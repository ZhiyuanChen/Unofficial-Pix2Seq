# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, DistributedSampler

torch.multiprocessing.set_sharing_strategy("file_system")

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from configs import TuneConfig


def main(config):
    utils.init_distributed_mode(config)
    print("git:\n  {}\n".format(utils.get_sha()))

    if config.model.frozen_weights is not None:
        assert config.masks, "Frozen training is meant for segmentation only"
    print(config)

    device = torch.device(config.device)

    # fix the seed for reproducibility
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(config)
    model.to(device)

    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": config.optim.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.optim.lr, weight_decay=config.optim.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    dataset_train = build_dataset(image_set="train", config=config)
    dataset_val = build_dataset(image_set="val", config=config)

    if config.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=config.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        config.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=config.num_workers,
    )

    if config.dataset.file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", config)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if config.model.frozen_weights is not None:
        checkpoint = torch.load(config.model.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    output_dir = Path(config.output_dir)
    if config.resume:
        if config.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                config.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(config.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            not config.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            config.start_epoch = checkpoint["epoch"] + 1

    if config.eval:
        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            config,
        )
        if config.output_dir:
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
            )
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            config,
        )
        lr_scheduler.step()
        if config.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % config.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "config": config,
                    },
                    checkpoint_path,
                )

        #        test_stats, coco_evaluator = evaluate(
        #            model, criterion, postprocessors, data_loader_val, base_ds, device, config.output_dir
        #        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            #                     **{f'test_{k}': v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if config.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
    #            if coco_evaluator is not None:
    #                (output_dir / 'eval').mkdir(exist_ok=True)
    #                if "bbox" in coco_evaluator.coco_eval:
    #                    filenames = ['latest.pth']
    #                    if epoch % 50 == 0:
    #                        filenames.append(f'{epoch:03}.pth')
    #                    for name in filenames:
    #                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
    #                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    config = TuneConfig()
    config.parse()
    if config.output_dir:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    main(config)
