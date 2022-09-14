# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import pdb
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from tqdm import tqdm

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from util import box_ops


def quantize(bbox, bins=1000):
    """Quantization of (normalized) bbox in [0, 1]."""
    bbox = torch.round(bbox * (bins - 1)).short()
    bbox = torch.clamp(bbox, 0, bins - 1)
    return bbox


def build_response_seq_from_bbox(
    bbox,
    label,
    quantization_bins,
    noise_bbox_weight,
    coord_vocab_shift,
    base_vocab_shift,
    fake_class_token,
    class_label_corruption="rand_cls",
):
    """ "Build target seq from bounding bboxes for object detection.

    Objects are serialized using the format of yxyxc.

    Args:
      bbox: `float` bounding box of shape (bsz, n, 4).
      label: `int` label of shape (bsz, n).
      quantization_bins: `int`.
      noise_bbox_weight: `float` on the token weights for noise bboxes.
      coord_vocab_shift: `int`, shifting coordinates by a specified integer.
      class_label_corruption: `string` specifying how labels are corrupted for the
        input_seq.

    Returns:
      discrete sequences with shape (bsz, seqlen).
    """
    # Bbox and label quantization.
    quantized_bbox = quantize(bbox.tensor, quantization_bins) + coord_vocab_shift
    quantized_bbox = torch.where(
        bbox.mask, torch.zeros_like(quantized_bbox), quantized_bbox
    )
    new_label = label.tensor + base_vocab_shift
    new_label = torch.where(label.mask, torch.zeros_like(new_label), new_label)
    new_label = new_label.unsqueeze(-1)
    lb_shape = new_label.shape

    # Bbox and label serialization.
    response_seq = torch.cat((quantized_bbox, new_label), -1)
    response_seq = rearrange(response_seq, "n l b -> n (l b)")
    rand_cls = (
        torch.rand(lb_shape) * (coord_vocab_shift - base_vocab_shift) + base_vocab_shift
    )
    rand_cls = rand_cls.to(label.device).to(label.dtype)
    fake_cls = fake_class_token + torch.zeros_like(new_label)
    rand_n_fake_cls = torch.where(
        torch.rand(lb_shape, device=rand_cls.device) > 0.5, rand_cls, fake_cls
    )
    real_n_fake_cls = torch.where(
        torch.rand(lb_shape, device=rand_cls.device) > 0.5, new_label, fake_cls
    )
    real_n_rand_n_fake_cls = torch.where(
        torch.rand(lb_shape, device=new_label.device) > 0.5, new_label, rand_n_fake_cls
    )
    label_mapping = {
        "none": new_label,
        "rand_cls": rand_cls,
        "real_n_fake_cls": real_n_fake_cls,
        "rand_n_fake_cls": rand_n_fake_cls,
        "real_n_rand_n_fake_cls": real_n_rand_n_fake_cls,
    }
    new_label_m = label_mapping[class_label_corruption]
    new_label_m = torch.where(
        label.mask.unsqueeze(-1), torch.zeros_like(new_label_m), new_label_m
    )
    response_seq_class_m = torch.cat((quantized_bbox, new_label_m), -1)
    response_seq_class_m = rearrange(response_seq_class_m, "n l b -> n (l b)")

    # Get token weights.
    is_real = (new_label != fake_class_token).float()
    bbox_weight = torch.tile(is_real, (1, 1, 4))
    label_weight = is_real + (1.0 - is_real) * noise_bbox_weight
    token_weights = torch.cat((bbox_weight, label_weight), -1)
    token_weights = rearrange(token_weights, "n l b -> n (l b)")

    return response_seq, response_seq_class_m, token_weights


def decode_object_seq_to_bbox(
    logits, pred_seq, quantization_bins, coord_vocab_shift, base_vocab_shift
):
    """Decode objects (label & bbox) for seq from `build_response_seq_from_bbox`.

    Assume yxyxc format with truncation at the end for any uneven extra tokens.
      Replace class tokens with argmax instead of sampling.

    Args:
      logits: `float` output logits in shape of (bsz, max_seq_len, vocab_size).
      pred_seq: `int` pred sequence in shape of (bsz, max_seq_len).
      quantization_bins: `int` for bins.
      coord_vocab_shift: `int`, shifting coordinates by a specified integer.

    Returns:
      pred_class: `int` of shape (bsz, max_instances_per_image).
      pred_bbox: `float` of shape (bsz, max_instances_per_image, 4).
      pred_score: `float` of shape (bsz, max_instances_per_image).
    """
    _, seqlen, vocab_size = logits.shape
    if seqlen % 5 != 0:  # truncate out the last few tokens.
        pred_seq = pred_seq[..., : -(seqlen % 5)]
        logits = logits[..., : -(seqlen % 5), :]
    pred_class_p = logits.softmax(-1)[:, 4::5]  # (bsz, instances, vocab_size)
    mask_s1 = [0.0] * base_vocab_shift  # reserved.
    mask_s2 = [1.0] * (coord_vocab_shift - base_vocab_shift)  # labels.
    mask_s3 = [0] * (vocab_size - coord_vocab_shift)  # coordinates and others.
    mask = torch.tensor(mask_s1 + mask_s2 + mask_s3)
    pred_class = torch.argmax(pred_class_p * mask[torch.newaxis, torch.newaxis, :], -1)
    pred_score = torch.sum(pred_class_p * F.one_hot(pred_class, vocab_size), -1)
    pred_class = torch.maximum(pred_class - base_vocab_shift, 0)
    pred_bbox = seq_to_bbox(pred_seq - coord_vocab_shift, quantization_bins)
    return pred_class, pred_bbox, pred_score


def seq_to_bbox(seq, quantization_bins, seq_format="yxyx_name"):
    """Returns [0, 1] normalized yxyx bbox from token sequence."""
    # [batch, 5*num_instances]
    assert seq.shape.rank == 2, seq.shape.as_list()
    # [batch, num_instances, 1]
    if seq_format.startswith("name"):
        ymin = torch.unsqueeze(seq[:, 1::5], -1)
        xmin = torch.unsqueeze(seq[:, 2::5], -1)
        ymax = torch.unsqueeze(seq[:, 3::5], -1)
        xmax = torch.unsqueeze(seq[:, 4::5], -1)
    else:
        ymin = torch.unsqueeze(seq[:, 0::5], -1)
        xmin = torch.unsqueeze(seq[:, 1::5], -1)
        ymax = torch.unsqueeze(seq[:, 2::5], -1)
        xmax = torch.unsqueeze(seq[:, 3::5], -1)
    if seq_format in ["name_cycxhw", "cycxhw_name"]:
        ycnt, xcnt, ysize, xsize = ymin, xmin, ymax, xmax
        ymin = ycnt - ysize // 2
        xmin = xcnt - xsize // 2
        ymax = ycnt + ysize // 2
        xmax = xcnt + xsize // 2
    quantized_box = torch.cat((ymin, xmin, ymax, xmax), axis=-1)
    quantized_box = utils.dequantize(quantized_box, quantization_bins)
    return torch.minimum(torch.maximum(quantized_box, 0), 1)


def dequantize(boxes, bins):
    """Dequantization of discrete tokens of coordinates in [0, bins-1]."""
    boxes = boxes.float()
    boxes = boxes / (bins - 1)
    return boxes


def pre_process(samples, targets, config, training=False):
    response_seq, response_seq_cm, token_weights = build_response_seq_from_bbox(
        targets["boxes"],
        targets["labels"],
        config.quantization_bins,
        config.noise_bbox_weight,
        config.coord_vocab_shift,
        config.base_vocab_shift,
        config.fake_class_token,
        config.class_label_corruption,
    )
    prompt_seq = (
        torch.zeros_like(response_seq[..., :1], dtype=torch.int64)
        + config.task_vocab_id
    )
    input_seq = torch.cat((prompt_seq, response_seq_cm), -1)
    target_seq = response_seq
    # target_seq = torch.cat((prompt_seq, response_seq), -1)
    seq_len = config.model.transformer.decoder.max_seq_len
    input_seq = F.pad(input_seq, (0, seq_len - input_seq.shape[-1]))
    target_seq = F.pad(target_seq, (0, seq_len - target_seq.shape[-1]))
    token_weights = F.pad(token_weights, (0, seq_len - token_weights.shape[-1]))
    token_weights = torch.where(
        target_seq == config.padding_token,
        torch.zeros_like(token_weights) + config.eos_token_weight,
        token_weights,
    )
    if training:
        return samples, input_seq, target_seq, token_weights
    else:
        return samples, response_seq, (samples, targets)


def post_process(
    samples,
    targets,
    pred_seq,
    logits,
    quantization_bins,
    coord_vocab_shift,
    base_vocab_shift,
    training=False,
):
    orig_image_size = targets["orig_size"]
    unpadded_image_size = targets["size"]

    # Decode sequence output.
    pred_classes, pred_bboxes, scores = decode_object_seq_to_bbox(
        logits, pred_seq, quantization_bins, coord_vocab_shift, base_vocab_shift
    )

    # Compute coordinate scaling from [0., 1.] to actual pixels in orig image.
    image_size = images.shape[1:3].as_list()
    if training:
        # scale points to whole image size during train.
        scale = utils.tf_float32(image_size)
    else:
        # scale points to original image size during eval.
        scale = utils.tf_float32(image_size)[torch.newaxis, :] / utils.tf_float32(
            unpadded_image_size
        )
        scale = scale * utils.tf_float32(orig_image_size)
        scale = torch.unsqueeze(scale, 1)
    pred_bboxes_rescaled = utils.scale_points(pred_bboxes, scale)

    gt_classes, gt_bboxes = targets["label"], targets["bbox"]
    gt_bboxes_rescaled = utils.scale_points(gt_bboxes, scale)
    area, is_crowd = targets["area"], targets["is_crowd"]

    return (
        images,
        pred_bboxes,
        pred_bboxes_rescaled,
        pred_classes,
        scores,
        gt_classes,
        gt_bboxes,
        gt_bboxes_rescaled,
        area,
        is_crowd,
    )


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # consider use accelerate to avoid manual device assignment
        samples = samples.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        preprocessed = pre_process(samples, targets, config, model.training)
        outputs = model.train(preprocessed)
        outputs = outputs.reshape(-1, 2003)
        box_labels = box_labels.unsqueeze(0).repeat(6, 1, 1).flatten()
        loss = criterion(outputs[box_labels != 2002], box_labels[box_labels != 2002])
        loss_dict = {"at": loss}
        weight_dict = {"at": 1}
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if config.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
        optimizer.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, config):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if "panoptic" in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(config.output_dir, "panoptic_eval"),
        )
    for samples, targets in tqdm(data_loader):
        # consider use accelerate to avoid manual device assignment
        samples = samples.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        response_seq, _, _ = build_response_seq_from_bbox(
            targets["boxes"],
            targets["labels"],
            config.quantization_bins,
            config.noise_bbox_weight,
            config.coord_vocab_shift,
            config.base_vocab_shift,
            config.fake_class_token,
            config.class_label_corruption,
        )
        prompt_seq = (
            torch.zeros_like(response_seq[..., :1], dtype=torch.int64)
            + config.task_vocab_id
        )
        pred_seq, logits = model(samples, prompt_seq)
        results = post_process(
            samples,
            targets,
            pred_seq,
            logits,
            config.quantization_bins,
            config.coord_vocab_shift,
            config.base_vocab_shift,
            model.training,
        )
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes
            )
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    if panoptic_res is not None:
        stats["PQ_all"] = panoptic_res["All"]
        stats["PQ_th"] = panoptic_res["Things"]
        stats["PQ_st"] = panoptic_res["Stuff"]
    return stats, coco_evaluator
