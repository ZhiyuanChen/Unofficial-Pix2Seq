import os
import re
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import torch
from einops import rearrange


def rename(name):
    name = name.replace("/", ".")
    name = name.replace("dense", "linear")
    name = name.replace("ln", "norm")
    name = name.replace("embedding", "embed")
    name = name.replace("kernel", "weight")
    name = name.replace("gamma", "weight")
    name = name.replace("output_weight", "weight")
    name = name.replace("beta", "bias")
    name = name.replace("output_bias", "bias")
    name = name.replace("moving", "running")
    name = name.replace("variance", "var")
    name = name.replace("..ATTRIBUTES.VARIABLE_VALUE", "")
    # process resnet
    if name.startswith("model.encoder.resnet"):
        name = name.replace("model.encoder.resnet", "backbone")
        name = name.replace("block_groups.", "body.layer")
        name = name.replace("initial_conv_relu_max_pool.2.bn", "body.bn1")
        name = name.replace("initial_conv_relu_max_pool.0.conv2d", "body.conv1")
        name = name.replace("layers.", "")
        name = name.replace("layer3", "layer4").replace("layer2", "layer3")
        name = name.replace("layer1", "layer2").replace("layer0", "layer1")
        name = name.replace("conv_relu_dropblock_0.conv2d", "conv1").replace(
            "conv_relu_dropblock_1.bn", "bn1"
        )
        name = name.replace("conv_relu_dropblock_3.conv2d", "conv2").replace(
            "conv_relu_dropblock_4.bn", "bn2"
        )
        name = name.replace("conv_relu_dropblock_6.conv2d", "conv3").replace(
            "conv_relu_dropblock_7.bn", "bn3"
        )
        name = name.replace("projection_", "downsample.")
        name = name.replace("downsample.0.conv2d", "downsample.0")
        name = name.replace("downsample.1.bn", "downsample.1")
    if name.startswith("model.proj"):
        name = name.replace("model", "transformer")
    if name.startswith("model.encoder.stem"):
        name = name.replace("model.encoder.", "")
        name = name.replace("projection", "proj")
    # process transformer encoder
    if name.startswith("model.encoder.transformer_encoder"):
        name = name.replace("model.encoder.transformer_encoder", "transformer.encoder")
        name = name.replace("enc_layers", "layers")
        name = name.replace("mha", "self_attn")
    # process transformer decoder
    if name.startswith("model.decoder.decoder"):
        name = name.replace("model.decoder", "transformer")
        name = name.replace("dec_layers", "layers")
        name = name.replace("self_mha", "self_attn")
        name = name.replace("cross_mha", "cross_attn")
    # process transformer in general
    if "transformer" in name:
        name = name.replace("ffn.", "")
        name = name.replace("mlp_layers.0.", "")
        name = name.replace("layernorms.0", "norm")
        name = name.replace("self_attn_norm", "self_norm")
        name = name.replace("_output_linear", "out_proj")
    if "output_norm" in name:
        name = name.replace("model", "transformer")
        name = name.replace("output_", "")
    if "ar_decoder" in name:
        name = name.replace("model", "transformer")
        name = name.replace("ar_decoder.S", "")
        name = name.replace("seq_", "")
        name = name.replace("inp", "input")
        name = name.replace("outp", "output")
    return name


def adjust_proj(state_dict):
    proj_dict = {}
    proj_keys = []
    for k, v in state_dict.items():
        # in projection
        if "transformer" in k and "_linear" in k:
            proj_keys.append(k)
            ks = k.split(".")
            # layer, qkv, weight/bias
            g, i, t = ".".join(ks[:5]), ks[5][1:-7], ks[6]
            if g not in proj_dict:
                proj_dict[g] = {}
            if t not in proj_dict[g]:
                proj_dict[g][t] = {}
            proj_dict[g][t][i] = v  # rearrange(v, 'k h c -> (h c) k')
        # out projection
        elif k.endswith("out_proj.weight"):
            state_dict[k] = rearrange(v, "h c k -> (h c) k")
    for k in proj_keys:
        del state_dict[k]
    for g, v1 in proj_dict.items():
        for t, v2 in v1.items():
            q, k, v = v2["query"], v2["key"], v2["value"]
            p = np.concatenate((q, k, v))
            if t == "weight":
                p = rearrange(p, "k h c -> (h c) k")
            elif t == "bias":
                p = rearrange(p, "h c -> (h c)")
            else:
                raise ValueError(
                    f"Invalid parameter specified, excepted to be weight or bias, but got {t}"
                )
            state_dict[g + ".in_proj_" + t] = p
    return state_dict


def adjust_weight(state_dict):
    state_dict = {
        k: np.transpose(v) if "weight" in k else v for k, v in state_dict.items()
    }
    state_dict = {k: torch.from_numpy(v) for k, v in state_dict.items()}
    if "transformer.decoder.token_embed" in state_dict:
        state_dict["transformer.decoder.input_embed"] = state_dict[
            "transformer.decoder.token_embed"
        ]
        state_dict["transformer.decoder.output_embed"] = state_dict[
            "transformer.decoder.token_embed"
        ]
        del state_dict["transformer.decoder.token_embed"]
    # state_dict["transformer.decoder.pos_embed"] = state_dict[
    #    "transformer.decoder.pos_embed"
    # ][None]
    return state_dict


def adjust(state_dict):
    state_dict = adjust_proj(state_dict)
    state_dict = adjust_weight(state_dict)
    return state_dict


def convert(tf_ckpt_path):
    """Load tf checkpoints in a pytorch model."""
    tf_ckpt_path = os.path.abspath(tf_ckpt_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_ckpt_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_ckpt_path)
    state_dict = {}
    for name, shape in init_vars:
        if (
            "optimizer" not in name
            and "global_step" not in name
            and "_CHECK" not in name
            and "save_counter" not in name
        ):
            print("Loading TF weight {} with shape {}".format(name, shape))
            array = tf.train.load_variable(tf_ckpt_path, name)
            state_dict[rename(name)] = array
    state_dict = adjust(state_dict)

    return OrderedDict(model=state_dict)


if __name__ == "__main__":
    torch.save(
        convert("checkpoints/tensorflow/resnet_640x640/ckpt-74844"),
        "checkpoints/pytorch/resnet_640x640/ckpt-74844.pth",
    )
