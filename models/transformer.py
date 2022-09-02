# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import pdb
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .position_encoding import build_position_encoding
from .utils import sample


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        dropout_attn=0.0,
        activation="relu",
        normalize_before=True,
        return_intermediate_dec=False,
        dec_proj_mode="mlp",
        use_cls_token=True,
        args=None,
    ):
        super().__init__()
        self.dec_proj_mode = dec_proj_mode
        self.use_cls_token = use_cls_token
        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            dropout_attn,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        # encoder_norm = None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm, use_cls_token, args
        )

        self.proj = nn.Linear(d_model, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        if self.dec_proj_mode in ("linear_p", "mlp"):
            self.pos_embed = build_position_encoding(args)
            if self.dec_proj_mode == "mlp":
                self.proj_mlp = FullyConnectedNetwork(
                    d_model, dim_feedforward, dropout, activation, normalize_before
                )

        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            dropout_attn,
            activation,
            normalize_before,
        )
        decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate_dec,
            args,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, seq):
        encoded = self.encoder(src, src_key_padding_mask=mask)
        encoded = self.proj_norm(self.proj(encoded))
        if self.dec_proj_mode in ("linear_p", "mlp"):
            pos_embed = self.pos_embed(encoded, mask)
            if self.use_cls_token:
                pos_embed = torch.cat(
                    (torch.zeros_like(pos_embed[:, :1]), pos_embed), dim=1
                )
            encoded = encoded + pos_embed
            if self.dec_proj_mode == "mlp":
                encoded = self.proj_mlp(encoded)

        return self.decoder(encoded, seq)


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm,
        use_cls_token,
        args,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pos_embed = build_position_encoding(args)
        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, encoder_layer.dim_model))
            if use_cls_token
            else None
        )

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        pos_embed = self.pos_embed(src, src_key_padding_mask)
        src_key_padding_mask = src_key_padding_mask.flatten(1)
        output = src + pos_embed
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(src.shape[0], -1, -1)
            output = torch.cat([cls_tokens, output], dim=1)

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm, return_intermediate, args):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.vocab_size, self.hidden_dim, self.max_seq_len = (
            args.vocab_size,
            args.hidden_dim,
            args.max_seq_len,
        )
        self.pos_embed = nn.Parameter(torch.randn(self.max_seq_len, self.hidden_dim))
        if args.shared_embed:
            self.input_embed = self.output_embed = nn.Parameter(
                torch.randn(self.vocab_size, self.hidden_dim)
            )
        else:
            self.input_embed = nn.Parameter(
                torch.randn(self.vocab_size, self.hidden_dim)
            )
            self.output_embed = nn.Parameter(
                torch.randn(self.vocab_size, self.hidden_dim)
            )
        self.output_bias = (
            nn.Parameter(torch.Tensor(self.vocab_size)) if args.output_bias else None
        )

    def fit(self, encoded, seq):
        _, seq_len = seq.shape
        tokens = self.input_embed[seq] + self.pos_embed[:seq_len]
        self_mask = torch.ones(seq_len, seq_len).tril().to(encoded.device)
        output, _ = self.decoder(tokens, encoded, None, self_mask)
        output = output @ self.output_embed.T
        if self.output_bias is not None:
            output = output + self.output_bias
        return output

    def inf(
        self,
        encoded,
        prompt,
        max_seq_len=None,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        sampling_callback=sample,
    ):
        bsz, prompt_len = prompt.shape
        seq_len = self.max_seq_len if max_seq_len is None else max_seq_len

        def step(steps, caches, tokens, logits, is_prompt=False):
            """
            Each step reads caches[:step] and tokens[step:next_step] and updates
            tokens[next_step], logits[next_step] and caches[step:next_step].
            On the first step, step=0, next_step=prompt_len. On subsequent steps
            next_step = step + 1.
            """
            if is_prompt:
                assert steps == 0
                x = self.input_embed[tokens[:prompt_len].T]
                x = x + self.pos_embed[:prompt_len]  # (bsz, prompt_len, d)
                self_mask = torch.ones(prompt_len, prompt_len).tril().to(encoded.device)
                caches_in = None
            else:
                x = self.input_embed[tokens[steps].T]
                x = x + self.pos_embed[steps]  # (bsz, d)
                x = x.unsqueeze(1)
                self_mask = torch.ones(1, 1).to(encoded.device)
                caches_in = caches[:steps].permute(1, 2, 0, 3)
            output, caches_out = self.decoder(x, encoded, caches_in, self_mask)
            if self.norm is not None:
                output = self.norm(output)
            next_logits = (output @ self.output_embed.T).squeeze(1)
            if self.output_bias is not None:
                next_logits = next_logits + self.output_bias

            # Scale and trunctate logits and sample next token.
            next_token = sampling_callback(
                next_logits, steps, temperature, top_k, top_p
            )

            # Update internal states.
            next_step = steps + (prompt_len if is_prompt else 1)
            caches_out = caches_out.permute(2, 0, 1, 3)
            caches = caches.index_copy_(0, torch.tensor([steps]).cuda(), caches_out)
            tokens = tokens.index_copy_(
                0, torch.tensor([next_step]).cuda(), next_token[None]
            )
            logits = logits.index_copy_(
                0, torch.tensor([next_step]).cuda(), next_logits[None]
            )
            return (next_step, caches, tokens, logits)

        caches_var = torch.zeros(seq_len - 1, self.num_layers, bsz, self.hidden_dim).to(
            encoded.device
        )
        tokens_var = torch.zeros(seq_len, bsz, dtype=torch.int64).to(encoded.device)
        logits_var = torch.zeros(seq_len, bsz, self.vocab_size, dtype=torch.float32).to(
            encoded.device
        )
        indices = torch.arange(prompt_len).to(encoded.device)
        tokens_var = tokens_var.index_copy_(0, indices, prompt.T)

        steps = 0
        steps, caches_var, tokens_var, logits_var = step(
            steps, caches_var, tokens_var, logits_var, is_prompt=True
        )
        if seq_len > prompt_len:
            while steps < seq_len - 1:
                steps, caches_var, tokens_var, logits_var = step(
                    steps, caches_var, tokens_var, logits_var
                )

        sampled_tokens = tokens_var[prompt_len:].T
        sampled_tokens_logits = logits_var[prompt_len:].permute(1, 0, 2)
        return sampled_tokens, sampled_tokens_logits

    def decoder(self, tgt, memory, caches, mask):
        output = tgt
        presents = []

        for i, layer in enumerate(self.layers):
            cache = None if caches is None else caches[i]
            output, cache = layer(
                output,
                memory,
                cache=cache,
                self_mask=mask,
            )
            if self.return_intermediate:
                presents.append(output)

        if self.norm is not None:
            output = self.norm(output)

        return output, torch.stack(presents)

    def forward(self, *args, **kwargs):
        return self.fit(*args, **kwargs) if self.training else self.inf(*args, **kwargs)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        dropout_attn=0.0,
        activation="gelu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout_attn, batch_first=True
        )
        self.mlp = FullyConnectedNetwork(
            d_model, dim_feedforward, dropout, activation, normalize_before
        )

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        output = src
        if self.normalize_before:
            output = self.self_norm(output)
        attn = self.self_attn(
            output,
            output,
            output,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        output = src + attn
        if not self.normalize_before:
            output = self.self_norm(output)
        output = self.mlp(output)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        dropout_attn=0.0,
        activation="gelu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout_attn, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout_attn, batch_first=True
        )
        self.mlp = FullyConnectedNetwork(
            d_model, dim_feedforward, dropout, activation, normalize_before
        )

    def forward(
        self,
        tgt,
        memory,
        cache: Optional[Tensor] = None,
        self_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None,
        self_key_padding_mask: Optional[Tensor] = None,
        cross_key_padding_mask: Optional[Tensor] = None,
    ):
        output = tgt
        if self.normalize_before:
            output = self.self_norm(tgt)
        x_for_cache = kv = output
        if cache is not None:  # Augment kv_ln with cache in (bsz, c_size, d).
            self_mask = (
                torch.ones(tgt.shape[1], cache.shape[1] + tgt.shape[1])
                .tril()
                .to(tgt.device)
            )
            kv = torch.cat([cache, kv], dim=1)
        attn = self.self_attn(
            query=output,
            key=kv,
            value=kv,
            attn_mask=self_mask,
            key_padding_mask=self_key_padding_mask,
        )[0]
        output = tgt + attn
        output = (
            self.cross_norm(output) if self.normalize_before else self.self_norm(output)
        )
        attn = self.cross_attn(
            query=output,
            key=memory,
            value=memory,
            attn_mask=cross_mask,
            key_padding_mask=cross_key_padding_mask,
        )[0]
        output = output + attn
        if not self.normalize_before:
            output = self.cross_norm(output)
        output = self.mlp(output)
        return output, x_for_cache


class FullyConnectedNetwork(nn.Module):
    def __init__(self, d_model, d_mlp, dropout, activation, normalize_before) -> None:
        super().__init__()
        self.normalize_before = normalize_before
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_mlp)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_mlp, d_model)

    def forward(self, x):
        return (
            x + self.linear2(self.dropout(self.activation(self.linear1(self.norm(x)))))
            if self.normalize_before
            else self.norm(
                x + self.linear2(self.dropout(self.activation(self.linear1(x))))
            )
        )


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        dropout_attn=args.dropout_attn,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        use_cls_token=args.use_cls_token,
        args=args,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
