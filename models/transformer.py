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
        encoder,
        decoder,
    ):
        super().__init__()
        self.use_cls_token = encoder.use_cls_token
        self.dec_proj_mode = decoder.proj_mode
        self.embed_dim = encoder.embed_dim
        encoder_layer = TransformerEncoderLayer(**encoder)
        encoder_norm = nn.LayerNorm(encoder.embed_dim) if encoder.norm_first else None
        # encoder_norm = None
        self.encoder = TransformerEncoder(
            layer=encoder_layer, norm=encoder_norm, **encoder
        )

        self.proj = nn.Linear(encoder.embed_dim, decoder.embed_dim)
        self.proj_norm = nn.LayerNorm(decoder.embed_dim)
        if self.dec_proj_mode in ("linear_p", "mlp"):
            self.pos_embed = build_position_encoding(**encoder)
            if self.dec_proj_mode == "mlp":
                self.proj_mlp = FullyConnectedNetwork(**decoder)

        decoder_layer = TransformerDecoderLayer(**decoder)
        decoder_norm = nn.LayerNorm(decoder.embed_dim) if decoder.norm_first else None
        self.decoder = TransformerDecoder(
            layer=decoder_layer, norm=decoder_norm, **decoder
        )

        self._reset_parameters()

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
        layer,
        num_layers,
        norm,
        use_cls_token,
        embed_dim,
        pos_embed,
        **kwargs,
    ):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pos_embed = build_position_encoding(pos_embed, embed_dim)
        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, embed_dim)) if use_cls_token else None
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
    def __init__(
        self,
        layer,
        num_layers,
        norm,
        return_intermediate,
        shared_embed,
        output_bias,
        vocab_size,
        embed_dim,
        max_seq_len,
        temperature,
        top_k,
        top_p,
        **kwargs,
    ):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.pos_embed = nn.Parameter(torch.randn(self.max_seq_len, self.embed_dim))
        if shared_embed:
            self.input_embed = self.output_embed = nn.Parameter(
                torch.randn(self.vocab_size, self.embed_dim)
            )
        else:
            self.input_embed = nn.Parameter(
                torch.randn(self.vocab_size, self.embed_dim)
            )
            self.output_embed = nn.Parameter(
                torch.randn(self.vocab_size, self.embed_dim)
            )
        self.output_bias = (
            nn.Parameter(torch.Tensor(self.vocab_size)) if output_bias else None
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
        sampling_callback=sample,
        max_seq_len=None,
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
                next_logits, steps, self.temperature, self.top_k, self.top_p
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

        caches_var = torch.zeros(seq_len - 1, self.num_layers, bsz, self.embed_dim).to(
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
        embed_dim,
        num_heads,
        ffn_dim=2048,
        dropout=0.1,
        dropout_attn=0.0,
        activation="gelu",
        norm_first=False,
        **kwargs,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.self_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_attn, batch_first=True
        )
        self.mlp = FullyConnectedNetwork(
            embed_dim, ffn_dim, dropout, activation, norm_first
        )

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        output = src
        if self.norm_first:
            output = self.self_norm(output)
        attn = self.self_attn(
            output,
            output,
            output,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        output = src + attn
        if not self.norm_first:
            output = self.self_norm(output)
        output = self.mlp(output)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ffn_dim=2048,
        dropout=0.1,
        dropout_attn=0.0,
        activation="gelu",
        norm_first=False,
        **kwargs,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.self_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_attn, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_attn, batch_first=True
        )
        self.mlp = FullyConnectedNetwork(
            embed_dim, ffn_dim, dropout, activation, norm_first
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
        if self.norm_first:
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
        output = self.cross_norm(output) if self.norm_first else self.self_norm(output)
        attn = self.cross_attn(
            query=output,
            key=memory,
            value=memory,
            attn_mask=cross_mask,
            key_padding_mask=cross_key_padding_mask,
        )[0]
        output = output + attn
        if not self.norm_first:
            output = self.cross_norm(output)
        output = self.mlp(output)
        return output, x_for_cache


class FullyConnectedNetwork(nn.Module):
    def __init__(
        self, embed_dim, ffn_dim, dropout, activation, norm_first, **kwargs
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        return (
            x + self.linear2(self.dropout(self.activation(self.linear1(self.norm(x)))))
            if self.norm_first
            else self.norm(
                x + self.linear2(self.dropout(self.activation(self.linear1(x))))
            )
        )


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def build_transformer(config):
    return Transformer(**config.model.transformer)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
