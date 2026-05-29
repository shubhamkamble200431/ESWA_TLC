"""
model.py — Vanilla Transformer (seq2seq) and Mamba-based seq2seq.

Vanilla Transformer  : standard encoder-decoder with sinusoidal positional encoding.
MambaSeq2Seq         : Mamba SSM encoder + autoregressive Mamba decoder.

The Mamba implementation here is a *pure-PyTorch* version that does NOT
require the CUDA-optimised mamba-ssm library, so it runs on any device.
If `mamba-ssm` is installed it will be swapped in automatically.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import TransformerConfig, MambaConfig


# ══════════════════════════════════════════════════════
#  Shared utilities
# ══════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ══════════════════════════════════════════════════════
#  Vanilla Transformer
# ══════════════════════════════════════════════════════

class VanillaTransformer(nn.Module):
    """
    Standard Transformer encoder-decoder for sequence-to-sequence translation.
    """

    def __init__(self, cfg: TransformerConfig, vocab_size: int, pad_id: int = 0):
        super().__init__()
        self.cfg      = cfg
        self.pad_id   = pad_id
        self.d_model  = cfg.d_model

        self.src_embed = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos_enc   = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)

        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.output_proj = nn.Linear(cfg.d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_key_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        return seq == self.pad_id          # (B, T)  True = ignore

    def _make_causal_mask(self, sz: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(
        self,
        src: torch.Tensor,    # (B, S)
        tgt: torch.Tensor,    # (B, T)
    ) -> torch.Tensor:        # (B, T, V)
        src_pad_mask = self._make_key_padding_mask(src)
        tgt_pad_mask = self._make_key_padding_mask(tgt)
        causal_mask  = self._make_causal_mask(tgt.size(1), src.device)

        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.output_proj(out)       # (B, T, V)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 128,
    ) -> torch.Tensor:
        """Greedy decode; returns token ids (B, T)."""
        device = src.device
        B      = src.size(0)

        src_pad_mask = self._make_key_padding_mask(src)
        src_emb      = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        memory       = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        ys      = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done    = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_emb    = self.pos_enc(self.tgt_embed(ys) * math.sqrt(self.d_model))
            causal_mask = self._make_causal_mask(ys.size(1), device)
            out        = self.transformer.decoder(
                tgt_emb, memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            logits     = self.output_proj(out[:, -1, :])   # (B, V)
            next_tok   = logits.argmax(-1)                  # (B,)
            ys         = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            done      |= next_tok == eos_id
            if done.all():
                break

        return ys


# ══════════════════════════════════════════════════════
#  Pure-PyTorch Mamba SSM block
# ══════════════════════════════════════════════════════

class MambaBlock(nn.Module):
    """
    Selective State Space block (S6) — pure PyTorch, device-agnostic.
    Reference: Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling
               with Selective State Spaces".
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state

        self.in_proj   = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d    = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )
        self.x_proj    = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj   = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # SSM parameters  (log-diagonal init)
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(self.d_inner, 1)
        self.A_log     = nn.Parameter(torch.log(A))
        self.D         = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj  = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm      = nn.LayerNorm(d_model)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_inner)"""
        B, L, _ = x.shape
        A = -torch.exp(self.A_log.float())          # (d_inner, d_state)

        xz = self.x_proj(x)                         # (B, L, 2*d_state + d_inner)
        delta, B_param, C = xz.split(
            [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))      # (B, L, d_inner)

        # Discretise  A, B  (ZOH)
        dA = torch.exp(
            torch.einsum("bld,dn->bldn", delta, A)
        )                                            # (B, L, d_inner, d_state)
        dB = torch.einsum("bld,bln->bldn", delta, B_param)  # (B, L, d_inner, d_state)

        # Scan (sequential; replace with parallel scan for speed if desired)
        hs = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            hs = dA[:, i] * hs + dB[:, i] * x[:, i, :].unsqueeze(-1)
            y  = torch.einsum("bdn,bn->bd", hs, C[:, i])
            ys.append(y)
        y = torch.stack(ys, dim=1)                  # (B, L, d_inner)
        return y + x * self.D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)

        # depthwise conv along sequence
        x_part = x_part.transpose(1, 2)             # (B, d_inner, L)
        x_part = self.conv1d(x_part)[..., :x.size(1)]
        x_part = x_part.transpose(1, 2)             # (B, L, d_inner)
        x_part = F.silu(x_part)

        y = self.ssm(x_part)
        y = y * F.silu(z)
        return self.out_proj(y) + residual


class MambaEncoder(nn.Module):
    def __init__(self, cfg: MambaConfig, vocab_size: int, pad_id: int = 0):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.layers  = nn.ModuleList([
            MambaBlock(cfg.d_model, cfg.d_state, cfg.d_conv, cfg.expand)
            for _ in range(cfg.num_layers)
        ])
        self.norm    = nn.LayerNorm(cfg.d_model)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.embed(src))
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MambaCrossAttentionDecoder(nn.Module):
    """Mamba blocks interleaved with cross-attention to encoder memory."""

    def __init__(self, cfg: MambaConfig, vocab_size: int, pad_id: int = 0):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.mamba_layers  = nn.ModuleList([
            MambaBlock(cfg.d_model, cfg.d_state, cfg.d_conv, cfg.expand)
            for _ in range(cfg.num_layers)
        ])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(cfg.d_model, num_heads=8, dropout=cfg.dropout, batch_first=True)
            for _ in range(cfg.num_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(cfg.num_layers)])
        self.norm        = nn.LayerNorm(cfg.d_model)
        self.proj        = nn.Linear(cfg.d_model, vocab_size)

    def forward(
        self,
        tgt: torch.Tensor,       # (B, T)
        memory: torch.Tensor,    # (B, S, d)
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.pos_enc(self.embed(tgt))
        for mamba, xattn, norm in zip(self.mamba_layers, self.cross_attn, self.cross_norms):
            x = mamba(x)
            residual = x
            x = norm(x)
            x, _ = xattn(
                x, memory, memory,
                key_padding_mask=memory_key_padding_mask,
            )
            x = x + residual
        return self.proj(self.norm(x))


class MambaSeq2Seq(nn.Module):
    """Mamba encoder + cross-attention Mamba decoder."""

    def __init__(self, cfg: MambaConfig, vocab_size: int, pad_id: int = 0):
        super().__init__()
        self.pad_id  = pad_id
        self.encoder = MambaEncoder(cfg, vocab_size, pad_id)
        self.decoder = MambaCrossAttentionDecoder(cfg, vocab_size, pad_id)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        mem_pad_mask = (src == self.pad_id)
        tgt_pad_mask = (tgt == self.pad_id)
        memory  = self.encoder(src)
        logits  = self.decoder(tgt, memory,
                               tgt_key_padding_mask=tgt_pad_mask,
                               memory_key_padding_mask=mem_pad_mask)
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 128,
    ) -> torch.Tensor:
        device = src.device
        B      = src.size(0)
        memory = self.encoder(src)
        mem_pad_mask = (src == self.pad_id)

        ys   = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits   = self.decoder(ys, memory, memory_key_padding_mask=mem_pad_mask)
            next_tok = logits[:, -1, :].argmax(-1)
            ys       = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            done    |= next_tok == eos_id
            if done.all():
                break
        return ys


# ══════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════

def build_model(cfg, vocab_size: int, pad_id: int = 0) -> nn.Module:
    if cfg.model_type == "transformer":
        return VanillaTransformer(cfg, vocab_size, pad_id)
    elif cfg.model_type == "mamba":
        return MambaSeq2Seq(cfg, vocab_size, pad_id)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)