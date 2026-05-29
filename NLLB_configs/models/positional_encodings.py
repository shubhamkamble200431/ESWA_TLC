import math
import torch
import torch.nn as nn
from typing import Optional


class RotaryEmbedding(nn.Module):

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int, device=None) -> None:
        if device is None:
            device = self.inv_freq.device
        if self.inv_freq.device != device:
            self.inv_freq = self.inv_freq.to(device)
        t     = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer(
            "cos_cache",
            emb.cos()[None, None, :, :],
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            emb.sin()[None, None, :, :],
            persistent=False,
        )
        self.max_seq_len = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        device   = q.device
        q_len    = q.shape[2]
        k_len    = k.shape[2]
        need_len = max(q_len, k_len)
        head_dim = q.shape[-1]

        if need_len > self.cos_cache.shape[2] or self.cos_cache.device != device:
            self._build_cache(need_len, device=device)

        q_cos = self.cos_cache[:, :, :q_len, :head_dim]
        q_sin = self.sin_cache[:, :, :q_len, :head_dim]
        k_cos = self.cos_cache[:, :, :k_len, :head_dim]
        k_sin = self.sin_cache[:, :, :k_len, :head_dim]

        return _apply_rotation(q, q_cos, q_sin), _apply_rotation(k, k_cos, k_sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotation(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


def apply_rope_to_model(
    model: nn.Module,
    head_dim: int,
    max_seq_len: int = 2048,
) -> nn.Module:
    patched = 0
    for name, module in model.named_modules():
        if "Attention" in type(module).__name__ and hasattr(module, "q_proj"):
            layer_rope = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len)
            _patch_attention_with_rope(module, layer_rope)
            patched += 1

    print(f"[RoPE] Patched {patched} attention layers "
          f"(each with its own RotaryEmbedding, non-persistent buffers).")
    return model


def _patch_attention_with_rope(
    attn_module: nn.Module,
    rope: RotaryEmbedding,
) -> None:
    import torch.nn.functional as F

    num_heads = attn_module.num_heads
    head_dim  = rope.dim

    q_linear: nn.Module = attn_module.q_proj
    k_linear: nn.Module = attn_module.k_proj

    object.__setattr__(attn_module, "_rope", rope)

    _orig_attn_forward = attn_module.forward

    class _Fixed(nn.Module):
        def __init__(self, out: torch.Tensor):
            super().__init__()
            self._out = out
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self._out

    def _rope_attn_forward(
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        kv_src = key_value_states if key_value_states is not None else hidden_states

        raw_q = F.linear(hidden_states, q_linear.weight, q_linear.bias)
        raw_k = F.linear(kv_src,        k_linear.weight, k_linear.bias)

        B_q, Sq, _ = raw_q.shape
        B_k, Sk, _ = raw_k.shape

        q4d = raw_q.view(B_q, Sq, num_heads, head_dim).transpose(1, 2)
        k4d = raw_k.view(B_k, Sk, num_heads, head_dim).transpose(1, 2)

        q_rot, k_rot = attn_module._rope(q4d, k4d)

        q_flat = q_rot.transpose(1, 2).contiguous().view(B_q, Sq, num_heads * head_dim)
        k_flat = k_rot.transpose(1, 2).contiguous().view(B_k, Sk, num_heads * head_dim)

        _save_q = attn_module.q_proj
        _save_k = attn_module.k_proj
        attn_module.q_proj = _Fixed(q_flat)
        attn_module.k_proj = _Fixed(k_flat)
        try:
            result = _orig_attn_forward(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
        finally:
            attn_module.q_proj = _save_q
            attn_module.k_proj = _save_k

        return result

    attn_module.forward = _rope_attn_forward


class ALiBiPositionalBias(nn.Module):

    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        slopes     = self._get_slopes(num_heads)
        alibi      = slopes.unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0)
        alibi_bias = alibi.unsqueeze(1) - alibi.unsqueeze(2)
        self.register_buffer(
            "bias",
            alibi_bias.unsqueeze(0),
            persistent=True,
        )

    @staticmethod
    def _get_slopes(n: int) -> torch.Tensor:
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return torch.tensor(get_slopes_power_of_2(n))
        else:
            closest = 2 ** math.floor(math.log2(n))
            base    = get_slopes_power_of_2(closest)
            extra   = get_slopes_power_of_2(2 * closest)[::2]
            return torch.tensor(base + extra[: n - closest])

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.bias[:, :, :seq_len, :seq_len]


def apply_alibi_to_model(
    model: nn.Module,
    num_heads: int,
    max_seq_len: int = 2048,
):
    alibi   = ALiBiPositionalBias(num_heads=num_heads, max_seq_len=max_seq_len)
    patched = 0
    for name, module in model.named_modules():
        if "Attention" in type(module).__name__ and hasattr(module, "q_proj"):
            module.alibi_bias = alibi
            patched += 1

    print(
        f"[ALiBi] Attached ALiBiPositionalBias to {patched} attention layers.\n"
        "        To activate: add `module.alibi_bias(seq_len)` to attention "
        "scores in each forward call."
    )
    return model, alibi
