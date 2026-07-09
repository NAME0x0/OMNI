"""PyTorch layers for the Stage 0 architecture comparison."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .recurrence import chunked_linear_recurrence, sequential_linear_recurrence

ROPE_BASE = 10_000.0


class RMSNorm(nn.Module):
    """RMSNorm: ``x * weight / sqrt(mean(x^2) + eps)``."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def _validate_heads(d_model: int, n_heads: int, n_kv_heads: int | None = None) -> int:
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if n_kv_heads is not None and n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    return d_model // n_heads


def _apply_rope(x: Tensor) -> Tensor:
    """Apply rotary positional embeddings to ``[batch, heads, seq, head_dim]``."""

    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")

    seq_len = x.shape[-2]
    pos = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    freq_idx = torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype)
    inv_freq = ROPE_BASE ** (-freq_idx / head_dim)
    angles = torch.outer(pos, inv_freq)
    cos = angles.cos().view(1, 1, seq_len, head_dim // 2)
    sin = angles.sin().view(1, 1, seq_len, head_dim // 2)

    even = x[..., 0::2]
    odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


class RecurrentBlock(nn.Module):
    """Pre-norm PDR/GLA recurrent mixer with residual output.

    The gate module supplies ``g_t``.  The shared recurrence applies
    ``S_t = diag(g_t) S_{t-1} + v_t k_t^T`` and reads ``S_t q_t``.
    """

    def __init__(self, d_model: int, rank: int, gate: nn.Module) -> None:
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.norm = RMSNorm(d_model)
        self.gate = gate
        self.W_k = nn.Linear(d_model, rank, bias=False)
        self.W_q = nn.Linear(d_model, rank, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        *,
        initial_state: Tensor | None = None,
        return_state: bool = False,
        use_chunked: bool = True,
    ) -> Tensor | tuple[Tensor, Tensor]:
        h = self.norm(x)
        g = self.gate(h)
        k = self.W_k(h)
        q = self.W_q(h)
        v = self.W_v(h)

        recurrence = chunked_linear_recurrence if use_chunked else sequential_linear_recurrence
        readout, state = recurrence(g, k, v, q, initial_state=initial_state)
        out = x + self.W_o(readout)
        return (out, state) if return_state else out


class SlidingWindowAttentionBlock(nn.Module):
    """Pre-norm grouped-query causal attention over a fixed recent window."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        window: int = 256,
    ) -> None:
        super().__init__()
        if window <= 0:
            raise ValueError("window must be positive")
        head_dim = _validate_heads(d_model, n_heads, n_kv_heads)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.window = window
        self.norm = RMSNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _attention_mask(self, seq_len: int, device: torch.device) -> Tensor:
        row = torch.arange(seq_len, device=device).view(seq_len, 1)
        col = torch.arange(seq_len, device=device).view(1, seq_len)
        return (col <= row) & ((row - col) < self.window)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        h = self.norm(x)
        q = self.W_q(h).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(h).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(h).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q)
        k = _apply_rope(k)
        repeat = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        mask = self._attention_mask(seq_len, x.device).view(1, 1, seq_len, seq_len)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return x + self.W_o(y)


class CausalAttentionBlock(nn.Module):
    """Pre-norm full causal self-attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int = 12) -> None:
        super().__init__()
        head_dim = _validate_heads(d_model, n_heads)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.norm = RMSNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        h = self.norm(x)
        q = self.W_q(h).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(h).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(h).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q)
        k = _apply_rope(k)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return x + self.W_o(y)


class SwiGLUFFN(nn.Module):
    """Pre-norm SwiGLU feed-forward block with residual output."""

    def __init__(self, d_model: int, intermediate: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.intermediate = intermediate
        self.norm = RMSNorm(d_model)
        self.W_gate = nn.Linear(d_model, intermediate, bias=False)
        self.W_up = nn.Linear(d_model, intermediate, bias=False)
        self.W_down = nn.Linear(intermediate, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        hidden = F.silu(self.W_gate(h)) * self.W_up(h)
        return x + self.W_down(hidden)


__all__ = [
    "CausalAttentionBlock",
    "RMSNorm",
    "RecurrentBlock",
    "SlidingWindowAttentionBlock",
    "SwiGLUFFN",
]
