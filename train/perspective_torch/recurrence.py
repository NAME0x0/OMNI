"""Chunked linear recurrence shared by PDR and GLA.

The state update is identical for the full-rank PDR gate and the low-rank GLA
gate:

    S_t = diag(g_t) S_{t-1} + v_t k_t^T
    y_t = S_t q_t

where ``S`` has shape ``[batch, d_model, rank]``.  The sequential reference is
kept for tests and tiny inputs.  The chunked path computes all states inside a
chunk with log-space cumulative products of the decay gates, then carries the
final state sequentially across chunks.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

DEFAULT_CHUNK_LEN = 128


def _validate_inputs(g: Tensor, k: Tensor, v: Tensor, q: Tensor) -> tuple[int, int, int, int]:
    if g.ndim != 3 or k.ndim != 3 or v.ndim != 3 or q.ndim != 3:
        raise ValueError("g, k, v, and q must all be rank-3 tensors")

    batch, seq_len, d_model = g.shape
    if v.shape != (batch, seq_len, d_model):
        raise ValueError("v must have shape [batch, seq, d_model]")
    if k.shape[:2] != (batch, seq_len) or q.shape[:2] != (batch, seq_len):
        raise ValueError("k and q must have shape [batch, seq, rank]")
    if k.shape[-1] != q.shape[-1]:
        raise ValueError("k and q ranks must match")
    if not (g.dtype == k.dtype == v.dtype == q.dtype):
        raise ValueError("g, k, v, and q must have the same dtype")
    if not (g.device == k.device == v.device == q.device):
        raise ValueError("g, k, v, and q must be on the same device")

    return batch, seq_len, d_model, k.shape[-1]


def _initial_state(
    initial_state: Tensor | None,
    batch: int,
    d_model: int,
    rank: int,
    like: Tensor,
) -> Tensor:
    if initial_state is None:
        return like.new_zeros(batch, d_model, rank)
    if initial_state.shape != (batch, d_model, rank):
        raise ValueError("initial_state must have shape [batch, d_model, rank]")
    return initial_state


def _maybe_project(readout: Tensor, output_weight: Tensor | None, output_bias: Tensor | None) -> Tensor:
    if output_weight is None:
        if output_bias is not None:
            raise ValueError("output_bias requires output_weight")
        return readout
    return F.linear(readout, output_weight, output_bias)


def sequential_linear_recurrence(
    g: Tensor,
    k: Tensor,
    v: Tensor,
    q: Tensor,
    *,
    initial_state: Tensor | None = None,
    output_weight: Tensor | None = None,
    output_bias: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Reference recurrence using a simple Python loop.

    Args:
        g: Decay gates in ``(0, 1)`` with shape ``[batch, seq, d_model]``.
        k: Keys with shape ``[batch, seq, rank]``.
        v: Values with shape ``[batch, seq, d_model]``.
        q: Queries with shape ``[batch, seq, rank]``.
        initial_state: Optional starting state ``[batch, d_model, rank]``.
        output_weight: Optional ``W_o`` for ``o_t = W_o (S_t q_t)``.
        output_bias: Optional output bias.

    Returns:
        ``(outputs, final_state)`` where outputs are ``[batch, seq, d_model]``.
    """

    batch, seq_len, d_model, rank = _validate_inputs(g, k, v, q)
    state = _initial_state(initial_state, batch, d_model, rank, g)

    outputs: list[Tensor] = []
    for t in range(seq_len):
        state = g[:, t].unsqueeze(-1) * state + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(1)
        outputs.append(torch.einsum("bdr,br->bd", state, q[:, t]))

    readout = torch.stack(outputs, dim=1) if outputs else g.new_zeros(batch, 0, d_model)
    return _maybe_project(readout, output_weight, output_bias), state


def chunked_linear_recurrence(
    g: Tensor,
    k: Tensor,
    v: Tensor,
    q: Tensor,
    *,
    initial_state: Tensor | None = None,
    output_weight: Tensor | None = None,
    output_bias: Tensor | None = None,
    chunk_len: int = DEFAULT_CHUNK_LEN,
) -> tuple[Tensor, Tensor]:
    """Chunked parallel formulation of the PDR/GLA state recurrence.

    Within each chunk, prefix sums of ``log(g)`` produce all decay products:

        prod_{i=j+1..t} g_i = exp(prefix_t - prefix_j)

    A batched matrix multiplication then accumulates all lower-triangular
    intra-chunk ``v_j k_j^T`` contributions.  Chunks are connected by carrying
    only the final ``S`` state.
    """

    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")

    batch, seq_len, d_model, rank = _validate_inputs(g, k, v, q)
    state = _initial_state(initial_state, batch, d_model, rank, g)
    if seq_len == 0:
        empty = g.new_zeros(batch, 0, d_model)
        return _maybe_project(empty, output_weight, output_bias), state

    outputs: list[Tensor] = []
    tiny = torch.finfo(g.dtype).tiny

    for start in range(0, seq_len, chunk_len):
        end = min(start + chunk_len, seq_len)
        g_chunk = g[:, start:end]
        k_chunk = k[:, start:end]
        v_chunk = v[:, start:end]
        q_chunk = q[:, start:end]
        length = end - start

        log_prefix = torch.cumsum(torch.log(g_chunk.clamp_min(tiny)), dim=1).transpose(1, 2)
        diff = log_prefix.unsqueeze(-1) - log_prefix.unsqueeze(-2)
        causal = torch.ones(length, length, dtype=torch.bool, device=g.device).tril()
        diff = diff.masked_fill(~causal, 0.0)
        weights = diff.exp() * causal.to(dtype=g.dtype)

        updates = torch.einsum("bld,blr->bdlr", v_chunk, k_chunk)
        intra_states = torch.matmul(weights, updates)
        carried_states = log_prefix.exp().unsqueeze(-1) * state.unsqueeze(2)
        states = carried_states + intra_states

        outputs.append(torch.einsum("bdlr,blr->bld", states, q_chunk))
        state = states[:, :, -1, :]

    readout = torch.cat(outputs, dim=1)
    return _maybe_project(readout, output_weight, output_bias), state


linear_recurrence = chunked_linear_recurrence


__all__ = [
    "DEFAULT_CHUNK_LEN",
    "chunked_linear_recurrence",
    "linear_recurrence",
    "sequential_linear_recurrence",
]
