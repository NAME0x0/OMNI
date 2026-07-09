"""Stage 0 language models for PDR, GLA, and transformer baselines."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import Stage0Config
from .gates import FullRankGate, LowRankGate
from .layers import CausalAttentionBlock, RMSNorm, RecurrentBlock, SlidingWindowAttentionBlock, SwiGLUFFN


class Stage0Block(nn.Module):
    """One sequence mixer followed by one SwiGLU FFN."""

    def __init__(self, mixer: nn.Module, ffn: SwiGLUFFN) -> None:
        super().__init__()
        self.mixer = mixer
        self.ffn = ffn

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(self.mixer(x))


class Stage0LM(nn.Module):
    """GPT-2-BPE-sized causal LM for the Stage 0 architecture gate."""

    def __init__(self, config: Stage0Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([self._make_block(i) for i in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)

    def _make_block(self, layer_idx: int) -> Stage0Block:
        config = self.config
        if config.variant == "transformer":
            mixer: nn.Module = CausalAttentionBlock(config.d_model, n_heads=config.n_heads)
        elif layer_idx % 4 == 3:
            mixer = SlidingWindowAttentionBlock(
                config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                window=config.sliding_window,
            )
        else:
            gate = FullRankGate(config.d_model) if config.variant == "pdr" else LowRankGate(
                config.d_model,
                gate_rank=config.low_rank_gate_rank,
            )
            mixer = RecurrentBlock(config.d_model, config.rank, gate)

        ffn = SwiGLUFFN(config.d_model, int(config.ffn_intermediate))
        return Stage0Block(mixer, ffn)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return F.linear(x, self.token_embedding.weight)


def count_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


__all__ = ["Stage0Block", "Stage0LM", "count_parameters"]
