"""Decay gate modules for the Stage 0 PDR/GLA comparison."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

GATE_INIT_MEAN = 0.95
GATE_BIAS_INIT = math.log(GATE_INIT_MEAN / (1.0 - GATE_INIT_MEAN))


class FullRankGate(nn.Module):
    """PDR gate ``g = sigmoid(W_p x + b_p)`` with full ``d x d`` capacity."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.W_p = nn.Linear(d_model, d_model, bias=False)
        self.b_p = nn.Parameter(torch.empty(d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.W_p.weight.zero_()
            self.W_p.weight.add_(torch.eye(self.d_model, device=self.W_p.weight.device))
            self.W_p.weight.add_(0.01 * torch.randn_like(self.W_p.weight))
            self.b_p.fill_(GATE_BIAS_INIT)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(F.linear(x, self.W_p.weight, self.b_p))


class LowRankGate(nn.Module):
    """GLA baseline gate ``g = sigmoid(W_2 W_1 x + b)`` with rank 16."""

    def __init__(self, d_model: int, gate_rank: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.gate_rank = gate_rank
        self.W_1 = nn.Linear(d_model, gate_rank, bias=False)
        self.W_2 = nn.Linear(gate_rank, d_model, bias=False)
        self.b = nn.Parameter(torch.empty(d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.W_1.weight.normal_(mean=0.0, std=0.02)
            self.W_2.weight.normal_(mean=0.0, std=0.02)
            self.b.fill_(GATE_BIAS_INIT)

    def forward(self, x: Tensor) -> Tensor:
        # Keep the bias inside F.linear so autocast casts it with the weights;
        # adding a fp32 Parameter outside would promote the result to fp32 and
        # break dtype agreement with k/v/q under bf16/fp16 autocast.
        return torch.sigmoid(F.linear(self.W_1(x), self.W_2.weight, self.b))


__all__ = ["GATE_BIAS_INIT", "GATE_INIT_MEAN", "FullRankGate", "LowRankGate"]
