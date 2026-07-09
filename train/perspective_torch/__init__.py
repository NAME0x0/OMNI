"""PyTorch Stage 0 models for the PERSPECTIVE research codebase."""

from .config import (
    GLA_FFN_INTERMEDIATE,
    PDR_FFN_INTERMEDIATE,
    TRANSFORMER_FFN_INTERMEDIATE,
    Stage0Config,
    param_table,
    parameter_breakdown,
    parameter_count,
)
from .gates import FullRankGate, LowRankGate
from .layers import CausalAttentionBlock, RecurrentBlock, SlidingWindowAttentionBlock, SwiGLUFFN
from .model import Stage0LM, count_parameters
from .recurrence import chunked_linear_recurrence, linear_recurrence, sequential_linear_recurrence

__all__ = [
    "CausalAttentionBlock",
    "FullRankGate",
    "GLA_FFN_INTERMEDIATE",
    "LowRankGate",
    "PDR_FFN_INTERMEDIATE",
    "RecurrentBlock",
    "SlidingWindowAttentionBlock",
    "Stage0Config",
    "Stage0LM",
    "SwiGLUFFN",
    "TRANSFORMER_FFN_INTERMEDIATE",
    "chunked_linear_recurrence",
    "count_parameters",
    "linear_recurrence",
    "param_table",
    "parameter_breakdown",
    "parameter_count",
    "sequential_linear_recurrence",
]
