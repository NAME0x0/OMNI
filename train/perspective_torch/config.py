"""Configuration and parameter accounting for Stage 0 models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

Variant = Literal["pdr", "gla", "transformer"]

BASE_D_MODEL = 768
BASE_RANK = 48
BASE_LAYERS = 12
BASE_VOCAB_SIZE = 50_257
BASE_N_HEADS = 12
BASE_N_KV_HEADS = 4
BASE_WINDOW = 256
LOW_RANK_GATE_RANK = 16

# Iso-parameter arithmetic, with bias-free non-gate projections:
# non-FFN totals at d=768, rank=48, vocab=50257:
#   PDR = 59,921,664; GLA = 54,834,432; transformer = 66,918,912.
# Use the largest base variant, transformer, with FFN=2048 as target:
#   66,918,912 + 12 * (3 * 768 * 2048 + 768) = 123,551,232.
# Solving round((target - non_ffn - 12 * 768) / (12 * 3 * 768)) gives:
PDR_FFN_INTERMEDIATE = 2301
GLA_FFN_INTERMEDIATE = 2485
TRANSFORMER_FFN_INTERMEDIATE = 2048


def default_ffn_intermediate(variant: Variant) -> int:
    if variant == "pdr":
        return PDR_FFN_INTERMEDIATE
    if variant == "gla":
        return GLA_FFN_INTERMEDIATE
    if variant == "transformer":
        return TRANSFORMER_FFN_INTERMEDIATE
    raise ValueError(f"unknown variant: {variant}")


@dataclass(frozen=True)
class Stage0Config:
    """Dataclass for Stage 0 language-model variants."""

    variant: Variant = "pdr"
    d_model: int = BASE_D_MODEL
    rank: int = BASE_RANK
    n_layers: int = BASE_LAYERS
    vocab_size: int = BASE_VOCAB_SIZE
    n_heads: int = BASE_N_HEADS
    n_kv_heads: int = BASE_N_KV_HEADS
    sliding_window: int = BASE_WINDOW
    low_rank_gate_rank: int = LOW_RANK_GATE_RANK
    ffn_intermediate: int | None = None

    def __post_init__(self) -> None:
        if self.variant not in ("pdr", "gla", "transformer"):
            raise ValueError(f"unknown variant: {self.variant}")
        if self.ffn_intermediate is None:
            object.__setattr__(self, "ffn_intermediate", default_ffn_intermediate(self.variant))
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    def with_variant(self, variant: Variant) -> "Stage0Config":
        return replace(self, variant=variant, ffn_intermediate=default_ffn_intermediate(variant))


def recurrent_layer_count(n_layers: int) -> int:
    return sum(1 for i in range(n_layers) if i % 4 != 3)


def sliding_attention_layer_count(n_layers: int) -> int:
    return n_layers - recurrent_layer_count(n_layers)


def _embedding_params(config: Stage0Config) -> int:
    return config.vocab_size * config.d_model


def _ffn_params(config: Stage0Config) -> int:
    return config.n_layers * (3 * config.d_model * int(config.ffn_intermediate) + config.d_model)


def _recurrent_mixer_params(config: Stage0Config) -> int:
    d = config.d_model
    r = config.rank
    base = d + 2 * d * r + 2 * d * d
    if config.variant == "pdr":
        gate = d * d + d
    elif config.variant == "gla":
        gate = 2 * config.low_rank_gate_rank * d + d
    else:
        gate = 0
    return recurrent_layer_count(config.n_layers) * (base + gate)


def _sliding_attention_params(config: Stage0Config) -> int:
    d = config.d_model
    head_dim = d // config.n_heads
    kv_dim = config.n_kv_heads * head_dim
    per_layer = d + 2 * d * d + 2 * d * kv_dim
    return sliding_attention_layer_count(config.n_layers) * per_layer


def _causal_attention_params(config: Stage0Config) -> int:
    d = config.d_model
    return config.n_layers * (d + 4 * d * d)


def parameter_breakdown(config: Stage0Config) -> dict[str, int]:
    """Return exact parameter counts by component for a config."""

    embedding = _embedding_params(config)
    if config.variant == "transformer":
        recurrent = 0
        sliding = 0
        causal = _causal_attention_params(config)
    else:
        recurrent = _recurrent_mixer_params(config)
        sliding = _sliding_attention_params(config)
        causal = 0
    ffn = _ffn_params(config)
    final_norm = config.d_model
    total = embedding + recurrent + sliding + causal + ffn + final_norm
    return {
        "embedding": embedding,
        "recurrent": recurrent,
        "sliding_attention": sliding,
        "causal_attention": causal,
        "ffn": ffn,
        "final_norm": final_norm,
        "total": total,
    }


def parameter_count(config: Stage0Config) -> int:
    return parameter_breakdown(config)["total"]


def _fmt(value: int) -> str:
    return f"{value:,}"


def param_table(*, print_table: bool = True) -> str:
    """Print and return the default Stage 0 parameter table."""

    configs = [Stage0Config(variant=v) for v in ("pdr", "gla", "transformer")]
    rows = []
    for config in configs:
        breakdown = parameter_breakdown(config)
        mixer = breakdown["recurrent"] + breakdown["sliding_attention"] + breakdown["causal_attention"]
        rows.append(
            [
                config.variant,
                str(config.ffn_intermediate),
                _fmt(breakdown["total"]),
                _fmt(breakdown["embedding"]),
                _fmt(mixer),
                _fmt(breakdown["ffn"]),
                _fmt(breakdown["final_norm"]),
            ]
        )

    headers = ["variant", "ffn", "total", "embedding", "mixers", "ffn_params", "final_norm"]
    widths = [max(len(row[i]) for row in rows + [headers]) for i in range(len(headers))]
    lines = ["  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows)
    table = "\n".join(lines)
    if print_table:
        print(table)
    return table


__all__ = [
    "BASE_D_MODEL",
    "BASE_LAYERS",
    "BASE_RANK",
    "BASE_VOCAB_SIZE",
    "GLA_FFN_INTERMEDIATE",
    "LOW_RANK_GATE_RANK",
    "PDR_FFN_INTERMEDIATE",
    "Stage0Config",
    "TRANSFORMER_FFN_INTERMEDIATE",
    "Variant",
    "default_ffn_intermediate",
    "param_table",
    "parameter_breakdown",
    "parameter_count",
    "recurrent_layer_count",
    "sliding_attention_layer_count",
]
