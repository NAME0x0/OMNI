from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perspective_torch import (  # noqa: E402
    CausalAttentionBlock,
    FullRankGate,
    LowRankGate,
    SlidingWindowAttentionBlock,
    Stage0Config,
    Stage0LM,
    chunked_linear_recurrence,
    param_table,
    parameter_count,
    sequential_linear_recurrence,
)


@pytest.mark.parametrize("gate_cls", [FullRankGate, LowRankGate])
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [(torch.float64, 1e-6, 1e-6), (torch.float32, 1e-4, 1e-4)],
)
def test_chunked_scan_matches_sequential_reference(gate_cls, dtype, atol, rtol):
    torch.manual_seed(100)
    batch, seq_len, d_model, rank = 2, 300, 16, 5
    gate = gate_cls(d_model).to(dtype=dtype)
    x = 0.1 * torch.randn(batch, seq_len, d_model, dtype=dtype)
    g = gate(x)
    k = 0.1 * torch.randn(batch, seq_len, rank, dtype=dtype)
    v = 0.1 * torch.randn(batch, seq_len, d_model, dtype=dtype)
    q = 0.1 * torch.randn(batch, seq_len, rank, dtype=dtype)
    w_o = 0.1 * torch.randn(d_model, d_model, dtype=dtype)

    seq_out, seq_state = sequential_linear_recurrence(g, k, v, q, output_weight=w_o)
    chunk_out, chunk_state = chunked_linear_recurrence(g, k, v, q, output_weight=w_o, chunk_len=128)

    torch.testing.assert_close(chunk_out, seq_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(chunk_state, seq_state, atol=atol, rtol=rtol)


def test_iso_parameter_variants_within_one_percent():
    table = param_table()
    print(table)
    totals = [parameter_count(Stage0Config(variant=variant)) for variant in ("pdr", "gla", "transformer")]
    assert (max(totals) - min(totals)) / max(totals) < 0.01


@pytest.mark.parametrize("variant", ["pdr", "gla", "transformer"])
def test_parameter_formula_matches_actual_model(variant):
    """The arithmetic in config.py must agree exactly with the real module,
    otherwise the iso-parameter guarantee is only checked on paper."""
    config = Stage0Config(variant=variant)
    model = Stage0LM(config)
    real = sum(p.numel() for p in model.parameters())
    assert real == parameter_count(config), (
        f"{variant}: model has {real:,} params but formula says {parameter_count(config):,}"
    )


@pytest.mark.parametrize("variant", ["pdr", "gla", "transformer"])
def test_forward_shape_all_variants(variant):
    torch.manual_seed(101)
    config = Stage0Config(
        variant=variant,
        d_model=64,
        rank=8,
        n_layers=12,
        vocab_size=50_257,
        n_heads=4,
        n_kv_heads=2,
        sliding_window=16,
        ffn_intermediate=96,
    )
    model = Stage0LM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 64))

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, 64, 50_257)


@pytest.mark.parametrize(
    ("variant", "gate_cls", "param_name"),
    [("pdr", FullRankGate, "W_p"), ("gla", LowRankGate, "W_1")],
)
def test_gate_parameters_receive_gradients(variant, gate_cls, param_name):
    torch.manual_seed(102)
    config = Stage0Config(
        variant=variant,
        d_model=32,
        rank=4,
        n_layers=4,
        vocab_size=97,
        n_heads=4,
        n_kv_heads=2,
        sliding_window=8,
        ffn_intermediate=64,
    )
    model = Stage0LM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    targets = torch.randint(0, config.vocab_size, (2, 12))

    loss = F.cross_entropy(model(input_ids).reshape(-1, config.vocab_size), targets.reshape(-1))
    loss.backward()

    gate = next(module for module in model.modules() if isinstance(module, gate_cls))
    grad = getattr(gate, param_name).weight.grad
    assert grad is not None
    assert grad.norm().item() > 0.0


@pytest.mark.parametrize(
    "block",
    [
        CausalAttentionBlock(d_model=64, n_heads=4),
        SlidingWindowAttentionBlock(d_model=64, n_heads=4, n_kv_heads=2, window=5),
    ],
)
def test_attention_logits_are_causal(block):
    torch.manual_seed(103)
    block = block.double().eval()
    batch, seq_len, d_model, vocab = 2, 16, 64, 113
    perturb_pos = 10
    x = torch.randn(batch, seq_len, d_model, dtype=torch.float64)
    x_perturbed = x.clone()
    x_perturbed[:, perturb_pos, :] += 5.0 * torch.randn(batch, d_model, dtype=torch.float64)
    lm_head = torch.randn(vocab, d_model, dtype=torch.float64)

    with torch.no_grad():
        logits = F.linear(block(x), lm_head)
        perturbed_logits = F.linear(block(x_perturbed), lm_head)

    torch.testing.assert_close(
        perturbed_logits[:, :perturb_pos],
        logits[:, :perturb_pos],
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize("gate_cls", [FullRankGate, LowRankGate])
def test_sigmoid_gate_bounds_and_initial_mean(gate_cls):
    torch.manual_seed(104)
    d_model = 32
    gate = gate_cls(d_model)
    zeros = torch.zeros(4, 7, d_model)
    g0 = gate(zeros)
    assert torch.all(g0 > 0.0)
    assert torch.all(g0 < 1.0)
    assert abs(g0.mean().item() - 0.95) <= 0.02

    random_g = gate(torch.randn(4, 7, d_model))
    assert torch.all(random_g > 0.0)
    assert torch.all(random_g < 1.0)
