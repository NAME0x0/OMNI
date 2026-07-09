from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import make_synthetic_batch_iterator  # noqa: E402
from eval_stage0 import evaluate_checkpoint, format_comparison_report  # noqa: E402
from perspective_torch import Stage0Config, Stage0LM  # noqa: E402
from trainer import HubCheckpointSync, Stage0Trainer, TrainerConfig  # noqa: E402


@pytest.fixture
def work_dir(request):
    root = Path("train/.pytest_tmp")
    path = root / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    yield path
    if path.exists():
        shutil.rmtree(path)


def test_eval_stage0_synthetic_checkpoint_returns_finite_ppl(work_dir):
    torch.manual_seed(501)
    output_dir = work_dir / "smoke-transformer"
    config = Stage0Config(
        variant="transformer",
        d_model=16,
        rank=4,
        n_layers=1,
        vocab_size=41,
        n_heads=4,
        n_kv_heads=1,
        sliding_window=8,
        ffn_intermediate=32,
    )
    trainer = Stage0Trainer(
        model=Stage0LM(config),
        train_iterator=make_synthetic_batch_iterator(seq_len=8, micro_batch_size=2, vocab_size=41, seed=5),
        config=TrainerConfig(
            output_dir=output_dir,
            seq_len=8,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=1,
            warmup_steps=1,
            eval_steps=0,
            save_minutes=1000.0,
            device="cpu",
        ),
        hub_sync=HubCheckpointSync(None),
    )
    trainer.train_one_step()
    checkpoint = trainer.save_checkpoint(reason="test", wait_for_upload=True)

    result = evaluate_checkpoint(
        variant="transformer",
        checkpoint=checkpoint.parent,
        eval_tokens=64,
        device="cpu",
        synthetic=True,
    )

    assert result["eval_tokens"] == 64
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["ppl"])
    assert result["ppl"] > 0.0


def test_comparison_report_gate_pass_and_fail():
    def fake_result(variant: str, ppl: float) -> dict[str, object]:
        return {
            "variant": variant,
            "checkpoint_step": 10,
            "eval_tokens": 100,
            "loss": math.log(ppl),
            "ppl": ppl,
            "data_source": "fake",
        }

    pass_report = format_comparison_report([fake_result("pdr", 9.0), fake_result("gla", 10.0)])
    fail_report = format_comparison_report([fake_result("pdr", 11.0), fake_result("gla", 10.0)])

    assert "GATE PASS: PDR ppl <= GLA ppl (9.000000 <= 10.000000)" in pass_report
    assert "Transformer reference: missing" in pass_report
    assert "GATE FAIL: PDR ppl > GLA ppl (11.000000 > 10.000000)" in fail_report
