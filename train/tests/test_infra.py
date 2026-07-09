from __future__ import annotations

import sys
import shutil
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import make_synthetic_batch_iterator, pack_tokens  # noqa: E402
from perspective_torch import Stage0Config, Stage0LM  # noqa: E402
from trainer import HubCheckpointSync, Stage0Trainer, TrainerConfig, warmup_cosine_lr  # noqa: E402


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


def _tiny_trainer(tmp_path: Path, *, max_hours: float | None = 1.0) -> Stage0Trainer:
    torch.manual_seed(300)
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
    model = Stage0LM(config)
    train_iter = make_synthetic_batch_iterator(seq_len=8, micro_batch_size=2, vocab_size=41, seed=5)
    trainer_config = TrainerConfig(
        output_dir=tmp_path,
        seq_len=8,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=10,
        warmup_steps=2,
        log_interval=1000,
        eval_steps=0,
        save_minutes=1000.0,
        max_hours=max_hours,
        device="cpu",
    )
    return Stage0Trainer(model=model, train_iterator=train_iter, config=trainer_config, hub_sync=HubCheckpointSync(None))


def test_pack_tokens_contiguous_blocks_without_overlap():
    blocks = list(pack_tokens(range(23), seq_len=4))

    assert [tuple(block.shape) for block in blocks] == [(5,), (5,), (5,), (5,)]
    flattened = torch.cat(blocks).tolist()
    assert flattened == list(range(20))


def test_checkpoint_round_trip_preserves_step_optimizer_and_next_batch(work_dir):
    interrupted = _tiny_trainer(work_dir / "interrupted")
    for _ in range(5):
        interrupted.train_one_step()
    checkpoint = interrupted.save_checkpoint(reason="test", wait_for_upload=True)

    resumed = _tiny_trainer(work_dir / "interrupted")
    resumed.load_checkpoint(checkpoint)
    assert resumed.global_step == 5
    assert len(resumed.optimizer.state) > 0
    resumed_loss = resumed.train_one_step()["loss"]

    uninterrupted = _tiny_trainer(work_dir / "uninterrupted")
    uninterrupted_losses = [uninterrupted.train_one_step()["loss"] for _ in range(6)]
    assert resumed_loss == pytest.approx(uninterrupted_losses[-1], abs=1e-7)


def test_budget_guard_exits_and_writes_checkpoint(work_dir):
    trainer = _tiny_trainer(work_dir / "budget", max_hours=0.0)

    trainer.run()

    assert trainer.global_step <= 1
    assert trainer.latest_checkpoint_path() is not None


def test_warmup_cosine_lr_known_values():
    kwargs = {"lr": 0.1, "min_lr": 0.01, "warmup_steps": 2, "max_steps": 6}

    assert warmup_cosine_lr(0, **kwargs) == pytest.approx(0.0)
    assert warmup_cosine_lr(1, **kwargs) == pytest.approx(0.05)
    assert warmup_cosine_lr(2, **kwargs) == pytest.approx(0.1)
    assert warmup_cosine_lr(4, **kwargs) == pytest.approx(0.055)
    assert warmup_cosine_lr(6, **kwargs) == pytest.approx(0.01)


def test_hub_sync_uses_injected_client_without_network(work_dir):
    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        def upload_folder(self, **kwargs):
            self.calls.append(kwargs)

    folder = work_dir / "checkpoints"
    folder.mkdir()
    (folder / "latest.json").write_text("{}", encoding="utf-8")
    fake = FakeClient()

    sync = HubCheckpointSync("user/repo", client=fake, token="token")
    sync.upload_async(folder)
    sync.wait()

    assert len(fake.calls) == 1
    assert fake.calls[0]["repo_id"] == "user/repo"
    assert fake.calls[0]["folder_path"] == str(folder)


def test_hub_sync_creates_repo_before_first_upload_when_client_supports_it(work_dir):
    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        def create_repo(self, repo_id, *, exist_ok, private, token):
            self.calls.append(("create_repo", repo_id, exist_ok, private, token))

        def upload_folder(self, **kwargs):
            self.calls.append(("upload_folder", kwargs))

    folder = work_dir / "checkpoints"
    folder.mkdir()
    (folder / "latest.json").write_text("{}", encoding="utf-8")
    fake = FakeClient()

    sync = HubCheckpointSync("user/repo", client=fake, token="token")
    sync.upload_async(folder)
    sync.wait()
    sync.upload_async(folder)
    sync.wait()

    assert fake.calls[0] == ("create_repo", "user/repo", True, True, "token")
    assert [call[0] for call in fake.calls].count("create_repo") == 1
    assert [call[0] for call in fake.calls].count("upload_folder") == 2


def test_hub_sync_upload_failures_do_not_raise(work_dir):
    class FailingClient:
        def upload_folder(self, **_kwargs):
            raise RuntimeError("boom")

    folder = work_dir / "checkpoints"
    folder.mkdir()
    sync = HubCheckpointSync("user/repo", client=FailingClient(), token="token", log_fn=lambda _message: None)

    sync.upload_async(folder)
    sync.wait()


@pytest.mark.parametrize("variant", ["pdr", "gla", "transformer"])
def test_forward_backward_under_bf16_autocast(variant):
    """Regression: LowRankGate once promoted its output to fp32 under autocast
    (fp32 bias added outside F.linear), crashing the recurrence dtype check.
    One training step per variant must survive bf16 autocast."""
    torch.manual_seed(305)
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
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size), targets.reshape(-1)
        )
    loss.backward()
    assert torch.isfinite(loss)
