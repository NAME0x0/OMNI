"""Interruption-tolerant Stage 0 trainer for free-tier GPU sessions."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import re
import shutil
import signal
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import perspective_torch.layers as stage0_layers

TARGET_TOKENS_PER_STEP = 262_144
DEFAULT_SEQ_LEN = 1024
DEFAULT_MICRO_BATCH_SIZE = 2
DEFAULT_CHUNK_LEN = 64

_ORIGINAL_CHUNKED_RECURRENCE = stage0_layers.chunked_linear_recurrence


def set_default_recurrent_chunk_len(chunk_len: int) -> None:
    """Override the Stage 0 recurrent chunk length without editing model code."""

    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")

    def _chunked_with_default(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("chunk_len", chunk_len)
        return _ORIGINAL_CHUNKED_RECURRENCE(*args, **kwargs)

    stage0_layers.chunked_linear_recurrence = _chunked_with_default


def resolve_gradient_accumulation_steps(
    *, seq_len: int, micro_batch_size: int, target_tokens_per_step: int = TARGET_TOKENS_PER_STEP
) -> int:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    tokens_per_micro_step = seq_len * micro_batch_size
    return max(1, round(target_tokens_per_step / tokens_per_micro_step))


def warmup_cosine_lr(step: int, *, lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Learning-rate value for a completed optimizer step index."""

    if step < 0:
        raise ValueError("step must be non-negative")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step / warmup_steps)
    if step >= max_steps:
        return min_lr
    decay_steps = max(1, max_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (lr - min_lr) * cosine


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
        last_step: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max(1, max_steps)
        self.last_step = last_step
        self.advance_to(last_step)

    def advance_to(self, step: int) -> float:
        self.last_step = int(step)
        value = warmup_cosine_lr(
            self.last_step,
            lr=self.lr,
            min_lr=self.min_lr,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value

    def get_last_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.lr = float(state["lr"])
        self.min_lr = float(state["min_lr"])
        self.warmup_steps = int(state["warmup_steps"])
        self.max_steps = int(state["max_steps"])
        self.advance_to(int(state["last_step"]))


@dataclass
class TrainerConfig:
    output_dir: str | Path
    seq_len: int = DEFAULT_SEQ_LEN
    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE
    gradient_accumulation_steps: int | None = None
    max_steps: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    warmup_steps: int = 2_000
    grad_clip: float = 1.0
    chunk_len: int = DEFAULT_CHUNK_LEN
    log_interval: int = 50
    eval_steps: int = 1_000
    eval_tokens: int = 200_000
    save_minutes: float = 15.0
    max_hours: float | None = 8.5
    hub_repo: str | None = None
    device: str | None = None

    def resolved_grad_accumulation(self) -> int:
        if self.gradient_accumulation_steps is not None:
            if self.gradient_accumulation_steps <= 0:
                raise ValueError("gradient_accumulation_steps must be positive")
            return self.gradient_accumulation_steps
        return resolve_gradient_accumulation_steps(seq_len=self.seq_len, micro_batch_size=self.micro_batch_size)


def build_adamw_optimizer(config: TrainerConfig, model: nn.Module) -> torch.optim.AdamW:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if _is_no_decay_parameter(module, param_name, full_name, param):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=config.betas,
    )


def _is_no_decay_parameter(module: nn.Module, param_name: str, full_name: str, param: nn.Parameter) -> bool:
    module_name = module.__class__.__name__.lower()
    return (
        param_name.endswith("bias")
        or param.ndim < 2
        or isinstance(module, nn.Embedding)
        or module_name.endswith("norm")
        or "embedding" in full_name
    )


class HubCheckpointSync:
    """Best-effort HuggingFace Hub checkpoint sync with injectable clients.

    Hub pruning matters because each Stage 0 checkpoint is roughly 1.5 GB
    (fp32 weights plus AdamW state at 123.5M parameters); without pruning,
    weeks of 15-minute saves would bloat a free Hub repo into the hundreds of
    GB.  Pruning is intentionally best effort and never allowed to stop
    training.
    """

    def __init__(
        self,
        repo_id: str | None,
        *,
        client: Any | None = None,
        token: str | None = None,
        path_in_repo: str = "checkpoints",
        keep_last: int = 2,
        log_fn: Callable[[str], None] = print,
    ) -> None:
        self.repo_id = repo_id
        self.client = client
        self.token = token
        self.path_in_repo = path_in_repo.strip("/")
        self.keep_last = keep_last
        self.log_fn = log_fn
        self._threads: list[threading.Thread] = []
        self._repo_exists_checked = False
        self._repo_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.repo_id)

    def _ensure_repo_exists(self, token: str | None) -> None:
        if not self.enabled or self._repo_exists_checked or not token:
            return
        with self._repo_lock:
            if self._repo_exists_checked:
                return
            try:
                if self.client is not None:
                    create_repo_fn = getattr(self.client, "create_repo", None)
                    if create_repo_fn is None:
                        self._repo_exists_checked = True
                        return
                else:
                    from huggingface_hub import create_repo

                    create_repo_fn = create_repo
                create_repo_fn(self.repo_id, exist_ok=True, private=True, token=token)
                self._repo_exists_checked = True
            except Exception as exc:  # pragma: no cover - network failure path
                self.log_fn(f"Warning: failed to ensure Hub checkpoint repo exists: {exc}")

    def upload_async(self, folder_path: str | Path) -> None:
        if not self.enabled:
            return
        thread = threading.Thread(target=self._upload_folder, args=(Path(folder_path),), daemon=True)
        thread.start()
        self._threads.append(thread)

    def wait(self) -> None:
        for thread in list(self._threads):
            thread.join()
            self._threads.remove(thread)

    def download_latest(self, checkpoint_root: str | Path) -> Path | None:
        if not self.enabled:
            return None
        token = self.token or os.getenv("HF_TOKEN")
        self._ensure_repo_exists(token)
        if self.client is not None and hasattr(self.client, "download_latest"):
            result = self.client.download_latest(self.repo_id, Path(checkpoint_root), self.path_in_repo)
            if result:
                self._download_metrics_file(Path(checkpoint_root), token)
            return Path(result) if result else None

        if not token:
            self.log_fn("HF_TOKEN is not set; skipping checkpoint download")
            return None
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            self.log_fn("huggingface_hub is not installed; skipping checkpoint download")
            return None

        try:
            latest_src = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"{self.path_in_repo}/latest.json",
                token=token,
            )
            latest = json.loads(Path(latest_src).read_text(encoding="utf-8"))
            checkpoint_name = latest["latest"]
            checkpoint_dir = Path(checkpoint_root) / checkpoint_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for filename in ("state.pt", "metadata.json"):
                src = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=f"{self.path_in_repo}/{checkpoint_name}/{filename}",
                    token=token,
                )
                shutil.copy2(src, checkpoint_dir / filename)
            self._download_metrics_file(Path(checkpoint_root), token)
            return checkpoint_dir / "state.pt"
        except Exception as exc:  # pragma: no cover - network failure path
            self.log_fn(f"Hub checkpoint download failed: {exc}")
            return None

    def prune_old_hub_checkpoints(self, keep_last: int | None = None) -> None:
        """Delete remote checkpoint folders older than the newest ``keep_last``.

        Each Stage 0 checkpoint is about 1.5 GB (fp32 weights plus AdamW state
        for a 123.5M parameter model).  If 15-minute checkpoint folders are
        never pruned, a multi-week free-tier run can grow into hundreds of GB
        on the Hub.  This method only removes ``checkpoint-step-*`` folders and
        leaves root files such as ``latest.json`` and ``metrics.jsonl`` intact.
        Missing client capabilities are treated as "pruning unavailable" and
        skipped silently; real Hub failures are logged as warnings.
        """

        if not self.enabled:
            return
        keep = self.keep_last if keep_last is None else int(keep_last)
        if keep < 0:
            return

        token = self.token or os.getenv("HF_TOKEN")
        try:
            client = self.client
            if client is None:
                try:
                    from huggingface_hub import HfApi
                except ImportError:
                    return

                client = HfApi()

            files = self._list_repo_files(client, token)
            if files is None:
                return
            folders = self._checkpoint_folders_from_repo_files(files)
            doomed = folders[:-keep] if keep else folders
            if not doomed:
                return
            if not self._delete_hub_folders(client, doomed, token):
                return
            self.log_fn(
                "Hub checkpoint pruning deleted "
                f"{len(doomed)} old folder(s): {', '.join(path.rsplit('/', 1)[-1] for path in doomed)}"
            )
        except Exception as exc:  # pragma: no cover - network failure path
            self.log_fn(f"Warning: failed to prune old Hub checkpoints: {exc}")

    def _upload_folder(self, folder_path: Path) -> None:
        token = self.token or os.getenv("HF_TOKEN")
        if not token:
            self.log_fn("HF_TOKEN is not set; skipping checkpoint upload")
            return
        try:
            self._ensure_repo_exists(token)
            client = self.client
            if client is None:
                from huggingface_hub import HfApi

                client = HfApi()
            client.upload_folder(
                repo_id=self.repo_id,
                folder_path=str(folder_path),
                path_in_repo=self.path_in_repo,
                token=token,
            )
            self.prune_old_hub_checkpoints(self.keep_last)
        except Exception as exc:
            self.log_fn(f"Hub checkpoint upload failed: {exc}")

    def _download_metrics_file(self, checkpoint_root: Path, token: str | None) -> None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        remote_name = f"{self.path_in_repo}/metrics.jsonl" if self.path_in_repo else "metrics.jsonl"
        target = checkpoint_root / "metrics.jsonl"

        if self.client is not None:
            download_file_fn = getattr(self.client, "download_file", None)
            if download_file_fn is None:
                return
            try:
                result = self._call_download_file(download_file_fn, remote_name, target, token)
                if result is not None:
                    self._copy_download_result(result, target)
            except FileNotFoundError:
                return
            except Exception as exc:
                self.log_fn(f"Warning: failed to download Hub metrics.jsonl: {exc}")
            return

        try:
            from huggingface_hub import hf_hub_download

            src = hf_hub_download(repo_id=self.repo_id, filename=remote_name, token=token)
            shutil.copy2(src, target)
        except Exception as exc:  # pragma: no cover - network/missing-file path
            self.log_fn(f"Warning: failed to download Hub metrics.jsonl: {exc}")

    def _call_download_file(
        self,
        download_file_fn: Callable[..., Any],
        remote_name: str,
        target: Path,
        token: str | None,
    ) -> Any:
        try:
            return download_file_fn(
                repo_id=self.repo_id,
                filename=remote_name,
                local_path=target,
                token=token,
            )
        except TypeError:
            try:
                return download_file_fn(self.repo_id, remote_name, target, token=token)
            except TypeError:
                return download_file_fn(self.repo_id, remote_name, target)

    def _copy_download_result(self, result: Any, target: Path) -> None:
        src = Path(result)
        if src.exists() and src.resolve() != target.resolve():
            shutil.copy2(src, target)

    def _list_repo_files(self, client: Any, token: str | None) -> list[str] | None:
        list_fn = getattr(client, "list_repo_files", None)
        if list_fn is None:
            return None
        try:
            return list(list_fn(repo_id=self.repo_id, token=token))
        except TypeError:
            try:
                return list(list_fn(self.repo_id, token=token))
            except TypeError:
                return list(list_fn(self.repo_id))

    def _checkpoint_folders_from_repo_files(self, files: list[str]) -> list[str]:
        base = f"{self.path_in_repo}/" if self.path_in_repo else ""
        pattern = re.compile(rf"^{re.escape(base)}(checkpoint-step-(\d+))(?:/|$)")
        by_step: dict[int, str] = {}
        for filename in files:
            normalized = str(filename).replace("\\", "/")
            match = pattern.match(normalized)
            if match:
                by_step[int(match.group(2))] = f"{base}{match.group(1)}"
        return [folder for _step, folder in sorted(by_step.items())]

    def _delete_hub_folders(self, client: Any, folders: list[str], token: str | None) -> bool:
        delete_folder_fn = getattr(client, "delete_folder", None)
        if delete_folder_fn is not None:
            for folder in folders:
                self._call_delete_folder(delete_folder_fn, folder, token)
            return True

        create_commit_fn = getattr(client, "create_commit", None)
        if create_commit_fn is None:
            return False
        try:
            from huggingface_hub import CommitOperationDelete
        except ImportError:
            return False

        operations = [CommitOperationDelete(path_in_repo=folder, is_folder=True) for folder in folders]
        message = f"Prune old Stage 0 checkpoints, keep last {self.keep_last}"
        try:
            create_commit_fn(repo_id=self.repo_id, operations=operations, commit_message=message, token=token)
        except TypeError:
            create_commit_fn(self.repo_id, operations, message)
        return True

    def _call_delete_folder(self, delete_folder_fn: Callable[..., Any], folder: str, token: str | None) -> None:
        message = f"Prune old Stage 0 checkpoint {folder.rsplit('/', 1)[-1]}"
        try:
            delete_folder_fn(
                repo_id=self.repo_id,
                path_in_repo=folder,
                token=token,
                commit_message=message,
            )
        except TypeError:
            try:
                delete_folder_fn(self.repo_id, folder, token=token, commit_message=message)
            except TypeError:
                delete_folder_fn(self.repo_id, folder)


class Stage0Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        train_iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
        config: TrainerConfig,
        eval_iterator_factory: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]] | None = None,
        hub_sync: HubCheckpointSync | None = None,
    ) -> None:
        self.config = config
        self.grad_accumulation_steps = config.resolved_grad_accumulation()
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        set_default_recurrent_chunk_len(config.chunk_len)

        self.model = model.to(self.device)
        self.train_iterator = train_iterator
        self.eval_iterator_factory = eval_iterator_factory
        self.output_dir = Path(config.output_dir)
        self.checkpoint_root = self.output_dir / "checkpoints"
        self.metrics_path = self.checkpoint_root / "metrics.jsonl"
        self.optimizer = build_adamw_optimizer(config, self.model)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            lr=config.lr,
            min_lr=config.min_lr,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )
        self.autocast_dtype, scaler_enabled = self._precision_policy()
        self.scaler = _make_grad_scaler(enabled=scaler_enabled)
        self.hub_sync = hub_sync or HubCheckpointSync(config.hub_repo)

        self.global_step = 0
        self._start_time = time.time()
        self._last_save_time = self._start_time
        self._last_log_time = self._start_time
        self._tokens_since_log = 0
        self._stop_requested = False
        self._previous_handlers: dict[int, Any] = {}

    def train_one_step(self) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_tokens = 0

        for _ in range(self.grad_accumulation_steps):
            input_ids, targets = next(self.train_iterator)
            input_ids = input_ids.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with self._autocast_context():
                logits = self.model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                scaled_loss = loss / self.grad_accumulation_steps

            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            total_loss += float(loss.detach().float().item())
            total_tokens += int(targets.numel())

        self.scheduler.advance_to(self.global_step + 1)
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.global_step += 1
        lr = self.scheduler.get_last_lr()
        return {
            "loss": total_loss / self.grad_accumulation_steps,
            "tokens": float(total_tokens),
            "lr": lr,
            "grad_norm": float(grad_norm.detach().float().item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
        }

    def run(self) -> None:
        self._start_time = time.time()
        self._last_save_time = self._start_time
        self._last_log_time = self._start_time
        self._install_signal_handlers()
        try:
            if self._budget_exceeded():
                self.save_checkpoint(reason="budget", wait_for_upload=True)
                return
            while self.global_step < self.config.max_steps:
                metrics = self.train_one_step()
                self._tokens_since_log += int(metrics["tokens"])

                if self.global_step % self.config.log_interval == 0:
                    self.log_metrics(metrics)
                if self.config.eval_steps > 0 and self.global_step % self.config.eval_steps == 0:
                    self.evaluate_and_log()
                if self._save_due():
                    self.save_checkpoint(reason="periodic")
                if self._stop_requested:
                    self.save_checkpoint(reason="signal", wait_for_upload=True)
                    return
                if self._budget_exceeded():
                    self.save_checkpoint(reason="budget", wait_for_upload=True)
                    return
            self.save_checkpoint(reason="complete", wait_for_upload=True)
        except KeyboardInterrupt:
            self.save_checkpoint(reason="keyboard_interrupt", wait_for_upload=True)
        finally:
            self._restore_signal_handlers()

    @torch.no_grad()
    def evaluate_and_log(self) -> dict[str, float] | None:
        if self.eval_iterator_factory is None:
            return None

        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        eval_iter = self.eval_iterator_factory()

        while total_tokens < self.config.eval_tokens:
            try:
                input_ids, targets = next(eval_iter)
            except StopIteration:
                break
            remaining = self.config.eval_tokens - total_tokens
            input_ids = input_ids.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with self._autocast_context():
                logits = self.model(input_ids)
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                )
            take = min(remaining, int(losses.numel()))
            total_loss += float(losses[:take].detach().float().sum().item())
            total_tokens += take

        if was_training:
            self.model.train()
        if total_tokens == 0:
            return None

        eval_loss = total_loss / total_tokens
        metrics = {
            "eval_loss": eval_loss,
            "eval_ppl": float(math.exp(min(eval_loss, 20.0))),
            "eval_tokens": float(total_tokens),
            "lr": self.scheduler.get_last_lr(),
        }
        self._write_metric({"step": self.global_step, **metrics})
        print(
            f"eval step={self.global_step} loss={metrics['eval_loss']:.4f} "
            f"ppl={metrics['eval_ppl']:.2f} tokens={int(metrics['eval_tokens'])}",
            flush=True,
        )
        return metrics

    def log_metrics(self, metrics: dict[str, float]) -> None:
        now = time.time()
        elapsed = max(1e-9, now - self._last_log_time)
        tokens_per_sec = self._tokens_since_log / elapsed
        record = {
            "step": self.global_step,
            "loss": metrics["loss"],
            "tokens_per_sec": tokens_per_sec,
            "lr": metrics["lr"],
            "grad_norm": metrics["grad_norm"],
            "tokens": int(metrics["tokens"]),
            **self._progress_fields(tokens_per_sec=tokens_per_sec),
        }
        self._write_metric(record)
        eta_hours = record["eta_hours"]
        eta_text = "unknown" if eta_hours is None else f"{eta_hours:.2f}h"
        print(
            f"step={self.global_step} loss={metrics['loss']:.4f} "
            f"tokens_done={record['tokens_total']:,} pct={record['pct_complete']:.2f}% eta={eta_text} "
            f"tokens/sec={tokens_per_sec:.1f} lr={metrics['lr']:.6g} "
            f"grad_norm={metrics['grad_norm']:.3f}",
            flush=True,
        )
        self._tokens_since_log = 0
        self._last_log_time = now

    def save_checkpoint(self, *, reason: str, wait_for_upload: bool = False) -> Path:
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = self.checkpoint_root / f"checkpoint-step-{self.global_step:010d}"
        tmp_dir = self.checkpoint_root / f".tmp-step-{self.global_step:010d}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        state = {
            "step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": _rng_state(),
            "data_state": self._data_state(),
            "data_skip_documents": self._data_skip_documents(),
            "trainer_config": asdict(self.config),
            "model_config": self._model_config_state(),
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "reason": reason,
        }
        torch.save(state, tmp_dir / "state.pt")
        metadata = {
            "step": self.global_step,
            "reason": reason,
            "created_unix": time.time(),
            "data_skip_documents": state["data_skip_documents"],
        }
        (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        tmp_dir.rename(checkpoint_dir)
        latest = {"latest": checkpoint_dir.name, "step": self.global_step, "reason": reason}
        (self.checkpoint_root / "latest.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")
        self._rotate_checkpoints(keep=2)
        self._last_save_time = time.time()
        print(f"checkpoint saved step={self.global_step} reason={reason} path={checkpoint_dir}", flush=True)
        self.hub_sync.upload_async(self.checkpoint_root)
        if wait_for_upload:
            self.hub_sync.wait()
        return checkpoint_dir / "state.pt"

    def load_checkpoint(self, checkpoint_path: str | Path | None = None) -> Path | None:
        path = Path(checkpoint_path) if checkpoint_path is not None else self.latest_checkpoint_path()
        if path is None and self.hub_sync.enabled:
            path = self.hub_sync.download_latest(self.checkpoint_root)
        if path is None:
            return None

        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state.get("scaler", {}))
        self.global_step = int(state["step"])
        if hasattr(self.train_iterator, "load_state_dict"):
            self.train_iterator.load_state_dict(state.get("data_state", {}))  # type: ignore[attr-defined]
        _restore_rng_state(state.get("rng", {}), self.device)
        print(f"checkpoint loaded step={self.global_step} path={path}", flush=True)
        return path

    def latest_checkpoint_path(self) -> Path | None:
        latest_file = self.checkpoint_root / "latest.json"
        if latest_file.exists():
            latest = json.loads(latest_file.read_text(encoding="utf-8"))
            candidate = self.checkpoint_root / latest["latest"] / "state.pt"
            if candidate.exists():
                return candidate
        checkpoints = sorted(self.checkpoint_root.glob("checkpoint-step-*/state.pt"))
        return checkpoints[-1] if checkpoints else None

    def _precision_policy(self) -> tuple[torch.dtype | None, bool]:
        if self.device.type != "cuda":
            return None, False
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        return torch.float16, True

    def _autocast_context(self) -> contextlib.AbstractContextManager[Any]:
        if self.device.type == "cuda" and self.autocast_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
        return contextlib.nullcontext()

    def _write_metric(self, record: dict[str, Any]) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _progress_fields(self, *, tokens_per_sec: float | None = None) -> dict[str, float | int | None]:
        tokens_per_step = self.config.seq_len * self.config.micro_batch_size * self.grad_accumulation_steps
        tokens_total = int(self.global_step * tokens_per_step)
        target_tokens = max(1, int(self.config.max_steps * tokens_per_step))
        pct_complete = min(100.0, 100.0 * tokens_total / target_tokens)
        eta_hours = None
        if tokens_per_sec is not None and tokens_per_sec > 0 and tokens_total < target_tokens:
            eta_hours = (target_tokens - tokens_total) / tokens_per_sec / 3600.0
        return {
            "tokens_total": tokens_total,
            "pct_complete": pct_complete,
            "eta_hours": eta_hours,
        }

    def _save_due(self) -> bool:
        return (time.time() - self._last_save_time) >= self.config.save_minutes * 60.0

    def _budget_exceeded(self) -> bool:
        if self.config.max_hours is None:
            return False
        return (time.time() - self._start_time) >= self.config.max_hours * 3600.0

    def _data_state(self) -> dict[str, Any] | None:
        if hasattr(self.train_iterator, "state_dict"):
            return self.train_iterator.state_dict()  # type: ignore[attr-defined]
        return None

    def _data_skip_documents(self) -> int | None:
        state = self._data_state() or {}
        block_state = state.get("block_iterator") if isinstance(state, dict) else None
        if isinstance(block_state, dict):
            value = block_state.get("data_skip_documents", block_state.get("documents_seen"))
            return int(value) if value is not None else None
        value = state.get("data_skip_documents", state.get("documents_seen")) if isinstance(state, dict) else None
        return int(value) if value is not None else None

    def _model_config_state(self) -> dict[str, Any] | None:
        config = getattr(self.model, "config", None)
        if config is None:
            return None
        try:
            return asdict(config)
        except TypeError:
            return dict(getattr(config, "__dict__", {}))

    def _rotate_checkpoints(self, *, keep: int) -> None:
        checkpoints = sorted(
            [path for path in self.checkpoint_root.glob("checkpoint-step-*") if path.is_dir()],
            key=lambda path: path.name,
        )
        for old in checkpoints[:-keep]:
            shutil.rmtree(old)

    def _install_signal_handlers(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._request_stop)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()

    def _request_stop(self, signum: int, _frame: Any) -> None:
        self._stop_requested = True
        print(f"received signal {signum}; checkpointing at next safe point", flush=True)


def _make_grad_scaler(*, enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # pragma: no cover - older torch compatibility
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state.get("cuda") is not None:
        cuda_states = []
        for item in state["cuda"]:
            if isinstance(item, torch.Tensor):
                cuda_states.append(item.detach().cpu().to(dtype=torch.uint8))
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)


__all__ = [
    "DEFAULT_CHUNK_LEN",
    "DEFAULT_MICRO_BATCH_SIZE",
    "DEFAULT_SEQ_LEN",
    "TARGET_TOKENS_PER_STEP",
    "HubCheckpointSync",
    "Stage0Trainer",
    "TrainerConfig",
    "WarmupCosineScheduler",
    "build_adamw_optimizer",
    "resolve_gradient_accumulation_steps",
    "set_default_recurrent_chunk_len",
    "warmup_cosine_lr",
]
