"""Standalone held-out evaluation for Stage 0 checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn import functional as F

from data import FineWebDataConfig, make_fineweb_eval_iterator, make_synthetic_batch_iterator
from perspective_torch import Stage0Config, Stage0LM
from trainer import DEFAULT_MICRO_BATCH_SIZE, DEFAULT_SEQ_LEN, HubCheckpointSync

VARIANTS = ("pdr", "gla", "transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 0 checkpoints on the held-out eval slice.")
    parser.add_argument("--variant", choices=VARIANTS, help="Model variant for single-checkpoint evaluation.")
    parser.add_argument("--checkpoint", help="Local checkpoint file/dir/root or Hugging Face Hub repo id.")
    parser.add_argument("--eval-tokens", type=int, default=1_000_000)
    parser.add_argument("--eval-docs", type=int, default=2_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compare", action="store_true", help="Evaluate all three stage0-* repos for a Hub user.")
    parser.add_argument("--hub-user", help="Hub username or organization used with --compare.")
    parser.add_argument("--synthetic", action="store_true", help="Use deterministic synthetic data instead of FineWeb.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare:
        if not args.hub_user:
            raise SystemExit("--compare requires --hub-user")
        results, missing = compare_hub_variants(
            hub_user=args.hub_user,
            eval_tokens=args.eval_tokens,
            eval_docs=args.eval_docs,
            device=args.device,
            synthetic=args.synthetic,
        )
        print(format_comparison_report(results, missing=missing), flush=True)
        return

    if not args.variant or not args.checkpoint:
        raise SystemExit("single-checkpoint evaluation requires --variant and --checkpoint")
    result = evaluate_checkpoint(
        variant=args.variant,
        checkpoint=args.checkpoint,
        eval_tokens=args.eval_tokens,
        eval_docs=args.eval_docs,
        device=args.device,
        synthetic=args.synthetic,
    )
    print(format_single_report(result), flush=True)


def compare_hub_variants(
    *,
    hub_user: str,
    eval_tokens: int,
    eval_docs: int = 2_000,
    device: str | None = None,
    synthetic: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for variant in VARIANTS:
        repo_id = f"{hub_user}/stage0-{variant}"
        print(f"\nEvaluating {variant}: repo={repo_id}", flush=True)
        try:
            results.append(
                evaluate_checkpoint(
                    variant=variant,
                    checkpoint=repo_id,
                    eval_tokens=eval_tokens,
                    eval_docs=eval_docs,
                    device=device,
                    synthetic=synthetic,
                )
            )
        except Exception as exc:  # pragma: no cover - network/auth failures vary
            message = f"{variant}: skipped {repo_id} ({exc})"
            print(f"SKIP: {message}", flush=True)
            missing.append(message)
    return results, missing


def evaluate_checkpoint(
    *,
    variant: str,
    checkpoint: str | Path,
    eval_tokens: int = 1_000_000,
    eval_docs: int = 2_000,
    device: str | None = None,
    synthetic: bool = False,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    if eval_tokens <= 0:
        raise ValueError("eval_tokens must be positive")

    checkpoint_path = resolve_checkpoint_path(checkpoint, cache_dir=cache_dir)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    model_config = stage0_config_from_checkpoint(variant=variant, state=state)
    model = Stage0LM(model_config).to(torch_device)
    model.load_state_dict(state["model"])
    model.eval()

    trainer_config = state.get("trainer_config") or {}
    seq_len = int(trainer_config.get("seq_len", DEFAULT_SEQ_LEN))
    micro_batch_size = int(trainer_config.get("micro_batch_size", DEFAULT_MICRO_BATCH_SIZE))
    use_synthetic = synthetic or _looks_like_smoke_checkpoint(state, checkpoint_path)
    data_source = "synthetic-smoke" if use_synthetic else "fineweb-heldout"
    iterator = _make_eval_iterator(
        model_config=model_config,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        eval_docs=eval_docs,
        synthetic=use_synthetic,
    )

    loss, actual_tokens = evaluate_model(model, iterator, eval_tokens=eval_tokens, device=torch_device)
    return {
        "variant": variant,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(state.get("step", _step_from_path(checkpoint_path) or -1)),
        "eval_tokens": actual_tokens,
        "loss": loss,
        "ppl": float(math.exp(min(loss, 20.0))),
        "data_source": data_source,
    }


@torch.no_grad()
def evaluate_model(
    model: Stage0LM,
    iterator: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    eval_tokens: int,
    device: torch.device,
) -> tuple[float, int]:
    total_loss = 0.0
    total_tokens = 0

    for input_ids, targets in iterator:
        if total_tokens >= eval_tokens:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast_context(device):
            logits = model(input_ids)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            )
        take = min(eval_tokens - total_tokens, int(losses.numel()))
        total_loss += float(losses[:take].detach().float().sum().item())
        total_tokens += take

    if total_tokens == 0:
        raise RuntimeError("evaluation produced zero tokens; check eval data availability")
    return total_loss / total_tokens, total_tokens


def resolve_checkpoint_path(checkpoint: str | Path, *, cache_dir: str | Path | None = None) -> Path:
    raw = str(checkpoint)
    local = Path(raw).expanduser()
    if local.exists():
        return latest_checkpoint_path(local)

    repo_id = raw
    root = Path(cache_dir or os.getenv("STAGE0_EVAL_CACHE", "train/runs/eval-cache"))
    repo_cache = root / re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id)
    state_path = HubCheckpointSync(repo_id).download_latest(repo_cache)
    if state_path is None:
        raise FileNotFoundError(f"could not download latest checkpoint from Hub repo {repo_id!r}")
    return Path(state_path)


def latest_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path
    direct = path / "state.pt"
    if direct.exists():
        return direct
    latest_file = path / "latest.json"
    if latest_file.exists():
        latest = json.loads(latest_file.read_text(encoding="utf-8"))
        candidate = path / latest["latest"] / "state.pt"
        if candidate.exists():
            return candidate
    checkpoints = sorted(path.glob("checkpoint-step-*/state.pt"))
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"no checkpoint state.pt found under {path}")


def stage0_config_from_checkpoint(*, variant: str, state: dict[str, Any]) -> Stage0Config:
    saved = state.get("model_config")
    if isinstance(saved, dict):
        valid_fields = {field.name for field in fields(Stage0Config)}
        kwargs = {key: value for key, value in saved.items() if key in valid_fields}
        kwargs["variant"] = variant
        return Stage0Config(**kwargs)
    return _infer_stage0_config(variant=variant, state=state)


def _infer_stage0_config(*, variant: str, state: dict[str, Any]) -> Stage0Config:
    model_state = state["model"]
    vocab_size, d_model = model_state["token_embedding.weight"].shape
    layer_indices = {
        int(match.group(1))
        for key in model_state
        if (match := re.match(r"blocks\.(\d+)\.", key))
    }
    n_layers = max(layer_indices) + 1 if layer_indices else 0
    ffn_intermediate = int(model_state["blocks.0.ffn.W_gate.weight"].shape[0])
    trainer_config = state.get("trainer_config") or {}

    n_heads = _infer_n_heads(d_model)
    rank = 48
    n_kv_heads = min(4, n_heads)
    low_rank_gate_rank = 16
    if variant in {"pdr", "gla"}:
        recurrent_key = next((key for key in model_state if key.endswith(".mixer.W_k.weight")), None)
        if recurrent_key:
            rank = int(model_state[recurrent_key].shape[0])
        sliding_key = next(
            (
                key
                for key in model_state
                if key.endswith(".mixer.W_k.weight")
                and int(re.match(r"blocks\.(\d+)\.", key).group(1)) % 4 == 3
            ),
            None,
        )
        if sliding_key:
            head_dim = d_model // n_heads
            n_kv_heads = max(1, int(model_state[sliding_key].shape[0]) // head_dim)
        gla_gate_key = next((key for key in model_state if key.endswith(".mixer.gate.W_1.weight")), None)
        if gla_gate_key:
            low_rank_gate_rank = int(model_state[gla_gate_key].shape[0])

    default_window = 256 if d_model == 768 else int(trainer_config.get("seq_len", 256))
    return Stage0Config(
        variant=variant,
        d_model=int(d_model),
        rank=int(rank),
        n_layers=int(n_layers),
        vocab_size=int(vocab_size),
        n_heads=int(n_heads),
        n_kv_heads=int(n_kv_heads),
        sliding_window=int(default_window),
        low_rank_gate_rank=int(low_rank_gate_rank),
        ffn_intermediate=int(ffn_intermediate),
    )


def _infer_n_heads(d_model: int) -> int:
    if d_model % 12 == 0:
        return 12
    if d_model % 8 == 0 and d_model >= 128:
        return 8
    if d_model % 4 == 0:
        return 4
    return 1


def _make_eval_iterator(
    *,
    model_config: Stage0Config,
    seq_len: int,
    micro_batch_size: int,
    eval_docs: int,
    synthetic: bool,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    if synthetic:
        return make_synthetic_batch_iterator(
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            vocab_size=model_config.vocab_size,
            seed=23,
        )
    data_config = FineWebDataConfig(seq_len=seq_len, eval_docs=eval_docs)
    return make_fineweb_eval_iterator(data_config, micro_batch_size=micro_batch_size)


def _looks_like_smoke_checkpoint(state: dict[str, Any], checkpoint_path: Path) -> bool:
    trainer_config = state.get("trainer_config") or {}
    output_dir = str(trainer_config.get("output_dir", "")).lower()
    if "smoke" in output_dir or "smoke" in str(checkpoint_path).lower():
        return True
    model_config = state.get("model_config") or {}
    return bool(model_config and int(model_config.get("d_model", 768)) < 128 and int(trainer_config.get("max_steps", 0)) <= 200)


def _autocast_context(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _step_from_path(path: Path) -> int | None:
    match = re.search(r"checkpoint-step-(\d+)", str(path))
    return int(match.group(1)) if match else None


def format_single_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Stage 0 held-out evaluation report",
            f"variant: {result['variant']}",
            f"checkpoint: {result['checkpoint']}",
            f"checkpoint step: {result['checkpoint_step']}",
            f"eval tokens: {result['eval_tokens']:,}",
            f"data source: {result['data_source']}",
            f"loss: {result['loss']:.6f}",
            f"perplexity: {result['ppl']:.6f}",
        ]
    )


def format_comparison_report(results: list[dict[str, Any]], *, missing: list[str] | None = None) -> str:
    by_variant = {result["variant"]: result for result in results}
    lines = ["Stage 0 comparison table"]
    if results:
        headers = ["variant", "step", "eval_tokens", "loss", "ppl", "data"]
        rows = [
            [
                result["variant"],
                str(result["checkpoint_step"]),
                f"{int(result['eval_tokens']):,}",
                f"{float(result['loss']):.6f}",
                f"{float(result['ppl']):.6f}",
                str(result.get("data_source", "")),
            ]
            for result in sorted(results, key=lambda item: VARIANTS.index(item["variant"]))
        ]
        widths = [max(len(row[i]) for row in rows + [headers]) for i in range(len(headers))]
        lines.append("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
        lines.append("  ".join("-" * width for width in widths))
        lines.extend("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows)
    else:
        lines.append("No variants could be evaluated.")

    for item in missing or []:
        lines.append(f"SKIPPED: {item}")

    pdr = by_variant.get("pdr")
    gla = by_variant.get("gla")
    transformer = by_variant.get("transformer")
    transformer_text = "missing"
    if transformer is not None:
        transformer_text = f"{float(transformer['ppl']):.6f}"

    if pdr is None or gla is None:
        lines.append("GATE UNKNOWN: PDR and GLA results are both required. Transformer reference: " + transformer_text)
    elif float(pdr["ppl"]) <= float(gla["ppl"]):
        lines.append(
            "GATE PASS: PDR ppl <= GLA ppl "
            f"({float(pdr['ppl']):.6f} <= {float(gla['ppl']):.6f}). "
            f"Transformer reference: {transformer_text}"
        )
    else:
        lines.append(
            "GATE FAIL: PDR ppl > GLA ppl "
            f"({float(pdr['ppl']):.6f} > {float(gla['ppl']):.6f}). "
            f"Transformer reference: {transformer_text}"
        )
    lines.append("Verdict meaning: this is the Stage 0 full-rank gate versus low-rank gate hypothesis check.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
