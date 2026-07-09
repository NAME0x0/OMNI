"""CLI entrypoint for Stage 0 architecture validation training."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from data import FineWebDataConfig, make_fineweb_eval_iterator, make_fineweb_train_iterator, make_synthetic_batch_iterator
from perspective_torch import Stage0Config, Stage0LM, count_parameters, param_table
from trainer import (
    DEFAULT_CHUNK_LEN,
    DEFAULT_MICRO_BATCH_SIZE,
    DEFAULT_SEQ_LEN,
    HubCheckpointSync,
    Stage0Trainer,
    TrainerConfig,
    resolve_gradient_accumulation_steps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 0 PDR/GLA/transformer training.")
    parser.add_argument("--variant", choices=["pdr", "gla", "transformer"], required=True)
    parser.add_argument("--tokens", type=int, default=2_500_000_000)
    parser.add_argument("--hub-repo", default=None, help="Optional HF Hub repo id, e.g. user/stage0-pdr")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--micro-batch", type=int, default=DEFAULT_MICRO_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--chunk-len", type=int, default=DEFAULT_CHUNK_LEN)
    parser.add_argument("--save-minutes", type=float, default=15.0)
    parser.add_argument("--eval-steps", type=int, default=1_000)
    parser.add_argument("--eval-tokens", type=int, default=200_000)
    parser.add_argument("--eval-docs", type=int, default=2_000)
    parser.add_argument("--max-hours", type=float, default=8.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny deterministic 200-step synthetic training job")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1234)

    if args.smoke:
        args.seq_len = min(args.seq_len, 32)
        args.micro_batch = min(args.micro_batch, 2)
        args.grad_accum = 1
        args.hub_repo = None
        args.eval_steps = 0
        args.save_minutes = 10_000.0
        args.max_hours = 1.0
        max_steps = 200
        config = Stage0Config(
            variant=args.variant,
            d_model=64,
            rank=8,
            n_layers=4,
            vocab_size=50_257,
            n_heads=4,
            n_kv_heads=2,
            sliding_window=32,
            ffn_intermediate=128,
        )
        output_dir = Path(args.output_dir or f"train/runs/smoke-{args.variant}")
        train_iter = make_synthetic_batch_iterator(
            seq_len=args.seq_len,
            micro_batch_size=args.micro_batch,
            vocab_size=config.vocab_size,
            seed=17,
        )
        eval_factory = None
    else:
        grad_accum = args.grad_accum or resolve_gradient_accumulation_steps(
            seq_len=args.seq_len,
            micro_batch_size=args.micro_batch,
        )
        tokens_per_step = args.seq_len * args.micro_batch * grad_accum
        max_steps = math.ceil(args.tokens / tokens_per_step)
        config = Stage0Config(variant=args.variant)
        output_dir = Path(args.output_dir or f"train/runs/stage0-{args.variant}")
        data_config = FineWebDataConfig(seq_len=args.seq_len, eval_docs=args.eval_docs)
        train_iter = make_fineweb_train_iterator(data_config, micro_batch_size=args.micro_batch)
        eval_factory = lambda: make_fineweb_eval_iterator(data_config, micro_batch_size=args.micro_batch)

    print(param_table(print_table=False), flush=True)
    model = Stage0LM(config)
    print(f"actual_parameters={count_parameters(model):,}", flush=True)
    grad_accum = args.grad_accum or resolve_gradient_accumulation_steps(
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch,
    )
    print(
        f"variant={args.variant} seq_len={args.seq_len} micro_batch={args.micro_batch} "
        f"grad_accum={grad_accum} chunk_len={args.chunk_len} max_steps={max_steps}",
        flush=True,
    )

    trainer_config = TrainerConfig(
        output_dir=output_dir,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        chunk_len=args.chunk_len,
        eval_steps=args.eval_steps,
        eval_tokens=args.eval_tokens,
        save_minutes=args.save_minutes,
        max_hours=args.max_hours,
        hub_repo=args.hub_repo,
    )
    trainer = Stage0Trainer(
        model=model,
        train_iterator=train_iter,
        eval_iterator_factory=eval_factory,
        config=trainer_config,
        hub_sync=HubCheckpointSync(args.hub_repo),
    )
    loaded = trainer.load_checkpoint()
    if loaded is None:
        print("starting from scratch", flush=True)
    trainer.run()


if __name__ == "__main__":
    main()
