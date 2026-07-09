# Stage 0 Training Runbook

This trains the three ~125M-parameter Stage 0 variants on FineWeb streaming data with checkpointing designed for free Kaggle and Colab sessions.

## Kaggle

1. Create a notebook, enable a free GPU accelerator, and add an `HF_TOKEN` secret with write access to your checkpoint repo.
2. Clone this repo into the notebook workspace.
3. Install dependencies:
   ```bash
   pip install -r train/requirements.txt
   ```
4. Run one variant:
   ```bash
   python train/run_stage0.py --variant pdr --tokens 2500000000 --hub-repo user/stage0-pdr --output-dir /kaggle/working/stage0-pdr
   ```

Defaults are free-tier oriented: `seq_len=1024`, `micro_batch=2`, `chunk_len=64`, fp16 on T4/P100 unless bf16 is supported, and gradient accumulation near 262,144 tokens per optimizer step. The trainer saves every 15 minutes, on interruption, and before the default 8.5 hour budget guard exits.

## Session Modes, Timeouts, and Idle Behaviour (Kaggle)

This plan is Kaggle-only. Two session modes exist; always use batch mode:

| Mode | GPU limit | Idle behaviour |
|---|---|---|
| Interactive (browser attached) | up to 12 h | killed after ~40–60 min without browser activity — avoid |
| **Save & Run All (batch)** | ~9 h | **no idle timeout, runs headless, browser can close** |

The trainer's default `--max-hours 8.5` budget guard exits with a final
checkpoint + Hub upload safely under the 9 h batch cap. Resume happens
automatically on the next run (local checkpoint first, then the Hub repo).

Quota check (do this after your first session): Kaggle has metered T4 x2
either as wall-clock hours (1x) or device-hours (2x) at different times.
Look at the GPU quota meter after the first run. If a T4 x2 session burned
double, switch the accelerator to P100 (single GPU, 1x metering) — the
trainer is single-GPU either way.

## Free-Quota Schedule (Kaggle-only)

Run one variant per batch session, rotating through:

```bash
python train/run_stage0.py --variant pdr --tokens 2500000000 --hub-repo user/stage0-pdr
python train/run_stage0.py --variant gla --tokens 2500000000 --hub-repo user/stage0-gla
python train/run_stage0.py --variant transformer --tokens 2500000000 --hub-repo user/stage0-transformer
```

At ~25–30 usable GPU-hours/week (about three 8.5 h batch runs), 2.5 B tokens
per variant works out to roughly **4–6 weeks calendar for all three
variants**. Levers if that is too slow: cut `--tokens` to 2e9 (saves ~20%,
still reasonable at 125M scale) or add 2-GPU DDP to use the second T4
(future work; roughly halves per-variant wall time).

## Gates

Stage 0 should pass before larger training:

- PDR quality: PDR matches or beats the GLA baseline perplexity at equal parameters and tokens.
- PDR implementation: parallel-scan PDR matches the sequential reference within numerical tolerance.
- Routing balance: N/A for Stage 0 dense baselines; apply to later MoE runs.
- Ternary viability: later gate; ternarizing the 125M PDR model with the Stage 4 STE recipe should cause less than 10% relative perplexity degradation.

Use only one Kaggle account. Multi-account quota farming violates the terms of service.

## Kaggle notebook

Use `train/notebooks/kaggle_stage0.ipynb` for the one-edit Kaggle flow:

1. Add the Kaggle secret `HF_TOKEN` once, with a Hugging Face write token.
2. Set `VARIANT` in the config cell.
3. Run All; repeat daily until the token budget is reached.

Kaggle's Save & Run All runs the notebook headless in the background, so you can leave it running after launch.
