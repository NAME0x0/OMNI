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

## Colab

Use the same setup in a GPU runtime:

```bash
git clone <repo-url>
cd OMNI
pip install -r train/requirements.txt
export HF_TOKEN=<token>
python train/run_stage0.py --variant gla --tokens 2500000000 --hub-repo user/stage0-gla
```

If the VM is preempted, rerun the same command. The trainer resumes from local checkpoints first, then from the Hugging Face Hub checkpoint folder when `--hub-repo` is set.

## Free-Quota Schedule

Run one variant per active session, rotating through:

```bash
python train/run_stage0.py --variant pdr --tokens 2500000000 --hub-repo user/stage0-pdr
python train/run_stage0.py --variant gla --tokens 2500000000 --hub-repo user/stage0-gla
python train/run_stage0.py --variant transformer --tokens 2500000000 --hub-repo user/stage0-transformer
```

Kaggle's 30 GPU-hour weekly quota plus free Colab preemptible T4 sessions should put the three-way comparison in the 2-3 week range if sessions are restarted promptly and checkpoints sync cleanly.

## Gates

Stage 0 should pass before larger training:

- PDR quality: PDR matches or beats the GLA baseline perplexity at equal parameters and tokens.
- PDR implementation: parallel-scan PDR matches the sequential reference within numerical tolerance.
- Routing balance: N/A for Stage 0 dense baselines; apply to later MoE runs.
- Ternary viability: later gate; ternarizing the 125M PDR model with the Stage 4 STE recipe should cause less than 10% relative perplexity degradation.

Use only one account per platform. Multi-account quota farming violates Kaggle and Colab terms of service.
