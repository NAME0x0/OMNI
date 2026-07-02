# § 10 — Training Plan

> You don't train 1T parameters from scratch on a laptop.
> Here's what it would take to validate and train this model honestly.

---

## 1  Training Strategy Overview

Training a 1.05T sparse MoE with natively ternary experts is a
multi-stage process.  The key insight: **not all parameters train
simultaneously**.  The shared layers train densely; experts train through
top-1 routing; ternary quantisation is progressive.

This plan is not yet an empirical result.  The PDR architecture, manifold
routing, ternary expert recipe, and loss interactions all require small-scale
validation before any large training spend is justified.

### Stages

| Stage | What trains | Precision | Hardware | Duration |
|-------|-----------|-----------|----------|----------|
| 0. Small-scale architecture validation | 125M PDR, GLA, transformer baselines; ternary gate | BF16/ternary | 1–8 consumer/A100 GPUs | ~2–5 days |
| 1. Dense seed | Shared layers only (6.83B) | BF16 | ~128 × A100-80GB | ~30 days |
| 2. Expert initialisation | Clone shared FFN → 128 experts | — | CPU/NVMe | 1–3 hours |
| 3. Expert differentiation | Route + train experts | BF16 | ~128 × A100-80GB | ~33 days |
| 4. Ternary distillation | Progressively quantise experts to ternary | BF16→ternary | ~128 × A100-80GB | ~18 days |
| 5. Manifold alignment | Train routing manifold + neighbour delta minimisation | Mixed | 32–128 × A100-80GB | ~1–4 days |
| 6. Safety anchoring | Compute SPP polytope | FP32 | CPU | 1 day |
| 7. Evaluation + tuning | Benchmarks, ablations, hyperparameter sweep | Ternary | Consumer HW + A100 eval nodes | ~1–2 weeks |

**Total: ~11–13 weeks with a realistic ~128 A100-80GB allocation.**  The
compute envelope is approximately **250,000 A100-hours**, or **$0.5M–1.5M
depending on MFU achieved, spot pricing, restarts, and eval overhead**.

---

### Stage 0: Small-Scale Architecture Validation

Stage 0 is a hard gate before committing large-scale compute.  The whole
architecture rests on the unvalidated hypothesis that a full-rank perspective
gate improves over lower-rank gated alternatives, and that ternary PDR/MoE
training remains stable.  A small run can falsify those assumptions for less
than the cost of a single large-scale restart.

#### Models

Train three ~125M-parameter models with matched dimensions and token budgets:

| Variant | Configuration | Purpose |
|---------|---------------|---------|
| PDR variant | $d_{\text{model}} = 768$, 12 layers, 9 PDR + 3 GQA with period-4 interleave | Tests full-rank perspective gate |
| GLA baseline | Same dimensions, low-rank gate | Controls against a known gated-linear design |
| Vanilla transformer | Same dimensions and depth | Controls against standard attention/MLP |

Use ~2.5B tokens, for example a SlimPajama subset.  This is roughly
$6 \cdot 125\text{M} \cdot 2.5\text{B} \approx 1.9 \times 10^{18}$ FLOPs per
model, so the full comparison fits on a single node with 1–8 consumer or A100
GPUs.  Target budget: **<$500**.

#### Gate Criteria

The large training plan should not proceed unless all gates pass:

| Gate | Requirement |
|------|-------------|
| PDR quality | PDR matches or beats the GLA baseline perplexity at equal parameters and tokens |
| PDR implementation | Parallel-scan PDR matches the sequential reference within numerical tolerance |
| Routing balance | Routing Gini remains < 0.15 during training, not just at initialisation |
| Ternary viability | Ternarising the 125M PDR model with the Stage 4 STE recipe causes <10% relative perplexity degradation |

The ternary gate matters because the large model assumes BitNet-style
ternarisation works for PDR and sparse experts.  That has not been shown here.

---

## 2  Stage 1: Dense Seed Model

### 2.1  Architecture

Train a **6.83B dense model** with the Perspective architecture but without
experts:

- 80 layers: 60 PDR + 20 windowed GQA
- All FFN layers are shared (no routing)
- BF16 mixed precision
- Standard AdamW optimiser

### 2.2  Data

| Dataset | Tokens | Purpose |
|---------|--------|---------|
| SlimPajama | 627B | General web text |
| The Stack v2 | 300B | Code |
| Wikipedia + Wikibooks | 6B | Factual knowledge |
| ArXiv papers | 50B | Scientific reasoning |
| OpenWebMath | 15B | Mathematical reasoning |
| **Total** | **~1T tokens** | |

### 2.3  Training Configuration

```yaml
model:
  d_model: 4096
  n_layers: 80
  pdr_layers: 60
  gqa_layers: 20
  pdr_rank: 256
  gqa_heads: 32
  gqa_kv_heads: 8
  ffn_intermediate: 11008
  vocab_size: 32768

optimizer:
  type: AdamW
  lr: 3e-4
  warmup_steps: 2000
  decay: cosine → 3e-5
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95

training:
  batch_size: 2048 sequences × 4096 tokens = 8.4M tokens/batch
  total_steps: 120000 (~1T tokens)
  grad_clip: 1.0
  precision: BF16 (mixed)

hardware:
  realistic_allocation: ~128 × A100-80GB
  parallelism: FSDP / ZeRO-style sharding
  compute: 6 × 6.83e9 × 1e12 ≈ 4.1e22 FLOPs
  effective_throughput: ~1.25e14 FLOP/s per A100 at 40% MFU
  a100_hours: ~91,000
  cloud_cost_at_3_usd_per_a100_hour: ~$275K
  time_on_8_a100s: ~470 days
  time_on_128_a100s: ~30 days
```

The 8-GPU version is not a realistic schedule.  With the standard
$6ND$ training FLOPs rule and ~40% MFU on A100-80GB BF16
($312$ TFLOP/s peak → ~$1.25 \times 10^{14}$ FLOP/s effective), the dense
seed alone is a month-scale 128-GPU job.

### 2.4  Expected Quality

These are **speculative targets (no empirical basis — nothing has been
trained)**.  They are useful only as a rough sanity check for evaluation
planning.

| Benchmark | LLaMA-7B | Speculative seed target |
|-----------|----------|-------------------------|
| MMLU | 35.1% | ~36% |
| ARC-Challenge | 51.7% | ~52% |
| GSM8K | 11.0% | ~12% |

---

## 3  Stage 2: Expert Initialisation

Clone the 20 shared FFN layers into 128 copies each for the 60 expert
FFN layers:

```python
for layer in expert_layers:  # 60 layers
    for expert_id in range(128):
        expert[layer][expert_id] = copy(shared_ffn[layer % 20])
```

Layer mapping: expert layer $\ell$ initialises from shared FFN layer
$(\ell \mod 20)$.  This gives each expert a reasonable starting point.

After cloning: total expert parameters = 128 × 60 × 135.3M = **1.04T**.
Not yet differentiated — all experts in the same layer are identical.

---

## 4  Stage 3: Expert Differentiation

### 4.1  Method

Train with the full MoE architecture, standard top-1 routing, and a loss that
encourages expert specialisation:

$$
\mathcal{L} = \mathcal{L}_{\text{LM}} + \alpha \, \mathcal{L}_{\text{balance}} + \beta \, \mathcal{L}_{\text{diversity}}
$$

where:

$$
\mathcal{L}_{\text{diversity}} = -\frac{1}{N^2} \sum_{i \neq j} \text{cos\_sim}(W_i, W_j)
$$

pushes experts apart, and $\mathcal{L}_{\text{balance}}$ ensures even
routing distribution.

This diversity objective is in tension with the Stage 5 delta-minimisation
objective, which pulls some experts together.  The intended reconciliation is
to apply diversity pressure to non-neighbour expert pairs and delta loss only
to manifold-neighbour pairs.  That is still a design hypothesis and should be
checked in Stage 0 or a follow-on small MoE run before full scale.

### 4.2  Router Pre-training

The manifold router needs to learn meaningful projections.  During this
stage, use **standard top-1 token-choice routing** with a simple linear
router — the manifold geometry is imposed in Stage 5.

### 4.3  Training Configuration

```yaml
training:
  data: same as Stage 1 but shuffled differently
  tokens: 500B (continued pre-training)
  batch_size: 1024 × 4096 = 4.2M tokens/batch
  optimizer: same as Stage 1, lr warmup from 1e-5 → 2e-4

moe:
  num_experts: 128
  active_params_per_token: 14.95B
  top_k: 1
  balance_loss_alpha: 0.01
  diversity_loss_beta: 0.001

hardware:
  realistic_allocation: ~128 × A100-80GB
  parallelism: Expert Parallelism + FSDP/ZeRO
  compute: 6 × 14.95e9 × 500e9 ≈ 4.5e22 FLOPs
  a100_hours: ~100,000
  cloud_cost_at_3_usd_per_a100_hour: ~$300K
  time_on_128_a100s: ~33 days
```

### 4.4  Expert Parallelism and Memory

The naive layout of 128 experts over 32 GPUs gives 4 experts per GPU.  That
does **not** fit in A100-80GB once gradients and AdamW states are counted:

$$
4 \times 8.12\text{B} = 32.5\text{B params/GPU}
$$

| Component | Memory per GPU |
|-----------|----------------|
| BF16 expert weights | ~65 GB |
| BF16 expert gradients | ~65 GB |
| FP32 AdamW $m, v$ states | ~260 GB |
| **Expert subtotal** | **~390 GB** |

This excludes shared layers, activations, routing buffers, and communication
workspace.  A feasible implementation needs sharding or offload:

| Option | Approximate expert-state memory per GPU | Tradeoff |
|--------|-----------------------------------------|----------|
| 1 expert/GPU across 128 GPUs | ~98 GB before sharding | Still needs sharded optimiser states or offload, but avoids 4-expert packing |
| ZeRO-1/2 sharded optimiser states | ~130–200 GB for 4 experts/GPU, depending on shard group | Reduces optimiser pressure but still tight without combining with fewer experts/GPU |
| 8-bit optimiser states | ~195 GB for 4 experts/GPU | Helps, but does not by itself make 4 experts/GPU fit |
| CPU optimiser offload | ~130 GB GPU for 4 experts/GPU plus large CPU RAM | Slower; may be acceptable for constrained runs |
| 1 expert/GPU + ZeRO/8-bit/offload | <80 GB target with careful activation/gradient policy | Most plausible A100-80GB path |

The training plan should assume ~128 GPUs with one expert resident per GPU
and sharded or compressed optimiser state, not 32 GPUs with 4 full AdamW
experts each.

---

## 5  Stage 4: Ternary Distillation

### 5.1  Progressive Quantisation

Convert expert weights from BF16 to ternary progressively:

```
Schedule (over 200B tokens):
  Step 0:    100% BF16 experts
  Step 50B:  Top 25% of weights (by magnitude) → ternary; rest BF16
  Step 100B: Top 50% → ternary
  Step 150B: Top 75% → ternary
  Step 200B: 100% ternary
```

At each step, the STE (Straight-Through Estimator) is used for ternary
weights:

```python
# Forward: quantise to ternary
w_ternary = ternary_round(w_fp16)  # {-1, 0, +1}

# Backward: gradient passes through as if unquantised
grad_w_fp16 = grad_w_ternary  # STE
```

### 5.2  Ternary Rounding Function

$$
\text{ternary\_round}(w) = \begin{cases}
+1 & \text{if } w > \tau \\
-1 & \text{if } w < -\tau \\
0 & \text{otherwise}
\end{cases}
$$

where $\tau$ is the per-tensor threshold, typically the mean absolute value:

$$\tau = \alpha \cdot \text{mean}(|W|), \quad \alpha \approx 0.7$$

This naturally produces ~50% zeros (the mean absolute value cuts off
roughly half the distribution).

### 5.3  Knowledge Distillation

During ternary training, a BF16 MoE checkpoint serves as teacher:

$$
\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{LM}} + 0.5 \cdot \text{KL}(p_{\text{ternary}} \| p_{\text{teacher}})
$$

The student-side training compute for 200B tokens at 14.95B active parameters
is:

$$
6 \cdot 14.95 \times 10^9 \cdot 200 \times 10^9 \approx 1.8 \times 10^{22}
$$

That is roughly **40,000 A100-hours** or **$120K** at $3/A100-hour.  Teacher
forward passes add about 30–50%, so the practical estimate is **~55,000
A100-hours** before restarts and evaluation.

### 5.4  Expected Quality After Ternarisation

These are **speculative targets (no empirical basis — nothing has been
trained)**.  The -3% degradation assumption is optimistic, based on BitNet
b1.58 results at ≤3.9B scale.  It is unvalidated at 8B-expert scale and for
sparse MoE routing.

| Benchmark | Speculative BF16 MoE target | Speculative ternary MoE target | Assumed degradation |
|-----------|-----------------------------|--------------------------------|---------------------|
| MMLU | ~65% | ~62% | -3% |
| GSM8K | ~45% | ~42% | -3% |
| HumanEval | ~40% | ~37% | -3% |
| ARC | ~68% | ~65% | -3% |

The hypothesis is that ternary degradation is limited because:

1. Only expert weights are ternary; shared layers remain higher precision or
   separately quantised.
2. 128 experts may provide redundancy, allowing routing to compensate for
   per-expert precision loss.
3. Knowledge distillation from a BF16 teacher may preserve quality.

Each point is an assumption until tested.

---

## 6  Stage 5: Manifold Alignment

### 6.1  Embed Experts on Torus

Using the weight similarity matrix from Stage 3/4, compute an expert
embedding via metric MDS (Multi-Dimensional Scaling):

```python
# Compute pairwise expert distance
D[i][j] = L1_distance(expert_i_weights, expert_j_weights)

# MDS to 3D
positions = MDS(D, n_components=3)

# Map to torus [0, 1)^3
positions = positions / positions.max() * 0.95 + 0.025  # margin from edges
```

### 6.2  Train Manifold Router

Replace the flat token-choice router with the manifold router:

```yaml
manifold_training:
  tokens: 50B
  lr: 1e-4 (only router and expert positions trainable)
  losses:
    - task_loss (cross-entropy)
    - delta_loss (L1 only between manifold-neighbour expert pairs)
    - balance_loss (Voronoi cell variance)
    - smooth_loss (routing continuity)
    - fold_loss (in-place manifold update regulariser; prevents append-style drift)
  hardware: 32–128 × A100 (experts frozen, router/positions train)
  compute: ~1.5e21 FLOPs (forward-dominated)
  a100_hours: ~3,300
  cloud_cost_at_3_usd_per_a100_hour: ~$10K
  time_on_128_a100s: ~1–2 days including overhead
```

The delta-minimisation loss and the Stage 3 diversity loss are opposing
objectives.  The only coherent version is local: delta loss applies to
manifold-neighbour pairs that should have compact deltas; diversity applies to
non-neighbour pairs that should remain specialised.  This reconciliation is a
design hypothesis, not a proven property.

### 6.3  Delta File Generation

After manifold alignment, pre-compute delta files:

```python
for expert_i in range(128):
    for expert_j in manifold_neighbours(expert_i):
        delta = compute_ternary_delta(expert_i, expert_j)
        save_delta_file(delta, f"delta_{i}_{j}.trd")
```

---

## 7  Stage 6: Safety Anchoring

### 7.1  Anchor Generation

```python
# Generate safe anchor embeddings
safe_vocab = get_safe_vocabulary(size=5000)
embeddings = model.embed(safe_vocab)

# Cluster to 1000 representative anchors
anchor_embeddings = kmeans(embeddings, k=1000)

# Compute half-space representation
halfspaces = convex_hull_halfspaces(anchor_embeddings, n_facets=500)

# Save
save_spp_config(anchor_embeddings, halfspaces)
```

### 7.2  Red-Team Calibration

Run 500 adversarial prompts through the model:

- Adjust $\epsilon$ (polytope inflation) to balance safety vs utility
- Add anchors in under-covered safe regions
- Repeat until block rate ≥ 98%

---

## 8  Stage 7: Evaluation

Full benchmark suite on consumer hardware (the target platform):

```yaml
evaluation:
  hardware: RTX 3060 (4GB) + 32GB RAM + NVMe
  benchmarks:
    - MMLU (57 tasks)
    - GSM8K (math reasoning)
    - HumanEval (code generation)
    - ARC-Challenge (science reasoning)
    - TruthfulQA (hallucination test)
    - WinoGrande (commonsense)

  system_tests:
    - VRAM usage ≤ 2,684 MB
    - RAM usage ≤ 26 GB
    - Throughput ≥ 7 tok/s (without MPD)
    - Throughput ≥ 5 tok/s (with MPD)
    - SPP block rate ≥ 98% on red-team set
    - MPD ECE ≤ 0.08
    - HDM retrieval accuracy ≥ 90%
```

---

## 9  Alternative: Community Training

For those without a 128-GPU allocation:

### 9.1  Progressive Expert Conversion

Start from an existing open model (e.g., Mixtral-8×7B, DBRX) and convert:

1. Replace attention with PDR in 3/4 of layers.
2. Replace flat routing with manifold routing.
3. Progressively add experts (8 → 16 → 32 → 64 → 128).
4. Progressively ternarise experts.
5. Fine-tune on each conversion step.

This lowers the cash requirement but produces a non-native architecture
(converted, not trained from scratch).  The attention→PDR conversion is itself
unvalidated; it should be treated as a research experiment, not a reliable
shortcut.

### 9.2  Federated Training

Distribute expert training across community GPUs:

- Each participant trains 1–4 experts
- Central server coordinates routing and shared-layer updates
- Expert weights are ternary → small upload/download per update

Federated training also changes optimisation dynamics, routing balance, and
data governance assumptions.  It needs its own validation before being used as
evidence for the centralised training plan.

---

## 10  Compute Budget Summary

Assumptions: standard $6ND$ training FLOPs, A100-80GB BF16 peak 312 TFLOP/s,
~40% MFU (~$1.25 \times 10^{14}$ FLOP/s effective), and $3/A100-hour.

| Stage | Tokens | FLOPs | A100-hours | Estimated cost |
|-------|--------|-------|-----------|---------------|
| 0. Small-scale validation | ~2.5B × 3 + ternary gate | ~6e18–1e19 | <170 GPU-hours on small hardware | <$500 |
| 1. Dense seed | 1T | ~4.1e22 | ~91,000 | ~$275K |
| 3. Expert diff | 500B | ~4.5e22 | ~100,000 | ~$300K |
| 4. Ternary distill | 200B | ~1.8e22 student; +30–50% teacher | ~55,000 with teacher | ~$165K |
| 5. Manifold align | 50B | ~1.5e21 | ~3,300 | ~$10K |
| 6–7. Safety + evaluation | — | workload-dependent | reserve margin | included in range |
| **Total** | **~1.75T large-scale tokens** | **~1.1e23 plus eval/restarts** | **~250,000** | **~$750K nominal** |

The honest budget range is **$0.5M–1.5M**, depending on MFU achieved, spot
pricing, restarts, data pipeline stalls, teacher-forward overhead, and
evaluation/ablation scope.

---

*Next: [§ 11 Inference Pipeline](11_inference_pipeline.md)*
