# § 10 — Training Plan

> You don't train 1T parameters from scratch on a laptop.
> Here's how to actually build this model.

---

## 1  Training Strategy Overview

Training a 1.05T sparse MoE with natively ternary experts is a
multi-stage process.  The key insight: **not all parameters train
simultaneously**.  The shared layers train densely; experts train in
rotation; ternary quantization is progressive.

### Stages

| Stage | What trains | Precision | Hardware | Duration |
|-------|-----------|-----------|----------|----------|
| 1. Dense seed | Shared layers only (6.83B) | FP16 | 8 × A100 (80GB) | ~2 weeks |
| 2. Expert initialisation | Clone shared FFN → 128 experts | — | CPU | 1 hour |
| 3. Expert differentiation | Route + train experts (FP16) | FP16 | 32 × A100 | ~4 weeks |
| 4. Ternary distillation | Progressively quantise experts to ternary | FP16→ternary | 32 × A100 | ~2 weeks |
| 5. Manifold alignment | Train routing manifold + delta minimisation | Mixed | 8 × A100 | ~1 week |
| 6. Safety anchoring | Compute SPP polytope | FP32 | CPU | 1 day |
| 7. Evaluation + tuning | Benchmarks, ablations, hyperparameter sweep | Ternary | Consumer HW | ~1 week |

**Total: ~10 weeks on 32 × A100-80GB** (estimated $50K–80K at cloud rates).

---

## 2  Stage 1: Dense Seed Model

### 2.1  Architecture

Train a **6.83B dense model** with the Perspective architecture but without
experts:

- 80 layers: 60 PDR + 20 windowed GQA
- All FFN layers are shared (no routing)
- Full FP16 precision
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
  gpus: 8 × A100-80GB
  parallelism: FSDP (Fully Sharded Data Parallel)
  compute: ~3.2e21 FLOPs
  time: ~14 days
```

### 2.4  Expected Quality

At 6.83B parameters trained on 1T tokens, the seed model should match or
exceed LLaMA-7B quality:

| Benchmark | LLaMA-7B | Expected seed |
|-----------|----------|---------------|
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

Train with the full MoE architecture, standard routing, and a loss that
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
  top_k: 1
  balance_loss_alpha: 0.01
  diversity_loss_beta: 0.001
  
hardware:
  gpus: 32 × A100-80GB
  parallelism: Expert Parallelism + FSDP
  experts_per_gpu: 4
  time: ~28 days
```

### 4.4  Expert Parallelism

With 128 experts and 32 GPUs, each GPU hosts 4 experts.  AlltoAll
communication routes tokens to the correct GPU for expert computation.

Memory per GPU:
- Shared model: 6.83B × 2 bytes = ~14 GB
- 4 experts × 8.12B × 2 bytes = ~65 GB
- Optimizer states: ~80 GB × 2 bytes (AdamW)
- Total: ~80 GB — fits in A100-80GB with careful memory management

---

## 5  Stage 4: Ternary Distillation

### 5.1  Progressive Quantisation

Convert expert weights from FP16 to ternary progressively:

```
Schedule (over 200B tokens):
  Step 0:    100% FP16 experts
  Step 50B:  Top 25% of weights (by magnitude) → ternary; rest FP16
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

During ternary training, the FP16 seed model serves as teacher:

$$
\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{LM}} + 0.5 \cdot \text{KL}(p_{\text{ternary}} \| p_{\text{teacher}})
$$

This prevents quality degradation during quantisation.

### 5.4  Expected Quality After Ternarisation

Based on BitNet b1.58 scaling laws applied to 1T MoE:

| Benchmark | FP16 MoE (post Stage 3) | Ternary MoE (post Stage 4) | Degradation |
|-----------|------------------------|---------------------------|-------------|
| MMLU | ~65% | ~62% | -3% |
| GSM8K | ~45% | ~42% | -3% |
| HumanEval | ~40% | ~37% | -3% |
| ARC | ~68% | ~65% | -3% |

Ternary degradation is modest (~3%) because:
1. Only expert weights are ternary; shared layers stay 2-bit GPTQ
2. 128 experts provide redundancy — routing compensates for per-expert
   precision loss
3. Knowledge distillation from the FP16 teacher preserves quality

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
    - delta_loss (L1 between neighbours)
    - balance_loss (Voronoi cell variance)
    - smooth_loss (routing continuity)
    - fold_loss (in-place manifold update regulariser; prevents append-style drift)
  hardware: 8 × A100 (experts frozen, only routing trains)
  time: ~5 days
```

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

For those without $50K+ cloud budget:

### 9.1  Progressive Expert Conversion

Start from an existing open model (e.g., Mixtral-8×7B, DBRX) and convert:

1. Replace attention with PDR in 3/4 of layers
2. Replace flat routing with manifold routing
3. Progressively add experts (8 → 16 → 32 → 64 → 128)
4. Progressively ternarise experts
5. Fine-tune on each conversion step

This reduces training to ~$5K–10K but produces a non-native architecture
(converted, not trained from scratch).

### 9.2  Federated Training

Distribute expert training across community GPUs:
- Each participant trains 1–4 experts
- Central server coordinates routing and shared-layer updates
- Expert weights are ternary → small upload/download per update

---

## 10  Compute Budget Summary

| Stage | Tokens | FLOPs | A100-hours | Estimated cost |
|-------|--------|-------|-----------|---------------|
| 1. Dense seed | 1T | 3.2e21 | 2,700 | $8,100 |
| 3. Expert diff | 500B | 8.1e22 | 21,500 | $64,500 |
| 4. Ternary distill | 200B | 3.2e22 | 8,600 | $25,800 |
| 5. Manifold align | 50B | 1.6e21 | 430 | $1,290 |
| **Total** | **1.75T** | **~1.2e23** | **~33,200** | **~$100K** |

At current cloud rates (~$3/A100-hr), total training cost is approximately
**$100K**.  This is 1000× cheaper than GPT-4 training but still substantial.

---

*Next: [§ 11 Inference Pipeline](11_inference_pipeline.md)*
