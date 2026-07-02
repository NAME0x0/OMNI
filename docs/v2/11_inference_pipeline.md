# § 11 — Inference Pipeline

> Every microsecond counts when you're streaming experts off NVMe.

---

## 1  Pipeline Overview

**Implementation status:** This document describes the intended inference
design.  The Rust runtime path is not functional end-to-end yet:
`src/runtime/pipeline.rs::process_token` intentionally returns an error, and
there is no tokenizer wired into the runtime.

A single token generation pass through PERSPECTIVE involves:

**Projected compute-only stage sketch (bandwidth model; unmeasured):**

```
Input token
  │
  ├─ 1. Embed (VRAM)                        ≈ 0.02 ms
  ├─ 2. PDR layers ×60 (VRAM)               ≈ 1.80 ms
  │     └─ Expert routing + execution ×60    (overlapped with PDR)
  ├─ 3. GQA layers ×20 (VRAM)               ≈ 0.60 ms
  │     └─ Windowed attention/shared compute
  ├─ 4. LM head (VRAM)                      ≈ 0.02 ms
  ├─ 5. SPP safety projection (CPU ref; device TBD) ≈ 0.01 ms
  ├─ 6. MPD (conditional, 20% of tokens)    ≈ 3.20 ms (amortised 0.64 ms)
  ├─ 7. HDM retrieval (conditional, 5%)     ≈ 0.03 ms (amortised ~0 ms)
  └─ 8. Sample + output                     ≈ 0.01 ms
                                     ─────────────────
                            Total:  ~3.10 ms/token (without MPD)
                                    ~3.74 ms/token (with MPD amortised)
```

This compute-only sketch yields **~322 tok/s compute-bound** in isolation,
but it is not the headline throughput.  The projected PCIe-bound baseline from
the § 02 bandwidth model is **~93 ms/token, ~10.8 tok/s**.

---

## 2  The PCIe Bottleneck

### 2.1  Per-Layer Expert Load

Each of the 60 expert layers requires loading one expert-layer (top-1 routing):

| Metric | Full load | Delta load | Unit |
|--------|-----------|------------|------|
| Expert size (ternary packed) | 27 MB | 2.7 MB | per expert |
| Sustained PCIe bandwidth assumption | 7 GB/s | 7 GB/s | unmeasured |
| Full load time | 3.86 ms | 0.40 ms | per expert-layer |
| 60 expert layers (all full) | 231.4 ms | — | per token |
| 60 expert layers (30% full, 70% delta) | — | **85.7 ms** | per token |

Add GQA + shared FFN compute (~7 ms) and embedding/output overhead:
**projected bottleneck: ~93 ms/token → ~10.8 tok/s** (PCIe bound).

### 2.2  Mitigation: Layer-Streamed Double Buffering

We don't wait for all 60 expert layers to load.  Instead, the pipeline overlaps
loading of layer $\ell + 1$'s expert with computation of layer $\ell$:

```
Timeline:

Buffer A: [Load expert ℓ][     idle     ][Load expert ℓ+2][     idle     ]
Buffer B: [     idle     ][Load expert ℓ+1][     idle     ][Load expert ℓ+3]
Compute:  [              ][Compute ℓ     ][Compute ℓ+1   ][Compute ℓ+2   ]
```

Effective per-layer time: `max(load_time, compute_time)`

- Load time (average): 1.43 ms (70% delta, 7 GB/s sustained)
- Compute time (average): ~0.14 ms (ternary GEMM target)
- **Effective: 1.43 ms/layer × 60 expert layers = 85.7 ms/token**

The design assumes compute is mostly hidden behind the load; this has not
been measured end-to-end.

### 2.3  Prediction-Ahead Prefetch

Look ahead $k$ layers using the router:

```
At layer ℓ, compute:
  expert_ℓ+1 = route(hidden_ℓ, routing_weights_ℓ+1)  // predicted
  expert_ℓ+2 = route(hidden_ℓ, routing_weights_ℓ+2)  // speculative

Start loading expert_ℓ+2 while computing layer ℓ.
```

Prediction accuracy target at $k = 1$: ~87% (based on the hypothesis that
hidden states change slowly between adjacent layers).

If the prediction is wrong: stall for one full expert-layer load (3.86 ms
worst case at the § 02 sustained-bandwidth assumption); target miss rate ~13%
at depth 1.

Expected mean load time with 1-ahead prediction:

$$
t = 0.87 \times 0 + 0.13 \times 3.86 \text{ ms} = 0.502 \text{ ms/layer (full miss)}
$$

Combined with delta streaming (70% of correct predictions use delta):

$$
t = 0.87 \times [0.7 \times 0 + 0.3 \times 0] + 0.13 \times [0.7 \times 0.40 + 0.3 \times 3.86]
$$
$$
t = 0 + 0.13 \times [0.280 + 1.158] = 0.13 \times 1.438 = 0.187 \text{ ms/layer}
$$

Wait — with prediction-ahead, correct predictions cost 0 because we
pre-loaded.  So:

$$
t_{\text{effective}} = 0.187 \text{ ms/layer} \times 60 = 11.2 \text{ ms/token}
$$

**Illustrative prediction-ahead upper bound: ~89 tok/s** before other runtime
costs — optimistic, unmeasured, and not used as the baseline.

### 2.4  Projected Throughput Estimates (Bandwidth Model; Unmeasured)

| Configuration | ms/token | tok/s |
|--------------|----------|-------|
| No prefetch, all full loads | ~238 | ~4.2 |
| No prefetch, 70% delta | ~93 | ~10.8 |
| 1-ahead prefetch, 70% delta | Target: 50–90 | Target: 11–20 |
| 2-ahead prefetch, 70% delta | Target: 35–70 | Target: 14–29 |
| **Conservative projected baseline** | **~93** | **~10.8** |
| **Aspirational tuned runtime** | **70–90** | **~11–14** |

We use **~10.8 tok/s projected** as the conservative baseline until the
runtime is implemented and benchmarked.

---

## 3  Memory Management

### 3.1  VRAM Layout (Static)

| Component | Size (MB) | Notes |
|-----------|----------|-------|
| Shared layers (2-bit GPTQ) | 1,707 | Permanent resident |
| Router weights | 1.9 | 80 layers × (4096→3) FP16 |
| KV cache (512-window ×20 GQA) | 42 | Architecture windowed cache budget |
| PDR states (60 layers) | 120 | Recurrent state |
| Active expert buffer A | 27 | Double buffer slot |
| Active expert buffer B | 27 | Double buffer slot |
| Activations + workspace | 200 | Computation scratch |
| Embeddings + LM head | 128 | Shared embedding |
| SPP anchors + halfspaces | 24 | Safety data |
| HDM codebook | 5 | Lookup table |
| **Total** | **2,281** | **55.7% of 4 GB** |
| **Free** | **1,815** | For OS + driver overlay |

### 3.2  RAM Layout

| Component | Size (GB) | Notes |
|-----------|----------|-------|
| Hot expert cache (8 experts) | 13.0 | 8 × 1.624 GB (ternary packed) |
| Read-ahead buffer | 0.5 | For NVMe prefetch |
| HDM banks (2000) | 0.0025 | Index only |
| FMEA LoRA workspace | 0.011 | During adaptation |
| MPD perspective cache | 0.8 | 4 copies of activations |
| OS + runtime overhead | ~4 | Estimated |
| **Total** | **~18.1** | **56.6% of 32 GB** |
| **Free** | **~13.9** | Headroom |

### 3.3  NVMe Layout

```
model/
├── shared/
│   ├── shared_layers.gptq2     (1.7 GB)
│   ├── embeddings.bin          (128 MB)
│   ├── router_weights.bin      (~2 MB)
│   ├── spp_config.bin          (24 MB)
│   └── hdm_codebook.bin        (5 MB)
├── experts/
│   ├── expert_000.trit         (1.624 GB each)
│   ├── expert_001.trit
│   ├── ...
│   └── expert_127.trit         (total: ~208 GB)
├── deltas/
│   ├── delta_000_001.trd       (~160 MB each, sparse)
│   ├── delta_000_016.trd       (neighbours on torus)
│   ├── ...
│   └── delta_127_111.trd       (total: ~20–40 GB)
└── meta/
    ├── manifold_positions.json  (128 × 3 floats)
    ├── expert_clusters.json     (cluster assignments)
    └── model_config.json        (hyperparameters)
```

Packed format: 5 trits/byte = 1.6 bits/param = 0.2 B/param.  One expert
file is 8.12B params × 0.2 B/param = **1.624 GB**.

**Total NVMe: ~230–253 GB**

---

## 4  Token Generation Loop

### 4.1  Main Loop (Pseudocode)

```rust
fn generate(prompt: &[u32], max_tokens: usize) -> Vec<u32> {
    let mut hidden = embed(prompt);          // [seq_len, d_model]
    let mut pdr_states = init_pdr_states();  // 60 × [d_model, rank]
    let mut kv_cache = KVCache::new();       // For 20 GQA layers
    let mut output_tokens = Vec::new();
    let mut prev_experts = [NONE; 60];       // Last expert per expert layer
    
    // Process prompt (prefill)
    for layer in 0..80 {
        hidden = forward_layer(layer, hidden, &mut pdr_states, &mut kv_cache,
                               &mut prev_experts);
    }
    
    // Autoregressive decode
    for step in 0..max_tokens {
        let token_hidden = hidden.last_token();
        
        // Layer-streamed forward pass
        let mut expert_slot = 0;
        for layer in 0..80 {
            // 1. Compute attention/PDR
            token_hidden = if layer < 60 {
                pdr_forward(token_hidden, &mut pdr_states[layer])
            } else {
                gqa_forward(token_hidden, &mut kv_cache, layer - 60)
            };

            // 2. Expert FFN on the 60 expert-bearing layers
            if layer < 60 {
                let expert_id = manifold_route(token_hidden, layer);
                let expert_future = load_expert_async(
                    layer, expert_id, prev_experts[expert_slot]);
                let expert_weights = expert_future.await;
                token_hidden = ternary_ffn(token_hidden, &expert_weights);
                prev_experts[expert_slot] = expert_id;
                expert_slot += 1;
            }
        }
        
        // LM head
        let logits = lm_head(token_hidden);
        
        // Safety projection
        let safe_logits = spp_project(logits);
        
        // Multi-perspective decoding (conditional)
        let final_logits = if mpd_should_activate(safe_logits) {
            mpd_decode(safe_logits, &pdr_states, &kv_cache)
        } else {
            safe_logits
        };
        
        // Sample
        let token = sample(final_logits, temperature, top_p);
        output_tokens.push(token);
        
        if token == EOS { break; }
    }
    
    output_tokens
}
```

### 4.2  Prefill Phase

For the prompt (may be hundreds of tokens), we process all tokens in
parallel through each layer:

- PDR: parallel scan across the prompt (O(n log n) work, O(n) depth)
- GQA: standard flash-attention over the prompt
- Expert routing: batch route all prompt tokens, load top-1 per layer
  (majority vote over tokens)

Projected prefill throughput: **~100–200 tok/s** (compute-bound, not
PCIe-bound, because one expert serves all prompt tokens per layer).  This is
unmeasured.

### 4.3  Decode Phase

Single-token generation, PCIe-bound.  The critical path:

```
Layer ℓ decode:
  [Route: 0.02 ms] → [Load expert: 0–2.13 ms] → [PDR/GQA: 0.03 ms] → [Ternary FFN: 0.03 ms]
                                                    ↑ overlap ↑
```

---

## 5  Expert Cache Management

### 5.1  Cache Policy

The 8-slot hot expert cache uses **frequency-weighted LRU**:

```
Score(expert) = frequency(expert) × recency(expert) × delta_cost(expert)
```

where `delta_cost` is 1.0 if the expert has no delta to the most likely
replacement, or 0.1 if it does (prefer caching experts that are expensive
to load).

### 5.2  Cache Warming

On startup, pre-load the 8 most commonly routed experts (intended to come from
training or deployment routing statistics):

```rust
fn warm_cache(cache: &mut ExpertCache) {
    let frequencies = load_expert_frequencies();
    let top_8 = frequencies.top_k(8);
    for expert_id in top_8 {
        cache.load_sync(expert_id);  // Blocking NVMe read
    }
}
```

Projected cache warming takes ~13.0 seconds (8 × 1.624 GB / 1 GB/s NVMe read).

### 5.3  Cache Hit Rates

Target cache hit rates, pending trained routing distributions:

| Scenario | Cache hit rate | Effective tok/s |
|----------|---------------|-----------------|
| Homogeneous topic | ~60% | ~20 |
| Mixed conversation | ~30% | ~15 |
| Topic switching | ~10% | ~12 |
| **Weighted average** | **Target: ~35%** | **Projected: ~10–11 baseline; higher if cache hits reduce transfers** |

---

## 6  Multi-Perspective Decoding Pipeline

### 6.1  Trigger Condition

MPD activates when the base logits indicate low confidence:

$$
\text{activate} = (\max(p) < 0.6) \lor (\text{entropy}(p) > 3.0)
$$

This fires for approximately 20% of generated tokens.

### 6.2  Four Perspectives

When MPD activates:

| Perspective | Modification | Purpose |
|-------------|-------------|---------|
| Base | No change | Reference distribution |
| Jitter | Hidden += N(0, 0.01) | Sensitivity test |
| Antipodal | Temperature × 2 | Explore alternatives |
| Random | Dropout(0.1) on hidden | Robustness check |

### 6.3  Agreement Protocol

```rust
fn mpd_decode(logits: &Tensor, states: &PdrStates, kv: &KVCache) -> Tensor {
    let perspectives = [
        logits.clone(),                           // base
        forward_with_jitter(states, kv),          // jitter
        forward_with_temperature(states, kv, 2.0), // antipodal
        forward_with_dropout(states, kv, 0.1),    // random
    ];
    
    // Pairwise JSD
    let agreement = pairwise_jsd(&perspectives);
    
    if agreement.max_jsd() < THRESHOLD {
        // All perspectives agree → high confidence
        geometric_mean(&perspectives)
    } else {
        // Disagreement → consult HDM
        let context = extract_context(states);
        let retrieved = hdm_retrieve(context);
        blend_with_memory(perspectives, retrieved)
    }
}
```

### 6.4  MPD Compute Cost (Projected; Unmeasured)

Each additional perspective requires a partial forward pass (only the last
few layers, not the full 80):

- Re-run last 5 layers with modification: ~0.4 ms compute
- 3 extra perspectives: ~1.2 ms compute
- Agreement computation: ~0.05 ms
- **Total MPD overhead: ~1.25 ms per activated token**
- Amortised (20% activation): **~0.25 ms/token**

---

## 7  Forward-Mode Evolutionary Adaptation

### 7.1  When It Runs

FMEA runs **between user messages** (during "think time"):

```
User sends message → Model generates response → User reads response
                                                  ↓
                                            FMEA runs here
                                            (2.25 seconds budget)
```

### 7.2  What It Adapts

| Component | Parameters | Method | Update budget |
|-----------|-----------|--------|---------------|
| PDR W_p matrices | 2M (LoRA rank-4) | JVP + NES | 1.5 s |
| Router manifold | 384 positions | NES only | 0.5 s |
| HDM codebook | 1000 entries | Binding update | 0.25 s |

### 7.3  Forgetting Prevention

LoRA adapters are isolated and elastic-anchored:

$$
\mathcal{L}_{\text{anchor}} = \lambda \sum_i \|a_i - a_i^0\|^2
$$

where $a_i^0$ is the initial adapter value and $\lambda = 0.01$.

---

## 8  Startup Sequence

```
1. Load model_config.json         (projected ~1 ms)
2. Load shared layers to VRAM     (projected ~1.7 s at 1 GB/s)
3. Load router weights to VRAM    (projected ~0.04 s)
4. Load SPP config to VRAM        (projected ~0.02 s)
5. Load HDM codebook to VRAM      (projected ~0.005 s)
6. Load embeddings to VRAM        (projected ~0.13 s)
7. Initialise PDR states          (projected ~0.001 s)
8. Initialise KV cache            (projected ~0.001 s)
9. Warm expert cache (8 experts)  (projected ~13.0 s)
10. Run health check              (target ~0.5 s)
                        ──────────────────
              Total startup target:  ~15.4 seconds
```

---

## 9  Shutdown + Persistence

On shutdown, persist:
1. **PDR states** (120 MB) — continuous recurrent memory
2. **FMEA LoRA adapters** (3.75 MB) — learned adaptations
3. **Expert cache frequency table** (1 KB) — for next warm start
4. **HDM bank updates** (variable) — session memory

Projected persistence write: ~124 MB → ~0.12 s.

---

*Next: [§ 12 Validation](12_validation_v2.md)*
