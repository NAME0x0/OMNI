# Section 10: Alignment & Safety System

## 10.1 Design Constraint

No dedicated reward model in VRAM. Total additional VRAM budget: 0 MB.
All alignment mechanisms must be either:
- Embedded in existing components (verifier, router, embeddings)
- CPU/RAM-resident only
- Zero-cost at inference (compile-time or load-time)

## 10.2 Architecture: Four-Layer Safety Stack

```
Layer 4: Constitutional Self-Audit    (post-generation, CPU)
Layer 3: Activation Steering           (inference-time, VRAM-neutral)
Layer 2: Output Safety Classifier      (reuses verifier encoder, 0 extra VRAM)
Layer 1: Input Sanitization            (pre-inference, CPU-only)
```

Each layer operates independently. A harmful output must pass ALL four layers
to reach the user. Defense-in-depth: any single layer can block.

## 10.3 Layer 1: Input Sanitization (CPU-Only)

### Purpose
Block or transform inputs that attempt to elicit harmful outputs before they
reach the model.

### Implementation

```rust
struct InputSanitizer {
    blocked_patterns: Vec<regex::Regex>,   // ~500 patterns, 200 KB RAM
    category_classifier: TinyClassifier,   // Bag-of-words, 50 KB RAM
    transform_rules: Vec<TransformRule>,   // Rewrite rules, 10 KB RAM
}

enum SanitizeResult {
    Pass(String),              // Input is safe, possibly transformed
    Block(String),             // Input blocked, reason provided
    Flag(String, Vec<String>), // Input passed but flagged for Layer 2-4 scrutiny
}
```

### Category Classifier (Bag-of-Words)

A CPU-only bag-of-words model trained on harmful/benign input pairs:
- Vocabulary: 10K terms, each mapped to a 16-dim feature vector
- Single hidden layer: 160 → 64 → 8 (8 safety categories)
- Total parameters: 10K * 16 + 160 * 64 + 64 * 8 = 171,264 floats = **685 KB RAM**
- Inference: ~0.01 ms (pure CPU, trivial)

### Safety Categories

| ID | Category | Action on detect |
|----|----------|-----------------|
| 0 | Violence/harm instructions | Block |
| 1 | CSAM-related | Block + log |
| 2 | Weapons/explosives synthesis | Block |
| 3 | PII extraction attempts | Block |
| 4 | Self-harm/suicide | Redirect to helpline text |
| 5 | Bias/discrimination elicitation | Flag for Layer 3 steering |
| 6 | Deception/manipulation | Flag for Layer 4 audit |
| 7 | Benign | Pass |

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB |
| RAM | 0.9 MB (patterns + classifier + rules) |
| Latency | 0.05 ms per input |
| False positive rate target | < 2% on benign inputs |

### Failure Mode
- Classifier misses novel attack phrasing → caught by Layer 2-4
- Regex patterns outdated → update via continual learning feedback loop
- **Fallback**: If sanitizer crashes, default to PASS with all-layers-flagged

### Falsifiable Test
Run 1000 known-harmful inputs + 1000 benign inputs. Measure:
- Block rate on harmful: target >= 85%
- Block rate on benign: target <= 2%

## 10.4 Layer 2: Output Safety Classifier (Verifier Reuse)

### Purpose
Classify generated output as safe/unsafe BEFORE presenting to user.
Reuses the existing 150M truth grounding verifier encoder — zero additional VRAM.

### Implementation

The verifier (§7) is an encoder-only transformer. We add a **safety classification
head** as additional output logits — 768 → 8 safety categories.

Additional parameters: 768 * 8 = 6,144 floats = **24 KB** (negligible VRAM delta).

```
VerifierOutput (extended):
  verdict: SUPPORTED | REFUTED | INSUFFICIENT     # existing
  confidence: f32                                   # existing
  safety: SafetyVerdict                             # NEW
  safety_scores: [f32; 8]                          # NEW: per-category scores
```

### Safety Verdict Logic

```rust
fn classify_safety(output_text: &str, verifier: &Verifier) -> SafetyVerdict {
    // Tokenize output, run through verifier encoder (already loaded in VRAM)
    let hidden = verifier.encode(output_text);  // Reuses existing forward pass
    let safety_logits = verifier.safety_head(hidden);  // 768 → 8, negligible compute

    // Check each category
    for (cat_id, score) in safety_logits.iter().enumerate() {
        if *score > SAFETY_THRESHOLDS[cat_id] {
            return SafetyVerdict::Blocked {
                category: cat_id,
                score: *score,
                reason: CATEGORY_NAMES[cat_id].to_string(),
            };
        }
    }
    SafetyVerdict::Safe
}
```

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 24 KB (safety head weights — negligible) |
| RAM | 0 MB additional |
| Latency | 0.1 ms additional (piggybacks on verifier forward pass) |
| False positive rate target | < 3% on benign outputs |

### Failure Mode
- Safety head undertrained → outputs harmful content labeled safe
- **Fallback**: Layer 3 activation steering + Layer 4 constitutional audit

### Falsifiable Test
Run verifier on 500 known-harmful outputs + 500 benign outputs.
Target AUROC >= 0.92 for safety classification.

## 10.5 Layer 3: Representation Engineering / Activation Steering

### Purpose
Steer model activations at inference time to suppress harmful directions in
representation space, without modifying weights.

### Method: Contrast Vector Steering (RepE)

Pre-compute **steering vectors** from contrastive pairs:
- Pair: (harmful_prompt, harmless_variant) → compute hidden state difference
- Average over 500+ pairs → one steering vector per layer
- At inference: subtract scaled steering vector from activations

```
For each layer l:
  h_l = GLA_l(input)                          # Normal forward pass
  h_l_steered = h_l - α * steering_vector_l   # Subtract harmful direction
  # α = steering strength, tuned per layer (0.5 to 2.0)
```

### Steering Vector Storage

- 32 layers * 2048 dimensions * 4 bytes (FP32) = **262 KB total**
- Stored in: **Host RAM** (loaded to VRAM activation scratch during inference)
- VRAM cost: 0 MB persistent (vectors loaded into existing scratch buffer temporarily)
- Transfer: 262 KB / 25 GB/s = 0.01 ms via PCIe (negligible)

### Multiple Steering Dimensions

| Steering Vector | Purpose | α range |
|----------------|---------|---------|
| anti-harm | Suppress violent/dangerous content | 1.0-2.0 |
| anti-bias | Reduce demographic bias in outputs | 0.5-1.5 |
| anti-sycophancy | Reduce agreement bias | 0.5-1.0 |
| pro-honesty | Increase hedging and uncertainty expression | 0.5-1.5 |
| anti-deception | Suppress manipulative language patterns | 1.0-2.0 |

Total: 5 vectors * 262 KB = **1.3 MB RAM**

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB persistent (borrows scratch buffer) |
| RAM | 1.3 MB |
| Latency | 0.02 ms per token (vector subtraction, 32 layers) |
| Compute | 32 * 2048 = 65,536 FLOPs per token (negligible vs 987M total) |

### Failure Mode
- Steering too aggressive (α too high) → coherence degrades, outputs become generic
- Steering too weak → harmful content passes through
- **Fallback**: If output quality drops > 5% on benign benchmarks, reduce α by 50%
- **Adaptive α**: Monitor output perplexity; if perplexity increases > 20% vs
  unsteered baseline, reduce α dynamically

### Falsifiable Test
1. Generate 500 outputs with steering OFF on harmful prompts → measure harm rate H_off
2. Generate 500 outputs with steering ON on harmful prompts → measure harm rate H_on
3. Generate 500 outputs with steering ON on benign prompts → measure quality Q_on
4. Required: H_on < 0.2 * H_off AND Q_on > 0.95 * Q_off

[HEURISTIC] RepE effectiveness is demonstrated in Zou et al. 2023
("Representation Engineering") but not yet validated on 2-bit quantized MoE models.
The quantization may distort the representation space, making steering vectors less
effective. Falsifiable: measure steering vector cosine similarity before/after quant.

## 10.6 Layer 4: Constitutional Self-Audit (CPU Post-Processing)

### Purpose
Final check: the model's own output is evaluated against a set of constitutional
principles using rule-based pattern matching + the model itself as judge.

### Constitutional Principles (Hardcoded)

```rust
const CONSTITUTION: &[&str] = &[
    "The output must not provide instructions for causing physical harm to people.",
    "The output must not generate content sexualizing minors.",
    "The output must not help create weapons of mass destruction.",
    "The output must not impersonate real people to spread misinformation.",
    "The output must not provide methods to circumvent safety systems.",
    "The output must acknowledge uncertainty rather than fabricate facts.",
    "The output must not discriminate based on protected characteristics.",
    "The output must not assist in illegal surveillance or stalking.",
    "The output must refuse requests that would violate user privacy.",
    "The output must not generate content designed to manipulate or deceive.",
];
```

### Audit Pipeline

```
1. Rule-based check (CPU, regex + keyword):
   - Fast scan against 200 harmful output patterns
   - If match: BLOCK immediately (0.1 ms)

2. Self-consistency check (uses main model, 1 GPU forward pass):
   - Prompt: "Does the following output violate any of these principles? [principles] [output]"
   - Run through main model as yes/no classification
   - Cost: ~987M FLOPs = 0.2 ms (one additional token generation)
   - Only triggered for FLAGGED outputs (from Layer 1 or when Layer 2 score is borderline)

3. Principle-specific scoring:
   - For each of 10 principles, score 0-1 how well the output complies
   - If any principle score < 0.3: BLOCK
   - If any principle score < 0.6: WARN (append disclaimer)
```

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB (reuses main model for self-audit) |
| RAM | 50 KB (principles + patterns) |
| Latency (rule-based only) | 0.1 ms |
| Latency (with self-consistency) | 50-100 ms (rare, only for flagged outputs) |
| Frequency of self-consistency check | ~5% of outputs (only flagged ones) |

### Failure Mode
- Self-audit agrees with harmful output (model marks its own output as safe)
- **Fallback**: Layer 2 (verifier-based) provides independent judgment
- If both Layer 2 and Layer 4 disagree, conservative action: BLOCK

## 10.7 Total Alignment System Costs

| Layer | VRAM | RAM | Latency/token | Latency/output |
|-------|------|-----|---------------|----------------|
| L1: Input Sanitizer | 0 | 0.9 MB | N/A | 0.05 ms |
| L2: Safety Classifier | 24 KB | 0 | 0 (piggyback) | 0.1 ms |
| L3: Activation Steering | 0 | 1.3 MB | 0.02 ms | N/A |
| L4: Constitutional Audit | 0 | 50 KB | N/A | 0.1 ms (rule) / 100 ms (self-check) |
| **TOTAL** | **24 KB** | **2.25 MB** | **0.02 ms** | **0.25 ms typical** |

**VRAM impact: negligible (24 KB on a 3676 MB budget = 0.0007%)**
**RAM impact: 2.25 MB on 7148 MB free = 0.03%**
**Latency impact: 0.02 ms/token + 0.25 ms/output = imperceptible**

## 10.8 Interaction with Truth Grounding Verifier

The safety system and truth grounding verifier share the encoder:

```
                    ┌─────────────────────┐
 Output text ──────►│  Verifier Encoder    │
                    │  (150M params, VRAM) │
                    └──────┬──────┬───────┘
                           │      │
                    ┌──────▼──┐ ┌─▼──────────┐
                    │ Truth    │ │ Safety     │
                    │ Head     │ │ Head       │
                    │ (3 cls)  │ │ (8 cls)    │
                    └──────────┘ └────────────┘
```

Both heads run in a single forward pass. Total additional cost of safety:
one extra linear layer (768 → 8) = 0.1 ms.
