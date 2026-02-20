# Section 14: Multilingual Equity, Debiasing, Tokenization & Interpretability

## 14.1 Multilingual Equity

### Problem
Base model is likely English-centric. Vocabulary, training distribution, and
embedding quality are uneven across languages.

### Fix: Language-Aware Expert Routing + Balanced Retrieval

**Mechanism 1: Dedicate 2 of 8 Experts to Non-English Languages**

The MoE architecture has 8 experts. We explicitly assign:
- Experts 0-5: General (English-primary, multi-domain)
- Expert 6: CJK + South/Southeast Asian languages
- Expert 7: European + Arabic + African languages

This is enforced during training of the expert FFN weights, not at inference.
At inference, the router naturally sends non-English tokens to experts 6-7
because those experts produce lower loss for those tokens.

**No VRAM cost** — expert structure already exists, just training allocation.

**Mechanism 2: Language-Balanced Topological Memory**

Reserve partitions of topological memory for non-English content:

| Language group | Reserved entries | RAM cost |
|---------------|-----------------|----------|
| English | 1,200,000 | 6.24 GB |
| CJK (Chinese, Japanese, Korean) | 300,000 | 1.56 GB |
| European (Romance, Germanic, Slavic) | 250,000 | 1.30 GB |
| Arabic + Hindi + Urdu | 100,000 | 0.52 GB |
| Other (African, SE Asian, etc.) | 150,000 | 0.78 GB |
| **Total** | **2,000,000** | **10.4 GB** |

RAM cost is already budgeted (§6). This is a partitioning policy, not additional storage.

**Mechanism 3: Language Detection Router**

Fast CPU-based language detection on input (using trigram frequency analysis):

```rust
fn detect_language(text: &str) -> Language {
    // Trigram frequency matching against 50 language profiles
    // Profiles: ~50 * 1000 trigrams * 4 bytes = 200 KB RAM
    // Latency: 0.02 ms
    let trigrams = extract_trigrams(text);
    let scores = LANGUAGE_PROFILES.iter()
        .map(|(lang, profile)| (lang, cosine_sim(&trigrams, profile)))
        .collect::<Vec<_>>();
    scores.into_iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
}
```

When a non-English language is detected:
- Bias retrieval toward same-language partition of topological memory
- Bias routing toward language-specialized experts (soft bias, not hard override)
- Select appropriate response language

### Costs

| Resource | VRAM | RAM | Latency |
|----------|------|-----|---------|
| Language detection | 0 | 200 KB | 0.02 ms |
| Expert routing bias | 0 | 0 | 0 ms |
| Memory partitioning | 0 | 0 (reallocation) | 0 ms |
| **Total** | **0** | **200 KB** | **0.02 ms** |

### Validation Test
Run multilingual benchmark (MMLU in 10 languages). Target: non-English accuracy
>= 80% of English accuracy for Tier-1 languages (Chinese, Spanish, French, German, Japanese).

[HEURISTIC] Expert specialization for languages assumes sufficient multilingual
training data. If base model was English-only, this mechanism has no effect.
Falsifiable: compare per-language accuracy with and without dedicated experts.

## 14.2 Debiasing

### Problem
Base model inherits biases from training data (gender, racial, cultural).
Continual learning can amplify biases if training samples are skewed.

### Fix: Activation Debiasing Vectors + Balanced Replay

**Mechanism 1: Debiasing via Representation Engineering**

Extends the activation steering from §10.5. Pre-compute bias direction vectors:

```
For each bias dimension (gender, race, age, religion):
  1. Create 200 contrastive pairs:
     ("The doctor arrived. He examined the patient.",
      "The doctor arrived. She examined the patient.")
  2. Compute hidden states for both at each layer
  3. Difference = bias direction vector for that layer
  4. At inference: project out bias direction from hidden states

  h_debiased = h - (h · bias_dir) * bias_dir
```

**Storage: 4 bias dimensions * 32 layers * 2048 * 4 bytes = 1 MB RAM**
(Added to the 1.3 MB steering vectors from §10.5, total 2.3 MB)

**Mechanism 2: Balanced Replay Buffer for Continual Learning**

The OGP-LoRA replay buffer (§5, 4096 entries) must maintain demographic balance:

```rust
struct BalancedReplayBuffer {
    buffer: Vec<Sample>,
    max_size: usize,  // 4096
    demographic_counters: HashMap<String, usize>,
    max_per_demographic: usize,  // 4096 / 8 = 512 per category
}

impl BalancedReplayBuffer {
    fn add(&mut self, sample: Sample) {
        let demo = classify_demographic(&sample);  // CPU heuristic
        if self.demographic_counters[&demo] >= self.max_per_demographic {
            // Evict oldest sample from this demographic
            self.evict_oldest(&demo);
        }
        self.buffer.push(sample);
        *self.demographic_counters.entry(demo).or_insert(0) += 1;
    }
}
```

This prevents continual learning from over-fitting to any single demographic.
Cost: 0 VRAM, ~1 KB RAM (counters).

### Costs

| Resource | VRAM | RAM | Latency |
|----------|------|-----|---------|
| Bias direction vectors | 0 | 1 MB | 0.01 ms/token (projection) |
| Balanced replay | 0 | 1 KB | 0 ms |
| **Total** | **0** | **~1 MB** | **0.01 ms/token** |

### Validation Test
Generate 1000 occupation-related sentences. Measure:
- Gender correlation with occupation (target: < 0.1 Pearson r)
- Demographic parity in sentiment across groups (target: max disparity < 5%)

## 14.3 Tokenization Fixes

### Problem
BPE tokenization causes: character counting errors, spelling mistakes on rare words,
poor handling of non-Latin scripts, code formatting issues.

### Fix: Byte-Level Fallback + Character-Aware Decoding

**Mechanism 1: Byte-Level Fallback Tokenizer**

When the standard BPE tokenizer produces unknown or low-confidence tokens, fall back
to byte-level encoding:

```rust
fn tokenize_robust(text: &str, tokenizer: &BPETokenizer) -> Vec<Token> {
    let tokens = tokenizer.encode(text);

    // Check for UNK tokens or high-entropy segments
    let mut result = Vec::new();
    for (i, token) in tokens.iter().enumerate() {
        if token.id == UNK_TOKEN_ID {
            // Fall back to byte-level encoding for this segment
            let text_segment = tokenizer.decode_span(i, i+1);
            let byte_tokens = byte_encode(text_segment);
            result.extend(byte_tokens);
        } else {
            result.push(*token);
        }
    }
    result
}

fn byte_encode(text: &str) -> Vec<Token> {
    // Each byte maps to token ID 256 + byte_value
    // This uses token IDs 256-511 (reserved range in vocabulary)
    text.as_bytes().iter()
        .map(|&b| Token { id: 256 + b as u32 })
        .collect()
}
```

**VRAM cost: 0** (just a different tokenization path, same embedding table).
The embedding table already contains entries for byte-level tokens (positions 256-511).

**Mechanism 2: Character-Aware Post-Processing**

After generation, apply character-level verification for:
- Spelling: check generated words against a dictionary (CPU, 2 MB dictionary in RAM)
- Character count: when the model claims "X has N characters", verify and correct
- Code formatting: re-indent generated code using tree-sitter parser (CPU, 0 VRAM)

```rust
fn post_process_tokens(output: &str) -> String {
    let mut result = output.to_string();

    // Fix spelling (only for words the model is uncertain about)
    result = spellcheck_uncertain_words(&result);

    // Fix character count claims
    result = verify_character_counts(&result);

    // Fix code formatting
    if contains_code_block(&result) {
        result = reformat_code_blocks(&result);
    }

    result
}
```

### Costs

| Resource | VRAM | RAM | Latency |
|----------|------|-----|---------|
| Byte-level fallback | 0 | 0 | 0 ms (same pipeline) |
| Dictionary | 0 | 2 MB | 0.1 ms per output |
| Tree-sitter | 0 | 5 MB | 1 ms per code block |
| **Total** | **0** | **7 MB** | **0.1-1 ms per output** |

### Validation Test
Generate 200 outputs containing character counts, non-Latin text, and code.
- Character count accuracy: target >= 95% (vs ~60% baseline)
- Non-Latin text coherence: manual eval by native speakers

## 14.4 Interpretability

### Problem
The model is a black box. Users and developers cannot understand why a specific
output was generated or which knowledge sources contributed.

### Fix: Attribution Trace + Expert Contribution Logging + Retrieval Citation

**Mechanism 1: Attribution Trace (Lightweight)**

For each generated token, record which components contributed most:

```rust
struct TokenAttribution {
    token_id: u32,
    token_text: String,
    expert_contributions: [f32; 2],     // Weight of each active expert
    retrieval_influence: f32,           // How much retrieval affected this token
    gla_state_norm: f32,               // How much history influenced this token
    top_attention_positions: [u16; 4],  // Which input positions were most influential
    verification_result: Option<Verdict>, // If this token was in a verified claim
}
```

**Storage: ~40 bytes per token. For a 500-token output: 20 KB RAM (temporary).**
Logged to SQLite for post-hoc analysis. Not stored in VRAM.

**Mechanism 2: Expert Contribution Logging**

After each forward pass, record which expert produced the largest activation:

```
For each layer l, token t:
  expert_1_activation = ||FFN_expert1(h_t)||
  expert_2_activation = ||FFN_expert2(h_t)||
  dominant_expert = argmax(expert_1_activation, expert_2_activation)
```

This adds one norm computation per layer per token:
- 32 layers * 2048 * 2 experts = 131K FLOPs per token (0.01% of total)
- Stored in attribution trace (RAM)

**Mechanism 3: Retrieval Citation**

When retrieval (§6) surfaces evidence that influences the output, include
citations in the output:

```
Model output: "Paris has a population of approximately 2.1 million [1]."

[1] Retrieved from topological memory entry #1234567,
    trust score: 0.9, source: "Wikipedia: Paris demographics, 2024-01"
```

This is a post-processing step:
- After verification, match verified claims to evidence entries
- Append footnotes for claims where evidence was used
- CPU-only, 0 VRAM, ~0.5 ms per output

### User-Facing Interpretability Dashboard

For developer/debug use, expose:

```json
{
  "output": "Paris has a population of approximately 2.1 million.",
  "attribution": {
    "expert_routing": {"layer_12": "expert_3", "layer_24": "expert_6"},
    "retrieval_used": true,
    "retrieval_entries": [1234567],
    "retrieval_influence": 0.72,
    "verification": {"verdict": "SUPPORTED", "confidence": 0.94},
    "gla_state_utilization": 0.45,
    "degradation_level": "NOMINAL"
  }
}
```

### Costs

| Resource | VRAM | RAM | Latency |
|----------|------|-----|---------|
| Attribution trace | 0 | 20 KB per output | 0.1 ms/token |
| Expert logging | 0 | (included in trace) | 0.01 ms/token |
| Citation generation | 0 | 0 | 0.5 ms per output |
| SQLite logging | 0 | 0 (disk-backed) | 1 ms per output |
| **Total** | **0** | **~20 KB per output** | **~0.6 ms per output** |

### Validation Test
- Generate 100 outputs with attribution traces
- Manually verify that cited evidence entries are actually relevant to claims
- Target: citation relevance >= 85% (human-judged)

## 14.5 Section Summary: Total Costs

| Subsystem | VRAM | RAM | Latency/token | Latency/output |
|-----------|------|-----|---------------|----------------|
| Multilingual equity | 0 | 200 KB | 0.02 ms | 0 |
| Debiasing | 0 | 1 MB | 0.01 ms | 0 |
| Tokenization fixes | 0 | 7 MB | 0 | 1 ms |
| Interpretability | 0 | 20 KB/output | 0.1 ms | 1.1 ms |
| **TOTAL** | **0 MB** | **~8.2 MB** | **0.13 ms** | **2.1 ms** |

All additions are VRAM-neutral and fit easily in RAM headroom.
