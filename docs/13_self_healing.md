# Section 13: Fixing OMNIS Self-Introduced Problems

## 13.1 Problem 1: Retrieval Poisoning

### Root Cause
Topological memory accepts any insertion. Adversarial, outdated, or factually
wrong entries degrade all future retrievals that surface them.

### Fix: Trust-Scored Memory with Quarantine + Cryptographic Provenance

(Detailed in §11.7, summarized here with additional hardening)

**Three-tier trust architecture:**

```
Tier 1 (Trust 0.9-1.0): System knowledge base — pre-verified, immutable
  Storage: Separate read-only memory partition
  Write policy: Only at system initialization or explicit update
  Size: ~500K entries, ~2.6 GB RAM

Tier 2 (Trust 0.5-0.8): User-verified content
  Storage: Main topological memory, marked as verified
  Write policy: User must confirm or entry auto-promotes after 5 retrievals
  without truth grounding rejection

Tier 3 (Trust 0.0-0.4): Quarantined/unverified
  Storage: Main topological memory, quarantine flag set
  Write policy: Auto on model-generated consolidations and external ingestion
  Retrieval: Only returned if Tier 1+2 return < k results
```

**Content integrity:**
```rust
fn compute_entry_hash(entry: &MemoryEntry) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(&entry.text.as_bytes());
    hasher.update(&entry.provenance.to_bytes());
    hasher.update(&entry.timestamp.to_le_bytes());
    hasher.finalize().into()
}
```

Any entry whose content_hash doesn't match its stored text has been tampered with
and is evicted immediately.

**Periodic audit (background, every 5 minutes):**
- Sample 50 quarantined entries
- Run through truth grounding verifier
- Promote verified entries (trust += 0.2), evict refuted entries
- Cost: 50 * 2ms verifier inference = 100 ms (imperceptible)

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB |
| RAM | 96 MB (48 bytes/entry * 2M — provenance + hash + flags) |
| Latency | 0 ms per retrieval (pre-filtered) |

### Validation Test
Insert 10K poisoned entries (factually wrong) among 2M correct entries.
Measure: retrieval of poisoned entries should be < 1% after quarantine system active.

## 13.2 Problem 2: GLA Lossy Compression

### Root Cause
Gated Linear Attention compresses all history into fixed-size state S ∈ R^(128×128)
per head. At conversation turn 50+, details from turn 1 are effectively lost.

### Fix: Hierarchical State Checkpointing + Selective Windowed Attention

**Mechanism 1: State Checkpoints**

Save full GLA state snapshots at key conversation boundaries:

```rust
struct StateCheckpoint {
    turn_number: u32,
    state: Vec<[f16; 128 * 128]>,  // Per-head state, 32 KB per head
    summary_embedding: [f16; 2048], // Dense summary of this turn
    timestamp: u64,
}

// Storage: per checkpoint = 16 heads * 32 KB + 4 KB = 516 KB
// Keep last 32 checkpoints = 16.5 MB (RAM)
// Policy: checkpoint at every user turn boundary
```

**When to restore:**
- If retrieval surfaces a memory entry timestamped to an old conversation turn,
  AND the current GLA state's gate values for that turn's tokens are < 0.01
  (indicating information was gated out):
  → Load the checkpoint closest to that turn
  → Re-process the retrieved entry through restored state
  → Merge result into current state via weighted addition

```
S_merged = (1 - α) * S_current + α * S_checkpoint_restored
α = relevance_score * 0.3  // Max 30% influence from restored state
```

**Mechanism 2: Selective Windowed Attention on Critical Layers**

For the **last 2 layers** (layers 31-32), use a small sliding window of standard
attention alongside GLA. This provides exact attention over the most recent 256 tokens.

```
Layer 31-32 output:
  h_gla = GLA(input, state)        // Existing O(d_k * d_v) recurrent
  h_attn = WindowedAttn(input, last_256_tokens)  // O(256 * d) standard attention
  h_combined = h_gla + 0.3 * h_attn   // Weighted combination
```

**VRAM cost for windowed attention:**
- KV-cache for 256 tokens * 2 layers: 256 * 2 * 2 * 128 * 128 * 2 bytes = 32 MB
- But we already budget 512 MB for KV-cache in §2 (4096 context full attention).
- Windowed attention uses LESS: repurpose 32 MB of the existing 512 MB KV-cache budget.
- **Net VRAM delta: 0 MB** (reallocation within existing budget).

**Quality impact:**
- Windowed attention on last 2 layers preserves fine-grained coherence for recent text
- GLA on first 30 layers handles long-range pattern recognition
- Checkpoints recover specific old details when retrieval triggers restoration

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB net (reallocation within existing KV-cache budget) |
| RAM | 16.5 MB (32 state checkpoints) |
| Latency | 0 ms normally; +2 ms when checkpoint restoration triggered (~5% of turns) |
| Compute overhead | +3% per token (windowed attention on 2 layers, 256 window) |

### Validation Test
Generate a 100-turn conversation. At turn 100, ask about a specific detail from turn 5.
- Without fix: model recalls detail < 20% of the time
- With fix (checkpoint + retrieval): target >= 70%

## 13.3 Problem 3: Expert Routing Instability

### Root Cause
2-bit quantized router weights have limited precision. Near decision boundaries,
similar inputs may route to different experts, causing inconsistent outputs.

### Fix: Router at FP16 + Routing Cache + Temperature Smoothing

**Mechanism 1: Keep Router at FP16**

The router is tiny: 2048 * 8 = 16,384 parameters per layer.
At FP16: 16,384 * 2 bytes * 32 layers = **1 MB VRAM total**.
This is already in the §2 budget (router listed at FP16, 8 MB with buffers).
Confirming: router is NOT quantized to 2-bit. It stays FP16. No change needed.

**Mechanism 2: Routing Cache (Deterministic)**

Cache routing decisions for recently-seen token contexts:

```rust
struct RoutingCache {
    // Key: hash of last 8 token embeddings (provides context sensitivity)
    // Value: (expert_1_idx, expert_2_idx, confidence)
    cache: LruCache<u64, (u8, u8, f32)>,
    capacity: usize,  // 8192 entries
}

impl RoutingCache {
    fn lookup_or_compute(&mut self, context_hash: u64, router: &Router) -> (u8, u8) {
        if let Some(&(e1, e2, _)) = self.cache.get(&context_hash) {
            return (e1, e2);  // Cache hit: deterministic routing
        }
        let (e1, e2, conf) = router.route(context_hash);
        self.cache.put(context_hash, (e1, e2, conf));
        (e1, e2)
    }
}

// Memory: 8192 * (8 + 2 + 1 + 4) = 123 KB RAM
```

**Mechanism 3: Softmax Temperature Smoothing**

When router logits are close (max - second_max < margin), apply higher temperature
to soften the decision, then take weighted combination of top-2 experts:

```
Standard:  weights = softmax(logits / T)           where T = 1.0
Smoothed:  weights = softmax(logits / T_smooth)    where T_smooth = 2.0

Trigger: when |logit_1 - logit_2| < 0.5
Effect: more balanced mixing of top-2 experts near decision boundary
```

This adds no VRAM cost (just changes the softmax temperature dynamically).

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB (router already FP16) |
| RAM | 123 KB (routing cache) |
| Latency | -0.01 ms (cache hits save router compute) |

### Validation Test
Run 1000 identical prompts. Measure variance in outputs.
- Without fix: output variance (measured as token-level edit distance) > 10%
- With fix: output variance < 2%

## 13.4 Problem 4: Weak 150M Verifier

### Root Cause
150M encoder is too small to reliably judge complex claims. It has limited world
knowledge and may confidently approve plausible-but-wrong statements.

### Fix: Cascaded Verification + Main Model Self-Check

**Architecture: Three-tier verification cascade**

```
Claim arrives for verification
        │
        ▼
┌─────────────────────────┐
│ Tier 1: 150M Verifier    │ ← Fast (2 ms), always runs
│ Confidence threshold: 0.9│
└──────────┬──────────────┘
           │ If confidence < 0.9:
           ▼
┌─────────────────────────┐
│ Tier 2: Main Model       │ ← Slow (100 ms), runs on uncertain claims
│ Self-verification prompt │
│ "Is this claim true given│
│  this evidence? Yes/No"  │
└──────────┬──────────────┘
           │ If Tier 1 and Tier 2 disagree:
           ▼
┌─────────────────────────┐
│ Tier 3: Evidence count   │ ← Heuristic tiebreaker
│ How many memory entries  │
│ support vs contradict?   │
│ Majority vote wins.      │
└─────────────────────────┘
```

**Tier 2 cost analysis:**
- Uses existing main model (no additional VRAM)
- One forward pass: ~987M FLOPs = ~100 ms including retrieval
- Only triggered for claims where Tier 1 confidence is in [0.3, 0.9] range
- Expected trigger rate: ~20% of claims (most claims are clearly true or clearly false)

**Tier 3 is zero-cost** — just counts evidence entries that support vs contradict.

### Per-Output Verification Cost

| Scenario | Probability | Latency |
|----------|------------|---------|
| Tier 1 sufficient (high confidence) | 80% | 2 ms |
| Tier 1 + Tier 2 needed | 18% | 102 ms |
| All three tiers | 2% | 103 ms |
| **Expected latency** | | **2 * 0.8 + 102 * 0.18 + 103 * 0.02 = 22 ms** |

vs previous single-tier: 26.8 ms per output (§7). **Comparable but more accurate.**

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB (reuses existing verifier + main model) |
| RAM | 0 MB |
| Latency | ~22 ms per output (slightly better than before, much more accurate) |

### Validation Test
Create 500 "tricky" claims (plausible but wrong). Measure:
- Tier 1 only: detection rate (baseline)
- Tier 1 + 2 cascade: detection rate (target: +15% improvement)

## 13.5 Problem 5: Latency Regression

### Root Cause
Multi-pass reasoning (verification, retrieval, branching) adds 2-3x overhead.

### Fix: Speculative Execution + Lazy Verification + Adaptive Depth

**Mechanism 1: Parallel Verification (Already Designed)**

The lookahead queue (§8.6) generates tokens ahead while CPU verifies previous tokens.
This overlaps GPU compute with CPU verification, hiding most latency.

**Mechanism 2: Lazy Verification**

Not every output needs full verification. Apply verification proportional to risk:

```rust
fn should_verify(output: &str, context: &Context) -> VerificationDepth {
    // Heuristic: estimate claim density and risk
    let factual_density = count_factual_claims(output) as f32 / word_count(output) as f32;
    let risk_score = context.safety_flags.max_score();
    let length = output.len();

    match (factual_density, risk_score, length) {
        (d, _, _) if d < 0.1 => VerificationDepth::None,    // Casual chat, few facts
        (_, r, _) if r > 0.5 => VerificationDepth::Full,    // Safety-flagged
        (_, _, l) if l < 50  => VerificationDepth::None,     // Short response
        (d, _, _) if d > 0.5 => VerificationDepth::Full,    // Highly factual
        _                     => VerificationDepth::Sample,   // Verify 30% of claims
    }
}

enum VerificationDepth {
    None,           // Skip verification entirely: 0 ms
    Sample,         // Verify 30% of claims: ~7 ms
    Full,           // Verify all claims: ~22 ms
}
```

**Expected depth distribution:**
- None: 40% of outputs (greetings, code, creative writing)
- Sample: 35% of outputs (general questions)
- Full: 25% of outputs (factual queries, safety-flagged)

**Expected average verification latency: 0.4 * 0 + 0.35 * 7 + 0.25 * 22 = 7.95 ms**
vs previous 26.8 ms: **70% latency reduction.**

**Mechanism 3: Adaptive CoT Depth**

Multi-pass chain-of-thought (CoT) runs 1-4 passes. Adapt based on task complexity:

```rust
fn select_cot_depth(prompt: &str) -> u32 {
    let complexity = estimate_complexity(prompt);
    match complexity {
        c if c < 0.3 => 1,  // Simple: single pass (60% of queries)
        c if c < 0.7 => 2,  // Medium: two passes (30% of queries)
        _            => 4,  // Complex: four passes (10% of queries)
    }
}
// Expected passes: 0.6*1 + 0.3*2 + 0.1*4 = 1.6 passes average
// vs fixed 3 passes: 47% compute reduction
```

### Revised Latency Budget

| Component | Before fix | After fix | Improvement |
|-----------|-----------|-----------|-------------|
| First token (TTFT) | 2000 ms | 1200 ms | 40% faster |
| Verification per output | 26.8 ms | 7.95 ms | 70% faster |
| CoT passes (average) | 3.0 | 1.6 | 47% fewer |
| Decode tok/s | 8 | 10 | 25% faster |

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB |
| RAM | 10 KB (complexity estimator heuristics) |
| Trade-off | Quality may decrease for incorrectly classified "simple" queries |

### Validation Test
Run 500 queries spanning simple-to-complex. Measure TTFT p50/p99 and CRS.
Target: TTFT p50 < 1200 ms AND CRS decrease < 0.01 vs full verification.

## 13.6 Problem 6: Cascading Failure

### Root Cause
If expert swap (PCIe), retrieval (RAM), and verification (GPU) all hit worst case
simultaneously, the system has no available compute path and stalls.

### Fix: Circuit Breakers + Graceful Degradation Ladder + Health Monitor

**Degradation Ladder (5 levels):**

```
Level 0: NOMINAL — all systems active
  ↓ (trigger: any subsystem latency > 2x normal)
Level 1: ELEVATED — disable speculative generation, reduce lookahead to 2
  ↓ (trigger: any subsystem latency > 5x normal)
Level 2: DEGRADED — disable verification, reduce to single-pass CoT
  ↓ (trigger: expert swap fails or VRAM OOM)
Level 3: FALLBACK — use single expert only (no MoE), disable retrieval
  ↓ (trigger: GPU completely unresponsive)
Level 4: EMERGENCY — CPU-only inference with smallest model
```

**Circuit Breaker Pattern:**

```rust
struct CircuitBreaker {
    state: BreakerState,
    failure_count: u32,
    failure_threshold: u32,    // 3 consecutive failures → trip
    recovery_timeout_ms: u64,  // 5000 ms before attempting recovery
    last_failure: Instant,
}

enum BreakerState {
    Closed,     // Normal operation
    Open,       // Subsystem disabled (tripped)
    HalfOpen,   // Testing if subsystem recovered
}

impl CircuitBreaker {
    fn should_execute(&mut self) -> bool {
        match self.state {
            BreakerState::Closed => true,
            BreakerState::Open => {
                if self.last_failure.elapsed().as_millis() as u64 > self.recovery_timeout_ms {
                    self.state = BreakerState::HalfOpen;
                    true  // Try once
                } else {
                    false  // Still tripped
                }
            }
            BreakerState::HalfOpen => true,  // Testing
        }
    }

    fn record_success(&mut self) {
        self.failure_count = 0;
        self.state = BreakerState::Closed;
    }

    fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure = Instant::now();
        if self.failure_count >= self.failure_threshold {
            self.state = BreakerState::Open;
        }
    }
}
```

**Independent circuit breakers for each subsystem:**

| Subsystem | Failure threshold | Recovery timeout | Fallback when tripped |
|-----------|------------------|-----------------|----------------------|
| Expert swap (PCIe) | 3 failures | 5 s | Use current resident experts only |
| Retrieval (topological memory) | 3 failures | 10 s | Generate without retrieval |
| Verification | 3 failures | 5 s | Output without verification + warning |
| Continual learning | 1 failure | 60 s | Freeze all LoRA updates |
| Tool execution | 2 failures | 30 s | Disable tool use for this conversation |

**Health Monitor (runs every 1 second):**

```rust
struct HealthMonitor {
    breakers: HashMap<Subsystem, CircuitBreaker>,
    degradation_level: DegradationLevel,
    metrics: SystemMetrics,
}

struct SystemMetrics {
    vram_used: u64,
    ram_used: u64,
    gpu_utilization: f32,
    cpu_utilization: f32,
    pcie_bandwidth_used: f64,
    tokens_per_second: f32,
    p99_latency_ms: f32,
}

impl HealthMonitor {
    fn tick(&mut self) {
        self.metrics = collect_metrics();

        // Auto-escalate degradation level
        let tripped_count = self.breakers.values()
            .filter(|b| matches!(b.state, BreakerState::Open))
            .count();

        self.degradation_level = match tripped_count {
            0 => DegradationLevel::Nominal,
            1 => DegradationLevel::Elevated,
            2 => DegradationLevel::Degraded,
            3 => DegradationLevel::Fallback,
            _ => DegradationLevel::Emergency,
        };

        // Log health status
        log::info!("Health: {:?} | VRAM: {}MB | tok/s: {:.1} | tripped: {}",
            self.degradation_level,
            self.metrics.vram_used / 1_000_000,
            self.metrics.tokens_per_second,
            tripped_count
        );
    }
}
```

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB |
| RAM | 10 KB (breaker state + metrics) |
| Latency | 0 ms (monitoring is async background) |
| CPU | < 0.1% of one core (1-second polling) |

### Validation Test (ST-7: Cascading Failure Test)
Simultaneously:
- Block PCIe transfers (simulate stuck DMA)
- Corrupt HNSW index (simulate memory corruption)
- Kill verifier process
Measure: system must degrade to Level 3/4 within 10 seconds and continue
producing outputs (degraded quality but no crash, no hang).
