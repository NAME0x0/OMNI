# Section 7: Truth Grounding Verifier

## 7.1 Overview

The truth grounding system is a dedicated verifier that can **halt** generation,
**branch** into alternative reasoning paths, and **retrieve** evidence from topological
memory. It operates as a post-hoc checker on the main model's outputs AND as an
inline monitor during chain-of-thought generation.

## 7.2 Verifier Architecture

### Model Specification

| Property | Value |
|----------|-------|
| Architecture | Encoder-only transformer (BERT-like) |
| Parameters | 150M |
| Quantization | 2.5-bit mixed (§4) |
| VRAM footprint | 400 MB (§2 budget) |
| Context window | 1024 tokens (claim + evidence) |
| Output | 3 logits: {SUPPORTED, REFUTED, INSUFFICIENT} + calibrated probability |
| Inference time | ~2 ms per claim (batch of 1) |

### Why Encoder-Only

- Verification is a classification task, not generation. Encoder-only is faster and
  more parameter-efficient for NLI (natural language inference).
- 150M params at 2.5-bit = 46.9 MB weights (fits easily alongside main model).
- Activation scratch: 1024 * 768 * 2 bytes = 1.5 MB per inference (within budget).

## 7.3 Formal Constraints

### Constraint 1: Completeness of Verification

Every output token sequence T of length > 32 tokens MUST be verified before being
presented to the user. Shorter outputs (greetings, acknowledgments) are exempt.

### Constraint 2: Evidence Requirement

The verifier MUST receive at least one evidence passage for each claim. If no evidence
is retrieved (topological memory returns empty), the output is classified as
INSUFFICIENT (not SUPPORTED or REFUTED).

### Constraint 3: Calibration Bound

The verifier's confidence scores must satisfy:
- For claims classified as SUPPORTED with confidence p >= 0.8: actual accuracy >= 0.75
- For claims classified as REFUTED with confidence p >= 0.8: actual accuracy >= 0.75
- Calibration error (ECE) <= 0.10 across all bins

[HEURISTIC] Calibration bound of ECE <= 0.10 is based on published NLI model
calibration (Desai & Durrett 2020). Falsifiable test: run verifier on 5000 claims
with known labels, compute ECE. If ECE > 0.10, apply temperature scaling.

### Constraint 4: Halting Criterion

If the verifier classifies any claim in the output as REFUTED with confidence > 0.7:
- **Halt** current generation
- **Branch** to alternative generation (re-run with different sampling or prompt)
- Maximum branches: 3 (then return best-scored alternative or abstain)

### Constraint 5: Abstention

If after 3 branches, no output passes verification:
- Return a structured abstention: "I cannot provide a verified answer to this question.
  [Reason: {verifier_explanation}]"
- Log the failure for continual learning (§5) to address the gap.

## 7.4 I/O Contract

### Input Contract

```
VerifierInput {
    claim: String,              // Extracted claim from model output (max 256 tokens)
    evidence: Vec<String>,      // 1-5 evidence passages from topological memory (max 512 tokens each)
    context: Option<String>,    // Original user query (max 256 tokens)
    mode: VerifyMode,           // INLINE (during generation) or POSTHOC (after generation)
}
```

### Output Contract

```
VerifierOutput {
    verdict: Verdict,           // SUPPORTED | REFUTED | INSUFFICIENT
    confidence: f32,            // [0.0, 1.0], calibrated probability
    explanation: String,        // Natural language explanation (max 128 tokens)
    evidence_used: Vec<u64>,    // IDs of evidence entries that contributed
    action: Action,             // CONTINUE | HALT | BRANCH | ABSTAIN
    latency_ms: f32,            // Self-reported inference time
}
```

### Action Decision Logic

```
fn decide_action(verdict: Verdict, confidence: f32, branch_count: u32) -> Action {
    match (verdict, confidence, branch_count) {
        (SUPPORTED, c, _) if c >= 0.6     => CONTINUE,
        (SUPPORTED, c, _) if c < 0.6      => CONTINUE, // Low confidence but not refuted
        (REFUTED, c, b) if c >= 0.7 && b < 3 => BRANCH,
        (REFUTED, c, _) if c >= 0.7       => ABSTAIN,  // Exhausted branches
        (REFUTED, c, _) if c < 0.7        => CONTINUE, // Low-confidence refutation, allow
        (INSUFFICIENT, _, b) if b < 3     => BRANCH,   // Try to find better evidence
        (INSUFFICIENT, _, _)              => CONTINUE,  // Accept with caveat
        _                                 => CONTINUE,
    }
}
```

## 7.5 Claim Extraction Pipeline

Before verification, the model output must be decomposed into atomic claims.

```
extract_claims(output_text: &str) -> Vec<String> {
    // Step 1: Sentence segmentation (rule-based, no ML)
    let sentences = segment_sentences(output_text);

    // Step 2: Filter non-factual sentences
    // Remove questions, imperatives, hedged opinions ("I think", "maybe")
    let factual = sentences.filter(|s| is_factual_assertion(s));

    // Step 3: Decompose compound claims
    // "Paris is in France and Berlin is in Germany" → 2 claims
    let atomic = factual.flat_map(|s| split_conjunctions(s));

    // Step 4: Deduplicate
    let unique = deduplicate(atomic);

    return unique;  // Typically 3-15 claims per output
}
```

**Claim extraction is CPU-only, rule-based, ~0.5 ms per output.**

## 7.6 Verification Pipeline

### Full Flow (Post-hoc Mode)

```
verify_output(output: &str, memory: &TopologicalMemory) -> VerificationResult {
    let claims = extract_claims(output);
    let mut results = Vec::new();

    for claim in &claims {
        // 1. Retrieve evidence from topological memory
        let query_vec = embed(claim);  // Reuse main model's embedding layer
        let evidence = memory.retrieve(query_vec, k=5);  // §6 retrieval, ~0.62ms

        // 2. Run verifier model
        let input = VerifierInput {
            claim: claim.clone(),
            evidence: evidence.iter().map(|e| e.text.clone()).collect(),
            context: None,
            mode: VerifyMode::PostHoc,
        };
        let result = verifier_model.infer(input);  // ~2ms GPU inference
        results.push(result);
    }

    // 3. Aggregate results
    let worst = results.iter().min_by_key(|r| match r.verdict {
        REFUTED => 0, INSUFFICIENT => 1, SUPPORTED => 2
    });

    return VerificationResult {
        per_claim: results,
        overall_action: worst.action,
        overall_confidence: results.iter().map(|r| r.confidence).min(),
    };
}
```

### Latency Analysis (Post-hoc)

| Step | Per claim | Total (10 claims) |
|------|-----------|-------------------|
| Claim extraction | N/A | 0.5 ms |
| Evidence retrieval | 0.62 ms | 6.2 ms |
| Verifier inference | 2 ms | 20 ms |
| Action decision | 0.01 ms | 0.1 ms |
| **Total** | | **26.8 ms** |

At 8 tok/s decode and ~50 tokens per output before verification: verification adds
26.8 ms after every ~6.25 seconds of generation. **Overhead: 0.4%** of generation time.

### Inline Mode (During Generation)

For inline monitoring, verification runs every 32 tokens on a sliding window:
- Extract claims from last 32 tokens + context
- Run verification in parallel with next token generation (async)
- If HALT action returned, interrupt generation before next token is committed

## 7.7 Measurable Validation Metrics

| Metric | Definition | Target | Measurement Method |
|--------|-----------|--------|-------------------|
| **Claim-check pass rate** | % of claims classified SUPPORTED | >= 85% on benign inputs | Run on 1000 factual paragraphs from Wikipedia; count SUPPORTED |
| **Abstain rate** | % of queries where system abstains | 5-15% (too low = overconfident, too high = useless) | Run on 1000 mixed queries (factual + unanswerable) |
| **False positive rate** | % of incorrect claims classified SUPPORTED | <= 10% | Run on 500 known-false claims; count SUPPORTED |
| **False negative rate** | % of correct claims classified REFUTED | <= 10% | Run on 500 known-true claims; count REFUTED |
| **Calibration (ECE)** | Expected calibration error | <= 0.10 | 10-bin histogram of confidence vs accuracy |
| **Latency p99** | 99th percentile verification time | <= 50 ms | Profile 10000 verifications |
| **Branch success rate** | % of branches that improve verdict | >= 40% | Track verdict changes across branches |

## 7.8 Memory Residency

| Component | Location | Size |
|-----------|----------|------|
| Verifier weights | VRAM | 400 MB (§2 budget) |
| Verifier activations | VRAM scratch | 1.5 MB (reuses activation scratch) |
| Claim cache (last 100 claims) | Host RAM | ~100 KB |
| Verification log (last 10K results) | SQLite on disk | ~10 MB |

## 7.9 Failure Modes

| Failure | Detection | Fallback |
|---------|-----------|----------|
| Verifier always returns SUPPORTED (degenerate) | Monitor SUPPORTED rate > 99% over 100 outputs | Disable verifier; flag to user; retrain |
| Verifier always returns REFUTED (degenerate) | Monitor REFUTED rate > 50% over 100 outputs | Increase confidence threshold to 0.9 for HALT action |
| Evidence retrieval returns irrelevant passages | Average cosine similarity of retrieved evidence < 0.3 | Mark all claims as INSUFFICIENT; do not HALT |
| Verifier latency spike > 100ms | Timer exceeds 100ms | Skip verification for this output; log |
| Verifier model corrupted (weights invalid) | NaN in output logits | Fall back to rule-based heuristic checker (keyword match) |
