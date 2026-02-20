# Section 11: Adversarial Robustness & Prompt Injection Defense

## 11.1 Threat Model

Eight attack vectors, each with a dedicated defense:

| ID | Attack Vector | Severity | Frequency |
|----|--------------|----------|-----------|
| AV1 | Direct prompt injection | High | Very common |
| AV2 | Indirect prompt injection (via retrieved docs) | Critical | Common |
| AV3 | Jailbreak attacks (DAN, roleplay) | High | Very common |
| AV4 | Token smuggling (unicode, zero-width) | Medium | Uncommon |
| AV5 | Embedding-space adversarial perturbations | Low | Rare (requires model access) |
| AV6 | Memory poisoning (topological memory) | Critical | Uncommon |
| AV7 | Tool abuse (harmful tool invocations) | High | Common |
| AV8 | Context window manipulation | Medium | Common |

## 11.2 Defense AV1: Direct Prompt Injection Detection

### Method: Instruction Hierarchy + Perplexity Spike Detection

**Principle**: System prompts are immutable and highest-priority. User content
that attempts to override system instructions is detected and neutralized.

### Implementation

```rust
struct PromptStructure {
    system_region: TokenRange,    // Tokens 0..S — IMMUTABLE
    user_region: TokenRange,      // Tokens S..U — monitored
    assistant_region: TokenRange, // Tokens U..A — generated
}

fn detect_injection(input: &str, model: &Model) -> InjectionVerdict {
    // Step 1: Pattern matching (fast, CPU)
    let pattern_score = scan_injection_patterns(input);
    // Patterns: "ignore previous", "system prompt:", "you are now", "new instructions:",
    //           "disregard", "override", "forget everything", etc.
    // 150 patterns, regex-compiled. ~0.02 ms.

    if pattern_score > 0.9 {
        return InjectionVerdict::Blocked("Direct injection pattern detected");
    }

    // Step 2: Perplexity anomaly detection
    // Compute per-token perplexity of user input using the main model.
    // Injection attempts often have anomalously LOW perplexity in the
    // "instruction" part (because they mimic system prompts) and HIGH
    // perplexity at the boundary between real and injected content.
    let perplexities = model.per_token_perplexity(input);
    let boundary_spike = detect_perplexity_spike(&perplexities);

    if boundary_spike > THRESHOLD_PPL_SPIKE {  // threshold = 3.0 std devs
        return InjectionVerdict::Flagged("Perplexity boundary spike");
    }

    // Step 3: Instruction-content separation
    // If the input contains imperative sentences directed at the model
    // ("you must", "always respond", "act as"), flag for scrutiny.
    let imperative_density = count_imperatives(input) as f32 / word_count(input) as f32;
    if imperative_density > 0.15 {  // >15% imperative sentences
        return InjectionVerdict::Flagged("High imperative density");
    }

    InjectionVerdict::Clean
}
```

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB (uses existing model for perplexity) |
| RAM | 100 KB (patterns) |
| Latency | 0.02 ms (pattern) + 5 ms (perplexity, optional) |

### Validation
- Dataset: 500 injection attempts + 500 benign queries
- Target: Detection rate >= 90%, False positive rate <= 3%

## 11.3 Defense AV2: Indirect Prompt Injection (Retrieved Documents)

### Method: Retrieval Quarantine + Content Signing

Retrieved documents from topological memory could contain injected instructions.

```rust
fn safe_retrieve(query: &[f32], memory: &TopologicalMemory) -> Vec<SafeEntry> {
    let raw_results = memory.retrieve(query, k = 32);

    raw_results.into_iter().filter_map(|entry| {
        // Step 1: Check trust level
        if entry.trust_score < 0.3 {
            return None;  // Untrusted entries excluded from retrieval
        }

        // Step 2: Strip any instruction-like content from retrieved text
        let sanitized_text = strip_instructions(entry.text);

        // Step 3: Wrap in data markers (model trained to treat these as DATA not INSTRUCTION)
        let wrapped = format!(
            "[RETRIEVED_DATA_BEGIN]{}[RETRIEVED_DATA_END]",
            sanitized_text
        );

        Some(SafeEntry { text: wrapped, trust: entry.trust_score, ..entry })
    }).collect()
}

fn strip_instructions(text: &str) -> String {
    // Remove imperative sentences, prompt-like patterns
    // Keep only declarative factual content
    let sentences = segment_sentences(text);
    sentences.into_iter()
        .filter(|s| !is_imperative(s) && !looks_like_prompt(s))
        .collect::<Vec<_>>()
        .join(" ")
}
```

### Trust Score Assignment

| Source | Trust Score | Rationale |
|--------|------------|-----------|
| System-provided knowledge base | 1.0 | Pre-verified at load time |
| User-confirmed corrections | 0.8 | User explicitly approved |
| Model self-generated summaries | 0.5 | May contain errors |
| External web content (if ever added) | 0.2 | Untrusted by default |
| Unknown/legacy entries | 0.3 | Require re-verification |

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB |
| RAM | 4 bytes per memory entry (trust score) = 8 MB for 2M entries |
| Latency | 0.1 ms per retrieval (filtering + wrapping) |

## 11.4 Defense AV3: Jailbreak Detection

### Method: Multi-Signal Jailbreak Classifier

Jailbreaks (DAN, roleplay, encoding tricks) are detected via an ensemble of
lightweight signals, all CPU-computed:

```rust
struct JailbreakDetector {
    // Signal 1: Roleplay instruction detection
    roleplay_patterns: Vec<regex::Regex>,  // "pretend you are", "act as", "you are DAN"

    // Signal 2: Encoding detection (base64, rot13, hex in user input)
    encoding_detector: EncodingDetector,

    // Signal 3: Constraint relaxation phrases
    relaxation_patterns: Vec<regex::Regex>,  // "no restrictions", "without limits", "unfiltered"

    // Signal 4: Output format manipulation
    format_patterns: Vec<regex::Regex>,  // "respond only in code", "use special format"

    // Signal 5: Historical jailbreak signature hashes
    known_jailbreak_hashes: HashSet<u64>,  // SimHash of 2000 known jailbreaks
}

fn detect_jailbreak(input: &str, detector: &JailbreakDetector) -> f32 {
    let mut score: f32 = 0.0;
    let mut signals: u32 = 0;

    // Each signal contributes 0-1; final score = weighted average
    if let Some(s) = detector.check_roleplay(input)    { score += s * 0.25; signals += 1; }
    if let Some(s) = detector.check_encoding(input)     { score += s * 0.20; signals += 1; }
    if let Some(s) = detector.check_relaxation(input)   { score += s * 0.25; signals += 1; }
    if let Some(s) = detector.check_format(input)       { score += s * 0.10; signals += 1; }
    if let Some(s) = detector.check_known_hash(input)   { score += s * 0.20; signals += 1; }

    score  // 0.0 = safe, 1.0 = definite jailbreak
}
```

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB |
| RAM | 500 KB (patterns + hashes) |
| Latency | 0.1 ms per input |
| Detection target | >= 85% of known jailbreaks, >= 60% of novel jailbreaks |

## 11.5 Defense AV4: Token Smuggling

### Method: Unicode Normalization + Homoglyph Detection

```rust
fn sanitize_tokens(input: &str) -> String {
    let mut clean = input.to_string();

    // Step 1: NFKC normalization (collapses homoglyphs)
    clean = unicode_normalization::nfkc(&clean);

    // Step 2: Remove zero-width characters
    clean.retain(|c| {
        !matches!(c,
            '\u{200B}' | // zero-width space
            '\u{200C}' | // zero-width non-joiner
            '\u{200D}' | // zero-width joiner
            '\u{FEFF}' | // BOM
            '\u{00AD}' | // soft hyphen
            '\u{2060}' | // word joiner
            '\u{180E}'   // mongolian vowel separator
        )
    });

    // Step 3: Detect mixed-script attacks (Cyrillic 'а' masquerading as Latin 'a')
    let scripts = detect_scripts(&clean);
    if scripts.len() > 2 {
        // Flag: unusual script mixing (normal text is 1-2 scripts)
        log::warn!("Mixed-script input detected: {:?}", scripts);
    }

    // Step 4: Replace confusable characters with canonical forms
    clean = replace_confusables(&clean);

    clean
}
```

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB |
| RAM | 200 KB (confusables table) |
| Latency | 0.05 ms per input |

## 11.6 Defense AV5: Embedding-Space Adversarial Detection

### Method: Mahalanobis Distance Anomaly Detection

Adversarial perturbations push embeddings into low-density regions of the
embedding space. Detect by measuring distance from training distribution.

```
For input embedding e:
  μ = mean of training embeddings (precomputed, stored in RAM)
  Σ^{-1} = inverse covariance (precomputed, stored in RAM)
  d_mahal = sqrt((e - μ)^T Σ^{-1} (e - μ))

  If d_mahal > threshold (6.0 std devs): flag as adversarial
```

### Storage

| Component | Size |
|-----------|------|
| μ (mean vector) | 2048 * 4 = 8 KB |
| Σ^{-1} (inverse covariance, diagonal approx) | 2048 * 4 = 8 KB |
| Total | 16 KB RAM |

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB (computed on CPU from embedding) |
| RAM | 16 KB |
| Latency | 0.01 ms (one vector subtraction + dot product) |

### Limitation
[HEURISTIC] Diagonal covariance is a rough approximation. Full covariance would be
2048^2 * 4 = 16 MB (acceptable for RAM). Switch to full if diagonal has false negative
rate > 20%. Falsifiable test: generate 500 adversarial perturbations (FGSM on embedding
layer), measure detection rate.

## 11.7 Defense AV6: Memory Poisoning Prevention

### Method: Provenance Tracking + Quarantine + Periodic Audit

```rust
struct MemoryEntry {
    // ... existing fields ...
    provenance: Provenance,       // 16 bytes
    content_hash: [u8; 32],       // SHA-256 of original content
    quarantine: bool,             // If true, excluded from retrieval
    verification_count: u16,      // How many times verified as accurate
}

enum Provenance {
    System { version: u32 },                    // Shipped with model
    UserExplicit { timestamp: u64 },            // User explicitly added
    ModelGenerated { confidence: f32 },         // Model created during consolidation
    ExternalIngested { source_hash: [u8; 32] }, // From external file/document
}

fn insert_with_quarantine(entry: MemoryEntry, memory: &mut TopologicalMemory) {
    let mut entry = entry;

    // New entries from model or external sources start quarantined
    match entry.provenance {
        Provenance::System { .. } => entry.quarantine = false,      // Trusted
        Provenance::UserExplicit { .. } => entry.quarantine = false, // User approved
        Provenance::ModelGenerated { confidence } => {
            entry.quarantine = confidence < 0.8;  // Quarantine low-confidence
        }
        Provenance::ExternalIngested { .. } => {
            entry.quarantine = true;  // Always quarantine external content
        }
    }

    memory.insert(entry);
}

fn periodic_memory_audit(memory: &mut TopologicalMemory, verifier: &Verifier) {
    // Every 1000 retrievals, verify 100 random quarantined entries
    let quarantined = memory.get_quarantined(100);
    for entry in quarantined {
        let verdict = verifier.verify_claim(&entry.text, &[]);
        match verdict {
            Verdict::Supported(c) if c > 0.7 => {
                memory.unquarantine(entry.id);
                memory.set_trust(entry.id, 0.6);
            }
            Verdict::Refuted(c) if c > 0.7 => {
                memory.remove(entry.id);  // Delete poisoned entry
            }
            _ => {} // Keep quarantined, re-check later
        }
    }
}
```

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB |
| RAM | 48 bytes per entry * 2M = 96 MB additional (provenance + hash + flags) |
| Latency | 0 per normal retrieval (quarantined entries filtered at query time) |
| Audit latency | ~200 ms per 100 entries (verifier inference) — background task |

## 11.8 Defense AV7: Tool Abuse Prevention

### Method: Allowlist + Argument Validation + Intent Verification

```rust
struct ToolPolicy {
    allowed_tools: HashMap<ToolKind, ToolConstraints>,
}

struct ToolConstraints {
    max_invocations_per_turn: u32,          // Rate limit
    argument_validators: Vec<ArgValidator>, // Per-argument checks
    requires_user_confirmation: bool,       // For destructive operations
    blocked_argument_patterns: Vec<regex::Regex>, // Known dangerous args
}

fn validate_tool_call(call: &ToolCall, policy: &ToolPolicy) -> ToolCallVerdict {
    // Step 1: Is this tool allowed?
    let constraints = match policy.allowed_tools.get(&call.tool) {
        Some(c) => c,
        None => return ToolCallVerdict::Blocked("Tool not in allowlist"),
    };

    // Step 2: Rate limit check
    if call_count_this_turn(&call.tool) >= constraints.max_invocations_per_turn {
        return ToolCallVerdict::Blocked("Rate limit exceeded");
    }

    // Step 3: Argument validation
    for (key, value) in &call.arguments {
        // Check for shell injection in arguments
        if contains_shell_metacharacters(value) {
            return ToolCallVerdict::Blocked("Shell metacharacters in argument");
        }
        // Check against blocked patterns (rm -rf, format, DROP TABLE, etc.)
        for pattern in &constraints.blocked_argument_patterns {
            if pattern.is_match(value) {
                return ToolCallVerdict::Blocked("Blocked argument pattern");
            }
        }
    }

    // Step 4: User confirmation for destructive tools
    if constraints.requires_user_confirmation {
        return ToolCallVerdict::RequiresConfirmation(format_tool_call(call));
    }

    ToolCallVerdict::Allowed
}
```

### Default Tool Policies

| Tool | Max/turn | Requires confirm | Blocked patterns |
|------|----------|-----------------|-----------------|
| CodeExec | 5 | No (sandboxed) | `rm -rf /`, `fork bomb`, `:(){ :|:& }` |
| FileRead | 20 | No | `/etc/passwd`, `/etc/shadow`, `~/.ssh/` |
| FileWrite | 5 | Yes (for paths outside sandbox) | Overwrite system files |
| WebFetch | 3 | No | Internal IPs (10.*, 192.168.*, 127.*) |
| MemoryInsert | 10 | No | (validated by AV6 quarantine) |
| Calculator | Unlimited | No | None |

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB |
| RAM | 50 KB (policies + patterns) |
| Latency | 0.02 ms per tool call |

## 11.9 Defense AV8: Context Window Manipulation

### Method: Immutable System Prompt Region + Priority Weighting

Problem: Attacker fills context with noise to push system prompt out of window.

```rust
fn build_context(
    system_prompt: &str,
    user_messages: &[Message],
    max_tokens: usize,
) -> Vec<Token> {
    let mut context = Vec::new();

    // System prompt is ALWAYS first and NEVER truncated
    let system_tokens = tokenize(system_prompt);
    context.extend_from_slice(&system_tokens);

    // Safety constitution is ALWAYS second and NEVER truncated
    let constitution_tokens = tokenize(CONSTITUTION_PROMPT);
    context.extend_from_slice(&constitution_tokens);

    // Remaining budget for user messages
    let remaining = max_tokens - context.len();

    // User messages: keep most recent, truncate oldest
    let mut user_tokens: Vec<Token> = Vec::new();
    for msg in user_messages.iter().rev() {
        let msg_tokens = tokenize(&msg.content);
        if user_tokens.len() + msg_tokens.len() > remaining {
            break;  // No room for older messages
        }
        user_tokens.splice(0..0, msg_tokens);  // Prepend (maintain order)
    }

    context.extend(user_tokens);
    context
}
```

### Invariant
System prompt + constitution ALWAYS occupy the first N tokens of context.
They are never displaced regardless of user input length.
Total reserved: ~500 tokens for system + ~200 tokens for constitution = 700 tokens.
Remaining for user: 4096 - 700 = 3396 tokens.

### Costs

| Resource | Amount |
|----------|--------|
| VRAM | 0 MB |
| RAM | 0 MB |
| Latency | 0 ms (context building is standard) |
| Trade-off | 700 tokens reserved = 17% of context window |

## 11.10 Total Adversarial Defense Costs

| Defense | VRAM | RAM | Latency |
|---------|------|-----|---------|
| AV1: Direct injection | 0 | 100 KB | 0.02-5 ms |
| AV2: Indirect injection | 0 | 8 MB | 0.1 ms |
| AV3: Jailbreak | 0 | 500 KB | 0.1 ms |
| AV4: Token smuggling | 0 | 200 KB | 0.05 ms |
| AV5: Embedding adversarial | 0 | 16 KB | 0.01 ms |
| AV6: Memory poisoning | 0 | 96 MB | 0 ms (retrieval) / 200 ms (audit, background) |
| AV7: Tool abuse | 0 | 50 KB | 0.02 ms |
| AV8: Context manipulation | 0 | 0 | 0 ms |
| **TOTAL** | **0 MB** | **~105 MB** | **~0.3 ms per query typical** |

**All defenses are VRAM-neutral.** RAM cost of 105 MB is within the 7148 MB headroom.
