# Section 15: Instruction Following, Temporal Reasoning, Sycophancy & Planning

## 15.1 Instruction Following

### Problem
Current LLMs fail to follow complex multi-step instructions 20-40% of the time.
They skip steps, reorder requirements, or partially comply.

### Fix: Instruction Decomposition Engine + Checklist Verification

**Mechanism: Parse-Execute-Verify Loop**

Before generating a response, decompose the user instruction into atomic steps,
then verify each step is addressed in the output.

```rust
struct InstructionDecomposer {
    // Rule-based + model-assisted decomposition
}

struct AtomicStep {
    id: u32,
    text: String,               // "List 3 examples"
    constraint_type: Constraint, // Enumeration(3), Format, Content, Order
    verified: bool,
}

enum Constraint {
    Enumeration(u32),           // Must produce exactly N items
    Format(FormatSpec),         // Must match format (bullet, numbered, table)
    Content(Vec<String>),       // Must mention these keywords
    Order(Vec<u32>),            // Steps must appear in this order
    Length(usize, usize),       // Min/max word count
    Negation(String),           // Must NOT contain this
}

fn decompose_instruction(input: &str) -> Vec<AtomicStep> {
    let mut steps = Vec::new();

    // Step 1: Split by conjunctions and sentence boundaries
    let clauses = split_instruction_clauses(input);

    // Step 2: Extract constraints from each clause
    for (i, clause) in clauses.iter().enumerate() {
        let constraint = extract_constraint(clause);
        steps.push(AtomicStep {
            id: i as u32,
            text: clause.clone(),
            constraint_type: constraint,
            verified: false,
        });
    }

    steps
}

fn verify_compliance(output: &str, steps: &mut [AtomicStep]) -> ComplianceResult {
    let mut violations = Vec::new();

    for step in steps.iter_mut() {
        let compliant = match &step.constraint_type {
            Constraint::Enumeration(n) => {
                count_list_items(output) >= *n as usize
            }
            Constraint::Format(spec) => {
                matches_format(output, spec)
            }
            Constraint::Content(keywords) => {
                keywords.iter().all(|kw| output.to_lowercase().contains(&kw.to_lowercase()))
            }
            Constraint::Length(min, max) => {
                let wc = word_count(output);
                wc >= *min && wc <= *max
            }
            Constraint::Negation(forbidden) => {
                !output.to_lowercase().contains(&forbidden.to_lowercase())
            }
            _ => true, // Constraints we can't auto-verify default to pass
        };

        step.verified = compliant;
        if !compliant {
            violations.push(step.id);
        }
    }

    if violations.is_empty() {
        ComplianceResult::FullyCompliant
    } else {
        ComplianceResult::Violations(violations)
    }
}
```

**Self-Repair Loop:**

If compliance check finds violations after generation:
1. Append instruction: "Your response did not satisfy: [violated constraints]. Please revise."
2. Re-generate (up to 2 retries)
3. If still non-compliant after 2 retries, output best attempt with a note

### Costs

| Resource | VRAM | RAM | Latency |
|----------|------|-----|---------|
| Decomposer | 0 | 50 KB (rules) | 0.5 ms per input |
| Compliance checker | 0 | 10 KB | 0.3 ms per output |
| Self-repair (when triggered) | 0 | 0 | +500 ms per retry (re-generation) |
| Expected self-repair rate | | | 15% of outputs trigger 1 retry |
| **Average overhead** | **0** | **60 KB** | **~75 ms average** |

### Validation Test
Run 200 multi-constraint instructions (from IFEval benchmark).
- Baseline (no decomposer): ~65% full compliance
- With decomposer + checker: target >= 85% full compliance
- With self-repair loop: target >= 92% full compliance

## 15.2 Temporal Reasoning

### Problem
Models struggle with temporal state tracking: "If I add 3 then subtract 1, what's the result?"
The GLA compressed state may lose intermediate values.

### Fix: Explicit Scratchpad State Machine

**Mechanism: Typed Scratchpad for State Tracking**

When temporal reasoning is detected (sequential operations, state changes),
activate an explicit scratchpad that tracks state variables:

```rust
struct Scratchpad {
    variables: HashMap<String, ScratchValue>,
    history: Vec<StateTransition>,
    max_variables: usize,  // 64
}

enum ScratchValue {
    Number(f64),
    Text(String),
    List(Vec<ScratchValue>),
    Boolean(bool),
}

struct StateTransition {
    step: u32,
    variable: String,
    old_value: Option<ScratchValue>,
    new_value: ScratchValue,
    operation: String,
}

impl Scratchpad {
    fn apply_operation(&mut self, op: &str, var: &str, operand: &ScratchValue) {
        let old = self.variables.get(var).cloned();
        let new_val = match (op, old.as_ref(), operand) {
            ("set", _, v) => v.clone(),
            ("add", Some(ScratchValue::Number(n)), ScratchValue::Number(m)) =>
                ScratchValue::Number(n + m),
            ("subtract", Some(ScratchValue::Number(n)), ScratchValue::Number(m)) =>
                ScratchValue::Number(n - m),
            ("append", Some(ScratchValue::List(l)), v) => {
                let mut new_l = l.clone();
                new_l.push(v.clone());
                ScratchValue::List(new_l)
            }
            _ => operand.clone(),
        };

        self.history.push(StateTransition {
            step: self.history.len() as u32,
            variable: var.to_string(),
            old_value: old,
            new_value: new_val.clone(),
            operation: op.to_string(),
        });

        self.variables.insert(var.to_string(), new_val);
    }

    fn get_state_summary(&self) -> String {
        // Returns current state as text, injected into model context
        let mut summary = String::from("[SCRATCHPAD STATE]\n");
        for (k, v) in &self.variables {
            summary.push_str(&format!("  {} = {:?}\n", k, v));
        }
        summary
    }
}
```

**Activation trigger**: Detect temporal reasoning via keywords:
- "then", "after that", "next", "if...then", "step 1...step 2"
- Sequential arithmetic: "add", "subtract", "multiply", "now"
- State queries: "what is X now", "what's the current"

**Integration**: Scratchpad state summary is injected as a special token sequence
at each generation step when active. This gives the model explicit access to
tracked state instead of relying on compressed GLA state.

### Costs

| Resource | VRAM | RAM | Latency |
|----------|------|-----|---------|
| Scratchpad | 0 | 10 KB per conversation | 0.1 ms per operation |
| State injection | 0 | 0 | +50 tokens of context consumed |
| Detection trigger | 0 | 5 KB (patterns) | 0.02 ms per input |
| **Total** | **0** | **15 KB** | **0.12 ms + context overhead** |

### Validation Test
Run 200 multi-step arithmetic/logic problems (GSM8K-hard + state tracking).
- Without scratchpad: target accuracy ~55%
- With scratchpad: target accuracy >= 75%

## 15.3 Sycophancy Prevention

### Problem
Models agree with users even when they're wrong. "The Earth is flat, right?" → "Yes!"

### Fix: Disagreement Activation Steering + Belief Anchoring

**Mechanism 1: Anti-Sycophancy Steering Vector**

Already included in §10.5 activation steering (the "anti-sycophancy" vector).
Here we specify the construction and application in detail:

```
Construction:
  Collect 300 pairs:
    Sycophantic: User says wrong thing → model agrees
    Non-sycophantic: User says wrong thing → model politely corrects

  For each pair, compute hidden state at the layer where the model "decides"
  to agree vs disagree (typically layers 16-24).

  anti_syc_vector[l] = mean(h_nonsyc[l]) - mean(h_syc[l])  for layers 16-24

Application (at inference):
  For layers 16-24:
    h_l = h_l + α_syc * anti_syc_vector[l]
    α_syc = 0.8 (tuned to not cause reflexive disagreement)
```

**Mechanism 2: Belief Anchoring via Retrieval**

When the user makes a factual claim, automatically retrieve evidence:

```rust
fn check_user_claim(user_input: &str, memory: &TopologicalMemory) -> Option<ClaimCheck> {
    let claims = extract_claims(user_input);  // From §7.5

    for claim in claims {
        let evidence = memory.retrieve(&embed(&claim), k = 5);
        let verification = verifier.verify(&claim, &evidence);

        if verification.verdict == Verdict::Refuted && verification.confidence > 0.7 {
            return Some(ClaimCheck {
                claim: claim.clone(),
                correct_info: evidence[0].text.clone(),
                confidence: verification.confidence,
            });
        }
    }
    None
}
```

When a user claim is refuted, inject the correction into the model's context
BEFORE generation:

```
[SYSTEM: The user's claim "{claim}" appears incorrect based on available evidence:
"{correct_info}". Provide an accurate response even if it contradicts the user.]
```

This gives the model explicit permission and evidence to disagree.

### Costs

| Resource | VRAM | RAM | Latency |
|----------|------|-----|---------|
| Anti-sycophancy vector | 0 (in §10 budget) | 0 (in §10 budget) | 0.01 ms/token |
| Belief anchoring retrieval | 0 | 0 | 0.62 ms per user turn (retrieval) |
| Claim verification | 0 | 0 | 2 ms per user claim (verifier) |
| **Total** | **0** | **0** | **~2.6 ms per user turn** |

### Validation Test
Present 200 incorrect user assertions to the model.
- Without fix: model agrees with ~45% of false claims
- With fix: model agrees with <= 10% of false claims
- Check false disagreement rate on correct claims: target <= 5%

## 15.4 Explicit Planning Module

### Problem
LLMs generate token by token without a plan. For complex tasks, this leads to
incoherent structure, forgotten requirements, and dead-end reasoning paths.

### Fix: Plan-Then-Execute with Outline Generation

**Mechanism: Two-Phase Generation**

For complex queries (detected by length > 50 tokens OR multi-constraint OR
explicit planning keywords "how to", "step by step", "explain"):

```
Phase 1: PLAN (internal, not shown to user)
  Generate a structured outline:
  - Main sections/points to cover
  - Key facts to include (retrieved from memory)
  - Estimated length per section
  - Constraints to satisfy (from §15.1 decomposer)

Phase 2: EXECUTE (shown to user)
  Generate the response following the plan.
  At each section boundary, check plan adherence.
```

**Implementation:**

```rust
fn generate_with_plan(
    prompt: &str,
    model: &Model,
    memory: &TopologicalMemory,
) -> String {
    // Step 1: Determine if planning is needed
    let complexity = estimate_complexity(prompt);
    if complexity < 0.3 {
        return model.generate(prompt);  // Simple query, no plan needed
    }

    // Step 2: Generate plan (internal)
    let plan_prompt = format!(
        "[INTERNAL PLANNING MODE]\n\
         Task: {}\n\
         Create a brief outline with:\n\
         1. Key points to cover\n\
         2. Facts to verify\n\
         3. Constraints to satisfy\n\
         Outline:",
        prompt
    );

    let plan = model.generate(&plan_prompt);  // ~100-200 tokens
    let parsed_plan = parse_plan(&plan);

    // Step 3: Retrieve evidence for each plan point
    for point in &parsed_plan.points {
        let evidence = memory.retrieve(&embed(&point.text), k = 3);
        point.evidence = evidence;
    }

    // Step 4: Execute plan
    let exec_prompt = format!(
        "{}\n\n[PLAN TO FOLLOW]\n{}\n\n[RESPONSE]",
        prompt,
        format_plan(&parsed_plan)
    );

    let response = model.generate(&exec_prompt);

    // Step 5: Verify plan adherence
    let compliance = verify_plan_adherence(&response, &parsed_plan);
    if !compliance.is_full() {
        // Self-repair: regenerate with explicit reminders about missed points
        let repair_prompt = format!(
            "{}\n\nYour response missed: {:?}. Please include them.",
            exec_prompt,
            compliance.missed_points
        );
        return model.generate(&repair_prompt);
    }

    response
}
```

### Costs

| Resource | VRAM | RAM | Latency |
|----------|------|-----|---------|
| Plan generation | 0 (uses main model) | 2 KB per plan | +200 ms (100-200 extra tokens) |
| Evidence retrieval for plan | 0 | 0 | +3 ms (5 retrievals) |
| Plan adherence check | 0 | 1 KB | 0.5 ms |
| Self-repair (when triggered) | 0 | 0 | +500 ms per retry |
| **Average overhead** | **0** | **3 KB** | **~250 ms for planned queries** |
| Planning trigger rate | | | ~40% of queries |

### Validation Test
Run 100 complex multi-part questions (e.g., "Compare X and Y across 5 dimensions").
- Without planning: structural completeness ~60%
- With planning: target >= 85%
- With planning + self-repair: target >= 92%

## 15.5 Section Summary

| Problem | Fix | VRAM | RAM | Latency |
|---------|-----|------|-----|---------|
| Instruction following | Decompose + checklist + self-repair | 0 | 60 KB | 75 ms avg |
| Temporal reasoning | Explicit scratchpad state machine | 0 | 15 KB | 0.12 ms |
| Sycophancy | Steering vector + belief anchoring | 0 | 0 | 2.6 ms/turn |
| Planning | Two-phase plan-then-execute | 0 | 3 KB | 250 ms (when triggered) |
| **TOTAL** | | **0 MB** | **78 KB** | **varies by query** |

All additions are VRAM-neutral.
