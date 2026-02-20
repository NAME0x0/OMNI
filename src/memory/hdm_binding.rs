//! HDM Binding — higher-level operations for associative memory.
//!
//! Implements structured memory traces:
//! - **Facts**: bind(subject_role, entity) ⊕ bind(predicate_role, relation) ⊕ bind(object_role, entity)
//! - **Episodes**: sequence of bound token-position pairs, superposed
//! - **Queries**: unbind a role to retrieve the filler

use crate::memory::hdm::{majority_vote, HyperVector};

/// A single memory trace (a composite hypervector).
#[derive(Clone, Debug)]
pub struct MemoryTrace {
    /// The composite hypervector (majority vote of bindings).
    pub vector: HyperVector,

    /// Individual bound vectors (role ⊕ filler) for proper bundling.
    bindings: Vec<HyperVector>,

    /// Number of bindings superposed into this trace.
    pub binding_count: usize,

    /// Human-readable label (for debugging).
    pub label: String,

    /// Timestamp (token index when created).
    pub timestamp: u64,
}

impl MemoryTrace {
    /// Create an empty trace.
    pub fn empty(label: &str, timestamp: u64) -> Self {
        Self {
            vector: HyperVector::zeros(),
            bindings: Vec::new(),
            binding_count: 0,
            label: label.to_string(),
            timestamp,
        }
    }

    /// Create a trace directly from a pre-computed vector (for testing/external use).
    pub fn from_vector(vector: HyperVector, label: &str, timestamp: u64) -> Self {
        Self {
            vector: vector.clone(),
            bindings: vec![vector],
            binding_count: 1,
            label: label.to_string(),
            timestamp,
        }
    }

    /// Create a trace from a single binding.
    pub fn from_binding(
        role: &HyperVector,
        filler: &HyperVector,
        label: &str,
        timestamp: u64,
    ) -> Self {
        let bound = role.xor(filler);
        Self {
            vector: bound.clone(),
            bindings: vec![bound],
            binding_count: 1,
            label: label.to_string(),
            timestamp,
        }
    }

    /// Recompute composite vector via majority vote of all bindings.
    fn recompute_vector(&mut self) {
        if self.bindings.is_empty() {
            self.vector = HyperVector::zeros();
        } else if self.bindings.len() == 1 {
            self.vector = self.bindings[0].clone();
        } else {
            let refs: Vec<&HyperVector> = self.bindings.iter().collect();
            self.vector = majority_vote(&refs);
        }
    }

    /// Add a role-filler binding to this trace (majority-vote bundling).
    pub fn add_binding(&mut self, role: &HyperVector, filler: &HyperVector) {
        let bound = role.xor(filler);
        self.bindings.push(bound);
        self.binding_count += 1;
        self.recompute_vector();
    }

    /// Query: unbind a role to retrieve the approximate filler.
    /// Since XOR is its own inverse: query(role) = trace ⊕ role.
    pub fn query(&self, role: &HyperVector) -> HyperVector {
        self.vector.xor(role)
    }

    /// Similarity of this trace to a probe vector.
    pub fn similarity(&self, probe: &HyperVector) -> f64 {
        self.vector.similarity(probe)
    }
}

/// Build a structured fact: S-P-O triple.
pub fn encode_fact(
    subject_role: &HyperVector,
    subject: &HyperVector,
    predicate_role: &HyperVector,
    predicate: &HyperVector,
    object_role: &HyperVector,
    object: &HyperVector,
    label: &str,
    timestamp: u64,
) -> MemoryTrace {
    let mut trace = MemoryTrace::from_binding(subject_role, subject, label, timestamp);
    trace.add_binding(predicate_role, predicate);
    trace.add_binding(object_role, object);
    trace
}

/// Encode a sequence as superposition of position-bound tokens.
pub fn encode_sequence(
    tokens: &[&HyperVector],
    position_base: &HyperVector,
    label: &str,
    timestamp: u64,
) -> MemoryTrace {
    let mut trace = MemoryTrace::empty(label, timestamp);

    for (i, token) in tokens.iter().enumerate() {
        let pos_vec = position_base.permute(i);
        trace.add_binding(&pos_vec, token);
    }

    trace
}

/// Merge multiple traces via majority vote (robust superposition).
pub fn merge_traces(traces: &[&MemoryTrace]) -> HyperVector {
    let vecs: Vec<&HyperVector> = traces.iter().map(|t| &t.vector).collect();
    majority_vote(&vecs)
}

/// Capacity estimation: how many bindings can a single trace hold
/// before retrieval similarity drops below threshold?
/// Theoretical limit ≈ D / (4 · ln(D)) for D-dimensional binary vectors.
pub fn estimated_capacity(dim: usize, min_similarity: f64) -> usize {
    // Each additional binding adds noise proportional to 1/√D.
    // After k bindings, expected similarity to a query ≈ 1/k.
    // So max k ≈ 1 / min_similarity.
    let theoretical = (dim as f64).sqrt() * min_similarity;
    theoretical as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::hdm::HyperVector;

    fn make_random_vecs(n: usize, base_seed: u64) -> Vec<HyperVector> {
        (0..n)
            .map(|i| HyperVector::random(base_seed + i as u64))
            .collect()
    }

    #[test]
    fn test_single_binding_recovery() {
        let role = HyperVector::random(100);
        let filler = HyperVector::random(200);

        let trace = MemoryTrace::from_binding(&role, &filler, "test", 0);
        let recovered = trace.query(&role);

        let sim = recovered.similarity(&filler);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "single binding should recover exactly: sim={}",
            sim
        );
    }

    #[test]
    fn test_two_binding_recovery() {
        let role_a = HyperVector::random(1);
        let filler_a = HyperVector::random(2);
        let role_b = HyperVector::random(3);
        let filler_b = HyperVector::random(4);

        let mut trace = MemoryTrace::from_binding(&role_a, &filler_a, "test", 0);
        trace.add_binding(&role_b, &filler_b);

        // Query role_a → should be more similar to filler_a than filler_b
        let recovered_a = trace.query(&role_a);
        let sim_a = recovered_a.similarity(&filler_a);
        let sim_b = recovered_a.similarity(&filler_b);

        assert!(
            sim_a > sim_b,
            "should recover filler_a: sim_a={}, sim_b={}",
            sim_a,
            sim_b
        );
        // With 10,000 dims and 2 bindings, similarity should be high
        assert!(sim_a > 0.3, "sim_a should be substantial: {}", sim_a);
    }

    #[test]
    fn test_fact_encoding() {
        let subj_role = HyperVector::random(10);
        let pred_role = HyperVector::random(20);
        let obj_role = HyperVector::random(30);

        let cat = HyperVector::random(100);
        let sits_on = HyperVector::random(200);
        let mat = HyperVector::random(300);

        let fact = encode_fact(
            &subj_role, &cat,
            &pred_role, &sits_on,
            &obj_role, &mat,
            "cat sits on mat",
            0,
        );

        assert_eq!(fact.binding_count, 3);

        // Query subject
        let recovered_subj = fact.query(&subj_role);
        let sim = recovered_subj.similarity(&cat);
        assert!(sim > 0.2, "subject recovery: sim={}", sim);
    }

    #[test]
    fn test_sequence_encoding() {
        let tokens = make_random_vecs(5, 1000);
        let pos_base = HyperVector::random(9999);

        let refs: Vec<&HyperVector> = tokens.iter().collect();
        let trace = encode_sequence(&refs, &pos_base, "seq", 0);

        assert_eq!(trace.binding_count, 5);
    }

    #[test]
    fn test_merge_traces() {
        let a = MemoryTrace::from_binding(
            &HyperVector::random(1),
            &HyperVector::random(2),
            "a",
            0,
        );
        let b = MemoryTrace::from_binding(
            &HyperVector::random(3),
            &HyperVector::random(4),
            "b",
            1,
        );

        let merged = merge_traces(&[&a, &b]);
        // Merged should have some structure
        assert!(merged.popcount() > 0);
    }

    #[test]
    fn test_estimated_capacity() {
        let cap = estimated_capacity(10_000, 0.1);
        // Should be reasonable: sqrt(10000) * 0.1 = 10
        assert_eq!(cap, 10);
    }
}
