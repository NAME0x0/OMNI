//! HDM Codebook — maps tokens/concepts to fixed hypervector representations.
//!
//! The codebook contains basis hypervectors for:
//! 1. Token embeddings (vocab_size vectors)
//! 2. Position encodings (max_seq_len vectors via permutation)
//! 3. Role markers (subject, predicate, object, etc.)
//! 4. Special markers (begin, end, separator, null)

use std::collections::HashMap;

use crate::config;
use crate::memory::hdm::HyperVector;

/// A codebook mapping symbols to hypervectors.
pub struct Codebook {
    /// Token-level basis vectors (one per vocab entry).
    pub token_vectors: Vec<HyperVector>,

    /// Role vectors for structured binding.
    pub role_vectors: HashMap<String, HyperVector>,

    /// Base position vector (permuted to create positional encodings).
    pub position_base: HyperVector,

    /// Cached position vectors for frequently-used positions.
    position_cache: HashMap<usize, HyperVector>,

    /// Seed used for generation (for reproducibility).
    pub seed: u64,
}

impl Codebook {
    /// Create a new codebook with deterministic random initialization.
    pub fn new(vocab_size: usize, seed: u64) -> Self {
        // Generate token vectors with sequential seeds
        let token_vectors: Vec<HyperVector> = (0..vocab_size)
            .map(|i| HyperVector::random(seed.wrapping_add(i as u64)))
            .collect();

        // Generate role vectors
        let roles = vec![
            "subject",
            "predicate",
            "object",
            "modifier",
            "temporal",
            "spatial",
            "causal",
            "negation",
        ];
        let role_base_seed = seed.wrapping_add(vocab_size as u64 + 1000);
        let role_vectors: HashMap<String, HyperVector> = roles
            .iter()
            .enumerate()
            .map(|(i, &name)| {
                (
                    name.to_string(),
                    HyperVector::random(role_base_seed.wrapping_add(i as u64)),
                )
            })
            .collect();

        // Position base vector
        let position_base = HyperVector::random(seed.wrapping_add(vocab_size as u64 + 2000));

        Self {
            token_vectors,
            role_vectors,
            position_base,
            position_cache: HashMap::new(),
            seed,
        }
    }

    /// Create with default PERSPECTIVE parameters.
    pub fn default_perspective() -> Self {
        Self::new(config::VOCAB_SIZE, 0xDEA5_BEC7)
    }

    /// Get the hypervector for a token ID.
    pub fn token(&self, token_id: usize) -> &HyperVector {
        &self.token_vectors[token_id]
    }

    /// Get the hypervector for a role.
    pub fn role(&self, name: &str) -> Option<&HyperVector> {
        self.role_vectors.get(name)
    }

    /// Get (or compute + cache) the position hypervector.
    /// position(i) = permute(position_base, i)
    pub fn position(&mut self, pos: usize) -> HyperVector {
        if let Some(cached) = self.position_cache.get(&pos) {
            return cached.clone();
        }
        let pv = self.position_base.permute(pos);
        self.position_cache.insert(pos, pv.clone());
        pv
    }

    /// Encode a token at a position: bind(token_vec, position_vec).
    pub fn encode_token_at_position(&mut self, token_id: usize, position: usize) -> HyperVector {
        let tok_vec = self.token_vectors[token_id].clone();
        let pos_vec = self.position(position);
        tok_vec.xor(&pos_vec)
    }

    /// Encode a role-filler pair: bind(role_vec, filler_vec).
    pub fn encode_role_filler(&self, role: &str, filler_token_id: usize) -> Option<HyperVector> {
        let role_vec = self.role_vectors.get(role)?;
        let filler_vec = &self.token_vectors[filler_token_id];
        Some(role_vec.xor(filler_vec))
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.token_vectors.len()
    }

    /// Number of defined roles.
    pub fn num_roles(&self) -> usize {
        self.role_vectors.len()
    }

    /// Total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let token_mem = self.token_vectors.len() * self.token_vectors[0].size_bytes();
        let role_mem = self.role_vectors.len() * HyperVector::zeros().size_bytes();
        let pos_mem = self.position_cache.len() * HyperVector::zeros().size_bytes();
        let base_mem = self.position_base.size_bytes();
        token_mem + role_mem + pos_mem + base_mem
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_codebook() -> Codebook {
        Codebook::new(100, 42)
    }

    #[test]
    fn test_codebook_creation() {
        let cb = small_codebook();
        assert_eq!(cb.vocab_size(), 100);
        assert_eq!(cb.num_roles(), 8);
    }

    #[test]
    fn test_token_vectors_distinct() {
        let cb = small_codebook();
        // Any two token vectors should be nearly orthogonal
        let sim = cb.token(0).similarity(cb.token(1));
        assert!(sim.abs() < 0.15, "sim = {}", sim);
    }

    #[test]
    fn test_role_vectors_exist() {
        let cb = small_codebook();
        assert!(cb.role("subject").is_some());
        assert!(cb.role("predicate").is_some());
        assert!(cb.role("nonexistent").is_none());
    }

    #[test]
    fn test_position_encoding_distinct() {
        let mut cb = small_codebook();
        let p0 = cb.position(0);
        let p1 = cb.position(1);
        let sim = p0.similarity(&p1);
        // Adjacent positions should differ
        assert!(sim.abs() < 0.3, "sim = {}", sim);
    }

    #[test]
    fn test_position_caching() {
        let mut cb = small_codebook();
        let p5a = cb.position(5);
        let p5b = cb.position(5);
        assert_eq!(p5a, p5b);
    }

    #[test]
    fn test_encode_token_at_position_recoverable() {
        let mut cb = small_codebook();
        let encoded = cb.encode_token_at_position(7, 3);
        let pos_vec = cb.position(3);
        let recovered = encoded.xor(&pos_vec);
        // recovered should be similar to token(7)
        let sim = recovered.similarity(cb.token(7));
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_role_filler_binding() {
        let cb = small_codebook();
        let bound = cb.encode_role_filler("subject", 10).unwrap();
        // Unbind with role → should recover filler
        let role_vec = cb.role("subject").unwrap();
        let recovered = bound.xor(role_vec);
        let sim = recovered.similarity(cb.token(10));
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memory_bytes_reasonable() {
        let cb = small_codebook();
        let mem = cb.memory_bytes();
        // 100 tokens × 1250 bytes + 8 roles × 1250 + base
        assert!(mem > 100 * 1250);
        assert!(mem < 200 * 1250);
    }
}
