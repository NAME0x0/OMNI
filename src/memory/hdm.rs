//! Core HDM hypervector type and operations.
//!
//! Each hypervector is a 10,000-bit binary vector stored as a packed
//! byte array (1,250 bytes). Operations are bit-parallel and cache-friendly.

use std::fmt;

use crate::config::HDM_DIM;

/// Number of bytes needed to store HDM_DIM bits.
pub const HDM_BYTES: usize = (HDM_DIM + 7) / 8; // 1250

/// A single holographic hypervector — 10,000 binary dimensions.
#[derive(Clone)]
pub struct HyperVector {
    /// Packed bits: bit i is at (data[i/8] >> (i%8)) & 1.
    pub data: Vec<u8>,
}

impl HyperVector {
    /// Create a zero vector.
    pub fn zeros() -> Self {
        Self {
            data: vec![0u8; HDM_BYTES],
        }
    }

    /// Create a random hypervector (each bit iid Bernoulli(0.5)).
    pub fn random(seed: u64) -> Self {
        let mut data = vec![0u8; HDM_BYTES];
        // Simple xorshift64 PRNG for reproducibility
        let mut state = seed ^ 0x5DEECE66D;
        for byte in data.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = (state & 0xFF) as u8;
        }
        Self { data }
    }

    /// Get bit at position i.
    pub fn get_bit(&self, i: usize) -> bool {
        assert!(i < HDM_DIM);
        (self.data[i / 8] >> (i % 8)) & 1 == 1
    }

    /// Set bit at position i.
    pub fn set_bit(&mut self, i: usize, val: bool) {
        assert!(i < HDM_DIM);
        if val {
            self.data[i / 8] |= 1 << (i % 8);
        } else {
            self.data[i / 8] &= !(1 << (i % 8));
        }
    }

    /// XOR with another vector (binding / unbinding).
    pub fn xor(&self, other: &HyperVector) -> HyperVector {
        let data: Vec<u8> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a ^ b)
            .collect();
        HyperVector { data }
    }

    /// In-place XOR.
    pub fn xor_inplace(&mut self, other: &HyperVector) {
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a ^= b;
        }
    }

    /// NOT (flip all bits) — used for complement / negation.
    pub fn not(&self) -> HyperVector {
        let data: Vec<u8> = self.data.iter().map(|&b| !b).collect();
        HyperVector { data }
    }

    /// AND with another vector.
    pub fn and(&self, other: &HyperVector) -> HyperVector {
        let data: Vec<u8> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a & b)
            .collect();
        HyperVector { data }
    }

    /// OR with another vector.
    pub fn or(&self, other: &HyperVector) -> HyperVector {
        let data: Vec<u8> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a | b)
            .collect();
        HyperVector { data }
    }

    /// Hamming distance to another vector.
    pub fn hamming_distance(&self, other: &HyperVector) -> u32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| (a ^ b).count_ones())
            .sum()
    }

    /// Cosine-like similarity in Hamming space: 1 - 2·d(x,y)/D.
    /// Range: [-1, 1]. 1 = identical, 0 = orthogonal, -1 = complement.
    pub fn similarity(&self, other: &HyperVector) -> f64 {
        let d = self.hamming_distance(other) as f64;
        1.0 - 2.0 * d / HDM_DIM as f64
    }

    /// Population count (number of 1-bits).
    pub fn popcount(&self) -> u32 {
        self.data.iter().map(|&b| b.count_ones()).sum()
    }

    /// Density: fraction of bits that are 1.
    pub fn density(&self) -> f64 {
        self.popcount() as f64 / HDM_DIM as f64
    }

    /// Permute (circular shift) by `amount` positions.
    /// Used for sequence encoding: role(i) = permute(base, i).
    pub fn permute(&self, amount: usize) -> HyperVector {
        let mut result = HyperVector::zeros();
        for i in 0..HDM_DIM {
            let src = (i + HDM_DIM - (amount % HDM_DIM)) % HDM_DIM;
            if self.get_bit(src) {
                result.set_bit(i, true);
            }
        }
        result
    }

    /// Memory footprint in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Create from raw bytes (must be exactly HDM_BYTES).
    pub fn from_bytes(data: Vec<u8>) -> Self {
        assert_eq!(data.len(), HDM_BYTES);
        Self { data }
    }

    /// Export raw bytes.
    pub fn to_bytes(&self) -> &[u8] {
        &self.data
    }
}

impl fmt::Debug for HyperVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HyperVector(dim={}, popcount={}, density={:.3})",
            HDM_DIM,
            self.popcount(),
            self.density()
        )
    }
}

impl PartialEq for HyperVector {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

/// Majority vote over multiple vectors (bundling / superposition).
/// For each bit position, the output bit is 1 if more than half
/// of the input vectors have a 1 at that position.
/// Ties (even count) are broken randomly (bit from first vector).
pub fn majority_vote(vectors: &[&HyperVector]) -> HyperVector {
    let n = vectors.len();
    assert!(n > 0, "majority_vote requires at least one vector");

    let threshold = n / 2;
    let mut result = HyperVector::zeros();

    for bit_idx in 0..HDM_DIM {
        let count: usize = vectors.iter().filter(|v| v.get_bit(bit_idx)).count();

        if count > threshold || (count == threshold && n % 2 == 0 && vectors[0].get_bit(bit_idx)) {
            result.set_bit(bit_idx, true);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let v = HyperVector::zeros();
        assert_eq!(v.popcount(), 0);
        assert_eq!(v.density(), 0.0);
    }

    #[test]
    fn test_random_density() {
        let v = HyperVector::random(42);
        let d = v.density();
        // Should be approximately 0.5 for a random vector
        assert!(d > 0.4 && d < 0.6, "density = {}", d);
    }

    #[test]
    fn test_xor_self_is_zero() {
        let v = HyperVector::random(123);
        let z = v.xor(&v);
        assert_eq!(z.popcount(), 0);
    }

    #[test]
    fn test_xor_inverse() {
        let a = HyperVector::random(1);
        let b = HyperVector::random(2);
        let bound = a.xor(&b);
        let recovered = bound.xor(&b);
        assert_eq!(a, recovered);
    }

    #[test]
    fn test_similarity_identical() {
        let v = HyperVector::random(99);
        let sim = v.similarity(&v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_complement() {
        let v = HyperVector::random(99);
        let c = v.not();
        let sim = v.similarity(&c);
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_random_near_zero() {
        let a = HyperVector::random(1);
        let b = HyperVector::random(2);
        let sim = a.similarity(&b);
        // Two independent random vectors should be nearly orthogonal
        assert!(sim.abs() < 0.1, "sim = {}", sim);
    }

    #[test]
    fn test_hamming_distance_symmetry() {
        let a = HyperVector::random(10);
        let b = HyperVector::random(20);
        assert_eq!(a.hamming_distance(&b), b.hamming_distance(&a));
    }

    #[test]
    fn test_permute_preserves_popcount() {
        let v = HyperVector::random(42);
        let p = v.permute(7);
        assert_eq!(v.popcount(), p.popcount());
    }

    #[test]
    fn test_majority_vote_single() {
        let v = HyperVector::random(42);
        let result = majority_vote(&[&v]);
        assert_eq!(result, v);
    }

    #[test]
    fn test_majority_vote_three() {
        let a = HyperVector::random(1);
        let b = HyperVector::random(2);

        // a, a, b → should be close to a
        let result = majority_vote(&[&a, &a, &b]);
        let sim_a = result.similarity(&a);
        let sim_b = result.similarity(&b);
        assert!(sim_a > sim_b, "sim_a={}, sim_b={}", sim_a, sim_b);
    }

    #[test]
    fn test_size_bytes() {
        let v = HyperVector::zeros();
        assert_eq!(v.size_bytes(), HDM_BYTES);
        assert_eq!(HDM_BYTES, 1250);
    }

    #[test]
    fn test_set_get_bit() {
        let mut v = HyperVector::zeros();
        v.set_bit(0, true);
        v.set_bit(100, true);
        v.set_bit(9999, true);
        assert!(v.get_bit(0));
        assert!(v.get_bit(100));
        assert!(v.get_bit(9999));
        assert!(!v.get_bit(1));
        assert_eq!(v.popcount(), 3);
    }
}
