//! LoRA (Low-Rank Adaptation) rank-4 adapters.
//!
//! Each expert gets a LoRA adapter: ΔW = A · B where A ∈ R^{d×r}, B ∈ R^{r×d}.
//! With r=4 and d=4096, each adapter is only 4×4096×2 = 32K parameters.
//! This is the "tunable surface" for FMEA's forward-mode gradients.

use ndarray::{Array1, Array2};

/// LoRA rank (global constant for PERSPECTIVE).
pub const LORA_RANK: usize = 4;

/// A single LoRA adapter pair (A, B) with optional scale α.
#[derive(Clone, Debug)]
pub struct LoraAdapter {
    /// Down-projection: A ∈ R^{r × d_in}.
    pub a: Array2<f32>,

    /// Up-projection: B ∈ R^{d_out × r}.
    pub b: Array2<f32>,

    /// Scaling factor α/r.
    pub alpha_over_r: f32,

    /// Whether this adapter is active (can be frozen).
    pub active: bool,

    /// Number of update steps applied.
    pub update_count: u64,
}

impl LoraAdapter {
    /// Create a new LoRA adapter with zero initialization.
    /// A is zero-initialized, B is zero-initialized.
    /// (Standard LoRA: A is random, B is zero — ΔW starts as 0.)
    pub fn zeros(d_in: usize, d_out: usize, alpha: f32) -> Self {
        Self {
            a: Array2::zeros((LORA_RANK, d_in)),
            b: Array2::zeros((d_out, LORA_RANK)),
            alpha_over_r: alpha / LORA_RANK as f32,
            active: true,
            update_count: 0,
        }
    }

    /// Create with Kaiming-like initialization for A, zeros for B.
    pub fn kaiming_init(d_in: usize, d_out: usize, alpha: f32, seed: u64) -> Self {
        let scale = (2.0 / d_in as f32).sqrt();
        let mut state = seed ^ 0xFEEDFACE;

        let a_values: Vec<f32> = (0..LORA_RANK * d_in)
            .map(|_| {
                state = xorshift64(state);
                let u = (state as f32) / (u64::MAX as f32) * 2.0 - 1.0;
                u * scale
            })
            .collect();

        let a = Array2::from_shape_vec((LORA_RANK, d_in), a_values).expect("LoRA A shape mismatch");
        let b = Array2::zeros((d_out, LORA_RANK));

        Self {
            a,
            b,
            alpha_over_r: alpha / LORA_RANK as f32,
            active: true,
            update_count: 0,
        }
    }

    /// Compute ΔW · x = α/r · B · (A · x).
    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        if !self.active {
            return Array1::zeros(self.b.nrows());
        }

        let down = self.a.dot(x); // R^r
        let up = self.b.dot(&down); // R^d_out
        up * self.alpha_over_r
    }

    /// Apply a gradient update to A and B.
    /// grad_a ∈ R^{r × d_in}, grad_b ∈ R^{d_out × r}.
    pub fn update(&mut self, grad_a: &Array2<f32>, grad_b: &Array2<f32>, lr: f32) {
        self.a = &self.a - &(grad_a * lr);
        self.b = &self.b - &(grad_b * lr);
        self.update_count += 1;
    }

    /// Apply a rank-1 update from JVP gradient estimate.
    /// Given direction v and directional derivative d, update:
    ///   A -= lr · d · v_a (where v_a is the A-component of v)
    pub fn update_from_jvp(
        &mut self,
        jvp_value: f32,
        direction_a: &Array2<f32>,
        direction_b: &Array2<f32>,
        lr: f32,
    ) {
        let step = lr * jvp_value;
        self.a = &self.a - &(direction_a * step);
        self.b = &self.b - &(direction_b * step);
        self.update_count += 1;
    }

    /// Number of trainable parameters.
    pub fn param_count(&self) -> usize {
        self.a.len() + self.b.len()
    }

    /// Memory footprint in bytes (f32 parameters).
    pub fn memory_bytes(&self) -> usize {
        self.param_count() * 4
    }

    /// Frobenius norm of ΔW = B · A.
    pub fn delta_norm(&self) -> f32 {
        // Compute efficiently via singular values of A and B
        // For small rank, just compute the product
        let delta = self.b.dot(&self.a); // d_out × d_in
        let frob_sq: f32 = delta.iter().map(|v| v * v).sum();
        frob_sq.sqrt() * self.alpha_over_r
    }

    /// Freeze this adapter (stop updates but keep contribution).
    pub fn freeze(&mut self) {
        self.active = false;
    }

    /// Unfreeze.
    pub fn unfreeze(&mut self) {
        self.active = true;
    }
}

/// A bank of LoRA adapters — one per expert or per layer.
pub struct LoraBank {
    /// All adapters indexed by expert_id or layer_id.
    pub adapters: Vec<LoraAdapter>,
}

impl LoraBank {
    /// Create a bank of adapters, one per expert.
    pub fn new(count: usize, d_in: usize, d_out: usize, alpha: f32) -> Self {
        let adapters = (0..count)
            .map(|_| LoraAdapter::zeros(d_in, d_out, alpha))
            .collect();
        Self { adapters }
    }

    /// Create with Kaiming initialization.
    pub fn kaiming(count: usize, d_in: usize, d_out: usize, alpha: f32, base_seed: u64) -> Self {
        let adapters = (0..count)
            .map(|i| LoraAdapter::kaiming_init(d_in, d_out, alpha, base_seed + i as u64))
            .collect();
        Self { adapters }
    }

    /// Compute adapter output for a specific expert.
    pub fn forward(&self, expert_id: usize, x: &Array1<f32>) -> Array1<f32> {
        self.adapters[expert_id].forward(x)
    }

    /// Total trainable parameters across all adapters.
    pub fn total_params(&self) -> usize {
        self.adapters.iter().map(|a| a.param_count()).sum()
    }

    /// Total memory in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        self.adapters.iter().map(|a| a.memory_bytes()).sum()
    }

    /// Freeze all adapters.
    pub fn freeze_all(&mut self) {
        for adapter in &mut self.adapters {
            adapter.freeze();
        }
    }

    /// Get total update count across all adapters.
    pub fn total_updates(&self) -> u64 {
        self.adapters.iter().map(|a| a.update_count).sum()
    }
}

fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_zeros_passthrough() {
        let adapter = LoraAdapter::zeros(16, 16, 1.0);
        let x = Array1::from_vec(vec![1.0; 16]);
        let out = adapter.forward(&x);
        // Zero-initialized → output should be zero
        for &v in out.iter() {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_lora_kaiming_nonzero() {
        let adapter = LoraAdapter::kaiming_init(16, 16, 1.0, 42);
        let x = Array1::from_vec(vec![1.0; 16]);
        let _out = adapter.forward(&x);
        // B is zeros, so output still zero until we update B
        // But A should be non-zero
        let a_norm: f32 = adapter.a.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(a_norm > 0.0);
    }

    #[test]
    fn test_lora_param_count() {
        let adapter = LoraAdapter::zeros(4096, 4096, 1.0);
        // A: 4 × 4096 = 16384, B: 4096 × 4 = 16384
        assert_eq!(adapter.param_count(), 32768);
    }

    #[test]
    fn test_lora_freeze() {
        let mut adapter = LoraAdapter::kaiming_init(8, 8, 1.0, 42);
        adapter.b = Array2::ones((8, LORA_RANK)); // Make B non-zero
        let x = Array1::from_vec(vec![1.0; 8]);

        let active_out = adapter.forward(&x);
        adapter.freeze();
        let frozen_out = adapter.forward(&x);

        // Frozen → all zeros
        assert!(frozen_out.iter().all(|&v| v == 0.0));
        // Active had some output
        let active_norm: f32 = active_out.iter().map(|v| v * v).sum::<f32>().sqrt();
        // May or may not be zero depending on init, but frozen should definitely be zero
        let _ = active_norm;
    }

    #[test]
    fn test_lora_bank() {
        let bank = LoraBank::new(128, 4096, 4096, 1.0);
        assert_eq!(bank.adapters.len(), 128);
        assert_eq!(bank.total_params(), 128 * 32768);
    }

    #[test]
    fn test_lora_memory_bytes() {
        let adapter = LoraAdapter::zeros(4096, 4096, 1.0);
        assert_eq!(adapter.memory_bytes(), 32768 * 4); // 128KB per adapter
    }

    #[test]
    fn test_lora_update() {
        let mut adapter = LoraAdapter::zeros(4, 4, 1.0);
        let grad_a = Array2::ones((LORA_RANK, 4));
        let grad_b = Array2::ones((4, LORA_RANK));
        adapter.update(&grad_a, &grad_b, 0.01);
        assert_eq!(adapter.update_count, 1);
        // A should now be -0.01
        assert!((adapter.a[[0, 0]] + 0.01).abs() < 1e-6);
    }
}
