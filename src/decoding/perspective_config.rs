//! Perspective configurations — controlled perturbations for MPD.
//!
//! Each perspective modifies the forward pass in a specific way:
//! 1. **Base**: Standard forward, temperature=1.0
//! 2. **Jitter**: Additive Gaussian noise to hidden states (σ=0.01)
//! 3. **Antipodal**: Temperature inversion (T=0.5 → sharper distribution)
//! 4. **Random**: Dropout mask on hidden units (p=0.05)

use ndarray::Array1;

/// The four canonical perspectives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PerspectiveKind {
    Base,
    Jitter,
    Antipodal,
    Random,
}

impl PerspectiveKind {
    /// All four perspectives in standard order.
    pub fn all() -> [PerspectiveKind; 4] {
        [
            PerspectiveKind::Base,
            PerspectiveKind::Jitter,
            PerspectiveKind::Antipodal,
            PerspectiveKind::Random,
        ]
    }

    pub fn index(&self) -> usize {
        match self {
            PerspectiveKind::Base => 0,
            PerspectiveKind::Jitter => 1,
            PerspectiveKind::Antipodal => 2,
            PerspectiveKind::Random => 3,
        }
    }
}

/// Configuration parameters for a single perspective.
#[derive(Clone, Debug)]
pub struct PerspectiveConfig {
    pub kind: PerspectiveKind,

    /// Sampling temperature (applied to logits).
    pub temperature: f32,

    /// Noise standard deviation (Jitter perspective).
    pub noise_sigma: f32,

    /// Dropout probability (Random perspective).
    pub dropout_p: f32,

    /// Whether this perspective is enabled.
    pub enabled: bool,
}

impl PerspectiveConfig {
    pub fn base() -> Self {
        Self {
            kind: PerspectiveKind::Base,
            temperature: 1.0,
            noise_sigma: 0.0,
            dropout_p: 0.0,
            enabled: true,
        }
    }

    pub fn jitter() -> Self {
        Self {
            kind: PerspectiveKind::Jitter,
            temperature: 1.0,
            noise_sigma: 0.01,
            dropout_p: 0.0,
            enabled: true,
        }
    }

    pub fn antipodal() -> Self {
        Self {
            kind: PerspectiveKind::Antipodal,
            temperature: 0.5,
            noise_sigma: 0.0,
            dropout_p: 0.0,
            enabled: true,
        }
    }

    pub fn random() -> Self {
        Self {
            kind: PerspectiveKind::Random,
            temperature: 1.0,
            noise_sigma: 0.0,
            dropout_p: 0.05,
            enabled: true,
        }
    }

    /// Default quad — all four perspectives.
    pub fn default_quad() -> [PerspectiveConfig; 4] {
        [
            Self::base(),
            Self::jitter(),
            Self::antipodal(),
            Self::random(),
        ]
    }
}

/// Apply perspective-specific perturbation to a logits vector.
pub fn apply_perspective(
    logits: &Array1<f32>,
    config: &PerspectiveConfig,
    rng_seed: u64,
) -> Array1<f32> {
    let mut result = logits.clone();

    // Apply temperature scaling
    if config.temperature != 1.0 && config.temperature > 0.0 {
        result.mapv_inplace(|v| v / config.temperature);
    }

    // Apply noise (Jitter)
    if config.noise_sigma > 0.0 {
        let noise = gaussian_noise(logits.len(), config.noise_sigma, rng_seed);
        result = &result + &noise;
    }

    // Apply dropout (Random)
    if config.dropout_p > 0.0 {
        let mask = dropout_mask(logits.len(), config.dropout_p, rng_seed);
        result = &result * &mask;
        // Scale by 1/(1-p) to maintain expected value
        let scale = 1.0 / (1.0 - config.dropout_p);
        result.mapv_inplace(|v| v * scale);
    }

    result
}

/// Generate Gaussian noise using Box-Muller transform.
fn gaussian_noise(n: usize, sigma: f32, seed: u64) -> Array1<f32> {
    let mut state = seed ^ 0xDEADBEEF;
    let mut values = Vec::with_capacity(n);

    for _ in 0..((n + 1) / 2) {
        // Generate two uniform random numbers
        state = xorshift64(state);
        let u1 = (state as f32) / (u64::MAX as f32);
        state = xorshift64(state);
        let u2 = (state as f32) / (u64::MAX as f32);

        let u1_clamped = u1.max(1e-10);
        let r = (-2.0 * u1_clamped.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;

        values.push(r * theta.cos() * sigma);
        values.push(r * theta.sin() * sigma);
    }

    values.truncate(n);
    Array1::from_vec(values)
}

/// Generate a dropout mask (0 or 1 per element).
fn dropout_mask(n: usize, p: f32, seed: u64) -> Array1<f32> {
    let mut state = seed ^ 0xCAFEBABE;
    let values: Vec<f32> = (0..n)
        .map(|_| {
            state = xorshift64(state);
            let u = (state as f32) / (u64::MAX as f32);
            if u < p { 0.0 } else { 1.0 }
        })
        .collect();
    Array1::from_vec(values)
}

/// Simple xorshift64 PRNG.
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
    fn test_perspective_kinds() {
        let all = PerspectiveKind::all();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0], PerspectiveKind::Base);
        assert_eq!(all[3], PerspectiveKind::Random);
    }

    #[test]
    fn test_base_passthrough() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let config = PerspectiveConfig::base();
        let result = apply_perspective(&logits, &config, 42);
        for i in 0..3 {
            assert!((result[i] - logits[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_antipodal_sharpens() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let config = PerspectiveConfig::antipodal();
        let result = apply_perspective(&logits, &config, 42);
        // T=0.5 → logits doubled
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);
        assert!((result[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_jitter_adds_noise() {
        let logits = Array1::from_vec(vec![0.0; 100]);
        let config = PerspectiveConfig::jitter();
        let result = apply_perspective(&logits, &config, 42);
        // Should not be all zeros anymore
        let sum: f32 = result.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0);
        // But noise should be small (σ=0.01)
        let max = result.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max < 0.5, "max noise = {}", max);
    }

    #[test]
    fn test_dropout_zeroes_some() {
        let logits = Array1::from_vec(vec![1.0; 1000]);
        let config = PerspectiveConfig::random();
        let result = apply_perspective(&logits, &config, 42);
        let zeros = result.iter().filter(|&&v| v == 0.0).count();
        // With p=0.05, expect ~50 zeros out of 1000
        assert!(zeros > 10 && zeros < 200, "zeros = {}", zeros);
    }

    #[test]
    fn test_default_quad() {
        let quad = PerspectiveConfig::default_quad();
        assert_eq!(quad.len(), 4);
        assert!(quad.iter().all(|c| c.enabled));
    }
}
