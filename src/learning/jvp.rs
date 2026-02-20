//! Forward-mode Jacobian-Vector Products (JVP).
//!
//! Instead of backpropagation (reverse-mode AD), we use forward-mode AD
//! to compute directional derivatives. This has key advantages:
//! - O(1) memory (no activation caching)
//! - Streaming-compatible (no backward pass)
//! - Naturally parallel (each direction is independent)
//!
//! The trade-off: O(p) cost per gradient estimate (vs O(1) for backprop),
//! but with LoRA rank-4 adapters, p is tiny (~65K parameters per layer).

use ndarray::{Array1, Array2};

/// A tangent vector for forward-mode AD.
/// Represents a perturbation direction in parameter space.
#[derive(Clone, Debug)]
pub struct TangentVector {
    /// Perturbation values (same shape as the parameter being perturbed).
    pub values: Array1<f32>,

    /// Scale factor (epsilon for finite differences).
    pub epsilon: f32,
}

impl TangentVector {
    /// Create a random unit tangent vector.
    pub fn random_unit(dim: usize, seed: u64) -> Self {
        let mut state = seed ^ 0xABCD1234;
        let values: Vec<f32> = (0..dim)
            .map(|_| {
                state = xorshift64(state);
                let u = (state as f32) / (u64::MAX as f32);
                u * 2.0 - 1.0 // [-1, 1]
            })
            .collect();

        // Normalize to unit length
        let norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        let values: Vec<f32> = values.iter().map(|&v| v / (norm + 1e-8)).collect();

        Self {
            values: Array1::from_vec(values),
            epsilon: 1e-3,
        }
    }

    /// Create a standard basis tangent (one-hot direction).
    pub fn basis(dim: usize, index: usize) -> Self {
        let mut values = vec![0.0f32; dim];
        values[index] = 1.0;
        Self {
            values: Array1::from_vec(values),
            epsilon: 1e-3,
        }
    }

    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

/// Compute JVP for a linear layer: y = Wx + b.
/// Given tangent dW (perturbation of W), the JVP is: dy = dW · x.
pub fn jvp_linear(
    x: &Array1<f32>,
    tangent_w: &Array2<f32>,
    tangent_b: Option<&Array1<f32>>,
) -> Array1<f32> {
    // dy = tangent_w · x + tangent_b
    let mut dy = tangent_w.dot(x);
    if let Some(db) = tangent_b {
        dy = &dy + db;
    }
    dy
}

/// Compute JVP for element-wise operations.
pub fn jvp_elementwise_mul(
    x: &Array1<f32>,
    dx: &Array1<f32>,
    y: &Array1<f32>,
    dy: &Array1<f32>,
) -> Array1<f32> {
    // d(x * y) = dx * y + x * dy
    &(dx * y) + &(x * dy)
}

/// Compute JVP for RMS normalization.
/// If y = x / rms(x) · scale, then dy ≈ (dx - y · (y · dx)/d) / rms · scale
pub fn jvp_rms_norm(x: &Array1<f32>, dx: &Array1<f32>, scale: &Array1<f32>) -> Array1<f32> {
    let d = x.len() as f32;
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / d).sqrt() + 1e-6;
    let y = x / rms;

    let y_dot_dx: f32 = y.iter().zip(dx.iter()).map(|(&a, &b)| a * b).sum();
    let dy = (dx - &(&y * (y_dot_dx / d))) / rms;
    &dy * scale
}

/// JVP for SiLU activation: silu(x) = x · σ(x).
/// d(silu(x)) = silu(x) + σ(x) · (1 - silu(x)) · dx
///            = (σ(x) + x · σ(x) · (1 - σ(x))) · dx
pub fn jvp_silu(x: &Array1<f32>, dx: &Array1<f32>) -> Array1<f32> {
    let result: Vec<f32> = x
        .iter()
        .zip(dx.iter())
        .map(|(&xi, &dxi)| {
            let sigma = 1.0 / (1.0 + (-xi).exp());
            let grad = sigma + xi * sigma * (1.0 - sigma);
            grad * dxi
        })
        .collect();
    Array1::from_vec(result)
}

/// JVP for softmax: d(softmax(x)) = softmax(x) · (dx - (softmax(x) · dx))
pub fn jvp_softmax(probs: &Array1<f32>, dx: &Array1<f32>) -> Array1<f32> {
    let dot: f32 = probs.iter().zip(dx.iter()).map(|(&p, &d)| p * d).sum();
    probs * &(dx - dot)
}

/// Estimate the gradient in a random direction using forward-mode JVP.
/// Returns (directional_derivative, direction).
///
/// This is the core of FMEA: instead of computing the full gradient,
/// we compute the directional derivative along a random direction,
/// then use it as a noisy gradient estimate:
///   g_hat ≈ (JVP · v) · v
pub fn estimate_gradient_direction(loss_jvp: f32, direction: &Array1<f32>) -> Array1<f32> {
    direction * loss_jvp
}

/// Multi-sample gradient estimator: average of K directional estimates.
/// More samples → lower variance but higher cost.
pub fn multi_sample_gradient(loss_jvps: &[f32], directions: &[Array1<f32>]) -> Array1<f32> {
    assert_eq!(loss_jvps.len(), directions.len());
    let k = loss_jvps.len() as f32;

    let mut grad = Array1::zeros(directions[0].len());
    for (jvp, dir) in loss_jvps.iter().zip(directions.iter()) {
        grad = &grad + &(dir * *jvp);
    }
    grad / k
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
    use ndarray::Array2;

    #[test]
    fn test_tangent_random_unit_norm() {
        let t = TangentVector::random_unit(100, 42);
        let norm: f32 = t.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "norm = {}", norm);
    }

    #[test]
    fn test_tangent_basis() {
        let t = TangentVector::basis(10, 3);
        assert_eq!(t.values[3], 1.0);
        assert_eq!(t.values[0], 0.0);
    }

    #[test]
    fn test_jvp_linear_identity() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        // Identity tangent: dW = I
        let dw = Array2::eye(3);
        let dy = jvp_linear(&x, &dw, None);
        // dy = I · x = x
        for i in 0..3 {
            assert!((dy[i] - x[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_jvp_silu_at_zero() {
        let x = Array1::from_vec(vec![0.0, 0.0]);
        let dx = Array1::from_vec(vec![1.0, 1.0]);
        let dy = jvp_silu(&x, &dx);
        // At x=0: σ(0) = 0.5, grad = 0.5 + 0 = 0.5
        assert!((dy[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_jvp_rms_norm_shape() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let dx = Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1]);
        let scale = Array1::ones(4);
        let dy = jvp_rms_norm(&x, &dx, &scale);
        assert_eq!(dy.len(), 4);
    }

    #[test]
    fn test_estimate_gradient() {
        let dir = Array1::from_vec(vec![0.6, 0.8]); // unit-ish
        let jvp_val = 2.0;
        let g = estimate_gradient_direction(jvp_val, &dir);
        assert!((g[0] - 1.2).abs() < 1e-6);
        assert!((g[1] - 1.6).abs() < 1e-6);
    }

    #[test]
    fn test_multi_sample_gradient() {
        let d1 = Array1::from_vec(vec![1.0, 0.0]);
        let d2 = Array1::from_vec(vec![0.0, 1.0]);
        let jvps = vec![2.0, 3.0];
        let g = multi_sample_gradient(&jvps, &[d1, d2]);
        // Average of [2, 0] and [0, 3] = [1, 1.5]
        assert!((g[0] - 1.0).abs() < 1e-6);
        assert!((g[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_jvp_softmax_sums_to_zero() {
        let probs = Array1::from_vec(vec![0.2, 0.3, 0.5]);
        let dx = Array1::from_vec(vec![0.1, 0.2, -0.3]);
        let dy = jvp_softmax(&probs, &dx);
        // Derivative of softmax output should sum to 0
        let sum: f32 = dy.iter().sum();
        assert!(sum.abs() < 1e-6, "sum = {}", sum);
    }
}
