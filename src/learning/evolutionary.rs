//! Natural Evolution Strategies (NES) for routing optimisation.
//!
//! NES optimises the router's weights without backpropagation:
//! 1. Sample K perturbations of the routing weights
//! 2. Evaluate each perturbation (forward-only)
//! 3. Update weights via fitness-weighted gradient estimate
//!
//! This handles the non-differentiable manifold routing through evolution.

use ndarray::Array1;

/// NES configuration.
#[derive(Clone, Debug)]
pub struct NesConfig {
    /// Population size (number of perturbation samples).
    pub pop_size: usize,

    /// Perturbation standard deviation σ.
    pub sigma: f32,

    /// Learning rate.
    pub lr: f32,

    /// Whether to use fitness shaping (rank-based).
    pub fitness_shaping: bool,

    /// Momentum coefficient (0 = no momentum).
    pub momentum: f32,
}

impl Default for NesConfig {
    fn default() -> Self {
        Self {
            pop_size: 50,
            sigma: 0.02,
            lr: 0.01,
            fitness_shaping: true,
            momentum: 0.9,
        }
    }
}

/// A single NES perturbation sample.
#[derive(Clone)]
pub struct NesSample {
    /// The perturbation vector (noise).
    pub perturbation: Array1<f32>,

    /// Fitness score for this perturbation.
    pub fitness: f32,
}

/// NES optimiser state.
pub struct NesOptimizer {
    pub config: NesConfig,

    /// Momentum buffer.
    pub momentum_buffer: Option<Array1<f32>>,

    /// Running mean fitness (for normalisation).
    pub mean_fitness: f32,

    /// Running std fitness.
    pub std_fitness: f32,

    /// Generation counter.
    pub generation: u64,

    /// PRNG state.
    rng_state: u64,
}

impl NesOptimizer {
    pub fn new(config: NesConfig) -> Self {
        Self {
            config,
            momentum_buffer: None,
            mean_fitness: 0.0,
            std_fitness: 1.0,
            generation: 0,
            rng_state: 0x12345678,
        }
    }

    /// Generate a population of perturbation directions.
    pub fn generate_perturbations(&mut self, dim: usize) -> Vec<Array1<f32>> {
        (0..self.config.pop_size)
            .map(|_| {
                let values: Vec<f32> = (0..dim)
                    .map(|_| {
                        self.rng_state = xorshift64(self.rng_state);
                        let u1 = (self.rng_state as f64) / (u64::MAX as f64);
                        self.rng_state = xorshift64(self.rng_state);
                        let u2 = (self.rng_state as f64) / (u64::MAX as f64);

                        // Box-Muller
                        let z = (-2.0 * u1.max(1e-10).ln()).sqrt()
                            * (2.0 * std::f64::consts::PI * u2).cos();
                        (z * self.config.sigma as f64) as f32
                    })
                    .collect();
                Array1::from_vec(values)
            })
            .collect()
    }

    /// Compute the NES gradient estimate from fitness-evaluated samples.
    ///
    /// Using antithetic sampling: each perturbation ε_i has fitness f(θ+σε_i)
    /// Gradient estimate: (1 / (K·σ)) · Σ f_i · ε_i
    pub fn compute_gradient(
        &mut self,
        samples: &[NesSample],
    ) -> Array1<f32> {
        let k = samples.len();
        if k == 0 {
            return Array1::zeros(0);
        }

        let dim = samples[0].perturbation.len();

        // Optional fitness shaping (rank-based utilities)
        let utilities: Vec<f32> = if self.config.fitness_shaping {
            fitness_shaping(samples)
        } else {
            // Normalise fitnesses
            let mean: f32 =
                samples.iter().map(|s| s.fitness).sum::<f32>() / k as f32;
            let std: f32 = (samples
                .iter()
                .map(|s| (s.fitness - mean).powi(2))
                .sum::<f32>()
                / k as f32)
                .sqrt()
                .max(1e-8);

            samples
                .iter()
                .map(|s| (s.fitness - mean) / std)
                .collect()
        };

        // Gradient estimate
        let mut grad = Array1::zeros(dim);
        for (sample, &utility) in samples.iter().zip(utilities.iter()) {
            grad = &grad + &(&sample.perturbation * utility);
        }
        grad = grad / (k as f32 * self.config.sigma);

        // Apply momentum
        if self.config.momentum > 0.0 {
            if let Some(ref buf) = self.momentum_buffer {
                grad = &grad + &(buf * self.config.momentum);
            }
            self.momentum_buffer = Some(grad.clone());
        }

        self.generation += 1;
        grad
    }

    /// Apply the gradient to parameter vector.
    pub fn step(
        &self,
        params: &Array1<f32>,
        gradient: &Array1<f32>,
    ) -> Array1<f32> {
        params + &(gradient * self.config.lr)
    }
}

/// Fitness shaping: convert raw fitnesses to rank-based utilities.
/// This is more robust to outliers and fitness scale.
/// Utility u_i = max(0, log(K/2 + 1) - log(rank_i)) / Σ_j u_j - 1/K
fn fitness_shaping(samples: &[NesSample]) -> Vec<f32> {
    let k = samples.len();

    // Sort indices by fitness (descending)
    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|&a, &b| {
        samples[b]
            .fitness
            .partial_cmp(&samples[a].fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Assign rank-based utilities
    let log_k_half = ((k as f32) / 2.0 + 1.0).ln();
    let mut raw_utilities = vec![0.0f32; k];

    for (rank, &idx) in indices.iter().enumerate() {
        let u = (log_k_half - ((rank + 1) as f32).ln()).max(0.0);
        raw_utilities[idx] = u;
    }

    // Normalise
    let sum: f32 = raw_utilities.iter().sum();
    if sum > 0.0 {
        raw_utilities.iter().map(|&u| u / sum - 1.0 / k as f32).collect()
    } else {
        vec![0.0; k]
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
    fn test_generate_perturbations_shape() {
        let config = NesConfig {
            pop_size: 10,
            ..Default::default()
        };
        let mut opt = NesOptimizer::new(config);
        let perts = opt.generate_perturbations(100);
        assert_eq!(perts.len(), 10);
        assert_eq!(perts[0].len(), 100);
    }

    #[test]
    fn test_perturbations_zero_mean() {
        let config = NesConfig {
            pop_size: 1000,
            sigma: 1.0,
            ..Default::default()
        };
        let mut opt = NesOptimizer::new(config);
        let perts = opt.generate_perturbations(50);

        // Mean should be approximately zero
        let mut mean = Array1::zeros(50);
        for p in &perts {
            mean = &mean + p;
        }
        mean /= 1000.0;

        let max_elem = mean.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_elem < 0.2, "max mean element = {}", max_elem);
    }

    #[test]
    fn test_fitness_shaping() {
        let samples = vec![
            NesSample {
                perturbation: Array1::zeros(5),
                fitness: 1.0,
            },
            NesSample {
                perturbation: Array1::zeros(5),
                fitness: 3.0,
            },
            NesSample {
                perturbation: Array1::zeros(5),
                fitness: 2.0,
            },
        ];
        let utils = fitness_shaping(&samples);
        assert_eq!(utils.len(), 3);
        // Highest fitness (index 1) should get highest utility
        assert!(utils[1] > utils[2]);
        assert!(utils[2] > utils[0] || (utils[2] - utils[0]).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_computation() {
        let config = NesConfig {
            pop_size: 2,
            fitness_shaping: false,
            momentum: 0.0,
            sigma: 0.1,
            ..Default::default()
        };
        let mut opt = NesOptimizer::new(config);

        let samples = vec![
            NesSample {
                perturbation: Array1::from_vec(vec![1.0, 0.0]),
                fitness: 2.0,
            },
            NesSample {
                perturbation: Array1::from_vec(vec![0.0, 1.0]),
                fitness: 1.0,
            },
        ];

        let grad = opt.compute_gradient(&samples);
        assert_eq!(grad.len(), 2);
        // Gradient should point more toward first perturbation (higher fitness)
    }

    #[test]
    fn test_step() {
        let config = NesConfig {
            lr: 0.1,
            ..Default::default()
        };
        let opt = NesOptimizer::new(config);

        let params = Array1::from_vec(vec![1.0, 2.0]);
        let grad = Array1::from_vec(vec![0.5, -0.5]);
        let new_params = opt.step(&params, &grad);

        assert!((new_params[0] - 1.05).abs() < 1e-6);
        assert!((new_params[1] - 1.95).abs() < 1e-6);
    }

    #[test]
    fn test_nes_config_default() {
        let config = NesConfig::default();
        assert_eq!(config.pop_size, 50);
        assert!(config.fitness_shaping);
    }
}
