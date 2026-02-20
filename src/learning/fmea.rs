//! Forward-Mode Expert Adaptation (FMEA) orchestrator.
//!
//! FMEA ties together JVP + LoRA + NES for continual learning without backprop:
//!
//! 1. JVP computes directional derivatives (cheap, O(1) memory)
//! 2. LoRA provides the small trainable surface (rank-4, ~32K params/adapter)
//! 3. NES evolves routing weights on the non-differentiable torus manifold
//!
//! The pipeline per update step:
//!   a) Route token → select expert
//!   b) Forward pass through expert with LoRA adapter
//!   c) Compute loss
//!   d) Sample K tangent vectors over LoRA params
//!   e) JVP along each tangent → K directional derivatives
//!   f) Multi-sample gradient estimate → update LoRA
//!   g) Periodically: NES step to evolve routing weights

use ndarray::{Array1, Array2};

use super::evolutionary::{NesConfig, NesOptimizer, NesSample};
use super::jvp::TangentVector;
use super::lora::{LoraBank, LORA_RANK};
use crate::config::MANIFOLD_DIM;

/// FMEA configuration.
#[derive(Clone, Debug)]
pub struct FmeaConfig {
    /// Number of JVP samples per update step.
    pub jvp_samples: usize,

    /// Learning rate for LoRA updates.
    pub lora_lr: f32,

    /// NES update frequency (every N steps).
    pub nes_update_freq: u64,

    /// NES configuration.
    pub nes_config: NesConfig,

    /// Exponential moving average decay for loss tracker.
    pub loss_ema_decay: f32,

    /// LoRA alpha scaling.
    pub lora_alpha: f32,

    /// Maximum LoRA delta norm before freezing.
    pub max_delta_norm: f32,
}

impl Default for FmeaConfig {
    fn default() -> Self {
        Self {
            jvp_samples: 8,
            lora_lr: 1e-3,
            nes_update_freq: 100,
            nes_config: NesConfig::default(),
            loss_ema_decay: 0.99,
            lora_alpha: 1.0,
            max_delta_norm: 10.0,
        }
    }
}

/// FMEA engine state.
pub struct FmeaEngine {
    pub config: FmeaConfig,

    /// LoRA adapters (one per expert).
    pub lora_bank: LoraBank,

    /// NES optimizer for routing weights.
    pub nes_optimizer: NesOptimizer,

    /// Step counter.
    pub step: u64,

    /// EMA of loss for monitoring.
    pub loss_ema: f32,

    /// Statistics.
    pub stats: FmeaStats,

    /// Internal routing parameter shadow used for scheduled NES updates.
    routing_shadow: Array1<f32>,

    /// PRNG state.
    rng_state: u64,
}

/// FMEA statistics.
#[derive(Clone, Debug, Default)]
pub struct FmeaStats {
    pub total_lora_updates: u64,
    pub total_nes_updates: u64,
    pub total_jvp_evals: u64,
    pub adapters_frozen: usize,
    pub avg_gradient_norm: f32,
}

/// Result of a single FMEA update step.
#[derive(Debug)]
pub struct FmeaStepResult {
    /// The loss value.
    pub loss: f32,

    /// The gradient norm.
    pub gradient_norm: f32,

    /// Whether a NES step was taken.
    pub nes_stepped: bool,

    /// Whether the adapter was frozen (delta too large).
    pub adapter_frozen: bool,
}

impl FmeaEngine {
    /// Create a new FMEA engine.
    pub fn new(config: FmeaConfig, n_experts: usize, d_model: usize) -> Self {
        let lora_bank = LoraBank::new(n_experts, d_model, d_model, config.lora_alpha);
        let nes_optimizer = NesOptimizer::new(config.nes_config.clone());

        Self {
            config,
            lora_bank,
            nes_optimizer,
            step: 0,
            loss_ema: 0.0,
            stats: FmeaStats::default(),
            routing_shadow: Array1::zeros(n_experts * MANIFOLD_DIM),
            rng_state: 0xF0EA_1234,
        }
    }

    /// Perform one FMEA update step for a given expert.
    ///
    /// - `expert_id`: Which expert was activated
    /// - `x`: Input hidden state
    /// - `expert_output`: Output from the base expert (without LoRA)
    /// - `target`: Target/label logits for loss computation
    /// - `loss_fn`: Function to compute scalar loss from (output, target)
    pub fn update_step<F>(
        &mut self,
        expert_id: usize,
        x: &Array1<f32>,
        expert_output: &Array1<f32>,
        target: &Array1<f32>,
        loss_fn: F,
    ) -> FmeaStepResult
    where
        F: Fn(&Array1<f32>, &Array1<f32>) -> f32,
    {
        let adapter = &self.lora_bank.adapters[expert_id];
        let d_in = adapter.a.ncols();
        let d_out = adapter.b.nrows();

        // Base output with current LoRA
        let lora_delta = adapter.forward(x);
        let output_with_lora = expert_output + &lora_delta;
        let base_loss = loss_fn(&output_with_lora, target);

        // Sample JVP directions and compute directional derivatives
        let mut jvp_values = Vec::with_capacity(self.config.jvp_samples);
        let mut dir_a_list = Vec::with_capacity(self.config.jvp_samples);
        let mut dir_b_list = Vec::with_capacity(self.config.jvp_samples);

        if self.config.jvp_samples > 0 {
            for k in 0..self.config.jvp_samples {
                self.rng_state = xorshift64(self.rng_state + k as u64);
                let seed = self.rng_state;

                // Generate random tangent for A and B
                let tangent_a = TangentVector::random_unit(LORA_RANK * d_in, seed);
                let tangent_b = TangentVector::random_unit(d_out * LORA_RANK, seed + 1);

                let dir_a =
                    match Array2::from_shape_vec((LORA_RANK, d_in), tangent_a.values.to_vec()) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                let dir_b =
                    match Array2::from_shape_vec((d_out, LORA_RANK), tangent_b.values.to_vec()) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                // Forward JVP: compute how loss changes in this direction
                // ΔW_ε = α/r · (B + ε·dir_b) · (A + ε·dir_a) · x
                // Approximate: JVP ≈ (loss(θ+εv) - loss(θ)) / ε
                let eps = tangent_a.epsilon;
                let perturbed_a = &adapter.a + &(&dir_a * eps);
                let perturbed_b = &adapter.b + &(&dir_b * eps);

                let down = perturbed_a.dot(x);
                let up = perturbed_b.dot(&down);
                let perturbed_output = expert_output + &(&up * adapter.alpha_over_r);
                let perturbed_loss = loss_fn(&perturbed_output, target);

                let jvp_val = (perturbed_loss - base_loss) / eps;
                jvp_values.push(jvp_val);
                dir_a_list.push(dir_a);
                dir_b_list.push(dir_b);
            }
        }

        self.stats.total_jvp_evals += jvp_values.len() as u64;

        // Multi-sample gradient estimate
        let (grad_a, grad_b) = multi_sample_lora_gradient(
            &jvp_values,
            &dir_a_list,
            &dir_b_list,
            (LORA_RANK, d_in),
            (d_out, LORA_RANK),
        );

        let gradient_norm = {
            let a_norm: f32 = grad_a.iter().map(|v| v * v).sum::<f32>();
            let b_norm: f32 = grad_b.iter().map(|v| v * v).sum::<f32>();
            (a_norm + b_norm).sqrt()
        };

        // Update EMA gradient norm
        let alpha = 0.01;
        self.stats.avg_gradient_norm =
            self.stats.avg_gradient_norm * (1.0 - alpha) + gradient_norm * alpha;

        // Apply update
        let adapter_mut = &mut self.lora_bank.adapters[expert_id];
        adapter_mut.update(&grad_a, &grad_b, self.config.lora_lr);
        self.stats.total_lora_updates += 1;

        // Check if adapter should be frozen
        let adapter_frozen = if adapter_mut.delta_norm() > self.config.max_delta_norm {
            adapter_mut.freeze();
            self.stats.adapters_frozen += 1;
            true
        } else {
            false
        };

        // Update loss EMA
        self.loss_ema = self.loss_ema * self.config.loss_ema_decay
            + base_loss * (1.0 - self.config.loss_ema_decay);

        // NES step for routing (periodic)
        let nes_stepped = if self.config.nes_update_freq > 0
            && (self.step + 1) % self.config.nes_update_freq == 0
        {
            let routing_params = self.routing_shadow.clone();
            let active_idx = if routing_params.is_empty() {
                None
            } else {
                let max_expert = routing_params.len() / MANIFOLD_DIM;
                if max_expert == 0 {
                    None
                } else {
                    Some(expert_id.min(max_expert - 1) * MANIFOLD_DIM)
                }
            };

            let new_params = self.nes_routing_step(&routing_params, |p| {
                let active_term = active_idx.map_or(0.0, |idx| {
                    p.iter().skip(idx).take(MANIFOLD_DIM).copied().sum::<f32>()
                        / MANIFOLD_DIM as f32
                });
                let l2_penalty = p.iter().map(|v| v * v).sum::<f32>();
                active_term - 0.01 * l2_penalty - base_loss
            });
            self.routing_shadow = new_params;
            true
        } else {
            false
        };

        self.step += 1;

        FmeaStepResult {
            loss: base_loss,
            gradient_norm,
            nes_stepped,
            adapter_frozen,
        }
    }

    /// Perform an NES step to evolve routing parameters.
    pub fn nes_routing_step(
        &mut self,
        routing_params: &Array1<f32>,
        fitness_evaluator: impl Fn(&Array1<f32>) -> f32,
    ) -> Array1<f32> {
        let perturbations = self
            .nes_optimizer
            .generate_perturbations(routing_params.len());

        let samples: Vec<NesSample> = perturbations
            .into_iter()
            .map(|pert| {
                let perturbed = routing_params + &pert;
                let fitness = fitness_evaluator(&perturbed);
                NesSample {
                    perturbation: pert,
                    fitness,
                }
            })
            .collect();

        let gradient = self.nes_optimizer.compute_gradient(&samples);
        let new_params = self.nes_optimizer.step(routing_params, &gradient);

        self.stats.total_nes_updates += 1;
        new_params
    }

    /// Get current statistics.
    pub fn stats(&self) -> &FmeaStats {
        &self.stats
    }

    /// Total trainable parameters.
    pub fn total_trainable_params(&self) -> usize {
        self.lora_bank.total_params()
    }

    /// Total memory usage of FMEA state.
    pub fn memory_bytes(&self) -> usize {
        self.lora_bank.total_memory_bytes()
    }
}

/// Compute multi-sample gradient for LoRA from JVP evaluations.
fn multi_sample_lora_gradient(
    jvp_values: &[f32],
    dirs_a: &[Array2<f32>],
    dirs_b: &[Array2<f32>],
    a_shape: (usize, usize),
    b_shape: (usize, usize),
) -> (Array2<f32>, Array2<f32>) {
    let samples = jvp_values.len().min(dirs_a.len()).min(dirs_b.len());

    let mut grad_a = Array2::zeros(a_shape);
    let mut grad_b = Array2::zeros(b_shape);

    if samples == 0 {
        return (grad_a, grad_b);
    }

    for i in 0..samples {
        grad_a = &grad_a + &(&dirs_a[i] * jvp_values[i]);
        grad_b = &grad_b + &(&dirs_b[i] * jvp_values[i]);
    }

    let k = samples as f32;
    grad_a /= k;
    grad_b /= k;

    (grad_a, grad_b)
}

fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

/// Compute mean squared error loss between two vectors.
pub fn mse_loss(output: &Array1<f32>, target: &Array1<f32>) -> f32 {
    let diff = output - target;
    diff.iter().map(|v| v * v).sum::<f32>() / diff.len() as f32
}

/// Compute cross-entropy loss from logits and target probabilities.
pub fn cross_entropy_loss(logits: &Array1<f32>, target_probs: &Array1<f32>) -> f32 {
    // Softmax
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|v| (v - max_val).exp()).collect();
    let sum_exp: f32 = exp.iter().sum();
    let log_probs: Vec<f32> = exp.iter().map(|e| (e / sum_exp).ln()).collect();

    -target_probs
        .iter()
        .zip(log_probs.iter())
        .map(|(&t, &lp)| t * lp)
        .sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmea_engine_creation() {
        let config = FmeaConfig::default();
        let engine = FmeaEngine::new(config, 128, 4096);
        assert_eq!(engine.lora_bank.adapters.len(), 128);
        assert_eq!(engine.step, 0);
    }

    #[test]
    fn test_fmea_step() {
        let config = FmeaConfig {
            jvp_samples: 4,
            lora_lr: 0.001,
            ..Default::default()
        };
        let mut engine = FmeaEngine::new(config, 8, 16);

        let x = Array1::from_vec(vec![1.0; 16]);
        let expert_output = Array1::from_vec(vec![0.5; 16]);
        let target = Array1::from_vec(vec![1.0; 16]);

        let result = engine.update_step(0, &x, &expert_output, &target, mse_loss);

        assert!(result.loss >= 0.0);
        assert!(result.gradient_norm >= 0.0);
        assert_eq!(engine.step, 1);
        assert_eq!(engine.stats.total_lora_updates, 1);
        assert_eq!(engine.stats.total_jvp_evals, 4);
    }

    #[test]
    fn test_fmea_step_zero_jvp_samples_safe() {
        let config = FmeaConfig {
            jvp_samples: 0,
            lora_lr: 0.001,
            ..Default::default()
        };
        let mut engine = FmeaEngine::new(config, 4, 8);

        let x = Array1::from_vec(vec![1.0; 8]);
        let expert_output = Array1::from_vec(vec![0.2; 8]);
        let target = Array1::from_vec(vec![0.7; 8]);

        let result = engine.update_step(0, &x, &expert_output, &target, mse_loss);

        assert!(result.gradient_norm.abs() < 1e-8);
        assert_eq!(engine.stats.total_jvp_evals, 0);
        assert_eq!(engine.stats.total_lora_updates, 1);
    }

    #[test]
    fn test_nes_runs_on_schedule_in_update_step() {
        let config = FmeaConfig {
            jvp_samples: 1,
            nes_update_freq: 2,
            nes_config: NesConfig {
                pop_size: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut engine = FmeaEngine::new(config, 4, 8);

        let x = Array1::from_vec(vec![1.0; 8]);
        let expert_output = Array1::from_vec(vec![0.0; 8]);
        let target = Array1::from_vec(vec![1.0; 8]);

        let step1 = engine.update_step(0, &x, &expert_output, &target, mse_loss);
        let step2 = engine.update_step(0, &x, &expert_output, &target, mse_loss);

        assert!(!step1.nes_stepped);
        assert!(step2.nes_stepped);
        assert_eq!(engine.stats.total_nes_updates, 1);
    }

    #[test]
    fn test_fmea_multiple_steps() {
        let config = FmeaConfig {
            jvp_samples: 2,
            lora_lr: 0.01,
            ..Default::default()
        };
        let mut engine = FmeaEngine::new(config, 4, 8);

        let x = Array1::from_vec(vec![1.0; 8]);
        let expert_output = Array1::from_vec(vec![0.0; 8]);
        let target = Array1::from_vec(vec![1.0; 8]);

        // Run 5 steps
        for _ in 0..5 {
            engine.update_step(0, &x, &expert_output, &target, mse_loss);
        }

        assert_eq!(engine.step, 5);
        assert_eq!(engine.stats.total_lora_updates, 5);
    }

    #[test]
    fn test_nes_routing_step() {
        let config = FmeaConfig::default();
        let mut engine = FmeaEngine::new(config, 4, 8);

        let routing_params = Array1::from_vec(vec![0.5; 10]);
        let new_params = engine.nes_routing_step(&routing_params, |p| {
            // Simple fitness: negative L2 norm (want to minimise)
            -p.iter().map(|v| v * v).sum::<f32>()
        });

        assert_eq!(new_params.len(), 10);
        assert_eq!(engine.stats.total_nes_updates, 1);
    }

    #[test]
    fn test_multi_sample_lora_gradient_empty_inputs() {
        let (grad_a, grad_b) =
            multi_sample_lora_gradient(&[], &[], &[], (LORA_RANK, 8), (8, LORA_RANK));

        assert_eq!(grad_a.dim(), (LORA_RANK, 8));
        assert_eq!(grad_b.dim(), (8, LORA_RANK));
        assert!(grad_a.iter().all(|v| v.abs() < 1e-8));
        assert!(grad_b.iter().all(|v| v.abs() < 1e-8));
    }

    #[test]
    fn test_mse_loss() {
        let output = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let target = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(mse_loss(&output, &target) < 1e-10);

        let target2 = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        assert!((mse_loss(&output, &target2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let logits = Array1::from_vec(vec![2.0, 1.0, 0.1]);
        let target = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let loss = cross_entropy_loss(&logits, &target);
        assert!(loss > 0.0);
        assert!(loss < 5.0); // Reasonable range
    }

    #[test]
    fn test_memory_bytes() {
        let config = FmeaConfig::default();
        let engine = FmeaEngine::new(config, 128, 4096);
        // 128 experts × 2 × 4 × 4096 × 4 bytes = 128 × 131072 = ~16MB
        let mem = engine.memory_bytes();
        assert!(mem > 0);
        assert_eq!(mem, 128 * 32768 * 4);
    }

    #[test]
    fn test_total_trainable_params() {
        let config = FmeaConfig::default();
        let engine = FmeaEngine::new(config, 128, 4096);
        // 128 × (4×4096 + 4096×4) = 128 × 32768 = 4,194,304
        assert_eq!(engine.total_trainable_params(), 128 * 32768);
    }
}
