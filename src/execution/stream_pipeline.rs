//! Layer-streamed execution pipeline.
//!
//! Orchestrates the per-token forward pass across 80 layers:
//!   1. For each layer ℓ:
//!      a. Route token through manifold router → expert_id
//!      b. Ensure expert is loaded via double-buffer
//!      c. Compute PDR/GQA layer (shared params, always resident)
//!      d. Compute expert FFN (ternary matvec, zero-multiply)
//!      e. Prefetch layer ℓ+1's predicted expert
//!   2. Final RMS-norm → logits via tied embedding
//!
//! The pipeline is **sequential** within a single token but enables
//! asynchronous expert prefetching across layers.

use ndarray::Array1;

use crate::config;
use crate::core::model::{LayerKind, PerspectiveModel};
use crate::core::pdr_state::PdrStateBank;
use crate::core::windowed_gqa::KVCacheBank;
use crate::execution::double_buffer::{DoubleBuffer, DoubleBufferStats};
use crate::execution::ternary_pack::TernaryExpertFfn;
use crate::routing::router::{ModelRouter, RoutingDecision};

/// Result of a single forward step through the full model.
pub struct StepResult {
    /// Output logits (vocab_size).
    pub logits: Array1<f32>,

    /// Per-layer routing decisions.
    pub routing_trace: Vec<RoutingDecision>,

    /// Which experts were loaded from NVMe vs cached.
    pub load_trace: Vec<LayerLoadEvent>,
}

/// Record of what happened during expert loading for one layer.
#[derive(Clone, Debug)]
pub struct LayerLoadEvent {
    pub layer_idx: usize,
    pub expert_id: usize,
    pub was_prefetched: bool,
    pub was_cached: bool,
}

/// A provider trait for loading expert weights from storage.
/// Implementations may read from NVMe, memory-mapped files, or cache.
pub trait ExpertProvider {
    /// Load expert weights for the given expert_id.
    fn load_expert(&self, expert_id: usize) -> TernaryExpertFfn;

    /// Check if an expert is in the hot RAM cache.
    fn is_cached(&self, expert_id: usize) -> bool;

    /// Number of experts available.
    fn num_experts(&self) -> usize;
}

/// Simple in-memory expert provider (for testing / small models).
pub struct InMemoryExpertProvider {
    pub experts: Vec<TernaryExpertFfn>,
}

impl ExpertProvider for InMemoryExpertProvider {
    fn load_expert(&self, expert_id: usize) -> TernaryExpertFfn {
        let e = &self.experts[expert_id];
        TernaryExpertFfn {
            w_gate: e.w_gate.clone(),
            w_up: e.w_up.clone(),
            w_down: e.w_down.clone(),
            gate_scale: e.gate_scale,
            up_scale: e.up_scale,
            down_scale: e.down_scale,
        }
    }

    fn is_cached(&self, _expert_id: usize) -> bool {
        true // everything is in memory
    }

    fn num_experts(&self) -> usize {
        self.experts.len()
    }
}

/// The main streaming execution pipeline.
pub struct StreamPipeline {
    /// Double-buffer system.
    double_buffer: DoubleBuffer,

    /// Running count of tokens processed.
    tokens_processed: u64,

    /// Whether to use prediction-ahead prefetching.
    prefetch_enabled: bool,

    /// Prediction-ahead hit counter.
    prefetch_hits: u64,

    /// Prediction-ahead miss counter.
    prefetch_misses: u64,
}

impl StreamPipeline {
    pub fn new() -> Self {
        Self {
            double_buffer: DoubleBuffer::new(),
            tokens_processed: 0,
            prefetch_enabled: true,
            prefetch_hits: 0,
            prefetch_misses: 0,
        }
    }

    pub fn with_prefetch(mut self, enabled: bool) -> Self {
        self.prefetch_enabled = enabled;
        self
    }

    /// Execute one full forward pass for a single token.
    ///
    /// Walks through all 80 layers, routing + loading experts + computing.
    pub fn forward_token(
        &mut self,
        token_id: usize,
        model: &PerspectiveModel,
        router: &ModelRouter,
        pdr_states: &mut PdrStateBank,
        kv_caches: &mut KVCacheBank,
        provider: &dyn ExpertProvider,
    ) -> StepResult {
        // Embed
        let mut h = model.embedding.embed(token_id as u32);

        let mut routing_trace = Vec::with_capacity(config::N_LAYERS);
        let mut load_trace = Vec::with_capacity(config::N_LAYERS);

        for layer_idx in 0..config::N_LAYERS {
            // --- Route ---
            let decision = router.route(layer_idx, &h);
            let expert_id = decision.expert_id;
            routing_trace.push(decision);

            // --- Load expert via double-buffer ---
            let was_cached = provider.is_cached(expert_id);
            let loader = |eid: usize| provider.load_expert(eid);

            let expert = self.double_buffer.get_expert_for_layer(expert_id, &loader);

            // --- Compute shared layer (PDR or GQA) ---
            match &model.layers[layer_idx] {
                LayerKind::Pdr(pdr_layer) => {
                    let pdr_idx = if layer_idx < config::N_LAYERS {
                        // Count PDR layers before this one
                        (0..layer_idx)
                            .filter(|&i| config::is_pdr_layer(i))
                            .count()
                    } else {
                        0
                    };
                    if pdr_idx < pdr_states.states.len() {
                        h = pdr_layer.forward_step(&h, &mut pdr_states.states[pdr_idx]);
                    }
                }
                LayerKind::Gqa(gqa_layer) => {
                    let gqa_idx = if layer_idx < config::N_LAYERS {
                        (0..layer_idx)
                            .filter(|&i| !config::is_pdr_layer(i))
                            .count()
                    } else {
                        0
                    };
                    if gqa_idx < kv_caches.caches.len() {
                        h = gqa_layer.forward_step(&h, &mut kv_caches.caches[gqa_idx]);
                    }
                }
            }

            // --- Compute expert FFN (ternary, zero-multiply) ---
            h = Array1::from_vec(expert.forward(h.as_slice().unwrap()));

            // --- Record load event ---
            load_trace.push(LayerLoadEvent {
                layer_idx,
                expert_id,
                was_prefetched: !was_cached
                    && self.double_buffer.stats.overlapped > 0,
                was_cached,
            });

            // --- Prefetch next layer's predicted expert ---
            if self.prefetch_enabled && layer_idx + 1 < config::N_LAYERS {
                let predicted = router.predict_ahead(&h, layer_idx);
                let next_eid = predicted.expert_id;
                self.double_buffer.prefetch(next_eid, &loader);
            }

            // Release compute buffer
            self.double_buffer.release_compute();
        }

        // --- Final norm + logits ---
        let normed = rms_norm(&h, &model.final_norm_scale);
        let logits = model.embedding.to_logits(&normed);

        self.tokens_processed += 1;

        StepResult {
            logits,
            routing_trace,
            load_trace,
        }
    }

    /// Get statistics from the double-buffer system.
    pub fn buffer_stats(&self) -> &DoubleBufferStats {
        &self.double_buffer.stats
    }

    /// Total tokens processed.
    pub fn tokens_processed(&self) -> u64 {
        self.tokens_processed
    }

    /// Prefetch hit rate.
    pub fn prefetch_hit_rate(&self) -> f64 {
        let total = self.prefetch_hits + self.prefetch_misses;
        if total == 0 {
            0.0
        } else {
            self.prefetch_hits as f64 / total as f64
        }
    }
}

/// RMS normalization (matches core implementation).
fn rms_norm(x: &Array1<f32>, scale: &Array1<f32>) -> Array1<f32> {
    let n = x.len() as f32;
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n).sqrt() + 1e-6;
    let normed = x / rms;
    &normed * scale
}

/// Pipeline configuration for tuning.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Enable prediction-ahead prefetching.
    pub prefetch_enabled: bool,

    /// Maximum number of layers to predict ahead.
    pub prefetch_depth: usize,

    /// Enable double-buffering.
    pub double_buffer_enabled: bool,

    /// Minimum expert residency time (in layers) before eviction.
    pub min_residency: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            prefetch_enabled: true,
            prefetch_depth: 1,
            double_buffer_enabled: true,
            min_residency: 2,
        }
    }
}

/// Throughput estimator based on hardware parameters.
pub struct ThroughputEstimator {
    /// NVMe sequential read bandwidth (GB/s).
    pub nvme_bandwidth_gbps: f64,

    /// PCIe bandwidth (GB/s).
    pub pcie_bandwidth_gbps: f64,

    /// Expert size (bytes).
    pub expert_size_bytes: u64,

    /// Compute time per layer (ms).
    pub compute_time_ms: f64,

    /// Prefetch hit rate (0..1).
    pub prefetch_hit_rate: f64,
}

impl ThroughputEstimator {
    /// Create estimator with default PERSPECTIVE parameters.
    pub fn default_perspective() -> Self {
        // Expert FFN: 3 matrices × 4096 × 11008 × 0.2 bytes/param
        let expert_bytes = 3 * 4096 * 11008 * 2 / 10; // ~2.7 MB in ternary
        Self {
            nvme_bandwidth_gbps: 7.0,  // Gen4 NVMe
            pcie_bandwidth_gbps: 14.0, // PCIe 4.0 x16
            expert_size_bytes: expert_bytes as u64,
            compute_time_ms: 3.1, // from docs
            prefetch_hit_rate: 0.87,
        }
    }

    /// Estimate tokens per second.
    pub fn estimate_tok_per_sec(&self) -> f64 {
        let load_time_ms = self.expert_size_bytes as f64
            / (self.nvme_bandwidth_gbps * 1e6); // ms

        // Effective load time accounts for prefetch hits (fully hidden)
        let effective_load_ms =
            load_time_ms * (1.0 - self.prefetch_hit_rate);

        // Total per-layer time = max(compute, effective_load)
        let per_layer_ms = self.compute_time_ms.max(effective_load_ms);

        // Total per-token time across 80 layers
        let per_token_ms = per_layer_ms * config::N_LAYERS as f64;

        1000.0 / per_token_ms
    }

    /// Estimate end-to-end latency for one token (ms).
    pub fn estimate_latency_ms(&self) -> f64 {
        1000.0 / self.estimate_tok_per_sec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ternary_pack::TernaryMatrix;

    fn make_dummy_expert(d: usize, ffn: usize) -> TernaryExpertFfn {
        TernaryExpertFfn {
            w_gate: TernaryMatrix::zeros(ffn, d),
            w_up: TernaryMatrix::zeros(ffn, d),
            w_down: TernaryMatrix::zeros(d, ffn),
            gate_scale: 1.0,
            up_scale: 1.0,
            down_scale: 1.0,
        }
    }

    #[test]
    fn test_pipeline_config_default() {
        let cfg = PipelineConfig::default();
        assert!(cfg.prefetch_enabled);
        assert_eq!(cfg.prefetch_depth, 1);
        assert!(cfg.double_buffer_enabled);
    }

    #[test]
    fn test_in_memory_provider() {
        let experts = vec![
            make_dummy_expert(8, 16),
            make_dummy_expert(8, 16),
        ];
        let provider = InMemoryExpertProvider { experts };
        assert_eq!(provider.num_experts(), 2);
        assert!(provider.is_cached(0));
        let _e = provider.load_expert(0);
    }

    #[test]
    fn test_throughput_estimator() {
        let est = ThroughputEstimator::default_perspective();
        let tps = est.estimate_tok_per_sec();
        // Should be positive and reasonable
        assert!(tps > 0.0);
        assert!(tps < 1000.0); // sanity bound
    }

    #[test]
    fn test_stream_pipeline_creation() {
        let pipe = StreamPipeline::new();
        assert_eq!(pipe.tokens_processed(), 0);
        assert_eq!(pipe.buffer_stats().stalls, 0);
    }

    #[test]
    fn test_pipeline_with_prefetch_toggle() {
        let pipe = StreamPipeline::new().with_prefetch(false);
        assert!(!pipe.prefetch_enabled);
    }

    #[test]
    fn test_rms_norm_unit() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let scale = Array1::ones(4);
        let normed = rms_norm(&x, &scale);
        // RMS of [1,2,3,4] = sqrt(30/4) ≈ 2.7386
        let rms = (30.0_f32 / 4.0).sqrt();
        for i in 0..4 {
            let expected = x[i] / (rms + 1e-6);
            assert!((normed[i] - expected).abs() < 1e-4);
        }
    }
}
