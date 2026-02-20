//! Full inference pipeline: token in → token out.
//!
//! Orchestrates all 7 novel components:
//! 1. PDR (O(1) recurrence) + GQA (windowed attention) — core layers
//! 2. Manifold routing (flat torus T²) — expert selection
//! 3. Ternary execution (1.58-bit weights) — compute
//! 4. HDM (hyperdimensional memory) — episodic memory
//! 5. MPD (multi-perspective decoding) — verification
//! 6. FMEA (forward-mode adaptation) — continual learning
//! 7. SPP (safety polytope projection) — hard safety

use ndarray::Array1;

/// Pipeline configuration.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Model dimension.
    pub d_model: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Number of layers.
    pub n_layers: usize,

    /// Number of experts total.
    pub n_experts: usize,

    /// Maximum sequence length for generation.
    pub max_gen_length: usize,

    /// Whether to enable FMEA (online learning).
    pub enable_fmea: bool,

    /// Whether to enable SPP (safety projection).
    pub enable_safety: bool,

    /// Whether to enable HDM (episodic memory).
    pub enable_memory: bool,

    /// Whether to enable MPD (multi-perspective decoding).
    pub enable_mpd: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            d_model: crate::config::D_MODEL,
            vocab_size: crate::config::VOCAB_SIZE,
            n_layers: crate::config::N_LAYERS,
            n_experts: crate::config::N_EXPERTS,
            max_gen_length: 2048,
            enable_fmea: true,
            enable_safety: true,
            enable_memory: true,
            enable_mpd: true,
        }
    }
}

/// A single generated token result.
#[derive(Clone, Debug)]
pub struct TokenResult {
    /// Selected token ID.
    pub token_id: usize,

    /// Probability of selected token.
    pub probability: f32,

    /// Whether MPD reached agreement.
    pub mpd_agreed: bool,

    /// Safety score [0, 1].
    pub safety_score: f32,

    /// Expert that was routed to.
    pub expert_id: usize,

    /// Layer-by-layer routing trace.
    pub routing_trace: Vec<usize>,
}

/// Per-token pipeline stage timings (in microseconds).
#[derive(Clone, Debug, Default)]
pub struct StageTimings {
    pub embed_us: u64,
    pub layer_compute_us: u64,
    pub routing_us: u64,
    pub expert_load_us: u64,
    pub safety_project_us: u64,
    pub mpd_decode_us: u64,
    pub memory_update_us: u64,
    pub total_us: u64,
}

impl StageTimings {
    /// Compute percentage breakdown.
    pub fn breakdown(&self) -> Vec<(&str, f32)> {
        let t = self.total_us as f32;
        if t == 0.0 {
            return vec![];
        }
        vec![
            ("embed", self.embed_us as f32 / t * 100.0),
            ("layers", self.layer_compute_us as f32 / t * 100.0),
            ("routing", self.routing_us as f32 / t * 100.0),
            ("expert_load", self.expert_load_us as f32 / t * 100.0),
            ("safety", self.safety_project_us as f32 / t * 100.0),
            ("mpd", self.mpd_decode_us as f32 / t * 100.0),
            ("memory", self.memory_update_us as f32 / t * 100.0),
        ]
    }
}

/// Generation result: sequence of tokens with metadata.
#[derive(Debug)]
pub struct GenerationResult {
    /// Generated token IDs.
    pub tokens: Vec<usize>,

    /// Per-token results.
    pub token_results: Vec<TokenResult>,

    /// Aggregate timing.
    pub total_time_ms: f32,

    /// Tokens per second.
    pub tokens_per_second: f32,

    /// Average safety score.
    pub avg_safety_score: f32,

    /// MPD agreement rate.
    pub mpd_agreement_rate: f32,
}

/// The inference pipeline orchestrator.
///
/// In a full implementation, this holds references to all subsystems.
/// Here we define the structure and token-processing skeleton.
pub struct InferencePipeline {
    pub config: PipelineConfig,

    /// Current generation state.
    tokens_generated: usize,

    /// Cumulative timing stats.
    pub cumulative_timings: StageTimings,
}

impl InferencePipeline {
    /// Create a new inference pipeline.
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            tokens_generated: 0,
            cumulative_timings: StageTimings::default(),
        }
    }

    /// Process a single token through the full pipeline.
    ///
    /// The full pipeline per token:
    /// 1. Embed token → hidden state h ∈ R^{d_model}
    /// 2. For each layer l ∈ [0, n_layers):
    ///    a. If PDR layer: h = PDR(h, state_l)
    ///    b. If GQA layer: h = GQA(h, kv_cache_l)
    ///    c. Route: expert_id = ManifoldRoute(h, l)
    ///    d. Load expert (from cache or NVMe stream)
    ///    e. h = h + ExpertFFN(h)  [ternary GEMM]
    ///    f. If FMEA enabled: update LoRA adapter
    /// 3. If SPP enabled: h = SafetyProject(h)
    /// 4. Compute logits = Embeddingᵀ · h
    /// 5. If MPD enabled: multi-perspective decode
    /// 6. If HDM enabled: update episodic memory
    /// 7. Return selected token
    pub fn process_token(
        &mut self,
        _token_id: usize,
        _hidden_state: &mut Array1<f32>,
    ) -> TokenResult {
        self.tokens_generated += 1;

        // Skeleton: in a full implementation, each stage would call
        // the respective subsystems. This structure defines the interface.
        TokenResult {
            token_id: 0,
            probability: 0.0,
            mpd_agreed: true,
            safety_score: 1.0,
            expert_id: 0,
            routing_trace: Vec::new(),
        }
    }

    /// Generate a sequence of tokens (autoregressive).
    pub fn generate(
        &mut self,
        prompt_tokens: &[usize],
        max_tokens: usize,
        stop_token: usize,
    ) -> GenerationResult {
        let start = std::time::Instant::now();
        let mut generated = Vec::new();
        let mut results = Vec::new();

        let mut hidden = Array1::zeros(self.config.d_model);

        // Process prompt (prefill)
        for &tok in prompt_tokens {
            let result = self.process_token(tok, &mut hidden);
            results.push(result);
        }

        // Generate new tokens
        for _ in 0..max_tokens {
            let last_token = generated.last().copied().unwrap_or(0);
            let result = self.process_token(last_token, &mut hidden);
            let token_id = result.token_id;
            generated.push(token_id);
            results.push(result);

            if token_id == stop_token {
                break;
            }
        }

        let elapsed = start.elapsed().as_secs_f32() * 1000.0;
        let total_tokens = generated.len() as f32;

        let avg_safety = if results.is_empty() {
            1.0
        } else {
            results.iter().map(|r| r.safety_score).sum::<f32>() / results.len() as f32
        };

        let mpd_rate = if results.is_empty() {
            1.0
        } else {
            results.iter().filter(|r| r.mpd_agreed).count() as f32 / results.len() as f32
        };

        GenerationResult {
            tokens: generated,
            token_results: results,
            total_time_ms: elapsed,
            tokens_per_second: if elapsed > 0.0 {
                total_tokens / (elapsed / 1000.0)
            } else {
                0.0
            },
            avg_safety_score: avg_safety,
            mpd_agreement_rate: mpd_rate,
        }
    }

    /// Reset pipeline state for a new conversation.
    pub fn reset(&mut self) {
        self.tokens_generated = 0;
        self.cumulative_timings = StageTimings::default();
    }

    /// Get tokens generated in current session.
    pub fn tokens_generated(&self) -> usize {
        self.tokens_generated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = InferencePipeline::new(config);
        assert_eq!(pipeline.tokens_generated(), 0);
    }

    #[test]
    fn test_pipeline_process_token() {
        let config = PipelineConfig {
            d_model: 32,
            ..Default::default()
        };
        let mut pipeline = InferencePipeline::new(config);
        let mut hidden = Array1::zeros(32);
        let result = pipeline.process_token(42, &mut hidden);
        assert_eq!(pipeline.tokens_generated(), 1);
        assert!(result.safety_score >= 0.0);
    }

    #[test]
    fn test_pipeline_generate() {
        let config = PipelineConfig {
            d_model: 16,
            max_gen_length: 10,
            ..Default::default()
        };
        let mut pipeline = InferencePipeline::new(config);
        let result = pipeline.generate(&[1, 2, 3], 5, 0);
        assert!(result.tokens.len() <= 5);
        assert!(result.total_time_ms >= 0.0);
    }

    #[test]
    fn test_pipeline_reset() {
        let config = PipelineConfig {
            d_model: 16,
            ..Default::default()
        };
        let mut pipeline = InferencePipeline::new(config);
        let mut hidden = Array1::zeros(16);
        pipeline.process_token(1, &mut hidden);
        assert_eq!(pipeline.tokens_generated(), 1);
        pipeline.reset();
        assert_eq!(pipeline.tokens_generated(), 0);
    }

    #[test]
    fn test_stage_timings() {
        let timings = StageTimings {
            embed_us: 10,
            layer_compute_us: 80,
            routing_us: 5,
            expert_load_us: 3,
            safety_project_us: 1,
            mpd_decode_us: 1,
            memory_update_us: 0,
            total_us: 100,
        };
        let breakdown = timings.breakdown();
        assert!(!breakdown.is_empty());
        assert!((breakdown[1].1 - 80.0).abs() < 1e-3); // layers = 80%
    }

    #[test]
    fn test_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.d_model, crate::config::D_MODEL);
        assert_eq!(config.n_layers, crate::config::N_LAYERS);
    }
}
