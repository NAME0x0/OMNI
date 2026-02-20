//! Multi-Perspective Decoding orchestrator.
//!
//! Coordinates 4 perspectives through the model, computes agreement,
//! and produces the final token sequence.

use ndarray::Array1;

use crate::decoding::agreement::{
    compute_agreement, should_accept, AcceptancePolicy, AgreementResult,
    RejectionStrategy,
};
use crate::decoding::perspective_config::{
    apply_perspective, PerspectiveConfig,
};

/// A single MPD decoding step result.
#[derive(Clone, Debug)]
pub struct MpdStepResult {
    /// The accepted token ID.
    pub token_id: usize,

    /// Agreement details.
    pub agreement: AgreementResult,

    /// Whether the token was accepted on first try.
    pub accepted_first_try: bool,

    /// Number of resampling attempts (0 if accepted first try).
    pub resample_attempts: usize,
}

/// MPD decoder state.
pub struct MpdDecoder {
    /// The four perspective configs.
    pub perspectives: [PerspectiveConfig; 4],

    /// Acceptance policy.
    pub policy: AcceptancePolicy,

    /// Rejection strategy.
    pub rejection_strategy: RejectionStrategy,

    /// Maximum resampling attempts before forcing acceptance.
    pub max_resample: usize,

    /// Running statistics.
    pub stats: MpdStats,
}

/// Decoding statistics.
#[derive(Clone, Debug, Default)]
pub struct MpdStats {
    pub total_steps: u64,
    pub unanimous_accepts: u64,
    pub majority_accepts: u64,
    pub forced_accepts: u64,
    pub total_resamples: u64,
    pub total_jsd: f64,
}

impl MpdStats {
    pub fn unanimity_rate(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.unanimous_accepts as f64 / self.total_steps as f64
        }
    }

    pub fn average_jsd(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.total_jsd / self.total_steps as f64
        }
    }

    pub fn forced_accept_rate(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.forced_accepts as f64 / self.total_steps as f64
        }
    }
}

impl MpdDecoder {
    /// Create a new MPD decoder with default settings.
    pub fn new() -> Self {
        Self {
            perspectives: PerspectiveConfig::default_quad(),
            policy: AcceptancePolicy::Majority,
            rejection_strategy: RejectionStrategy::Resample { temperature: 0.7 },
            max_resample: 3,
            stats: MpdStats::default(),
        }
    }

    /// Create with strict unanimous policy.
    pub fn strict() -> Self {
        Self {
            perspectives: PerspectiveConfig::default_quad(),
            policy: AcceptancePolicy::Unanimous,
            rejection_strategy: RejectionStrategy::Resample { temperature: 0.5 },
            max_resample: 5,
            stats: MpdStats::default(),
        }
    }

    /// Create with fast (no-check) mode for benchmarking.
    pub fn fast() -> Self {
        Self {
            perspectives: PerspectiveConfig::default_quad(),
            policy: AcceptancePolicy::AlwaysAccept,
            rejection_strategy: RejectionStrategy::Resample { temperature: 1.0 },
            max_resample: 0,
            stats: MpdStats::default(),
        }
    }

    /// Run one MPD decoding step given base logits.
    ///
    /// `logits_fn` is called for each perspective with modified parameters,
    /// returning the logits for that perspective. In practice, this would
    /// re-run the forward pass with perturbations or apply them post-hoc.
    pub fn decode_step(
        &mut self,
        base_logits: &Array1<f32>,
        step_seed: u64,
    ) -> MpdStepResult {
        self.stats.total_steps += 1;

        // Apply each perspective's perturbation to the base logits
        let perspective_logits: Vec<Array1<f32>> = self
            .perspectives
            .iter()
            .enumerate()
            .map(|(i, config)| {
                if config.enabled {
                    apply_perspective(base_logits, config, step_seed + i as u64)
                } else {
                    base_logits.clone()
                }
            })
            .collect();

        // Compute agreement
        let agreement = compute_agreement(&perspective_logits);
        self.stats.total_jsd += agreement.jsd;

        let accepted = should_accept(&agreement, self.policy);

        if accepted {
            if agreement.unanimous {
                self.stats.unanimous_accepts += 1;
            } else {
                self.stats.majority_accepts += 1;
            }

            return MpdStepResult {
                token_id: agreement.token_id,
                agreement,
                accepted_first_try: true,
                resample_attempts: 0,
            };
        }

        // Token rejected — attempt resampling
        let mut attempts = 0;
        let mut best_agreement = agreement.clone();
        let mut best_jsd = agreement.jsd;

        while attempts < self.max_resample {
            attempts += 1;
            self.stats.total_resamples += 1;

            // Re-run with modified seed and potentially lower temperature
            let resample_seed = step_seed + 1000 + attempts as u64;
            let resample_logits: Vec<Array1<f32>> = self
                .perspectives
                .iter()
                .enumerate()
                .map(|(i, config)| {
                    let mut modified = config.clone();
                    // Lower temperature on resamples
                    if let RejectionStrategy::Resample { temperature } =
                        self.rejection_strategy
                    {
                        modified.temperature *= temperature;
                    }
                    apply_perspective(base_logits, &modified, resample_seed + i as u64)
                })
                .collect();

            let new_agreement = compute_agreement(&resample_logits);

            if should_accept(&new_agreement, self.policy) {
                if new_agreement.unanimous {
                    self.stats.unanimous_accepts += 1;
                } else {
                    self.stats.majority_accepts += 1;
                }

                return MpdStepResult {
                    token_id: new_agreement.token_id,
                    agreement: new_agreement,
                    accepted_first_try: false,
                    resample_attempts: attempts,
                };
            }

            if new_agreement.jsd < best_jsd {
                best_jsd = new_agreement.jsd;
                best_agreement = new_agreement;
            }
        }

        // Exhausted resamples — force accept
        self.stats.forced_accepts += 1;

        MpdStepResult {
            token_id: best_agreement.token_id,
            agreement: best_agreement,
            accepted_first_try: false,
            resample_attempts: attempts,
        }
    }

    /// Decode a full sequence of `max_tokens` tokens.
    pub fn decode_sequence(
        &mut self,
        logits_stream: &[Array1<f32>],
    ) -> Vec<MpdStepResult> {
        logits_stream
            .iter()
            .enumerate()
            .map(|(i, logits)| self.decode_step(logits, i as u64 * 7919))
            .collect()
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = MpdStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sharp_logits(winner: usize, vocab: usize) -> Array1<f32> {
        let mut v = vec![-10.0f32; vocab];
        v[winner] = 10.0;
        Array1::from_vec(v)
    }

    fn flat_logits(vocab: usize) -> Array1<f32> {
        Array1::from_vec(vec![0.0; vocab])
    }

    #[test]
    fn test_mpd_unanimous_accept() {
        let mut decoder = MpdDecoder::new();
        // Very sharp logits → all perspectives should agree
        let logits = sharp_logits(5, 100);
        let result = decoder.decode_step(&logits, 42);
        assert_eq!(result.token_id, 5);
        assert!(result.accepted_first_try);
    }

    #[test]
    fn test_mpd_stats_tracking() {
        let mut decoder = MpdDecoder::new();
        let logits = sharp_logits(3, 50);
        decoder.decode_step(&logits, 0);
        decoder.decode_step(&logits, 1);
        assert_eq!(decoder.stats.total_steps, 2);
    }

    #[test]
    fn test_mpd_fast_mode() {
        let mut decoder = MpdDecoder::fast();
        let logits = flat_logits(100);
        let result = decoder.decode_step(&logits, 42);
        // Fast mode always accepts
        assert!(result.accepted_first_try);
    }

    #[test]
    fn test_mpd_sequence_decode() {
        let mut decoder = MpdDecoder::new();
        let stream: Vec<Array1<f32>> = (0..5)
            .map(|i| sharp_logits(i, 20))
            .collect();
        let results = decoder.decode_sequence(&stream);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_mpd_strict_mode() {
        let decoder = MpdDecoder::strict();
        assert_eq!(decoder.policy, AcceptancePolicy::Unanimous);
        assert_eq!(decoder.max_resample, 5);
    }

    #[test]
    fn test_mpd_stats_rates() {
        let stats = MpdStats {
            total_steps: 100,
            unanimous_accepts: 80,
            majority_accepts: 15,
            forced_accepts: 5,
            total_resamples: 10,
            total_jsd: 5.0,
        };
        assert!((stats.unanimity_rate() - 0.8).abs() < 1e-10);
        assert!((stats.average_jsd() - 0.05).abs() < 1e-10);
        assert!((stats.forced_accept_rate() - 0.05).abs() < 1e-10);
    }
}
