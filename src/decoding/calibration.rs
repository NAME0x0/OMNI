//! Calibration — Expected Calibration Error (ECE) and geometric mean scoring.
//!
//! Ensures the model's confidence (from MPD agreement) aligns with actual
//! accuracy. A well-calibrated model that says "80% confident" should be
//! correct ~80% of the time.

/// A single prediction with confidence and correctness.
#[derive(Clone, Debug)]
pub struct CalibrationSample {
    /// Model's reported confidence (from MPD agreement: 1 - JSD).
    pub confidence: f64,

    /// Whether the prediction was correct.
    pub correct: bool,

    /// Token ID predicted.
    pub token_id: usize,
}

/// Binned calibration statistics.
#[derive(Clone, Debug)]
pub struct CalibrationBin {
    /// Average confidence in this bin.
    pub avg_confidence: f64,

    /// Empirical accuracy in this bin.
    pub accuracy: f64,

    /// Number of samples.
    pub count: usize,

    /// Gap: |accuracy - confidence|.
    pub gap: f64,
}

/// Expected Calibration Error (ECE) computation.
pub struct CalibrationAnalyzer {
    /// Number of bins for ECE.
    pub num_bins: usize,

    /// All collected samples.
    samples: Vec<CalibrationSample>,
}

impl CalibrationAnalyzer {
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins,
            samples: Vec::new(),
        }
    }

    /// Default with 15 bins.
    pub fn default_15() -> Self {
        Self::new(15)
    }

    /// Add a sample.
    pub fn add_sample(&mut self, confidence: f64, correct: bool, token_id: usize) {
        self.samples.push(CalibrationSample {
            confidence,
            correct,
            token_id,
        });
    }

    /// Compute binned calibration.
    pub fn compute_bins(&self) -> Vec<CalibrationBin> {
        let mut bins: Vec<Vec<&CalibrationSample>> =
            (0..self.num_bins).map(|_| Vec::new()).collect();

        for sample in &self.samples {
            let bin_idx =
                ((sample.confidence * self.num_bins as f64) as usize).min(self.num_bins - 1);
            bins[bin_idx].push(sample);
        }

        bins.iter()
            .enumerate()
            .filter(|(_, bin)| !bin.is_empty())
            .map(|(_i, bin)| {
                let avg_conf: f64 =
                    bin.iter().map(|s| s.confidence).sum::<f64>() / bin.len() as f64;
                let accuracy = bin.iter().filter(|s| s.correct).count() as f64 / bin.len() as f64;
                CalibrationBin {
                    avg_confidence: avg_conf,
                    accuracy,
                    count: bin.len(),
                    gap: (accuracy - avg_conf).abs(),
                }
            })
            .collect()
    }

    /// Expected Calibration Error: weighted average of |bin_accuracy - bin_confidence|.
    pub fn ece(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let bins = self.compute_bins();
        let n = self.samples.len() as f64;

        bins.iter().map(|bin| bin.count as f64 / n * bin.gap).sum()
    }

    /// Maximum Calibration Error: max gap across bins.
    pub fn mce(&self) -> f64 {
        let bins = self.compute_bins();
        bins.iter().map(|bin| bin.gap).fold(0.0f64, f64::max)
    }

    /// Overall accuracy.
    pub fn accuracy(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().filter(|s| s.correct).count() as f64 / self.samples.len() as f64
    }

    /// Average confidence.
    pub fn avg_confidence(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().map(|s| s.confidence).sum::<f64>() / self.samples.len() as f64
    }

    /// Number of samples.
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Geometric mean of per-token probabilities (perplexity-related).
/// Given a sequence of per-token log-probabilities, compute:
///   geo_mean = exp((1/N) · Σ log_prob)
/// This is equivalent to perplexity^(-1).
pub fn geometric_mean_probability(log_probs: &[f64]) -> f64 {
    if log_probs.is_empty() {
        return 0.0;
    }
    let avg_log = log_probs.iter().sum::<f64>() / log_probs.len() as f64;
    avg_log.exp()
}

/// Compute perplexity from log-probabilities.
pub fn perplexity(log_probs: &[f64]) -> f64 {
    if log_probs.is_empty() {
        return f64::INFINITY;
    }
    let avg_neg_log = -log_probs.iter().sum::<f64>() / log_probs.len() as f64;
    avg_neg_log.exp()
}

/// Reliability diagram data for plotting.
pub struct ReliabilityDiagram {
    pub bins: Vec<(f64, f64)>, // (confidence, accuracy) pairs
    pub ece: f64,
    pub mce: f64,
    pub num_samples: usize,
}

impl CalibrationAnalyzer {
    /// Generate reliability diagram data.
    pub fn reliability_diagram(&self) -> ReliabilityDiagram {
        let bins = self.compute_bins();
        let bin_points: Vec<(f64, f64)> = bins
            .iter()
            .map(|b| (b.avg_confidence, b.accuracy))
            .collect();

        ReliabilityDiagram {
            bins: bin_points,
            ece: self.ece(),
            mce: self.mce(),
            num_samples: self.num_samples(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_calibration() {
        let mut analyzer = CalibrationAnalyzer::new(10);
        // Add perfectly calibrated samples
        for _ in 0..80 {
            analyzer.add_sample(0.8, true, 0);
        }
        for _ in 0..20 {
            analyzer.add_sample(0.8, false, 0);
        }
        // accuracy ≈ 0.8, avg_confidence ≈ 0.8 → low ECE
        let ece = analyzer.ece();
        assert!(ece < 0.05, "ece = {}", ece);
    }

    #[test]
    fn test_overconfident() {
        let mut analyzer = CalibrationAnalyzer::new(10);
        // High confidence but low accuracy
        for _ in 0..50 {
            analyzer.add_sample(0.95, false, 0);
        }
        for _ in 0..50 {
            analyzer.add_sample(0.95, true, 0);
        }
        // accuracy ≈ 0.5 but confidence ≈ 0.95 → high ECE
        let ece = analyzer.ece();
        assert!(ece > 0.3, "ece = {}", ece);
    }

    #[test]
    fn test_ece_empty() {
        let analyzer = CalibrationAnalyzer::new(10);
        assert_eq!(analyzer.ece(), 0.0);
    }

    #[test]
    fn test_geometric_mean() {
        // All same probability
        let log_probs = vec![-1.0, -1.0, -1.0];
        let gm = geometric_mean_probability(&log_probs);
        assert!((gm - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_perplexity_uniform() {
        // Uniform over 100 tokens: log_prob = ln(1/100)
        let log_prob = (1.0 / 100.0_f64).ln();
        let log_probs = vec![log_prob; 50];
        let ppl = perplexity(&log_probs);
        assert!((ppl - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_perplexity_perfect() {
        // Perfect prediction: log_prob = 0 (prob = 1)
        let log_probs = vec![0.0; 10];
        let ppl = perplexity(&log_probs);
        assert!((ppl - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy() {
        let mut analyzer = CalibrationAnalyzer::new(10);
        analyzer.add_sample(0.9, true, 0);
        analyzer.add_sample(0.8, false, 0);
        assert!((analyzer.accuracy() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_reliability_diagram() {
        let mut analyzer = CalibrationAnalyzer::new(5);
        for i in 0..100 {
            let conf = i as f64 / 100.0;
            let correct = (i % 3) != 0; // ~66% accuracy
            analyzer.add_sample(conf, correct, i);
        }
        let diagram = analyzer.reliability_diagram();
        assert!(!diagram.bins.is_empty());
        assert_eq!(diagram.num_samples, 100);
    }
}
