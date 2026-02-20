//! Safety anchors for the Safety Polytope.
//!
//! Anchors are reference points in the representation space that define
//! "known safe" regions. The polytope is the convex hull of these anchors.
//! ~1000 anchors are extracted from curated safe outputs during training.

use ndarray::Array1;

/// A safety anchor point in representation space.
#[derive(Clone, Debug)]
pub struct SafetyAnchor {
    /// The anchor vector in R^d.
    pub vector: Array1<f32>,

    /// Semantic label for this anchor.
    pub label: String,

    /// Category (e.g., "factual", "helpful", "harmless").
    pub category: AnchorCategory,

    /// Confidence score [0, 1] — how reliably safe this anchor is.
    pub confidence: f32,
}

/// Categories of safety anchors.
#[derive(Clone, Debug, PartialEq)]
pub enum AnchorCategory {
    /// Factually grounded response patterns.
    Factual,
    /// Helpful and informative patterns.
    Helpful,
    /// Explicitly harmless patterns.
    Harmless,
    /// Refusal patterns (for unsafe requests).
    Refusal,
    /// Uncertainty expression patterns ("I don't know").
    Uncertainty,
    /// Custom category.
    Custom(String),
}

impl SafetyAnchor {
    /// Create a new anchor.
    pub fn new(vector: Array1<f32>, label: &str, category: AnchorCategory) -> Self {
        Self {
            vector,
            label: label.to_string(),
            category,
            confidence: 1.0,
        }
    }

    /// Set confidence level.
    pub fn with_confidence(mut self, c: f32) -> Self {
        self.confidence = c.clamp(0.0, 1.0);
        self
    }

    /// Dimension of this anchor.
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Compute distance to a point.
    pub fn distance_to(&self, x: &Array1<f32>) -> f32 {
        let diff = &self.vector - x;
        diff.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Cosine similarity to a point.
    pub fn cosine_similarity(&self, x: &Array1<f32>) -> f32 {
        let dot: f32 = self.vector.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
        let norm_a: f32 = self.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        dot / (norm_a * norm_b + 1e-8)
    }
}

/// A collection of safety anchors.
pub struct AnchorSet {
    /// All anchors.
    pub anchors: Vec<SafetyAnchor>,

    /// Precomputed centroid for fast distance checks.
    centroid: Option<Array1<f32>>,
}

impl AnchorSet {
    pub fn new() -> Self {
        Self {
            anchors: Vec::new(),
            centroid: None,
        }
    }

    /// Add an anchor and invalidate centroid cache.
    pub fn add(&mut self, anchor: SafetyAnchor) {
        self.anchors.push(anchor);
        self.centroid = None;
    }

    /// Compute the centroid of all anchors.
    pub fn centroid(&mut self) -> &Array1<f32> {
        if self.centroid.is_none() && !self.anchors.is_empty() {
            let dim = self.anchors[0].dim();
            let mut c = Array1::zeros(dim);
            for a in &self.anchors {
                c = &c + &a.vector;
            }
            c /= self.anchors.len() as f32;
            self.centroid = Some(c);
        }
        self.centroid.as_ref().unwrap()
    }

    /// Find the K nearest anchors to a point.
    pub fn nearest_k(&self, x: &Array1<f32>, k: usize) -> Vec<(usize, f32)> {
        let mut dists: Vec<(usize, f32)> = self
            .anchors
            .iter()
            .enumerate()
            .map(|(i, a)| (i, a.distance_to(x)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.truncate(k);
        dists
    }

    /// Find the nearest anchor to a point.
    pub fn nearest(&self, x: &Array1<f32>) -> Option<(usize, f32)> {
        self.nearest_k(x, 1).into_iter().next()
    }

    /// Maximum cosine similarity to any anchor (safety score proxy).
    pub fn max_similarity(&self, x: &Array1<f32>) -> f32 {
        self.anchors
            .iter()
            .map(|a| a.cosine_similarity(x))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Compute a convex combination of the K nearest anchors
    /// that is closest to x (approximate projection onto convex hull).
    /// Uses iterative weight adjustment.
    pub fn project_to_convex_hull(&self, x: &Array1<f32>, k: usize) -> Array1<f32> {
        let nearest = self.nearest_k(x, k);
        if nearest.is_empty() {
            return x.clone();
        }

        // Simple projection: weighted average by inverse distance
        let mut weights: Vec<f32> = nearest.iter().map(|(_, d)| 1.0 / (d + 1e-6)).collect();
        let sum: f32 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }

        let dim = x.len();
        let mut result = Array1::zeros(dim);
        for (i, &(anchor_idx, _)) in nearest.iter().enumerate() {
            result = &result + &(&self.anchors[anchor_idx].vector * weights[i]);
        }
        result
    }

    /// Filter anchors by category.
    pub fn by_category(&self, category: &AnchorCategory) -> Vec<&SafetyAnchor> {
        self.anchors
            .iter()
            .filter(|a| &a.category == category)
            .collect()
    }

    /// Number of anchors.
    pub fn len(&self) -> usize {
        self.anchors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty()
    }

    /// Generate synthetic anchors for testing (random points in a safe region).
    pub fn generate_synthetic(dim: usize, count: usize, seed: u64) -> Self {
        let mut set = Self::new();
        let mut state = seed;

        let categories = [
            AnchorCategory::Factual,
            AnchorCategory::Helpful,
            AnchorCategory::Harmless,
            AnchorCategory::Refusal,
            AnchorCategory::Uncertainty,
        ];

        for i in 0..count {
            let values: Vec<f32> = (0..dim)
                .map(|_| {
                    state = xorshift64(state);
                    let u = (state as f32) / (u64::MAX as f32);
                    (u - 0.5) * 2.0 // [-1, 1] — safe region
                })
                .collect();

            let cat = &categories[i % categories.len()];
            set.add(
                SafetyAnchor::new(
                    Array1::from_vec(values),
                    &format!("synth_anchor_{}", i),
                    cat.clone(),
                )
                .with_confidence(0.9),
            );
        }
        set
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
    fn test_anchor_distance() {
        let a = SafetyAnchor::new(
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
            "origin-ish",
            AnchorCategory::Factual,
        );
        let x = Array1::from_vec(vec![4.0, 0.0, 0.0]);
        assert!((a.distance_to(&x) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = SafetyAnchor::new(
            Array1::from_vec(vec![1.0, 0.0]),
            "right",
            AnchorCategory::Helpful,
        );
        // Same direction → 1.0
        assert!((a.cosine_similarity(&Array1::from_vec(vec![2.0, 0.0])) - 1.0).abs() < 1e-5);
        // Orthogonal → 0.0
        assert!(a.cosine_similarity(&Array1::from_vec(vec![0.0, 1.0])).abs() < 1e-5);
    }

    #[test]
    fn test_nearest_k() {
        let mut set = AnchorSet::new();
        set.add(SafetyAnchor::new(
            Array1::from_vec(vec![0.0, 0.0]),
            "origin",
            AnchorCategory::Factual,
        ));
        set.add(SafetyAnchor::new(
            Array1::from_vec(vec![10.0, 0.0]),
            "far",
            AnchorCategory::Helpful,
        ));
        set.add(SafetyAnchor::new(
            Array1::from_vec(vec![1.0, 1.0]),
            "near",
            AnchorCategory::Harmless,
        ));

        let x = Array1::from_vec(vec![0.5, 0.5]);
        let nearest = set.nearest_k(&x, 2);
        assert_eq!(nearest.len(), 2);
        // Origin and (1,1) should be nearest
        assert!(nearest[0].0 == 0 || nearest[0].0 == 2);
    }

    #[test]
    fn test_centroid() {
        let mut set = AnchorSet::new();
        set.add(SafetyAnchor::new(
            Array1::from_vec(vec![0.0, 0.0]),
            "a",
            AnchorCategory::Factual,
        ));
        set.add(SafetyAnchor::new(
            Array1::from_vec(vec![2.0, 4.0]),
            "b",
            AnchorCategory::Factual,
        ));

        let c = set.centroid();
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_synthetic_generation() {
        let set = AnchorSet::generate_synthetic(32, 100, 42);
        assert_eq!(set.len(), 100);
        assert_eq!(set.anchors[0].dim(), 32);
    }

    #[test]
    fn test_by_category() {
        let mut set = AnchorSet::new();
        set.add(SafetyAnchor::new(
            Array1::zeros(4),
            "f1",
            AnchorCategory::Factual,
        ));
        set.add(SafetyAnchor::new(
            Array1::zeros(4),
            "h1",
            AnchorCategory::Helpful,
        ));
        set.add(SafetyAnchor::new(
            Array1::zeros(4),
            "f2",
            AnchorCategory::Factual,
        ));

        let factual = set.by_category(&AnchorCategory::Factual);
        assert_eq!(factual.len(), 2);
    }

    #[test]
    fn test_convex_hull_projection() {
        let mut set = AnchorSet::new();
        set.add(SafetyAnchor::new(
            Array1::from_vec(vec![0.0, 0.0]),
            "a",
            AnchorCategory::Factual,
        ));
        set.add(SafetyAnchor::new(
            Array1::from_vec(vec![1.0, 0.0]),
            "b",
            AnchorCategory::Factual,
        ));

        // Project a faraway point
        let x = Array1::from_vec(vec![100.0, 0.0]);
        let proj = set.project_to_convex_hull(&x, 2);
        // Should be between the two anchors (weighted by inverse distance)
        assert!(proj[0] >= 0.0 && proj[0] <= 1.0);
    }
}
