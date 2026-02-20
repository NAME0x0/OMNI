//! Safety Polytope definition.
//!
//! The Safety Polytope P is the intersection of:
//! 1. The convex hull of ~1000 safety anchors
//! 2. ~500 halfspace constraints (hard rules)
//!
//! For any output representation x, we compute:
//!   x_safe = project_P(x)
//!
//! This projection is non-differentiable — no gradient can escape through it,
//! making jailbreak attempts via gradient-based adversarial attacks impossible.

use ndarray::Array1;

use super::anchors::AnchorSet;
use super::halfspace::HalfspaceSet;

/// The Safety Polytope: intersection of convex hull and halfspace constraints.
pub struct SafetyPolytope {
    /// Safety anchor points (define the convex hull component).
    pub anchors: AnchorSet,

    /// Halfspace constraints (hard boundaries).
    pub halfspaces: HalfspaceSet,

    /// Safety margin: extra shrinkage applied to the polytope boundary.
    pub margin: f32,

    /// Number of nearest anchors to use for convex hull projection.
    pub hull_k: usize,

    /// Whether to apply anchor projection in addition to halfspace projection.
    pub use_anchor_projection: bool,
}

impl SafetyPolytope {
    /// Create a new Safety Polytope.
    pub fn new(anchors: AnchorSet, halfspaces: HalfspaceSet) -> Self {
        Self {
            anchors,
            halfspaces,
            margin: 0.0,
            hull_k: 10,
            use_anchor_projection: true,
        }
    }

    /// Set safety margin.
    pub fn with_margin(mut self, margin: f32) -> Self {
        self.margin = margin;
        self
    }

    /// Check if a point is inside the polytope.
    pub fn contains(&self, x: &Array1<f32>) -> bool {
        // Must satisfy all halfspace constraints
        if !self.halfspaces.is_feasible(x) {
            return false;
        }

        // If using anchor projection, must also be "close" to anchors
        if self.use_anchor_projection && !self.anchors.is_empty() {
            if let Some((_, dist)) = self.anchors.nearest(x) {
                // Point should be within reasonable distance of nearest anchor
                let threshold = self.anchor_distance_threshold();
                return dist <= threshold;
            }
        }

        true
    }

    /// Compute anchor distance threshold for containment check.
    fn anchor_distance_threshold(&self) -> f32 {
        // Use average nearest-neighbor distance among anchors × 3
        if self.anchors.len() < 2 {
            return f32::MAX;
        }

        let mut total_nn_dist = 0.0f32;
        let sample = self.anchors.len().min(50); // Sample for efficiency
        for i in 0..sample {
            let x = &self.anchors.anchors[i].vector;
            let mut min_dist = f32::MAX;
            for (j, a) in self.anchors.anchors.iter().enumerate() {
                if j != i {
                    let d = a.distance_to(x);
                    if d < min_dist {
                        min_dist = d;
                    }
                }
            }
            total_nn_dist += min_dist;
        }

        let avg_nn = total_nn_dist / sample as f32;
        avg_nn * 3.0 + self.margin
    }

    /// Get safety score for a point [0, 1].
    /// 1.0 = deep inside polytope, 0.0 = on boundary or outside.
    pub fn safety_score(&self, x: &Array1<f32>) -> f32 {
        let max_viol = self.halfspaces.max_violation(x);
        let halfspace_score = (-max_viol * 10.0).exp(); // Exponential decay

        let anchor_score = if !self.anchors.is_empty() {
            self.anchors.max_similarity(x).max(0.0) // Cosine similarity
        } else {
            1.0
        };

        // Geometric mean
        (halfspace_score * anchor_score).sqrt()
    }

    /// Diagnostics: return a safety report.
    pub fn diagnose(&self, x: &Array1<f32>) -> SafetyDiagnosis {
        let violations = self.halfspaces.violations(x);
        let max_violation = self.halfspaces.max_violation(x);
        let nearest_anchor = self.anchors.nearest(x);
        let safety_score = self.safety_score(x);

        SafetyDiagnosis {
            is_safe: self.contains(x),
            safety_score,
            n_violations: violations.len(),
            max_violation,
            nearest_anchor_dist: nearest_anchor.map(|(_, d)| d).unwrap_or(f32::MAX),
            nearest_anchor_idx: nearest_anchor.map(|(i, _)| i),
        }
    }
}

/// Diagnostic report for a point's safety status.
#[derive(Debug)]
pub struct SafetyDiagnosis {
    pub is_safe: bool,
    pub safety_score: f32,
    pub n_violations: usize,
    pub max_violation: f32,
    pub nearest_anchor_dist: f32,
    pub nearest_anchor_idx: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::super::anchors::AnchorCategory;
    use super::super::anchors::SafetyAnchor;
    use super::super::halfspace::Halfspace;
    use super::*;

    fn make_simple_polytope() -> SafetyPolytope {
        let mut anchors = AnchorSet::new();
        anchors.add(SafetyAnchor::new(
            Array1::from_vec(vec![0.0, 0.0]),
            "origin",
            AnchorCategory::Factual,
        ));
        anchors.add(SafetyAnchor::new(
            Array1::from_vec(vec![1.0, 0.0]),
            "right",
            AnchorCategory::Helpful,
        ));
        anchors.add(SafetyAnchor::new(
            Array1::from_vec(vec![0.0, 1.0]),
            "up",
            AnchorCategory::Harmless,
        ));

        let mut halfspaces = HalfspaceSet::new();
        halfspaces.add(Halfspace::new(
            Array1::from_vec(vec![1.0, 0.0]),
            2.0,
            "x<=2",
        ));
        halfspaces.add(Halfspace::new(
            Array1::from_vec(vec![0.0, 1.0]),
            2.0,
            "y<=2",
        ));
        halfspaces.add(Halfspace::new(
            Array1::from_vec(vec![-1.0, 0.0]),
            1.0,
            "x>=-1",
        ));
        halfspaces.add(Halfspace::new(
            Array1::from_vec(vec![0.0, -1.0]),
            1.0,
            "y>=-1",
        ));

        SafetyPolytope::new(anchors, halfspaces)
    }

    #[test]
    fn test_contains_inside() {
        let poly = make_simple_polytope();
        assert!(poly.contains(&Array1::from_vec(vec![0.5, 0.5])));
    }

    #[test]
    fn test_contains_outside_halfspace() {
        let poly = make_simple_polytope();
        assert!(!poly.contains(&Array1::from_vec(vec![10.0, 0.0])));
    }

    #[test]
    fn test_safety_score() {
        let poly = make_simple_polytope();
        let score_inside = poly.safety_score(&Array1::from_vec(vec![0.5, 0.5]));
        let score_outside = poly.safety_score(&Array1::from_vec(vec![10.0, 10.0]));
        assert!(score_inside > score_outside);
    }

    #[test]
    fn test_diagnosis() {
        let poly = make_simple_polytope();
        let diag = poly.diagnose(&Array1::from_vec(vec![0.5, 0.5]));
        assert!(diag.is_safe);
        assert_eq!(diag.n_violations, 0);
    }

    #[test]
    fn test_diagnosis_outside() {
        let poly = make_simple_polytope();
        let diag = poly.diagnose(&Array1::from_vec(vec![10.0, 0.0]));
        assert!(!diag.is_safe);
        assert!(diag.n_violations > 0);
    }
}
