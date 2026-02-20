//! Halfspace constraints for the Safety Polytope.
//!
//! Each halfspace is defined as: a · x ≤ b
//! where a ∈ R^d is the normal vector and b ∈ R is the offset.
//! These encode hard safety rules that cannot be violated.

use ndarray::Array1;

/// A single halfspace constraint: a · x ≤ b.
#[derive(Clone, Debug)]
pub struct Halfspace {
    /// Normal vector (unit-normalised).
    pub normal: Array1<f32>,

    /// Offset (upper bound of the dot product).
    pub offset: f32,

    /// Human-readable description of what this constraint encodes.
    pub label: String,

    /// Priority level (higher = more important, projected later in Dykstra's).
    pub priority: u32,
}

impl Halfspace {
    /// Create a new halfspace, normalising the normal vector.
    pub fn new(normal: Array1<f32>, offset: f32, label: &str) -> Self {
        let norm: f32 = normal.iter().map(|v| v * v).sum::<f32>().sqrt();
        let unit_normal = if norm > 1e-8 {
            &normal / norm
        } else {
            normal.clone()
        };
        let adjusted_offset = if norm > 1e-8 { offset / norm } else { offset };

        Self {
            normal: unit_normal,
            offset: adjusted_offset,
            label: label.to_string(),
            priority: 0,
        }
    }

    /// Create with explicit priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Check if a point satisfies this constraint.
    pub fn contains(&self, x: &Array1<f32>) -> bool {
        self.normal.dot(x) <= self.offset + 1e-6 // small tolerance
    }

    /// Compute the signed distance from x to this halfspace boundary.
    /// Negative means inside, positive means outside.
    pub fn signed_distance(&self, x: &Array1<f32>) -> f32 {
        self.normal.dot(x) - self.offset
    }

    /// Project a point onto this halfspace (closest point inside).
    /// If already inside, returns the point unchanged.
    pub fn project(&self, x: &Array1<f32>) -> Array1<f32> {
        let dist = self.signed_distance(x);
        if dist <= 0.0 {
            x.clone()
        } else {
            x - &(&self.normal * dist)
        }
    }

    /// Violation amount (0 if satisfied, positive if violated).
    pub fn violation(&self, x: &Array1<f32>) -> f32 {
        self.signed_distance(x).max(0.0)
    }
}

/// A set of halfspace constraints.
pub struct HalfspaceSet {
    /// All halfspace constraints.
    pub constraints: Vec<Halfspace>,
}

impl HalfspaceSet {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Add a constraint.
    pub fn add(&mut self, h: Halfspace) {
        self.constraints.push(h);
    }

    /// Check if a point satisfies all constraints.
    pub fn is_feasible(&self, x: &Array1<f32>) -> bool {
        self.constraints.iter().all(|h| h.contains(x))
    }

    /// Return list of violated constraints and their violations.
    pub fn violations(&self, x: &Array1<f32>) -> Vec<(usize, f32)> {
        self.constraints
            .iter()
            .enumerate()
            .filter_map(|(i, h)| {
                let v = h.violation(x);
                if v > 1e-6 {
                    Some((i, v))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get constraints sorted by priority (ascending, so high-priority last).
    /// In Dykstra's algorithm, projecting last = having the final say.
    pub fn sorted_by_priority(&self) -> Vec<&Halfspace> {
        let mut sorted: Vec<&Halfspace> = self.constraints.iter().collect();
        sorted.sort_by_key(|h| h.priority);
        sorted
    }

    /// Total number of constraints.
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Is the constraint set empty?
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Maximum violation across all constraints.
    pub fn max_violation(&self, x: &Array1<f32>) -> f32 {
        self.constraints
            .iter()
            .map(|h| h.violation(x))
            .fold(0.0f32, f32::max)
    }

    /// Generate default safety halfspaces for a given dimension.
    /// These encode basic safety rules in representation space.
    pub fn default_safety_set(dim: usize) -> Self {
        let mut set = Self::new();

        // Constraint 1: Bound the L∞ norm (no extreme activations).
        // Each dimension: x_i ≤ MAX_ACT and -x_i ≤ MAX_ACT
        let max_activation = 50.0;
        for i in 0..dim.min(50) {
            // Positive bound
            let mut normal_pos = Array1::zeros(dim);
            normal_pos[i] = 1.0;
            set.add(
                Halfspace::new(
                    normal_pos,
                    max_activation,
                    &format!("activation_upper_{}", i),
                )
                .with_priority(1),
            );

            // Negative bound
            let mut normal_neg = Array1::zeros(dim);
            normal_neg[i] = -1.0;
            set.add(
                Halfspace::new(
                    normal_neg,
                    max_activation,
                    &format!("activation_lower_{}", i),
                )
                .with_priority(1),
            );
        }

        // Constraint 2: Bound the L2 norm (total energy constraint).
        // ||x||_2 ≤ R is not a halfspace, but we can approximate with
        // multiple halfspaces (polytope approximation of the L2 ball).
        // Use a few random directions.
        let l2_bound = max_activation * (dim as f32).sqrt();
        let mut state = 0x5AFE_1234u64;
        for k in 0..20 {
            let normal: Vec<f32> = (0..dim)
                .map(|_| {
                    state = xorshift64(state);
                    let u = (state as f32) / (u64::MAX as f32) * 2.0 - 1.0;
                    u
                })
                .collect();
            let norm: f32 = normal.iter().map(|v| v * v).sum::<f32>().sqrt();
            let unit_normal: Vec<f32> =
                normal.iter().map(|v| v / (norm + 1e-8)).collect();

            set.add(
                Halfspace::new(
                    Array1::from_vec(unit_normal),
                    l2_bound,
                    &format!("l2_approx_{}", k),
                )
                .with_priority(2),
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
    fn test_halfspace_contains() {
        let h = Halfspace::new(
            Array1::from_vec(vec![1.0, 0.0]),
            2.0,
            "x1<=2",
        );
        assert!(h.contains(&Array1::from_vec(vec![1.0, 5.0])));
        assert!(h.contains(&Array1::from_vec(vec![2.0, 0.0])));
        assert!(!h.contains(&Array1::from_vec(vec![3.0, 0.0])));
    }

    #[test]
    fn test_halfspace_project() {
        let h = Halfspace::new(
            Array1::from_vec(vec![1.0, 0.0]),
            2.0,
            "x1<=2",
        );
        // Point inside → unchanged
        let p = h.project(&Array1::from_vec(vec![1.0, 3.0]));
        assert!((p[0] - 1.0).abs() < 1e-6);
        assert!((p[1] - 3.0).abs() < 1e-6);

        // Point outside → projected
        let p2 = h.project(&Array1::from_vec(vec![5.0, 3.0]));
        assert!((p2[0] - 2.0).abs() < 1e-5);
        assert!((p2[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_halfspace_set_feasible() {
        let mut set = HalfspaceSet::new();
        set.add(Halfspace::new(
            Array1::from_vec(vec![1.0, 0.0]),
            2.0,
            "x1<=2",
        ));
        set.add(Halfspace::new(
            Array1::from_vec(vec![0.0, 1.0]),
            3.0,
            "x2<=3",
        ));

        assert!(set.is_feasible(&Array1::from_vec(vec![1.0, 2.0])));
        assert!(!set.is_feasible(&Array1::from_vec(vec![3.0, 2.0])));
    }

    #[test]
    fn test_violations() {
        let mut set = HalfspaceSet::new();
        set.add(Halfspace::new(
            Array1::from_vec(vec![1.0, 0.0]),
            1.0,
            "x1<=1",
        ));
        set.add(Halfspace::new(
            Array1::from_vec(vec![0.0, 1.0]),
            1.0,
            "x2<=1",
        ));

        let v = set.violations(&Array1::from_vec(vec![2.0, 0.5]));
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].0, 0); // First constraint violated
    }

    #[test]
    fn test_signed_distance() {
        let h = Halfspace::new(
            Array1::from_vec(vec![0.0, 1.0]),
            1.0,
            "x2<=1",
        );
        assert!(h.signed_distance(&Array1::from_vec(vec![0.0, 0.5])) < 0.0);
        assert!(h.signed_distance(&Array1::from_vec(vec![0.0, 2.0])) > 0.0);
    }

    #[test]
    fn test_priority_sorting() {
        let mut set = HalfspaceSet::new();
        set.add(
            Halfspace::new(Array1::from_vec(vec![1.0]), 1.0, "low").with_priority(1),
        );
        set.add(
            Halfspace::new(Array1::from_vec(vec![1.0]), 1.0, "high").with_priority(10),
        );
        set.add(
            Halfspace::new(Array1::from_vec(vec![1.0]), 1.0, "mid").with_priority(5),
        );

        let sorted = set.sorted_by_priority();
        assert_eq!(sorted[0].priority, 1);
        assert_eq!(sorted[1].priority, 5);
        assert_eq!(sorted[2].priority, 10);
    }
}
