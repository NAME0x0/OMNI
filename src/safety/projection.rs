//! Dykstra's Alternating Projection onto the Safety Polytope.
//!
//! Projects any point x ∈ R^d onto the intersection of convex sets
//! (halfspaces + approximate convex hull) using Dykstra's algorithm.
//!
//! Key property: THIS IS NON-DIFFERENTIABLE.
//! No gradient flows through the projection, making it immune to
//! gradient-based adversarial attacks. The safety boundary is a hard wall.

use ndarray::Array1;

use super::halfspace::HalfspaceSet;
use super::polytope::SafetyPolytope;

/// Configuration for the projection algorithm.
#[derive(Clone, Debug)]
pub struct ProjectionConfig {
    /// Maximum number of Dykstra iterations.
    pub max_iterations: usize,

    /// Convergence tolerance (L2 change between iterations).
    pub tolerance: f32,

    /// Whether to use priority ordering (higher priority projected last).
    pub priority_order: bool,

    /// Whether to also project onto the convex hull of anchors.
    pub project_to_hull: bool,

    /// Number of nearest anchors for hull projection.
    pub hull_k: usize,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            priority_order: true,
            project_to_hull: true,
            hull_k: 10,
        }
    }
}

/// Result of a projection operation.
#[derive(Debug)]
pub struct ProjectionResult {
    /// The projected point.
    pub point: Array1<f32>,

    /// Number of iterations taken.
    pub iterations: usize,

    /// Whether convergence was reached.
    pub converged: bool,

    /// Final residual (L2 change in last iteration).
    pub residual: f32,

    /// Distance moved from original point.
    pub distance_moved: f32,

    /// Number of active constraints at the solution.
    pub active_constraints: usize,
}

/// Project a point onto a single halfspace: argmin ||y - x|| s.t. a·y ≤ b.
/// If a·x ≤ b, returns x. Otherwise, returns x - ((a·x - b) / ||a||²) · a.
fn project_halfspace(x: &Array1<f32>, normal: &Array1<f32>, offset: f32) -> Array1<f32> {
    let dot = normal.dot(x);
    if dot <= offset {
        x.clone()
    } else {
        let norm_sq: f32 = normal.iter().map(|v| v * v).sum();
        let shift = (dot - offset) / (norm_sq + 1e-10);
        x - &(normal * shift)
    }
}

/// Dykstra's alternating projection algorithm.
///
/// Projects x onto the intersection of convex sets C_1 ∩ C_2 ∩ ... ∩ C_m.
/// The algorithm maintains incremental corrections to ensure convergence
/// to the true projection (unlike simple cyclic projection which may not).
///
/// Algorithm:
///   Initialise: y_0 = x, p_i = 0 for all i
///   Repeat:
///     For each constraint i:
///       y = project_C_i(y + p_i)    // project with correction
///       p_i = (y_prev + p_i) - y    // update correction
///     Until ||y_new - y_old|| < tolerance
pub fn dykstra_project(
    x: &Array1<f32>,
    halfspaces: &HalfspaceSet,
    config: &ProjectionConfig,
) -> ProjectionResult {
    let d = x.len();
    let m = halfspaces.len();

    if m == 0 {
        return ProjectionResult {
            point: x.clone(),
            iterations: 0,
            converged: true,
            residual: 0.0,
            distance_moved: 0.0,
            active_constraints: 0,
        };
    }

    // Get constraints in priority order if requested
    let ordered: Vec<usize> = if config.priority_order {
        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by_key(|&i| halfspaces.constraints[i].priority);
        indices
    } else {
        (0..m).collect()
    };

    let mut y = x.clone();
    let mut corrections: Vec<Array1<f32>> = vec![Array1::zeros(d); m];
    let mut iterations = 0;
    let mut residual = f32::MAX;

    for iter in 0..config.max_iterations {
        let y_old = y.clone();

        for &i in &ordered {
            let h = &halfspaces.constraints[i];

            // Add correction
            let y_corrected = &y + &corrections[i];

            // Project onto halfspace
            let y_new = project_halfspace(&y_corrected, &h.normal, h.offset);

            // Update correction (Dykstra's key step)
            corrections[i] = &y_corrected - &y_new;

            y = y_new;
        }

        // Check convergence
        let diff = &y - &y_old;
        residual = diff.iter().map(|v| v * v).sum::<f32>().sqrt();
        iterations = iter + 1;

        if residual < config.tolerance {
            break;
        }
    }

    // Count active constraints
    let active = halfspaces
        .constraints
        .iter()
        .filter(|h| {
            let d = h.signed_distance(&y);
            d.abs() < 1e-4 // On the boundary
        })
        .count();

    let distance_moved = {
        let diff = &y - x;
        diff.iter().map(|v| v * v).sum::<f32>().sqrt()
    };

    ProjectionResult {
        point: y,
        iterations,
        converged: residual < config.tolerance,
        residual,
        distance_moved,
        active_constraints: active,
    }
}

/// Full Safety Polytope projection.
/// Combines halfspace projection with optional anchor hull projection.
pub fn project_onto_polytope(
    x: &Array1<f32>,
    polytope: &SafetyPolytope,
    config: &ProjectionConfig,
) -> ProjectionResult {
    // Step 1: Dykstra's projection onto halfspaces
    let mut result = dykstra_project(x, &polytope.halfspaces, config);

    // Step 2: Optional projection toward anchor convex hull
    if config.project_to_hull && !polytope.anchors.is_empty() {
        let hull_proj = polytope
            .anchors
            .project_to_convex_hull(&result.point, config.hull_k);

        // Blend: move toward hull projection if far from anchors
        if let Some((_, anchor_dist)) = polytope.anchors.nearest(&result.point) {
            // If far from nearest anchor, pull toward hull
            let blend_factor = (anchor_dist / 10.0).min(1.0) * 0.5;
            result.point = &result.point * (1.0 - blend_factor) + &hull_proj * blend_factor;
        }
    }

    result
}

/// Simple (non-Dykstra) cyclic projection for comparison.
/// Less accurate but faster — good for rough enforcement.
pub fn cyclic_project(x: &Array1<f32>, halfspaces: &HalfspaceSet, max_iters: usize) -> Array1<f32> {
    let mut y = x.clone();
    for _ in 0..max_iters {
        for h in &halfspaces.constraints {
            y = project_halfspace(&y, &h.normal, h.offset);
        }
    }
    y
}

/// Project onto a single L∞ box constraint:
/// Each dimension i: lo_i ≤ x_i ≤ hi_i.
pub fn project_box(x: &Array1<f32>, lo: f32, hi: f32) -> Array1<f32> {
    x.mapv(|v| v.clamp(lo, hi))
}

#[cfg(test)]
mod tests {
    use super::super::halfspace::Halfspace;
    use super::*;

    fn make_box_constraints(dim: usize, bound: f32) -> HalfspaceSet {
        let mut set = HalfspaceSet::new();
        for i in 0..dim {
            let mut n_pos = Array1::zeros(dim);
            n_pos[i] = 1.0;
            set.add(Halfspace::new(n_pos, bound, &format!("+{}", i)));

            let mut n_neg = Array1::zeros(dim);
            n_neg[i] = -1.0;
            set.add(Halfspace::new(n_neg, bound, &format!("-{}", i)));
        }
        set
    }

    #[test]
    fn test_project_halfspace_inside() {
        let x = Array1::from_vec(vec![1.0, 0.0]);
        let n = Array1::from_vec(vec![1.0, 0.0]);
        let p = project_halfspace(&x, &n, 2.0);
        assert!((p[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_project_halfspace_outside() {
        let x = Array1::from_vec(vec![5.0, 3.0]);
        let n = Array1::from_vec(vec![1.0, 0.0]);
        let p = project_halfspace(&x, &n, 2.0);
        assert!((p[0] - 2.0).abs() < 1e-6);
        assert!((p[1] - 3.0).abs() < 1e-6); // y unchanged
    }

    #[test]
    fn test_dykstra_box() {
        let constraints = make_box_constraints(2, 1.0);
        let config = ProjectionConfig::default();

        let x = Array1::from_vec(vec![5.0, -3.0]);
        let result = dykstra_project(&x, &constraints, &config);

        assert!(result.converged);
        assert!(
            (result.point[0] - 1.0).abs() < 1e-3,
            "x0 = {}",
            result.point[0]
        );
        assert!(
            (result.point[1] + 1.0).abs() < 1e-3,
            "x1 = {}",
            result.point[1]
        );
    }

    #[test]
    fn test_dykstra_already_feasible() {
        let constraints = make_box_constraints(3, 10.0);
        let config = ProjectionConfig::default();

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = dykstra_project(&x, &constraints, &config);

        assert!(result.converged);
        assert!(result.distance_moved < 1e-5);
    }

    #[test]
    fn test_dykstra_convergence() {
        // More complex: intersection of diagonal halfspaces
        let mut set = HalfspaceSet::new();
        set.add(Halfspace::new(
            Array1::from_vec(vec![1.0, 1.0]),
            2.0,
            "x+y<=2",
        ));
        set.add(Halfspace::new(
            Array1::from_vec(vec![1.0, -1.0]),
            2.0,
            "x-y<=2",
        ));
        set.add(Halfspace::new(
            Array1::from_vec(vec![-1.0, 0.0]),
            0.0,
            "x>=0",
        ));
        set.add(Halfspace::new(
            Array1::from_vec(vec![0.0, -1.0]),
            0.0,
            "y>=0",
        ));

        let config = ProjectionConfig::default();
        let x = Array1::from_vec(vec![10.0, 10.0]);
        let result = dykstra_project(&x, &set, &config);

        assert!(
            result.converged,
            "did not converge after {} iters (residual={})",
            result.iterations, result.residual
        );
        // Should be inside all constraints
        assert!(set.is_feasible(&result.point));
    }

    #[test]
    fn test_cyclic_vs_dykstra() {
        let constraints = make_box_constraints(2, 1.0);
        let x = Array1::from_vec(vec![5.0, 5.0]);

        let config = ProjectionConfig::default();
        let dykstra_result = dykstra_project(&x, &constraints, &config);
        let cyclic_result = cyclic_project(&x, &constraints, 50);

        // Both should give approximately [1, 1]
        assert!((dykstra_result.point[0] - 1.0).abs() < 1e-2);
        assert!((cyclic_result[0] - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_project_box() {
        let x = Array1::from_vec(vec![-5.0, 0.5, 10.0]);
        let p = project_box(&x, -1.0, 1.0);
        assert!((p[0] + 1.0).abs() < 1e-6);
        assert!((p[1] - 0.5).abs() < 1e-6);
        assert!((p[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_constraints() {
        let set = HalfspaceSet::new();
        let config = ProjectionConfig::default();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = dykstra_project(&x, &set, &config);
        assert!(result.converged);
        assert!(result.distance_moved < 1e-10);
    }
}
