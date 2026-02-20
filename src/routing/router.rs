//! Expert router — projects hidden states onto the torus and selects top-1 expert.
//!
//! The router maps each token's hidden state h ∈ R^{d_model} to a point on the
//! torus via a learned linear projection W_route ∈ R^{2 × d_model}, then selects
//! the nearest expert on the manifold.
//!
//! Total router overhead: 2 × 4096 = 8,192 parameters per layer × 80 layers
//! = 655,360 parameters (~0.00006% of total model).

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::config::{D_MODEL, N_EXPERTS, N_LAYERS};
use crate::routing::manifold::{ExpertManifold, TorusPoint};

/// Routing weights for a single layer.
#[derive(Clone, Serialize, Deserialize)]
pub struct LayerRouter {
    /// Projection: W_route ∈ R^{2 × d_model}
    pub w_route: Array2<f32>,

    /// Bias: b_route ∈ R^{2}
    pub b_route: Array1<f32>,
}

impl LayerRouter {
    pub fn zeros() -> Self {
        Self {
            w_route: Array2::zeros((2, D_MODEL)),
            b_route: Array1::zeros(2),
        }
    }

    /// Project hidden state to a point on the torus.
    pub fn project(&self, h: &Array1<f32>) -> TorusPoint {
        let raw = self.w_route.dot(h) + &self.b_route;
        // Sigmoid to map to [0, 1) then onto torus
        let x = sigmoid(raw[0]);
        let y = sigmoid(raw[1]);
        TorusPoint::new(x, y)
    }

    /// Route: project hidden state to torus, find nearest expert.
    pub fn route(&self, h: &Array1<f32>, manifold: &ExpertManifold) -> RoutingDecision {
        let point = self.project(h);
        let expert_id = manifold.nearest_expert(&point);
        let distance = point.distance(&manifold.positions[expert_id]);

        RoutingDecision {
            expert_id,
            torus_point: point,
            distance,
        }
    }

    pub fn param_count(&self) -> usize {
        2 * D_MODEL + 2
    }
}

/// The result of routing a single token.
#[derive(Clone, Debug)]
pub struct RoutingDecision {
    /// Selected expert index (0..127).
    pub expert_id: usize,

    /// The token's position on the torus.
    pub torus_point: TorusPoint,

    /// Distance from the token's point to the selected expert.
    pub distance: f32,
}

/// Full router for all 80 layers.
#[derive(Clone, Serialize, Deserialize)]
pub struct ModelRouter {
    /// Per-layer routing weights.
    pub layers: Vec<LayerRouter>,

    /// The expert manifold (shared across all layers).
    pub manifold: ExpertManifold,
}

impl ModelRouter {
    /// Create with zero weights and default grid manifold.
    pub fn zeros() -> Self {
        Self {
            layers: (0..N_LAYERS).map(|_| LayerRouter::zeros()).collect(),
            manifold: ExpertManifold::default_grid(),
        }
    }

    /// Route a token through a specific layer.
    pub fn route(&self, layer: usize, h: &Array1<f32>) -> RoutingDecision {
        self.layers[layer].route(h, &self.manifold)
    }

    /// Route a token through all layers (used for prefetch planning).
    /// Returns the predicted expert for each layer.
    pub fn route_all_layers(&self, h: &Array1<f32>) -> Vec<RoutingDecision> {
        self.layers
            .iter()
            .map(|router| router.route(h, &self.manifold))
            .collect()
    }

    /// Predict the expert for layer `target_layer` using the hidden state
    /// from `current_layer`.  Used for prefetch-ahead.
    pub fn predict_ahead(
        &self,
        current_h: &Array1<f32>,
        target_layer: usize,
    ) -> RoutingDecision {
        self.layers[target_layer].route(current_h, &self.manifold)
    }

    /// Total parameter count for all routers.
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|r| r.param_count()).sum()
    }

    /// Compute load balance statistics across a batch of routing decisions.
    pub fn load_balance_stats(decisions: &[RoutingDecision]) -> LoadBalanceStats {
        let mut counts = vec![0u64; N_EXPERTS];
        for d in decisions {
            counts[d.expert_id] += 1;
        }

        let total = decisions.len() as f64;
        let mean = total / N_EXPERTS as f64;

        let variance: f64 = counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / N_EXPERTS as f64;

        let max_count = *counts.iter().max().unwrap_or(&0);
        let min_count = *counts.iter().min().unwrap_or(&0);

        // Gini coefficient
        let sorted: Vec<f64> = {
            let mut s: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s
        };
        let n = sorted.len() as f64;
        let gini = if total > 0.0 {
            let numerator: f64 = sorted
                .iter()
                .enumerate()
                .map(|(i, &x)| (2.0 * (i as f64 + 1.0) - n - 1.0) * x)
                .sum();
            numerator / (n * total)
        } else {
            0.0
        };

        LoadBalanceStats {
            counts,
            mean,
            variance,
            max_count,
            min_count,
            gini,
        }
    }
}

/// Load balance statistics for routing decisions.
#[derive(Debug)]
pub struct LoadBalanceStats {
    pub counts: Vec<u64>,
    pub mean: f64,
    pub variance: f64,
    pub max_count: u64,
    pub min_count: u64,
    pub gini: f64,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_produces_valid_expert() {
        let router = ModelRouter::zeros();
        let h = Array1::from_vec(vec![0.1; D_MODEL]);
        let decision = router.route(0, &h);
        assert!(decision.expert_id < N_EXPERTS);
    }

    #[test]
    fn test_router_deterministic() {
        let router = ModelRouter::zeros();
        let h = Array1::from_vec(vec![0.5; D_MODEL]);
        let d1 = router.route(0, &h);
        let d2 = router.route(0, &h);
        assert_eq!(d1.expert_id, d2.expert_id);
    }

    #[test]
    fn test_route_all_layers() {
        let router = ModelRouter::zeros();
        let h = Array1::from_vec(vec![0.1; D_MODEL]);
        let decisions = router.route_all_layers(&h);
        assert_eq!(decisions.len(), N_LAYERS);
    }

    #[test]
    fn test_layer_router_param_count() {
        let r = LayerRouter::zeros();
        assert_eq!(r.param_count(), 2 * D_MODEL + 2);
    }

    #[test]
    fn test_load_balance() {
        let decisions: Vec<RoutingDecision> = (0..1000)
            .map(|i| RoutingDecision {
                expert_id: i % N_EXPERTS,
                torus_point: TorusPoint::new(0.0, 0.0),
                distance: 0.0,
            })
            .collect();
        let stats = ModelRouter::load_balance_stats(&decisions);
        // Nearly balanced: gini should be very small
        assert!(stats.gini < 0.05, "gini = {}", stats.gini);
        assert_eq!(stats.max_count - stats.min_count, 1); // 1000 / 128 ≈ 7.8
    }
}
