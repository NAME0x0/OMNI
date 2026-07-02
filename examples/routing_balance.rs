//! Manifold routing load-balance measurement.
//!
//! docs/v2/12_validation_v2.md sets two routing targets that were previously
//! asserted without measurement:
//!   - Load balance: Gini coefficient < 0.15 over 10K random tokens
//!   - Coverage: all 128 experts reachable
//!
//! This example measures both against the actual `routing::` implementation,
//! using seeded random Gaussian router weights (the crate has no trained
//! weights yet, so this characterises the *initialisation regime*, not a
//! trained router). Two initialisation scales are reported:
//!   - std = 1/sqrt(d_model): standard fan-in init. Projected coordinates
//!     concentrate near the bias before `rem_euclid(1.0)` wrapping.
//!   - std = 1.0: large-scale init. Coordinates wrap many times around the
//!     torus, approximating a uniform query distribution.

use ndarray::{Array1, Array2};
use perspective::config::{D_MODEL, N_EXPERTS};
use perspective::routing::manifold::ExpertManifold;
use perspective::routing::router::LayerRouter;

const N_TOKENS: usize = 10_000;

/// xorshift64* PRNG — deterministic, dependency-free.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box–Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn random_router(rng: &mut Rng, std: f64) -> LayerRouter {
    let mut router = LayerRouter::zeros();
    router.w_route = Array2::from_shape_fn((3, D_MODEL), |_| (rng.normal() * std) as f32);
    router.b_route = Array1::from_shape_fn(3, |_| (rng.normal() * std) as f32);
    router
}

fn random_hidden(rng: &mut Rng) -> Array1<f32> {
    Array1::from_shape_fn(D_MODEL, |_| rng.normal() as f32)
}

/// Gini coefficient of a count distribution (0 = perfectly even).
fn gini(counts: &[usize]) -> f64 {
    let n = counts.len() as f64;
    let mut sorted: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total: f64 = sorted.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    let weighted: f64 = sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as f64 + 1.0) * x)
        .sum();
    (2.0 * weighted) / (n * total) - (n + 1.0) / n
}

fn measure(init_std: f64, seed: u64) -> (f64, usize, Vec<(usize, usize)>) {
    let mut rng = Rng::new(seed);
    let router = random_router(&mut rng, init_std);
    let manifold = ExpertManifold::default_grid();

    let mut counts = vec![0usize; N_EXPERTS];
    for _ in 0..N_TOKENS {
        let h = random_hidden(&mut rng);
        let decision = router.route(&h, &manifold);
        counts[decision.expert_id] += 1;
    }

    let g = gini(&counts);
    let unreached = counts.iter().filter(|&&c| c == 0).count();

    let mut ranked: Vec<(usize, usize)> = counts.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    ranked.truncate(5);

    (g, unreached, ranked)
}

fn main() {
    println!("# Manifold Routing Load Balance — measured\n");
    println!(
        "{} random Gaussian hidden states (d={}), untrained router, {} experts.\n",
        N_TOKENS, D_MODEL, N_EXPERTS
    );
    println!("| Init std | Gini | Unreached experts | Top-5 hottest (id: count) |");
    println!("|---|---|---|---|");

    for (label, std) in [
        ("1/sqrt(d) ≈ 0.0156", 1.0 / (D_MODEL as f64).sqrt()),
        ("1.0", 1.0),
    ] {
        let (g, unreached, top) = measure(std, 42);
        let top_str = top
            .iter()
            .map(|(id, c)| format!("{}: {}", id, c))
            .collect::<Vec<_>>()
            .join(", ");
        println!("| {} | {:.4} | {} / {} | {} |", label, g, unreached, N_EXPERTS, top_str);
    }

    println!("\nTargets (docs/v2/12_validation_v2.md): Gini < 0.15; all experts reachable.");
    println!("Note: untrained-router characterisation only; trained routing may differ.");
}
