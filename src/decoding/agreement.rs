//! Agreement computation for Multi-Perspective Decoding.
//!
//! Given logit distributions from 4 perspectives, compute:
//! - Per-token agreement (do all perspectives agree on the same token?)
//! - Jensen-Shannon Divergence (JSD) — measures distribution similarity
//! - Confidence scores for accepting/rejecting a token

use ndarray::Array1;

/// Agreement result for a single decoding step.
#[derive(Clone, Debug)]
pub struct AgreementResult {
    /// Whether all perspectives agree on the argmax token.
    pub unanimous: bool,

    /// The agreed-upon token (if unanimous), or the base perspective's choice.
    pub token_id: usize,

    /// Number of perspectives that chose the same token as the base.
    pub agreement_count: usize,

    /// Jensen-Shannon Divergence across perspectives (lower = more agreement).
    pub jsd: f64,

    /// Per-perspective argmax tokens.
    pub per_perspective_tokens: Vec<usize>,

    /// Confidence: 1 - JSD (higher = more confident).
    pub confidence: f64,
}

/// Softmax a logits vector into probabilities.
pub fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Array1<f32> = logits.mapv(|v| (v - max).exp());
    let sum: f32 = exp.sum();
    if sum > 0.0 {
        exp / sum
    } else {
        Array1::from_vec(vec![1.0 / logits.len() as f32; logits.len()])
    }
}

/// Argmax of a vector.
pub fn argmax(v: &Array1<f32>) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// KL divergence: KL(P || Q) = Σ P(x) · log(P(x) / Q(x)).
/// Assumes P and Q are valid probability distributions.
pub fn kl_divergence(p: &Array1<f32>, q: &Array1<f32>) -> f64 {
    let eps = 1e-10;
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let pi = pi as f64 + eps;
            let qi = qi as f64 + eps;
            pi * (pi / qi).ln()
        })
        .sum()
}

/// Jensen-Shannon Divergence: JSD(P1, P2, ..., Pk) = H(M) - (1/k)·Σ H(Pi)
/// where M = (1/k)·Σ Pi is the mixture distribution.
/// Returns a value in [0, ln(k)]. Normalized to [0, 1] by dividing by ln(k).
pub fn jensen_shannon_divergence(distributions: &[Array1<f32>]) -> f64 {
    let k = distributions.len();
    if k <= 1 {
        return 0.0;
    }

    let n = distributions[0].len();

    // Compute mixture M = average of all distributions
    let mut mixture = Array1::zeros(n);
    for dist in distributions {
        mixture = &mixture + dist;
    }
    mixture /= k as f32;

    // JSD = average KL divergence from each distribution to mixture
    let avg_kl: f64 = distributions
        .iter()
        .map(|dist| kl_divergence(dist, &mixture))
        .sum::<f64>()
        / k as f64;

    // Normalize by ln(k) to get [0, 1] range
    let max_jsd = (k as f64).ln();
    if max_jsd > 0.0 {
        avg_kl / max_jsd
    } else {
        0.0
    }
}

/// Compute agreement across multiple perspective logit outputs.
pub fn compute_agreement(
    perspective_logits: &[Array1<f32>],
) -> AgreementResult {
    assert!(!perspective_logits.is_empty());

    let k = perspective_logits.len();

    // Convert each to probability distribution
    let distributions: Vec<Array1<f32>> = perspective_logits
        .iter()
        .map(|logits| softmax(logits))
        .collect();

    // Get argmax per perspective
    let per_perspective_tokens: Vec<usize> = distributions
        .iter()
        .map(|probs| argmax(probs))
        .collect();

    // Base perspective's choice
    let base_token = per_perspective_tokens[0];

    // Count agreement with base
    let agreement_count = per_perspective_tokens
        .iter()
        .filter(|&&t| t == base_token)
        .count();

    let unanimous = agreement_count == k;

    // Compute JSD
    let jsd = jensen_shannon_divergence(&distributions);

    AgreementResult {
        unanimous,
        token_id: base_token,
        agreement_count,
        jsd,
        per_perspective_tokens,
        confidence: 1.0 - jsd,
    }
}

/// Acceptance policy: should we accept this token?
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AcceptancePolicy {
    /// Accept only if all perspectives agree (strictest).
    Unanimous,

    /// Accept if ≥ 3 out of 4 agree.
    Majority,

    /// Accept if JSD < threshold.
    JsdThreshold(f64),

    /// Always accept base perspective (fastest, no safety).
    AlwaysAccept,
}

/// Decide whether to accept a token based on agreement and policy.
pub fn should_accept(result: &AgreementResult, policy: AcceptancePolicy) -> bool {
    match policy {
        AcceptancePolicy::Unanimous => result.unanimous,
        AcceptancePolicy::Majority => result.agreement_count >= 3,
        AcceptancePolicy::JsdThreshold(threshold) => result.jsd < threshold,
        AcceptancePolicy::AlwaysAccept => true,
    }
}

/// When a token is rejected, what to do?
#[derive(Clone, Copy, Debug)]
pub enum RejectionStrategy {
    /// Resample with lower temperature.
    Resample { temperature: f32 },

    /// Fall back to the token with lowest JSD across perspectives.
    MinJsd,

    /// Emit a special "uncertain" token.
    UncertainToken(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);
        let sum: f32 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_argmax_basic() {
        let v = Array1::from_vec(vec![0.1, 0.7, 0.2]);
        assert_eq!(argmax(&v), 1);
    }

    #[test]
    fn test_kl_divergence_same() {
        let p = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-6);
    }

    #[test]
    fn test_jsd_identical() {
        let p = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let jsd = jensen_shannon_divergence(&[p.clone(), p.clone(), p.clone()]);
        assert!(jsd.abs() < 1e-6);
    }

    #[test]
    fn test_jsd_different() {
        let p = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let q = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
        let jsd = jensen_shannon_divergence(&[p, q]);
        assert!(jsd > 0.5, "jsd = {}", jsd);
    }

    #[test]
    fn test_agreement_unanimous() {
        let logits = Array1::from_vec(vec![-10.0, -10.0, 10.0, -10.0]);
        let all_logits = vec![logits.clone(), logits.clone(), logits.clone(), logits.clone()];
        let result = compute_agreement(&all_logits);
        assert!(result.unanimous);
        assert_eq!(result.token_id, 2);
        assert_eq!(result.agreement_count, 4);
        assert!(result.jsd < 0.01);
    }

    #[test]
    fn test_agreement_split() {
        let a = Array1::from_vec(vec![10.0, -10.0, -10.0]);
        let b = Array1::from_vec(vec![-10.0, 10.0, -10.0]);
        let result = compute_agreement(&[a.clone(), a.clone(), b.clone(), b.clone()]);
        assert!(!result.unanimous);
        assert_eq!(result.agreement_count, 2);
        assert!(result.jsd > 0.1);
    }

    #[test]
    fn test_acceptance_policy_unanimous() {
        let result = AgreementResult {
            unanimous: true,
            token_id: 5,
            agreement_count: 4,
            jsd: 0.01,
            per_perspective_tokens: vec![5, 5, 5, 5],
            confidence: 0.99,
        };
        assert!(should_accept(&result, AcceptancePolicy::Unanimous));
        assert!(should_accept(&result, AcceptancePolicy::Majority));
    }

    #[test]
    fn test_acceptance_policy_majority() {
        let result = AgreementResult {
            unanimous: false,
            token_id: 5,
            agreement_count: 3,
            jsd: 0.15,
            per_perspective_tokens: vec![5, 5, 5, 7],
            confidence: 0.85,
        };
        assert!(!should_accept(&result, AcceptancePolicy::Unanimous));
        assert!(should_accept(&result, AcceptancePolicy::Majority));
    }

    #[test]
    fn test_acceptance_jsd_threshold() {
        let result = AgreementResult {
            unanimous: false,
            token_id: 5,
            agreement_count: 2,
            jsd: 0.3,
            per_perspective_tokens: vec![5, 5, 7, 7],
            confidence: 0.7,
        };
        assert!(should_accept(&result, AcceptancePolicy::JsdThreshold(0.5)));
        assert!(!should_accept(&result, AcceptancePolicy::JsdThreshold(0.2)));
    }
}
