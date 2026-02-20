//! Health monitoring and diagnostics for the PERSPECTIVE runtime.
//!
//! Provides real-time metrics, anomaly detection, and self-healing triggers.

use std::collections::VecDeque;

/// Health status levels.
#[derive(Clone, Debug, PartialEq)]
pub enum HealthStatus {
    /// All systems nominal.
    Healthy,
    /// Some metrics are degraded but within tolerance.
    Degraded(String),
    /// Critical issue requiring attention.
    Critical(String),
}

/// A health metric sample.
#[derive(Clone, Debug)]
pub struct MetricSample {
    pub timestamp_ms: u64,
    pub value: f32,
}

/// A rolling metric tracker.
pub struct MetricTracker {
    /// Metric name.
    pub name: String,

    /// Rolling window of samples.
    pub samples: VecDeque<MetricSample>,

    /// Maximum window size.
    pub max_samples: usize,

    /// Warning threshold (upper bound).
    pub warn_threshold: Option<f32>,

    /// Critical threshold (upper bound).
    pub critical_threshold: Option<f32>,
}

impl MetricTracker {
    pub fn new(name: &str, max_samples: usize) -> Self {
        Self {
            name: name.to_string(),
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            warn_threshold: None,
            critical_threshold: None,
        }
    }

    /// Set thresholds.
    pub fn with_thresholds(mut self, warn: f32, critical: f32) -> Self {
        self.warn_threshold = Some(warn);
        self.critical_threshold = Some(critical);
        self
    }

    /// Record a new sample.
    pub fn record(&mut self, timestamp_ms: u64, value: f32) {
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(MetricSample {
            timestamp_ms,
            value,
        });
    }

    /// Latest value.
    pub fn latest(&self) -> Option<f32> {
        self.samples.back().map(|s| s.value)
    }

    /// Mean over the window.
    pub fn mean(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.samples.iter().map(|s| s.value).sum();
        sum / self.samples.len() as f32
    }

    /// Standard deviation over the window.
    pub fn std_dev(&self) -> f32 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let var: f32 = self
            .samples
            .iter()
            .map(|s| (s.value - mean).powi(2))
            .sum::<f32>()
            / (self.samples.len() - 1) as f32;
        var.sqrt()
    }

    /// Maximum in window.
    pub fn max(&self) -> f32 {
        self.samples
            .iter()
            .map(|s| s.value)
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Minimum in window.
    pub fn min(&self) -> f32 {
        self.samples
            .iter()
            .map(|s| s.value)
            .fold(f32::INFINITY, f32::min)
    }

    /// P99 (99th percentile).
    pub fn p99(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut values: Vec<f32> = self.samples.iter().map(|s| s.value).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((values.len() as f32) * 0.99) as usize;
        values[idx.min(values.len() - 1)]
    }

    /// Check health status based on latest value.
    pub fn status(&self) -> HealthStatus {
        if let Some(val) = self.latest() {
            if let Some(crit) = self.critical_threshold {
                if val >= crit {
                    return HealthStatus::Critical(format!(
                        "{}: {:.2} >= critical {}",
                        self.name, val, crit
                    ));
                }
            }
            if let Some(warn) = self.warn_threshold {
                if val >= warn {
                    return HealthStatus::Degraded(format!(
                        "{}: {:.2} >= warn {}",
                        self.name, val, warn
                    ));
                }
            }
        }
        HealthStatus::Healthy
    }

    /// Detect anomalies (z-score > 3).
    pub fn is_anomalous(&self) -> bool {
        if let Some(val) = self.latest() {
            let mean = self.mean();
            let std = self.std_dev();
            if std > 1e-8 {
                return ((val - mean) / std).abs() > 3.0;
            }
        }
        false
    }
}

/// Comprehensive health monitor.
pub struct HealthMonitor {
    /// Token generation latency (ms/token).
    pub latency: MetricTracker,

    /// Expert cache hit rate [0, 1].
    pub cache_hit_rate: MetricTracker,

    /// VRAM usage (MB).
    pub vram_usage: MetricTracker,

    /// RAM usage (MB).
    pub ram_usage: MetricTracker,

    /// Safety score [0, 1].
    pub safety_score: MetricTracker,

    /// MPD agreement rate [0, 1].
    pub mpd_agreement: MetricTracker,

    /// Expert load stalls per second.
    pub load_stalls: MetricTracker,

    /// Total tokens generated.
    pub total_tokens: u64,
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self {
            latency: MetricTracker::new("latency_ms", 1000).with_thresholds(100.0, 500.0),
            cache_hit_rate: MetricTracker::new("cache_hit_rate", 100),
            vram_usage: MetricTracker::new("vram_mb", 100).with_thresholds(3500.0, 3900.0),
            ram_usage: MetricTracker::new("ram_mb", 100).with_thresholds(28000.0, 31000.0),
            safety_score: MetricTracker::new("safety_score", 1000),
            mpd_agreement: MetricTracker::new("mpd_agreement", 1000),
            load_stalls: MetricTracker::new("load_stalls", 100).with_thresholds(5.0, 20.0),
            total_tokens: 0,
        }
    }

    /// Record a token generation event.
    pub fn record_token(
        &mut self,
        timestamp_ms: u64,
        latency_ms: f32,
        safety: f32,
        mpd_agreed: bool,
    ) {
        self.total_tokens += 1;
        self.latency.record(timestamp_ms, latency_ms);
        self.safety_score.record(timestamp_ms, safety);
        self.mpd_agreement
            .record(timestamp_ms, if mpd_agreed { 1.0 } else { 0.0 });
    }

    /// Record system metrics.
    pub fn record_system(
        &mut self,
        timestamp_ms: u64,
        vram_mb: f32,
        ram_mb: f32,
        cache_hit: f32,
        stalls: f32,
    ) {
        self.vram_usage.record(timestamp_ms, vram_mb);
        self.ram_usage.record(timestamp_ms, ram_mb);
        self.cache_hit_rate.record(timestamp_ms, cache_hit);
        self.load_stalls.record(timestamp_ms, stalls);
    }

    /// Get overall health status.
    pub fn overall_status(&self) -> HealthStatus {
        let checks = [
            self.latency.status(),
            self.vram_usage.status(),
            self.ram_usage.status(),
            self.load_stalls.status(),
        ];

        for check in &checks {
            if matches!(check, HealthStatus::Critical(_)) {
                return check.clone();
            }
        }

        for check in &checks {
            if matches!(check, HealthStatus::Degraded(_)) {
                return check.clone();
            }
        }

        HealthStatus::Healthy
    }

    /// Generate a summary report.
    pub fn report(&self) -> HealthReport {
        HealthReport {
            status: self.overall_status(),
            total_tokens: self.total_tokens,
            avg_latency_ms: self.latency.mean(),
            p99_latency_ms: self.latency.p99(),
            avg_safety_score: self.safety_score.mean(),
            mpd_agreement_rate: self.mpd_agreement.mean(),
            cache_hit_rate: self.cache_hit_rate.mean(),
            vram_mb: self.vram_usage.latest().unwrap_or(0.0),
            ram_mb: self.ram_usage.latest().unwrap_or(0.0),
        }
    }
}

/// A health report snapshot.
#[derive(Debug)]
pub struct HealthReport {
    pub status: HealthStatus,
    pub total_tokens: u64,
    pub avg_latency_ms: f32,
    pub p99_latency_ms: f32,
    pub avg_safety_score: f32,
    pub mpd_agreement_rate: f32,
    pub cache_hit_rate: f32,
    pub vram_mb: f32,
    pub ram_mb: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_tracker_basic() {
        let mut t = MetricTracker::new("test", 100);
        t.record(0, 1.0);
        t.record(1, 2.0);
        t.record(2, 3.0);

        assert!((t.mean() - 2.0).abs() < 1e-6);
        assert!((t.max() - 3.0).abs() < 1e-6);
        assert!((t.min() - 1.0).abs() < 1e-6);
        assert_eq!(t.latest(), Some(3.0));
    }

    #[test]
    fn test_metric_tracker_window() {
        let mut t = MetricTracker::new("test", 3);
        for i in 0..5 {
            t.record(i as u64, i as f32);
        }
        assert_eq!(t.samples.len(), 3);
        assert!((t.mean() - 3.0).abs() < 1e-6); // [2, 3, 4]
    }

    #[test]
    fn test_health_status_thresholds() {
        let mut t = MetricTracker::new("latency", 100).with_thresholds(50.0, 100.0);

        t.record(0, 30.0);
        assert_eq!(t.status(), HealthStatus::Healthy);

        t.record(1, 60.0);
        assert!(matches!(t.status(), HealthStatus::Degraded(_)));

        t.record(2, 200.0);
        assert!(matches!(t.status(), HealthStatus::Critical(_)));
    }

    #[test]
    fn test_anomaly_detection() {
        let mut t = MetricTracker::new("test", 100);
        // Normal values around 10
        for i in 0..50 {
            t.record(i, 10.0 + (i as f32 % 3.0) * 0.1);
        }
        assert!(!t.is_anomalous());

        // Spike
        t.record(100, 1000.0);
        assert!(t.is_anomalous());
    }

    #[test]
    fn test_health_monitor() {
        let mut mon = HealthMonitor::new();
        mon.record_token(0, 50.0, 0.99, true);
        mon.record_token(1, 55.0, 0.98, true);

        assert_eq!(mon.total_tokens, 2);
        assert!(matches!(mon.overall_status(), HealthStatus::Healthy));
    }

    #[test]
    fn test_health_report() {
        let mut mon = HealthMonitor::new();
        for i in 0..10 {
            mon.record_token(i, 40.0 + i as f32, 0.95, true);
        }

        let report = mon.report();
        assert_eq!(report.total_tokens, 10);
        assert!(report.avg_latency_ms > 40.0);
        assert!(report.avg_safety_score > 0.9);
    }

    #[test]
    fn test_std_dev() {
        let mut t = MetricTracker::new("test", 100);
        t.record(0, 10.0);
        t.record(1, 10.0);
        t.record(2, 10.0);
        assert!(t.std_dev() < 1e-6); // Zero variance

        t.record(3, 20.0);
        assert!(t.std_dev() > 0.0); // Non-zero variance
    }
}
