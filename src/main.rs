//! PERSPECTIVE — Perspective Is All You Need
//!
//! A 1.05T parameter sparse Mixture-of-Experts architecture
//! designed to run on 4 GB VRAM + 32 GB RAM consumer hardware.
//!
//! This is the CLI binary entry point.

use clap::Parser;
use perspective::runtime::health::HealthMonitor;
use perspective::runtime::pipeline::{InferencePipeline, PipelineConfig};
use perspective::runtime::zeroclaw::{ZeroclawClient, ZeroclawConfig};
use std::path::PathBuf;

/// PERSPECTIVE inference CLI.
#[derive(Parser, Debug)]
#[command(
    name = "perspective",
    about = "PERSPECTIVE — 1.05T sparse MoE on consumer hardware",
    version
)]
struct Cli {
    /// Path to the model directory containing weight shards.
    #[arg(short, long, default_value = "./model")]
    model_dir: String,

    /// Maximum number of tokens to generate.
    #[arg(short = 'n', long, default_value_t = 256)]
    max_tokens: usize,

    /// Enable safety polytope projection.
    #[arg(long, default_value_t = true)]
    safety: bool,

    /// Enable multi-perspective decoding.
    #[arg(long, default_value_t = true)]
    mpd: bool,

    /// Enable continual learning (FMEA).
    #[arg(long, default_value_t = false)]
    learn: bool,

    /// Enable episodic memory (HDM).
    #[arg(long, default_value_t = true)]
    memory: bool,

    /// Print health diagnostics after generation.
    #[arg(long, default_value_t = false)]
    health: bool,

    /// Input prompt.
    #[arg(trailing_var_arg = true)]
    prompt: Vec<String>,

    /// Optional ZeroClaw task to execute via swarm orchestration.
    #[arg(long)]
    swarm_task: Option<String>,

    /// ZeroClaw executable name/path.
    #[arg(long, default_value = "zeroclaw")]
    zeroclaw_bin: String,

    /// Working directory for ZeroClaw execution.
    #[arg(long, default_value = ".")]
    swarm_workspace: String,
}

fn main() {
    // Initialise tracing
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    tracing::info!("PERSPECTIVE v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Model directory: {}", cli.model_dir);

    if let Some(task) = cli.swarm_task.as_deref() {
        let config = ZeroclawConfig {
            executable: PathBuf::from(&cli.zeroclaw_bin),
            workspace: PathBuf::from(&cli.swarm_workspace),
            ..Default::default()
        };
        let client = ZeroclawClient::new(config);
        if let Err(err) = client.check_available() {
            tracing::error!("ZeroClaw unavailable: {}", err);
            return;
        }
        match client.run_task(task) {
            Ok(result) => {
                tracing::info!("ZeroClaw run exit code: {}", result.status_code);
                if !result.stdout.trim().is_empty() {
                    tracing::info!("ZeroClaw output:\n{}", result.stdout.trim());
                }
                if !result.stderr.trim().is_empty() {
                    tracing::warn!("ZeroClaw stderr:\n{}", result.stderr.trim());
                }
            }
            Err(err) => {
                tracing::error!("ZeroClaw task failed: {}", err);
            }
        }
        return;
    }

    let config = PipelineConfig {
        enable_fmea: cli.learn,
        enable_safety: cli.safety,
        enable_memory: cli.memory,
        enable_mpd: cli.mpd,
        max_gen_length: cli.max_tokens,
        ..Default::default()
    };

    tracing::info!(
        "Config: {} layers, {} experts, d_model={}, vocab={}",
        config.n_layers,
        config.n_experts,
        config.d_model,
        config.vocab_size,
    );

    let mut pipeline = InferencePipeline::new(config);
    let health = HealthMonitor::new();

    let prompt_text = cli.prompt.join(" ");
    if prompt_text.is_empty() {
        tracing::warn!("No prompt provided. Running empty generation.");
    }

    // In a full implementation, we would:
    // 1. Tokenise the prompt
    // 2. Load model weights via WeightProvider
    // 3. Run the inference pipeline
    // 4. Detokenise and print output

    tracing::info!("Prompt: \"{}\"", prompt_text);
    tracing::info!("Generating up to {} tokens...", cli.max_tokens);

    // Placeholder: simulate generation
    let prompt_tokens: Vec<usize> = (0..prompt_text.len().min(100)).collect();
    let result = match pipeline.generate(&prompt_tokens, cli.max_tokens, 0) {
        Ok(result) => result,
        Err(err) => {
            tracing::error!("Generation failed: {}", err);
            return;
        }
    };

    tracing::info!(
        "Generated {} tokens in {:.1} ms ({:.1} tok/s)",
        result.tokens.len(),
        result.total_time_ms,
        result.tokens_per_second,
    );

    if cli.health {
        let report = health.report();
        tracing::info!("Health: {:?}", report.status);
        tracing::info!("  Avg latency: {:.1} ms", report.avg_latency_ms);
        tracing::info!("  P99 latency: {:.1} ms", report.p99_latency_ms);
        tracing::info!("  Safety score: {:.3}", report.avg_safety_score);
        tracing::info!("  MPD agreement: {:.1}%", report.mpd_agreement_rate * 100.0);
    }

    tracing::info!("Done.");
}
