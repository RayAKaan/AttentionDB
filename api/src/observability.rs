//! Observability — Structured Logging, Prometheus Metrics, Request Tracing
//!
//! # Logging
//!
//! Uses `tracing` with configurable levels via `RUST_LOG` env var:
//! ```bash
//! RUST_LOG=attentiondb=debug,tower_http=info cargo run --bin attentiondb-server
//! ```
//!
//! # Metrics
//!
//! Prometheus endpoint at `/metrics` (port 9090 by default).
//! ```bash
//! curl http://localhost:9090/metrics
//! ```

use metrics::{counter, gauge, histogram};
use std::time::Instant;
use tracing::info;

pub type MetricsHandle = metrics_exporter_prometheus::PrometheusHandle;

/// Initialize the tracing subscriber for structured logging.
/// Reads `RUST_LOG` env var for filter configuration.
/// Defaults to `attentiondb=info,tower_http=info` if unset.
pub fn init_logging() {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("attentiondb=info,tower_http=info,attentiondb_api=info,attentiondb_core=info"));

    let subscriber = fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .with_ansi(atty::is(atty::Stream::Stdout))
        .compact()
        .finish();

    let _ = tracing::subscriber::set_global_default(subscriber);
}

/// Initialize the Prometheus metrics exporter.
/// Returns a handle to the recorder. Metrics are served at `/metrics`.
pub fn init_metrics() -> Option<MetricsHandle> {
    use metrics_exporter_prometheus::PrometheusBuilder;

    match PrometheusBuilder::new().install_recorder() {
        Ok(handle) => {
            info!("Prometheus metrics exporter installed");
            Some(handle)
        }
        Err(e) => {
            tracing::warn!("Failed to install Prometheus exporter: {} (metrics will be no-ops)", e);
            None
        }
    }
}

/// Record an ATTEND query with all relevant dimensions.
pub fn record_attend(collection: &str, heads: &[String], top_k: usize, result_count: usize, latency_ms: f64) {
    counter!("attentiondb_attend_total").increment(1);
    histogram!("attentiondb_attend_latency_ms").record(latency_ms);
    histogram!("attentiondb_attend_result_count").record(result_count as f64);
    gauge!("attentiondb_attend_top_k").set(top_k as f64);

    tracing::info!(
        collection = collection,
        heads = ?heads,
        top_k = top_k,
        result_count = result_count,
        latency_ms = format!("{:.2}", latency_ms),
        "ATTEND query completed"
    );
}

/// Record a document insertion.
pub fn record_insert(collection: &str, doc_id: &str, num_vectors: usize, latency_ms: f64) {
    counter!("attentiondb_insert_total").increment(1);
    histogram!("attentiondb_insert_latency_ms").record(latency_ms);

    tracing::info!(
        collection = collection,
        doc_id = doc_id,
        num_vectors = num_vectors,
        latency_ms = format!("{:.2}", latency_ms),
        "Document inserted"
    );
}

/// Record a document deletion.
pub fn record_delete(collection: &str, doc_id: &str, success: bool) {
    counter!("attentiondb_delete_total").increment(1);
    if !success {
        counter!("attentiondb_delete_not_found_total").increment(1);
    }
    tracing::info!(collection = collection, doc_id = doc_id, success = success, "Document deleted");
}

/// Record collection creation.
pub fn record_create_collection(collection: &str, heads: &[&str], ef_search: usize) {
    counter!("attentiondb_collections_created_total").increment(1);
    tracing::info!(
        collection = collection,
        heads = ?heads,
        ef_search = ef_search,
        "Collection created"
    );
}

/// Record engine stats as gauges.
pub fn record_engine_stats(collection_count: usize, total_heads: usize, total_vectors: usize) {
    gauge!("attentiondb_collections_count").set(collection_count as f64);
    gauge!("attentiondb_heads_count").set(total_heads as f64);
    gauge!("attentiondb_vectors_count").set(total_vectors as f64);
}

/// Record an error.
pub fn record_error(operation: &str, error: &str) {
    counter!("attentiondb_errors_total").increment(1);
    tracing::error!(operation = operation, error = error, "Operation failed");
}

/// A timing guard that records latency on drop.
pub struct LatencyTimer {
    operation: &'static str,
    start: Instant,
}

impl LatencyTimer {
    pub fn new(operation: &'static str) -> Self {
        Self { operation, start: Instant::now() }
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

impl Drop for LatencyTimer {
    fn drop(&mut self) {
        let ms = self.elapsed_ms();
        histogram!("attentiondb_operation_latency_ms").record(ms);
        tracing::debug!(operation = self.operation, latency_ms = format!("{:.2}", ms), "Operation latency recorded");
    }
}
