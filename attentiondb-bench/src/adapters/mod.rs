pub mod translation;
pub mod attentiondb_cpu;
pub mod attentiondb_gpu;
#[cfg(feature = "qdrant")]
pub mod qdrant;
#[cfg(feature = "pgvector")]
pub mod pgvector;
#[cfg(feature = "elasticsearch")]
pub mod elasticsearch;

#[cfg(feature = "milvus")]
pub mod milvus;
#[cfg(feature = "weaviate")]
pub mod weaviate;
#[cfg(feature = "pinecone")]
pub mod pinecone;

use async_trait::async_trait;
use crate::workload::{Document, Query};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub ranked_ids: Vec<String>,
    pub scores: Vec<f32>,
    pub latency: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertResult {
    pub count: usize,
    pub duration: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub vector_count: usize,
    pub dimension: usize,
    pub index_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub alive: bool,
    pub version: String,
    pub latency_ms: f64,
}

#[derive(Debug, Clone)]
pub struct AdapterCapabilities {
    pub supports_multi_vector: bool,
    pub supports_filtering: bool,
    pub supports_hybrid_search: bool,
    pub max_batch_size: usize,
    pub requires_index_before_query: bool,
}

#[async_trait]
pub trait DatabaseAdapter: Send + Sync {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> AdapterCapabilities;

    async fn connect(&mut self, config: &ConnectionConfig) -> anyhow::Result<()>;
    async fn health_check(&self) -> anyhow::Result<HealthStatus>;
    async fn setup_collection(&self, dimension: usize) -> anyhow::Result<()>;
    async fn teardown(&self) -> anyhow::Result<()>;
    async fn disconnect(&self) -> anyhow::Result<()>;

    async fn insert_batch(&self, documents: &[Document]) -> anyhow::Result<InsertResult>;
    async fn flush(&self) -> anyhow::Result<()>;
    async fn build_index(&self) -> anyhow::Result<Duration>;

    async fn query(&self, query: &Query, top_k: usize) -> anyhow::Result<QueryResult>;

    async fn query_single_vector(
        &self,
        vector: &[f32],
        top_k: usize,
    ) -> anyhow::Result<QueryResult>;
}

#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    pub host: String,
    pub port: u16,
    pub use_tls: bool,
    pub api_key: Option<String>,
    pub collection_name: String,
    pub transport: String,
}
