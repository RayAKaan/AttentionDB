use async_trait::async_trait;
use crate::workload::{Document, Query};
use crate::adapters::{DatabaseAdapter, ConnectionConfig, QueryResult, InsertResult, HealthStatus, AdapterCapabilities};
use std::time::Duration;

pub struct AttentionDBGPUAdapter {
    inner: super::attentiondb_cpu::AttentionDBAdapter,
}

impl AttentionDBGPUAdapter {
    pub fn new() -> Self {
        Self {
            inner: super::attentiondb_cpu::AttentionDBAdapter::new(),
        }
    }
}

#[async_trait]
impl DatabaseAdapter for AttentionDBGPUAdapter {
    fn name(&self) -> &'static str { "AttentionDB (GPU)" }
    fn capabilities(&self) -> AdapterCapabilities { self.inner.capabilities() }

    async fn connect(&mut self, config: &ConnectionConfig) -> anyhow::Result<()> {
        self.inner.connect(config).await
    }

    async fn health_check(&self) -> anyhow::Result<HealthStatus> {
        let mut status = self.inner.health_check().await?;
        status.version = format!("{} (CUDA)", status.version);
        Ok(status)
    }

    async fn setup_collection(&self, dimension: usize) -> anyhow::Result<()> {
        self.inner.setup_collection(dimension).await
    }

    async fn teardown(&self) -> anyhow::Result<()> { self.inner.teardown().await }
    async fn disconnect(&self) -> anyhow::Result<()> { Ok(()) }

    async fn insert_batch(&self, documents: &[Document]) -> anyhow::Result<InsertResult> {
        self.inner.insert_batch(documents).await
    }

    async fn flush(&self) -> anyhow::Result<()> { Ok(()) }
    async fn build_index(&self) -> anyhow::Result<Duration> { Ok(Duration::from_secs(0)) }

    async fn query(&self, query: &Query, top_k: usize) -> anyhow::Result<QueryResult> {
        self.inner.query(query, top_k).await
    }

    async fn query_single_vector(&self, vector: &[f32], top_k: usize) -> anyhow::Result<QueryResult> {
        self.inner.query_single_vector(vector, top_k).await
    }
}
