use crate::adapters::DatabaseAdapter;
use crate::workload::Document;
use std::time::{Duration, Instant};

pub struct IsolationManager;

impl IsolationManager {
    pub async fn verify_empty(
        adapter: &dyn DatabaseAdapter,
        dimension: usize,
    ) -> anyhow::Result<()> {
        let vector = vec![0.1; dimension];
        let result = adapter.query_single_vector(&vector, 1).await?;

        if !result.ranked_ids.is_empty() {
            anyhow::bail!(
                "Isolation error: database '{}' should be empty but returned {} results",
                adapter.name(),
                result.ranked_ids.len()
            );
        }
        Ok(())
    }

    pub async fn verify_count(
        adapter: &dyn DatabaseAdapter,
        expected_count: usize,
        dimension: usize,
    ) -> anyhow::Result<()> {
        let vector = vec![0.1; dimension];
        let result = adapter.query_single_vector(&vector, expected_count * 2).await?;

        if result.ranked_ids.len() < expected_count {
            anyhow::bail!(
                "Isolation error: database '{}' expected ~{} docs but got {} results",
                adapter.name(),
                expected_count,
                result.ranked_ids.len()
            );
        }
        Ok(())
    }

    pub async fn full_teardown_rebuild(
        adapter: &dyn DatabaseAdapter,
        documents: &[Document],
        dimension: usize,
    ) -> anyhow::Result<Duration> {
        let start = Instant::now();

        adapter.teardown().await?;
        adapter.setup_collection(dimension).await?;

        for chunk in documents.chunks(500) {
            adapter.insert_batch(chunk).await?;
        }
        adapter.flush().await?;
        let _ = adapter.build_index().await;

        Ok(start.elapsed())
    }
}
