use async_trait::async_trait;
use crate::workload::{Document, Query};
use crate::adapters::{DatabaseAdapter, ConnectionConfig, QueryResult, InsertResult, HealthStatus, AdapterCapabilities, translation};
use sqlx::postgres::PgPoolOptions;
use std::time::{Duration, Instant};

pub struct PgvectorAdapter {
    pool: Option<sqlx::PgPool>,
    collection_name: String,
}

impl PgvectorAdapter {
    pub fn new() -> Self {
        Self {
            pool: None,
            collection_name: String::new(),
        }
    }
}

#[async_trait]
impl DatabaseAdapter for PgvectorAdapter {
    fn name(&self) -> &'static str { "pgvector" }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            supports_multi_vector: false,
            supports_filtering: true,
            supports_hybrid_search: false,
            max_batch_size: 5000,
            requires_index_before_query: true,
        }
    }

    async fn connect(&mut self, config: &ConnectionConfig) -> anyhow::Result<()> {
        self.collection_name = config.collection_name.clone();
        let conn_str = format!("postgres://postgres:postgres@{}:{}/postgres", config.host, config.port);

        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(&conn_str)
            .await?;

        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&pool)
            .await?;

        self.pool = Some(pool);

        let health = self.health_check().await?;
        anyhow::ensure!(health.alive, "pgvector not reachable");
        tracing::info!("Connected to pgvector");
        Ok(())
    }

    async fn health_check(&self) -> anyhow::Result<HealthStatus> {
        let pool = self.pool.as_ref().unwrap();
        let start = Instant::now();
        let row: (i32,) = sqlx::query_as("SELECT 1")
            .fetch_one(pool)
            .await?;
        let latency = start.elapsed();
        Ok(HealthStatus {
            alive: row.0 == 1,
            version: "pgvector 0.7".into(),
            latency_ms: latency.as_secs_f64() * 1000.0,
        })
    }

    async fn setup_collection(&self, dimension: usize) -> anyhow::Result<()> {
        let pool = self.pool.as_ref().unwrap();
        sqlx::query(&format!("DROP TABLE IF EXISTS {}", self.collection_name))
            .execute(pool)
            .await?;

        sqlx::query(&format!(
            "CREATE TABLE {} (
                id TEXT PRIMARY KEY,
                embedding vector({})
            )", self.collection_name, dimension
        ))
        .execute(pool)
        .await?;

        tracing::info!("Created pgvector table '{}' (dim={})", self.collection_name, dimension);
        Ok(())
    }

    async fn teardown(&self) -> anyhow::Result<()> {
        let pool = self.pool.as_ref().unwrap();
        sqlx::query(&format!("DROP TABLE IF EXISTS {}", self.collection_name))
            .execute(pool)
            .await?;
        Ok(())
    }

    async fn disconnect(&self) -> anyhow::Result<()> {
        self.pool.as_ref().unwrap().close().await;
        Ok(())
    }

    async fn insert_batch(&self, documents: &[Document]) -> anyhow::Result<InsertResult> {
        let pool = self.pool.as_ref().unwrap();
        let start = Instant::now();
        let head = crate::workload::HeadType::Semantic;

        for doc in documents {
            let vector = translation::translate_document_vector(doc, translation::TranslationStrategy::SemanticOnly, head.clone())?;
            let vector_str: String = format!("[{}]", vector.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","));

            sqlx::query(&format!(
                "INSERT INTO {} (id, embedding) VALUES ($1, $2::vector) ON CONFLICT (id) DO UPDATE SET embedding = $2::vector",
                self.collection_name
            ))
            .bind(&doc.id)
            .bind(&vector_str)
            .execute(pool)
            .await?;
        }

        Ok(InsertResult { count: documents.len(), duration: start.elapsed() })
    }

    async fn flush(&self) -> anyhow::Result<()> { Ok(()) }

    async fn build_index(&self) -> anyhow::Result<Duration> {
        let pool = self.pool.as_ref().unwrap();
        let start = Instant::now();

        sqlx::query(&format!(
            "CREATE INDEX ON {} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)",
            self.collection_name
        ))
        .execute(pool)
        .await?;

        tracing::info!("Built HNSW index on pgvector table '{}'", self.collection_name);
        Ok(start.elapsed())
    }

    async fn query(&self, query: &Query, top_k: usize) -> anyhow::Result<QueryResult> {
        let pool = self.pool.as_ref().unwrap();
        let head = crate::workload::HeadType::Semantic;

        let q_vec = query.embeddings.iter()
            .find(|e| e.head_name == head)
            .ok_or_else(|| anyhow::anyhow!("Query missing semantic head"))?;

        let vector_str: String = format!("[{}]", q_vec.vector.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(","));

        let start = Instant::now();

        let rows: Vec<(String, f32)> = sqlx::query_as(&format!(
            "SELECT id, 1 - (embedding <=> $1::vector) AS score FROM {} ORDER BY embedding <=> $1::vector LIMIT {}",
            self.collection_name, top_k
        ))
        .bind(&vector_str)
        .fetch_all(pool)
        .await?;

        let latency = start.elapsed();
        let (ranked_ids, scores): (Vec<String>, Vec<f32>) = rows.into_iter().unzip();

        Ok(QueryResult { ranked_ids, scores, latency })
    }

    async fn query_single_vector(&self, vector: &[f32], top_k: usize) -> anyhow::Result<QueryResult> {
        let pool = self.pool.as_ref().unwrap();
        let vector_str: String = format!("[{}]", vector.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(","));

        let start = Instant::now();
        let rows: Vec<(String, f32)> = sqlx::query_as(&format!(
            "SELECT id, 1 - (embedding <=> $1::vector) AS score FROM {} ORDER BY embedding <=> $1::vector LIMIT {}",
            self.collection_name, top_k
        ))
        .bind(&vector_str)
        .fetch_all(pool)
        .await?;

        let latency = start.elapsed();
        let (ranked_ids, scores): (Vec<String>, Vec<f32>) = rows.into_iter().unzip();

        Ok(QueryResult { ranked_ids, scores, latency })
    }
}
