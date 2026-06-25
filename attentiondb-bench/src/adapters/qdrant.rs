use async_trait::async_trait;
use crate::workload::{Document, Query};
use crate::adapters::{DatabaseAdapter, ConnectionConfig, QueryResult, InsertResult, HealthStatus, AdapterCapabilities, translation};
use std::time::{Duration, Instant};

pub struct QdrantAdapter {
    host: String,
    port: u16,
    collection_name: String,
    client: Option<qdrant_client::Qdrant>,
}

impl QdrantAdapter {
    pub fn new() -> Self {
        Self {
            host: String::new(),
            port: 6334,
            collection_name: String::new(),
            client: None,
        }
    }
}

#[async_trait]
impl DatabaseAdapter for QdrantAdapter {
    fn name(&self) -> &'static str { "Qdrant" }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            supports_multi_vector: true,
            supports_filtering: true,
            supports_hybrid_search: false,
            max_batch_size: 1000,
            requires_index_before_query: false,
        }
    }

    async fn connect(&mut self, config: &ConnectionConfig) -> anyhow::Result<()> {
        self.host = config.host.clone();
        self.port = config.port;
        self.collection_name = config.collection_name.clone();

        let url = format!("http://{}:{}", self.host, self.port);
        self.client = Some(qdrant_client::Qdrant::from_url(&url)
            .build()
            .map_err(|e| anyhow::anyhow!("Qdrant client build failed: {}", e))?);

        let health = self.health_check().await?;
        anyhow::ensure!(health.alive, "Qdrant not reachable at {}", url);
        tracing::info!("Connected to Qdrant at {} (v{})", url, health.version);
        Ok(())
    }

    async fn health_check(&self) -> anyhow::Result<HealthStatus> {
        let client = self.client.as_ref().unwrap();
        let start = Instant::now();
        let resp = client.health_check().await;
        let latency = start.elapsed();

        match resp {
            Ok(r) => Ok(HealthStatus {
                alive: true,
                version: r.version,
                latency_ms: latency.as_secs_f64() * 1000.0,
            }),
            Err(e) => Ok(HealthStatus {
                alive: false,
                version: e.to_string(),
                latency_ms: latency.as_secs_f64() * 1000.0,
            }),
        }
    }

    async fn setup_collection(&self, dimension: usize) -> anyhow::Result<()> {
        let client = self.client.as_ref().unwrap();
        let _ = client
            .delete_collection(&self.collection_name)
            .await;

        client
            .create_collection(
                qdrant_client::qdrant::CreateCollectionBuilder::new(&self.collection_name)
                    .vectors_config(
                        qdrant_client::qdrant::VectorParamsBuilder::new(dimension as u64, qdrant_client::qdrant::Distance::Cosine),
                    ),
            )
            .await?;

        tracing::info!("Created Qdrant collection '{}' (dim={})", self.collection_name, dimension);
        Ok(())
    }

    async fn teardown(&self) -> anyhow::Result<()> {
        let client = self.client.as_ref().unwrap();
        let _ = client
            .delete_collection(&self.collection_name)
            .await;
        Ok(())
    }

    async fn disconnect(&self) -> anyhow::Result<()> { Ok(()) }

    async fn insert_batch(&self, documents: &[Document]) -> anyhow::Result<InsertResult> {
        let client = self.client.as_ref().unwrap();
        let start = Instant::now();

        let mut points = Vec::with_capacity(documents.len());
        for doc in documents {
            let head = crate::workload::HeadType::Semantic;
            let vector = translation::translate_document_vector(doc, translation::TranslationStrategy::SemanticOnly, head)?;
            use qdrant_client::qdrant::{point_id, PointId};
            let id = PointId {
                point_id_options: Some(point_id::PointIdOptions::Uuid(doc.id.clone())),
            };
            points.push(qdrant_client::qdrant::PointStruct {
                id: Some(id),
                vectors: Some(vector.into()),
                payload: std::collections::HashMap::new(),
            });
        }

        client.upsert_points(
            qdrant_client::qdrant::UpsertPointsBuilder::new(&self.collection_name, points)
        ).await?;

        Ok(InsertResult { count: documents.len(), duration: start.elapsed() })
    }

    async fn flush(&self) -> anyhow::Result<()> { Ok(()) }

    async fn build_index(&self) -> anyhow::Result<Duration> {
        let client = self.client.as_ref().unwrap();
        let start = Instant::now();
        client
            .update_collection(
                qdrant_client::qdrant::UpdateCollectionBuilder::new(&self.collection_name)
                    .optimizers_config(
                        qdrant_client::qdrant::OptimizersConfigDiffBuilder::default()
                            .default_segment_number(1)
                            .indexing_threshold(0),
                    ),
            )
            .await?;
        tokio::time::sleep(Duration::from_secs(1)).await;
        Ok(start.elapsed())
    }

    async fn query(&self, query: &Query, top_k: usize) -> anyhow::Result<QueryResult> {
        let client = self.client.as_ref().unwrap();
        let head = crate::workload::HeadType::Semantic;
        let q_vec = query.embeddings.iter()
            .find(|e| e.head_name == head)
            .ok_or_else(|| anyhow::anyhow!("Query missing semantic head"))?;

        let start = Instant::now();
        let search = qdrant_client::qdrant::SearchPointsBuilder::new(
            &self.collection_name,
            q_vec.vector.clone(),
            top_k as u64,
        ).build();
        let resp = client.search_points(search).await?;

        let latency = start.elapsed();
        let ranked_ids: Vec<String> = resp.result.iter()
            .filter_map(|p| {
                p.id.as_ref().and_then(|id| {
                    id.point_id_options.as_ref().map(|opts| match opts {
                        qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u) => u.to_string(),
                        qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => n.to_string(),
                    })
                })
            })
            .collect();
        let scores: Vec<f32> = resp.result.iter().map(|p| p.score).collect();

        Ok(QueryResult { ranked_ids, scores, latency })
    }

    async fn query_single_vector(&self, vector: &[f32], top_k: usize) -> anyhow::Result<QueryResult> {
        let client = self.client.as_ref().unwrap();
        let start = Instant::now();
        let search = qdrant_client::qdrant::SearchPointsBuilder::new(
            &self.collection_name,
            vector.to_vec(),
            top_k as u64,
        ).build();
        let resp = client.search_points(search).await?;

        let latency = start.elapsed();
        let ranked_ids: Vec<String> = resp.result.iter()
            .filter_map(|p| {
                p.id.as_ref().and_then(|id| {
                    id.point_id_options.as_ref().map(|opts| match opts {
                        qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u) => u.to_string(),
                        qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => n.to_string(),
                    })
                })
            })
            .collect();
        let scores: Vec<f32> = resp.result.iter().map(|p| p.score).collect();

        Ok(QueryResult { ranked_ids, scores, latency })
    }
}
