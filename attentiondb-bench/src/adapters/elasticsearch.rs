use async_trait::async_trait;
use crate::workload::{Document, Query};
use crate::adapters::{DatabaseAdapter, ConnectionConfig, QueryResult, InsertResult, HealthStatus, AdapterCapabilities, translation};
use elasticsearch::{Elasticsearch, http::transport::Transport, http::request::JsonBody, SearchParts, IndexParts, BulkParts, IndicesCreateParts, IndicesDeleteParts};
use serde_json::json;
use std::time::{Duration, Instant};

pub struct ElasticsearchAdapter {
    client: Option<Elasticsearch>,
    collection_name: String,
    dimension: usize,
}

impl ElasticsearchAdapter {
    pub fn new() -> Self {
        Self {
            client: None,
            collection_name: "bench".into(),
            dimension: 0,
        }
    }
}

#[async_trait]
impl DatabaseAdapter for ElasticsearchAdapter {
    fn name(&self) -> &'static str { "Elasticsearch" }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            supports_multi_vector: false,
            supports_filtering: true,
            supports_hybrid_search: true,
            max_batch_size: 1000,
            requires_index_before_query: false,
        }
    }

    async fn connect(&mut self, config: &ConnectionConfig) -> anyhow::Result<()> {
        self.collection_name = config.collection_name.clone();
        let url = format!("http://{}:{}", config.host, config.port);
        let transport = Transport::single_node(&url)
            .map_err(|e| anyhow::anyhow!("ES transport: {}", e))?;
        self.client = Some(Elasticsearch::new(transport));

        let health = self.health_check().await?;
        anyhow::ensure!(health.alive, "Elasticsearch not reachable at {}", url);
        tracing::info!("Connected to Elasticsearch at {}", url);
        Ok(())
    }

    async fn health_check(&self) -> anyhow::Result<HealthStatus> {
        let client = self.client.as_ref().unwrap();
        let start = Instant::now();
        let resp = client.ping().send().await?;
        let latency = start.elapsed();
        Ok(HealthStatus {
            alive: resp.status_code().is_success(),
            version: "8.x".into(),
            latency_ms: latency.as_secs_f64() * 1000.0,
        })
    }

    async fn setup_collection(&self, dimension: usize) -> anyhow::Result<()> {
        let client = self.client.as_ref().unwrap();
        self.dimension = dimension;

        let _ = client.indices().delete(IndicesDeleteParts::Index(&[&self.collection_name]))
            .send().await;

        let mapping = json!({
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dimension,
                        "index": true,
                        "similarity": "cosine"
                    },
                    "text": { "type": "text" }
                }
            }
        });

        client.indices().create(IndicesCreateParts::Index(&self.collection_name))
            .body(mapping)
            .send()
            .await?;

        tracing::info!("Created ES index '{}' (dim={})", self.collection_name, dimension);
        Ok(())
    }

    async fn teardown(&self) -> anyhow::Result<()> {
        let client = self.client.as_ref().unwrap();
        let _ = client.indices().delete(IndicesDeleteParts::Index(&[&self.collection_name]))
            .send().await;
        Ok(())
    }

    async fn disconnect(&self) -> anyhow::Result<()> { Ok(()) }

    async fn insert_batch(&self, documents: &[Document]) -> anyhow::Result<InsertResult> {
        let client = self.client.as_ref().unwrap();
        let start = Instant::now();
        let head = crate::workload::HeadType::Semantic;

        for doc in documents {
            let vector = translation::translate_document_vector(doc, translation::TranslationStrategy::SemanticOnly, head)?;
            let body = json!({
                "embedding": vector,
                "text": doc.metadata.get("text").cloned().unwrap_or_default(),
            });

            client.index(IndexParts::IndexId(&self.collection_name, &doc.id))
                .body(body)
                .send()
                .await?;
        }

        // Refresh to make documents searchable immediately
        tokio::time::sleep(Duration::from_millis(200)).await;

        Ok(InsertResult { count: documents.len(), duration: start.elapsed() })
    }

    async fn flush(&self) -> anyhow::Result<()> { Ok(()) }

    async fn build_index(&self) -> anyhow::Result<Duration> {
        // ES builds index automatically
        Ok(Duration::from_secs(0))
    }

    async fn query(&self, query: &Query, top_k: usize) -> anyhow::Result<QueryResult> {
        let client = self.client.as_ref().unwrap();
        let head = crate::workload::HeadType::Semantic;

        let q_vec = query.embeddings.iter()
            .find(|e| e.head_name == head)
            .ok_or_else(|| anyhow::anyhow!("Query missing semantic head"))?;

        let body = json!({
            "size": top_k,
            "query": {
                "script_score": {
                    "query": { "match_all": {} },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": { "query_vector": q_vec.vector }
                    }
                }
            }
        });

        let start = Instant::now();
        let resp = client.search(SearchParts::Index(&[&self.collection_name]))
            .body(body)
            .send()
            .await?;
        let latency = start.elapsed();

        let response_body: serde_json::Value = resp.json().await?;
        let hits = response_body["hits"]["hits"].as_array()
            .ok_or_else(|| anyhow::anyhow!("ES query failed: {:?}", response_body))?;

        let ranked_ids: Vec<String> = hits.iter()
            .map(|h| h["_id"].as_str().unwrap_or("").to_string())
            .collect();
        let scores: Vec<f32> = hits.iter()
            .map(|h| h["_score"].as_f64().unwrap_or(0.0) as f32)
            .collect();

        Ok(QueryResult { ranked_ids, scores, latency })
    }

    async fn query_single_vector(&self, vector: &[f32], top_k: usize) -> anyhow::Result<QueryResult> {
        let query = Query {
            id: "es_single".into(),
            embeddings: vec![crate::workload::HeadEmbedding {
                head_name: crate::workload::HeadType::Semantic,
                vector: vector.to_vec(),
            }],
            enabled_heads: vec![crate::workload::HeadType::Semantic],
            ground_truth: vec![],
            difficulty: crate::workload::difficulty::DifficultyLevel::Easy,
            failure_mode: None,
        };
        self.query(&query, top_k).await
    }
}
