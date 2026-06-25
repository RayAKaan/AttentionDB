use async_trait::async_trait;
use crate::workload::{Document, Query, HeadEmbedding};
use crate::adapters::{DatabaseAdapter, ConnectionConfig, QueryResult, InsertResult, HealthStatus, AdapterCapabilities};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::hash::{Hash, Hasher};
use std::fmt::Write as FmtWrite;
use tokio::sync::Semaphore;

pub struct AttentionDBAdapter {
    client: Option<reqwest::Client>,
    base_url: String,
    collection_name: String,
    /// Maps server-side numeric IDs (hash of server-assigned UUID) to original document IDs
    id_map: Arc<Mutex<HashMap<String, String>>>,
}

impl AttentionDBAdapter {
    pub fn new() -> Self {
        Self {
            client: None,
            base_url: String::new(),
            collection_name: "bench".into(),
            id_map: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn vec_to_csv(v: &[f32]) -> String {
        if v.is_empty() { return String::new(); }
        let cap = v.len() * 9;
        let mut s = String::with_capacity(cap);
        for (i, val) in v.iter().enumerate() {
            if i > 0 { s.push(','); }
            let _ = write!(s, "{}", val);
        }
        s
    }
}

#[async_trait]
impl DatabaseAdapter for AttentionDBAdapter {
    fn name(&self) -> &'static str {
        "AttentionDB (CPU)"
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            supports_multi_vector: true,
            supports_filtering: true,
            supports_hybrid_search: true,
            max_batch_size: 1,
            requires_index_before_query: false,
        }
    }

    async fn connect(&mut self, config: &ConnectionConfig) -> anyhow::Result<()> {
        self.base_url = format!("http://{}:{}", config.host, config.port);
        self.collection_name = config.collection_name.clone();
        self.client = Some(reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?);

        let health = self.health_check().await?;
        anyhow::ensure!(health.alive, "AttentionDB not reachable at {}", self.base_url);
        tracing::info!("Connected to AttentionDB at {} (v{})", self.base_url, health.version);
        Ok(())
    }

    async fn health_check(&self) -> anyhow::Result<HealthStatus> {
        let client = self.client.as_ref().unwrap();
        let start = Instant::now();
        let resp = client.get(&format!("{}/health", self.base_url))
            .send()
            .await?;
        let latency = start.elapsed();
        Ok(HealthStatus {
            alive: resp.status().is_success(),
            version: resp.text().await.unwrap_or_default(),
            latency_ms: latency.as_secs_f64() * 1000.0,
        })
    }

    async fn setup_collection(&self, dimension: usize) -> anyhow::Result<()> {
        let client = self.client.as_ref().unwrap();
        let body = serde_json::json!({
            "collection": self.collection_name,
            "dimension": dimension,
            "head_settings": {
                "semantic": {},
                "temporal": {},
                "structural": {}
            },
            "fields": []
        });

        let resp = client.post(&format!("{}/v1/collections", self.base_url))
            .json(&body)
            .send()
            .await?;

        let result: serde_json::Value = resp.json().await?;
        let success = result["success"].as_bool().unwrap_or(false);
        let message = result["message"].as_str().unwrap_or("");
        if !success && message.contains("already exists") {
            tracing::info!("Collection '{}' already exists, reusing", self.collection_name);
            return Ok(());
        }
        anyhow::ensure!(success, "Failed to create collection: {}", message);
        Ok(())
    }

    async fn teardown(&self) -> anyhow::Result<()> {
        self.id_map.lock().unwrap().clear();
        Ok(())
    }

    async fn disconnect(&self) -> anyhow::Result<()> { Ok(()) }

    async fn insert_batch(&self, documents: &[Document]) -> anyhow::Result<InsertResult> {
        let start = Instant::now();
        let semaphore = Arc::new(Semaphore::new(8));
        let client = self.client.clone().unwrap();
        let base_url = self.base_url.clone();
        let collection_name = self.collection_name.clone();
        let id_map = self.id_map.clone();
        let mut handles = Vec::with_capacity(documents.len());

        for doc in documents {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| anyhow::anyhow!("Semaphore error: {}", e))?;
            let client = client.clone();
            let base_url = base_url.clone();
            let collection_name = collection_name.clone();
            let id_map = id_map.clone();
            let doc = doc.clone();

            handles.push(tokio::spawn(async move {
                let _permit = permit;
                let mut fields = HashMap::new();
                for emb in &doc.embeddings {
                    let key = format!("{}_vector", emb.head_name.name());
                    fields.insert(key, Self::vec_to_csv(&emb.vector));
                }

                let body = serde_json::json!({
                    "collection": collection_name,
                    "fields": fields,
                });

                let resp = client
                    .post(&format!("{}/v1/insert", base_url))
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| anyhow::anyhow!("Insert request: {}", e))?;

                if !resp.status().is_success() {
                    anyhow::bail!("Insert failed: {}", resp.text().await?);
                }

                let resp_json: serde_json::Value = resp.json().await?;
                if let (Some(uuid_str), Some(true)) = (
                    resp_json["id"].as_str(),
                    resp_json["success"].as_bool(),
                ) {
                    if let Ok(parsed) = uuid::Uuid::parse_str(uuid_str) {
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        parsed.hash(&mut hasher);
                        let numeric_id = hasher.finish().to_string();
                        id_map.lock().unwrap().insert(numeric_id, doc.id.clone());
                    }
                }

                Ok::<_, anyhow::Error>(())
            }));
        }

        for handle in handles {
            handle.await??;
        }

        Ok(InsertResult {
            count: documents.len(),
            duration: start.elapsed(),
        })
    }

    async fn flush(&self) -> anyhow::Result<()> { Ok(()) }

    async fn build_index(&self) -> anyhow::Result<Duration> { Ok(Duration::from_secs(0)) }

    async fn query(&self, query: &Query, top_k: usize) -> anyhow::Result<QueryResult> {
        let client = self.client.as_ref().unwrap();
        let start = Instant::now();

        let primary = query.embeddings.first()
            .ok_or_else(|| anyhow::anyhow!("Query has no embeddings"))?;
        let query_str = Self::vec_to_csv(&primary.vector);
        let heads: Vec<&str> = query.enabled_heads.iter().map(|h| h.name()).collect();

        let body = serde_json::json!({
            "collection": self.collection_name,
            "query": query_str,
            "heads": heads,
            "top_k": top_k,
        });

        let resp = client.post(&format!("{}/v1/attend", self.base_url))
            .json(&body)
            .send()
            .await?;

        let latency = start.elapsed();

        if !resp.status().is_success() {
            anyhow::bail!("Query failed: {}", resp.text().await?);
        }

        let result: serde_json::Value = resp.json().await?;
        let id_map_guard = self.id_map.lock().unwrap();
        let ranked_ids: Vec<String> = result["results"].as_array()
            .map(|arr| arr.iter().map(|v| {
                let numeric_id = v["id"].as_str().unwrap_or("");
                id_map_guard.get(numeric_id).cloned().unwrap_or_else(|| numeric_id.to_string())
            }).collect())
            .unwrap_or_default();
        let scores: Vec<f32> = result["results"].as_array()
            .map(|arr| arr.iter().map(|v| v["score"].as_f64().unwrap_or(0.0) as f32).collect())
            .unwrap_or_default();

        Ok(QueryResult { ranked_ids, scores, latency })
    }

    async fn query_single_vector(&self, vector: &[f32], top_k: usize) -> anyhow::Result<QueryResult> {
        let query = Query {
            id: "single_vector_query".into(),
            embeddings: vec![HeadEmbedding {
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
