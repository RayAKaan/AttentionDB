use tonic::{Request, Response, Status};
use std::collections::HashMap;
use std::sync::Arc;

pub mod attentiondb {
    tonic::include_proto!("attentiondb");
}

use attentiondb::attention_db_server::AttentionDb;
use attentiondb::{
    AttendRequest, AttendResponse, InsertRequest, InsertResponse,
    DeleteRequest, DeleteResponse, HealthRequest, HealthResponse,
    CreateCollectionRequest, CreateCollectionResponse,
    GetCollectionSettingsRequest, GetCollectionSettingsResponse,
    AlterCollectionRequest, AlterCollectionResponse,
    CollectionSettings,
};

#[derive(Clone)]
pub struct AttentionDBService {
    pub engine: Arc<attentiondb_core::engine::AttentionEngine>,
}

impl AttentionDBService {
    pub fn new(engine: Arc<attentiondb_core::engine::AttentionEngine>) -> Self {
        Self { engine }
    }
}

impl Default for AttentionDBService {
    fn default() -> Self {
        Self::new(Arc::new(attentiondb_core::engine::AttentionEngine::new()))
    }
}

pub fn parse_float_vector(s: &str) -> Result<Vec<f32>, ()> {
    let s = s.trim();
    let s = if s.starts_with('[') && s.ends_with(']') {
        &s[1..s.len()-1]
    } else {
        s
    };
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        return Err(());
    }
    let mut vec = Vec::with_capacity(parts.len());
    for p in parts {
        if let Ok(f) = p.parse::<f32>() {
            vec.push(f);
        } else {
            return Err(());
        }
    }
    Ok(vec)
}

#[tonic::async_trait]
impl AttentionDb for AttentionDBService {
    async fn attend(
        &self,
        request: Request<AttendRequest>,
    ) -> Result<Response<AttendResponse>, Status> {
        let req = request.into_inner();
        println!("[gRPC] Attend: collection={}, query=\"{}\", heads={:?}, top_k={}",
                 req.collection, req.query, req.heads, req.top_k);

        let query_vec = parse_float_vector(&req.query).unwrap_or_else(|_| vec![0.1; 64]);

        let heads = if req.heads.is_empty() || req.heads == ["default"] {
            if let Ok(coll) = self.engine.get_collection(&req.collection) {
                coll.list_heads()
            } else {
                vec!["default".to_string()]
            }
        } else {
            req.heads.clone()
        };

        let start = std::time::Instant::now();
        let raw_results = self.engine.attend(&req.collection, &heads, &query_vec, req.top_k as usize)
            .map_err(|e| Status::internal(e.to_string()))?;
        let latency = start.elapsed().as_secs_f64() * 1000.0;

        let results = raw_results.into_iter().map(|(numeric_id, score)| {
            let fields = self.engine.get_document_fields(numeric_id);
            attentiondb::Result {
                id: numeric_id.to_string(),
                score,
                fields,
            }
        }).collect();

        Ok(Response::new(AttendResponse {
            results,
            latency_ms: latency,
            effective_sample_size: 1.0,
        }))
    }

    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();

        let mut json_fields = HashMap::new();
        let mut k_vecs = HashMap::new();

        for (k, v) in &req.fields {
            if let Ok(vec) = parse_float_vector(v) {
                if !vec.is_empty() {
                    let head_name = if k.ends_with("_vector") || k.ends_with("_embedding") || k.ends_with("_head") {
                        k.split('_').next().unwrap_or("default").to_string()
                    } else {
                        k.clone()
                    };
                    k_vecs.insert(head_name, vec);
                }
            }
            json_fields.insert(k.clone(), serde_json::Value::String(v.clone()));
        }

        let mut record = attentiondb_storage::Record::new(json_fields);
        record.k_vecs = k_vecs;

        if record.k_vecs.is_empty() {
            record.k_vecs.insert("default".to_string(), vec![0.1; 64]);
        }

        let id_str = self.engine.insert_document(&req.collection, record)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(InsertResponse {
            id: id_str,
            success: true,
        }))
    }

    async fn delete(
        &self,
        request: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        let req = request.into_inner();
        let success = self.engine.delete_document(&req.collection, &req.id)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(DeleteResponse { success }))
    }

    async fn health_check(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let stats = self.engine.stats();
        Ok(Response::new(HealthResponse {
            status: format!("healthy (collections: {}, heads: {}, vectors: {})", stats.collection_count, stats.total_heads, stats.total_vectors),
            version: "0.5.0".to_string(),
        }))
    }

    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        println!("[gRPC] CreateCollection: collection={}", req.collection);

        let mut hnsw_settings = attentiondb_hnsw::CollectionSettings::default();
        if let Some(ref s) = req.settings {
            hnsw_settings.ef_search = s.ef_search as usize;
            hnsw_settings.ef_construction = s.ef_construction as usize;
            hnsw_settings.max_nb_connection = s.max_connections as usize;
            hnsw_settings.similarity_metric = s.similarity.clone();
            hnsw_settings.enable_exact_reranking = s.exact_rerank;
            hnsw_settings.enable_gpu_fusion = s.enable_gpu_fusion;
            hnsw_settings.enable_gpu_projections = s.enable_gpu_projections;
        }

        hnsw_settings.validate().map_err(|e| Status::invalid_argument(e))?;

        let heads: Vec<String> = if req.head_settings.is_empty() {
            vec!["default".to_string()]
        } else {
            req.head_settings.keys().cloned().collect()
        };
        let head_refs: Vec<&str> = heads.iter().map(|s| s.as_str()).collect();

        self.engine.create_collection_with_settings(&req.collection, 64, &head_refs, hnsw_settings.clone())
            .map_err(|e| Status::internal(e.to_string()))?;

        let field_info: Vec<String> = req.fields.iter()
            .map(|f| format!("{}:{}", f.name, f.r#type))
            .collect();

        let per_head_count = req.head_settings.len();

        let mut msg = format!(
            "Created collection '{}' with fields [{}] and settings (ef_search={}, ef_construction={}, max_connections={}, similarity={}, exact_rerank={})",
            req.collection,
            field_info.join(", "),
            hnsw_settings.ef_search,
            hnsw_settings.ef_construction,
            hnsw_settings.max_nb_connection,
            hnsw_settings.similarity_metric,
            hnsw_settings.enable_exact_reranking,
        );

        if per_head_count > 0 {
            let head_info: Vec<String> = req.head_settings.iter()
                .map(|(name, s)| format!("{}: (ef_search={})", name, s.ef_search))
                .collect();
            msg.push_str(&format!(". Per-head settings: [{}]", head_info.join(", ")));
        }

        Ok(Response::new(CreateCollectionResponse {
            success: true,
            message: msg,
        }))
    }

    async fn get_collection_settings(
        &self,
        request: Request<GetCollectionSettingsRequest>,
    ) -> Result<Response<GetCollectionSettingsResponse>, Status> {
        let req = request.into_inner();
        let coll = self.engine.get_collection(&req.collection)
            .map_err(|e| Status::not_found(e.to_string()))?;

        let s = coll.settings.read();
        let settings = CollectionSettings {
            ef_search: s.ef_search as u32,
            ef_construction: s.ef_construction as u32,
            max_connections: s.max_nb_connection as u32,
            similarity: s.similarity_metric.clone(),
            exact_rerank: s.enable_exact_reranking,
            enable_gpu_fusion: s.enable_gpu_fusion,
            enable_gpu_projections: s.enable_gpu_projections,
        };
        Ok(Response::new(GetCollectionSettingsResponse {
            settings: Some(settings),
        }))
    }

    async fn alter_collection(
        &self,
        request: Request<AlterCollectionRequest>,
    ) -> Result<Response<AlterCollectionResponse>, Status> {
        let req = request.into_inner();
        println!("[gRPC] AlterCollection: collection={}", req.collection);

        let s = req.settings.unwrap_or_default();
        let hnsw_settings = attentiondb_hnsw::CollectionSettings {
            ef_search: s.ef_search as usize,
            ef_construction: s.ef_construction as usize,
            max_nb_connection: s.max_connections as usize,
            similarity_metric: s.similarity,
            enable_exact_reranking: s.exact_rerank,
            enable_gpu_fusion: s.enable_gpu_fusion,
            enable_gpu_projections: s.enable_gpu_projections,
        };

        hnsw_settings.validate().map_err(|e| Status::invalid_argument(e))?;

        self.engine.alter_collection_settings(&req.collection, hnsw_settings.clone())
            .map_err(|e| Status::internal(e.to_string()))?;

        let per_head_count = req.head_settings.len();

        let mut msg = format!(
            "Altered collection '{}' settings to (ef_search={}, ef_construction={}, max_connections={}, similarity={}, exact_rerank={})",
            req.collection,
            hnsw_settings.ef_search,
            hnsw_settings.ef_construction,
            hnsw_settings.max_nb_connection,
            hnsw_settings.similarity_metric,
            hnsw_settings.enable_exact_reranking,
        );

        if per_head_count > 0 {
            let head_info: Vec<String> = req.head_settings.iter()
                .map(|(name, s)| format!("{}: (ef_search={})", name, s.ef_search))
                .collect();
            msg.push_str(&format!(". Per-head settings: [{}]", head_info.join(", ")));
        }

        Ok(Response::new(AlterCollectionResponse {
            success: true,
            message: msg,
        }))
    }
}
