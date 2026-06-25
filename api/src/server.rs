use crate::auth::ApiKeyStore;
use crate::observability;
use crate::validation;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{debug, warn};

pub mod attentiondb {
    tonic::include_proto!("attentiondb");
}

use attentiondb::attention_db_server::AttentionDb;
use attentiondb::{
    AlterCollectionRequest, AlterCollectionResponse, AttendRequest, AttendResponse, BackupInfo,
    CollectionSettings, CreateBackupRequest, CreateBackupResponse, CreateCollectionRequest,
    CreateCollectionResponse, DeleteRequest, DeleteResponse, GetCollectionSettingsRequest,
    GetCollectionSettingsResponse, HealthRequest, HealthResponse, InsertRequest, InsertResponse,
    ListBackupsRequest, ListBackupsResponse, RestoreBackupRequest, RestoreBackupResponse,
};

#[derive(Clone)]
pub struct AttentionDBService {
    pub engine: Arc<attentiondb_core::engine::AttentionEngine>,
    pub api_keys: Arc<ApiKeyStore>,
}

impl AttentionDBService {
    pub fn new(engine: Arc<attentiondb_core::engine::AttentionEngine>) -> Self {
        Self {
            engine,
            api_keys: Arc::new(ApiKeyStore::disabled()),
        }
    }

    pub fn with_auth(
        engine: Arc<attentiondb_core::engine::AttentionEngine>,
        api_keys: Arc<ApiKeyStore>,
    ) -> Self {
        Self { engine, api_keys }
    }
}

impl Default for AttentionDBService {
    fn default() -> Self {
        Self::new(Arc::new(attentiondb_core::engine::AttentionEngine::new()))
    }
}

#[allow(clippy::result_unit_err)]
pub fn parse_float_vector(s: &str) -> Result<Vec<f32>, ()> {
    let s = s.trim();
    let s = if s.starts_with('[') && s.ends_with(']') {
        &s[1..s.len() - 1]
    } else {
        s
    };
    let parts: Vec<&str> = s
        .split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();
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

#[allow(clippy::result_large_err)]
fn check_grpc_auth(
    api_keys: &ApiKeyStore,
    metadata: &tonic::metadata::MetadataMap,
) -> Result<(), Status> {
    if !api_keys.enabled {
        return Ok(());
    }

    if let Some(val) = metadata.get("authorization") {
        if let Ok(s) = val.to_str() {
            if let Some(token) = s
                .strip_prefix("Bearer ")
                .or_else(|| s.strip_prefix("bearer "))
            {
                if api_keys.validate(token) {
                    return Ok(());
                }
                warn!("gRPC: invalid API key via Authorization header");
                return Err(Status::unauthenticated("Invalid API key"));
            }
        }
    }

    if let Some(val) = metadata.get("x-api-key") {
        if let Ok(s) = val.to_str() {
            if api_keys.validate(s) {
                return Ok(());
            }
            warn!("gRPC: invalid API key via x-api-key header");
            return Err(Status::unauthenticated("Invalid API key"));
        }
    }

    warn!("gRPC: missing API key");
    Err(Status::unauthenticated(
        "API key required. Set Authorization: Bearer <key> or x-api-key metadata.",
    ))
}

#[tonic::async_trait]
impl AttentionDb for AttentionDBService {
    async fn attend(
        &self,
        request: Request<AttendRequest>,
    ) -> Result<Response<AttendResponse>, Status> {
        check_grpc_auth(&self.api_keys, request.metadata())?;
        let req = request.into_inner();
        let _timer = observability::LatencyTimer::new("attend");

        validation::validate_collection_name(&req.collection)?;
        validation::validate_top_k(req.top_k)?;
        validation::validate_heads(&req.heads)?;

        let query_vec = parse_float_vector(&req.query).map_err(|_| {
            Status::invalid_argument(format!("Invalid query vector format: '{}'", req.query))
        })?;
        validation::validate_vector_dimension(query_vec.len())?;

        debug!(collection = %req.collection, heads = ?req.heads, top_k = req.top_k, offset = req.offset, "ATTEND request received");

        let heads = if req.heads.is_empty() || req.heads == ["default"] {
            if let Ok(coll) = self.engine.get_collection(&req.collection) {
                coll.list_heads()
            } else {
                vec!["default".to_string()]
            }
        } else {
            req.heads.clone()
        };

        let offset = req.offset as usize;
        let fetch_count = offset + req.top_k as usize;

        let start = std::time::Instant::now();
        let raw_results = if req.hybrid {
            let query_text = if req.query_text.is_empty() {
                &req.query
            } else {
                &req.query_text
            };
            self.engine
                .attend_hybrid(&req.collection, &heads, &query_vec, query_text, fetch_count)
        } else {
            self.engine
                .attend(&req.collection, &heads, &query_vec, fetch_count)
        }
        .map_err(|e| {
            observability::record_error("attend", &e.to_string());
            Status::internal(e.to_string())
        })?;
        let latency = start.elapsed().as_secs_f64() * 1000.0;

        let total_count = raw_results.len() as u32;
        let paged: Vec<(u64, f32)> = raw_results
            .into_iter()
            .skip(offset)
            .take(req.top_k as usize)
            .collect();
        let has_more = (offset + paged.len()) < total_count as usize;

        let results: Vec<attentiondb::Result> = paged
            .into_iter()
            .map(|(numeric_id, score)| {
                let fields = self.engine.get_document_fields(numeric_id);
                attentiondb::Result {
                    id: numeric_id.to_string(),
                    score,
                    fields,
                }
            })
            .collect();

        observability::record_attend(
            &req.collection,
            &heads,
            req.top_k as usize,
            results.len(),
            latency,
        );

        Ok(Response::new(AttendResponse {
            results,
            latency_ms: latency,
            effective_sample_size: 1.0,
            total_count,
            offset: req.offset,
            has_more,
        }))
    }

    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        check_grpc_auth(&self.api_keys, request.metadata())?;
        let req = request.into_inner();
        let _timer = observability::LatencyTimer::new("insert");

        validation::validate_collection_name(&req.collection)?;
        validation::validate_fields(&req.fields)?;

        let mut json_fields = HashMap::new();
        let mut k_vecs = HashMap::new();

        for (k, v) in &req.fields {
            if let Ok(vec) = parse_float_vector(v) {
                if !vec.is_empty() {
                    let head_name = if k.ends_with("_vector")
                        || k.ends_with("_embedding")
                        || k.ends_with("_head")
                    {
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
            return Err(Status::invalid_argument(
                "No vector embeddings found in fields. Provide at least one field parseable as a float vector."
            ));
        }

        let num_vecs = record.k_vecs.len();
        let start = std::time::Instant::now();
        let id_str = self
            .engine
            .insert_document(&req.collection, record)
            .map_err(|e| {
                observability::record_error("insert", &e.to_string());
                Status::internal(e.to_string())
            })?;
        let latency = start.elapsed().as_secs_f64() * 1000.0;

        observability::record_insert(&req.collection, &id_str, num_vecs, latency);

        Ok(Response::new(InsertResponse {
            id: id_str,
            success: true,
        }))
    }

    async fn delete(
        &self,
        request: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        check_grpc_auth(&self.api_keys, request.metadata())?;
        let req = request.into_inner();
        validation::validate_collection_name(&req.collection)?;

        let success = self
            .engine
            .delete_document(&req.collection, &req.id)
            .map_err(|e| {
                observability::record_error("delete", &e.to_string());
                Status::internal(e.to_string())
            })?;

        observability::record_delete(&req.collection, &req.id, success);
        Ok(Response::new(DeleteResponse { success }))
    }

    async fn health_check(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let stats = self.engine.stats();
        observability::record_engine_stats(
            stats.collection_count,
            stats.total_heads,
            stats.total_vectors,
        );

        Ok(Response::new(HealthResponse {
            status: format!(
                "healthy (collections: {}, heads: {}, vectors: {})",
                stats.collection_count, stats.total_heads, stats.total_vectors
            ),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }))
    }

    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        validation::validate_collection_name(&req.collection)?;

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

        hnsw_settings.validate().map_err(Status::invalid_argument)?;

        let heads: Vec<String> = if req.head_settings.is_empty() {
            vec!["default".to_string()]
        } else {
            req.head_settings.keys().cloned().collect()
        };
        validation::validate_heads(&heads)?;
        let head_refs: Vec<&str> = heads.iter().map(|s| s.as_str()).collect();

        let dimension = if req.dimension > 0 {
            validation::validate_vector_dimension(req.dimension as usize)?;
            req.dimension as usize
        } else {
            64
        };

        self.engine
            .create_collection_with_settings(
                &req.collection,
                dimension,
                &head_refs,
                hnsw_settings.clone(),
            )
            .map_err(|e| {
                observability::record_error("create_collection", &e.to_string());
                Status::internal(e.to_string())
            })?;

        observability::record_create_collection(
            &req.collection,
            &head_refs,
            hnsw_settings.ef_search,
        );

        let per_head_count = req.head_settings.len();
        let mut msg = format!(
            "Created collection '{}' with settings (ef_search={}, ef_construction={}, max_connections={}, similarity={}, exact_rerank={}, enable_gpu_fusion={}, enable_gpu_projections={})",
            req.collection,
            hnsw_settings.ef_search,
            hnsw_settings.ef_construction,
            hnsw_settings.max_nb_connection,
            hnsw_settings.similarity_metric,
            hnsw_settings.enable_exact_reranking,
            hnsw_settings.enable_gpu_fusion,
            hnsw_settings.enable_gpu_projections,
        );

        if per_head_count > 0 {
            let head_info: Vec<String> = req
                .head_settings
                .iter()
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
        check_grpc_auth(&self.api_keys, request.metadata())?;
        let req = request.into_inner();
        validation::validate_collection_name(&req.collection)?;
        let coll = self
            .engine
            .get_collection(&req.collection)
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
        check_grpc_auth(&self.api_keys, request.metadata())?;
        let req = request.into_inner();
        validation::validate_collection_name(&req.collection)?;

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

        hnsw_settings.validate().map_err(Status::invalid_argument)?;

        self.engine
            .alter_collection_settings(&req.collection, hnsw_settings.clone())
            .map_err(|e| {
                observability::record_error("alter_collection", &e.to_string());
                Status::internal(e.to_string())
            })?;

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
            let head_info: Vec<String> = req
                .head_settings
                .iter()
                .map(|(name, s)| format!("{}: (ef_search={})", name, s.ef_search))
                .collect();
            msg.push_str(&format!(". Per-head settings: [{}]", head_info.join(", ")));
        }

        Ok(Response::new(AlterCollectionResponse {
            success: true,
            message: msg,
        }))
    }

    async fn create_backup(
        &self,
        request: Request<CreateBackupRequest>,
    ) -> Result<Response<CreateBackupResponse>, Status> {
        check_grpc_auth(&self.api_keys, request.metadata())?;
        let req = request.into_inner();
        let engine = &self.engine;
        let collections = engine.list_collections();

        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let backup_id = format!("backup_{}", timestamp);

        let dest = if req.destination.is_empty() {
            std::path::PathBuf::from(
                std::env::var("ATTENTIONDB_DATA_DIR").unwrap_or_else(|_| "/data".into()),
            )
            .parent()
            .unwrap_or(std::path::Path::new("/data"))
            .join("backups")
            .join(&backup_id)
        } else {
            std::path::PathBuf::from(&req.destination)
        };
        std::fs::create_dir_all(&dest)
            .map_err(|e| Status::internal(format!("Failed to create backup directory: {}", e)))?;

        let mut backed_up_collections = Vec::new();
        let mut total_bytes = 0u64;

        for coll_name in &collections {
            let coll = engine
                .get_collection(coll_name)
                .map_err(|e| Status::internal(e.to_string()))?;
            let heads = coll.list_heads();
            let coll_dir = dest.join(coll_name);
            std::fs::create_dir_all(&coll_dir).map_err(|e| Status::internal(e.to_string()))?;

            for head_name in &heads {
                let head_dir = coll_dir.join(head_name);
                std::fs::create_dir_all(&head_dir).map_err(|e| Status::internal(e.to_string()))?;

                match coll.head_manager.read().get_head(head_name) {
                    Ok(idx) => {
                        let idx_guard = idx.read();
                        attentiondb_hnsw::persistence::save_index(&idx_guard, &head_dir).map_err(
                            |e| Status::internal(format!("Failed to save index: {}", e)),
                        )?;
                        if let Ok(meta) = std::fs::metadata(&head_dir) {
                            total_bytes += meta.len();
                        }
                    }
                    Err(e) => {
                        return Err(Status::internal(format!("Head not found: {}", e)));
                    }
                }
            }
            backed_up_collections.push(coll_name.clone());
        }

        // Save manifest
        std::fs::write(
            dest.join("manifest.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "backup_id": backup_id,
                "timestamp": timestamp,
                "collections": backed_up_collections,
            }))
            .map_err(|e| Status::internal(e.to_string()))?,
        )
        .map_err(|e| Status::internal(e.to_string()))?;

        // Copy WAL if persistent
        if engine.is_persistent() {
            let wal_path = std::path::PathBuf::from(
                std::env::var("ATTENTIONDB_DATA_DIR").unwrap_or_else(|_| "/data".into()),
            )
            .join("engine.wal");
            if wal_path.exists() {
                std::fs::copy(&wal_path, dest.join("engine.wal"))
                    .map_err(|e| Status::internal(format!("Failed to copy WAL: {}", e)))?;
                if let Ok(meta) = wal_path.metadata() {
                    total_bytes += meta.len();
                }
            }
        }

        Ok(Response::new(CreateBackupResponse {
            backup_id,
            timestamp,
            collections: backed_up_collections,
            path: dest.to_string_lossy().to_string(),
            size_bytes: total_bytes,
        }))
    }

    async fn list_backups(
        &self,
        _request: Request<ListBackupsRequest>,
    ) -> Result<Response<ListBackupsResponse>, Status> {
        check_grpc_auth(&self.api_keys, _request.metadata())?;
        let data_dir = std::env::var("ATTENTIONDB_DATA_DIR").unwrap_or_else(|_| "/data".into());
        let backup_root = std::path::Path::new(&data_dir)
            .parent()
            .unwrap_or(std::path::Path::new("/data"))
            .join("backups");

        let mut backups = Vec::new();
        if backup_root.exists() {
            for entry in std::fs::read_dir(&backup_root)
                .map_err(|e| Status::internal(e.to_string()))?
                .flatten()
            {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let dir_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();
                if !dir_name.starts_with("backup_") {
                    continue;
                }

                let mut collections = Vec::new();
                let mut size_bytes = 0u64;
                if let Ok(read_dir) = std::fs::read_dir(&path) {
                    for sub in read_dir.flatten() {
                        let sub_path = sub.path();
                        if sub_path.is_dir() {
                            collections.push(sub.file_name().to_string_lossy().to_string());
                        }
                        if let Ok(meta) = sub_path.metadata() {
                            size_bytes += meta.len();
                        }
                    }
                }

                backups.push(BackupInfo {
                    backup_id: dir_name.clone(),
                    timestamp: dir_name.trim_start_matches("backup_").replace('_', " "),
                    collections,
                    path: path.to_string_lossy().to_string(),
                    size_bytes,
                });
            }
        }

        backups.sort_by(|a, b| b.backup_id.cmp(&a.backup_id));
        Ok(Response::new(ListBackupsResponse { backups }))
    }

    async fn restore_backup(
        &self,
        request: Request<RestoreBackupRequest>,
    ) -> Result<Response<RestoreBackupResponse>, Status> {
        check_grpc_auth(&self.api_keys, request.metadata())?;
        let req = request.into_inner();
        let data_dir = std::env::var("ATTENTIONDB_DATA_DIR").unwrap_or_else(|_| "/data".into());
        let backup_dir = std::path::Path::new(&data_dir)
            .parent()
            .unwrap_or(std::path::Path::new("/data"))
            .join("backups")
            .join(&req.backup_id);

        if !backup_dir.exists() {
            return Err(Status::not_found(format!(
                "Backup '{}' not found",
                req.backup_id
            )));
        }

        let manifest_path = backup_dir.join("manifest.json");
        if !manifest_path.exists() {
            return Err(Status::failed_precondition(format!(
                "Backup '{}' is corrupted — manifest missing",
                req.backup_id
            )));
        }

        Ok(Response::new(RestoreBackupResponse {
            success: true,
            message: format!("Backup '{}' validated. To restore, restart the server with data directory pointing to this backup.", req.backup_id),
        }))
    }
}
