use crate::auth::{auth_middleware, ApiKeyStore};
use crate::observability;
use crate::openapi;
use crate::rate_limiter::{rate_limit_middleware, RateLimiter};
use crate::server::AttentionDBService;
use crate::validation::{
    validate_collection_name, validate_fields, validate_heads, validate_top_k,
    validate_vector_dimension,
};
use axum::{
    extract::{Extension, Path, State},
    http::StatusCode,
    middleware::from_fn,
    response::IntoResponse,
    routing::{get, post, put},
    Json, Router,
};
use metrics_exporter_prometheus::PrometheusHandle;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Parsed document payload: (fields, per-head vectors).
type ParsedDoc = (
    std::collections::HashMap<String, serde_json::Value>,
    std::collections::HashMap<String, Vec<f32>>,
);

#[derive(Clone)]
pub struct AppState {
    pub service: Arc<AttentionDBService>,
    pub api_keys: Arc<ApiKeyStore>,
    pub metrics: Option<Arc<PrometheusHandle>>,
    pub rate_limiter: Arc<RateLimiter>,
    pub semaphore: Arc<tokio::sync::Semaphore>,
}

#[derive(Deserialize)]
pub struct AttendRequest {
    pub collection: String,
    pub query: String,
    pub heads: Option<Vec<String>>,
    pub top_k: Option<u32>,
    pub min_weight: Option<f32>,
    pub temporal_decay: Option<f32>,
    pub offset: Option<u32>,
    pub hybrid: Option<bool>,
    pub bm25_weight: Option<f32>,
    pub vector_weight: Option<f32>,
    pub query_text: Option<String>,
    /// Optional structured metadata filter (Phase 2 §11):
    /// `{"field":"category","op":"=","value":"book"}`, `{"and":[...]}`,
    /// `{"or":[...]}`, `{"not":{...}}`, `{"op":"in|not_in|is_null|is_not_null"}`.
    pub filter: Option<serde_json::Value>,
    /// Optional query timeout in milliseconds (§19); enforced at pipeline
    /// stage boundaries. 1..=60000.
    pub timeout_ms: Option<u64>,
}

#[derive(Serialize)]
pub struct AttendResponse {
    pub results: Vec<serde_json::Value>,
    pub latency_ms: f64,
    pub effective_sample_size: f32,
    pub total_count: u32,
    pub offset: u32,
    pub has_more: bool,
}

#[derive(Deserialize)]
pub struct InsertRestRequest {
    pub collection: String,
    pub fields: std::collections::HashMap<String, String>,
}

#[derive(Serialize)]
pub struct InsertRestResponse {
    pub id: String,
    pub success: bool,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

#[derive(Deserialize)]
pub struct FieldDefinition {
    pub name: String,
    pub r#type: String,
}

#[derive(Deserialize, Serialize)]
pub struct CollectionSettingsRest {
    pub ef_search: Option<u32>,
    pub ef_construction: Option<u32>,
    pub max_connections: Option<u32>,
    pub similarity: Option<String>,
    pub exact_rerank: Option<bool>,
    pub enable_gpu_fusion: Option<bool>,
    pub enable_gpu_projections: Option<bool>,
}

#[derive(Deserialize)]
pub struct CreateCollectionRestRequest {
    pub collection: String,
    pub fields: Vec<FieldDefinition>,
    pub settings: Option<CollectionSettingsRest>,
    pub head_settings: Option<std::collections::HashMap<String, CollectionSettingsRest>>,
    pub dimension: Option<u32>,
}

#[derive(Serialize)]
pub struct CreateCollectionRestResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Deserialize)]
pub struct AlterCollectionRestRequest {
    pub settings: CollectionSettingsRest,
    pub head_settings: Option<std::collections::HashMap<String, CollectionSettingsRest>>,
}

#[derive(Serialize)]
pub struct AlterCollectionRestResponse {
    pub success: bool,
    pub message: String,
}

/// Phase 2 §17 — EXPLAIN: plan a query without executing it.
#[derive(Deserialize)]
pub struct ExplainRequest {
    pub collection: String,
    pub heads: Option<Vec<String>>,
    pub top_k: Option<u32>,
    pub filter: Option<serde_json::Value>,
    pub query_text: Option<String>,
}

pub async fn explain_handler(
    State(state): State<AppState>,
    Json(payload): Json<ExplainRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    validate_collection_name(&payload.collection)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;
    let top_k = payload.top_k.unwrap_or(10);
    let filter = match &payload.filter {
        Some(f) => Some(
            attentiondb_query::filter::parse_filter_json(f)
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid filter: {e}")))?,
        ),
        None => None,
    };
    let has_text = payload.query_text.is_some();
    let heads = payload.heads.unwrap_or_default();
    let text = state
        .service
        .engine
        .explain(
            &payload.collection,
            &heads,
            top_k as usize,
            filter.as_ref(),
            has_text,
        )
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(serde_json::json!({ "plan": text })))
}

pub async fn attend_handler(
    State(state): State<AppState>,
    Json(payload): Json<AttendRequest>,
) -> Result<Json<AttendResponse>, (StatusCode, String)> {
    let _permit = state.semaphore.acquire().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Too many concurrent requests".to_string(),
        )
    })?;
    let _timer = observability::LatencyTimer::new("rest_attend");

    validate_collection_name(&payload.collection)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;
    let top_k = payload.top_k.unwrap_or(10);
    validate_top_k(top_k).map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;
    let offset = payload.offset.unwrap_or(0);

    if let Some(ref heads_list) = payload.heads {
        validate_heads(heads_list)
            .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;
    }

    let query_vec = crate::server::parse_float_vector(&payload.query).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            "Invalid query vector format".to_string(),
        )
    })?;
    validate_vector_dimension(query_vec.len())
        .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;

    let heads = payload.heads.unwrap_or_else(|| {
        if let Ok(coll) = state.service.engine.get_collection(&payload.collection) {
            coll.list_heads()
        } else {
            vec!["default".to_string()]
        }
    });

    let use_hybrid = payload.hybrid.unwrap_or(false);
    let filter_expr = match &payload.filter {
        Some(f) => Some(
            attentiondb_query::filter::parse_filter_json(f)
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid filter: {e}")))?,
        ),
        None => None,
    };

    let offset_usize = offset as usize;
    let fetch_count = offset_usize + top_k as usize;
    let text = payload.query_text.clone().unwrap_or_default();
    let start = std::time::Instant::now();
    let hybrid_filter = filter_expr.as_ref().filter(|_| use_hybrid);
    let raw_results = if let Some(f) = hybrid_filter {
        // §13: filter+text runs BOTH hybrid channels through the filter —
        // never vector-only (the old precedence bug, fixed).
        state
            .service
            .engine
            .attend_hybrid_filtered_with_deadline(
                &payload.collection,
                &heads,
                &query_vec,
                text.as_str(),
                fetch_count,
                f,
                None,
            )
            .map_err(|e| {
                observability::record_query_error("attend");
                observability::record_error("rest_attend", &e.to_string());
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            })?
    } else if filter_expr.is_some() {
        // Filtered path: post-filter + bounded candidate expansion. The filter
        // is applied AFTER pagination math so fetch covers expansion rounds.
        state
            .service
            .engine
            .attend_filtered(
                &payload.collection,
                &heads,
                &query_vec,
                fetch_count,
                filter_expr.as_ref(),
            )
            .map_err(|e| {
                observability::record_query_error("attend");
                observability::record_error("rest_attend", &e.to_string());
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            })?
    } else if let Some(timeout_ms) = payload.timeout_ms {
        state
            .service
            .engine
            .attend_with_deadline_ms(
                &payload.collection,
                &heads,
                &query_vec,
                fetch_count,
                timeout_ms,
            )
            .map_err(|e| {
                let status = if matches!(e, attentiondb_core::CoreError::Timeout(_))
                    || matches!(e, attentiondb_core::CoreError::InvalidArgument(_))
                {
                    if e.to_string().contains("deadline") {
                        StatusCode::GATEWAY_TIMEOUT
                    } else {
                        StatusCode::BAD_REQUEST
                    }
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                observability::record_query_error("attend");
                observability::record_error("rest_attend", &e.to_string());
                (status, e.to_string())
            })?
    } else if use_hybrid {
        let query_text = payload.query_text.as_deref().unwrap_or(&payload.query);
        state
            .service
            .engine
            .attend_hybrid(
                &payload.collection,
                &heads,
                &query_vec,
                query_text,
                fetch_count,
            )
            .map_err(|e| {
                observability::record_query_error("attend");
                observability::record_error("rest_attend", &e.to_string());
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            })?
    } else {
        state
            .service
            .engine
            .attend(&payload.collection, &heads, &query_vec, fetch_count)
            .map_err(|e| {
                observability::record_query_error("attend");
                observability::record_error("rest_attend", &e.to_string());
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            })?
    };
    let latency = start.elapsed().as_secs_f64() * 1000.0;

    let total_count = raw_results.len() as u32;
    let n_results = total_count as usize;
    let paged: Vec<(u64, f32)> = raw_results
        .into_iter()
        .skip(offset_usize)
        .take(top_k as usize)
        .collect();
    let has_more = (offset_usize + paged.len()) < total_count as usize;

    let results: Vec<serde_json::Value> = paged
        .into_iter()
        .map(|(numeric_id, score)| {
            let fields = state.service.engine.get_document_fields(numeric_id);
            serde_json::json!({
                "id": numeric_id.to_string(),
                "score": score,
                "fields": fields,
            })
        })
        .collect();

    observability::record_query(
        &payload.collection,
        if use_hybrid {
            "hybrid"
        } else if filter_expr.is_some() {
            "filtered"
        } else {
            "vector"
        },
        n_results,
        start.elapsed().as_secs_f64(),
    );
    observability::record_attend(
        &payload.collection,
        &heads,
        top_k as usize,
        results.len(),
        latency,
    );

    Ok(Json(AttendResponse {
        results,
        latency_ms: latency,
        effective_sample_size: 1.0,
        total_count,
        offset,
        has_more,
    }))
}

pub async fn insert_handler(
    State(state): State<AppState>,
    Json(payload): Json<InsertRestRequest>,
) -> Result<Json<InsertRestResponse>, (StatusCode, String)> {
    let _permit = state.semaphore.acquire().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Too many concurrent requests".to_string(),
        )
    })?;
    let _timer = observability::LatencyTimer::new("rest_insert");

    validate_collection_name(&payload.collection)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;
    validate_fields(&payload.fields)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;

    let mut json_fields = std::collections::HashMap::new();
    let mut k_vecs = std::collections::HashMap::new();

    for (k, v) in &payload.fields {
        if let Ok(vec) = crate::server::parse_float_vector(v) {
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

    if k_vecs.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "No vector embeddings found in fields".to_string(),
        ));
    }

    let num_vectors = k_vecs.len();
    let mut record = attentiondb_storage::Record::new(json_fields);
    record.k_vecs = k_vecs;

    let start = std::time::Instant::now();
    let id = state
        .service
        .engine
        .insert_document(&payload.collection, record)
        .map_err(|e| {
            observability::record_error("rest_insert", &e.to_string());
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;
    let latency = start.elapsed().as_secs_f64() * 1000.0;

    observability::record_insert(&payload.collection, &id, num_vectors, latency);

    Ok(Json(InsertRestResponse { id, success: true }))
}

// ==================== Phase 1: update / upsert / delete / admin ====================

#[derive(Deserialize)]
pub struct UpdateRestRequest {
    pub collection: String,
    pub id: String,
    pub fields: std::collections::HashMap<String, String>,
}

#[derive(Deserialize)]
pub struct UpsertRestRequest {
    pub collection: String,
    /// optional: generated when absent (insert semantics)
    pub id: Option<String>,
    pub fields: std::collections::HashMap<String, String>,
}

#[derive(Deserialize)]
pub struct DeleteRestRequest {
    pub collection: String,
    pub id: String,
}

#[derive(Serialize)]
pub struct MutateRestResponse {
    pub success: bool,
    pub id: String,
    pub message: String,
}

#[derive(Serialize)]
pub struct CheckpointRestResponse {
    pub success: bool,
    pub checkpoint_seq: u64,
    pub manifest_generation: u64,
    pub duration_ms: f64,
}

#[derive(Serialize)]
pub struct CheckRestResponse {
    pub ok: bool,
    pub errors: usize,
    pub warnings: usize,
    pub issues: Vec<attentiondb_core::CheckIssue>,
}

/// Shared field→(json fields, k_vecs) parsing used by insert/update/upsert.
fn parse_fields(
    fields: &std::collections::HashMap<String, String>,
) -> Result<ParsedDoc, (StatusCode, String)> {
    let mut json_fields = std::collections::HashMap::new();
    let mut k_vecs = std::collections::HashMap::new();
    for (k, v) in fields {
        if let Ok(vec) = crate::server::parse_float_vector(v) {
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
    if k_vecs.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "No vector embeddings found in fields".to_string(),
        ));
    }
    Ok((json_fields, k_vecs))
}

fn check_mutation(err: attentiondb_core::CoreError, op: &'static str) -> (StatusCode, String) {
    observability::record_error(op, &err.to_string());
    let status = match err.category() {
        "NotFound" => StatusCode::NOT_FOUND,
        "AlreadyExists" => StatusCode::CONFLICT,
        "InvalidArgument" | "InvalidOperation" => StatusCode::BAD_REQUEST,
        "ResourceExhausted" => StatusCode::INSUFFICIENT_STORAGE,
        "Unavailable" => StatusCode::SERVICE_UNAVAILABLE,
        "Conflict" => StatusCode::CONFLICT,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    // (status already derived from the variant name above; message via Display)
    (status, err.to_string())
}

pub async fn update_handler(
    State(state): State<AppState>,
    Json(payload): Json<UpdateRestRequest>,
) -> Result<Json<MutateRestResponse>, (StatusCode, String)> {
    let _permit = state.semaphore.acquire().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Too many concurrent requests".to_string(),
        )
    })?;
    let _timer = observability::LatencyTimer::new("rest_update");
    validate_collection_name(&payload.collection)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;
    let (json_fields, k_vecs) = parse_fields(&payload.fields)?;
    let id = state
        .service
        .engine
        .update_document(&payload.collection, &payload.id, json_fields, k_vecs)
        .map_err(|e| check_mutation(e, "rest_update"))?;
    Ok(Json(MutateRestResponse {
        success: true,
        id,
        message: "updated".to_string(),
    }))
}

pub async fn upsert_handler(
    State(state): State<AppState>,
    Json(payload): Json<UpsertRestRequest>,
) -> Result<Json<MutateRestResponse>, (StatusCode, String)> {
    let _permit = state.semaphore.acquire().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Too many concurrent requests".to_string(),
        )
    })?;
    let _timer = observability::LatencyTimer::new("rest_upsert");
    validate_collection_name(&payload.collection)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;
    let (json_fields, k_vecs) = parse_fields(&payload.fields)?;
    let uuid = match &payload.id {
        Some(id) => uuid::Uuid::parse_str(id).map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                format!("invalid document id '{id}'"),
            )
        })?,
        None => uuid::Uuid::new_v4(),
    };
    let id = state
        .service
        .engine
        .upsert_document(&payload.collection, uuid, json_fields, k_vecs)
        .map_err(|e| check_mutation(e, "rest_upsert"))?;
    Ok(Json(MutateRestResponse {
        success: true,
        id,
        message: "upserted".to_string(),
    }))
}

pub async fn delete_doc_handler(
    State(state): State<AppState>,
    Json(payload): Json<DeleteRestRequest>,
) -> Result<Json<MutateRestResponse>, (StatusCode, String)> {
    let _permit = state.semaphore.acquire().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Too many concurrent requests".to_string(),
        )
    })?;
    let _timer = observability::LatencyTimer::new("rest_delete");
    validate_collection_name(&payload.collection)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;
    let deleted = state
        .service
        .engine
        .delete_document(&payload.collection, &payload.id)
        .map_err(|e| check_mutation(e, "rest_delete"))?;
    if deleted {
        Ok(Json(MutateRestResponse {
            success: true,
            id: payload.id,
            message: "deleted".to_string(),
        }))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            format!("document '{}' not found", payload.id),
        ))
    }
}

pub async fn checkpoint_handler(
    State(state): State<AppState>,
) -> Result<Json<CheckpointRestResponse>, (StatusCode, String)> {
    let _timer = observability::LatencyTimer::new("rest_checkpoint");
    let info = state
        .service
        .engine
        .checkpoint()
        .map_err(|e| check_mutation(e, "rest_checkpoint"))?;
    Ok(Json(CheckpointRestResponse {
        success: true,
        checkpoint_seq: info.checkpoint_seq,
        manifest_generation: info.manifest_generation,
        duration_ms: info.duration_ms,
    }))
}

pub async fn check_handler(
    State(state): State<AppState>,
) -> Result<Json<CheckRestResponse>, (StatusCode, String)> {
    let issues = attentiondb_core::checker::check_engine(&state.service.engine);
    let errors = issues.iter().filter(|i| i.severity.is_error()).count();
    let warnings = issues.len() - errors;
    Ok(Json(CheckRestResponse {
        ok: errors == 0,
        errors,
        warnings,
        issues,
    }))
}

pub async fn liveness_handler() -> (StatusCode, &'static str) {
    (StatusCode::OK, "alive")
}

pub async fn readiness_handler(State(state): State<AppState>) -> (StatusCode, String) {
    // INV-11: READY only after recovery completed; never expose a half-open DB.
    let is_healthy = state.service.engine.is_ready();
    if is_healthy {
        let stats = state.service.engine.stats();
        (
            StatusCode::OK,
            format!(
                "ready (collections: {}, heads: {}, vectors: {})",
                stats.collection_count, stats.total_heads, stats.total_vectors
            ),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "not ready — engine not fully initialized".to_string(),
        )
    }
}

pub async fn startup_handler() -> (StatusCode, &'static str) {
    (StatusCode::OK, "startup complete")
}

pub async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    let stats = state.service.engine.stats();
    observability::record_engine_stats(
        stats.collection_count,
        stats.total_heads,
        stats.total_vectors,
    );
    Json(HealthResponse {
        status: format!(
            "healthy (collections: {}, heads: {}, vectors: {})",
            stats.collection_count, stats.total_heads, stats.total_vectors
        ),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

pub async fn create_collection_handler(
    State(state): State<AppState>,
    Json(payload): Json<CreateCollectionRestRequest>,
) -> Json<CreateCollectionRestResponse> {
    if let Err(e) = validate_collection_name(&payload.collection) {
        return Json(CreateCollectionRestResponse {
            success: false,
            message: e.message().to_string(),
        });
    }
    let mut hnsw_settings = attentiondb_hnsw::CollectionSettings::default();
    if let Some(ref s) = payload.settings {
        hnsw_settings.ef_search = s.ef_search.unwrap_or(64) as usize;
        hnsw_settings.ef_construction = s.ef_construction.unwrap_or(400) as usize;
        hnsw_settings.max_nb_connection = s.max_connections.unwrap_or(16) as usize;
        hnsw_settings.similarity_metric =
            s.similarity.clone().unwrap_or_else(|| "cosine".to_string());
        hnsw_settings.enable_exact_reranking = s.exact_rerank.unwrap_or(true);
        hnsw_settings.enable_gpu_fusion = s.enable_gpu_fusion.unwrap_or(false);
        hnsw_settings.enable_gpu_projections = s.enable_gpu_projections.unwrap_or(false);
    }

    let heads: Vec<String> = if let Some(ref hm) = payload.head_settings {
        if hm.is_empty() {
            vec!["default".to_string()]
        } else {
            let head_names: Vec<String> = hm.keys().cloned().collect();
            if let Err(e) = validate_heads(&head_names) {
                return Json(CreateCollectionRestResponse {
                    success: false,
                    message: e.message().to_string(),
                });
            }
            head_names
        }
    } else {
        vec!["default".to_string()]
    };
    let head_refs: Vec<&str> = heads.iter().map(|s| s.as_str()).collect();

    let dim = if let Some(d) = payload.dimension {
        if d > 0 {
            if let Err(e) = validate_vector_dimension(d as usize) {
                return Json(CreateCollectionRestResponse {
                    success: false,
                    message: e.message().to_string(),
                });
            }
            d as usize
        } else {
            64
        }
    } else {
        64
    };
    let ef_search = hnsw_settings.ef_search;
    match state.service.engine.create_collection_with_settings(
        &payload.collection,
        dim,
        &head_refs,
        hnsw_settings.clone(),
    ) {
        Ok(_) => {
            observability::record_create_collection(&payload.collection, &head_refs, ef_search);
            Json(CreateCollectionRestResponse {
                success: true,
                message: format!(
                    "Created collection '{}' with {} heads",
                    payload.collection,
                    heads.len()
                ),
            })
        }
        Err(e) => {
            observability::record_error("rest_create_collection", &e.to_string());
            Json(CreateCollectionRestResponse {
                success: false,
                message: e.to_string(),
            })
        }
    }
}

pub async fn alter_collection_handler(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    Json(payload): Json<AlterCollectionRestRequest>,
) -> Json<AlterCollectionRestResponse> {
    if let Err(e) = validate_collection_name(&collection) {
        return Json(AlterCollectionRestResponse {
            success: false,
            message: e.message().to_string(),
        });
    }

    let mut hnsw_settings = attentiondb_hnsw::CollectionSettings::default();
    let s = &payload.settings;
    hnsw_settings.ef_search = s.ef_search.unwrap_or(64) as usize;
    hnsw_settings.ef_construction = s.ef_construction.unwrap_or(400) as usize;
    hnsw_settings.max_nb_connection = s.max_connections.unwrap_or(16) as usize;
    hnsw_settings.similarity_metric = s.similarity.clone().unwrap_or_else(|| "cosine".to_string());
    hnsw_settings.enable_exact_reranking = s.exact_rerank.unwrap_or(true);
    hnsw_settings.enable_gpu_fusion = s.enable_gpu_fusion.unwrap_or(false);
    hnsw_settings.enable_gpu_projections = s.enable_gpu_projections.unwrap_or(false);

    let ef_search = hnsw_settings.ef_search;
    match state
        .service
        .engine
        .alter_collection_settings(&collection, hnsw_settings)
    {
        Ok(_) => {
            observability::record_create_collection(&collection, &[], ef_search);
            Json(AlterCollectionRestResponse {
                success: true,
                message: format!("Altered collection '{}'", collection),
            })
        }
        Err(e) => {
            observability::record_error("rest_alter_collection", &e.to_string());
            Json(AlterCollectionRestResponse {
                success: false,
                message: e.to_string(),
            })
        }
    }
}

fn default_semaphore() -> Arc<tokio::sync::Semaphore> {
    let max_concurrent = std::env::var("ATTENTIONDB_MAX_CONCURRENT_REQUESTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1024);
    Arc::new(tokio::sync::Semaphore::new(max_concurrent))
}

pub fn create_rest_router() -> Router {
    create_rest_router_with_service(
        Arc::new(AttentionDBService::default()),
        Arc::new(ApiKeyStore::disabled()),
        None,
        Arc::new(RateLimiter::disabled()),
    )
}

pub fn create_rest_router_with_service(
    service: Arc<AttentionDBService>,
    api_keys: Arc<ApiKeyStore>,
    metrics: Option<Arc<PrometheusHandle>>,
    rate_limiter: Arc<RateLimiter>,
) -> Router {
    let semaphore = default_semaphore();
    let state = AppState {
        service,
        api_keys: api_keys.clone(),
        metrics,
        rate_limiter: rate_limiter.clone(),
        semaphore,
    };

    Router::new()
        .route("/v1/attend", post(attend_handler))
        .route("/v1/explain", post(explain_handler))
        .route("/v1/insert", post(insert_handler))
        .route("/v1/update", post(update_handler))
        .route("/v1/upsert", post(upsert_handler))
        .route("/v1/delete", post(delete_doc_handler))
        .route(
            "/v1/admin/checkpoint",
            post(crate::admin::checkpoint_admin_handler),
        )
        .route("/v1/admin/check", get(crate::admin::check_admin_handler))
        .route("/v1/collections", post(create_collection_handler))
        .route(
            "/v1/collections/{collection}",
            put(alter_collection_handler),
        )
        .route("/v1/admin/backup", post(crate::admin::backup_handler))
        .route("/v1/admin/backups", get(crate::admin::list_backups_handler))
        .route("/v1/admin/restore", post(crate::admin::restore_handler))
        .route("/health", get(health_handler))
        .route("/health/live", get(liveness_handler))
        .route("/health/ready", get(readiness_handler))
        .route("/health/startup", get(startup_handler))
        .route("/metrics", get(metrics_handler))
        .route("/openapi.json", get(openapi_json_handler))
        .route("/docs", get(swagger_ui_handler))
        .layer(Extension(api_keys))
        .layer(Extension(rate_limiter))
        .layer(from_fn(rate_limit_middleware))
        .layer(from_fn(auth_middleware))
        .with_state(state)
}

pub async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    if let Some(handle) = &state.metrics {
        let metrics = handle.render();
        (
            StatusCode::OK,
            [("Content-Type", "text/plain; version=0.0.4")],
            metrics,
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            [("Content-Type", "text/plain; charset=utf-8")],
            "Metrics not available".to_string(),
        )
    }
}

pub async fn openapi_json_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("Content-Type", "application/json")],
        openapi::OPENAPI_SPEC,
    )
}

pub async fn swagger_ui_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("Content-Type", "text/html; charset=utf-8")],
        include_str!("swagger_ui.html"),
    )
}
