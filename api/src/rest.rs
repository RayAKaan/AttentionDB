use axum::{
    extract::{Extension, Path, State},
    http::StatusCode,
    middleware::from_fn,
    response::IntoResponse,
    routing::{get, post, put},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use metrics_exporter_prometheus::PrometheusHandle;
use std::sync::Arc;
use crate::auth::{ApiKeyStore, auth_middleware};
use crate::rate_limiter::{RateLimiter, rate_limit_middleware};
use crate::server::AttentionDBService;
use crate::observability;
use crate::validation::{validate_collection_name, validate_fields, validate_heads, validate_top_k, validate_vector_dimension};
use crate::openapi;

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

pub async fn attend_handler(
    State(state): State<AppState>,
    Json(payload): Json<AttendRequest>,
) -> Result<Json<AttendResponse>, (StatusCode, String)> {
    let _permit = state.semaphore.acquire().await.map_err(|_| {
        (StatusCode::SERVICE_UNAVAILABLE, "Too many concurrent requests".to_string())
    })?;
    let _timer = observability::LatencyTimer::new("rest_attend");

    validate_collection_name(&payload.collection)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;
    let top_k = payload.top_k.unwrap_or(10);
    validate_top_k(top_k)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;
    let offset = payload.offset.unwrap_or(0);

    if let Some(ref heads_list) = payload.heads {
        validate_heads(heads_list)
            .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;
    }

    let query_vec = crate::server::parse_float_vector(&payload.query)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid query vector format".to_string()))?;
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

    let offset_usize = offset as usize;
    let fetch_count = offset_usize + top_k as usize;
    let start = std::time::Instant::now();
    let raw_results = if use_hybrid {
        let query_text = payload.query_text.as_deref().unwrap_or(&payload.query);
        state.service.engine.attend_hybrid(&payload.collection, &heads, &query_vec, query_text, fetch_count)
            .map_err(|e| {
                observability::record_error("rest_attend", &e.to_string());
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            })?
    } else {
        state.service.engine.attend(&payload.collection, &heads, &query_vec, fetch_count)
            .map_err(|e| {
                observability::record_error("rest_attend", &e.to_string());
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            })?
    };
    let latency = start.elapsed().as_secs_f64() * 1000.0;

    let total_count = raw_results.len() as u32;
    let paged: Vec<(u64, f32)> = raw_results.into_iter().skip(offset_usize).take(top_k as usize).collect();
    let has_more = (offset_usize + paged.len()) < total_count as usize;

    let results: Vec<serde_json::Value> = paged.into_iter().map(|(numeric_id, score)| {
        let fields = state.service.engine.get_document_fields(numeric_id);
        serde_json::json!({
            "id": numeric_id.to_string(),
            "score": score,
            "fields": fields,
        })
    }).collect();

    observability::record_attend(&payload.collection, &heads, top_k as usize, results.len(), latency);

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
        (StatusCode::SERVICE_UNAVAILABLE, "Too many concurrent requests".to_string())
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

    if k_vecs.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No vector embeddings found in fields".to_string()));
    }

    let num_vectors = k_vecs.len();
    let mut record = attentiondb_storage::Record::new(json_fields);
    record.k_vecs = k_vecs;

    let start = std::time::Instant::now();
    let id = state.service.engine.insert_document(&payload.collection, record)
        .map_err(|e| {
            observability::record_error("rest_insert", &e.to_string());
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;
    let latency = start.elapsed().as_secs_f64() * 1000.0;

    observability::record_insert(&payload.collection, &id, num_vectors, latency);

    Ok(Json(InsertRestResponse {
        id,
        success: true,
    }))
}

pub async fn liveness_handler() -> (StatusCode, &'static str) {
    (StatusCode::OK, "alive")
}

pub async fn readiness_handler(State(state): State<AppState>) -> (StatusCode, String) {
    let is_healthy = state.service.engine.is_persistent();
    if is_healthy {
        let stats = state.service.engine.stats();
        (StatusCode::OK, format!("ready (collections: {}, heads: {}, vectors: {})", stats.collection_count, stats.total_heads, stats.total_vectors))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not ready — engine not fully initialized".to_string())
    }
}

pub async fn startup_handler() -> (StatusCode, &'static str) {
    (StatusCode::OK, "startup complete")
}

pub async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    let stats = state.service.engine.stats();
    observability::record_engine_stats(stats.collection_count, stats.total_heads, stats.total_vectors);
    Json(HealthResponse {
        status: format!("healthy (collections: {}, heads: {}, vectors: {})", stats.collection_count, stats.total_heads, stats.total_vectors),
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
        hnsw_settings.similarity_metric = s.similarity.clone().unwrap_or_else(|| "cosine".to_string());
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
    match state.service.engine.create_collection_with_settings(&payload.collection, dim, &head_refs, hnsw_settings.clone()) {
        Ok(_) => {
            observability::record_create_collection(&payload.collection, &head_refs, ef_search);
            Json(CreateCollectionRestResponse {
                success: true,
                message: format!("Created collection '{}' with {} heads", payload.collection, heads.len()),
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
    match state.service.engine.alter_collection_settings(&collection, hnsw_settings) {
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
    let state = AppState { service, api_keys: api_keys.clone(), metrics, rate_limiter: rate_limiter.clone(), semaphore };

    Router::new()
        .route("/v1/attend", post(attend_handler))
        .route("/v1/insert", post(insert_handler))
        .route("/v1/collections", post(create_collection_handler))
        .route("/v1/collections/{collection}", put(alter_collection_handler))
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
        (StatusCode::OK, [("Content-Type", "text/plain; version=0.0.4")], metrics)
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, [("Content-Type", "text/plain; charset=utf-8")], "Metrics not available".to_string())
    }
}

pub async fn openapi_json_handler() -> impl IntoResponse {
    (StatusCode::OK, [("Content-Type", "application/json")], openapi::OPENAPI_SPEC)
}

pub async fn swagger_ui_handler() -> impl IntoResponse {
    (StatusCode::OK, [("Content-Type", "text/html; charset=utf-8")], include_str!("swagger_ui.html"))
}
