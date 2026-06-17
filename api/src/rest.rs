use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post, put},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::server::AttentionDBService;
use crate::validation::{validate_collection_name, validate_fields, validate_heads, validate_top_k};

#[derive(Clone)]
pub struct AppState {
    pub service: Arc<AttentionDBService>,
}

#[derive(Deserialize)]
pub struct AttendRequest {
    pub collection: String,
    pub query: String,
    pub heads: Option<Vec<String>>,
    pub top_k: Option<u32>,
    pub min_weight: Option<f32>,
    pub temporal_decay: Option<f32>,
}

#[derive(Serialize)]
pub struct AttendResponse {
    pub results: Vec<serde_json::Value>,
    pub latency_ms: f64,
    pub effective_sample_size: f32,
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
    validate_collection_name(&payload.collection)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;
    let top_k = payload.top_k.unwrap_or(10);
    validate_top_k(top_k)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;

    let query_vec = crate::server::parse_float_vector(&payload.query)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid query vector format".to_string()))?;

    if let Some(ref heads_list) = payload.heads {
        validate_heads(heads_list)
            .map_err(|e| (StatusCode::BAD_REQUEST, e.message().to_string()))?;
    }

    let heads = payload.heads.unwrap_or_else(|| {
        if let Ok(coll) = state.service.engine.get_collection(&payload.collection) {
            coll.list_heads()
        } else {
            vec!["default".to_string()]
        }
    });

    let start = std::time::Instant::now();
    let raw_results = state.service.engine.attend(&payload.collection, &heads, &query_vec, payload.top_k.unwrap_or(10) as usize)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let latency = start.elapsed().as_secs_f64() * 1000.0;

    let results = raw_results.into_iter().map(|(numeric_id, score)| {
        let fields = state.service.engine.get_document_fields(numeric_id);
        serde_json::json!({
            "id": numeric_id.to_string(),
            "score": score,
            "fields": fields,
        })
    }).collect();

    Ok(Json(AttendResponse {
        results,
        latency_ms: latency,
        effective_sample_size: 1.0,
    }))
}

pub async fn insert_handler(
    State(state): State<AppState>,
    Json(payload): Json<InsertRestRequest>,
) -> Result<Json<InsertRestResponse>, (StatusCode, String)> {
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

    let mut record = attentiondb_storage::Record::new(json_fields);
    record.k_vecs = k_vecs;

    let id = state.service.engine.insert_document(&payload.collection, record)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(InsertRestResponse {
        id,
        success: true,
    }))
}

pub async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    let stats = state.service.engine.stats();
    Json(HealthResponse {
        status: format!("healthy (collections: {}, heads: {}, vectors: {})", stats.collection_count, stats.total_heads, stats.total_vectors),
        version: "0.5.0".to_string(),
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

    let success = state.service.engine.create_collection_with_settings(&payload.collection, 64, &head_refs, hnsw_settings).is_ok();

    Json(CreateCollectionRestResponse {
        success,
        message: format!("Created collection '{}' with {} heads", payload.collection, heads.len()),
    })
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

    let success = state.service.engine.alter_collection_settings(&collection, hnsw_settings).is_ok();

    Json(AlterCollectionRestResponse {
        success,
        message: format!("Altered collection '{}'", collection),
    })
}

pub fn create_rest_router() -> Router {
    create_rest_router_with_service(Arc::new(AttentionDBService::default()))
}

pub fn create_rest_router_with_service(service: Arc<AttentionDBService>) -> Router {
    let state = AppState { service };

    Router::new()
        .route("/v1/attend", post(attend_handler))
        .route("/v1/insert", post(insert_handler))
        .route("/v1/collections", post(create_collection_handler))
        .route("/v1/collections/{collection}", put(alter_collection_handler))
        .route("/health", get(health_handler))
        .with_state(state)
}
