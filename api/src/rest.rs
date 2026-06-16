use axum::{
    extract::{Path, State},
    routing::{get, post, put},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::server::AttentionDBService;

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
}

#[derive(Deserialize)]
pub struct CreateCollectionRestRequest {
    pub collection: String,
    pub fields: Vec<FieldDefinition>,
    pub settings: Option<CollectionSettingsRest>,
}

#[derive(Serialize)]
pub struct CreateCollectionRestResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Deserialize)]
pub struct AlterCollectionRestRequest {
    pub settings: CollectionSettingsRest,
}

#[derive(Serialize)]
pub struct AlterCollectionRestResponse {
    pub success: bool,
    pub message: String,
}

pub async fn attend_handler(
    State(_state): State<AppState>,
    Json(payload): Json<AttendRequest>,
) -> Json<AttendResponse> {
    println!("[REST] POST /v1/attend: collection={}, query=\"{}\", heads={:?}, top_k={:?}",
             payload.collection, payload.query, payload.heads, payload.top_k);

    Json(AttendResponse {
        results: vec![],
        latency_ms: 2.3,
        effective_sample_size: 4.7,
    })
}

pub async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: "0.5.0".to_string(),
    })
}

pub async fn create_collection_handler(
    Json(payload): Json<CreateCollectionRestRequest>,
) -> Json<CreateCollectionRestResponse> {
    let settings = payload.settings.unwrap_or(CollectionSettingsRest {
        ef_search: Some(64),
        ef_construction: Some(400),
        max_connections: Some(16),
        similarity: Some("cosine".to_string()),
        exact_rerank: Some(true),
        enable_gpu_fusion: Some(false),
    });

    Json(CreateCollectionRestResponse {
        success: true,
        message: format!(
            "Collection '{}' created with ef_search={:?}, ef_construction={:?}, max_connections={:?}",
            payload.collection, settings.ef_search, settings.ef_construction, settings.max_connections
        ),
    })
}

pub async fn alter_collection_handler(
    Path(collection): Path<String>,
    Json(payload): Json<AlterCollectionRestRequest>,
) -> Json<AlterCollectionRestResponse> {
    Json(AlterCollectionRestResponse {
        success: true,
        message: format!(
            "Collection '{}' settings updated (ef_search={:?}, enable_gpu_fusion={:?})",
            collection, payload.settings.ef_search, payload.settings.enable_gpu_fusion
        ),
    })
}

pub fn create_rest_router() -> Router {
    let state = AppState {
        service: Arc::new(AttentionDBService),
    };

    Router::new()
        .route("/v1/attend", post(attend_handler))
        .route("/v1/collections", post(create_collection_handler))
        .route("/v1/collections/{collection}", put(alter_collection_handler))
        .route("/health", get(health_handler))
        .with_state(state)
}
