use axum::{
    extract::State,
    routing::{get, post},
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

pub fn create_rest_router() -> Router {
    let state = AppState {
        service: Arc::new(AttentionDBService),
    };

    Router::new()
        .route("/v1/attend", post(attend_handler))
        .route("/health", get(health_handler))
        .with_state(state)
}
