use attentiondb_api::server::attentiondb::attention_db_server::AttentionDb;
use attentiondb_api::server::attentiondb::{
    AttendRequest, CollectionSettings, CreateCollectionRequest, InsertRequest,
};
use attentiondb_api::{create_rest_router_with_service, ApiKeyStore, AttentionDBService};
use axum::{
    body::Body,
    http::{self, Request as AxumRequest, StatusCode},
};
use std::sync::Arc;
use tonic::Request;
use tower::ServiceExt;

#[tokio::test]
async fn test_grpc_proto_compiles() {
    let _svc = AttentionDBService::default();
}

#[tokio::test]
async fn test_rest_router_created() {
    let router = attentiondb_api::create_rest_router();
    let _cloned = router.clone();
}

#[tokio::test]
async fn test_client_connect_fails_on_bad_addr() {
    let result = attentiondb_api::AttentionDBClient::connect("127.0.0.1:1").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_grpc_end_to_end_insert_and_attend() {
    let svc = AttentionDBService::default();

    let create_req = CreateCollectionRequest {
        collection: "papers".to_string(),
        fields: vec![],
        settings: Some(CollectionSettings {
            ef_search: 64,
            ef_construction: 400,
            max_connections: 16,
            similarity: "cosine".to_string(),
            exact_rerank: true,
            enable_gpu_fusion: false,
            enable_gpu_projections: false,
        }),
        head_settings: std::collections::HashMap::new(),
        dimension: 64,
    };
    svc.create_collection(Request::new(create_req))
        .await
        .unwrap();

    let mut test_vec = vec![0.0f32; 64];
    test_vec[0] = 1.0;
    let vec_str = serde_json::to_string(&test_vec).unwrap();

    let mut fields = std::collections::HashMap::new();
    fields.insert("title".to_string(), "Attention Is All You Need".to_string());
    fields.insert("semantic".to_string(), vec_str.clone());

    let insert_req = InsertRequest {
        collection: "papers".to_string(),
        fields,
    };
    let insert_res = svc
        .insert(Request::new(insert_req))
        .await
        .unwrap()
        .into_inner();
    assert!(insert_res.success);

    let attend_req = AttendRequest {
        collection: "papers".to_string(),
        query: vec_str,
        heads: vec!["semantic".to_string()],
        top_k: 5,
        min_weight: 0.0,
        temporal_decay: None,
        offset: 0,
        hybrid: false,
        bm25_weight: 0.3,
        vector_weight: 0.7,
        query_text: String::new(),
    };
    let attend_res = svc
        .attend(Request::new(attend_req))
        .await
        .unwrap()
        .into_inner();

    assert_eq!(attend_res.results.len(), 1);
    assert_eq!(
        attend_res.results[0]
            .fields
            .get("title")
            .map(|s| s.as_str()),
        Some("Attention Is All You Need")
    );
}

#[tokio::test]
async fn test_rest_end_to_end_insert_and_attend() {
    let svc = Arc::new(AttentionDBService::default());
    let api_keys = Arc::new(ApiKeyStore::disabled());
    let rate_limiter = Arc::new(attentiondb_api::RateLimiter::disabled());
    let app = create_rest_router_with_service(svc.clone(), api_keys, None, rate_limiter);

    let coll_body = serde_json::json!({
        "collection": "rest_papers",
        "fields": []
    });
    let req = AxumRequest::builder()
        .method(http::Method::POST)
        .uri("/v1/collections")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&coll_body).unwrap()))
        .unwrap();
    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let mut test_vec = vec![0.0f32; 64];
    test_vec[1] = 1.0;
    let vec_str = serde_json::to_string(&test_vec).unwrap();

    let insert_body = serde_json::json!({
        "collection": "rest_papers",
        "fields": {
            "title": "RESTful Attention",
            "semantic": vec_str
        }
    });
    let req = AxumRequest::builder()
        .method(http::Method::POST)
        .uri("/v1/insert")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&insert_body).unwrap()))
        .unwrap();
    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let attend_body = serde_json::json!({
        "collection": "rest_papers",
        "query": vec_str,
        "heads": ["semantic"],
        "top_k": 5
    });
    let req = AxumRequest::builder()
        .method(http::Method::POST)
        .uri("/v1/attend")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&attend_body).unwrap()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let attend_res: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

    let results = attend_res["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["fields"]["title"], "RESTful Attention");
}
