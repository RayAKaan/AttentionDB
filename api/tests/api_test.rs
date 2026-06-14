//! Integration tests for Phase 5 API

#[tokio::test]
async fn test_grpc_proto_compiles() {
    let _svc = attentiondb_api::AttentionDBService;
}

#[tokio::test]
async fn test_rest_router_created() {
    let router = attentiondb_api::create_rest_router();
    // Router is clonable and usable — basic sanity check
    let _cloned = router.clone();
}

#[tokio::test]
async fn test_client_connect_fails_on_bad_addr() {
    let result = attentiondb_api::AttentionDBClient::connect("127.0.0.1:1").await;
    // Should fail to connect (nothing listening on that port)
    assert!(result.is_err());
}
