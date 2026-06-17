use attentiondb_api::server::AttentionDBService;
use attentiondb_api::{create_rest_router, create_rest_router_with_service};
use tonic::transport::Server;
use std::net::SocketAddr;
use tower_http::cors::CorsLayer;
use tokio::net::TcpListener;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grpc_addr: SocketAddr = "0.0.0.0:7400".parse()?;
    let rest_addr: SocketAddr = "0.0.0.0:8080".parse()?;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     AttentionDB Phase 5 — API Server                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  gRPC  → {}:{}", grpc_addr.ip(), grpc_addr.port());
    println!("  REST  → {}:{}", rest_addr.ip(), rest_addr.port());
    println!("  Press Ctrl+C to stop.\n");

    let engine = Arc::new(attentiondb_core::engine::AttentionEngine::new());
    let svc = AttentionDBService::new(engine.clone());
    let rest_svc = Arc::new(AttentionDBService::new(engine));

    let grpc_server = Server::builder()
        .add_service(attentiondb_api::server::attentiondb::attention_db_server::AttentionDbServer::new(svc))
        .serve(grpc_addr);

    let listener = TcpListener::bind(&rest_addr).await?;
    let app = create_rest_router_with_service(rest_svc).layer(CorsLayer::permissive());
    let rest_server = axum::serve(listener, app.into_make_service());

    tokio::select! {
        _ = grpc_server => {},
        _ = rest_server => {},
    }

    Ok(())
}
