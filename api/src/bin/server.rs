use attentiondb_api::server::AttentionDBService;
use attentiondb_api::create_rest_router_with_service;
use attentiondb_api::auth::{ApiKeyStore, grpc_auth_interceptor};
use attentiondb_api::observability;
use attentiondb_api::tls;
use tonic::transport::Server;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tokio::signal;
use tracing::{info, error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Initialize Observability ─────────────────────────────────────────
    observability::init_logging();
    let metrics_handle = observability::init_metrics().map(Arc::new);

    // ── Configuration ────────────────────────────────────────────────────
    let grpc_port = std::env::var("ATTENTIONDB_GRPC_PORT").unwrap_or_else(|_| "7400".into());
    let rest_port = std::env::var("ATTENTIONDB_REST_PORT").unwrap_or_else(|_| "8080".into());
    let grpc_addr: SocketAddr = format!("0.0.0.0:{}", grpc_port).parse()?;
    let rest_addr: SocketAddr = format!("0.0.0.0:{}", rest_port).parse()?;

    // ── Initialize Authentication ────────────────────────────────────────
    let api_keys = Arc::new(ApiKeyStore::from_env());

    // ── Initialize TLS ───────────────────────────────────────────────────
    let tls_mode = tls::resolve_tls().await;
    let grpc_tls_config = tls::resolve_grpc_tls().await;

    // ── Initialize Engine ────────────────────────────────────────────────
    let wal_dir = std::env::var("ATTENTIONDB_DATA_DIR").unwrap_or_else(|_| "/data".into());
    let wal_path = format!("{}/engine.wal", wal_dir);

    let engine = match attentiondb_core::AttentionEngine::open(&wal_path, attentiondb_storage::Durability::GroupCommit) {
        Ok(e) => {
            info!(wal_path = %wal_path, "Engine opened with persistent WAL");
            Arc::new(e)
        }
        Err(e) => {
            info!(error = %e, "WAL open failed, starting with in-memory engine");
            Arc::new(attentiondb_core::AttentionEngine::new())
        }
    };

    let tls_label = match &tls_mode {
        tls::TlsMode::Enabled(_) => "HTTPS",
        tls::TlsMode::Disabled => "HTTP",
    };
    info!("╔══════════════════════════════════════════════════════════════╗");
    info!("║     AttentionDB — Production API Server                    ║");
    info!("╚══════════════════════════════════════════════════════════════╝");
    info!(grpc = %grpc_addr, rest = %rest_addr, protocol = tls_label, auth = api_keys.enabled, "Server starting");

    let svc = AttentionDBService::new(engine.clone());
    let rest_svc = Arc::new(AttentionDBService::new(engine));

    // ── gRPC Server ──────────────────────────────────────────────────────
    let grpc_service = attentiondb_api::server::attentiondb::attention_db_server::AttentionDbServer::with_interceptor(
        svc,
        grpc_auth_interceptor(api_keys.clone()),
    );
    let mut grpc_builder = Server::builder();
    if let Some(tls_config) = grpc_tls_config {
        grpc_builder = grpc_builder.tls_config(tls_config)?;
    }
    let grpc_server = grpc_builder
        .add_service(grpc_service)
        .serve_with_shutdown(grpc_addr, shutdown_signal("gRPC"));

    // ── REST Server (HTTP or HTTPS) ──────────────────────────────────────
    let app = create_rest_router_with_service(rest_svc, api_keys.clone(), metrics_handle.clone())
        .layer(CorsLayer::permissive())
        .layer(RequestBodyLimitLayer::new(
            attentiondb_api::validation::MAX_REQUEST_BODY_BYTES,
        ));

    info!("Server ready — press Ctrl+C for graceful shutdown");

    match tls_mode {
        tls::TlsMode::Enabled(tls_config) => {
            // HTTPS mode
            let rest_server = axum_server::bind_rustls(rest_addr, tls_config)
                .serve(app.into_make_service());

            tokio::select! {
                result = grpc_server => {
                    if let Err(e) = result { error!(error = %e, "gRPC server error"); }
                }
                result = rest_server => {
                    if let Err(e) = result { error!(error = %e, "HTTPS REST server error"); }
                }
                _ = shutdown_signal("main") => {
                    info!("Shutdown signal received");
                }
            }
        }
        tls::TlsMode::Disabled => {
            // HTTP mode
            let listener = tokio::net::TcpListener::bind(&rest_addr).await?;
            let rest_server = axum::serve(listener, app.into_make_service())
                .with_graceful_shutdown(shutdown_signal("REST"));

            tokio::select! {
                result = grpc_server => {
                    if let Err(e) = result { error!(error = %e, "gRPC server error"); }
                }
                result = rest_server => {
                    if let Err(e) = result { error!(error = %e, "HTTP REST server error"); }
                }
            }
        }
    }

    info!("Server shutdown complete");
    Ok(())
}

async fn shutdown_signal(name: &str) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!(server = name, "Received Ctrl+C, initiating graceful shutdown");
        }
        _ = terminate => {
            info!(server = name, "Received SIGTERM, initiating graceful shutdown");
        }
    }
}
