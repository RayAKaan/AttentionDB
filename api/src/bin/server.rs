use attentiondb_api::server::AttentionDBService;
use attentiondb_api::create_rest_router_with_service;
use attentiondb_api::auth::{ApiKeyStore, grpc_auth_interceptor};
use attentiondb_api::observability;
use attentiondb_api::tls;
use attentiondb_api::RateLimiter;
use tonic::transport::Server;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tokio::signal;
use tracing::{info, error, warn};

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
    let shutdown_timeout_secs: u64 = std::env::var("ATTENTIONDB_SHUTDOWN_TIMEOUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);

    // ── Initialize Authentication ────────────────────────────────────────
    let api_keys = Arc::new(ApiKeyStore::from_env());

    // ── Initialize Rate Limiting ─────────────────────────────────────────
    let rate_limiter = Arc::new(RateLimiter::from_env());

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

    // ── Shared Shutdown Notify ───────────────────────────────────────────
    let shutdown = Arc::new(tokio::sync::Notify::new());

    let tls_label = match &tls_mode {
        tls::TlsMode::Enabled(_) => "HTTPS",
        tls::TlsMode::Disabled => "HTTP",
    };
    info!("╔══════════════════════════════════════════════════════════════╗");
    info!("║     AttentionDB — Production API Server                    ║");
    info!("╚══════════════════════════════════════════════════════════════╝");
    info!(grpc = %grpc_addr, rest = %rest_addr, protocol = tls_label, auth = api_keys.enabled, shutdown_timeout = shutdown_timeout_secs, "Server starting");

    let engine_for_wal = engine.clone();
    let shutdown_for_handler = shutdown.clone();

    // ── OS Signal → Notify ────────────────────────────────────────────────
    tokio::spawn(async move {
        shutdown_signal().await;
        info!("Shutdown signal received — stopping new connections, draining in-flight requests");
        shutdown_for_handler.notify_waiters();
    });

    // ── gRPC Server ──────────────────────────────────────────────────────
    let svc = AttentionDBService::new(engine.clone());
    let grpc_service = attentiondb_api::server::attentiondb::attention_db_server::AttentionDbServer::with_interceptor(
        svc,
        grpc_auth_interceptor(api_keys.clone()),
    );
    let mut grpc_builder = Server::builder();
    if let Some(tls_config) = grpc_tls_config {
        grpc_builder = grpc_builder.tls_config(tls_config)?;
    }
    let grpc_shutdown = shutdown.clone();
    let grpc_server = grpc_builder
        .add_service(grpc_service)
        .serve_with_shutdown(grpc_addr, async move {
            grpc_shutdown.notified().await;
            info!("gRPC server stopping (shutdown requested)");
        });

    // ── REST Server (HTTP or HTTPS) ──────────────────────────────────────
    let rest_svc = Arc::new(AttentionDBService::new(engine.clone()));
    let app = create_rest_router_with_service(rest_svc, api_keys.clone(), metrics_handle.clone(), rate_limiter.clone())
        .layer(CorsLayer::permissive())
        .layer(RequestBodyLimitLayer::new(
            attentiondb_api::validation::MAX_REQUEST_BODY_BYTES,
        ));

    info!("Server ready — press Ctrl+C for graceful shutdown");

    let rest_shutdown = shutdown.clone();
    let rest_shutdown_fut = async move {
        rest_shutdown.notified().await;
        info!("REST server stopping (shutdown requested)");
    };

    let rest_server = match tls_mode {
        tls::TlsMode::Enabled(tls_config) => {
            let result = axum_server::bind_rustls(rest_addr, tls_config)
                .serve(app.into_make_service());
            tokio::select! {
                r = result => { r.map_err(|e| Box::new(e) as Box<dyn std::error::Error>) }
                _ = rest_shutdown_fut => { Ok(()) }
            }
        }
        tls::TlsMode::Disabled => {
            let listener = tokio::net::TcpListener::bind(&rest_addr).await?;
            let server = axum::serve(listener, app.into_make_service())
                .with_graceful_shutdown(rest_shutdown_fut);
            server.await.map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        }
    };

    if let Err(e) = rest_server {
        error!(error = %e, "REST server error");
    }

    // ── Wait for gRPC server (with timeout) ──────────────────────────────
    let grpc_result = tokio::time::timeout(
        Duration::from_secs(shutdown_timeout_secs),
        grpc_server,
    ).await;

    match grpc_result {
        Ok(Ok(())) => info!("gRPC server stopped cleanly"),
        Ok(Err(e)) => error!(error = %e, "gRPC server error"),
        Err(_) => warn!("gRPC server shutdown timed out after {}s", shutdown_timeout_secs),
    }

    // ── Flush WAL ──────────────────────────────────────────────────────────
    info!("Flushing WAL before exit");
    if let Err(e) = engine_for_wal.flush_wal() {
        warn!(error = %e, "WAL flush failed");
    }

    info!("Server shutdown complete");
    Ok(())
}

async fn shutdown_signal() {
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
            info!("Received Ctrl+C, initiating graceful shutdown");
        }
        _ = terminate => {
            info!("Received SIGTERM, initiating graceful shutdown");
        }
    }
}
