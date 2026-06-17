use attentiondb_api::auth::{grpc_auth_interceptor, ApiKeyStore};
use attentiondb_api::create_rest_router_with_service;
use attentiondb_api::observability::{init_logging, init_metrics, MetricsHandle};
use attentiondb_api::server::AttentionDBService;
use attentiondb_api::validation::MAX_REQUEST_BODY_BYTES;
use axum::extract::Extension;
use axum::middleware::from_fn;
use axum::routing::get;
use axum::Router;
use axum::http::StatusCode;
use axum_server::Server as AxumServer;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::signal;
use tonic::service::interceptor::InterceptedService;
use tonic::transport::{Identity, Server, ServerTlsConfig};
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();
    let metrics_handle = Arc::new(init_metrics());

    let grpc_port = std::env::var("ATTENTIONDB_GRPC_PORT").unwrap_or_else(|_| "7400".into());
    let rest_port = std::env::var("ATTENTIONDB_REST_PORT").unwrap_or_else(|_| "8080".into());
    let metrics_port = std::env::var("ATTENTIONDB_METRICS_PORT").unwrap_or_else(|_| "9090".into());

    let grpc_addr: SocketAddr = format!("0.0.0.0:{}", grpc_port).parse()?;
    let rest_addr: SocketAddr = format!("0.0.0.0:{}", rest_port).parse()?;
    let metrics_addr: SocketAddr = format!("0.0.0.0:{}", metrics_port).parse()?;

    let api_keys = Arc::new(ApiKeyStore::from_env());
    let wal_dir = std::env::var("ATTENTIONDB_DATA_DIR").unwrap_or_else(|_| "/data".into());
    let wal_path = format!("{}/engine.wal", wal_dir);

    let engine = match attentiondb_core::engine::AttentionEngine::open(&wal_path, attentiondb_storage::Durability::GroupCommit) {
        Ok(e) => {
            info!(wal_path = %wal_path, "Engine opened with persistent WAL");
            Arc::new(e)
        }
        Err(e) => {
            info!(error = %e, "WAL open failed, starting with in-memory engine");
            Arc::new(attentiondb_core::engine::AttentionEngine::new())
        }
    };

    info!(grpc = %grpc_addr, rest = %rest_addr, metrics = %metrics_addr, auth = api_keys.enabled, "Server starting");

    let svc = AttentionDBService::new(engine.clone());
    let rest_svc = Arc::new(AttentionDBService::new(engine));

    // Configure optional TLS for gRPC if cert/key env vars are provided
    let mut server_builder = Server::builder();
    if let (Ok(cert_path), Ok(key_path)) = (
        std::env::var("ATTENTIONDB_TLS_CERT"),
        std::env::var("ATTENTIONDB_TLS_KEY"),
    ) {
        match (std::fs::read(&cert_path), std::fs::read(&key_path)) {
            (Ok(cert), Ok(key)) => {
                let identity = Identity::from_pem(cert, key);
                let tls = ServerTlsConfig::new().identity(identity);
                server_builder = server_builder.tls_config(tls)?;
                info!(cert = %cert_path, "gRPC TLS enabled");
            }
            _ => {
                info!(cert = %cert_path, key = %key_path, "Failed to read TLS cert/key, continuing without TLS");
            }
        }
    }

    let grpc_server = server_builder
        .add_service(InterceptedService::new(
            attentiondb_api::server::attentiondb::attention_db_server::AttentionDbServer::new(svc),
            grpc_auth_interceptor(api_keys.clone()),
        ))
        .serve_with_shutdown(grpc_addr, shutdown_signal("gRPC"));

    let rest_app = create_rest_router_with_service(rest_svc)
        .layer(Extension(api_keys.clone()))
        .layer(from_fn(attentiondb_api::auth::auth_middleware))
        .layer(CorsLayer::permissive())
        .layer(RequestBodyLimitLayer::new(MAX_REQUEST_BODY_BYTES));

    let listener = TcpListener::bind(&rest_addr).await?;
    let rest_server = axum::serve(listener, rest_app.into_make_service())
        .with_graceful_shutdown(shutdown_signal("REST"));

    let metrics_app = Router::new()
        .route("/metrics", get(metrics_handler))
        .layer(Extension(metrics_handle.clone()));

    let metrics_server = async move {
        AxumServer::bind(metrics_addr)
            .serve(metrics_app.into_make_service())
            .await
    };

    info!("Server ready — press Ctrl+C for graceful shutdown");

    tokio::select! {
        result = grpc_server => {
            if let Err(e) = result {
                error!(error = %e, "gRPC server error");
            }
        }
        result = rest_server => {
            if let Err(e) = result {
                error!(error = %e, "REST server error");
            }
        }
        result = metrics_server => {
            if let Err(e) = result {
                error!(error = %e, "Metrics server error");
            }
        }
    }

    info!("Server shutdown complete");
    Ok(())
}

async fn metrics_handler(Extension(handle): Extension<Arc<Option<MetricsHandle>>>) -> axum::response::Response {
    match handle.as_ref() {
        Some(handle) => {
            let body = handle.render();
            axum::response::Response::builder()
                .status(StatusCode::OK)
                .header(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")
                .body(body.into())
                .unwrap()
        }
        None => axum::response::Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .body("Metrics unavailable".into())
            .unwrap(),
    }
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
