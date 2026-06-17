pub mod server;
pub mod rest;
pub mod client;
pub mod error;
pub mod auth;
pub mod observability;
pub mod validation;
pub mod tls;

pub use server::AttentionDBService;
pub use rest::{create_rest_router, create_rest_router_with_service};
pub use client::AttentionDBClient;
pub use error::ApiError;
pub use auth::ApiKeyStore;
pub use observability::{init_logging, init_metrics};
pub use tls::{TlsMode, resolve_tls};
