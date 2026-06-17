//! API Key Authentication Middleware
//!
//! Supports bearer token authentication via the `Authorization` header.
//! API keys are SHA-256 hashed and compared in constant-time-equivalent fashion.
//!
//! # Configuration
//! Set `ATTENTIONDB_API_KEYS` environment variable with comma-separated keys:
//! ```bash
//! export ATTENTIONDB_API_KEYS="key1,key2,admin-key-3"
//! ```
//! If unset, authentication is **disabled** (open access for development).

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::sync::Arc;
use subtle::ConstantTimeEq;
use tonic::{Request as GrpcRequest, Status};

/// Holds the set of valid API key hashes.
#[derive(Clone)]
pub struct ApiKeyStore {
    /// SHA-256 hashes of valid API keys (hex-encoded).
    key_hashes: HashSet<String>,
    /// Whether authentication is enabled.
    pub enabled: bool,
}

impl ApiKeyStore {
    /// Create from environment variable `ATTENTIONDB_API_KEYS`.
    /// If the variable is unset or empty, auth is disabled.
    pub fn from_env() -> Self {
        match std::env::var("ATTENTIONDB_API_KEYS") {
            Ok(keys_str) if !keys_str.trim().is_empty() => {
                let key_hashes: HashSet<String> = keys_str
                    .split(',')
                    .map(|k| k.trim().to_string())
                    .filter(|k| !k.is_empty())
                    .map(|k| Self::hash_key(&k))
                    .collect();
                let count = key_hashes.len();
                tracing::info!(count, "API key authentication enabled");
                Self { key_hashes, enabled: true }
            }
            _ => {
                tracing::warn!("ATTENTIONDB_API_KEYS not set — authentication DISABLED (open access)");
                Self { key_hashes: HashSet::new(), enabled: false }
            }
        }
    }

    /// Create with explicit keys (for testing).
    pub fn with_keys(keys: &[&str]) -> Self {
        let key_hashes: HashSet<String> = keys.iter()
            .map(|k| Self::hash_key(k))
            .collect();
        Self { key_hashes, enabled: !keys.is_empty() }
    }

    /// Create with auth disabled.
    pub fn disabled() -> Self {
        Self { key_hashes: HashSet::new(), enabled: false }
    }

    fn hash_key(key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Validate a raw API key against stored hashes.
    pub fn validate(&self, key: &str) -> bool {
        if !self.enabled {
            return true;
        }
        let hash = Self::hash_key(key);
        self.key_hashes
            .iter()
            .any(|stored| stored.as_bytes().ct_eq(hash.as_bytes()).unwrap_u8() == 1)
    }
}

/// Extract the bearer token from the Authorization header.
fn extract_bearer_token(req: &Request) -> Option<String> {
    let header = req.headers().get("authorization")?;
    let value = header.to_str().ok()?;
    if let Some(token) = value.strip_prefix("Bearer ") {
        Some(token.to_string())
    } else if let Some(token) = value.strip_prefix("bearer ") {
        Some(token.to_string())
    } else {
        None
    }
}

fn extract_api_key(req: &Request) -> Option<String> {
    if let Some(token) = extract_bearer_token(req) {
        return Some(token);
    }
    if let Some(header) = req.headers().get("x-api-key") {
        if let Ok(key) = header.to_str() {
            return Some(key.to_string());
        }
    }
    None
}

/// Axum middleware layer for API key authentication.
pub async fn auth_middleware(
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let store = req.extensions().get::<Arc<ApiKeyStore>>().cloned();

    let store = match store {
        Some(s) => s,
        None => return Ok(next.run(req).await),
    };

    if !store.enabled {
        return Ok(next.run(req).await);
    }

    if req.uri().path() == "/health" || req.uri().path() == "/metrics" {
        return Ok(next.run(req).await);
    }

    match extract_api_key(&req) {
        Some(key) if store.validate(&key) => {
            tracing::debug!(path = %req.uri().path(), "API key authentication successful");
            Ok(next.run(req).await)
        }
        Some(_) => {
            tracing::warn!(path = %req.uri().path(), "Invalid API key");
            Err(StatusCode::UNAUTHORIZED)
        }
        None => {
            tracing::warn!(path = %req.uri().path(), "Missing API key");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

/// Create a gRPC interceptor for API key authentication.
pub fn grpc_auth_interceptor(
    store: Arc<ApiKeyStore>,
) -> impl Fn(GrpcRequest<()>) -> Result<GrpcRequest<()>, Status> + Clone {
    move |req: GrpcRequest<()>| {
        if !store.enabled {
            return Ok(req);
        }

        let metadata = req.metadata();
        let auth_value = metadata
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .or_else(|| {
                metadata
                    .get("x-api-key")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.to_string())
            });

        if let Some(value) = auth_value {
            let token = if let Some(bearer) = value.strip_prefix("Bearer ") {
                bearer.to_string()
            } else if let Some(bearer) = value.strip_prefix("bearer ") {
                bearer.to_string()
            } else {
                value
            };

            if store.validate(&token) {
                return Ok(req);
            }
        }

        Err(Status::unauthenticated("Invalid or missing API key"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_correct_key() {
        let store = ApiKeyStore::with_keys(&["my-secret-key", "another-key"]);
        assert!(store.validate("my-secret-key"));
        assert!(store.validate("another-key"));
    }

    #[test]
    fn test_validate_wrong_key() {
        let store = ApiKeyStore::with_keys(&["my-secret-key"]);
        assert!(!store.validate("wrong-key"));
        assert!(!store.validate(""));
    }

    #[test]
    fn test_disabled_always_passes() {
        let store = ApiKeyStore::disabled();
        assert!(store.validate("anything"));
        assert!(store.validate(""));
    }

    #[test]
    fn test_hash_is_deterministic() {
        let h1 = ApiKeyStore::hash_key("test-key");
        let h2 = ApiKeyStore::hash_key("test-key");
        assert_eq!(h1, h2);
        assert_ne!(h1, ApiKeyStore::hash_key("other-key"));
    }

    #[test]
    fn test_from_empty_env_disables_auth() {
        std::env::remove_var("ATTENTIONDB_API_KEYS");
        let store = ApiKeyStore::from_env();
        assert!(!store.enabled);
    }
}
