//! TLS Configuration for the REST API server.
//!
//! Supports two modes:
//!   1. **File-based:** Provide cert/key paths via env vars
//!      `ATTENTIONDB_TLS_CERT` and `ATTENTIONDB_TLS_KEY`
//!   2. **Self-signed (dev only):** Set `ATTENTIONDB_TLS_SELF_SIGNED=true`
//!      to generate an ephemeral self-signed cert at startup.
//!
//! If neither is set, TLS is **disabled** and the server runs plain HTTP.
//!
//! # Examples
//! ```bash
//! # Production: provide your own cert + key
//! ATTENTIONDB_TLS_CERT=/etc/tls/cert.pem \
//! ATTENTIONDB_TLS_KEY=/etc/tls/key.pem \
//!   cargo run --bin attentiondb-server --release
//!
//! # Development: auto-generated self-signed cert
//! ATTENTIONDB_TLS_SELF_SIGNED=true cargo run --bin attentiondb-server --release
//!
//! # Default: plain HTTP (no env vars = TLS disabled)
//! cargo run --bin attentiondb-server --release
//! ```

use axum_server::tls_rustls::RustlsConfig;
use std::path::PathBuf;
use tracing::{info, warn};

/// Result of TLS configuration attempt.
pub enum TlsMode {
    /// TLS enabled with the given rustls config.
    Enabled(RustlsConfig),
    /// TLS disabled — run plain HTTP.
    Disabled,
}

/// Resolve TLS configuration from environment variables.
///
/// Priority:
/// 1. `ATTENTIONDB_TLS_CERT` + `ATTENTIONDB_TLS_KEY` → file-based
/// 2. `ATTENTIONDB_TLS_SELF_SIGNED=true` → ephemeral self-signed
/// 3. Neither → disabled
pub async fn resolve_tls() -> TlsMode {
    // ── File-based TLS ──────────────────────────────────────────────────
    if let (Ok(cert_path), Ok(key_path)) = (
        std::env::var("ATTENTIONDB_TLS_CERT"),
        std::env::var("ATTENTIONDB_TLS_KEY"),
    ) {
        let cert = PathBuf::from(&cert_path);
        let key = PathBuf::from(&key_path);

        if !cert.exists() {
            warn!(path = %cert_path, "TLS cert file not found — falling back to plain HTTP");
            return TlsMode::Disabled;
        }
        if !key.exists() {
            warn!(path = %key_path, "TLS key file not found — falling back to plain HTTP");
            return TlsMode::Disabled;
        }

        match RustlsConfig::from_pem_file(&cert, &key).await {
            Ok(config) => {
                info!(cert = %cert_path, key = %key_path, "TLS enabled (file-based)");
                return TlsMode::Enabled(config);
            }
            Err(e) => {
                warn!(error = %e, "Failed to load TLS cert/key — falling back to plain HTTP");
                return TlsMode::Disabled;
            }
        }
    }

    // ── Self-signed (dev only) ──────────────────────────────────────────
    if std::env::var("ATTENTIONDB_TLS_SELF_SIGNED")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false)
    {
        match generate_self_signed() {
            Ok(config) => {
                warn!("TLS enabled with SELF-SIGNED certificate (development only — do NOT use in production)");
                return TlsMode::Enabled(config);
            }
            Err(e) => {
                warn!(error = %e, "Failed to generate self-signed cert — falling back to plain HTTP");
                return TlsMode::Disabled;
            }
        }
    }

    // ── Disabled ────────────────────────────────────────────────────────
    info!("TLS disabled (set ATTENTIONDB_TLS_CERT/KEY or ATTENTIONDB_TLS_SELF_SIGNED=true to enable)");
    TlsMode::Disabled
}

/// Generate an ephemeral self-signed certificate using rcgen.
fn generate_self_signed() -> Result<RustlsConfig, Box<dyn std::error::Error>> {
    use rcgen::generate_simple_self_signed;

    let subject_alt_names = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        "0.0.0.0".to_string(),
        "attentiondb".to_string(),
    ];

    let rcgen::CertifiedKey { cert, key_pair } = generate_simple_self_signed(subject_alt_names)?;
    let cert_pem = cert.pem();
    let key_pem = key_pair.serialize_pem();

    let dir = std::env::temp_dir().join("attentiondb_tls");
    std::fs::create_dir_all(&dir)?;
    let cert_path = dir.join("self_signed.crt");
    let key_path = dir.join("self_signed.key");
    std::fs::write(&cert_path, cert_pem)?;
    std::fs::write(&key_path, key_pem)?;

    // Use current tokio runtime to call async builder
    let rt = tokio::runtime::Handle::current();
    let config = rt.block_on(RustlsConfig::from_pem_file(&cert_path, &key_path))?;

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_signed_cert_generation() {
        use rcgen::generate_simple_self_signed;
        let result = generate_simple_self_signed(vec!["localhost".into()]);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_resolve_tls_disabled_by_default() {
        std::env::remove_var("ATTENTIONDB_TLS_CERT");
        std::env::remove_var("ATTENTIONDB_TLS_KEY");
        std::env::remove_var("ATTENTIONDB_TLS_SELF_SIGNED");
        let mode = resolve_tls().await;
        assert!(matches!(mode, TlsMode::Disabled));
    }
}
