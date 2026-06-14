//! AttentionDB Phase 1 — Storage Engine
//!
//! This crate provides the physical storage foundation for AttentionDB:
//! - Record model with projections
//! - Write-ahead log (WAL)
//! - Document store (.adb)
//! - Projection store (.kv)

pub mod record;
pub mod wal;
pub mod projection_store;
pub mod document_store;
pub mod error;

pub use record::Record;
pub use error::StorageError;
pub use document_store::DocumentStore;
pub use wal::{Wal, OpType, WalEntry};
pub use projection_store::ProjectionStore;
