//! Production-Grade HNSW Index Persistence
//!
//! Provides robust save/load for HNSW indexes.
//!
//! ## Strategy
//! Because `hnsw_rs` does not expose its internal graph structure for serialization,
//! we store vectors + configuration + metadata and rebuild the graph on load.
//! This is the same approach used by many production vector databases.

pub mod async_compaction;
pub mod async_persistence;
pub mod backup;
pub mod compaction;
pub mod error;
pub mod graph_persistence;
pub mod index_persistence;
pub mod remote_backup;
pub mod strategy;
pub mod r#trait;

pub use async_compaction::compact_index_async;
pub use async_persistence::save_index_async;
pub use backup::{create_backup, list_backups};
pub use compaction::compact_index;
pub use error::PersistenceError;
pub use graph_persistence::GraphPersistence;
pub use index_persistence::{append_vectors, load_index, save_index, IndexMetadata, LoadProgress};
pub use r#trait::{IndexPersistence, VectorPersistence};
pub use remote_backup::{download_backup, upload_backup};
pub use strategy::{PersistenceStrategy, VectorRebuildPersistence};
