//! Production-Grade HNSW Index Persistence
//!
//! Provides robust save/load for HNSW indexes.
//!
//! ## Strategy
//! Because `hnsw_rs` does not expose its internal graph structure for serialization,
//! we store vectors + configuration + metadata and rebuild the graph on load.
//! This is the same approach used by many production vector databases.

pub mod error;
pub mod index_persistence;
pub mod r#trait;
pub mod strategy;
pub mod graph_persistence;
pub mod async_persistence;
pub mod compaction;
pub mod backup;
pub mod async_compaction;
pub mod remote_backup;

pub use error::PersistenceError;
pub use index_persistence::{save_index, load_index, append_vectors, IndexMetadata, LoadProgress};
pub use r#trait::{IndexPersistence, VectorPersistence};
pub use strategy::{PersistenceStrategy, VectorRebuildPersistence};
pub use graph_persistence::GraphPersistence;
pub use async_persistence::save_index_async;
pub use compaction::compact_index;
pub use backup::{create_backup, list_backups};
pub use async_compaction::compact_index_async;
pub use remote_backup::{upload_backup, download_backup};
