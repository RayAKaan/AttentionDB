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

pub use error::PersistenceError;
pub use index_persistence::{save_index, load_index, append_vectors, IndexMetadata, LoadProgress};
pub use r#trait::{IndexPersistence, VectorPersistence};
pub use strategy::{PersistenceStrategy, VectorRebuildPersistence};
