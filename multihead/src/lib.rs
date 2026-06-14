pub mod head;
pub mod gating;
pub mod fusion;
pub mod manager;
pub mod error;

pub use head::{HeadType, HeadConfig};
pub use gating::GatingNetwork;
pub use fusion::{fuse_scores, normalize_scores, weighted_fuse};
pub use manager::MultiHeadManager;
pub use error::MultiHeadError;
