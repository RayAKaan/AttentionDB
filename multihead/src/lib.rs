pub mod error;
pub mod fusion;
pub mod gating;
pub mod head;
pub mod manager;

pub use error::MultiHeadError;
pub use fusion::{fuse_scores, normalize_scores, weighted_fuse};
pub use gating::GatingNetwork;
pub use head::{HeadConfig, HeadType};
pub use manager::MultiHeadManager;
