pub mod contrastive;
pub mod error;
pub mod eval;
pub mod gating_v2;
pub mod projection;
pub mod registry;
pub mod reprojection;
pub mod trainer;

pub use contrastive::ContrastiveLoss;
pub use error::LearnedError;
pub use projection::{ProjectionConfig, ProjectionMatrix};
pub use reprojection::ReprojectionJob;
pub use trainer::ProjectionTrainer;
