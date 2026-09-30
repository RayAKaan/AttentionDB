pub mod adaptive;
pub mod backup;
pub mod bm25;
pub mod checker;
pub mod collection;
pub mod constants;
pub mod engine;
pub mod error;
pub mod planner;
pub mod retrieval;
pub mod transaction;

pub use adaptive::{AdaptiveTrace, HeadAllocation, RetrievalBudget};
pub use bm25::{reciprocal_rank_fusion, Bm25Index};
pub use checker::{CheckIssue, Severity};
pub use collection::Collection;
pub use engine::{AttentionEngine, CheckpointInfo, EngineState, EngineStats, IdMapper};
pub use error::CoreError;
pub use retrieval::{
    AdaptivePolicyType, AdaptiveRetrievalConfig, C7CandidateTrace, C7Trace, CandidateSet,
    FusionWeights, RankedCandidate, ScoreNormalization,
};
pub use transaction::{TransactionManager, TxnOp};
