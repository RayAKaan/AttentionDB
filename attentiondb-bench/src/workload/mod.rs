pub mod generator;
pub mod difficulty;
pub mod ground_truth;
pub mod dataset_loader;
pub mod failure_modes;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HeadType {
    Semantic,
    Temporal,
    Structural,
}

impl HeadType {
    pub fn all() -> Vec<HeadType> {
        vec![HeadType::Semantic, HeadType::Temporal, HeadType::Structural]
    }

    pub fn from_name(name: &str) -> Option<HeadType> {
        match name.to_lowercase().as_str() {
            "semantic" => Some(HeadType::Semantic),
            "temporal" => Some(HeadType::Temporal),
            "structural" => Some(HeadType::Structural),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            HeadType::Semantic => "semantic",
            HeadType::Temporal => "temporal",
            HeadType::Structural => "structural",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadEmbedding {
    pub head_name: HeadType,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub embeddings: Vec<HeadEmbedding>,
    pub metadata: HashMap<String, String>,
    pub timestamp: Option<i64>,
}

pub type DocumentId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub id: String,
    pub embeddings: Vec<HeadEmbedding>,
    pub enabled_heads: Vec<HeadType>,
    pub ground_truth: Vec<DocumentId>,
    pub difficulty: difficulty::DifficultyLevel,
    pub failure_mode: Option<failure_modes::FailureMode>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HeadCombination {
    SemanticOnly,
    TemporalOnly,
    StructuralOnly,
    AllHeads,
    AllHeadsLearnedGating,
    SemanticTemporal,
    SemanticStructural,
    TemporalStructural,
}

impl HeadCombination {
    pub fn enabled_heads(&self) -> Vec<HeadType> {
        match self {
            HeadCombination::SemanticOnly => vec![HeadType::Semantic],
            HeadCombination::TemporalOnly => vec![HeadType::Temporal],
            HeadCombination::StructuralOnly => vec![HeadType::Structural],
            HeadCombination::SemanticTemporal => vec![HeadType::Semantic, HeadType::Temporal],
            HeadCombination::SemanticStructural => vec![HeadType::Semantic, HeadType::Structural],
            HeadCombination::TemporalStructural => vec![HeadType::Temporal, HeadType::Structural],
            HeadCombination::AllHeads | HeadCombination::AllHeadsLearnedGating => {
                vec![HeadType::Semantic, HeadType::Temporal, HeadType::Structural]
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ExperimentType {
    Standard,
    SemanticOnly,
    TemporalOnly,
    StructuralOnly,
    SemanticTemporal,
    SemanticStructural,
    TemporalStructural,
    AllHeads,
    AllHeadsLearnedGating,
    Corruption10,
    Corruption25,
    Corruption50,
    MissingHead,
    ConflictingHead,
    AdversarialQuery,
    NoisyQuery,
}

impl ExperimentType {
    pub fn from_config(name: &str) -> Option<ExperimentType> {
        match name {
            "Standard" => Some(ExperimentType::Standard),
            "SemanticOnly" => Some(ExperimentType::SemanticOnly),
            "TemporalOnly" => Some(ExperimentType::TemporalOnly),
            "StructuralOnly" => Some(ExperimentType::StructuralOnly),
            "SemanticTemporal" => Some(ExperimentType::SemanticTemporal),
            "SemanticStructural" => Some(ExperimentType::SemanticStructural),
            "TemporalStructural" => Some(ExperimentType::TemporalStructural),
            "AllHeads" => Some(ExperimentType::AllHeads),
            "AllHeadsLearnedGating" => Some(ExperimentType::AllHeadsLearnedGating),
            "Corruption10" => Some(ExperimentType::Corruption10),
            "Corruption25" => Some(ExperimentType::Corruption25),
            "Corruption50" => Some(ExperimentType::Corruption50),
            "MissingHead" => Some(ExperimentType::MissingHead),
            "ConflictingHead" => Some(ExperimentType::ConflictingHead),
            "AdversarialQuery" => Some(ExperimentType::AdversarialQuery),
            "NoisyQuery" => Some(ExperimentType::NoisyQuery),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}
