use crate::workload::{Document, HeadType, Query};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FailureMode {
    Corruption { head: HeadType, noise_std: f32, fraction: f32 },
    MissingHead { head: HeadType },
    ConflictingHead { head: HeadType, shift_vector: f32 },
    AdversarialQuery { head: HeadType, epsilon: f32 },
    NoisyQuery { head: HeadType, noise_std: f32 },
}

pub fn apply_failure_to_document(
    doc: &mut Document,
    mode: &FailureMode,
    rng: &mut impl Rng,
) {
    match mode {
        FailureMode::Corruption { head, noise_std, fraction } => {
            if rng.gen::<f32>() < *fraction {
                if let Some(emb) = doc.embeddings.iter_mut().find(|e| e.head_name == *head) {
                    for v in emb.vector.iter_mut() {
                        *v += rng.gen::<f32>() * noise_std * 2.0 - noise_std;
                    }
                }
            }
        }
        FailureMode::MissingHead { head } => {
            doc.embeddings.retain(|e| e.head_name != *head);
        }
        FailureMode::ConflictingHead { head, shift_vector } => {
            if let Some(emb) = doc.embeddings.iter_mut().find(|e| e.head_name == *head) {
                for v in emb.vector.iter_mut() {
                    *v = -(*v) + shift_vector;
                }
            }
        }
        _ => {}
    }
}

pub fn apply_failure_to_query(
    query: &mut Query,
    mode: &FailureMode,
    rng: &mut impl Rng,
) {
    match mode {
        FailureMode::AdversarialQuery { head, epsilon } => {
            if let Some(emb) = query.embeddings.iter_mut().find(|e| e.head_name == *head) {
                for v in emb.vector.iter_mut() {
                    *v += rng.gen::<f32>() * epsilon * 2.0 - epsilon;
                }
            }
        }
        FailureMode::NoisyQuery { head, noise_std } => {
            if let Some(emb) = query.embeddings.iter_mut().find(|e| e.head_name == *head) {
                for v in emb.vector.iter_mut() {
                    *v += rng.gen::<f32>() * noise_std * 2.0 - noise_std;
                }
            }
        }
        _ => {}
    }
}

pub fn get_failure_mode_for_experiment(
    experiment_type: crate::workload::ExperimentType,
) -> Option<FailureMode> {
    match experiment_type {
        crate::workload::ExperimentType::Corruption10 => {
            Some(FailureMode::Corruption { head: HeadType::Temporal, noise_std: 0.5, fraction: 0.1 })
        }
        crate::workload::ExperimentType::Corruption25 => {
            Some(FailureMode::Corruption { head: HeadType::Temporal, noise_std: 0.5, fraction: 0.25 })
        }
        crate::workload::ExperimentType::Corruption50 => {
            Some(FailureMode::Corruption { head: HeadType::Temporal, noise_std: 0.5, fraction: 0.50 })
        }
        crate::workload::ExperimentType::MissingHead => {
            Some(FailureMode::MissingHead { head: HeadType::Temporal })
        }
        crate::workload::ExperimentType::ConflictingHead => {
            Some(FailureMode::ConflictingHead { head: HeadType::Temporal, shift_vector: 0.0 })
        }
        crate::workload::ExperimentType::AdversarialQuery => {
            Some(FailureMode::AdversarialQuery { head: HeadType::Temporal, epsilon: 0.3 })
        }
        crate::workload::ExperimentType::NoisyQuery => {
            Some(FailureMode::NoisyQuery { head: HeadType::Temporal, noise_std: 0.2 })
        }
        _ => None,
    }
}
