use crate::workload::{Document, HeadType, Query, DocumentId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TranslationStrategy {
    SemanticOnly,
    Concatenation,
    PerHeadRRF { rrf_k: usize },
}

impl TranslationStrategy {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "SemanticOnly" => Some(TranslationStrategy::SemanticOnly),
            "Concatenation" => Some(TranslationStrategy::Concatenation),
            "PerHeadRRF" => Some(TranslationStrategy::PerHeadRRF { rrf_k: 60 }),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::SemanticOnly => "semantic_only",
            Self::Concatenation => "concatenation",
            Self::PerHeadRRF { .. } => "per_head_rrf",
        }
    }

    pub fn requires_multi_vector_native(&self) -> bool {
        matches!(self, Self::PerHeadRRF { .. })
    }
}

pub fn translate_document_vector(
    doc: &Document,
    strategy: TranslationStrategy,
    primary_head: HeadType,
) -> anyhow::Result<Vec<f32>> {
    match strategy {
        TranslationStrategy::SemanticOnly => {
            doc.embeddings
                .iter()
                .find(|e| e.head_name == primary_head)
                .map(|e| e.vector.clone())
                .ok_or_else(|| anyhow::anyhow!(
                    "Document {} missing primary head {:?}", doc.id, primary_head
                ))
        }
        TranslationStrategy::Concatenation => {
            let mut result = Vec::new();
            for head in [HeadType::Semantic, HeadType::Temporal, HeadType::Structural] {
                if let Some(emb) = doc.embeddings.iter().find(|e| e.head_name == head) {
                    result.extend_from_slice(&emb.vector);
                }
            }
            if result.is_empty() {
                anyhow::bail!("Document {} has no embeddings", doc.id);
            }
            Ok(result)
        }
        TranslationStrategy::PerHeadRRF { .. } => {
            anyhow::bail!(
                "PerHeadRRF requires adapter-level handling, not flat vector translation. \
                 Use translate_query_per_head()."
            )
        }
    }
}

pub fn translate_query_per_head(query: &Query) -> Vec<(HeadType, Vec<f32>)> {
    query.embeddings
        .iter()
        .filter(|e| query.enabled_heads.contains(&e.head_name))
        .map(|e| (e.head_name.clone(), e.vector.clone()))
        .collect()
}

pub fn reciprocal_rank_fusion(
    ranked_lists: &[Vec<DocumentId>],
    top_k: usize,
    rrf_k: usize,
) -> Vec<DocumentId> {
    let mut scores: HashMap<DocumentId, f64> = HashMap::new();

    for ranked in ranked_lists {
        for (rank, doc_id) in ranked.iter().enumerate() {
            *scores.entry(doc_id.clone()).or_insert(0.0)
                += 1.0 / (rrf_k as f64 + rank as f64 + 1.0);
        }
    }

    let mut scored: Vec<(DocumentId, f64)> = scores.into_iter().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(top_k).map(|(id, _)| id).collect()
}

#[derive(Debug, Clone)]
pub struct AdapterCapabilities {
    pub supports_multi_vector: bool,
    pub supports_filtering: bool,
    pub supports_hybrid_search: bool,
    pub max_batch_size: usize,
}

pub fn validate_strategy_for_adapter(
    strategy: TranslationStrategy,
    capabilities: &AdapterCapabilities,
) -> anyhow::Result<()> {
    if strategy == (TranslationStrategy::PerHeadRRF { rrf_k: 60 })
        && !capabilities.supports_multi_vector
    {
        anyhow::bail!(
            "TranslationStrategy::PerHeadRRF requires native multi-vector support \
             but this adapter does not support it."
        );
    }
    Ok(())
}
