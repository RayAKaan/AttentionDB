use std::cmp::Ordering;
use std::collections::HashMap;
use parking_lot::RwLock;

pub struct Posting {
    pub doc_id: u64,
    pub term_frequency: usize,
}

pub struct Bm25Index {
    pub index: RwLock<HashMap<String, Vec<Posting>>>,
    pub document_lengths: RwLock<HashMap<u64, usize>>,
    pub average_doc_length: RwLock<f32>,
    pub total_documents: RwLock<usize>,
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self {
            index: RwLock::new(HashMap::new()),
            document_lengths: RwLock::new(HashMap::new()),
            average_doc_length: RwLock::new(0.0),
            total_documents: RwLock::new(0),
            k1: 1.5,
            b: 0.75,
        }
    }
}

impl Bm25Index {
    pub fn insert(&self, doc_id: u64, text: &str) {
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|token| token.to_lowercase())
            .collect();

        let mut frequencies: HashMap<String, usize> = HashMap::new();
        for token in tokens.iter() {
            *frequencies.entry(token.clone()).or_default() += 1;
        }

        let length = tokens.len();
        {
            let mut lengths = self.document_lengths.write();
            lengths.insert(doc_id, length);
        }

        let mut added_new = false;
        let mut index = self.index.write();
        for (token, frequency) in frequencies {
            let postings = index.entry(token).or_default();
            if postings.iter().any(|p| p.doc_id == doc_id) {
                continue;
            }
            postings.push(Posting { doc_id, term_frequency: frequency });
            added_new = true;
        }

        if added_new {
            let mut total_docs = self.total_documents.write();
            *total_docs += 1;
            let lengths = self.document_lengths.read();
            let average = lengths.values().copied().sum::<usize>() as f32 / *total_docs as f32;
            *self.average_doc_length.write() = average;
        }
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<(u64, f32)> {
        let query_terms: Vec<String> = query
            .split_whitespace()
            .map(|token| token.to_lowercase())
            .collect();

        let total_documents = *self.total_documents.read();
        let average_doc_length = *self.average_doc_length.read();
        if total_documents == 0 || query_terms.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<u64, f32> = HashMap::new();
        let index = self.index.read();
        let lengths = self.document_lengths.read();

        for term in query_terms.iter() {
            if let Some(postings) = index.get(term) {
                let document_frequency = postings.len();
                let idf = ((total_documents as f32 - document_frequency as f32 + 0.5)
                    / (document_frequency as f32 + 0.5)
                    + 1.0)
                    .ln();

                for posting in postings {
                    let doc_length = *lengths.get(&posting.doc_id).unwrap_or(&0) as f32;
                    let freq = posting.term_frequency as f32;
                    let denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_length / average_doc_length));
                    let score = idf * ((freq * (self.k1 + 1.0)) / denom);
                    *scores.entry(posting.doc_id).or_default() += score;
                }
            }
        }

        let mut results: Vec<(u64, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(top_k);
        results
    }
}

pub fn reciprocal_rank_fusion(
    dense: &[(u64, f32)],
    sparse: &[(u64, f32)],
    top_k: usize,
) -> Vec<(u64, f32)> {
    let mut fused: HashMap<u64, f32> = HashMap::new();

    for (rank, (doc_id, _score)) in dense.iter().enumerate() {
        *fused.entry(*doc_id).or_default() += 1.0 / (rank as f32 + 1.0);
    }

    for (rank, (doc_id, _score)) in sparse.iter().enumerate() {
        *fused.entry(*doc_id).or_default() += 1.0 / (rank as f32 + 1.0);
    }

    let mut fused_results: Vec<(u64, f32)> = fused.into_iter().collect();
    fused_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    fused_results.truncate(top_k);
    fused_results
}
