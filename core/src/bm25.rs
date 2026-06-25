use std::cmp::Ordering;
use std::collections::HashMap;
use parking_lot::RwLock;


/// Built-in Porter stemming (no external dependency).
fn porter_stem(word: &str) -> String {
    let w = word.to_lowercase();
    if w.len() <= 2 {
        return w;
    }
    let _s = w.as_bytes();
    let mut stem = w.clone();

    // Step 1a
    if stem.ends_with("sses") { stem = stem[..stem.len()-2].to_string(); }
    else if stem.ends_with("ies") { stem = stem[..stem.len()-2].to_string(); }
    else if stem.ends_with("ss") { /* keep */ }
    else if stem.ends_with("s") { stem = stem[..stem.len()-1].to_string(); }

    // Step 1b
    if stem.ends_with("eed") {
        let before = &stem[..stem.len()-3];
        if count_consonant_sequence(before) > 0 {
            stem = stem[..stem.len()-1].to_string();
        }
    } else if stem.ends_with("ed") {
        let before = &stem[..stem.len()-2];
        if before.contains(|c: char| "aeiou".contains(c)) {
            stem = before.to_string();
            if stem.ends_with("at") || stem.ends_with("bl") || stem.ends_with("iz") {
                stem.push('e');
            } else if let Some(c) = stem.chars().last() {
                let last_two: String = stem.chars().rev().take(2).collect::<Vec<_>>().into_iter().rev().collect();
                let is_double = last_two.len() == 2 && last_two.chars().nth(0) == last_two.chars().nth(1);
                if is_double && "bcdfghjklmnpqrstvwxz".contains(c) {
                    stem = stem[..stem.len()-1].to_string();
                }
            }
        }
    } else if stem.ends_with("ing") {
        let before = &stem[..stem.len()-3];
        if before.contains(|c: char| "aeiou".contains(c)) {
            stem = before.to_string();
            if stem.ends_with("at") || stem.ends_with("bl") || stem.ends_with("iz") {
                stem.push('e');
            } else if let Some(c) = stem.chars().last() {
                let last_two: String = stem.chars().rev().take(2).collect::<Vec<_>>().into_iter().rev().collect();
                let is_double = last_two.len() == 2 && last_two.chars().nth(0) == last_two.chars().nth(1);
                if is_double && "bcdfghjklmnpqrstvwxz".contains(c) {
                    stem = stem[..stem.len()-1].to_string();
                }
            }
        }
    }

    // Step 1c: replace trailing y with i
    if stem.ends_with("y") && stem.len() > 1 {
        let before = &stem[..stem.len()-1];
        if before.contains(|c: char| "aeiou".contains(c)) {
            if let Some(c) = stem.chars().last() {
                if "bcdfghjklmnpqrstvwxz".contains(c) {
                    let mut chars: Vec<char> = stem.chars().collect();
                    let last_idx = chars.len() - 1;
                    if last_idx > 0 {
                        chars[last_idx] = 'i';
                        stem = chars.into_iter().collect();
                    }
                }
            }
        }
    }

    stem
}

fn count_consonant_sequence(s: &str) -> usize {
    let vowels = ['a', 'e', 'i', 'o', 'u'];
    let vc: Vec<char> = s.chars().collect();
    let mut count = 0;
    let mut i = 0;
    while i < vc.len() {
        if vowels.contains(&vc[i]) {
            count += 1;
            i += 1;
        } else {
            let mut j = i;
            while j < vc.len() && !vowels.contains(&vc[j]) { j += 1; }
            if j > i { count += 1; i = j; } else { i += 1; }
        }
    }
    count
}

/// Stopwords set.
pub struct Stopwords {
    words: Vec<&'static str>,
}

impl Stopwords {
    pub fn new() -> Self {
        Self {
            words: vec![
                "a", "an", "the", "and", "or", "but", "in", "on", "at",
                "to", "for", "of", "by", "with", "from", "is", "are", "was",
                "were", "be", "been", "being", "have", "has", "had", "do", "does",
                "did", "will", "would", "can", "could", "shall", "should", "may",
                "might", "must", "i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "us", "them", "my", "your", "his", "its",
                "our", "their", "this", "that", "these", "those", "am", "not",
                "no", "nor", "so", "if", "as", "up", "down", "out", "about",
                "into", "over", "after", "before", "between", "under", "again",
                "further", "then", "once", "here", "there", "when", "where",
                "why", "how", "all", "each", "every", "both", "few", "more",
                "most", "other", "some", "such", "only", "own", "same", "than",
                "too", "very", "just", "because",
            ],
        }
    }

    pub fn is_stopword(&self, word: &str) -> bool {
        self.words.contains(&word)
    }
}

/// A token with position information for phrase matching.
#[derive(Debug, Clone)]
pub struct Token {
    pub term: String,
    pub position: usize,
}

pub struct Posting {
    pub doc_id: u64,
    pub term_frequency: usize,
    pub positions: Vec<usize>,
}

pub struct Bm25Index {
    pub index: RwLock<HashMap<String, Vec<Posting>>>,
    pub document_lengths: RwLock<HashMap<u64, usize>>,
    pub average_doc_length: RwLock<f32>,
    pub total_documents: RwLock<usize>,
    pub k1: f32,
    pub b: f32,
    #[allow(dead_code)]
    stopwords: Stopwords,
    idf_cache: RwLock<HashMap<String, f32>>,
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
            stopwords: Stopwords::new(),
            idf_cache: RwLock::new(HashMap::new()),
        }
    }
}

fn tokenize_full(text: &str) -> Vec<Token> {
    text.split_whitespace()
        .enumerate()
        .map(|(pos, t)| Token {
            term: t.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase(),
            position: pos,
        })
        .filter(|t| !t.term.is_empty())
        .filter(|t| !Stopwords::new().is_stopword(&t.term))
        .map(|t| Token { term: porter_stem(&t.term), position: t.position })
        .collect()
}

impl Bm25Index {
    pub fn insert(&self, doc_id: u64, text: &str) {
        let tokens = tokenize_full(text);
        let mut frequencies: HashMap<String, (usize, Vec<usize>)> = HashMap::new();
        for token in tokens.iter() {
            let entry = frequencies.entry(token.term.clone()).or_default();
            entry.0 += 1;
            entry.1.push(token.position);
        }
        let length = tokens.len();
        {
            let mut lengths = self.document_lengths.write();
            lengths.insert(doc_id, length);
        }
        let mut added_new = false;
        let mut index = self.index.write();
        for (term, (frequency, positions)) in frequencies {
            let postings = index.entry(term.clone()).or_default();
            if postings.iter().any(|p| p.doc_id == doc_id) {
                continue;
            }
            postings.push(Posting { doc_id, term_frequency: frequency, positions });
            added_new = true;
        }
        if added_new {
            let mut total_docs = self.total_documents.write();
            *total_docs += 1;
            let lengths = self.document_lengths.read();
            let average = lengths.values().copied().sum::<usize>() as f32 / *total_docs as f32;
            *self.average_doc_length.write() = average;
            // Invalidate IDF cache
            self.idf_cache.write().clear();
        }
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<(u64, f32)> {
        let query_tokens = tokenize_full(query);
        let query_terms: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            query_tokens.iter().filter(|t| seen.insert(t.term.clone())).map(|t| t.term.clone()).collect()
        };
        let total_documents = *self.total_documents.read();
        let average_doc_length = *self.average_doc_length.read();
        if total_documents == 0 || query_terms.is_empty() {
            return Vec::new();
        }
        let mut scores: HashMap<u64, f32> = HashMap::new();
        let index = self.index.read();
        let lengths = self.document_lengths.read();
        let mut idf_cache = self.idf_cache.write();
        for term in query_terms.iter() {
            let idf = if let Some(&cached) = idf_cache.get(term) {
                cached
            } else {
                let df = index.get(term).map(|p| p.len()).unwrap_or(0);
                let computed = if df == 0 {
                    0.0
                } else {
                    ((total_documents as f32 - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln()
                };
                idf_cache.insert(term.clone(), computed);
                computed
            };
            if idf == 0.0 {
                continue;
            }
            if let Some(postings) = index.get(term) {
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

    /// Phrase query: match documents containing tokens in order within window.
    pub fn search_phrase(&self, phrase: &str, top_k: usize, _window: usize) -> Vec<(u64, f32)> {
        let tokens = tokenize_full(phrase);
        if tokens.is_empty() {
            return Vec::new();
        }
        let terms: Vec<String> = tokens.iter().map(|t| t.term.clone()).collect();
        let first_term = &terms[0];
        let index = self.index.read();
        let some_postings = match index.get(first_term) {
            Some(p) => p,
            None => return Vec::new(),
        };
        let mut candidate_docs: Vec<u64> = Vec::new();
        for posting in some_postings {
            if terms.len() == 1 {
                candidate_docs.push(posting.doc_id);
                continue;
            }
            let mut found = false;
            for &pos in &posting.positions {
                let mut all_match = true;
                for (offset, term) in terms.iter().enumerate().skip(1) {
                    let target_pos = pos + offset;
                    if let Some(next_posting) = index.get(term) {
                        if let Some(np) = next_posting.iter().find(|p| p.doc_id == posting.doc_id) {
                            if !np.positions.contains(&target_pos) {
                                all_match = false;
                                break;
                            }
                        } else {
                            all_match = false;
                            break;
                        }
                    } else {
                        all_match = false;
                        break;
                    }
                }
                if all_match {
                    found = true;
                    break;
                }
            }
            if found {
                candidate_docs.push(posting.doc_id);
            }
        }
        let total_documents = *self.total_documents.read();
        let average_doc_length = *self.average_doc_length.read();
        let lengths = self.document_lengths.read();
        let mut scores: HashMap<u64, f32> = HashMap::new();
        for doc_id in &candidate_docs {
            let mut doc_score = 0.0f32;
            for term in &terms {
                if let Some(postings) = index.get(term) {
                    let df = postings.len();
                    if df == 0 { continue; }
                    let idf = ((total_documents as f32 - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
                    if let Some(posting) = postings.iter().find(|p| p.doc_id == *doc_id) {
                        let doc_length = *lengths.get(doc_id).unwrap_or(&0) as f32;
                        let freq = posting.term_frequency as f32;
                        let denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_length / average_doc_length));
                        doc_score += idf * ((freq * (self.k1 + 1.0)) / denom);
                    }
                }
            }
            scores.insert(*doc_id, doc_score);
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