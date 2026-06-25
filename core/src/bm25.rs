use parking_lot::RwLock;
use std::collections::HashMap;

pub const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "by", "with",
    "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "can", "could", "shall", "should", "may", "might", "must", "i", "you",
    "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "this", "that", "these", "those", "am", "not", "no", "nor", "so", "if", "as",
    "up", "down", "out", "about", "into", "over", "after", "before", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "only", "own", "same", "than",
    "too", "very", "just", "because",
];

fn is_stopword(w: &str) -> bool {
    STOP_WORDS.contains(&w)
}

/// Full Porter stemmer (steps 1a–5b).
fn porter_stem(word: &str) -> String {
    let w = word.to_lowercase();
    if w.len() <= 2 {
        return w;
    }
    let mut s = w;

    // Step 1a
    if s.ends_with("sses") || s.ends_with("ies") {
        s = s[..s.len() - 2].to_string();
    } else if s.ends_with("ss") { /* keep */
    } else if s.ends_with("s") {
        s = s[..s.len() - 1].to_string();
    }

    // Step 1b
    let r1b = if s.ends_with("eed") {
        let stem = &s[..s.len() - 3];
        if measure(stem) > 0 {
            s = stem.to_string() + "ee";
            true
        } else {
            false
        }
    } else if s.ends_with("ed") {
        let stem = s[..s.len() - 2].to_string();
        if stem.contains(|c: char| "aeiou".contains(c)) {
            s = stem;
            true
        } else {
            false
        }
    } else if s.ends_with("ing") {
        let stem = s[..s.len() - 3].to_string();
        if stem.contains(|c: char| "aeiou".contains(c)) {
            s = stem;
            true
        } else {
            false
        }
    } else {
        false
    };
    if r1b {
        if s.ends_with("at") || s.ends_with("bl") || s.ends_with("iz") {
            s += "e";
        } else if let Some(c) = s.chars().last() {
            if let Some(c2) = s.chars().nth(s.len().saturating_sub(2)) {
                if c == c2 && "bcdfghjklmnpqrstvwxz".contains(c) && !"lsz".contains(c) {
                    s.pop();
                }
            }
        }
        if measure(&s) == 1 && ends_with_cvc(&s) {
            s += "e";
        }
    }

    // Step 1c
    if s.ends_with("y") && s.len() > 1 {
        let before = &s[..s.len() - 1];
        if before.contains(|c: char| "aeiou".contains(c)) {
            s = before.to_string() + "i";
        }
    }

    // Step 2
    let step2 = |s: &mut String| {
        let subs = [
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        ];
        for (suff, repl) in &subs {
            if s.ends_with(suff) {
                let stem = &s[..s.len() - suff.len()];
                if measure(stem) > 0 {
                    *s = stem.to_string() + repl;
                    return true;
                }
            }
        }
        false
    };
    step2(&mut s);

    // Step 3
    let step3 = |s: &mut String| {
        let subs = [
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ];
        for (suff, repl) in &subs {
            if s.ends_with(suff) {
                let stem = &s[..s.len() - suff.len()];
                if measure(stem) > 0 {
                    *s = stem.to_string() + repl;
                    return true;
                }
            }
        }
        false
    };
    step3(&mut s);

    // Step 4
    let step4 = |s: &mut String| {
        let suffs = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ism",
            "ate", "iti", "ous", "ive", "ize",
        ];
        for suff in &suffs {
            if s.ends_with(suff) {
                let stem = &s[..s.len() - suff.len()];
                if measure(stem) > 1 {
                    *s = stem.to_string();
                    return true;
                }
            }
        }
        if s.ends_with("ion") && s.len() > 3 {
            let stem = &s[..s.len() - 3];
            if measure(stem) > 1 && (stem.ends_with('s') || stem.ends_with('t')) {
                *s = stem.to_string();
                return true;
            }
        }
        false
    };
    step4(&mut s);

    // Step 5a
    if s.ends_with('e') {
        let stem = &s[..s.len() - 1];
        if measure(stem) > 1 || (measure(stem) == 1 && !ends_with_cvc(stem)) {
            s = stem.to_string();
        }
    }

    // Step 5b
    if s.ends_with('l') && s.len() > 1 {
        let stem = &s[..s.len() - 1];
        if measure(stem) > 1 && stem.ends_with('l') {
            s.pop();
        }
    }

    s
}

fn measure(s: &str) -> usize {
    let vowels = ['a', 'e', 'i', 'o', 'u'];
    let mut count = 0;
    let mut in_vowel = false;
    for c in s.chars() {
        if vowels.contains(&c) {
            if !in_vowel {
                count += 1;
            }
            in_vowel = true;
        } else if c != 'y' || !in_vowel {
            in_vowel = false;
        }
    }
    if count > 0 {
        count - 1
    } else {
        0
    }
}

fn ends_with_cvc(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 3 {
        return false;
    }
    let c1 = chars[chars.len() - 3];
    let c2 = chars[chars.len() - 2];
    let c3 = chars[chars.len() - 1];
    let cons = |c: char| !"aeiou".contains(c) && c != 'y';
    cons(c1) && "aeiou".contains(c2) && cons(c3) && !"wxy".contains(c3)
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|t| !t.is_empty() && !is_stopword(t))
        .map(|t| porter_stem(&t))
        .collect()
}

fn tokenize_with_positions(text: &str) -> Vec<(String, usize)> {
    text.split_whitespace()
        .enumerate()
        .map(|(pos, t)| {
            (
                t.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase(),
                pos,
            )
        })
        .filter(|(t, _)| !t.is_empty() && !is_stopword(t))
        .map(|(t, pos)| (porter_stem(&t), pos))
        .collect()
}

/// Extract query terms, respecting "quoted strings" as phrase queries.
fn extract_query_terms(query: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let mut terms = Vec::new();
    let mut phrases = Vec::new();
    for part in query.split('"').enumerate() {
        let (i, seg) = part;
        if i % 2 == 1 {
            let phrase: Vec<String> = seg
                .split_whitespace()
                .map(|t| {
                    t.trim_matches(|c: char| !c.is_alphanumeric())
                        .to_lowercase()
                })
                .filter(|t| !t.is_empty())
                .map(|t| porter_stem(&t))
                .collect();
            if !phrase.is_empty() {
                phrases.push(phrase);
            }
        } else {
            for t in seg.split_whitespace() {
                let clean = t
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase();
                if !clean.is_empty() && !is_stopword(&clean) {
                    terms.push(porter_stem(&clean));
                }
            }
        }
    }
    (terms, phrases)
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
    pub idf_cache: RwLock<HashMap<String, f32>>,
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
            idf_cache: RwLock::new(HashMap::new()),
        }
    }
}

impl Bm25Index {
    pub fn insert(&self, doc_id: u64, text: &str) {
        let tokens = tokenize_with_positions(text);
        let len = tokens.len();
        let mut freq: HashMap<String, (usize, Vec<usize>)> = HashMap::new();
        for (term, pos) in tokens {
            let e = freq.entry(term).or_default();
            e.0 += 1;
            e.1.push(pos);
        }
        self.document_lengths.write().insert(doc_id, len);
        let mut added = false;
        let mut idx = self.index.write();
        for (term, (tf, positions)) in freq {
            let ps = idx.entry(term.clone()).or_default();
            if ps.iter().any(|p| p.doc_id == doc_id) {
                continue;
            }
            ps.push(Posting {
                doc_id,
                term_frequency: tf,
                positions,
            });
            added = true;
        }
        if added {
            *self.total_documents.write() += 1;
            let avg = self
                .document_lengths
                .read()
                .values()
                .copied()
                .sum::<usize>() as f32
                / *self.total_documents.read() as f32;
            *self.average_doc_length.write() = avg;
            self.idf_cache.write().clear();
        }
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<(u64, f32)> {
        let (terms, phrases) = extract_query_terms(query);
        if terms.is_empty() {
            return Vec::new();
        }
        let td = *self.total_documents.read();
        let avg = *self.average_doc_length.read();
        if td == 0 {
            return Vec::new();
        }
        let idx = self.index.read();
        let lens = self.document_lengths.read();
        let mut cache = self.idf_cache.write();
        let mut scores: HashMap<u64, f32> = HashMap::new();
        for term in &terms {
            let idf = *cache.entry(term.clone()).or_insert_with(|| {
                let df = idx.get(term).map(|p| p.len()).unwrap_or(0);
                if df == 0 {
                    0.0
                } else {
                    ((td as f32 - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln()
                }
            });
            if idf == 0.0 {
                continue;
            }
            if let Some(ps) = idx.get(term) {
                for p in ps {
                    let dl = *lens.get(&p.doc_id).unwrap_or(&0) as f32;
                    let d = p.term_frequency as f32 + self.k1 * (1.0 - self.b + self.b * dl / avg);
                    *scores.entry(p.doc_id).or_default() +=
                        idf * (p.term_frequency as f32 * (self.k1 + 1.0)) / d;
                }
            }
        }
        // Phrase match bonus
        for phrase in &phrases {
            if let Some(first) = phrase.first() {
                if let Some(ps) = idx.get(first) {
                    for posting in ps {
                        for &pos in &posting.positions {
                            let mut match_all = true;
                            for (offset, term) in phrase.iter().enumerate().skip(1) {
                                if let Some(next) = idx.get(term) {
                                    if let Some(np) =
                                        next.iter().find(|p| p.doc_id == posting.doc_id)
                                    {
                                        if !np.positions.contains(&(pos + offset)) {
                                            match_all = false;
                                            break;
                                        }
                                    } else {
                                        match_all = false;
                                        break;
                                    }
                                } else {
                                    match_all = false;
                                    break;
                                }
                            }
                            if match_all {
                                *scores.entry(posting.doc_id).or_default() += 2.0;
                                break;
                            }
                        }
                    }
                }
            }
        }
        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    pub fn search_phrase(&self, phrase: &str, top_k: usize, _window: usize) -> Vec<(u64, f32)> {
        let terms: Vec<String> = tokenize(phrase);
        if terms.is_empty() {
            return Vec::new();
        }
        let idx = self.index.read();
        let Some(first_ps) = idx.get(&terms[0]) else {
            return Vec::new();
        };
        let mut cand = Vec::new();
        for p in first_ps {
            for &pos in &p.positions {
                let mut ok = true;
                for (off, term) in terms.iter().enumerate().skip(1) {
                    if let Some(np) = idx
                        .get(term)
                        .and_then(|ps| ps.iter().find(|x| x.doc_id == p.doc_id))
                    {
                        if !np.positions.contains(&(pos + off)) {
                            ok = false;
                            break;
                        }
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    cand.push(p.doc_id);
                    break;
                }
            }
        }
        let td = *self.total_documents.read();
        let avg = *self.average_doc_length.read();
        let lens = self.document_lengths.read();
        let mut scores: HashMap<u64, f32> = HashMap::new();
        for &doc in &cand {
            let mut s = 0.0;
            for term in &terms {
                if let Some(ps) = idx.get(term) {
                    let df = ps.len();
                    let idf = if df == 0 {
                        0.0
                    } else {
                        ((td as f32 - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln()
                    };
                    if let Some(p) = ps.iter().find(|p| p.doc_id == doc) {
                        let dl = *lens.get(&doc).unwrap_or(&0) as f32;
                        let d =
                            p.term_frequency as f32 + self.k1 * (1.0 - self.b + self.b * dl / avg);
                        s += idf * (p.term_frequency as f32 * (self.k1 + 1.0)) / d;
                    }
                }
            }
            scores.insert(doc, s);
        }
        let mut r: Vec<(u64, f32)> = scores.into_iter().collect();
        r.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        r.truncate(top_k);
        r
    }
}

pub fn reciprocal_rank_fusion(
    dense: &[(u64, f32)],
    sparse: &[(u64, f32)],
    top_k: usize,
) -> Vec<(u64, f32)> {
    let mut f: HashMap<u64, f32> = HashMap::new();
    for (i, (id, _)) in dense.iter().enumerate() {
        *f.entry(*id).or_default() += 1.0 / (i as f32 + 1.0);
    }
    for (i, (id, _)) in sparse.iter().enumerate() {
        *f.entry(*id).or_default() += 1.0 / (i as f32 + 1.0);
    }
    let mut r: Vec<_> = f.into_iter().collect();
    r.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    r.truncate(top_k);
    r
}
