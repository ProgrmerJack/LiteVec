//! BM25 full-text search index on metadata string fields.

use std::collections::{HashMap, HashSet};

/// Stop words filtered during tokenization.
const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// Tokenize text into lowercase alphanumeric tokens, filtering stop words and short tokens.
fn tokenize(text: &str) -> Vec<String> {
    let stop: HashSet<&str> = STOP_WORDS.iter().copied().collect();
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .filter(|t| !stop.contains(t))
        .map(String::from)
        .collect()
}

/// BM25 full-text search index on metadata string fields.
pub struct FullTextIndex {
    /// Term frequency saturation parameter (typically 1.2).
    k1: f32,
    /// Document length normalization parameter (typically 0.75).
    b: f32,
    /// Inverted index: term -> vec of (doc_id, term_frequency).
    inverted_index: HashMap<String, Vec<(u64, u32)>>,
    /// Document lengths (in tokens).
    doc_lengths: HashMap<u64, u32>,
    /// Total number of documents.
    doc_count: u64,
    /// Average document length.
    avg_doc_length: f32,
}

impl FullTextIndex {
    /// Create a new index with default BM25 parameters (k1=1.2, b=0.75).
    pub fn new() -> Self {
        Self::with_params(1.2, 0.75)
    }

    /// Create a new index with custom BM25 parameters.
    pub fn with_params(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            inverted_index: HashMap::new(),
            doc_lengths: HashMap::new(),
            doc_count: 0,
            avg_doc_length: 0.0,
        }
    }

    /// Recalculate average document length from current state.
    fn recalculate_avg_doc_length(&mut self) {
        if self.doc_count == 0 {
            self.avg_doc_length = 0.0;
        } else {
            let total: u64 = self.doc_lengths.values().map(|&l| l as u64).sum();
            self.avg_doc_length = total as f32 / self.doc_count as f32;
        }
    }

    /// Index a document's text content. If the document already exists, it is replaced.
    pub fn add_document(&mut self, doc_id: u64, text: &str) {
        // Remove existing entry first to avoid duplicates.
        if self.doc_lengths.contains_key(&doc_id) {
            self.remove_document(doc_id);
        }

        let tokens = tokenize(text);
        let doc_len = tokens.len() as u32;

        // Count term frequencies.
        let mut tf: HashMap<&str, u32> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.as_str()).or_insert(0) += 1;
        }

        // Insert into inverted index.
        for (term, freq) in tf {
            self.inverted_index
                .entry(term.to_string())
                .or_default()
                .push((doc_id, freq));
        }

        self.doc_lengths.insert(doc_id, doc_len);
        self.doc_count += 1;
        self.recalculate_avg_doc_length();
    }

    /// Remove a document from the index.
    pub fn remove_document(&mut self, doc_id: u64) {
        if self.doc_lengths.remove(&doc_id).is_none() {
            return;
        }

        self.doc_count -= 1;

        // Remove from all posting lists.
        self.inverted_index.retain(|_, postings| {
            postings.retain(|&(id, _)| id != doc_id);
            !postings.is_empty()
        });

        self.recalculate_avg_doc_length();
    }

    /// Search for documents matching a query string.
    /// Returns `(doc_id, bm25_score)` pairs sorted by score descending.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(u64, f32)> {
        let query_terms = tokenize(query);
        if query_terms.is_empty() || self.doc_count == 0 {
            return Vec::new();
        }

        let mut scores: HashMap<u64, f32> = HashMap::new();
        let n = self.doc_count as f32;

        for term in &query_terms {
            let postings = match self.inverted_index.get(term.as_str()) {
                Some(p) => p,
                None => continue,
            };

            let df = postings.len() as f32;
            // IDF = ln((N - df + 0.5) / (df + 0.5) + 1)
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            for &(doc_id, tf) in postings {
                let doc_len = *self.doc_lengths.get(&doc_id).unwrap_or(&0) as f32;
                let tf_f = tf as f32;
                // BM25 term score
                let numerator = tf_f * (self.k1 + 1.0);
                let denominator =
                    tf_f + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_length);
                let term_score = idf * numerator / denominator;
                *scores.entry(doc_id).or_insert(0.0) += term_score;
            }
        }

        let mut results: Vec<(u64, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Get the number of indexed documents.
    pub fn len(&self) -> usize {
        self.doc_count as usize
    }

    /// Returns true if the index contains no documents.
    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }
}

impl Default for FullTextIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, World! This is a test.");
        // "this", "is", "a" are stop words
        assert_eq!(tokens, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_tokenize_short_tokens_filtered() {
        let tokens = tokenize("I am a b c testing");
        // "I" -> "i" (len 1, filtered), "am" (len 2, kept), "a" (stop), "b" (len 1), "c" (len 1)
        assert_eq!(tokens, vec!["am", "testing"]);
    }

    #[test]
    fn test_tokenize_numbers() {
        let tokens = tokenize("version 42 release 2024");
        assert_eq!(tokens, vec!["version", "42", "release", "2024"]);
    }

    #[test]
    fn test_tokenize_empty() {
        assert!(tokenize("").is_empty());
        assert!(tokenize("   ").is_empty());
        assert!(tokenize("a").is_empty()); // single char, filtered
    }

    #[test]
    fn test_basic_index_and_search() {
        let mut idx = FullTextIndex::new();
        idx.add_document(1, "the quick brown fox");
        idx.add_document(2, "the lazy brown dog");
        idx.add_document(3, "quick fox jumps high");

        let results = idx.search("quick fox", 10);
        assert!(!results.is_empty());
        // Doc 1 and 3 both contain "quick" and "fox"; doc 2 has neither
        let ids: Vec<u64> = results.iter().map(|r| r.0).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&2));
    }

    #[test]
    fn test_bm25_ranking_relevance() {
        let mut idx = FullTextIndex::new();
        // Doc 1: "rust" appears twice
        idx.add_document(1, "rust programming in rust");
        // Doc 2: "rust" appears once in longer doc
        idx.add_document(2, "learning python java ruby rust go");
        // Doc 3: no "rust" at all
        idx.add_document(3, "python programming language");

        let results = idx.search("rust", 10);
        assert!(results.len() >= 2);
        // Doc 1 should rank higher than doc 2 (higher tf, shorter doc)
        assert_eq!(results[0].0, 1);
        assert_eq!(results[1].0, 2);
        // Scores should be positive and in descending order
        assert!(results[0].1 > results[1].1);
        assert!(results[1].1 > 0.0);
    }

    #[test]
    fn test_document_removal() {
        let mut idx = FullTextIndex::new();
        idx.add_document(1, "hello world");
        idx.add_document(2, "hello universe");

        assert_eq!(idx.len(), 2);
        let results = idx.search("hello", 10);
        assert_eq!(results.len(), 2);

        idx.remove_document(1);
        assert_eq!(idx.len(), 1);

        let results = idx.search("hello", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);

        // Searching for "world" should return nothing
        let results = idx.search("world", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_nonexistent_document() {
        let mut idx = FullTextIndex::new();
        idx.add_document(1, "test document");
        idx.remove_document(999); // should be a no-op
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_multi_term_query() {
        let mut idx = FullTextIndex::new();
        idx.add_document(1, "machine learning algorithms");
        idx.add_document(2, "deep learning neural networks");
        idx.add_document(3, "machine vision systems");
        idx.add_document(4, "database query optimization");

        let results = idx.search("machine learning", 10);
        // Doc 1 has both terms, should rank highest
        assert_eq!(results[0].0, 1);
        // Docs 2 and 3 each have one term
        let ids: Vec<u64> = results.iter().map(|r| r.0).collect();
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
        // Doc 4 has neither term
        assert!(!ids.contains(&4));
    }

    #[test]
    fn test_empty_query() {
        let mut idx = FullTextIndex::new();
        idx.add_document(1, "some text");
        let results = idx.search("", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_all_stop_words() {
        let mut idx = FullTextIndex::new();
        idx.add_document(1, "some text");
        let results = idx.search("the a an is", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_empty_index() {
        let idx = FullTextIndex::new();
        let results = idx.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_limit_results() {
        let mut idx = FullTextIndex::new();
        for i in 0..20 {
            idx.add_document(i, &format!("common term document {i}"));
        }
        let results = idx.search("common", 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut idx = FullTextIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);

        idx.add_document(1, "test");
        assert!(!idx.is_empty());
        assert_eq!(idx.len(), 1);

        idx.remove_document(1);
        assert!(idx.is_empty());
    }

    #[test]
    fn test_duplicate_document_replaces() {
        let mut idx = FullTextIndex::new();
        idx.add_document(1, "original content");
        idx.add_document(1, "replacement content");

        assert_eq!(idx.len(), 1);

        let results = idx.search("original", 10);
        assert!(results.is_empty());

        let results = idx.search("replacement", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_with_custom_params() {
        let idx = FullTextIndex::with_params(2.0, 0.5);
        assert_eq!(idx.k1, 2.0);
        assert_eq!(idx.b, 0.5);
        assert!(idx.is_empty());
    }

    #[test]
    fn test_default_trait() {
        let idx = FullTextIndex::default();
        assert!(idx.is_empty());
        assert_eq!(idx.k1, 1.2);
        assert_eq!(idx.b, 0.75);
    }

    #[test]
    fn test_scores_descending_order() {
        let mut idx = FullTextIndex::new();
        idx.add_document(1, "rust rust rust");
        idx.add_document(2, "rust programming");
        idx.add_document(
            3,
            "some other thing mentioning rust once in a long sentence with many words",
        );

        let results = idx.search("rust", 10);
        for window in results.windows(2) {
            assert!(window[0].1 >= window[1].1);
        }
    }

    #[test]
    fn test_special_characters_in_text() {
        let mut idx = FullTextIndex::new();
        idx.add_document(1, "user@example.com sent an email!");
        idx.add_document(2, "check out https://example.com/path?q=test");

        let results = idx.search("example", 10);
        assert_eq!(results.len(), 2);

        let results = idx.search("user", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }
}
