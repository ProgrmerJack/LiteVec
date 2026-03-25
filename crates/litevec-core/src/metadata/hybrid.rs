//! Hybrid search combining vector similarity and BM25 keyword search.
//!
//! Uses Reciprocal Rank Fusion (RRF) to merge results from both search types
//! into a single ranked list.

use std::collections::HashMap;

/// A single result from hybrid search.
#[derive(Debug, Clone)]
pub struct HybridResult {
    /// Document ID.
    pub id: u64,
    /// Combined score (higher = more relevant).
    pub score: f32,
    /// Vector search rank (None if not in vector results).
    pub vector_rank: Option<usize>,
    /// Keyword search rank (None if not in keyword results).
    pub keyword_rank: Option<usize>,
}

/// Fusion strategy for combining vector and keyword search results.
#[derive(Debug, Clone, Copy)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion: score = Σ 1/(k + rank).
    /// `k` is a smoothing constant (typically 60).
    Rrf { k: f32 },
    /// Weighted linear combination of normalized scores.
    /// `vector_weight` + `keyword_weight` should sum to 1.0.
    WeightedSum {
        vector_weight: f32,
        keyword_weight: f32,
    },
}

impl Default for FusionStrategy {
    fn default() -> Self {
        FusionStrategy::Rrf { k: 60.0 }
    }
}

/// Merge vector search results and keyword search results using score fusion.
///
/// # Arguments
/// * `vector_results` — (doc_id, distance) pairs from vector search, sorted by distance ascending
/// * `keyword_results` — (doc_id, bm25_score) pairs from BM25, sorted by score descending
/// * `strategy` — the fusion strategy to use
/// * `limit` — max results to return
///
/// # Returns
/// Combined results sorted by fused score descending (most relevant first).
pub fn hybrid_search(
    vector_results: &[(u64, f32)],
    keyword_results: &[(u64, f32)],
    strategy: FusionStrategy,
    limit: usize,
) -> Vec<HybridResult> {
    let mut scores: HashMap<u64, HybridResult> = HashMap::new();

    match strategy {
        FusionStrategy::Rrf { k } => {
            // Vector results: rank 1 = best (lowest distance)
            for (rank, &(id, _distance)) in vector_results.iter().enumerate() {
                let rrf_score = 1.0 / (k + (rank + 1) as f32);
                let entry = scores.entry(id).or_insert(HybridResult {
                    id,
                    score: 0.0,
                    vector_rank: None,
                    keyword_rank: None,
                });
                entry.score += rrf_score;
                entry.vector_rank = Some(rank + 1);
            }

            // Keyword results: rank 1 = best (highest BM25 score)
            for (rank, &(id, _bm25_score)) in keyword_results.iter().enumerate() {
                let rrf_score = 1.0 / (k + (rank + 1) as f32);
                let entry = scores.entry(id).or_insert(HybridResult {
                    id,
                    score: 0.0,
                    vector_rank: None,
                    keyword_rank: None,
                });
                entry.score += rrf_score;
                entry.keyword_rank = Some(rank + 1);
            }
        }
        FusionStrategy::WeightedSum {
            vector_weight,
            keyword_weight,
        } => {
            // Normalize vector distances to [0, 1] scores (invert: lower distance = higher score)
            let max_dist = vector_results
                .iter()
                .map(|(_, d)| *d)
                .fold(f32::NEG_INFINITY, f32::max);
            let min_dist = vector_results
                .iter()
                .map(|(_, d)| *d)
                .fold(f32::INFINITY, f32::min);
            let dist_range = (max_dist - min_dist).max(f32::EPSILON);

            for (rank, &(id, distance)) in vector_results.iter().enumerate() {
                let normalized = 1.0 - (distance - min_dist) / dist_range;
                let entry = scores.entry(id).or_insert(HybridResult {
                    id,
                    score: 0.0,
                    vector_rank: None,
                    keyword_rank: None,
                });
                entry.score += vector_weight * normalized;
                entry.vector_rank = Some(rank + 1);
            }

            // Normalize BM25 scores to [0, 1]
            let max_bm25 = keyword_results
                .iter()
                .map(|(_, s)| *s)
                .fold(f32::NEG_INFINITY, f32::max);
            let min_bm25 = keyword_results
                .iter()
                .map(|(_, s)| *s)
                .fold(f32::INFINITY, f32::min);
            let bm25_range = (max_bm25 - min_bm25).max(f32::EPSILON);

            for (rank, &(id, bm25_score)) in keyword_results.iter().enumerate() {
                let normalized = (bm25_score - min_bm25) / bm25_range;
                let entry = scores.entry(id).or_insert(HybridResult {
                    id,
                    score: 0.0,
                    vector_rank: None,
                    keyword_rank: None,
                });
                entry.score += keyword_weight * normalized;
                entry.keyword_rank = Some(rank + 1);
            }
        }
    }

    let mut results: Vec<HybridResult> = scores.into_values().collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_basic() {
        // Doc 1 is rank 1 in vector, rank 2 in keyword
        // Doc 2 is rank 2 in vector, rank 1 in keyword
        // Doc 3 is only in vector (rank 3)
        let vector = vec![(1, 0.1), (2, 0.3), (3, 0.5)];
        let keyword = vec![(2, 5.0), (1, 3.0)];

        let results = hybrid_search(&vector, &keyword, FusionStrategy::Rrf { k: 60.0 }, 10);

        assert_eq!(results.len(), 3);
        // Doc 1 and Doc 2 should have higher scores than Doc 3 (they appear in both)
        let doc1 = results.iter().find(|r| r.id == 1).unwrap();
        let doc2 = results.iter().find(|r| r.id == 2).unwrap();
        let doc3 = results.iter().find(|r| r.id == 3).unwrap();

        assert!(doc1.score > doc3.score);
        assert!(doc2.score > doc3.score);
        // Doc 1 and Doc 2 have symmetric ranks, so their scores should be equal
        assert!((doc1.score - doc2.score).abs() < 1e-6);
        assert_eq!(doc1.vector_rank, Some(1));
        assert_eq!(doc1.keyword_rank, Some(2));
        assert_eq!(doc3.keyword_rank, None);
    }

    #[test]
    fn test_rrf_limit() {
        let vector = vec![(1, 0.1), (2, 0.2), (3, 0.3)];
        let keyword = vec![(4, 5.0), (5, 3.0)];

        let results = hybrid_search(&vector, &keyword, FusionStrategy::Rrf { k: 60.0 }, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_rrf_empty_inputs() {
        let results = hybrid_search(&[], &[], FusionStrategy::Rrf { k: 60.0 }, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_rrf_vector_only() {
        let vector = vec![(1, 0.1), (2, 0.3)];
        let results = hybrid_search(&vector, &[], FusionStrategy::Rrf { k: 60.0 }, 10);
        assert_eq!(results.len(), 2);
        // Rank 1 should have higher score
        assert!(results[0].id == 1);
    }

    #[test]
    fn test_rrf_keyword_only() {
        let keyword = vec![(10, 5.0), (20, 3.0)];
        let results = hybrid_search(&[], &keyword, FusionStrategy::Rrf { k: 60.0 }, 10);
        assert_eq!(results.len(), 2);
        assert!(results[0].id == 10);
    }

    #[test]
    fn test_weighted_sum() {
        let vector = vec![(1, 0.1), (2, 0.5)];
        let keyword = vec![(2, 5.0), (1, 1.0)];

        let strategy = FusionStrategy::WeightedSum {
            vector_weight: 0.5,
            keyword_weight: 0.5,
        };
        let results = hybrid_search(&vector, &keyword, strategy, 10);

        assert_eq!(results.len(), 2);
        // Both docs appear in both lists
        for r in &results {
            assert!(r.vector_rank.is_some());
            assert!(r.keyword_rank.is_some());
        }
    }

    #[test]
    fn test_weighted_sum_vector_heavy() {
        // With all weight on vector search, rank should follow vector order
        let vector = vec![(1, 0.1), (2, 0.9)];
        let keyword = vec![(2, 10.0), (1, 0.1)];

        let strategy = FusionStrategy::WeightedSum {
            vector_weight: 1.0,
            keyword_weight: 0.0,
        };
        let results = hybrid_search(&vector, &keyword, strategy, 10);
        assert_eq!(results[0].id, 1); // Best vector result wins
    }

    #[test]
    fn test_overlapping_and_unique_docs() {
        let vector = vec![(1, 0.1), (2, 0.2), (3, 0.3)];
        let keyword = vec![(2, 5.0), (4, 3.0), (5, 1.0)];

        let results = hybrid_search(&vector, &keyword, FusionStrategy::Rrf { k: 60.0 }, 10);
        assert_eq!(results.len(), 5); // 3 from vector + 2 unique from keyword

        // Doc 2 appears in both → highest score
        assert_eq!(results[0].id, 2);
        assert!(results[0].vector_rank.is_some());
        assert!(results[0].keyword_rank.is_some());
    }
}
