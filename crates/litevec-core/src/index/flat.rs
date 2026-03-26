//! Brute-force flat index.
//!
//! Used for small collections (< 1000 vectors). Computes distance to every vector
//! and returns the top-k results using a binary heap.

use std::collections::{BinaryHeap, HashSet};

use ordered_float::OrderedFloat;

use super::{VectorIndex, VectorStore};
use crate::distance::DistanceFn;

/// Flat (brute-force) index. No additional data structures needed.
pub struct FlatIndex {
    /// Set of indexed internal IDs.
    ids: HashSet<u64>,
}

impl FlatIndex {
    pub fn new() -> Self {
        Self {
            ids: HashSet::new(),
        }
    }
}

impl Default for FlatIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex for FlatIndex {
    fn add(
        &mut self,
        internal_id: u64,
        _vector: &[f32],
        _distance_fn: &dyn DistanceFn,
        _vectors: &VectorStore,
    ) {
        self.ids.insert(internal_id);
    }

    fn remove(&mut self, internal_id: u64) {
        self.ids.remove(&internal_id);
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        _ef_search: usize,
        allowed_ids: Option<&HashSet<u64>>,
        distance_fn: &dyn DistanceFn,
        vectors: &VectorStore,
    ) -> Vec<(u64, f32)> {
        if k == 0 {
            return Vec::new();
        }

        // Max-heap: stores (distance, id), pop removes the largest distance
        let mut heap: BinaryHeap<(OrderedFloat<f32>, u64)> = BinaryHeap::with_capacity(k + 1);

        for &id in &self.ids {
            if let Some(allowed) = allowed_ids
                && !allowed.contains(&id)
            {
                continue;
            }

            if let Some(vector) = vectors.get(id) {
                let dist = distance_fn.compute(query, vector);
                heap.push((OrderedFloat(dist), id));
                if heap.len() > k {
                    heap.pop();
                }
            }
        }

        // Convert heap to sorted results (ascending by distance)
        let mut results: Vec<_> = heap.into_iter().map(|(d, id)| (id, d.0)).collect();
        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance;
    use crate::types::DistanceType;

    #[test]
    fn test_flat_index_basic() {
        let distance_fn = distance::get_distance_fn(DistanceType::Euclidean);
        let mut store = VectorStore::new();
        let mut index = FlatIndex::new();

        store.insert(0, vec![0.0, 0.0]);
        store.insert(1, vec![1.0, 0.0]);
        store.insert(2, vec![0.0, 1.0]);
        store.insert(3, vec![1.0, 1.0]);

        for id in 0..4 {
            index.add(id, store.get(id).unwrap(), distance_fn.as_ref(), &store);
        }

        let query = vec![0.1, 0.1];
        let results = index.search(&query, 2, 0, None, distance_fn.as_ref(), &store);

        assert_eq!(results.len(), 2);
        // Closest should be (0,0)
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_flat_index_with_filter() {
        let distance_fn = distance::get_distance_fn(DistanceType::Euclidean);
        let mut store = VectorStore::new();
        let mut index = FlatIndex::new();

        store.insert(0, vec![0.0, 0.0]);
        store.insert(1, vec![1.0, 0.0]);
        store.insert(2, vec![10.0, 10.0]);

        for id in 0..3 {
            index.add(id, store.get(id).unwrap(), distance_fn.as_ref(), &store);
        }

        let query = vec![0.0, 0.0];
        let allowed: HashSet<u64> = [1, 2].into_iter().collect();
        let results = index.search(&query, 1, 0, Some(&allowed), distance_fn.as_ref(), &store);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1); // (1,0) is closer than (10,10) among allowed
    }

    #[test]
    fn test_flat_index_empty() {
        let distance_fn = distance::get_distance_fn(DistanceType::Cosine);
        let store = VectorStore::new();
        let index = FlatIndex::new();

        let results = index.search(&[1.0, 0.0], 5, 0, None, distance_fn.as_ref(), &store);
        assert!(results.is_empty());
    }

    #[test]
    fn test_flat_index_self_search() {
        let distance_fn = distance::get_distance_fn(DistanceType::Cosine);
        let mut store = VectorStore::new();
        let mut index = FlatIndex::new();

        for i in 0..100 {
            let v: Vec<f32> = (0..32).map(|j| ((i * 32 + j) as f32).sin()).collect();
            store.insert(i, v.clone());
            index.add(i, &v, distance_fn.as_ref(), &store);
        }

        // Search for each vector — it should be the top result
        for i in 0..100 {
            let v = store.get(i).unwrap();
            let results = index.search(v, 1, 0, None, distance_fn.as_ref(), &store);
            assert_eq!(results[0].0, i, "Self-search failed for vector {i}");
            assert!(results[0].1 < 1e-5, "Self-distance should be ~0");
        }
    }

    #[test]
    fn test_flat_index_remove() {
        let distance_fn = distance::get_distance_fn(DistanceType::Euclidean);
        let mut store = VectorStore::new();
        let mut index = FlatIndex::new();

        store.insert(0, vec![0.0, 0.0]);
        store.insert(1, vec![1.0, 1.0]);
        index.add(0, store.get(0).unwrap(), distance_fn.as_ref(), &store);
        index.add(1, store.get(1).unwrap(), distance_fn.as_ref(), &store);

        assert_eq!(index.len(), 2);
        index.remove(0);
        assert_eq!(index.len(), 1);

        let results = index.search(&[0.0, 0.0], 10, 0, None, distance_fn.as_ref(), &store);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }
}
