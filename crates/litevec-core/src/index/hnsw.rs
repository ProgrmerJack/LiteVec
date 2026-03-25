//! HNSW (Hierarchical Navigable Small World) index.
//!
//! O(log N) approximate nearest neighbor search with high recall.
//! Based on the original paper by Malkov & Yashunin (2016).

use std::collections::{BinaryHeap, HashMap, HashSet};

use ordered_float::OrderedFloat;
use rand::Rng;

use super::{VectorIndex, VectorStore};
use crate::distance::DistanceFn;

/// HNSW index.
pub struct HnswIndex {
    /// Maximum connections per node per layer.
    m: usize,
    /// Maximum connections at layer 0 (typically 2 * M).
    m0: usize,
    /// Dynamic candidate list size for construction.
    ef_construction: usize,
    /// Current maximum layer.
    max_layer: usize,
    /// Entry point node ID.
    entry_point: Option<u64>,
    /// Layers: layer → { node → neighbors }.
    layers: Vec<HashMap<u64, Vec<u64>>>,
    /// Highest layer each node appears in.
    node_layer: HashMap<u64, usize>,
    /// Level generation multiplier: 1 / ln(M).
    level_mult: f64,
    /// Total number of vectors.
    count: usize,
}

impl HnswIndex {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            m,
            m0: m * 2,
            ef_construction,
            max_layer: 0,
            entry_point: None,
            layers: vec![HashMap::new()],
            node_layer: HashMap::new(),
            level_mult: 1.0 / (m.max(2) as f64).ln(),
            count: 0,
        }
    }

    /// Generate a random layer for a new node.
    fn random_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        let f: f64 = rng.r#gen::<f64>();
        let level = (-f.ln() * self.level_mult).floor() as usize;
        level.min(self.max_layer + 1) // Don't jump too many levels
    }

    /// Greedy search from a single entry point — returns the closest node.
    fn search_layer_greedy(
        &self,
        query: &[f32],
        entry: u64,
        layer: usize,
        distance_fn: &dyn DistanceFn,
        vectors: &VectorStore,
    ) -> u64 {
        let mut current = entry;
        let mut current_dist = vectors
            .get(current)
            .map(|v| distance_fn.compute(query, v))
            .unwrap_or(f32::MAX);

        loop {
            let mut changed = false;
            if let Some(neighbors) = self.layers[layer].get(&current) {
                for &neighbor in neighbors {
                    if let Some(v) = vectors.get(neighbor) {
                        let d = distance_fn.compute(query, v);
                        if d < current_dist {
                            current_dist = d;
                            current = neighbor;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    #[allow(clippy::too_many_arguments)]
    /// Beam search at a single layer — returns ef nearest neighbors.
    fn search_layer_ef(
        &self,
        query: &[f32],
        entry_points: &[u64],
        layer: usize,
        ef: usize,
        allowed_ids: Option<&HashSet<u64>>,
        distance_fn: &dyn DistanceFn,
        vectors: &VectorStore,
    ) -> Vec<(u64, f32)> {
        let mut visited: HashSet<u64> = HashSet::new();

        // Min-heap for candidates (closest first)
        let mut candidates: BinaryHeap<std::cmp::Reverse<(OrderedFloat<f32>, u64)>> =
            BinaryHeap::new();
        // Max-heap for results (farthest first for easy pruning)
        let mut results: BinaryHeap<(OrderedFloat<f32>, u64)> = BinaryHeap::new();

        for &ep in entry_points {
            if visited.insert(ep)
                && let Some(v) = vectors.get(ep)
            {
                let d = distance_fn.compute(query, v);
                candidates.push(std::cmp::Reverse((OrderedFloat(d), ep)));
                results.push((OrderedFloat(d), ep));
            }
        }

        while let Some(std::cmp::Reverse((cand_dist, cand_id))) = candidates.pop() {
            // If this candidate is farther than the worst in our result set and we have enough, stop
            if results.len() >= ef
                && let Some(&(worst_dist, _)) = results.peek()
                && cand_dist > worst_dist
            {
                break;
            }

            if let Some(neighbors) = self.layers[layer].get(&cand_id) {
                for &neighbor in neighbors {
                    if !visited.insert(neighbor) {
                        continue;
                    }

                    if let Some(v) = vectors.get(neighbor) {
                        let d = distance_fn.compute(query, v);
                        let should_add = results.len() < ef
                            || d < results.peek().map(|(od, _)| od.0).unwrap_or(f32::MAX);

                        if should_add {
                            candidates.push(std::cmp::Reverse((OrderedFloat(d), neighbor)));
                            results.push((OrderedFloat(d), neighbor));
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Filter by allowed_ids and collect
        let mut out: Vec<(u64, f32)> = results
            .into_iter()
            .filter(|(_, id)| allowed_ids.is_none_or(|allowed| allowed.contains(id)))
            .map(|(d, id)| (id, d.0))
            .collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        out
    }

    /// Select the M closest neighbors (simple heuristic).
    fn select_neighbors(candidates: &[(u64, f32)], m: usize) -> Vec<u64> {
        candidates.iter().take(m).map(|&(id, _)| id).collect()
    }

    /// Ensure a layer exists.
    fn ensure_layer(&mut self, layer: usize) {
        while self.layers.len() <= layer {
            self.layers.push(HashMap::new());
        }
    }

    /// Add bidirectional edge, pruning if necessary.
    fn connect(
        &mut self,
        node: u64,
        neighbor: u64,
        layer: usize,
        max_connections: usize,
        distance_fn: &dyn DistanceFn,
        vectors: &VectorStore,
    ) {
        // node → neighbor
        self.layers[layer].entry(node).or_default().push(neighbor);

        // neighbor → node
        let neighbors = self.layers[layer].entry(neighbor).or_default();
        neighbors.push(node);

        // Prune neighbor's connections if too many
        if neighbors.len() > max_connections
            && let Some(nv) = vectors.get(neighbor)
        {
            let mut scored: Vec<(u64, f32)> = neighbors
                .iter()
                .filter_map(|&n| vectors.get(n).map(|v| (n, distance_fn.compute(nv, v))))
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            *neighbors = scored
                .into_iter()
                .take(max_connections)
                .map(|(id, _)| id)
                .collect();
        }
    }
}

impl VectorIndex for HnswIndex {
    fn add(
        &mut self,
        internal_id: u64,
        _vector: &[f32],
        distance_fn: &dyn DistanceFn,
        vectors: &VectorStore,
    ) {
        let new_layer = self.random_layer();
        self.ensure_layer(new_layer);
        self.node_layer.insert(internal_id, new_layer);
        self.count += 1;

        // Ensure the node exists in all layers from 0 to new_layer
        for l in 0..=new_layer {
            self.layers[l].entry(internal_id).or_default();
        }

        // First node
        if self.entry_point.is_none() {
            self.entry_point = Some(internal_id);
            self.max_layer = new_layer;
            return;
        }

        let entry_point = self.entry_point.unwrap();

        // Phase 1: Greedily descend from top layer to new_layer + 1
        let mut current_entry = entry_point;
        let top = self.max_layer;
        if top > new_layer {
            for layer in (new_layer + 1..=top).rev() {
                if self.layers.len() > layer {
                    current_entry = self.search_layer_greedy(
                        _vector,
                        current_entry,
                        layer,
                        distance_fn,
                        vectors,
                    );
                }
            }
        }

        // Phase 2: Insert at layers new_layer down to 0
        for layer in (0..=new_layer.min(top)).rev() {
            let ef = self.ef_construction;
            let neighbors_found = self.search_layer_ef(
                _vector,
                &[current_entry],
                layer,
                ef,
                None,
                distance_fn,
                vectors,
            );

            let max_conn = if layer == 0 { self.m0 } else { self.m };
            let selected = Self::select_neighbors(&neighbors_found, max_conn);

            for &neighbor in &selected {
                self.connect(internal_id, neighbor, layer, max_conn, distance_fn, vectors);
            }

            // Update entry for next lower layer
            if let Some(&(closest, _)) = neighbors_found.first() {
                current_entry = closest;
            }
        }

        // Update entry point if new node has higher layer
        if new_layer > self.max_layer {
            self.entry_point = Some(internal_id);
            self.max_layer = new_layer;
        }
    }

    fn remove(&mut self, internal_id: u64) {
        if let Some(max_layer) = self.node_layer.remove(&internal_id) {
            for layer in 0..=max_layer {
                // Remove this node from all neighbors' adjacency lists
                if let Some(neighbors) = self.layers[layer].remove(&internal_id) {
                    for neighbor in &neighbors {
                        if let Some(adj) = self.layers[layer].get_mut(neighbor) {
                            adj.retain(|&id| id != internal_id);
                        }
                    }
                }
            }
            self.count -= 1;

            // Update entry point if needed
            if self.entry_point == Some(internal_id) {
                self.entry_point = self.layers[0].keys().next().copied();
                if self.entry_point.is_none() {
                    self.max_layer = 0;
                }
            }
        }
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        allowed_ids: Option<&HashSet<u64>>,
        distance_fn: &dyn DistanceFn,
        vectors: &VectorStore,
    ) -> Vec<(u64, f32)> {
        if self.entry_point.is_none() || k == 0 {
            return Vec::new();
        }

        let entry_point = self.entry_point.unwrap();
        let ef = ef_search.max(k);

        // Phase 1: Greedily descend from top layer to layer 1
        let mut current_entry = entry_point;
        for layer in (1..=self.max_layer).rev() {
            if layer < self.layers.len() {
                current_entry =
                    self.search_layer_greedy(query, current_entry, layer, distance_fn, vectors);
            }
        }

        // Phase 2: Search layer 0 with beam width = ef
        let mut results = self.search_layer_ef(
            query,
            &[current_entry],
            0,
            ef,
            allowed_ids,
            distance_fn,
            vectors,
        );
        results.truncate(k);
        results
    }

    fn len(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance;
    use crate::index::flat::FlatIndex;
    use crate::types::DistanceType;

    #[test]
    fn test_hnsw_basic_insert_and_search() {
        let distance_fn = distance::get_distance_fn(DistanceType::Euclidean);
        let mut store = VectorStore::new();
        let mut index = HnswIndex::new(16, 200);

        let vectors_data: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..32)
                    .map(|j| ((i * 32 + j) as f32 * 0.01).sin())
                    .collect()
            })
            .collect();

        for (i, v) in vectors_data.iter().enumerate() {
            let id = i as u64;
            store.insert(id, v.clone());
            index.add(id, v, distance_fn.as_ref(), &store);
        }

        assert_eq!(index.len(), 100);

        // Search for first vector — should find itself
        let results = index.search(&vectors_data[0], 1, 100, None, distance_fn.as_ref(), &store);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_hnsw_self_search_recall() {
        let distance_fn = distance::get_distance_fn(DistanceType::Cosine);
        let mut store = VectorStore::new();
        let mut index = HnswIndex::new(16, 200);

        let n = 500;
        let dim = 32;
        let mut rng = rand::thread_rng();

        let vectors_data: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() - 0.5).collect())
            .collect();

        for (i, v) in vectors_data.iter().enumerate() {
            let id = i as u64;
            store.insert(id, v.clone());
            index.add(id, v, distance_fn.as_ref(), &store);
        }

        // Self-search: each vector should be its own top result
        let mut self_found = 0;
        for (i, v) in vectors_data.iter().enumerate() {
            let results = index.search(v, 1, 100, None, distance_fn.as_ref(), &store);
            if !results.is_empty() && results[0].0 == i as u64 {
                self_found += 1;
            }
        }

        let recall = self_found as f64 / n as f64;
        assert!(
            recall >= 0.95,
            "Self-search recall@1 = {recall:.3} (expected >= 0.95)"
        );
    }

    #[test]
    fn test_hnsw_vs_flat_recall() {
        let distance_fn = distance::get_distance_fn(DistanceType::Cosine);
        let mut store = VectorStore::new();
        let mut hnsw = HnswIndex::new(16, 200);
        let mut flat = FlatIndex::new();

        let n = 500;
        let dim = 32;
        let k = 10;
        let mut rng = rand::thread_rng();

        let vectors_data: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() - 0.5).collect())
            .collect();

        for (i, v) in vectors_data.iter().enumerate() {
            let id = i as u64;
            store.insert(id, v.clone());
            hnsw.add(id, v, distance_fn.as_ref(), &store);
            flat.add(id, v, distance_fn.as_ref(), &store);
        }

        // Compare HNSW results to flat (ground truth)
        let num_queries = 50;
        let query_vectors: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() - 0.5).collect())
            .collect();

        let mut total_recall = 0.0;

        for q in &query_vectors {
            let flat_results = flat.search(q, k, 0, None, distance_fn.as_ref(), &store);
            let hnsw_results = hnsw.search(q, k, 200, None, distance_fn.as_ref(), &store);

            let flat_ids: HashSet<u64> = flat_results.iter().map(|r| r.0).collect();
            let hnsw_ids: HashSet<u64> = hnsw_results.iter().map(|r| r.0).collect();

            let overlap = flat_ids.intersection(&hnsw_ids).count();
            total_recall += overlap as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall >= 0.90,
            "HNSW recall@{k} = {avg_recall:.3} (expected >= 0.90)"
        );
    }

    #[test]
    fn test_hnsw_remove() {
        let distance_fn = distance::get_distance_fn(DistanceType::Euclidean);
        let mut store = VectorStore::new();
        let mut index = HnswIndex::new(16, 200);

        store.insert(0, vec![0.0, 0.0]);
        store.insert(1, vec![1.0, 0.0]);
        store.insert(2, vec![2.0, 0.0]);

        for id in 0..3 {
            index.add(id, store.get(id).unwrap(), distance_fn.as_ref(), &store);
        }

        assert_eq!(index.len(), 3);
        index.remove(1);
        assert_eq!(index.len(), 2);

        let results = index.search(&[1.0, 0.0], 10, 100, None, distance_fn.as_ref(), &store);
        // Should NOT contain id=1
        for r in &results {
            assert_ne!(r.0, 1);
        }
    }

    #[test]
    fn test_hnsw_with_filter() {
        let distance_fn = distance::get_distance_fn(DistanceType::Euclidean);
        let mut store = VectorStore::new();
        let mut index = HnswIndex::new(16, 200);

        for i in 0..100u64 {
            let v = vec![i as f32, 0.0];
            store.insert(i, v.clone());
            index.add(i, &v, distance_fn.as_ref(), &store);
        }

        // Only allow even IDs
        let allowed: HashSet<u64> = (0..100).filter(|x| x % 2 == 0).collect();
        let results = index.search(
            &[5.0, 0.0],
            3,
            100,
            Some(&allowed),
            distance_fn.as_ref(),
            &store,
        );

        for r in &results {
            assert!(r.0 % 2 == 0, "Expected only even IDs, got {}", r.0);
        }
    }
}
