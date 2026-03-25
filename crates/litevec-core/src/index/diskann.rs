//! DiskANN-inspired Vamana graph index.
//!
//! A simplified but functional implementation of the Vamana graph-based
//! approximate nearest neighbor index from the DiskANN paper (Subramanya et al., 2019).
//! Supports beam search from a medoid entry point and robust pruning to bound
//! the maximum out-degree of every node.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use ordered_float::OrderedFloat;

use super::{VectorIndex, VectorStore};
use crate::distance::DistanceFn;

/// Vamana graph-based approximate nearest neighbor index.
pub struct VamanaIndex {
    /// Maximum out-degree per node.
    max_degree: usize,
    /// Distance stretch factor for robust pruning (typically 1.2).
    alpha: f32,
    /// Beam width used during graph search (L parameter).
    search_list_size: usize,
    /// Adjacency lists: node → sorted neighbor list.
    graph: HashMap<u64, Vec<u64>>,
    /// Entry point (approximate medoid) of the graph.
    medoid: Option<u64>,
    /// Number of insertions since last medoid recalculation.
    inserts_since_medoid: usize,
}

impl VamanaIndex {
    /// Create a new Vamana index.
    ///
    /// * `max_degree` — maximum out-degree R (e.g. 64).
    /// * `alpha` — robust-pruning stretch factor (e.g. 1.2).
    /// * `search_list_size` — beam width L for search (e.g. 128).
    pub fn new(max_degree: usize, alpha: f32, search_list_size: usize) -> Self {
        Self {
            max_degree,
            alpha,
            search_list_size,
            graph: HashMap::new(),
            medoid: None,
            inserts_since_medoid: 0,
        }
    }

    // ── Beam search (greedy search with beam width) ──────────────────────

    /// Search the graph starting from the medoid, returning up to `list_size`
    /// closest candidates as `(id, distance)` pairs sorted by distance.
    fn greedy_search(
        &self,
        query: &[f32],
        list_size: usize,
        distance_fn: &dyn DistanceFn,
        store: &VectorStore,
    ) -> Vec<(u64, f32)> {
        let entry = match self.medoid {
            Some(id) => id,
            None => return Vec::new(),
        };

        let mut visited: HashSet<u64> = HashSet::new();
        // Min-heap of candidates (closest first)
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, u64)>> = BinaryHeap::new();
        // Max-heap of best results (farthest first for easy pruning)
        let mut results: BinaryHeap<(OrderedFloat<f32>, u64)> = BinaryHeap::new();

        if let Some(v) = store.get(entry) {
            let d = distance_fn.compute(query, v);
            visited.insert(entry);
            candidates.push(Reverse((OrderedFloat(d), entry)));
            results.push((OrderedFloat(d), entry));
        }

        while let Some(Reverse((cand_dist, cand_id))) = candidates.pop() {
            // Early termination: candidate farther than worst result and we have enough
            if results.len() >= list_size
                && let Some(&(worst, _)) = results.peek()
                && cand_dist > worst
            {
                break;
            }

            if let Some(neighbors) = self.graph.get(&cand_id) {
                for &nbr in neighbors {
                    if !visited.insert(nbr) {
                        continue;
                    }
                    if let Some(v) = store.get(nbr) {
                        let d = distance_fn.compute(query, v);
                        let dominated = results.len() >= list_size
                            && results
                                .peek()
                                .is_some_and(|&(worst, _)| OrderedFloat(d) >= worst);
                        if !dominated {
                            candidates.push(Reverse((OrderedFloat(d), nbr)));
                            results.push((OrderedFloat(d), nbr));
                            if results.len() > list_size {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<(u64, f32)> = results.into_iter().map(|(d, id)| (id, d.0)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        out
    }

    // ── Robust pruning ───────────────────────────────────────────────────

    /// Robust pruning (Algorithm 2 from the DiskANN paper).
    ///
    /// Given a node `p` and a set of `candidates` (id, distance-to-p),
    /// return a neighbor list of at most `max_degree` nodes such that
    /// no candidate is "dominated" by an already-selected neighbor.
    fn robust_prune(
        &self,
        candidates: &[(u64, f32)],
        max_degree: usize,
        alpha: f32,
        distance_fn: &dyn DistanceFn,
        store: &VectorStore,
    ) -> Vec<u64> {
        // Sort candidates by distance to p (ascending)
        let mut sorted: Vec<(u64, f32)> = candidates.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut selected: Vec<u64> = Vec::with_capacity(max_degree);

        for &(cand_id, cand_dist) in &sorted {
            if selected.len() >= max_degree {
                break;
            }
            // Check if `cand_id` is dominated by any already-selected neighbor
            let dominated = selected.iter().any(|&sel_id| {
                if let (Some(sel_vec), Some(cand_vec)) = (store.get(sel_id), store.get(cand_id)) {
                    let sel_to_cand = distance_fn.compute(sel_vec, cand_vec);
                    sel_to_cand < alpha * cand_dist
                } else {
                    false
                }
            });
            if !dominated {
                selected.push(cand_id);
            }
        }

        selected
    }

    // ── Medoid computation ───────────────────────────────────────────────

    /// Recompute the medoid (the node closest to the centroid of all vectors).
    fn recompute_medoid(&mut self, distance_fn: &dyn DistanceFn, store: &VectorStore) {
        let ids: Vec<u64> = self.graph.keys().copied().collect();
        if ids.is_empty() {
            self.medoid = None;
            return;
        }

        // Compute centroid
        let dim = store.get(ids[0]).map_or(0, |v| v.len());
        if dim == 0 {
            return;
        }
        let mut centroid = vec![0.0f32; dim];
        let mut count = 0usize;
        for &id in &ids {
            if let Some(v) = store.get(id) {
                for (c, &val) in centroid.iter_mut().zip(v.iter()) {
                    *c += val;
                }
                count += 1;
            }
        }
        if count == 0 {
            return;
        }
        for c in &mut centroid {
            *c /= count as f32;
        }

        // Find closest node to centroid
        let mut best_id = ids[0];
        let mut best_dist = f32::MAX;
        for &id in &ids {
            if let Some(v) = store.get(id) {
                let d = distance_fn.compute(&centroid, v);
                if d < best_dist {
                    best_dist = d;
                    best_id = id;
                }
            }
        }
        self.medoid = Some(best_id);
        self.inserts_since_medoid = 0;
    }

    /// Interval at which to recalculate the medoid. Recalculation is O(n·d)
    /// so we amortize it over a batch of inserts.
    fn medoid_refresh_interval(&self) -> usize {
        let n = self.graph.len();
        // Refresh every ~sqrt(n) inserts, minimum 32.
        (n as f64).sqrt().max(32.0) as usize
    }
}

// ── VectorIndex implementation ────────────────────────────────────────────

impl VectorIndex for VamanaIndex {
    fn add(
        &mut self,
        internal_id: u64,
        _vector: &[f32],
        distance_fn: &dyn DistanceFn,
        vectors: &VectorStore,
    ) {
        // Ensure node exists in the graph
        self.graph.entry(internal_id).or_default();

        // First node — becomes the medoid
        if self.graph.len() == 1 {
            self.medoid = Some(internal_id);
            self.inserts_since_medoid = 0;
            return;
        }

        // Greedy search to find candidate neighbors for the new node
        let search_size = self.search_list_size.max(self.max_degree);
        let candidates = self.greedy_search(_vector, search_size, distance_fn, vectors);

        // Robust-prune to select neighbors
        let neighbors = self.robust_prune(
            &candidates,
            self.max_degree,
            self.alpha,
            distance_fn,
            vectors,
        );

        // Set forward edges from internal_id
        self.graph.insert(internal_id, neighbors.clone());

        // Add reverse edges and prune back-neighbors if needed
        for &nbr in &neighbors {
            let back_list = self.graph.entry(nbr).or_default();
            if !back_list.contains(&internal_id) {
                back_list.push(internal_id);
            }
            // If the neighbor now exceeds max_degree, prune it
            if back_list.len() > self.max_degree
                && let Some(nbr_vec) = vectors.get(nbr)
            {
                let nbr_candidates: Vec<(u64, f32)> = back_list
                    .iter()
                    .filter_map(|&n| vectors.get(n).map(|v| (n, distance_fn.compute(nbr_vec, v))))
                    .collect();
                let pruned = self.robust_prune(
                    &nbr_candidates,
                    self.max_degree,
                    self.alpha,
                    distance_fn,
                    vectors,
                );
                *self.graph.get_mut(&nbr).unwrap() = pruned;
            }
        }

        // Periodically refresh the medoid
        self.inserts_since_medoid += 1;
        if self.inserts_since_medoid >= self.medoid_refresh_interval() {
            self.recompute_medoid(distance_fn, vectors);
        }
    }

    fn remove(&mut self, internal_id: u64) {
        if let Some(neighbors) = self.graph.remove(&internal_id) {
            // Remove internal_id from every neighbor's adjacency list
            for &nbr in &neighbors {
                if let Some(adj) = self.graph.get_mut(&nbr) {
                    adj.retain(|&id| id != internal_id);
                }
            }
            // Also scan all nodes (handles reverse edges not in our neighbor list)
            for adj in self.graph.values_mut() {
                adj.retain(|&id| id != internal_id);
            }

            // Update medoid if it was removed
            if self.medoid == Some(internal_id) {
                self.medoid = self.graph.keys().next().copied();
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
        if self.medoid.is_none() || k == 0 {
            return Vec::new();
        }

        let list_size = ef_search.max(k).max(self.search_list_size);
        let results = self.greedy_search(query, list_size, distance_fn, vectors);

        // Apply filter and take top-k
        let mut filtered: Vec<(u64, f32)> = results
            .into_iter()
            .filter(|(id, _)| allowed_ids.is_none_or(|allowed| allowed.contains(id)))
            .collect();
        filtered.truncate(k);
        filtered
    }

    fn len(&self) -> usize {
        self.graph.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance;
    use crate::index::flat::FlatIndex;
    use crate::types::DistanceType;
    use rand::Rng;
    use std::collections::VecDeque;

    /// Helper: generate deterministic-ish random vectors.
    fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() - 0.5).collect())
            .collect()
    }

    /// Build a VamanaIndex and VectorStore from a set of vectors.
    fn build_index(vecs: &[Vec<f32>], distance_fn: &dyn DistanceFn) -> (VamanaIndex, VectorStore) {
        let mut store = VectorStore::new();
        let mut index = VamanaIndex::new(64, 1.2, 128);
        for (i, v) in vecs.iter().enumerate() {
            let id = i as u64;
            store.insert(id, v.clone());
            index.add(id, v, distance_fn, &store);
        }
        (index, store)
    }

    #[test]
    fn test_basic_add_and_search() {
        let distance_fn = distance::get_distance_fn(DistanceType::Euclidean);
        let vecs: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32, 0.0, 0.0, 0.0]).collect();
        let (index, store) = build_index(&vecs, distance_fn.as_ref());

        assert_eq!(index.len(), 50);

        // Search for vec near id=25 — should return id=25 first
        let results = index.search(
            &[25.0, 0.0, 0.0, 0.0],
            5,
            128,
            None,
            distance_fn.as_ref(),
            &store,
        );
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 25, "Expected id 25 as closest");
    }

    #[test]
    fn test_search_recall_at_10() {
        let distance_fn = distance::get_distance_fn(DistanceType::Euclidean);
        let n = 500;
        let dim = 32;
        let k = 10;

        let vecs = random_vectors(n, dim);
        let (vamana, store) = build_index(&vecs, distance_fn.as_ref());

        // Ground truth via flat index
        let mut flat = FlatIndex::new();
        for (i, v) in vecs.iter().enumerate() {
            flat.add(i as u64, v, distance_fn.as_ref(), &store);
        }

        let queries = random_vectors(50, dim);
        let mut total_recall = 0.0;
        for q in &queries {
            let truth: HashSet<u64> = flat
                .search(q, k, 0, None, distance_fn.as_ref(), &store)
                .iter()
                .map(|r| r.0)
                .collect();
            let approx: HashSet<u64> = vamana
                .search(q, k, 256, None, distance_fn.as_ref(), &store)
                .iter()
                .map(|r| r.0)
                .collect();
            let overlap = truth.intersection(&approx).count();
            total_recall += overlap as f64 / k as f64;
        }

        let avg_recall = total_recall / queries.len() as f64;
        assert!(
            avg_recall >= 0.80,
            "Recall@{k} = {avg_recall:.3} (expected >= 0.80)"
        );
    }

    #[test]
    fn test_filtered_search() {
        let distance_fn = distance::get_distance_fn(DistanceType::Euclidean);
        let vecs: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32, 0.0]).collect();
        let (index, store) = build_index(&vecs, distance_fn.as_ref());

        // Only allow even ids
        let allowed: HashSet<u64> = (0..100).filter(|x| x % 2 == 0).collect();
        let results = index.search(
            &[5.0, 0.0],
            5,
            128,
            Some(&allowed),
            distance_fn.as_ref(),
            &store,
        );

        assert!(!results.is_empty());
        for (id, _) in &results {
            assert!(id % 2 == 0, "Expected only even IDs, got {id}");
        }
    }

    #[test]
    fn test_empty_index_search() {
        let distance_fn = distance::get_distance_fn(DistanceType::Euclidean);
        let index = VamanaIndex::new(64, 1.2, 128);
        let store = VectorStore::new();

        let results = index.search(&[1.0, 2.0], 5, 128, None, distance_fn.as_ref(), &store);
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_and_research() {
        let distance_fn = distance::get_distance_fn(DistanceType::Euclidean);
        let mut store = VectorStore::new();
        let mut index = VamanaIndex::new(64, 1.2, 128);

        for i in 0..20u64 {
            let v = vec![i as f32, 0.0];
            store.insert(i, v.clone());
            index.add(i, &v, distance_fn.as_ref(), &store);
        }

        assert_eq!(index.len(), 20);

        // Remove id=10
        index.remove(10);
        assert_eq!(index.len(), 19);

        // Searching near 10.0 should not return id=10
        let results = index.search(&[10.0, 0.0], 10, 128, None, distance_fn.as_ref(), &store);
        for (id, _) in &results {
            assert_ne!(*id, 10, "Removed id=10 should not appear");
        }
    }

    #[test]
    fn test_graph_connectivity() {
        let distance_fn = distance::get_distance_fn(DistanceType::Euclidean);
        let n = 200;
        let dim = 8;
        let vecs = random_vectors(n, dim);
        let (index, _store) = build_index(&vecs, distance_fn.as_ref());

        let medoid = index.medoid.expect("medoid should exist");

        // BFS from medoid
        let mut visited: HashSet<u64> = HashSet::new();
        let mut queue: VecDeque<u64> = VecDeque::new();
        visited.insert(medoid);
        queue.push_back(medoid);

        while let Some(node) = queue.pop_front() {
            if let Some(neighbors) = index.graph.get(&node) {
                for &nbr in neighbors {
                    if visited.insert(nbr) {
                        queue.push_back(nbr);
                    }
                }
            }
        }

        assert_eq!(
            visited.len(),
            n,
            "Not all nodes reachable from medoid: {}/{n} reachable",
            visited.len()
        );
    }
}
