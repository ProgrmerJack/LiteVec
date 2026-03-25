//! Index algorithms.
//!
//! Provides a trait for vector indexes and implementations (Flat, HNSW).

pub mod diskann;
pub mod flat;
pub mod hnsw;
pub mod pq;
pub mod quantize;

use std::collections::HashSet;

use crate::distance::DistanceFn;

/// Trait for a vector search index.
pub trait VectorIndex: Send + Sync {
    /// Add a vector to the index.
    fn add(
        &mut self,
        internal_id: u64,
        vector: &[f32],
        distance_fn: &dyn DistanceFn,
        vectors: &VectorStore,
    );

    /// Remove a vector from the index.
    fn remove(&mut self, internal_id: u64);

    /// Search for the top-k nearest neighbors.
    ///
    /// `allowed_ids` — if Some, only consider vectors with these internal IDs.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        allowed_ids: Option<&HashSet<u64>>,
        distance_fn: &dyn DistanceFn,
        vectors: &VectorStore,
    ) -> Vec<(u64, f32)>;

    /// Number of indexed vectors.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Simple in-memory vector store: maps internal_id → vector data.
pub struct VectorStore {
    vectors: std::collections::HashMap<u64, Vec<f32>>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            vectors: std::collections::HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        self.vectors.insert(id, vector);
    }

    pub fn get(&self, id: u64) -> Option<&[f32]> {
        self.vectors.get(&id).map(|v| v.as_slice())
    }

    pub fn remove(&mut self, id: u64) -> Option<Vec<f32>> {
        self.vectors.remove(&id)
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Iterate over all (id, vector) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &[f32])> {
        self.vectors.iter().map(|(&id, v)| (id, v.as_slice()))
    }
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}
