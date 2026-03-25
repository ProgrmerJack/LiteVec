//! Collection CRUD operations.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::distance::{self, DistanceFn};
use crate::error::{Error, Result};
use crate::index::flat::FlatIndex;
use crate::index::hnsw::HnswIndex;
use crate::index::{VectorIndex, VectorStore};
use crate::metadata::filter::evaluate_filter;
use crate::metadata::fulltext::FullTextIndex;
use crate::metadata::hybrid::{self, FusionStrategy};
use crate::metadata::secondary::SecondaryIndexManager;
use crate::metadata::store::MetadataStore;
use crate::persistence::{CollectionSnapshot, VectorEntry};
use crate::query::SearchQuery;
use crate::types::{CollectionConfig, DistanceType, Filter, IndexType, SearchResult, VectorRecord};

/// Internal shared state of a collection.
#[derive(Clone)]
pub(crate) struct CollectionInner {
    inner: Arc<RwLock<CollectionData>>,
}

struct CollectionData {
    name: String,
    config: CollectionConfig,
    distance_fn: Box<dyn DistanceFn>,
    index: Box<dyn VectorIndex>,
    vector_store: VectorStore,
    metadata_store: MetadataStore,
    fulltext_index: FullTextIndex,
    secondary_indexes: SecondaryIndexManager,
    id_map: HashMap<String, u64>,      // string ID → internal ID
    reverse_map: HashMap<u64, String>, // internal ID → string ID
    next_internal_id: u64,
}

impl CollectionData {
    /// Concatenate all string values from metadata for FTS indexing.
    fn extract_text_content(metadata: &serde_json::Value) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(obj) = metadata.as_object() {
            for value in obj.values() {
                if let Some(s) = value.as_str() {
                    parts.push(s.to_string());
                }
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" "))
        }
    }

    /// Helper to add a vector to the index with split borrows.
    fn add_to_index(&mut self, internal_id: u64, vector: &[f32]) {
        // Split borrow: get mutable ref to index and immutable refs to the rest
        let distance_fn = self.distance_fn.as_ref();
        let vector_store = &self.vector_store;
        self.index
            .add(internal_id, vector, distance_fn, vector_store);
    }

    /// Create a secondary index, using split borrows on struct fields.
    fn create_secondary_index(&mut self, field: &str) {
        self.secondary_indexes
            .create_index(field, &self.metadata_store);
    }
}

impl CollectionInner {
    pub(crate) fn new(name: &str, config: CollectionConfig) -> Self {
        let distance_fn = distance::get_distance_fn(config.distance);
        let index: Box<dyn VectorIndex> = match config.index {
            IndexType::Flat => Box::new(FlatIndex::new()),
            IndexType::Hnsw | IndexType::Auto => {
                Box::new(HnswIndex::new(config.hnsw.m, config.hnsw.ef_construction))
            }
        };

        Self {
            inner: Arc::new(RwLock::new(CollectionData {
                name: name.to_string(),
                config,
                distance_fn,
                index,
                vector_store: VectorStore::new(),
                metadata_store: MetadataStore::new(),
                fulltext_index: FullTextIndex::new(),
                secondary_indexes: SecondaryIndexManager::new(),
                id_map: HashMap::new(),
                reverse_map: HashMap::new(),
                next_internal_id: 0,
            })),
        }
    }

    pub(crate) fn execute_search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        ef_search_override: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        let data = self.inner.read();

        if query.len() as u32 != data.config.dimension {
            return Err(Error::DimensionMismatch {
                expected: data.config.dimension,
                got: query.len() as u32,
            });
        }

        let allowed_ids = filter.map(|f| evaluate_filter(f, &data.metadata_store));

        let ef = ef_search_override.unwrap_or(data.config.hnsw.ef_search);
        let raw_results = data.index.search(
            query,
            k,
            ef,
            allowed_ids.as_ref(),
            data.distance_fn.as_ref(),
            &data.vector_store,
        );

        let results = raw_results
            .into_iter()
            .filter_map(|(internal_id, dist)| {
                let string_id = data.reverse_map.get(&internal_id)?.clone();
                let metadata = data
                    .metadata_store
                    .get(internal_id)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                Some(SearchResult {
                    id: string_id,
                    distance: dist,
                    metadata,
                })
            })
            .collect();

        Ok(results)
    }
}

/// A collection of vectors with metadata.
///
/// Thread-safe. Clone to share across threads.
#[derive(Clone)]
pub struct Collection {
    pub(crate) inner: CollectionInner,
}

impl Collection {
    pub(crate) fn new(name: &str, config: CollectionConfig) -> Self {
        Self {
            inner: CollectionInner::new(name, config),
        }
    }

    /// Insert a vector with a string ID and JSON metadata.
    ///
    /// If a vector with this ID already exists, it is overwritten (upsert).
    pub fn insert(&self, id: &str, vector: &[f32], metadata: serde_json::Value) -> Result<()> {
        let mut data = self.inner.inner.write();

        if vector.len() as u32 != data.config.dimension {
            return Err(Error::DimensionMismatch {
                expected: data.config.dimension,
                got: vector.len() as u32,
            });
        }

        // If ID exists, remove old entry first (upsert)
        if let Some(&old_internal) = data.id_map.get(id) {
            let old_metadata = data.metadata_store.get(old_internal).cloned();
            data.index.remove(old_internal);
            data.vector_store.remove(old_internal);
            data.metadata_store.remove(old_internal);
            data.reverse_map.remove(&old_internal);
            data.fulltext_index.remove_document(old_internal);
            if let Some(ref meta) = old_metadata {
                data.secondary_indexes.on_remove(old_internal, meta);
            }
        }

        let internal_id = data.next_internal_id;
        data.next_internal_id += 1;

        // Index for full-text and secondary indexes before metadata is moved into store
        if let Some(text) = CollectionData::extract_text_content(&metadata) {
            data.fulltext_index.add_document(internal_id, &text);
        }
        data.secondary_indexes.on_insert(internal_id, &metadata);

        data.vector_store.insert(internal_id, vector.to_vec());
        data.metadata_store.insert(internal_id, metadata);
        data.id_map.insert(id.to_string(), internal_id);
        data.reverse_map.insert(internal_id, id.to_string());
        data.add_to_index(internal_id, vector);

        Ok(())
    }

    /// Insert multiple vectors in a batch.
    pub fn insert_batch(&self, items: &[(&str, &[f32], serde_json::Value)]) -> Result<()> {
        for (id, vector, metadata) in items {
            self.insert(id, vector, metadata.clone())?;
        }
        Ok(())
    }

    /// Search for the k nearest vectors to the query.
    pub fn search(&self, query: &[f32], k: usize) -> SearchQuery {
        SearchQuery {
            query: query.to_vec(),
            k,
            filter: None,
            ef_search: None,
            collection: self.inner.clone(),
        }
    }

    /// Get a vector and its metadata by ID.
    pub fn get(&self, id: &str) -> Result<Option<VectorRecord>> {
        let data = self.inner.inner.read();

        let internal_id = match data.id_map.get(id) {
            Some(&id) => id,
            None => return Ok(None),
        };

        let vector = data
            .vector_store
            .get(internal_id)
            .map(|v| v.to_vec())
            .unwrap_or_default();
        let metadata = data
            .metadata_store
            .get(internal_id)
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        Ok(Some(VectorRecord {
            id: id.to_string(),
            vector,
            metadata,
        }))
    }

    /// Delete a vector by ID. Returns true if the vector existed.
    pub fn delete(&self, id: &str) -> Result<bool> {
        let mut data = self.inner.inner.write();

        let internal_id = match data.id_map.remove(id) {
            Some(id) => id,
            None => return Ok(false),
        };

        let old_metadata = data.metadata_store.get(internal_id).cloned();

        data.index.remove(internal_id);
        data.vector_store.remove(internal_id);
        data.metadata_store.remove(internal_id);
        data.reverse_map.remove(&internal_id);

        // Clean up full-text and secondary indexes
        data.fulltext_index.remove_document(internal_id);
        if let Some(ref meta) = old_metadata {
            data.secondary_indexes.on_remove(internal_id, meta);
        }

        Ok(true)
    }

    /// Update metadata for a vector.
    pub fn update_metadata(&self, id: &str, metadata: serde_json::Value) -> Result<()> {
        let mut data = self.inner.inner.write();

        let internal_id = match data.id_map.get(id) {
            Some(&id) => id,
            None => return Err(Error::VectorNotFound(id.to_string())),
        };

        // Remove old entries from indexes
        let old_metadata = data.metadata_store.get(internal_id).cloned();
        data.fulltext_index.remove_document(internal_id);
        if let Some(ref meta) = old_metadata {
            data.secondary_indexes.on_remove(internal_id, meta);
        }

        // Add new entries to indexes before metadata is moved into store
        if let Some(text) = CollectionData::extract_text_content(&metadata) {
            data.fulltext_index.add_document(internal_id, &text);
        }
        data.secondary_indexes.on_insert(internal_id, &metadata);

        data.metadata_store.insert(internal_id, metadata);
        Ok(())
    }

    /// Number of vectors in this collection.
    pub fn len(&self) -> usize {
        self.inner.inner.read().vector_store.len()
    }

    /// Is the collection empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector dimension.
    pub fn dimension(&self) -> u32 {
        self.inner.inner.read().config.dimension
    }

    /// Collection name.
    pub fn name(&self) -> String {
        self.inner.inner.read().name.clone()
    }

    /// Distance type.
    pub fn distance_type(&self) -> DistanceType {
        self.inner.inner.read().config.distance
    }

    /// Full-text search across metadata string fields.
    /// Returns results sorted by BM25 relevance score (highest first).
    pub fn text_search(&self, query: &str, limit: usize) -> Vec<SearchResult> {
        let data = self.inner.inner.read();
        let fts_results = data.fulltext_index.search(query, limit);

        fts_results
            .into_iter()
            .filter_map(|(internal_id, score)| {
                let string_id = data.reverse_map.get(&internal_id)?.clone();
                let metadata = data
                    .metadata_store
                    .get(internal_id)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                Some(SearchResult {
                    id: string_id,
                    distance: score,
                    metadata,
                })
            })
            .collect()
    }

    /// Hybrid search combining vector similarity and keyword matching.
    /// Uses Reciprocal Rank Fusion (RRF) to merge results.
    pub fn hybrid_search(
        &self,
        vector: &[f32],
        text_query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        self.hybrid_search_with_strategy(vector, text_query, k, FusionStrategy::default())
    }

    /// Hybrid search with a custom fusion strategy.
    pub fn hybrid_search_with_strategy(
        &self,
        vector: &[f32],
        text_query: &str,
        k: usize,
        strategy: FusionStrategy,
    ) -> Result<Vec<SearchResult>> {
        let data = self.inner.inner.read();

        if vector.len() as u32 != data.config.dimension {
            return Err(Error::DimensionMismatch {
                expected: data.config.dimension,
                got: vector.len() as u32,
            });
        }

        // Vector search
        let ef = data.config.hnsw.ef_search;
        let raw_vector =
            data.index
                .search(vector, k, ef, None, data.distance_fn.as_ref(), &data.vector_store);

        // Keyword search
        let raw_keyword = data.fulltext_index.search(text_query, k);

        // Fuse
        let fused = hybrid::hybrid_search(&raw_vector, &raw_keyword, strategy, k);

        let results = fused
            .into_iter()
            .filter_map(|hr| {
                let string_id = data.reverse_map.get(&hr.id)?.clone();
                let metadata = data
                    .metadata_store
                    .get(hr.id)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                Some(SearchResult {
                    id: string_id,
                    distance: hr.score,
                    metadata,
                })
            })
            .collect();

        Ok(results)
    }

    /// Create a secondary index on a metadata field for faster filtered search.
    pub fn create_index(&self, field: &str) {
        let mut data = self.inner.inner.write();
        data.create_secondary_index(field);
    }

    /// Drop a secondary index.
    pub fn drop_index(&self, field: &str) {
        let mut data = self.inner.inner.write();
        data.secondary_indexes.drop_index(field);
    }

    /// List currently indexed fields.
    pub fn indexed_fields(&self) -> Vec<String> {
        self.inner.inner.read().secondary_indexes.indexed_fields()
    }

    /// Create a serializable snapshot of this collection.
    pub(crate) fn to_snapshot(&self) -> CollectionSnapshot {
        let data = self.inner.inner.read();

        let mut vectors = Vec::with_capacity(data.id_map.len());
        for (string_id, &internal_id) in &data.id_map {
            let vector = data
                .vector_store
                .get(internal_id)
                .map(|v| v.to_vec())
                .unwrap_or_default();
            let metadata = data
                .metadata_store
                .get(internal_id)
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            vectors.push(VectorEntry {
                id: string_id.clone(),
                internal_id,
                vector,
                metadata,
            });
        }

        CollectionSnapshot {
            name: data.name.clone(),
            config: (&data.config).into(),
            vectors,
            next_internal_id: data.next_internal_id,
        }
    }

    /// Restore a collection from a snapshot, rebuilding the index.
    pub(crate) fn from_snapshot(snap: &CollectionSnapshot) -> Self {
        let config: CollectionConfig = (&snap.config).into();
        let col = Self::new(&snap.name, config);

        {
            let mut data = col.inner.inner.write();
            data.next_internal_id = snap.next_internal_id;

            for entry in &snap.vectors {
                data.vector_store
                    .insert(entry.internal_id, entry.vector.clone());
                data.metadata_store
                    .insert(entry.internal_id, entry.metadata.clone());
                data.id_map.insert(entry.id.clone(), entry.internal_id);
                data.reverse_map.insert(entry.internal_id, entry.id.clone());
            }

            // Rebuild index from stored vectors
            for entry in &snap.vectors {
                data.add_to_index(entry.internal_id, &entry.vector);
            }

            // Rebuild full-text index from stored metadata
            for entry in &snap.vectors {
                if let Some(text) = CollectionData::extract_text_content(&entry.metadata) {
                    data.fulltext_index.add_document(entry.internal_id, &text);
                }
            }
        }

        col
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_collection(dim: u32) -> Collection {
        Collection::new("test", CollectionConfig::new(dim))
    }

    #[test]
    fn test_insert_and_get() {
        let col = make_collection(3);
        col.insert("v1", &[1.0, 2.0, 3.0], json!({"key": "value"}))
            .unwrap();

        let record = col.get("v1").unwrap().unwrap();
        assert_eq!(record.id, "v1");
        assert_eq!(record.vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(record.metadata, json!({"key": "value"}));
    }

    #[test]
    fn test_dimension_mismatch() {
        let col = make_collection(3);
        let result = col.insert("v1", &[1.0, 2.0], json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_upsert() {
        let col = make_collection(2);
        col.insert("v1", &[1.0, 0.0], json!({"v": 1})).unwrap();
        col.insert("v1", &[0.0, 1.0], json!({"v": 2})).unwrap();

        assert_eq!(col.len(), 1);
        let record = col.get("v1").unwrap().unwrap();
        assert_eq!(record.vector, vec![0.0, 1.0]);
        assert_eq!(record.metadata, json!({"v": 2}));
    }

    #[test]
    fn test_delete() {
        let col = make_collection(2);
        col.insert("v1", &[1.0, 0.0], json!({})).unwrap();
        assert_eq!(col.len(), 1);

        assert!(col.delete("v1").unwrap());
        assert_eq!(col.len(), 0);
        assert!(col.get("v1").unwrap().is_none());
        assert!(!col.delete("v1").unwrap()); // Already deleted
    }

    #[test]
    fn test_search() {
        let col = make_collection(3);
        col.insert("a", &[1.0, 0.0, 0.0], json!({})).unwrap();
        col.insert("b", &[0.0, 1.0, 0.0], json!({})).unwrap();
        col.insert("c", &[0.0, 0.0, 1.0], json!({})).unwrap();

        let results = col.search(&[1.0, 0.1, 0.0], 2).execute().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a"); // Closest to [1,0.1,0]
    }

    #[test]
    fn test_search_with_filter() {
        let col = make_collection(2);
        col.insert("a", &[1.0, 0.0], json!({"type": "A"})).unwrap();
        col.insert("b", &[0.9, 0.1], json!({"type": "B"})).unwrap();
        col.insert("c", &[0.0, 1.0], json!({"type": "A"})).unwrap();

        let results = col
            .search(&[1.0, 0.0], 10)
            .filter(Filter::Eq("type".into(), json!("A")))
            .execute()
            .unwrap();

        for r in &results {
            assert_ne!(r.id, "b", "Filtered results should not contain 'b'");
        }
        assert!(results.iter().any(|r| r.id == "a"));
    }

    #[test]
    fn test_update_metadata() {
        let col = make_collection(2);
        col.insert("v1", &[1.0, 0.0], json!({"old": true})).unwrap();
        col.update_metadata("v1", json!({"new": true})).unwrap();

        let record = col.get("v1").unwrap().unwrap();
        assert_eq!(record.metadata, json!({"new": true}));
    }

    #[test]
    fn test_batch_insert() {
        let col = make_collection(2);
        let items: Vec<(&str, &[f32], serde_json::Value)> = vec![
            ("a", &[1.0, 0.0], json!({})),
            ("b", &[0.0, 1.0], json!({})),
            ("c", &[1.0, 1.0], json!({})),
        ];
        col.insert_batch(&items).unwrap();
        assert_eq!(col.len(), 3);
    }

    #[test]
    fn test_collection_metadata() {
        let col = make_collection(384);
        assert_eq!(col.dimension(), 384);
        assert_eq!(col.name(), "test");
        assert!(col.is_empty());
    }

    #[test]
    fn test_text_search() {
        let col = make_collection(3);
        col.insert(
            "d1",
            &[1.0, 0.0, 0.0],
            json!({"title": "rust programming language"}),
        )
        .unwrap();
        col.insert(
            "d2",
            &[0.0, 1.0, 0.0],
            json!({"title": "python scripting"}),
        )
        .unwrap();
        col.insert(
            "d3",
            &[0.0, 0.0, 1.0],
            json!({"title": "rust web framework"}),
        )
        .unwrap();

        let results = col.text_search("rust", 10);
        assert!(results.len() >= 2);
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"d1"));
        assert!(ids.contains(&"d3"));
    }

    #[test]
    fn test_hybrid_search() {
        let col = make_collection(3);
        col.insert(
            "d1",
            &[1.0, 0.0, 0.0],
            json!({"title": "machine learning algorithms"}),
        )
        .unwrap();
        col.insert(
            "d2",
            &[0.0, 1.0, 0.0],
            json!({"title": "deep learning neural networks"}),
        )
        .unwrap();
        col.insert(
            "d3",
            &[0.9, 0.1, 0.0],
            json!({"title": "statistical methods"}),
        )
        .unwrap();

        // Hybrid: vector close to d1/d3, text matches "learning" in d1/d2
        let results = col
            .hybrid_search(&[0.95, 0.05, 0.0], "learning", 3)
            .unwrap();
        assert!(!results.is_empty());
        // d1 should rank high (matches both vector and text)
        assert_eq!(results[0].id, "d1");
    }

    #[test]
    fn test_secondary_index() {
        let col = make_collection(2);
        col.insert("a", &[1.0, 0.0], json!({"category": "tech"}))
            .unwrap();
        col.insert("b", &[0.0, 1.0], json!({"category": "science"}))
            .unwrap();
        col.insert("c", &[0.5, 0.5], json!({"category": "tech"}))
            .unwrap();

        col.create_index("category");
        let fields = col.indexed_fields();
        assert!(fields.contains(&"category".to_string()));

        // Verify the index doesn't break search
        let results = col
            .search(&[1.0, 0.0], 10)
            .filter(Filter::Eq("category".into(), json!("tech")))
            .execute()
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_text_search_after_delete() {
        let col = make_collection(2);
        col.insert("a", &[1.0, 0.0], json!({"title": "hello world"}))
            .unwrap();
        col.insert("b", &[0.0, 1.0], json!({"title": "hello universe"}))
            .unwrap();

        col.delete("a").unwrap();

        let results = col.text_search("hello", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }
}
