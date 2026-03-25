//! Core type definitions for LiteVec.

use serde::{Deserialize, Serialize};

/// Distance function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceType {
    /// Cosine distance (1 - cosine_similarity). Default.
    Cosine,
    /// Euclidean (L2) distance.
    Euclidean,
    /// Negative dot product (so lower = more similar).
    DotProduct,
}

/// Index type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// Flat (brute force). Best for < 1000 vectors.
    Flat,
    /// HNSW. Best for >= 1000 vectors.
    Hnsw,
    /// Automatic: uses Flat for < 1000, HNSW otherwise.
    Auto,
}

/// Configuration for opening a database.
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// Page size in bytes. Default: 4096.
    pub page_size: usize,
    /// Whether to enable WAL mode. Default: true.
    pub wal_enabled: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            page_size: 4096,
            wal_enabled: true,
        }
    }
}

/// Configuration for creating a collection.
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    /// Vector dimension (required).
    pub dimension: u32,
    /// Distance function. Default: Cosine.
    pub distance: DistanceType,
    /// Index type. Default: Auto.
    pub index: IndexType,
    /// HNSW configuration (only used if index = HNSW or Auto).
    pub hnsw: HnswConfig,
}

impl CollectionConfig {
    /// Create a new config with the given dimension and sensible defaults.
    pub fn new(dimension: u32) -> Self {
        Self {
            dimension,
            distance: DistanceType::Cosine,
            index: IndexType::Auto,
            hnsw: HnswConfig::default(),
        }
    }
}

/// HNSW index configuration.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Max connections per node per layer. Default: 16.
    pub m: usize,
    /// Dynamic candidate list size during construction. Default: 200.
    pub ef_construction: usize,
    /// Dynamic candidate list size during search. Default: 100.
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
        }
    }
}

/// A search result containing the matched vector's ID, distance, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The string ID of the vector.
    pub id: String,
    /// The distance from the query vector (lower = more similar).
    pub distance: f32,
    /// The metadata associated with this vector.
    pub metadata: serde_json::Value,
}

/// A stored vector record.
#[derive(Debug, Clone)]
pub struct VectorRecord {
    /// The string ID of the vector.
    pub id: String,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// The metadata associated with this vector.
    pub metadata: serde_json::Value,
}

/// Metadata filter for constraining search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Filter {
    /// Field equals value.
    Eq(String, serde_json::Value),
    /// Field not equals value.
    Ne(String, serde_json::Value),
    /// Field greater than value (numeric).
    Gt(String, f64),
    /// Field greater than or equal (numeric).
    Gte(String, f64),
    /// Field less than value (numeric).
    Lt(String, f64),
    /// Field less than or equal (numeric).
    Lte(String, f64),
    /// Field value is in list.
    In(String, Vec<serde_json::Value>),
    /// Field exists.
    Exists(String),
    /// Logical AND of multiple filters.
    And(Vec<Filter>),
    /// Logical OR of multiple filters.
    Or(Vec<Filter>),
    /// Logical NOT.
    Not(Box<Filter>),
}

/// Information about a backup snapshot.
#[derive(Debug, Clone)]
pub struct BackupInfo {
    /// Snapshot format version.
    pub version: u32,
    /// Number of collections in the backup.
    pub num_collections: usize,
    /// Total number of vectors across all collections.
    pub total_vectors: usize,
    /// Per-collection details.
    pub collections: Vec<BackupCollectionInfo>,
}

/// Per-collection information inside a backup snapshot.
#[derive(Debug, Clone)]
pub struct BackupCollectionInfo {
    /// Collection name.
    pub name: String,
    /// Vector dimension.
    pub dimension: u32,
    /// Number of vectors in this collection.
    pub num_vectors: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_database_config() {
        let config = DatabaseConfig::default();
        assert_eq!(config.page_size, 4096);
        assert!(config.wal_enabled);
    }

    #[test]
    fn test_default_hnsw_config() {
        let config = HnswConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 100);
    }

    #[test]
    fn test_collection_config_new() {
        let config = CollectionConfig::new(384);
        assert_eq!(config.dimension, 384);
        assert_eq!(config.distance, DistanceType::Cosine);
        assert_eq!(config.index, IndexType::Auto);
    }

    #[test]
    fn test_distance_type_serialization() {
        let dt = DistanceType::Cosine;
        let json = serde_json::to_string(&dt).unwrap();
        let deserialized: DistanceType = serde_json::from_str(&json).unwrap();
        assert_eq!(dt, deserialized);
    }

    #[test]
    fn test_filter_construction() {
        let filter = Filter::And(vec![
            Filter::Eq("category".into(), serde_json::json!("science")),
            Filter::Gte("year".into(), 2024.0),
        ]);
        // Just verify it constructs without panic
        match &filter {
            Filter::And(filters) => assert_eq!(filters.len(), 2),
            _ => panic!("Expected And filter"),
        }
    }

    #[test]
    fn test_search_result_serialization() {
        let result = SearchResult {
            id: "doc1".to_string(),
            distance: 0.123,
            metadata: serde_json::json!({"title": "Hello"}),
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: SearchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "doc1");
        assert!((deserialized.distance - 0.123).abs() < 1e-6);
    }
}
