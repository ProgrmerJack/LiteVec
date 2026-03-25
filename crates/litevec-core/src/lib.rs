//! # LiteVec Core
//!
//! Core engine for the LiteVec embedded vector database.

pub mod collection;
pub mod database;
pub mod distance;
pub mod error;
pub mod index;
pub mod metadata;
pub(crate) mod persistence;
pub mod query;
pub mod storage;
pub mod types;

// Re-export public API types
pub use collection::Collection;
pub use database::Database;
pub use error::{Error, Result};
pub use metadata::hybrid::FusionStrategy;
pub use query::SearchQuery;
pub use types::{
    BackupCollectionInfo, BackupInfo, CollectionConfig, DatabaseConfig, DistanceType, Filter,
    HnswConfig, IndexType, SearchResult, VectorRecord,
};
