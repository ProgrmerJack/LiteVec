//! Database persistence — serialize/deserialize collections to disk.
//!
//! Uses a JSON snapshot file alongside the main database file.
//! On close, all collections are serialized. On open, they are restored
//! and indexes are rebuilt from the vector data.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::types::{CollectionConfig, DistanceType, HnswConfig, IndexType};

/// Serializable snapshot of the entire database state.
#[derive(Serialize, Deserialize)]
pub(crate) struct DatabaseSnapshot {
    pub version: u32,
    pub collections: Vec<CollectionSnapshot>,
}

/// Serializable snapshot of a single collection.
#[derive(Serialize, Deserialize)]
pub(crate) struct CollectionSnapshot {
    pub name: String,
    pub config: CollectionConfigSnapshot,
    pub vectors: Vec<VectorEntry>,
    pub next_internal_id: u64,
}

/// Serializable collection config.
#[derive(Serialize, Deserialize)]
pub(crate) struct CollectionConfigSnapshot {
    pub dimension: u32,
    pub distance: DistanceType,
    pub index: IndexType,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
}

impl From<&CollectionConfig> for CollectionConfigSnapshot {
    fn from(c: &CollectionConfig) -> Self {
        Self {
            dimension: c.dimension,
            distance: c.distance,
            index: c.index,
            hnsw_m: c.hnsw.m,
            hnsw_ef_construction: c.hnsw.ef_construction,
            hnsw_ef_search: c.hnsw.ef_search,
        }
    }
}

impl From<&CollectionConfigSnapshot> for CollectionConfig {
    fn from(c: &CollectionConfigSnapshot) -> Self {
        Self {
            dimension: c.dimension,
            distance: c.distance,
            index: c.index,
            hnsw: HnswConfig {
                m: c.hnsw_m,
                ef_construction: c.hnsw_ef_construction,
                ef_search: c.hnsw_ef_search,
            },
        }
    }
}

/// A single vector entry for serialization.
#[derive(Serialize, Deserialize)]
pub(crate) struct VectorEntry {
    pub id: String,
    pub internal_id: u64,
    pub vector: Vec<f32>,
    pub metadata: serde_json::Value,
}

/// Get the snapshot file path for a database path.
pub(crate) fn snapshot_path(db_path: &Path) -> PathBuf {
    let mut snap = db_path.to_path_buf();
    let ext = match snap.extension() {
        Some(e) => format!("{}.snap", e.to_string_lossy()),
        None => "snap".to_string(),
    };
    snap.set_extension(ext);
    snap
}

/// Write a database snapshot to disk.
///
/// The snapshot is written next to the database file with a `.snap` suffix
/// derived from `path` via [`snapshot_path`].
pub fn save_snapshot(path: &Path, snapshot: &DatabaseSnapshot) -> Result<()> {
    let snap_path = snapshot_path(path);
    let data = serde_json::to_vec(snapshot)
        .map_err(|e| Error::Serialization(format!("failed to serialize snapshot: {e}")))?;
    std::fs::write(&snap_path, data)?;
    Ok(())
}

/// Load a database snapshot from disk, if it exists.
///
/// The snapshot file is located via [`snapshot_path`].
pub fn load_snapshot(path: &Path) -> Result<Option<DatabaseSnapshot>> {
    let snap_path = snapshot_path(path);
    if !snap_path.exists() {
        return Ok(None);
    }
    let data = std::fs::read(&snap_path)?;
    let snapshot: DatabaseSnapshot = serde_json::from_slice(&data)
        .map_err(|e| Error::Serialization(format!("failed to deserialize snapshot: {e}")))?;
    Ok(Some(snapshot))
}

/// Write a database snapshot directly to the given file path.
///
/// Unlike [`save_snapshot`], this writes to `path` exactly as given,
/// without deriving a `.snap` suffix.
pub fn save_snapshot_to(path: &Path, snapshot: &DatabaseSnapshot) -> Result<()> {
    let data = serde_json::to_vec(snapshot)
        .map_err(|e| Error::Serialization(format!("failed to serialize snapshot: {e}")))?;
    std::fs::write(path, data)?;
    Ok(())
}

/// Load a database snapshot directly from the given file path.
///
/// Unlike [`load_snapshot`], this reads from `path` exactly as given.
/// Returns `Ok(None)` if the file does not exist.
pub fn load_snapshot_from(path: &Path) -> Result<Option<DatabaseSnapshot>> {
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read(path)?;
    let snapshot: DatabaseSnapshot = serde_json::from_slice(&data)
        .map_err(|e| Error::Serialization(format!("failed to deserialize snapshot: {e}")))?;
    Ok(Some(snapshot))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_path() {
        let p = snapshot_path(Path::new("/tmp/test.lv"));
        assert_eq!(p, PathBuf::from("/tmp/test.lv.snap"));

        let p2 = snapshot_path(Path::new("/tmp/mydb"));
        assert_eq!(p2, PathBuf::from("/tmp/mydb.snap"));
    }

    #[test]
    fn test_roundtrip_snapshot() {
        let snap = DatabaseSnapshot {
            version: 1,
            collections: vec![CollectionSnapshot {
                name: "test".into(),
                config: CollectionConfigSnapshot {
                    dimension: 3,
                    distance: DistanceType::Cosine,
                    index: IndexType::Auto,
                    hnsw_m: 16,
                    hnsw_ef_construction: 200,
                    hnsw_ef_search: 100,
                },
                vectors: vec![VectorEntry {
                    id: "v1".into(),
                    internal_id: 0,
                    vector: vec![1.0, 2.0, 3.0],
                    metadata: serde_json::json!({"key": "val"}),
                }],
                next_internal_id: 1,
            }],
        };

        let json = serde_json::to_vec(&snap).unwrap();
        let restored: DatabaseSnapshot = serde_json::from_slice(&json).unwrap();

        assert_eq!(restored.version, 1);
        assert_eq!(restored.collections.len(), 1);
        assert_eq!(restored.collections[0].name, "test");
        assert_eq!(restored.collections[0].vectors.len(), 1);
        assert_eq!(restored.collections[0].vectors[0].id, "v1");
    }
}
