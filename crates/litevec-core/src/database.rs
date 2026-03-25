//! Database management.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::collection::Collection;
use crate::error::{Error, Result};
use crate::persistence::{self, DatabaseSnapshot};
use crate::storage::file::FileStorage;
use crate::storage::page::DEFAULT_PAGE_SIZE;
use crate::storage::wal::Wal;
use crate::storage::{MemoryStorage, StorageBackend};
use crate::types::{BackupCollectionInfo, BackupInfo, CollectionConfig, DatabaseConfig};

/// A LiteVec database backed by a single file.
///
/// Thread-safe. Clone to share across threads.
#[derive(Clone)]
pub struct Database {
    inner: Arc<RwLock<DatabaseInner>>,
}

struct DatabaseInner {
    _storage: Box<dyn StorageBackend>,
    _wal: Wal,
    collections: HashMap<String, Collection>,
    _config: DatabaseConfig,
    /// Path to the database file (None for in-memory databases).
    db_path: Option<PathBuf>,
}

impl Database {
    /// Open or create a database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_config(path, DatabaseConfig::default())
    }

    /// Open with custom configuration.
    pub fn open_with_config<P: AsRef<Path>>(path: P, config: DatabaseConfig) -> Result<Self> {
        let path = path.as_ref();
        let storage = FileStorage::open(path, config.page_size)?;
        let wal = if config.wal_enabled {
            Wal::open(path)?
        } else {
            Wal::in_memory()
        };

        // Load existing collections from snapshot
        let collections = match persistence::load_snapshot(path)? {
            Some(snapshot) => {
                let mut map = HashMap::new();
                for col_snap in &snapshot.collections {
                    let col = Collection::from_snapshot(col_snap);
                    map.insert(col_snap.name.clone(), col);
                }
                map
            }
            None => HashMap::new(),
        };

        Ok(Self {
            inner: Arc::new(RwLock::new(DatabaseInner {
                _storage: Box::new(storage),
                _wal: wal,
                collections,
                _config: config,
                db_path: Some(path.to_path_buf()),
            })),
        })
    }

    /// Open an in-memory database (no file, data lost when dropped).
    pub fn open_memory() -> Result<Self> {
        let config = DatabaseConfig::default();
        let storage = MemoryStorage::new(DEFAULT_PAGE_SIZE);
        let wal = Wal::in_memory();

        Ok(Self {
            inner: Arc::new(RwLock::new(DatabaseInner {
                _storage: Box::new(storage),
                _wal: wal,
                collections: HashMap::new(),
                _config: config,
                db_path: None,
            })),
        })
    }

    /// Create a new collection with the given name and vector dimension.
    pub fn create_collection(&self, name: &str, dimension: u32) -> Result<Collection> {
        self.create_collection_with_config(name, CollectionConfig::new(dimension))
    }

    /// Create a collection with custom configuration.
    pub fn create_collection_with_config(
        &self,
        name: &str,
        config: CollectionConfig,
    ) -> Result<Collection> {
        let mut inner = self.inner.write();

        if inner.collections.contains_key(name) {
            return Err(Error::CollectionExists(name.to_string()));
        }

        let collection = Collection::new(name, config);
        inner
            .collections
            .insert(name.to_string(), collection.clone());

        Ok(collection)
    }

    /// Get an existing collection by name.
    pub fn get_collection(&self, name: &str) -> Option<Collection> {
        let inner = self.inner.read();
        inner.collections.get(name).cloned()
    }

    /// Delete a collection and all its data.
    pub fn delete_collection(&self, name: &str) -> Result<()> {
        let mut inner = self.inner.write();

        if inner.collections.remove(name).is_none() {
            return Err(Error::CollectionNotFound(name.to_string()));
        }

        Ok(())
    }

    /// List all collection names.
    pub fn list_collections(&self) -> Vec<String> {
        let inner = self.inner.read();
        inner.collections.keys().cloned().collect()
    }

    /// Force a WAL checkpoint and save all collections to disk.
    pub fn checkpoint(&self) -> Result<()> {
        let inner = self.inner.read();
        if let Some(ref path) = inner.db_path {
            let snapshot = DatabaseSnapshot {
                version: 1,
                collections: inner
                    .collections
                    .values()
                    .map(|c| c.to_snapshot())
                    .collect(),
            };
            persistence::save_snapshot(path, &snapshot)?;
        }
        Ok(())
    }

    /// Close the database gracefully, persisting all data.
    pub fn close(self) -> Result<()> {
        self.checkpoint()
    }

    /// Create a backup snapshot at the given path.
    ///
    /// This creates a complete, self-contained backup that can be restored
    /// later with [`Database::restore_from_backup`].
    pub fn create_backup<P: AsRef<Path>>(&self, backup_path: P) -> Result<()> {
        let inner = self.inner.read();
        let snapshot = DatabaseSnapshot {
            version: 1,
            collections: inner
                .collections
                .values()
                .map(|c| c.to_snapshot())
                .collect(),
        };
        persistence::save_snapshot_to(backup_path.as_ref(), &snapshot)?;
        Ok(())
    }

    /// Restore a database from a backup snapshot file.
    ///
    /// Returns a new in-memory `Database` with all collections and vectors
    /// restored from the backup.
    pub fn restore_from_backup<P: AsRef<Path>>(backup_path: P) -> Result<Self> {
        let snapshot = persistence::load_snapshot_from(backup_path.as_ref())?.ok_or_else(|| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "backup snapshot not found",
            ))
        })?;

        let db = Self::open_memory()?;
        {
            let mut inner = db.inner.write();
            for col_snap in &snapshot.collections {
                let col = Collection::from_snapshot(col_snap);
                inner.collections.insert(col_snap.name.clone(), col);
            }
        }
        Ok(db)
    }

    /// Get backup metadata without fully loading the backup.
    ///
    /// Returns information such as the number of collections, total vectors,
    /// and per-collection details.
    pub fn backup_info<P: AsRef<Path>>(backup_path: P) -> Result<BackupInfo> {
        let snapshot = persistence::load_snapshot_from(backup_path.as_ref())?.ok_or_else(|| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "backup snapshot not found",
            ))
        })?;

        let collections: Vec<BackupCollectionInfo> = snapshot
            .collections
            .iter()
            .map(|c| BackupCollectionInfo {
                name: c.name.clone(),
                dimension: c.config.dimension,
                num_vectors: c.vectors.len(),
            })
            .collect();

        let total_vectors = collections.iter().map(|c| c.num_vectors).sum();

        Ok(BackupInfo {
            version: snapshot.version,
            num_collections: collections.len(),
            total_vectors,
            collections,
        })
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // Only persist if this is the last reference
        if Arc::strong_count(&self.inner) == 1 {
            let _ = self.checkpoint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_create_and_restore_backup() {
        let backup_path =
            std::env::temp_dir().join(format!("litevec_backup_{}.snap", std::process::id()));
        let _ = std::fs::remove_file(&backup_path);

        // Create a database with data
        let db = Database::open_memory().unwrap();
        let col1 = db.create_collection("docs", 3).unwrap();
        col1.insert("d1", &[1.0, 0.0, 0.0], json!({"title": "Hello"}))
            .unwrap();
        col1.insert("d2", &[0.0, 1.0, 0.0], json!({"title": "World"}))
            .unwrap();

        let col2 = db.create_collection("images", 2).unwrap();
        col2.insert("i1", &[0.5, 0.5], json!({"format": "png"}))
            .unwrap();

        // Backup
        db.create_backup(&backup_path).unwrap();

        // Restore
        let restored = Database::restore_from_backup(&backup_path).unwrap();

        // Verify collections
        let mut names = restored.list_collections();
        names.sort();
        assert_eq!(names, vec!["docs", "images"]);

        // Verify docs collection data
        let docs = restored.get_collection("docs").unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs.dimension(), 3);
        let rec = docs.get("d1").unwrap().unwrap();
        assert_eq!(rec.vector, vec![1.0, 0.0, 0.0]);
        assert_eq!(rec.metadata["title"], "Hello");

        // Verify search works after restore
        let results = docs.search(&[0.9, 0.1, 0.0], 2).execute().unwrap();
        assert_eq!(results[0].id, "d1");

        // Verify images collection
        let images = restored.get_collection("images").unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images.dimension(), 2);

        let _ = std::fs::remove_file(&backup_path);
    }

    #[test]
    fn test_backup_info() {
        let backup_path =
            std::env::temp_dir().join(format!("litevec_info_{}.snap", std::process::id()));
        let _ = std::fs::remove_file(&backup_path);

        let db = Database::open_memory().unwrap();
        let col = db.create_collection("vectors", 4).unwrap();
        col.insert("a", &[1.0, 0.0, 0.0, 0.0], json!({})).unwrap();
        col.insert("b", &[0.0, 1.0, 0.0, 0.0], json!({})).unwrap();
        col.insert("c", &[0.0, 0.0, 1.0, 0.0], json!({})).unwrap();

        db.create_collection("empty", 2).unwrap();
        db.create_backup(&backup_path).unwrap();

        let info = Database::backup_info(&backup_path).unwrap();
        assert_eq!(info.version, 1);
        assert_eq!(info.num_collections, 2);
        assert_eq!(info.total_vectors, 3);

        let mut col_infos = info.collections.clone();
        col_infos.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(col_infos[0].name, "empty");
        assert_eq!(col_infos[0].dimension, 2);
        assert_eq!(col_infos[0].num_vectors, 0);
        assert_eq!(col_infos[1].name, "vectors");
        assert_eq!(col_infos[1].dimension, 4);
        assert_eq!(col_infos[1].num_vectors, 3);

        let _ = std::fs::remove_file(&backup_path);
    }

    #[test]
    fn test_restore_nonexistent_backup() {
        let result = Database::restore_from_backup("/tmp/does_not_exist_litevec.snap");
        assert!(result.is_err());
    }

    #[test]
    fn test_backup_empty_database() {
        let backup_path =
            std::env::temp_dir().join(format!("litevec_empty_{}.snap", std::process::id()));
        let _ = std::fs::remove_file(&backup_path);

        let db = Database::open_memory().unwrap();
        db.create_backup(&backup_path).unwrap();

        let info = Database::backup_info(&backup_path).unwrap();
        assert_eq!(info.num_collections, 0);
        assert_eq!(info.total_vectors, 0);
        assert!(info.collections.is_empty());

        let restored = Database::restore_from_backup(&backup_path).unwrap();
        assert!(restored.list_collections().is_empty());

        let _ = std::fs::remove_file(&backup_path);
    }

    #[test]
    fn test_open_memory() {
        let db = Database::open_memory().unwrap();
        assert!(db.list_collections().is_empty());
    }

    #[test]
    fn test_create_collection() {
        let db = Database::open_memory().unwrap();
        let col = db.create_collection("test", 128).unwrap();
        assert_eq!(col.dimension(), 128);
        assert_eq!(col.name(), "test");
    }

    #[test]
    fn test_create_duplicate_collection() {
        let db = Database::open_memory().unwrap();
        db.create_collection("test", 128).unwrap();
        let result = db.create_collection("test", 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_collection() {
        let db = Database::open_memory().unwrap();
        db.create_collection("test", 128).unwrap();

        let col = db.get_collection("test");
        assert!(col.is_some());
        assert_eq!(col.unwrap().dimension(), 128);

        assert!(db.get_collection("nonexistent").is_none());
    }

    #[test]
    fn test_delete_collection() {
        let db = Database::open_memory().unwrap();
        db.create_collection("test", 128).unwrap();
        db.delete_collection("test").unwrap();

        assert!(db.get_collection("test").is_none());
        assert!(db.list_collections().is_empty());
    }

    #[test]
    fn test_list_collections() {
        let db = Database::open_memory().unwrap();
        db.create_collection("a", 64).unwrap();
        db.create_collection("b", 128).unwrap();
        db.create_collection("c", 256).unwrap();

        let mut names = db.list_collections();
        names.sort();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_full_workflow() {
        let db = Database::open_memory().unwrap();
        let col = db.create_collection("docs", 3).unwrap();

        col.insert("doc1", &[1.0, 0.0, 0.0], json!({"title": "Hello"}))
            .unwrap();
        col.insert("doc2", &[0.0, 1.0, 0.0], json!({"title": "World"}))
            .unwrap();

        let results = col.search(&[0.9, 0.1, 0.0], 2).execute().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc1");

        let record = col.get("doc1").unwrap().unwrap();
        assert_eq!(record.metadata["title"], "Hello");

        col.delete("doc1").unwrap();
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn test_file_backed_database() {
        let path = std::env::temp_dir().join(format!("litevec_test_db_{}.lv", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let snap_path = path.with_extension("lv.snap");
        let _ = std::fs::remove_file(&snap_path);

        // Create and populate database
        {
            let db = Database::open(&path).unwrap();
            let col = db.create_collection("test", 3).unwrap();
            col.insert("v1", &[1.0, 0.0, 0.0], json!({"label": "first"}))
                .unwrap();
            col.insert("v2", &[0.0, 1.0, 0.0], json!({"label": "second"}))
                .unwrap();
            assert_eq!(col.len(), 2);
            db.close().unwrap();
        }

        // Reopen and verify data persisted
        {
            let db = Database::open(&path).unwrap();
            let names = db.list_collections();
            assert_eq!(names, vec!["test"]);

            let col = db.get_collection("test").unwrap();
            assert_eq!(col.len(), 2);
            assert_eq!(col.dimension(), 3);

            let record = col.get("v1").unwrap().unwrap();
            assert_eq!(record.vector, vec![1.0, 0.0, 0.0]);
            assert_eq!(record.metadata["label"], "first");

            let record2 = col.get("v2").unwrap().unwrap();
            assert_eq!(record2.vector, vec![0.0, 1.0, 0.0]);

            // Search should work after reload
            let results = col.search(&[0.9, 0.1, 0.0], 2).execute().unwrap();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].id, "v1");
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&snap_path);
    }

    #[test]
    fn test_thread_safety() {
        let db = Database::open_memory().unwrap();
        let col = db.create_collection("test", 3).unwrap();

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let col = col.clone();
                std::thread::spawn(move || {
                    for i in 0..25 {
                        let id = format!("t{t}_v{i}");
                        let v = vec![t as f32, i as f32, 0.0];
                        col.insert(&id, &v, json!({})).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(col.len(), 100);
    }
}
