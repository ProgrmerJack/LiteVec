#![deny(clippy::all)]

use litevec_core::{Collection as CoreCollection, Database as CoreDatabase};
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// A LiteVec database instance.
#[napi]
pub struct Database {
    inner: CoreDatabase,
}

#[napi]
impl Database {
    /// Open a file-backed database at the given path.
    #[napi(factory)]
    pub fn open(path: String) -> Result<Database> {
        let db = CoreDatabase::open(&path).map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Database { inner: db })
    }

    /// Open an in-memory database.
    #[napi(factory)]
    pub fn open_memory() -> Result<Database> {
        let db = CoreDatabase::open_memory().map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Database { inner: db })
    }

    /// Create a new collection.
    #[napi]
    pub fn create_collection(&self, name: String, dimension: u32) -> Result<Collection> {
        let col = self
            .inner
            .create_collection(&name, dimension)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Collection { inner: col })
    }

    /// Get an existing collection.
    #[napi]
    pub fn get_collection(&self, name: String) -> Result<Collection> {
        self.inner
            .get_collection(&name)
            .map(|col| Collection { inner: col })
            .ok_or_else(|| Error::from_reason("collection not found"))
    }

    /// Delete a collection.
    #[napi]
    pub fn delete_collection(&self, name: String) -> Result<()> {
        self.inner
            .delete_collection(&name)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// List all collection names.
    #[napi]
    pub fn list_collections(&self) -> Vec<String> {
        self.inner.list_collections()
    }

    /// Persist data to disk (for file-backed databases).
    #[napi]
    pub fn checkpoint(&self) -> Result<()> {
        self.inner
            .checkpoint()
            .map_err(|e| Error::from_reason(e.to_string()))
    }
}

/// A search result.
#[napi(object)]
pub struct SearchResult {
    pub id: String,
    pub distance: f64,
    pub metadata: String,
}

/// A vector record.
#[napi(object)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f64>,
    pub metadata: String,
}

/// A LiteVec collection.
#[napi]
pub struct Collection {
    inner: CoreCollection,
}

#[napi]
impl Collection {
    /// Insert a vector with an ID and optional JSON metadata string.
    #[napi]
    pub fn insert(&self, id: String, vector: Vec<f64>, metadata: Option<String>) -> Result<()> {
        let f32_vec: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
        let meta = match metadata {
            Some(json) => serde_json::from_str(&json)
                .map_err(|e| Error::from_reason(format!("invalid metadata JSON: {e}")))?,
            None => serde_json::Value::Object(serde_json::Map::new()),
        };
        self.inner
            .insert(&id, &f32_vec, meta)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Search for the k nearest vectors. Returns results as an array of objects.
    #[napi]
    pub fn search(&self, query: Vec<f64>, k: u32) -> Result<Vec<SearchResult>> {
        let f32_query: Vec<f32> = query.iter().map(|&v| v as f32).collect();
        let results = self
            .inner
            .search(&f32_query, k as usize)
            .execute()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                distance: r.distance as f64,
                metadata: serde_json::to_string(&r.metadata).unwrap_or_default(),
            })
            .collect())
    }

    /// Get a vector by ID.
    #[napi]
    pub fn get(&self, id: String) -> Result<Option<VectorRecord>> {
        let record = self
            .inner
            .get(&id)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(record.map(|r| VectorRecord {
            id: r.id,
            vector: r.vector.iter().map(|&v| v as f64).collect(),
            metadata: serde_json::to_string(&r.metadata).unwrap_or_default(),
        }))
    }

    /// Delete a vector by ID.
    #[napi]
    pub fn delete(&self, id: String) -> Result<bool> {
        self.inner
            .delete(&id)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Number of vectors in the collection.
    #[napi]
    pub fn len(&self) -> u32 {
        self.inner.len() as u32
    }

    /// Whether the collection is empty.
    #[napi]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Vector dimension.
    #[napi]
    pub fn dimension(&self) -> u32 {
        self.inner.dimension()
    }

    /// Collection name.
    #[napi]
    pub fn name(&self) -> String {
        self.inner.name()
    }
}
