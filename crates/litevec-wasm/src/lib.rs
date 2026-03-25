use litevec_core::{Collection, Database};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmDatabase {
    inner: Database,
}

#[wasm_bindgen]
impl WasmDatabase {
    /// Create an in-memory database (browser use only).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmDatabase, JsError> {
        let db = Database::open_memory().map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmDatabase { inner: db })
    }

    /// Create a new collection with the given name and dimension.
    #[wasm_bindgen(js_name = "createCollection")]
    pub fn create_collection(&self, name: &str, dimension: u32) -> Result<WasmCollection, JsError> {
        let col = self
            .inner
            .create_collection(name, dimension)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmCollection { inner: col })
    }

    /// Get an existing collection by name.
    #[wasm_bindgen(js_name = "getCollection")]
    pub fn get_collection(&self, name: &str) -> Result<WasmCollection, JsError> {
        self.inner
            .get_collection(name)
            .map(|col| WasmCollection { inner: col })
            .ok_or_else(|| JsError::new("collection not found"))
    }

    /// Delete a collection.
    #[wasm_bindgen(js_name = "deleteCollection")]
    pub fn delete_collection(&self, name: &str) -> Result<(), JsError> {
        self.inner
            .delete_collection(name)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// List all collection names.
    #[wasm_bindgen(js_name = "listCollections")]
    pub fn list_collections(&self) -> Vec<String> {
        self.inner.list_collections()
    }
}

#[wasm_bindgen]
pub struct WasmCollection {
    inner: Collection,
}

/// Serializable wrapper for VectorRecord (which lacks a Serialize derive).
#[derive(Serialize)]
struct JsVectorRecord {
    id: String,
    vector: Vec<f32>,
    metadata: serde_json::Value,
}

#[wasm_bindgen]
impl WasmCollection {
    /// Insert a vector with an ID and optional metadata JSON.
    pub fn insert(
        &self,
        id: &str,
        vector: &[f32],
        metadata_json: Option<String>,
    ) -> Result<(), JsError> {
        let metadata = match metadata_json {
            Some(json) => serde_json::from_str(&json).map_err(|e| JsError::new(&e.to_string()))?,
            None => serde_json::Value::Object(serde_json::Map::new()),
        };
        self.inner
            .insert(id, vector, metadata)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Search for the k nearest vectors. Returns a JS array of results.
    pub fn search(&self, query: &[f32], k: u32) -> Result<JsValue, JsError> {
        let results = self
            .inner
            .search(query, k as usize)
            .execute()
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&results).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get a vector by ID. Returns null if not found.
    pub fn get(&self, id: &str) -> Result<JsValue, JsError> {
        let record = self
            .inner
            .get(id)
            .map_err(|e| JsError::new(&e.to_string()))?;
        match record {
            Some(r) => {
                let js_record = JsVectorRecord {
                    id: r.id,
                    vector: r.vector,
                    metadata: r.metadata,
                };
                serde_wasm_bindgen::to_value(&js_record).map_err(|e| JsError::new(&e.to_string()))
            }
            None => Ok(JsValue::NULL),
        }
    }

    /// Delete a vector by ID. Returns true if it existed.
    pub fn delete(&self, id: &str) -> Result<bool, JsError> {
        self.inner
            .delete(id)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Number of vectors in the collection.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Vector dimension of the collection.
    pub fn dimension(&self) -> u32 {
        self.inner.dimension()
    }
}
