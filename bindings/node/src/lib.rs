#![deny(clippy::all)]

use litevec_core::{Collection as CoreCollection, Database as CoreDatabase, Filter};
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ─────────────────────── helper ───────────────────────

fn parse_filter(json: &str) -> Result<Option<Filter>> {
    let val: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Error::from_reason(format!("invalid filter JSON: {e}")))?;
    parse_filter_value(&val).map(Some)
}

fn parse_filter_value(val: &serde_json::Value) -> Result<Filter> {
    let obj = val
        .as_object()
        .ok_or_else(|| Error::from_reason("filter must be an object"))?;

    if let Some(and) = obj.get("$and") {
        let arr = and
            .as_array()
            .ok_or_else(|| Error::from_reason("$and must be an array"))?;
        let filters: Result<Vec<Filter>> = arr.iter().map(parse_filter_value).collect();
        return Ok(Filter::And(filters?));
    }
    if let Some(or) = obj.get("$or") {
        let arr = or
            .as_array()
            .ok_or_else(|| Error::from_reason("$or must be an array"))?;
        let filters: Result<Vec<Filter>> = arr.iter().map(parse_filter_value).collect();
        return Ok(Filter::Or(filters?));
    }

    let mut filters = Vec::new();
    for (field, value) in obj {
        if let Some(inner) = value.as_object() {
            for (op, v) in inner {
                let f = match op.as_str() {
                    "$eq" => Filter::Eq(field.clone(), v.clone()),
                    "$ne" => Filter::Ne(field.clone(), v.clone()),
                    "$gt" => Filter::Gt(
                        field.clone(),
                        v.as_f64()
                            .ok_or_else(|| Error::from_reason("$gt requires a number"))?,
                    ),
                    "$gte" => Filter::Gte(
                        field.clone(),
                        v.as_f64()
                            .ok_or_else(|| Error::from_reason("$gte requires a number"))?,
                    ),
                    "$lt" => Filter::Lt(
                        field.clone(),
                        v.as_f64()
                            .ok_or_else(|| Error::from_reason("$lt requires a number"))?,
                    ),
                    "$lte" => Filter::Lte(
                        field.clone(),
                        v.as_f64()
                            .ok_or_else(|| Error::from_reason("$lte requires a number"))?,
                    ),
                    "$in" => {
                        let arr = v
                            .as_array()
                            .ok_or_else(|| Error::from_reason("$in requires an array"))?;
                        Filter::In(field.clone(), arr.clone())
                    }
                    _ => return Err(Error::from_reason(format!("unknown filter operator: {op}"))),
                };
                filters.push(f);
            }
        } else {
            filters.push(Filter::Eq(field.clone(), value.clone()));
        }
    }

    match filters.len() {
        0 => Err(Error::from_reason("empty filter")),
        1 => Ok(filters.into_iter().next().unwrap()),
        _ => Ok(Filter::And(filters)),
    }
}

// ─────────────────────── Database ───────────────────────

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

    /// Create a backup snapshot at the given path.
    #[napi]
    pub fn create_backup(&self, path: String) -> Result<()> {
        self.inner
            .create_backup(&path)
            .map_err(|e| Error::from_reason(e.to_string()))
    }
}

// ─────────────────────── types ───────────────────────

/// A search result.
#[napi(object)]
pub struct SearchResult {
    pub id: String,
    pub distance: f64,
    /// JSON-encoded metadata string.
    pub metadata: String,
}

/// A vector record.
#[napi(object)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f64>,
    /// JSON-encoded metadata string.
    pub metadata: String,
}

/// An item for batch insertion.
#[napi(object)]
pub struct BatchItem {
    pub id: String,
    pub vector: Vec<f64>,
    /// Optional JSON-encoded metadata string.
    pub metadata: Option<String>,
}

// ─────────────────────── Collection ───────────────────────

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

    /// Insert multiple vectors in a batch. Returns the number of inserted items.
    #[napi]
    pub fn insert_batch(&self, items: Vec<BatchItem>) -> Result<u32> {
        let mut converted: Vec<(String, Vec<f32>, serde_json::Value)> =
            Vec::with_capacity(items.len());
        for item in &items {
            let f32_vec: Vec<f32> = item.vector.iter().map(|&v| v as f32).collect();
            let meta = match &item.metadata {
                Some(json) => serde_json::from_str(json)
                    .map_err(|e| Error::from_reason(format!("invalid metadata JSON: {e}")))?,
                None => serde_json::Value::Object(serde_json::Map::new()),
            };
            converted.push((item.id.clone(), f32_vec, meta));
        }
        let batch: Vec<(&str, &[f32], serde_json::Value)> = converted
            .iter()
            .map(|(id, vec, meta)| (id.as_str(), vec.as_slice(), meta.clone()))
            .collect();
        self.inner
            .insert_batch(&batch)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(converted.len() as u32)
    }

    /// Search for the k nearest vectors. An optional JSON filter string restricts results.
    ///
    /// Supports operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$and`, `$or`.
    #[napi]
    pub fn search(
        &self,
        query: Vec<f64>,
        k: u32,
        filter: Option<String>,
    ) -> Result<Vec<SearchResult>> {
        let f32_query: Vec<f32> = query.iter().map(|&v| v as f32).collect();
        let f = match filter {
            Some(ref json) => parse_filter(json)?,
            None => None,
        };
        let mut search = self.inner.search(&f32_query, k as usize);
        if let Some(f) = f {
            search = search.filter(f);
        }
        let results = search
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

    /// Full-text keyword search across metadata string fields.
    ///
    /// Returns results ranked by BM25 relevance.
    #[napi]
    pub fn text_search(&self, query: String, limit: Option<u32>) -> Result<Vec<SearchResult>> {
        let limit = limit.unwrap_or(10) as usize;
        let results = self.inner.text_search(&query, limit);
        Ok(results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                distance: r.distance as f64,
                metadata: serde_json::to_string(&r.metadata).unwrap_or_default(),
            })
            .collect())
    }

    /// Combined vector + keyword search using Reciprocal Rank Fusion.
    #[napi]
    pub fn hybrid_search(
        &self,
        vector: Vec<f64>,
        query: String,
        k: Option<u32>,
    ) -> Result<Vec<SearchResult>> {
        let f32_query: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
        let k = k.unwrap_or(10) as usize;
        let results = self
            .inner
            .hybrid_search(&f32_query, &query, k)
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

    /// Update metadata for an existing vector. Metadata is a JSON string.
    #[napi]
    pub fn update_metadata(&self, id: String, metadata: String) -> Result<()> {
        let meta: serde_json::Value = serde_json::from_str(&metadata)
            .map_err(|e| Error::from_reason(format!("invalid JSON: {e}")))?;
        self.inner
            .update_metadata(&id, meta)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Create a secondary index on a metadata field for faster filtered search.
    #[napi]
    pub fn create_index(&self, field: String) {
        self.inner.create_index(&field);
    }

    /// Drop a secondary index on a metadata field.
    #[napi]
    pub fn drop_index(&self, field: String) {
        self.inner.drop_index(&field);
    }

    /// List the currently indexed metadata fields.
    #[napi]
    pub fn indexed_fields(&self) -> Vec<String> {
        self.inner.indexed_fields()
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
