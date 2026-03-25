//! PyO3 bindings for LiteVec.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

use litevec_core::{Collection, Database, Filter};

fn to_py_err(e: litevec_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Convert a Python object to a `serde_json::Value` by round-tripping through JSON.
fn dict_to_value(py: Python<'_>, obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Value> {
    let json_mod = py.import("json")?;
    let json_str: String = json_mod.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&json_str).map_err(|e| PyValueError::new_err(format!("Invalid JSON: {e}")))
}

/// Convert a `serde_json::Value` to a Python object by round-tripping through JSON.
fn value_to_py(py: Python<'_>, val: &Value) -> PyResult<PyObject> {
    let json_mod = py.import("json")?;
    let json_str = serde_json::to_string(val)
        .map_err(|e| PyValueError::new_err(format!("JSON error: {e}")))?;
    let result = json_mod.call_method1("loads", (json_str,))?;
    Ok(result.into_pyobject(py)?.into_any().unbind())
}

/// Convert a Python dict (possibly containing filter operators) to a `Filter`.
///
/// Supports operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$and`, `$or`.
/// Plain `{"field": value}` is treated as `$eq`.
fn dict_to_filter(py: Python<'_>, obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Option<Filter>> {
    if obj.is_none() {
        return Ok(None);
    }
    let val = dict_to_value(py, obj)?;
    parse_filter_value(&val).map(Some)
}

fn parse_filter_value(val: &Value) -> PyResult<Filter> {
    let obj = val
        .as_object()
        .ok_or_else(|| PyValueError::new_err("filter must be a dict"))?;

    if let Some(and) = obj.get("$and") {
        let arr = and
            .as_array()
            .ok_or_else(|| PyValueError::new_err("$and must be an array"))?;
        let filters: PyResult<Vec<Filter>> = arr.iter().map(parse_filter_value).collect();
        return Ok(Filter::And(filters?));
    }
    if let Some(or) = obj.get("$or") {
        let arr = or
            .as_array()
            .ok_or_else(|| PyValueError::new_err("$or must be an array"))?;
        let filters: PyResult<Vec<Filter>> = arr.iter().map(parse_filter_value).collect();
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
                            .ok_or_else(|| PyValueError::new_err("$gt requires a number"))?,
                    ),
                    "$gte" => Filter::Gte(
                        field.clone(),
                        v.as_f64()
                            .ok_or_else(|| PyValueError::new_err("$gte requires a number"))?,
                    ),
                    "$lt" => Filter::Lt(
                        field.clone(),
                        v.as_f64()
                            .ok_or_else(|| PyValueError::new_err("$lt requires a number"))?,
                    ),
                    "$lte" => Filter::Lte(
                        field.clone(),
                        v.as_f64()
                            .ok_or_else(|| PyValueError::new_err("$lte requires a number"))?,
                    ),
                    "$in" => {
                        let arr = v
                            .as_array()
                            .ok_or_else(|| PyValueError::new_err("$in requires an array"))?;
                        Filter::In(field.clone(), arr.clone())
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown filter operator: {op}"
                        )))
                    }
                };
                filters.push(f);
            }
        } else {
            filters.push(Filter::Eq(field.clone(), value.clone()));
        }
    }

    match filters.len() {
        0 => Err(PyValueError::new_err("Empty filter")),
        1 => Ok(filters.into_iter().next().unwrap()),
        _ => Ok(Filter::And(filters)),
    }
}

// ─────────────────────── SearchResult ───────────────────────

/// A single search result containing the vector ID, distance, and metadata.
#[pyclass(name = "PySearchResult")]
#[derive(Clone)]
pub struct PySearchResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub distance: f32,
    metadata_val: Value,
}

#[pymethods]
impl PySearchResult {
    /// The metadata dict associated with this result.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        value_to_py(py, &self.metadata_val)
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(id='{}', distance={:.6})",
            self.id, self.distance
        )
    }
}

// ─────────────────────── Collection ───────────────────────

/// A named collection of vectors with a fixed dimension.
#[pyclass(name = "PyCollection")]
#[derive(Clone)]
pub struct PyCollection {
    inner: Collection,
}

#[pymethods]
impl PyCollection {
    /// The name of this collection.
    #[getter]
    fn name(&self) -> String {
        self.inner.name()
    }

    /// The vector dimension of this collection.
    #[getter]
    fn dimension(&self) -> u32 {
        self.inner.dimension()
    }

    /// The distance metric used by this collection (e.g. "Cosine").
    #[getter]
    fn distance_type(&self) -> String {
        format!("{:?}", self.inner.distance_type())
    }

    /// Return the number of vectors in this collection.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Insert a single vector with an ID and optional metadata dict.
    #[pyo3(signature = (id, vector, metadata=None))]
    fn insert(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<()> {
        let meta = match &metadata {
            Some(obj) => dict_to_value(py, obj)?,
            None => Value::Object(Default::default()),
        };
        let inner = self.inner.clone();
        let id = id.to_string();
        py.allow_threads(move || inner.insert(&id, &vector, meta).map_err(to_py_err))
    }

    /// Insert multiple vectors in a batch.
    ///
    /// Each item is a tuple of ``(id, vector, metadata)`` where *metadata* may
    /// be ``None`` or a dict.
    #[pyo3(signature = (items))]
    fn insert_batch(
        &self,
        py: Python<'_>,
        items: Vec<(String, Vec<f32>, Option<Bound<'_, pyo3::types::PyAny>>)>,
    ) -> PyResult<()> {
        let mut converted: Vec<(String, Vec<f32>, Value)> = Vec::with_capacity(items.len());
        for (id, vector, metadata) in &items {
            let meta = match metadata {
                Some(obj) => dict_to_value(py, obj)?,
                None => Value::Object(Default::default()),
            };
            converted.push((id.clone(), vector.clone(), meta));
        }
        let inner = self.inner.clone();
        py.allow_threads(move || {
            let batch: Vec<(&str, &[f32], Value)> = converted
                .iter()
                .map(|(id, vec, meta)| (id.as_str(), vec.as_slice(), meta.clone()))
                .collect();
            inner.insert_batch(&batch).map_err(to_py_err)
        })
    }

    /// Search for the *k* nearest vectors.
    ///
    /// An optional *filter* dict may be supplied to restrict results. Supports
    /// operators ``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``, ``$lte``,
    /// ``$in``, ``$and``, and ``$or``.
    #[pyo3(signature = (vector, k=10, filter=None))]
    fn search(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        k: usize,
        filter: Option<Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<Vec<PySearchResult>> {
        let f = match &filter {
            Some(obj) => dict_to_filter(py, obj)?,
            None => None,
        };

        let inner = self.inner.clone();
        let results = py.allow_threads(move || {
            let mut search = inner.search(&vector, k);
            if let Some(f) = f {
                search = search.filter(f);
            }
            search.execute().map_err(to_py_err)
        })?;

        Ok(results
            .into_iter()
            .map(|r| PySearchResult {
                id: r.id,
                distance: r.distance,
                metadata_val: r.metadata,
            })
            .collect())
    }

    /// Full-text keyword search across metadata string fields.
    ///
    /// Returns results ranked by BM25 relevance (highest first).
    #[pyo3(signature = (query, limit=10))]
    fn text_search(&self, query: &str, limit: usize) -> PyResult<Vec<PySearchResult>> {
        let results = self.inner.text_search(query, limit);
        Ok(results
            .into_iter()
            .map(|r| PySearchResult {
                id: r.id,
                distance: r.distance,
                metadata_val: r.metadata,
            })
            .collect())
    }

    /// Combined vector + keyword search using Reciprocal Rank Fusion.
    ///
    /// Merges nearest-neighbour and BM25 results into a single ranked list.
    #[pyo3(signature = (vector, text_query, k=10))]
    fn hybrid_search(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        text_query: &str,
        k: usize,
    ) -> PyResult<Vec<PySearchResult>> {
        let inner = self.inner.clone();
        let text_query = text_query.to_string();
        let results = py.allow_threads(move || {
            inner
                .hybrid_search(&vector, &text_query, k)
                .map_err(to_py_err)
        })?;
        Ok(results
            .into_iter()
            .map(|r| PySearchResult {
                id: r.id,
                distance: r.distance,
                metadata_val: r.metadata,
            })
            .collect())
    }

    /// Retrieve a vector record by ID, or ``None`` if not found.
    fn get(&self, py: Python<'_>, id: &str) -> PyResult<Option<PyObject>> {
        let inner = self.inner.clone();
        let id = id.to_string();
        let record = py.allow_threads(move || inner.get(&id).map_err(to_py_err))?;

        match record {
            Some(r) => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("id", &r.id)?;
                dict.set_item("vector", r.vector)?;
                dict.set_item("metadata", value_to_py(py, &r.metadata)?)?;
                Ok(Some(dict.into_pyobject(py)?.into_any().unbind()))
            }
            None => Ok(None),
        }
    }

    /// Delete a vector by ID. Returns ``True`` if it existed.
    fn delete(&self, py: Python<'_>, id: &str) -> PyResult<bool> {
        let inner = self.inner.clone();
        let id = id.to_string();
        py.allow_threads(move || inner.delete(&id).map_err(to_py_err))
    }

    /// Update the metadata dict for an existing vector.
    fn update_metadata(
        &self,
        py: Python<'_>,
        id: &str,
        metadata: Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<()> {
        let meta = dict_to_value(py, &metadata)?;
        let inner = self.inner.clone();
        let id = id.to_string();
        py.allow_threads(move || inner.update_metadata(&id, meta).map_err(to_py_err))
    }

    /// Create a secondary index on a metadata field for faster filtered search.
    fn create_index(&self, field: &str) {
        self.inner.create_index(field);
    }

    /// Drop a secondary index on a metadata field.
    fn drop_index(&self, field: &str) {
        self.inner.drop_index(field);
    }

    /// List the currently indexed metadata fields.
    fn indexed_fields(&self) -> Vec<String> {
        self.inner.indexed_fields()
    }

    fn __repr__(&self) -> String {
        format!(
            "Collection(name='{}', dimension={}, len={})",
            self.inner.name(),
            self.inner.dimension(),
            self.inner.len()
        )
    }
}

// ─────────────────────── Database ───────────────────────

/// A LiteVec database that holds one or more collections.
#[pyclass(name = "PyDatabase")]
#[derive(Clone)]
pub struct PyDatabase {
    inner: Database,
}

#[pymethods]
impl PyDatabase {
    /// Open (or create) a file-backed database at *path*.
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let db = Database::open(path).map_err(to_py_err)?;
        Ok(PyDatabase { inner: db })
    }

    /// Open an ephemeral in-memory database.
    #[staticmethod]
    fn open_memory() -> PyResult<Self> {
        let db = Database::open_memory().map_err(to_py_err)?;
        Ok(PyDatabase { inner: db })
    }

    /// Create a new collection with the given name and vector dimension.
    #[pyo3(signature = (name, dimension))]
    fn create_collection(&self, name: &str, dimension: u32) -> PyResult<PyCollection> {
        let col = self
            .inner
            .create_collection(name, dimension)
            .map_err(to_py_err)?;
        Ok(PyCollection { inner: col })
    }

    /// Get an existing collection by name, or ``None``.
    fn get_collection(&self, name: &str) -> Option<PyCollection> {
        self.inner
            .get_collection(name)
            .map(|c| PyCollection { inner: c })
    }

    /// Delete a collection by name.
    fn delete_collection(&self, name: &str) -> PyResult<()> {
        self.inner.delete_collection(name).map_err(to_py_err)
    }

    /// List the names of all collections in this database.
    fn list_collections(&self) -> Vec<String> {
        self.inner.list_collections()
    }

    /// Force a checkpoint, persisting all data to disk.
    fn checkpoint(&self) -> PyResult<()> {
        self.inner.checkpoint().map_err(to_py_err)
    }

    /// Create a backup snapshot at the given path.
    fn create_backup(&self, path: &str) -> PyResult<()> {
        self.inner.create_backup(path).map_err(to_py_err)
    }

    /// Restore a database from a backup snapshot.
    #[staticmethod]
    fn restore_from_backup(path: &str) -> PyResult<Self> {
        let db = Database::restore_from_backup(path).map_err(to_py_err)?;
        Ok(PyDatabase { inner: db })
    }

    /// Return metadata about a backup file (number of collections, vectors, etc.).
    #[staticmethod]
    fn backup_info(path: &str) -> PyResult<PyObject> {
        let info = Database::backup_info(path).map_err(to_py_err)?;
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("version", info.version)?;
            dict.set_item("num_collections", info.num_collections)?;
            dict.set_item("total_vectors", info.total_vectors)?;
            let cols = pyo3::types::PyList::empty(py);
            for c in &info.collections {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("name", &c.name)?;
                d.set_item("dimension", c.dimension)?;
                d.set_item("num_vectors", c.num_vectors)?;
                cols.append(d)?;
            }
            dict.set_item("collections", cols)?;
            Ok(dict.into_pyobject(py)?.into_any().unbind())
        })
    }

    fn __repr__(&self) -> String {
        let count = self.inner.list_collections().len();
        format!("Database(collections={count})")
    }
}

// ─────────────────────── Module ───────────────────────

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PySearchResult>()?;
    Ok(())
}
