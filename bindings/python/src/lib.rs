//! PyO3 bindings for LiteVec.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

use litevec_core::{Collection, Database, Filter};

fn to_py_err(e: litevec_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn dict_to_value(py: Python<'_>, obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Value> {
    // Convert Python object to serde_json::Value via JSON string
    let json_mod = py.import("json")?;
    let json_str: String = json_mod.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&json_str).map_err(|e| PyValueError::new_err(format!("Invalid JSON: {e}")))
}

fn value_to_py(py: Python<'_>, val: &Value) -> PyResult<PyObject> {
    let json_mod = py.import("json")?;
    let json_str = serde_json::to_string(val)
        .map_err(|e| PyValueError::new_err(format!("JSON error: {e}")))?;
    let result = json_mod.call_method1("loads", (json_str,))?;
    Ok(result.into_pyobject(py)?.into_any().unbind())
}

fn dict_to_filter(py: Python<'_>, obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Option<Filter>> {
    if obj.is_none() {
        return Ok(None);
    }
    let val = dict_to_value(py, obj)?;
    match &val {
        Value::Object(map) => {
            let filters: Vec<Filter> = map
                .iter()
                .map(|(k, v)| Filter::Eq(k.clone(), v.clone()))
                .collect();
            match filters.len() {
                0 => Ok(None),
                1 => Ok(Some(filters.into_iter().next().unwrap())),
                _ => Ok(Some(Filter::And(filters))),
            }
        }
        _ => Err(PyValueError::new_err("filter must be a dict")),
    }
}

// ─────────────────────── SearchResult ───────────────────────

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

#[pyclass(name = "PyCollection")]
#[derive(Clone)]
pub struct PyCollection {
    inner: Collection,
}

#[pymethods]
impl PyCollection {
    #[getter]
    fn name(&self) -> String {
        self.inner.name()
    }

    #[getter]
    fn dimension(&self) -> u32 {
        self.inner.dimension()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[pyo3(signature = (id, vector, metadata=None))]
    fn insert(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<()> {
        let meta = match metadata {
            Some(ref obj) => dict_to_value(py, obj)?,
            None => Value::Object(Default::default()),
        };
        let inner = self.inner.clone();
        let vector = vector.clone();
        let id = id.to_string();
        py.allow_threads(move || inner.insert(&id, &vector, meta).map_err(to_py_err))
    }

    #[pyo3(signature = (vector, k=10, filter=None))]
    fn search(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        k: usize,
        filter: Option<Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<Vec<PySearchResult>> {
        let f = match filter {
            Some(ref obj) => dict_to_filter(py, obj)?,
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

    fn delete(&self, py: Python<'_>, id: &str) -> PyResult<bool> {
        let inner = self.inner.clone();
        let id = id.to_string();
        py.allow_threads(move || inner.delete(&id).map_err(to_py_err))
    }

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

#[pyclass(name = "PyDatabase")]
#[derive(Clone)]
pub struct PyDatabase {
    inner: Database,
}

#[pymethods]
impl PyDatabase {
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let db = Database::open(path).map_err(to_py_err)?;
        Ok(PyDatabase { inner: db })
    }

    #[staticmethod]
    fn open_memory() -> PyResult<Self> {
        let db = Database::open_memory().map_err(to_py_err)?;
        Ok(PyDatabase { inner: db })
    }

    #[pyo3(signature = (name, dimension))]
    fn create_collection(&self, name: &str, dimension: u32) -> PyResult<PyCollection> {
        let col = self
            .inner
            .create_collection(name, dimension)
            .map_err(to_py_err)?;
        Ok(PyCollection { inner: col })
    }

    fn get_collection(&self, name: &str) -> Option<PyCollection> {
        self.inner
            .get_collection(name)
            .map(|c| PyCollection { inner: c })
    }

    fn delete_collection(&self, name: &str) -> PyResult<()> {
        self.inner.delete_collection(name).map_err(to_py_err)
    }

    fn list_collections(&self) -> Vec<String> {
        self.inner.list_collections()
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
