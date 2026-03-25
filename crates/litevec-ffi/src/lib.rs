//! C FFI bindings for LiteVec.
#![allow(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::ptr;
use std::slice;

use litevec_core::{Collection, Database};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Opaque wrapper types
// ---------------------------------------------------------------------------

/// Opaque handle to a LiteVec database (wraps `litevec_core::Database`).
pub struct LiteVecDb {
    inner: Database,
}

/// Opaque handle to a LiteVec collection (wraps `litevec_core::Collection`).
pub struct LiteVecCollection {
    inner: Collection,
}

/// A single search result, exposed across the FFI boundary.
#[repr(C)]
pub struct LiteVecSearchResult {
    pub id: *mut c_char,
    pub distance: f32,
    pub metadata_json: *mut c_char,
}

// ---------------------------------------------------------------------------
// Thread-local error storage
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a `*const c_char` to `&str`. Returns `None` on null or invalid UTF-8.
///
/// # Safety
/// If non-null, `ptr` must point to a valid null-terminated C string.
unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

/// Leak a Rust `String` into a `*mut c_char` for the C caller to own.
fn string_to_c(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

// ---------------------------------------------------------------------------
// Open / Close
// ---------------------------------------------------------------------------

/// Open (or create) a database at `path`. Returns a heap-allocated handle, or
/// `NULL` on error (check `litevec_last_error`).
///
/// # Safety
/// `path` must be a valid null-terminated C string or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_open(path: *const c_char) -> *mut LiteVecDb {
    clear_last_error();
    let Some(p) = cstr_to_str(path) else {
        set_last_error("null or invalid path");
        return ptr::null_mut();
    };
    match Database::open(p) {
        Ok(db) => Box::into_raw(Box::new(LiteVecDb { inner: db })),
        Err(e) => {
            set_last_error(&e.to_string());
            ptr::null_mut()
        }
    }
}

/// Open an in-memory database. Returns `NULL` on error.
///
/// # Safety
/// No special requirements. Always safe to call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_open_memory() -> *mut LiteVecDb {
    clear_last_error();
    match Database::open_memory() {
        Ok(db) => Box::into_raw(Box::new(LiteVecDb { inner: db })),
        Err(e) => {
            set_last_error(&e.to_string());
            ptr::null_mut()
        }
    }
}

/// Close a database handle and free its memory. Passing `NULL` is a no-op.
///
/// # Safety
/// `db` must be a valid pointer from `litevec_open`/`litevec_open_memory`, or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_close(db: *mut LiteVecDb) {
    if db.is_null() {
        return;
    }
    let boxed = Box::from_raw(db);
    if let Err(e) = boxed.inner.close() {
        set_last_error(&e.to_string());
    }
}

// ---------------------------------------------------------------------------
// Collections
// ---------------------------------------------------------------------------

/// Create a new collection. Returns a heap-allocated handle, or `NULL` on error.
///
/// # Safety
/// `db` must be a valid `LiteVecDb` pointer. `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_create_collection(
    db: *mut LiteVecDb,
    name: *const c_char,
    dimension: u32,
) -> *mut LiteVecCollection {
    clear_last_error();
    if db.is_null() {
        set_last_error("null db pointer");
        return ptr::null_mut();
    }
    let Some(n) = cstr_to_str(name) else {
        set_last_error("null or invalid collection name");
        return ptr::null_mut();
    };
    match (*db).inner.create_collection(n, dimension) {
        Ok(col) => Box::into_raw(Box::new(LiteVecCollection { inner: col })),
        Err(e) => {
            set_last_error(&e.to_string());
            ptr::null_mut()
        }
    }
}

/// Get an existing collection by name. Returns `NULL` if not found.
///
/// # Safety
/// `db` must be a valid `LiteVecDb` pointer. `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_get_collection(
    db: *mut LiteVecDb,
    name: *const c_char,
) -> *mut LiteVecCollection {
    clear_last_error();
    if db.is_null() {
        set_last_error("null db pointer");
        return ptr::null_mut();
    }
    let Some(n) = cstr_to_str(name) else {
        set_last_error("null or invalid collection name");
        return ptr::null_mut();
    };
    match (*db).inner.get_collection(n) {
        Some(col) => Box::into_raw(Box::new(LiteVecCollection { inner: col })),
        None => {
            set_last_error("collection not found");
            ptr::null_mut()
        }
    }
}

/// Delete a collection. Returns 0 on success, -1 on error.
///
/// # Safety
/// `db` must be a valid `LiteVecDb` pointer. `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_delete_collection(db: *mut LiteVecDb, name: *const c_char) -> i32 {
    clear_last_error();
    if db.is_null() {
        set_last_error("null db pointer");
        return -1;
    }
    let Some(n) = cstr_to_str(name) else {
        set_last_error("null or invalid collection name");
        return -1;
    };
    match (*db).inner.delete_collection(n) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Free a collection handle. Passing `NULL` is a no-op.
///
/// # Safety
/// `col` must be a valid pointer from `litevec_create_collection`/`litevec_get_collection`, or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_free_collection(col: *mut LiteVecCollection) {
    if !col.is_null() {
        drop(Box::from_raw(col));
    }
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

/// Insert (or upsert) a vector into a collection.
///
/// * `id`            – null-terminated vector id
/// * `vector`        – pointer to `dim` floats
/// * `dim`           – number of dimensions
/// * `metadata_json` – optional JSON string (may be `NULL` for empty metadata)
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// `col` must be a valid `LiteVecCollection` pointer. `vector` must point to at least `dim` floats.
/// `id` must be a valid C string. `metadata_json`, if non-NULL, must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_insert(
    col: *mut LiteVecCollection,
    id: *const c_char,
    vector: *const f32,
    dim: u32,
    metadata_json: *const c_char,
) -> i32 {
    clear_last_error();
    if col.is_null() || vector.is_null() {
        set_last_error("null pointer argument");
        return -1;
    }
    let Some(id_str) = cstr_to_str(id) else {
        set_last_error("null or invalid id");
        return -1;
    };
    let vec_slice = slice::from_raw_parts(vector, dim as usize);

    let metadata: Value = if metadata_json.is_null() {
        Value::Object(serde_json::Map::new())
    } else {
        match cstr_to_str(metadata_json) {
            Some(s) if !s.is_empty() => match serde_json::from_str(s) {
                Ok(v) => v,
                Err(e) => {
                    set_last_error(&format!("invalid metadata JSON: {e}"));
                    return -1;
                }
            },
            _ => Value::Object(serde_json::Map::new()),
        }
    };

    match (*col).inner.insert(id_str, vec_slice, metadata) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Search for the `k` nearest neighbours.
///
/// On success the function writes a heap-allocated array of `LiteVecSearchResult`
/// into `*results` and the count into `*result_count`. The caller must free them
/// with `litevec_free_results`.
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// `col` must be valid. `query` must point to at least `dim` floats.
/// `results` and `result_count` must be valid writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_search(
    col: *mut LiteVecCollection,
    query: *const f32,
    dim: u32,
    k: u32,
    results: *mut *mut LiteVecSearchResult,
    result_count: *mut u32,
) -> i32 {
    clear_last_error();
    if col.is_null() || query.is_null() || results.is_null() || result_count.is_null() {
        set_last_error("null pointer argument");
        return -1;
    }
    let query_slice = slice::from_raw_parts(query, dim as usize);

    let search_results = match (*col).inner.search(query_slice, k as usize).execute() {
        Ok(r) => r,
        Err(e) => {
            set_last_error(&e.to_string());
            return -1;
        }
    };

    let count = search_results.len();
    if count == 0 {
        *results = ptr::null_mut();
        *result_count = 0;
        return 0;
    }

    let mut out: Vec<LiteVecSearchResult> = Vec::with_capacity(count);
    for sr in search_results {
        let meta_str = serde_json::to_string(&sr.metadata).unwrap_or_default();
        out.push(LiteVecSearchResult {
            id: string_to_c(sr.id),
            distance: sr.distance,
            metadata_json: string_to_c(meta_str),
        });
    }

    let mut boxed = out.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);

    *results = ptr;
    *result_count = count as u32;
    0
}

/// Retrieve a vector by id.
///
/// On success, `*vector` receives a heap-allocated float array, `*dim` receives
/// its length, and `*metadata_json` receives a heap-allocated JSON string.
/// The caller must free them with `litevec_free_vector` / `litevec_free_string`.
///
/// Returns 0 on success, -1 on error (including "not found").
///
/// # Safety
/// `col` must be valid. `id` must be a valid C string. `vector`, `dim`, and
/// `metadata_json` must be valid writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_get(
    col: *mut LiteVecCollection,
    id: *const c_char,
    vector: *mut *mut f32,
    dim: *mut u32,
    metadata_json: *mut *mut c_char,
) -> i32 {
    clear_last_error();
    if col.is_null() || vector.is_null() || dim.is_null() || metadata_json.is_null() {
        set_last_error("null pointer argument");
        return -1;
    }
    let Some(id_str) = cstr_to_str(id) else {
        set_last_error("null or invalid id");
        return -1;
    };

    match (*col).inner.get(id_str) {
        Ok(Some(record)) => {
            let len = record.vector.len();
            let mut boxed = record.vector.into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            std::mem::forget(boxed);
            register_vector(ptr, len);
            *vector = ptr;
            *dim = len as u32;

            let meta_str = serde_json::to_string(&record.metadata).unwrap_or_default();
            *metadata_json = string_to_c(meta_str);
            0
        }
        Ok(None) => {
            set_last_error("vector not found");
            -1
        }
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Delete a vector by id. Returns 0 on success, -1 on error.
///
/// # Safety
/// `col` must be a valid `LiteVecCollection` pointer. `id` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_delete(col: *mut LiteVecCollection, id: *const c_char) -> i32 {
    clear_last_error();
    if col.is_null() {
        set_last_error("null collection pointer");
        return -1;
    }
    let Some(id_str) = cstr_to_str(id) else {
        set_last_error("null or invalid id");
        return -1;
    };
    match (*col).inner.delete(id_str) {
        Ok(_) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

/// Free an array of `LiteVecSearchResult` returned by `litevec_search`.
///
/// # Safety
/// `results` must be a pointer returned by `litevec_search`, or NULL.
/// `count` must match the `result_count` from that call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_free_results(results: *mut LiteVecSearchResult, count: u32) {
    if results.is_null() {
        return;
    }
    let slice = slice::from_raw_parts_mut(results, count as usize);
    for item in slice.iter_mut() {
        if !item.id.is_null() {
            drop(CString::from_raw(item.id));
            item.id = ptr::null_mut();
        }
        if !item.metadata_json.is_null() {
            drop(CString::from_raw(item.metadata_json));
            item.metadata_json = ptr::null_mut();
        }
    }
    // Reconstruct the boxed slice so it is properly deallocated.
    let _ = Box::from_raw(slice as *mut [LiteVecSearchResult]);
}

/// Free a C string returned by the library (e.g. metadata_json from `litevec_get`).
///
/// # Safety
/// `s` must be a pointer returned by this library, or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Free a float vector returned by `litevec_get`.
///
/// # Safety
/// `v` must be a pointer returned by `litevec_get`, or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn litevec_free_vector(v: *mut f32) {
    if !v.is_null() {
        // Look up the original length so we can properly reconstruct the Vec.
        if let Some(len) = VECTOR_LENGTHS.with(|m| m.borrow_mut().remove(&(v as usize))) {
            let _ = Vec::from_raw_parts(v, len, len);
        }
    }
}

thread_local! {
#[allow(clippy::missing_const_for_thread_local)]
    static VECTOR_LENGTHS: RefCell<std::collections::HashMap<usize, usize>> =
        RefCell::new(std::collections::HashMap::new());
}

/// Internal helper: register the length of a vector we're about to hand out.
fn register_vector(ptr: *mut f32, len: usize) {
    VECTOR_LENGTHS.with(|m| {
        m.borrow_mut().insert(ptr as usize, len);
    });
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

/// Return the last error message for the current thread, or `NULL` if none.
/// The returned pointer is valid until the next FFI call on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn litevec_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match e.borrow().as_ref() {
        Some(cs) => cs.as_ptr(),
        None => ptr::null(),
    })
}
