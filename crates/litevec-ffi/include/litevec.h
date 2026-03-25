/**
 * @file litevec.h
 * @brief C API for LiteVec embedded vector database.
 *
 * All functions that return an `int` use the convention:
 *   0  = success
 *  -1  = error (call `litevec_last_error()` for details)
 *
 * Pointer-returning functions return NULL on error.
 */

#ifndef LITEVEC_H
#define LITEVEC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles --------------------------------------------------------- */

typedef struct LiteVecDb LiteVecDb;
typedef struct LiteVecCollection LiteVecCollection;

/* Search result ---------------------------------------------------------- */

typedef struct {
    char  *id;
    float  distance;
    char  *metadata_json;
} LiteVecSearchResult;

/* Open / Close ----------------------------------------------------------- */

/**
 * Open (or create) a database at the given file path.
 * @param path  Null-terminated UTF-8 file path.
 * @return      Database handle, or NULL on error.
 */
LiteVecDb *litevec_open(const char *path);

/**
 * Open an in-memory database (no persistence).
 * @return  Database handle, or NULL on error.
 */
LiteVecDb *litevec_open_memory(void);

/**
 * Close a database and free its resources.
 * Passing NULL is a safe no-op.
 */
void litevec_close(LiteVecDb *db);

/* Collections ------------------------------------------------------------ */

/**
 * Create a new collection with the given vector dimension.
 * @return  Collection handle, or NULL on error.
 */
LiteVecCollection *litevec_create_collection(LiteVecDb *db,
                                             const char *name,
                                             uint32_t dimension);

/**
 * Get an existing collection by name.
 * @return  Collection handle, or NULL if not found.
 */
LiteVecCollection *litevec_get_collection(LiteVecDb *db, const char *name);

/**
 * Delete a collection and all its data.
 * @return 0 on success, -1 on error.
 */
int litevec_delete_collection(LiteVecDb *db, const char *name);

/**
 * Free a collection handle. Passing NULL is a safe no-op.
 */
void litevec_free_collection(LiteVecCollection *col);

/* CRUD ------------------------------------------------------------------- */

/**
 * Insert (or upsert) a vector.
 *
 * @param col            Collection handle.
 * @param id             Null-terminated vector ID.
 * @param vector         Pointer to `dim` floats.
 * @param dim            Number of dimensions.
 * @param metadata_json  Optional JSON string (may be NULL).
 * @return 0 on success, -1 on error.
 */
int litevec_insert(LiteVecCollection *col,
                   const char *id,
                   const float *vector,
                   uint32_t dim,
                   const char *metadata_json);

/**
 * Search for the k nearest neighbours.
 *
 * On success, `*results` receives a heap-allocated array of
 * LiteVecSearchResult and `*result_count` receives its length.
 * Free with `litevec_free_results()`.
 *
 * @return 0 on success, -1 on error.
 */
int litevec_search(LiteVecCollection *col,
                   const float *query,
                   uint32_t dim,
                   uint32_t k,
                   LiteVecSearchResult **results,
                   uint32_t *result_count);

/**
 * Retrieve a vector by ID.
 *
 * On success, `*vector` receives a heap-allocated float array (length
 * written to `*dim`) and `*metadata_json` receives a heap-allocated JSON
 * string. Free them with `litevec_free_vector()` / `litevec_free_string()`.
 *
 * @return 0 on success, -1 on error (including "not found").
 */
int litevec_get(LiteVecCollection *col,
                const char *id,
                float **vector,
                uint32_t *dim,
                char **metadata_json);

/**
 * Delete a vector by ID.
 * @return 0 on success, -1 on error.
 */
int litevec_delete(LiteVecCollection *col, const char *id);

/* Memory management ------------------------------------------------------ */

/**
 * Free search results returned by `litevec_search`.
 */
void litevec_free_results(LiteVecSearchResult *results, uint32_t count);

/**
 * Free a C string returned by the library.
 */
void litevec_free_string(char *s);

/**
 * Free a float vector returned by `litevec_get`.
 */
void litevec_free_vector(float *v);

/* Error handling --------------------------------------------------------- */

/**
 * Return the last error message for the calling thread, or NULL if none.
 * The pointer is valid until the next litevec_* call on the same thread.
 */
const char *litevec_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* LITEVEC_H */
