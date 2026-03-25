# C FFI Reference

LiteVec provides a C-compatible foreign function interface for integration with C, C++, Go, Zig, and any language with C FFI support.

## Header File

Include `litevec.h` from `crates/litevec-ffi/include/litevec.h`.

## Building

```bash
cargo build --release -p litevec-ffi

# Produces:
#   target/release/liblitevec_ffi.so    (Linux)
#   target/release/liblitevec_ffi.dylib (macOS)
#   target/release/litevec_ffi.dll      (Windows)
```

## API

### Database Management

```c
// Open a file-backed database
LiteVecDb* litevec_db_open(const char* path);

// Open an in-memory database
LiteVecDb* litevec_db_open_memory(void);

// Close and free database
void litevec_db_close(LiteVecDb* db);

// Create a collection
LiteVecCollection* litevec_create_collection(LiteVecDb* db, const char* name, uint32_t dimension);

// Get an existing collection
LiteVecCollection* litevec_get_collection(LiteVecDb* db, const char* name);

// Delete a collection
int litevec_delete_collection(LiteVecDb* db, const char* name);
// Returns 0 on success, -1 on error
```

### Vector Operations

```c
// Insert a vector with JSON metadata
int litevec_insert(LiteVecCollection* col, const char* id,
                   const float* vector, uint32_t dimension,
                   const char* metadata_json);
// Returns 0 on success, -1 on error

// Delete a vector
int litevec_delete(LiteVecCollection* col, const char* id);

// Search for k nearest neighbors
LiteVecSearchResults* litevec_search(LiteVecCollection* col,
                                      const float* query, uint32_t dimension,
                                      uint32_t k);
```

### Search Results

```c
typedef struct {
    char* id;
    float distance;
    char* metadata_json;
} LiteVecSearchResult;

typedef struct {
    LiteVecSearchResult* results;
    uint32_t count;
} LiteVecSearchResults;

// Free search results
void litevec_free_search_results(LiteVecSearchResults* results);
```

### Error Handling

```c
// Get the last error message (thread-local)
const char* litevec_last_error(void);
```

## Example (C)

```c
#include "litevec.h"
#include <stdio.h>

int main() {
    LiteVecDb* db = litevec_db_open_memory();
    if (!db) {
        printf("Error: %s\n", litevec_last_error());
        return 1;
    }

    LiteVecCollection* col = litevec_create_collection(db, "docs", 3);
    
    float v1[] = {1.0f, 0.0f, 0.0f};
    litevec_insert(col, "doc1", v1, 3, "{\"title\": \"Hello\"}");

    float v2[] = {0.0f, 1.0f, 0.0f};
    litevec_insert(col, "doc2", v2, 3, "{\"title\": \"World\"}");

    float query[] = {0.9f, 0.1f, 0.0f};
    LiteVecSearchResults* results = litevec_search(col, query, 3, 2);

    for (uint32_t i = 0; i < results->count; i++) {
        printf("%s: distance=%.4f\n",
               results->results[i].id,
               results->results[i].distance);
    }

    litevec_free_search_results(results);
    litevec_db_close(db);
    return 0;
}
```

## Memory Management

- `LiteVecDb*` — freed by `litevec_db_close()`
- `LiteVecCollection*` — lifetime tied to database, do NOT free manually
- `LiteVecSearchResults*` — freed by `litevec_free_search_results()`
- Error strings — static/thread-local, do NOT free
