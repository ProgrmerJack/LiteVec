# Architecture

LiteVec is designed as a zero-dependency embedded vector database with a layered architecture similar to SQLite.

## Crate Structure

```
litevec/
├── litevec-core      # Core engine: indexes, storage, collections
├── litevec           # Public API re-exports
├── litevec-cli       # Command-line interface
├── litevec-ffi       # C foreign function interface
├── litevec-mcp       # Model Context Protocol server
├── litevec-wasm      # WebAssembly bindings
├── bindings/python   # Python (PyO3) bindings
└── bindings/node     # Node.js (NAPI-RS) bindings
```

## Core Engine Layers

```
┌─────────────────────────────────┐
│          Public API             │  Database, Collection, SearchQuery
├─────────────────────────────────┤
│       Collection Layer          │  Thread-safe CRUD, search orchestration
├──────────┬──────────┬───────────┤
│  Index   │ Metadata │ Full-text │  HNSW, Flat, PQ, DiskANN │ Filters │ BM25
├──────────┴──────────┴───────────┤
│       Vector Store              │  In-memory vector storage
├─────────────────────────────────┤
│     Persistence Layer           │  JSON snapshots, backup/restore
├─────────────────────────────────┤
│      Storage Backend            │  Memory-mapped files, WAL, pages
└─────────────────────────────────┘
```

## Index Algorithms

### HNSW (Hierarchical Navigable Small World)
The default index. Builds a multi-layer graph where upper layers are sparse (for fast navigation) and lower layers are dense (for accurate search).

**Parameters:**
- `m` (default 16) — Max connections per node per layer. Higher = better recall, more memory.
- `ef_construction` (default 200) — Search width during build. Higher = better graph quality, slower insert.
- `ef_search` (default 100) — Search width during query. Higher = better recall, slower search.

**Complexity:**
- Insert: O(log n) amortized
- Search: O(log n)
- Memory: O(n × m)

### Flat Index
Brute-force exact search. Compares the query against every vector.

**Complexity:**
- Insert: O(1)
- Search: O(n)
- Memory: O(n × d)

### Product Quantization (PQ)
Compresses vectors by dividing them into sub-vectors and clustering each. Reduces memory by 32× for 128-dim vectors (128 floats → 16 bytes).

### DiskANN (Vamana Graph)
Disk-optimized graph index for billion-scale datasets. Uses a Vamana graph with robust pruning to maintain high recall even when data exceeds memory.

## SIMD Acceleration

Distance computations are SIMD-accelerated on supported platforms:
- **x86_64**: AVX2 (256-bit, 8 floats per instruction) with runtime detection
- **aarch64**: NEON (128-bit, 4 floats per instruction), always available

Falls back to scalar code on unsupported platforms.

## Thread Safety

All public types (`Database`, `Collection`) are `Send + Sync`. Internally, collections use `parking_lot::RwLock` for concurrent read access with exclusive writes.

## Persistence Model

LiteVec uses a dual-file approach:
1. **Main database file** (`.lv`) — Memory-mapped storage with page management
2. **Snapshot file** (`.lv.snap`) — JSON serialization of all collection data

On close, collections are serialized to the snapshot file. On open, they are deserialized and indexes are rebuilt from vector data. This approach prioritizes simplicity and correctness over startup speed for large datasets.

## WAL (Write-Ahead Log)

The WAL provides crash safety. All mutations are first written to the WAL (with CRC32 checksums), then applied to the main storage. On recovery, uncommitted WAL entries are replayed.
