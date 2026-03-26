# Changelog

All notable changes to LiteVec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

### Added
- **Core engine** — embedded vector database with single-file `.lv` storage
- **Index types** — HNSW, Flat (brute-force), DiskANN (Vamana graph), Product Quantization
- **Distance functions** — Cosine, Euclidean (L2), Dot Product with SIMD acceleration (AVX2 + NEON)
- **Metadata** — JSON metadata on every vector, 11 filter operators (Eq, Ne, Gt, Gte, Lt, Lte, In, Exists, And, Or, Not)
- **Full-text search** — BM25 ranking on metadata text fields
- **Hybrid search** — combined vector + keyword search with RRF and WeightedSum fusion
- **Secondary indexes** — B-tree indexes on metadata fields for fast filtered queries
- **WAL** — write-ahead log with CRC32 checksums for crash safety
- **Backup & restore** — `create_backup()`, `restore_from_backup()`, `backup_info()`
- **Thread safety** — `Arc<RwLock>` for concurrent read/write access
- **Python bindings** — PyO3 with GIL release for long operations, numpy support
- **Node.js bindings** — NAPI-RS with TypeScript type definitions
- **C FFI** — 15 functions with `litevec.h` header for any language with C interop
- **WebAssembly** — `wasm-bindgen` build for browser and edge use
- **MCP server** — Model Context Protocol server with 11 tools for AI agent integration
- **CLI** — `litevec` command-line tool (create, insert, search, get, delete, list, info, serve, bench)
- **Benchmarks** — Criterion benchmarks for distance, search, and insert operations
- **Test suite** — 182 tests: 150 unit, 24 integration (lifecycle, crash recovery, concurrent access, large dataset recall, all filter operators, full-text/hybrid search, backup/restore), 8 property-based tests (proptest)
- **SECURITY.md** — vulnerability reporting policy
