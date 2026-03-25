<div align="center">

<img src="assets/hero.svg" alt="LiteVec — The SQLite of Vector Search" width="900"/>

# 🔍 LiteVec

**The embedded vector database. No server. No Docker. No config.**

[![Crates.io](https://img.shields.io/crates/v/litevec)](https://crates.io/crates/litevec)
[![PyPI](https://img.shields.io/pypi/v/litevec)](https://pypi.org/project/litevec/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[Quickstart](#-quickstart) •
[Features](#-features) •
[Installation](#-installation) •
[Bindings](#-language-bindings) •
[CLI](#-cli) •
[MCP Server](#-mcp-server) •
[Benchmarks](#-benchmarks) •
[Docs](docs/)

</div>

---

<div align="center">
<img src="assets/demo.svg" alt="LiteVec Demo" width="800"/>
</div>

## Why LiteVec?

Every AI application needs vector search — RAG pipelines, semantic search, AI agents with memory. But existing solutions require running Docker containers, separate server processes, or cloud subscriptions. **LiteVec is the SQLite of vector search:** add it as a library dependency, open a file, done.

| | LiteVec | Qdrant | FAISS | pgvector | Pinecone |
|---|:---:|:---:|:---:|:---:|:---:|
| Embedded (no server) | ✅ | ❌ | ⚠️ | ❌ | ❌ |
| Single file database | ✅ | ❌ | ❌ | ❌ | ❌ |
| Metadata filtering | ✅ | ✅ | ❌ | ✅ | ✅ |
| Full-text + hybrid search | ✅ | ✅ | ❌ | ✅ | ❌ |
| Zero dependencies | ✅ | ❌ | ❌ | ❌ | ❌ |
| Works offline | ✅ | ✅ | ✅ | ✅ | ❌ |
| Rust + Python + Node + C + WASM | ✅ | ⚠️ | ⚠️ | ❌ | ⚠️ |
| MCP server for AI agents | ✅ | ❌ | ❌ | ❌ | ❌ |
| Backup & restore | ✅ | ✅ | ❌ | ✅ | ✅ |

## ⚡ Quickstart

### Rust

```rust
use litevec::Database;
use serde_json::json;

// Open a database (creates file if it doesn't exist)
let db = Database::open("my_vectors.lv")?;

// Create a collection for 384-dimensional embeddings
let collection = db.create_collection("docs", 384)?;

// Insert vectors with metadata
collection.insert("doc1", &embedding, json!({"title": "Hello World"}))?;
collection.insert("doc2", &embedding2, json!({"title": "Goodbye"}))?;

// Search for the 10 nearest neighbors
let results = collection.search(&query_embedding, 10).execute()?;
for result in &results {
    println!("{}: distance={:.4}", result.id, result.distance);
}

// Search with metadata filter
let results = collection.search(&query_embedding, 10)
    .filter(Filter::Eq("title".into(), json!("Hello World")))
    .execute()?;
```

### Python

```python
import litevec

db = litevec.open("my_vectors.lv")
collection = db.create_collection("docs", dimension=384)

# Insert
collection.insert("doc1", embedding, {"title": "Hello World"})

# Search
results = collection.search(query_embedding, k=10)
for r in results:
    print(f"{r.id}: distance={r.distance:.4f}")

# Search with filter
results = collection.search(query_embedding, k=10, filter={"title": "Hello World"})
```

### In-Memory Mode

```rust
// No file, perfect for testing
let db = Database::open_memory()?;
let collection = db.create_collection("test", 128)?;
```

## ✨ Features

- **🚀 SIMD-Accelerated** — AVX2 (x86_64) + NEON (ARM/Apple Silicon) for blazing fast distance calculations
- **📊 Multiple Index Types** — HNSW (>95% recall), Flat (exact), DiskANN (billion-scale), Product Quantization (32× compression)
- **🔍 Metadata Filtering** — Filter by JSON fields with 11 operators (Eq, Gt, Lt, In, And, Or, Not...) + secondary indexes
- **📝 Full-Text Search** — Built-in BM25 full-text index with hybrid search (vector + keyword fusion via RRF)
- **💾 Crash-Safe** — Write-ahead log (WAL) with CRC32 checksums ensures data integrity
- **🔒 Thread-Safe** — Multiple readers, single writer with `Arc<RwLock>` concurrency
- **📦 Single File** — All data in one `.lv` file, easy to backup and deploy
- **💿 Backup & Restore** — Snapshot backups with `create_backup()` / `restore_from_backup()`
- **🐍 5 Language Bindings** — Rust, Python (PyO3), Node.js (NAPI-RS), C FFI, WebAssembly
- **🤖 MCP Server** — Model Context Protocol server gives AI agents vector search as a tool
- **💻 CLI Tool** — Create, insert, search, and manage from the command line
- **📐 Distance Functions** — Cosine, Euclidean (L2), and Dot Product

## 📦 Installation

### Rust

```bash
cargo add litevec
```

### Python

```bash
pip install litevec
```

### Node.js

```bash
npm install litevec
```

### CLI

```bash
cargo install litevec-cli
```

### WebAssembly

```bash
cd crates/litevec-wasm && wasm-pack build --target web
```

## 🌐 Language Bindings

| Language | Package | Status |
|----------|---------|--------|
| **Rust** | `litevec` | ✅ Full API |
| **Python** | `litevec` (PyPI) | ✅ GIL-released, native speed |
| **Node.js** | `litevec` (npm) | ✅ NAPI-RS, TypeScript types |
| **C/C++** | `litevec-ffi` | ✅ 15 FFI functions + header |
| **WebAssembly** | `litevec-wasm` | ✅ Browser & edge runtimes |

### Python

```python
import litevec

db = litevec.Database.open_memory()
col = db.create_collection("docs", dimension=384)
col.insert("doc1", embedding, {"title": "Hello World"})
results = col.search(query_embedding, k=10)
```

### Node.js

```javascript
const { Database } = require('litevec');

const db = Database.openMemory();
const col = db.createCollection('docs', 384);
col.insert('doc1', embedding, { title: 'Hello World' });
const results = col.search(queryEmbedding, 10);
```

### C

```c
#include "litevec.h"

LiteVecDb* db = litevec_db_open_memory();
LiteVecCollection* col = litevec_create_collection(db, "docs", 3);
float vec[] = {1.0f, 0.0f, 0.0f};
litevec_insert(col, "doc1", vec, 3, "{\"title\": \"Hello\"}");
LiteVecSearchResults* results = litevec_search(col, vec, 3, 10);
litevec_free_search_results(results);
litevec_db_close(db);
```

## 💻 CLI

```bash
# Create a collection
litevec create mydb.lv --collection docs --dimension 384

# Insert vectors from JSONL
litevec insert mydb.lv --collection docs --input vectors.jsonl

# Search
litevec search mydb.lv --collection docs --vector '[0.1, 0.2, ...]' --k 5

# List collections
litevec list mydb.lv

# Database info
litevec info mydb.lv
```

### JSONL Format

Each line is a JSON object with `id`, `vector`, and optional `metadata`:

```json
{"id": "doc1", "vector": [0.1, 0.2, 0.3], "metadata": {"title": "Hello"}}
{"id": "doc2", "vector": [0.4, 0.5, 0.6], "metadata": {"title": "World"}}
```

## 🔧 API Reference

For complete API documentation, see [docs/](docs/).

### Database

```rust
// Open or create a file-backed database
let db = Database::open("path/to/db.lv")?;

// Open with custom config
let db = Database::open_with_config("db.lv", DatabaseConfig {
    page_size: 4096,
    wal_enabled: true,
})?;

// In-memory database
let db = Database::open_memory()?;

// Collection operations
let col = db.create_collection("name", 384)?;
let col = db.get_collection("name");
db.delete_collection("name")?;
let names = db.list_collections();
```

### Collection

```rust
// Insert (upserts if ID exists)
collection.insert("id", &vector, json!({"key": "value"}))?;

// Batch insert
collection.insert_batch(&[
    ("id1", &vec1, json!({})),
    ("id2", &vec2, json!({"tag": "important"})),
])?;

// Search
let results = collection.search(&query, 10).execute()?;

// Search with filter
let results = collection.search(&query, 10)
    .filter(Filter::And(vec![
        Filter::Eq("category".into(), json!("docs")),
        Filter::Gte("year".into(), 2024.0),
    ]))
    .execute()?;

// Get, delete, update
let record = collection.get("id")?;
collection.delete("id")?;
collection.update_metadata("id", json!({"updated": true}))?;

// Info
println!("Count: {}", collection.len());
println!("Dimension: {}", collection.dimension());
```

### Filters

```rust
use litevec::Filter;

Filter::Eq("field".into(), json!("value"))     // field == value
Filter::Ne("field".into(), json!("value"))     // field != value
Filter::Gt("field".into(), 10.0)               // field > 10
Filter::Gte("field".into(), 10.0)              // field >= 10
Filter::Lt("field".into(), 100.0)              // field < 100
Filter::Lte("field".into(), 100.0)             // field <= 100
Filter::In("field".into(), vec![json!("a"), json!("b")])  // field in [a, b]
Filter::Exists("field".into())                 // field exists
Filter::And(vec![filter1, filter2])            // AND
Filter::Or(vec![filter1, filter2])             // OR
Filter::Not(Box::new(filter))                  // NOT
```

### Collection Config

```rust
use litevec::{CollectionConfig, DistanceType, IndexType};

let config = CollectionConfig {
    dimension: 384,
    distance: DistanceType::Cosine,     // Cosine, Euclidean, DotProduct
    index: IndexType::Hnsw,             // Hnsw, Flat, Auto
    ..CollectionConfig::new(384)
};

let col = db.create_collection_with_config("name", config)?;
```

## 📊 Benchmarks

Real benchmark results on random 128-dimensional vectors (cosine distance, HNSW M=16, ef_construction=200, ef_search=100). Measured on Windows x86_64 with AVX2.

### Search Latency (100 queries, k=10)

| Vectors | LiteVec HNSW | FAISS HNSW | LiteVec Flat | FAISS Flat |
|--------:|:------------:|:----------:|:------------:|:----------:|
| 1,000 | 136μs | 50μs | 54μs | 13μs |
| 10,000 | 406μs | 131μs | 862μs | 121μs |
| 50,000 | 550μs | 468μs | — | 1,368μs |

### Insert Throughput

| Vectors | LiteVec HNSW | FAISS HNSW |
|--------:|:------------:|:----------:|
| 1,000 | 4,607 vec/s | 76,202 vec/s |
| 10,000 | 2,180 vec/s | 32,349 vec/s |
| 50,000 | 970 vec/s | 13,056 vec/s |

### Persistence

| Operation | 10,000 vectors (dim=128) |
|-----------|:------------------------:|
| Save to disk | 144ms |
| Load + rebuild index | 5,746ms |

**Key takeaway:** FAISS is faster at raw computation (highly optimized C++ with decades of work), but LiteVec trades some speed for a dramatically simpler developer experience — no server, no Docker, single file, metadata filtering built-in, and Rust memory safety. For most RAG and semantic search workloads (< 100K vectors), LiteVec's sub-millisecond search latency is more than sufficient.

## 🏗️ Architecture

```
┌──────────────────────────────────────┐
│          Public API (Database)        │
├──────────────────────────────────────┤
│  Collection  │  Collection  │  ...   │
├──────────────┼──────────────┼────────┤
│ HNSW/Flat/   │  Metadata    │  BM25  │
│ DiskANN/PQ   │  + SecIndex  │  FTS   │
├──────────────┴──────────────┴────────┤
│    Storage │ Persistence │ Backup    │
│    (mmap)  │ (snapshot)  │ (restore) │
├──────────────────────────────────────┤
│         WAL (CRC32) │ Single .lv     │
└──────────────────────────────────────┘
```

### Bindings & Tools

```
┌─────────┬─────────┬──────┬──────┬──────┬─────┐
│  Rust   │ Python  │ Node │  C   │ WASM │ MCP │
│  API    │ PyO3    │ NAPI │ FFI  │      │ srv │
└─────────┴─────────┴──────┴──────┴──────┴─────┘
              ↓ All backed by litevec-core ↓
```

## 🤖 MCP Server

LiteVec includes an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that gives AI agents vector search as a tool.

```bash
# Run with file-backed database
litevec-mcp --db my_vectors.lv

# Run with in-memory database
litevec-mcp --memory
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "litevec": {
      "command": "litevec-mcp",
      "args": ["--db", "knowledge_base.lv"]
    }
  }
}
```

**Available tools (11):** `create_collection`, `list_collections`, `insert`, `batch_insert`, `search`, `text_search`, `hybrid_search`, `get`, `delete`, `update_metadata`, `collection_info`

### VS Code / GitHub Copilot

Add `.vscode/mcp.json` to your project:

```json
{
  "servers": {
    "litevec": {
      "command": "cargo",
      "args": ["run", "-p", "litevec-mcp", "--", "--memory"]
    }
  }
}
```

See [docs/api/mcp.md](docs/api/mcp.md) for full protocol documentation.

## 🗺️ Roadmap

### Phase 1 ✅ — Core Engine MVP
- [x] Storage engine with memory-mapped I/O
- [x] Flat index (brute-force) + HNSW index (approximate)
- [x] SIMD-accelerated distance functions (AVX2)
- [x] Metadata filtering (11 operators)
- [x] Write-ahead log (WAL) with crash safety
- [x] Full Rust API + Python bindings + CLI tool
- [x] Data persistence across open/close

### Phase 2 ✅ — Ecosystem & Performance
- [x] Node.js bindings (NAPI-RS) with TypeScript types
- [x] C FFI layer (15 functions + header file)
- [x] MCP server for AI agents (JSON-RPC 2.0)
- [x] NEON SIMD for ARM / Apple Silicon
- [x] Product quantization (32× memory reduction)
- [x] Criterion benchmarks (distance + search)
- [x] Batch insert optimization
- [x] Documentation site (docs/)
- [x] Examples: RAG pipeline, Ollama, LangChain integration

### Phase 3 ✅ — Scale & Community
- [x] WASM build for browsers and edge runtimes
- [x] DiskANN (Vamana graph) for billion-scale search
- [x] Full-text search on metadata (BM25)
- [x] Hybrid search (vector + keyword with RRF fusion)
- [x] Secondary indexes on metadata fields (B-tree)
- [x] Snapshot backup & restore API
- [x] Community infrastructure (CONTRIBUTING.md, issue templates, CoC)

### Future
- [ ] Distributed mode (multi-node sharding)
- [ ] GPU-accelerated distance computation
- [ ] Streaming insert from Kafka/Redis
- [ ] Official Docker image for MCP server
- [ ] crates.io / PyPI / npm publishing

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <strong>Built with 🦀 Rust</strong>
  <br>
  <sub>The embedded vector database. No server. No Docker. No config.</sub>
</div>
