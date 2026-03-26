# Comparison: LiteVec vs Alternatives

How does LiteVec compare to other vector search solutions?

## Overview

| Feature | LiteVec | Qdrant | FAISS | pgvector | Pinecone | lancedb |
|---------|:-------:|:------:|:-----:|:--------:|:--------:|:-------:|
| **Embedded (no server)** | ✅ | ❌ | ⚠️ Library | ❌ | ❌ | ✅ |
| **Single file database** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ (directory) |
| **Metadata filtering** | ✅ 11 ops | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Full-text + hybrid search** | ✅ BM25+RRF | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Zero external deps** | ✅ Pure Rust | ❌ | ❌ C++ | ❌ PostgreSQL | ❌ Cloud | ❌ Arrow |
| **Works offline** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Language bindings** | Rust, Python, Node, C, WASM | Rust, Python, Go, JS | C++, Python | SQL | Python, Node, Go | Python, JS |
| **MCP server** | ✅ Built-in | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Backup & restore** | ✅ | ✅ | ❌ | ✅ (pg_dump) | ✅ | ✅ |
| **Crash safety** | ✅ WAL | ✅ | ❌ | ✅ | ✅ | ✅ |

## Detailed Comparisons

### LiteVec vs Qdrant

**Qdrant** is a production-grade vector search engine with rich features (multi-tenancy, distributed mode, quantization). However, it requires running a separate server process or Docker container.

**Choose LiteVec when:**
- You want to embed vector search directly into your application
- You don't want to manage a separate server process
- You want a single-file database for easy backup/deploy
- You're building a CLI tool, desktop app, or edge device

**Choose Qdrant when:**
- You need distributed multi-node deployment
- You need multi-tenancy and access control
- You're building a high-traffic web service
- You need production monitoring and observability

### LiteVec vs FAISS

**FAISS** (Facebook AI Similarity Search) is the gold standard for raw vector search performance, with decades of optimization in C++.

**Choose LiteVec when:**
- You need metadata filtering (FAISS has none)
- You need data persistence (FAISS is in-memory only by default)
- You want a complete database (not just an index)
- You need crash safety and data integrity

**Choose FAISS when:**
- You need maximum raw search speed on huge datasets (100M+ vectors)
- You're doing offline batch processing
- You need GPU-accelerated search
- You only need vector indexing, not a full database

### LiteVec vs pgvector

**pgvector** adds vector search capabilities to PostgreSQL.

**Choose LiteVec when:**
- You don't want to install/manage PostgreSQL
- You want an embedded, zero-config solution
- You're building a standalone app (CLI, desktop, mobile)
- You want SIMD-optimized search without PostgreSQL overhead

**Choose pgvector when:**
- You already use PostgreSQL
- You need SQL joins between vector data and relational data
- You need PostgreSQL's ecosystem (replication, backup, extensions)

### LiteVec vs Pinecone

**Pinecone** is a fully managed cloud vector database.

**Choose LiteVec when:**
- You want to run locally / offline
- You don't want vendor lock-in or recurring costs
- You need to embed vector search in your application
- Privacy or data sovereignty matters

**Choose Pinecone when:**
- You want zero operational overhead (fully managed)
- You need automatic scaling
- You're building a cloud-native SaaS application

### LiteVec vs lancedb

**lancedb** is an embedded vector database built on Apache Arrow and the Lance columnar format.

**Choose LiteVec when:**
- You want a single-file database (lancedb uses a directory of files)
- You need C FFI or WASM bindings
- You want built-in MCP server for AI agents
- You prefer Rust-native with no Arrow dependency

**Choose lancedb when:**
- You need tight integration with the Arrow/Parquet ecosystem
- You're doing hybrid analytics + vector search
- You want versioned datasets with time travel

## Performance

For performance benchmarks, see the [performance guide](guide/performance.md) and the benchmark data in the [README](../README.md#-benchmarks).

**Key takeaway:** LiteVec trades some raw compute speed (vs FAISS) for a dramatically simpler developer experience — no server, no Docker, single file, metadata filtering built-in, and Rust memory safety. For most RAG and semantic search workloads (< 100K vectors), LiteVec's sub-millisecond search latency is more than sufficient.
