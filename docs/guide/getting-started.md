# Getting Started

LiteVec is an embedded vector database — no server, no Docker, no configuration. Add it as a dependency and start storing and searching vectors in seconds.

## Installation

### Rust

```toml
[dependencies]
litevec = "0.1"
```

### Python

```bash
pip install litevec
```

### Node.js

```bash
npm install litevec
```

## Quickstart (Rust)

```rust
use litevec::Database;
use serde_json::json;

fn main() -> litevec::Result<()> {
    // Open a database (file-backed, persistent)
    let db = Database::open("my_vectors.lv")?;

    // Create a collection with 3-dimensional vectors
    let col = db.create_collection("documents", 3)?;

    // Insert vectors with metadata
    col.insert("doc1", &[1.0, 0.0, 0.0], json!({"title": "Introduction to AI"}))?;
    col.insert("doc2", &[0.0, 1.0, 0.0], json!({"title": "Machine Learning 101"}))?;
    col.insert("doc3", &[0.0, 0.0, 1.0], json!({"title": "Deep Learning Guide"}))?;

    // Search for nearest neighbors
    let results = col.search(&[0.9, 0.1, 0.0], 2).execute()?;

    for result in &results {
        println!("{}: distance={:.4}, title={}", 
            result.id, result.distance, result.metadata["title"]);
    }

    Ok(())
}
```

## Quickstart (Python)

```python
import litevec

db = litevec.Database.open_memory()
col = db.create_collection("docs", dimension=3)

col.insert("doc1", [1.0, 0.0, 0.0], {"title": "Introduction to AI"})
col.insert("doc2", [0.0, 1.0, 0.0], {"title": "Machine Learning 101"})

results = col.search([0.9, 0.1, 0.0], k=2)
for r in results:
    print(f"{r['id']}: {r['distance']:.4f}")
```

## Core Concepts

### Database
A database is a container for collections. It can be file-backed (persistent) or in-memory (ephemeral).

```rust
// File-backed — data persists across restarts
let db = Database::open("path/to/db.lv")?;

// In-memory — fast, but lost when process exits
let db = Database::open_memory()?;
```

### Collection
A collection stores vectors of a fixed dimension along with JSON metadata. Each vector has a unique string ID.

```rust
let col = db.create_collection("my_collection", 384)?;  // 384-dim vectors
```

### Search
Search finds the k nearest vectors to a query vector. You can optionally filter by metadata.

```rust
use litevec::Filter;

let results = col.search(&query_vector, 10)
    .filter(Filter::Eq("category".into(), "science".into()))
    .execute()?;
```

### Index Types
- **HNSW** (default) — Fast approximate search with >95% recall. Best for most workloads.
- **Flat** — Exact brute-force search. Best for small collections (<10K vectors).

```rust
use litevec::{CollectionConfig, IndexType};

let config = CollectionConfig {
    dimension: 384,
    index: IndexType::Flat,
    ..Default::default()
};
let col = db.create_collection_with_config("exact_search", config)?;
```

### Distance Metrics
- **Cosine** (default) — Best for normalized embeddings (OpenAI, sentence-transformers)
- **Euclidean** — Raw L2 distance
- **DotProduct** — Inner product similarity

### Metadata Filtering
Filter search results by metadata fields:

```rust
use litevec::Filter;

// Equality
Filter::Eq("status".into(), "published".into())

// Numeric range
Filter::And(vec![
    Filter::Gte("price".into(), json!(10.0)),
    Filter::Lt("price".into(), json!(100.0)),
])

// Set membership
Filter::In("category".into(), vec![json!("tech"), json!("science")])

// Boolean logic
Filter::Or(vec![
    Filter::Eq("author".into(), "alice".into()),
    Filter::Eq("author".into(), "bob".into()),
])
```

## Next Steps

- [Architecture Guide](architecture.md) — Understand how LiteVec works internally
- [Performance Tuning](performance.md) — Optimize HNSW parameters for your workload
- [RAG Pipeline Example](../examples/rag.md) — Build a complete retrieval system
