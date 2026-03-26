# Performance Tuning

LiteVec performance depends on three factors: index parameters, distance metric choice, and hardware capabilities.

## HNSW Parameters

| Parameter | Default | Effect | Recommendation |
|-----------|---------|--------|----------------|
| `m` | 16 | Connections per node | 12–48. Higher = better recall, more RAM |
| `ef_construction` | 200 | Build-time search width | 100–500. Higher = slower insert, better graph |
| `ef_search` | 100 | Query-time search width | 50–500. Tune for recall vs latency |

### Tuning for Recall

```rust
use litevec::{CollectionConfig, HnswConfig};

let config = CollectionConfig {
    dimension: 384,
    hnsw: HnswConfig {
        m: 32,                // More connections
        ef_construction: 400, // Better graph quality
        ef_search: 200,       // Wider search beam
    },
    ..Default::default()
};
```

### Tuning for Speed

```rust
let config = CollectionConfig {
    dimension: 384,
    hnsw: HnswConfig {
        m: 12,
        ef_construction: 100,
        ef_search: 50,
    },
    ..Default::default()
};
```

## Distance Metric Selection

| Metric | Best For | Notes |
|--------|----------|-------|
| Cosine | Normalized embeddings (OpenAI, sentence-transformers) | Default. Insensitive to vector magnitude. |
| Euclidean | Raw feature vectors, geographic data | Sensitive to scale differences. |
| DotProduct | Already-normalized vectors, maximum inner product search | Fastest computation. |

## SIMD Acceleration

LiteVec automatically uses SIMD instructions when available:
- **AVX2** (x86_64): ~4× speedup for distance computations
- **NEON** (aarch64/Apple Silicon): ~3× speedup

No configuration needed — runtime detection selects the fastest path.

## Memory Usage

Approximate memory per vector:

| Component | Bytes per Vector (128-dim) |
|-----------|---------------------------|
| Raw vector | 512 (128 × 4 bytes) |
| HNSW graph | ~256 (m=16, ~16 links × 16 bytes) |
| Metadata | Variable (JSON) |
| ID mapping | ~64 |
| **Total** | **~832 + metadata** |

### Product Quantization

For memory-constrained scenarios, product quantization compresses vectors:

| Dimension | Raw Size | PQ Size (16 subvectors) | Compression |
|-----------|----------|------------------------|-------------|
| 128 | 512 B | 16 B | 32× |
| 384 | 1536 B | 48 B | 32× |
| 768 | 3072 B | 96 B | 32× |

Trade-off: ~5-10% recall reduction for 32× memory savings.

## Batch Operations

Insert vectors in batches for better throughput:

```rust
let items: Vec<(&str, &[f32], serde_json::Value)> = vec![
    ("id1", &vector1, json!({})),
    ("id2", &vector2, json!({})),
    // ... thousands more
];
col.insert_batch(&items)?;
```

## Filtered Search

Metadata filters are evaluated before vector comparison when secondary indexes exist, reducing the search space. For non-indexed fields, filters are applied post-retrieval.

**Create secondary indexes on frequently filtered fields:**

```rust
col.create_index("category");
col.create_index("timestamp");

// List indexed fields
let fields = col.indexed_fields();

// Drop an index you no longer need
col.drop_index("timestamp");
```

## Benchmarks

See the [README benchmarks section](../../README.md#-benchmarks) for real numbers comparing LiteVec against FAISS.
