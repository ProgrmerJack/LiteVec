# Semantic Search Example

Combine LiteVec's vector search with BM25 full-text search for hybrid retrieval.

## Vector Search (Semantic)

Vector search finds documents by meaning, not keywords. A query like "How do animals show affection?" matches "Dogs wag their tails when happy" even though they share no words.

```python
import litevec
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
db = litevec.Database.open_memory()
col = db.create_collection("articles", dimension=384)

articles = [
    ("a1", "Dogs wag their tails when happy"),
    ("a2", "Cats purr to show contentment"),
    ("a3", "Stock markets rallied on Tuesday"),
    ("a4", "Python 3.12 introduces new type features"),
]

for aid, text in articles:
    embedding = model.encode(text).tolist()
    col.insert(aid, embedding, {"text": text})

# Semantic search
query = model.encode("How do animals show affection?").tolist()
results = col.search(query, k=2)
# Returns: a1 (dogs/happy), a2 (cats/contentment)
```

## Hybrid Search (Vector + Keyword)

LiteVec's hybrid search module combines vector similarity with BM25 keyword scores using fusion strategies.

### Reciprocal Rank Fusion (RRF)

```rust
use litevec_core::metadata::hybrid::{HybridSearch, FusionStrategy};

let hybrid = HybridSearch::new(FusionStrategy::Rrf { k: 60 });

// Vector search results: (id, distance)
let vector_results = vec![(1, 0.1), (2, 0.3), (3, 0.5)];

// BM25 keyword results: (id, score)
let keyword_results = vec![(2, 5.2), (4, 3.1), (1, 2.0)];

// Fuse results
let fused = hybrid.fuse(&vector_results, &keyword_results, 10);
// Doc 2 ranks high in both → top result
// Doc 1 appears in both → second
// Docs 3, 4 appear in one each → lower
```

### Weighted Sum

```rust
let hybrid = HybridSearch::new(FusionStrategy::WeightedSum {
    vector_weight: 0.7,
    keyword_weight: 0.3,
});

let fused = hybrid.fuse(&vector_results, &keyword_results, 10);
```

## Full-Text Search (BM25)

LiteVec includes a built-in BM25 full-text index:

```rust
use litevec_core::metadata::fulltext::FullTextIndex;

let mut fts = FullTextIndex::new();
fts.add_document(1, "The quick brown fox jumps over the lazy dog");
fts.add_document(2, "A fast brown canine leaps across a sleepy hound");

let results = fts.search("quick fox", 10);
// Returns doc 1 (exact keyword match) ranked highest
```

## When to Use Which

| Approach | Best For | Weakness |
|----------|----------|----------|
| Vector only | Semantic similarity, cross-lingual | Misses exact keyword matches |
| BM25 only | Exact term matching, known terms | Can't understand meaning |
| Hybrid (RRF) | General-purpose retrieval | Requires tuning k parameter |
| Hybrid (Weighted) | When you know the balance | Requires weight tuning |
