# Python API Reference

## Installation

```bash
pip install litevec
```

## Database

```python
import litevec

# File-backed
db = litevec.Database("my_vectors.lv")

# In-memory
db = litevec.Database.open_memory()
```

### Methods

| Method | Description |
|--------|-------------|
| `db.create_collection(name, dimension)` | Create a new collection |
| `db.get_collection(name)` | Get existing collection |
| `db.delete_collection(name)` | Delete a collection |
| `db.list_collections()` | List collection names |

## Collection

```python
col = db.create_collection("documents", dimension=384)
```

### Insert

```python
# Single insert
col.insert("doc1", [0.1, 0.2, ...], {"title": "Hello World"})

# Metadata is optional
col.insert("doc2", [0.3, 0.4, ...])
```

### Search

```python
results = col.search([0.1, 0.2, ...], k=10)

for result in results:
    print(f"ID: {result['id']}")
    print(f"Distance: {result['distance']:.4f}")
    print(f"Metadata: {result['metadata']}")
```

### Get / Delete

```python
record = col.get("doc1")
# Returns: {"id": "doc1", "vector": [...], "metadata": {...}}

col.delete("doc1")
```

### Properties

| Property | Description |
|----------|-------------|
| `col.len()` | Number of vectors |
| `col.dimension` | Vector dimension |
| `col.name` | Collection name |

## Filtered Search

```python
# Equality filter
results = col.search(query, k=10, filter={"category": "science"})

# The filter parameter accepts a dict that is matched against metadata.
```

## Thread Safety

The Python bindings release the GIL during search and insert operations, allowing concurrent access from multiple threads.

## Example: RAG Pipeline

```python
import litevec
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
db = litevec.Database.open_memory()
col = db.create_collection("docs", dimension=384)

# Index documents
documents = ["The cat sat on the mat", "Dogs are loyal animals", ...]
for i, doc in enumerate(documents):
    embedding = model.encode(doc).tolist()
    col.insert(f"doc_{i}", embedding, {"text": doc})

# Search
query_embedding = model.encode("What animals are friendly?").tolist()
results = col.search(query_embedding, k=5)

for r in results:
    print(f"{r['distance']:.4f}: {r['metadata']['text']}")
```
