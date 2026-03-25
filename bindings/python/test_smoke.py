"""Quick smoke test for litevec Python bindings."""
import litevec
import random

# Test in-memory database
db = litevec.open_memory()
print(f"Database: {db}")

# Create collection
col = db.create_collection("test", dimension=64)
print(f"Collection: {col}")

# Insert vectors
for i in range(20):
    vec = [random.random() for _ in range(64)]
    col.insert(f"doc_{i}", vec, {"category": "even" if i % 2 == 0 else "odd", "value": i})

print(f"Inserted {len(col)} vectors")

# Search
query = [random.random() for _ in range(64)]
results = col.search(query, k=5)
for r in results:
    print(f"  {r.id}: distance={r.distance:.4f}")

# Search with filter
results = col.search(query, k=5, filter={"category": "even"})
print(f"Filtered results: {len(results)}")
for r in results:
    meta = r.metadata
    cat = meta["category"]
    assert cat == "even", f"Expected 'even', got '{cat}'"
    print(f"  {r.id}: distance={r.distance:.4f}, category={cat}")

# Get
record = col.get("doc_0")
assert record is not None
print(f"Get doc_0: id={record['id']}, dim={len(record['vector'])}")

# Delete
deleted = col.delete("doc_0")
assert deleted is True
print(f"Deleted: {deleted}, count: {len(col)}")

# List collections
names = db.list_collections()
assert "test" in names
print(f"Collections: {names}")

print("\nAll Python smoke tests passed!")
