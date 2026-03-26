"""LiteVec — Basic Usage Example

Demonstrates the core operations: create, insert, search, get, delete.

Install:
    pip install litevec

Run:
    python basic_usage.py
"""

import litevec


def main():
    # Open an in-memory database (use litevec.open("my_vectors.lv") for persistence)
    db = litevec.open_memory()

    # Create a collection with 3-dimensional vectors
    collection = db.create_collection("docs", dimension=3)
    print(f"Created collection: {collection.name}, dimension={collection.dimension}")

    # Insert vectors with metadata
    collection.insert("doc1", [1.0, 0.0, 0.0], {"title": "Introduction to AI", "category": "tech"})
    collection.insert("doc2", [0.0, 1.0, 0.0], {"title": "Machine Learning 101", "category": "tech"})
    collection.insert("doc3", [0.0, 0.0, 1.0], {"title": "Cooking with Rust", "category": "food"})
    print(f"Inserted {len(collection)} vectors")

    # Search for the nearest neighbors
    query = [0.9, 0.1, 0.0]
    results = collection.search(query, k=3)
    print("\nSearch results:")
    for r in results:
        print(f"  {r.id}: distance={r.distance:.4f}, metadata={r.metadata}")

    # Search with a metadata filter
    results = collection.search(query, k=10, filter={"category": "tech"})
    print("\nFiltered search (category=tech):")
    for r in results:
        print(f"  {r.id}: distance={r.distance:.4f}, metadata={r.metadata}")

    # Get a specific vector by ID
    record = collection.get("doc1")
    print(f"\nGet doc1: {record}")

    # Update metadata
    collection.update_metadata("doc1", {"title": "Updated Title", "category": "tech"})
    record = collection.get("doc1")
    print(f"After update: {record}")

    # Delete a vector
    deleted = collection.delete("doc3")
    print(f"\nDeleted doc3: {deleted}")
    print(f"Collection size: {len(collection)}")

    # List collections
    names = db.list_collections()
    print(f"\nCollections: {names}")


if __name__ == "__main__":
    main()
