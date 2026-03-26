"""LiteVec + Sentence Transformers Example

Demonstrates using LiteVec with the sentence-transformers library for
real semantic search over text documents.

Install:
    pip install litevec sentence-transformers

Run:
    python with_sentence_transformers.py
"""

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("This example requires sentence-transformers:")
    print("  pip install sentence-transformers")
    raise SystemExit(1)

import litevec


def main():
    # Load a sentence-transformer model (downloads on first run, ~90MB)
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional embeddings
    dimension = model.get_sentence_embedding_dimension()

    # Open a LiteVec database
    db = litevec.open_memory()
    collection = db.create_collection("articles", dimension=dimension)

    # Sample documents to index
    documents = [
        {"id": "doc1", "text": "Python is a popular programming language for data science and machine learning."},
        {"id": "doc2", "text": "Rust provides memory safety without garbage collection through ownership."},
        {"id": "doc3", "text": "Vector databases enable semantic search over unstructured data."},
        {"id": "doc4", "text": "Neural networks learn representations by adjusting weights during training."},
        {"id": "doc5", "text": "SQLite is the most widely deployed database engine in the world."},
        {"id": "doc6", "text": "Transformers use self-attention mechanisms for natural language processing."},
        {"id": "doc7", "text": "Embeddings convert text into dense numerical vectors for similarity comparison."},
        {"id": "doc8", "text": "HNSW graphs provide fast approximate nearest neighbor search."},
    ]

    # Encode and insert all documents
    print(f"Encoding {len(documents)} documents...")
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts)

    for doc, embedding in zip(documents, embeddings):
        collection.insert(doc["id"], embedding.tolist(), {"text": doc["text"]})
    print(f"Indexed {len(collection)} documents (dimension={dimension})")

    # Semantic search
    queries = [
        "How do databases work?",
        "What programming language should I learn?",
        "How does AI understand text?",
    ]

    for query in queries:
        print(f"\n--- Query: \"{query}\" ---")
        query_embedding = model.encode(query).tolist()
        results = collection.search(query_embedding, k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.distance:.4f}] {r.metadata['text']}")


if __name__ == "__main__":
    main()
