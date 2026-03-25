"""
LiteVec RAG (Retrieval-Augmented Generation) Pipeline Example

This example demonstrates how to build a simple RAG pipeline using LiteVec
as the vector store. It shows:
1. Chunking documents into passages
2. Generating embeddings (simulated here; replace with your embedding model)
3. Storing vectors in LiteVec
4. Retrieving relevant context for a query
5. Constructing a prompt with retrieved context

Requirements:
    pip install litevec
    # For real embeddings, also install: sentence-transformers or openai
"""

import hashlib
import json

# For demonstration, we simulate embeddings.
# In production, replace with real embedding model (see commented code below).


def fake_embedding(text: str, dim: int = 128) -> list[float]:
    """Generate a deterministic pseudo-embedding from text (for demo only)."""
    h = hashlib.sha256(text.encode()).hexdigest()
    values = []
    for i in range(dim):
        byte_val = int(h[(i * 2) % len(h) : (i * 2 + 2) % len(h) + 2][:2], 16)
        values.append((byte_val / 255.0) * 2 - 1)  # normalize to [-1, 1]
    # Normalize to unit vector
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values] if norm > 0 else values


# ---- Real embedding example (uncomment to use) ----
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim
# def real_embedding(text: str) -> list[float]:
#     return model.encode(text).tolist()


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_rag_prompt(query: str, context_passages: list[str]) -> str:
    """Build a prompt with retrieved context for an LLM."""
    context = "\n\n---\n\n".join(context_passages)
    return f"""Answer the question based on the following context. If the answer
is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""


def main():
    # In production, use: import litevec
    # For this demo, we'll use the litevec Python API
    try:
        import litevec
    except ImportError:
        print("litevec not installed. Install with: pip install litevec")
        print("Running with mock database for demonstration...\n")
        litevec = None

    # Sample documents
    documents = [
        {
            "title": "Introduction to Vector Databases",
            "content": (
                "Vector databases are specialized database systems designed to store, "
                "index, and query high-dimensional vector embeddings. They are essential "
                "for modern AI applications including semantic search, recommendation "
                "systems, and retrieval-augmented generation. Unlike traditional databases "
                "that use exact matching, vector databases find similar items using "
                "distance metrics like cosine similarity, euclidean distance, or dot product."
            ),
        },
        {
            "title": "HNSW Algorithm",
            "content": (
                "Hierarchical Navigable Small World (HNSW) is a graph-based algorithm "
                "for approximate nearest neighbor search. It builds a multi-layer graph "
                "where higher layers provide long-range connections for fast navigation, "
                "while lower layers provide precise local search. HNSW achieves excellent "
                "recall rates (>95%) with sub-millisecond query latency, making it the "
                "most popular index type for vector databases."
            ),
        },
        {
            "title": "Embeddings in NLP",
            "content": (
                "Text embeddings are dense vector representations of text that capture "
                "semantic meaning. Models like OpenAI's text-embedding-ada-002, Sentence "
                "Transformers, and Cohere's embed models convert text into fixed-dimensional "
                "vectors. Similar texts produce vectors that are close together in the "
                "embedding space, enabling semantic search and similarity comparison."
            ),
        },
    ]

    dim = 128
    print("=== LiteVec RAG Pipeline Example ===\n")

    if litevec:
        # Use real LiteVec
        db = litevec.Database.open_memory()
        collection = db.create_collection("documents", dimension=dim)

        # Chunk and index documents
        print("Indexing documents...")
        chunk_id = 0
        for doc in documents:
            chunks = chunk_text(doc["content"], chunk_size=50, overlap=10)
            for i, chunk in enumerate(chunks):
                embedding = fake_embedding(chunk, dim)
                metadata = {
                    "title": doc["title"],
                    "chunk_index": i,
                    "text": chunk,
                }
                collection.insert(f"chunk_{chunk_id}", embedding, metadata)
                chunk_id += 1

        print(f"Indexed {collection.len()} chunks from {len(documents)} documents\n")

        # Query
        query = "How does HNSW work for vector search?"
        print(f"Query: {query}\n")

        query_embedding = fake_embedding(query, dim)
        results = collection.search(query_embedding, k=3)

        print("Retrieved passages:")
        context_passages = []
        for i, result in enumerate(results):
            meta = result.metadata if hasattr(result, "metadata") else {}
            text = meta.get("text", "N/A") if isinstance(meta, dict) else "N/A"
            title = meta.get("title", "N/A") if isinstance(meta, dict) else "N/A"
            print(f"  {i + 1}. [{title}] (distance: {result.distance:.4f})")
            print(f"     {text[:100]}...\n")
            context_passages.append(text)

        # Build RAG prompt
        prompt = build_rag_prompt(query, context_passages)
        print("=" * 60)
        print("Generated RAG Prompt:")
        print("=" * 60)
        print(prompt)
    else:
        # Mock demonstration
        print("Indexing documents...")
        chunk_id = 0
        all_chunks = []
        for doc in documents:
            chunks = chunk_text(doc["content"], chunk_size=50, overlap=10)
            for i, chunk in enumerate(chunks):
                all_chunks.append(
                    {"id": f"chunk_{chunk_id}", "title": doc["title"], "text": chunk}
                )
                chunk_id += 1

        print(f"Would index {len(all_chunks)} chunks from {len(documents)} documents")
        print("\nQuery: How does HNSW work for vector search?")
        print("\n(Install litevec to see actual search results)")


if __name__ == "__main__":
    main()
