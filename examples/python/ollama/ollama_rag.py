"""
LiteVec + Ollama Integration Example

This example shows how to use LiteVec with Ollama for a fully local,
offline RAG pipeline:
1. Use Ollama to generate embeddings from text
2. Store embeddings in LiteVec
3. Search for similar documents
4. Use Ollama to generate answers with retrieved context

Requirements:
    pip install litevec requests
    # Ollama must be running: ollama serve
    # Pull an embedding model: ollama pull nomic-embed-text
    # Pull a chat model: ollama pull llama3.2
"""

import json
import sys

try:
    import requests
except ImportError:
    print("requests not installed. Install with: pip install requests")
    sys.exit(1)

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"


def ollama_embed(text: str) -> list[float]:
    """Generate an embedding using Ollama."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"][0]


def ollama_chat(prompt: str) -> str:
    """Generate a response using Ollama."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def main():
    try:
        import litevec
    except ImportError:
        print("litevec not installed. Install with: pip install litevec")
        sys.exit(1)

    # Check Ollama is running
    try:
        requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    except requests.ConnectionError:
        print("Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    print("=== LiteVec + Ollama: Fully Local RAG ===\n")

    # Get embedding dimension by generating a test embedding
    test_embed = ollama_embed("test")
    dim = len(test_embed)
    print(f"Embedding dimension: {dim}")

    # Create database
    db = litevec.Database.open_memory()
    collection = db.create_collection("knowledge", dimension=dim)

    # Knowledge base
    documents = [
        "LiteVec is an embedded vector database written in Rust. It requires no server, no Docker, and no configuration.",
        "HNSW (Hierarchical Navigable Small World) is the primary index type in LiteVec, providing >95% recall at sub-millisecond latency.",
        "LiteVec supports cosine similarity, euclidean distance, and dot product as distance metrics.",
        "LiteVec has bindings for Python, Node.js, and C. It also includes a built-in MCP server for AI agents.",
        "Product Quantization in LiteVec reduces memory usage by 10-32x by compressing vectors into compact codes.",
    ]

    # Index documents
    print(f"\nIndexing {len(documents)} documents...")
    for i, doc in enumerate(documents):
        embedding = ollama_embed(doc)
        collection.insert(f"doc_{i}", embedding, {"text": doc})
    print(f"Indexed {collection.len()} documents\n")

    # Interactive query loop
    print("Ask questions about LiteVec (type 'quit' to exit):\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        # Search for relevant documents
        query_embedding = ollama_embed(query)
        results = collection.search(query_embedding, k=3)

        # Build context from results
        context_parts = []
        for r in results:
            meta = r.metadata if hasattr(r, "metadata") else {}
            text = meta.get("text", "") if isinstance(meta, dict) else ""
            if text:
                context_parts.append(text)

        context = "\n".join(f"- {p}" for p in context_parts)

        # Generate answer with Ollama
        prompt = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {query}

Answer:"""

        answer = ollama_chat(prompt)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
