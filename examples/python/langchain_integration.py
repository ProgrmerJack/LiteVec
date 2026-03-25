"""
LiteVec + LangChain Integration Example

This example demonstrates using LiteVec as a vector store with LangChain
for building RAG applications.

Requirements:
    pip install litevec langchain langchain-community sentence-transformers
"""

import hashlib
import sys


def fake_embedding(text: str, dim: int = 384) -> list[float]:
    """Deterministic pseudo-embedding for demo purposes."""
    h = hashlib.sha256(text.encode()).hexdigest()
    values = []
    for i in range(dim):
        byte_val = int(h[(i * 2) % len(h) : (i * 2 + 2) % len(h) + 2][:2], 16)
        values.append((byte_val / 255.0) * 2 - 1)
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values] if norm > 0 else values


class LiteVecVectorStore:
    """A LangChain-compatible vector store backed by LiteVec.

    This adapter wraps LiteVec to work with LangChain's retriever interface.

    Usage with LangChain:
        from langchain.chains import RetrievalQA
        from langchain.llms import Ollama

        vectorstore = LiteVecVectorStore(collection, embed_fn)
        retriever = vectorstore.as_retriever(k=5)

        qa = RetrievalQA.from_chain_type(
            llm=Ollama(model="llama3.2"),
            retriever=retriever,
        )
        answer = qa.run("What is LiteVec?")
    """

    def __init__(self, collection, embedding_fn):
        """
        Args:
            collection: A LiteVec Collection instance.
            embedding_fn: A function that takes a string and returns a list of floats.
        """
        self.collection = collection
        self.embedding_fn = embedding_fn
        self._doc_counter = 0

    def add_texts(self, texts, metadatas=None):
        """Add texts to the vector store."""
        ids = []
        for i, text in enumerate(texts):
            doc_id = f"doc_{self._doc_counter}"
            self._doc_counter += 1
            embedding = self.embedding_fn(text)
            metadata = metadatas[i] if metadatas else {}
            metadata["text"] = text
            self.collection.insert(doc_id, embedding, metadata)
            ids.append(doc_id)
        return ids

    def similarity_search(self, query, k=4):
        """Search for documents similar to the query."""
        query_embedding = self.embedding_fn(query)
        results = self.collection.search(query_embedding, k=k)
        documents = []
        for r in results:
            meta = r.metadata if hasattr(r, "metadata") else {}
            text = meta.get("text", "") if isinstance(meta, dict) else ""
            documents.append(
                {"page_content": text, "metadata": meta, "distance": r.distance}
            )
        return documents

    def as_retriever(self, k=4):
        """Return a retriever-like object."""
        store = self

        class Retriever:
            def get_relevant_documents(self, query):
                return store.similarity_search(query, k=k)

        return Retriever()


def main():
    try:
        import litevec
    except ImportError:
        print("litevec not installed. Install with: pip install litevec")
        print("Showing LangChain integration pattern...\n")
        litevec = None

    print("=== LiteVec + LangChain Integration ===\n")

    if litevec:
        db = litevec.Database.open_memory()
        collection = db.create_collection("langchain_docs", dimension=384)
    else:
        collection = None

    # Documents to index
    texts = [
        "LiteVec is an embedded vector database written in Rust.",
        "It supports HNSW indexing for fast approximate nearest neighbor search.",
        "LiteVec has Python, Node.js, and C bindings.",
        "The database uses SIMD acceleration (AVX2/NEON) for fast distance computation.",
        "LiteVec includes a built-in MCP server for AI agent integration.",
    ]

    if collection:
        # Create vector store adapter
        vectorstore = LiteVecVectorStore(collection, lambda t: fake_embedding(t, 384))

        # Add documents
        ids = vectorstore.add_texts(texts)
        print(f"Indexed {len(ids)} documents\n")

        # Search
        query = "How does LiteVec handle vector search?"
        print(f"Query: {query}\n")
        results = vectorstore.similarity_search(query, k=3)

        print("Results:")
        for i, doc in enumerate(results):
            print(f"  {i + 1}. (distance: {doc['distance']:.4f})")
            print(f"     {doc['page_content'][:100]}\n")

        # Show LangChain integration pattern
        print("=" * 60)
        print("To use with LangChain RetrievalQA:")
        print("=" * 60)
        print(
            """
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

retriever = vectorstore.as_retriever(k=5)
qa = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama3.2"),
    retriever=retriever,
)
answer = qa.run("What is LiteVec?")
print(answer)
"""
        )
    else:
        print("LiteVecVectorStore adapter pattern:")
        print("  vectorstore = LiteVecVectorStore(collection, embedding_fn)")
        print("  vectorstore.add_texts(texts)")
        print('  results = vectorstore.similarity_search("query", k=5)')
        print("  retriever = vectorstore.as_retriever(k=5)")


if __name__ == "__main__":
    main()
