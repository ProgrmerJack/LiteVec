"""
LangChain-compatible LiteVec Vector Store

Provides a VectorStore adapter that integrates LiteVec with the LangChain framework.

Usage:
    pip install litevec langchain-core sentence-transformers

    from langchain_litevec import LiteVecVectorStore
    from langchain_core.documents import Document

    store = LiteVecVectorStore(embedding=my_embedding, dimension=384)
    store.add_documents([Document(page_content="Hello world", metadata={"source": "test"})])
    results = store.similarity_search("hello", k=3)
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

try:
    import litevec
except ImportError:
    raise ImportError("Install litevec: pip install litevec")

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError:
    raise ImportError("Install langchain-core: pip install langchain-core")


class LiteVecVectorStore(VectorStore):
    """LangChain VectorStore backed by LiteVec."""

    def __init__(
        self,
        embedding: Embeddings,
        collection_name: str = "langchain",
        db_path: str | None = None,
        dimension: int = 384,
    ):
        self.embedding = embedding
        self.collection_name = collection_name
        self.dimension = dimension

        if db_path:
            self.db = litevec.Database(db_path)
        else:
            self.db = litevec.Database.open_memory()

        try:
            self.collection = self.db.get_collection(collection_name)
        except Exception:
            self.collection = self.db.create_collection(collection_name, dimension=dimension)

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vector store."""
        texts = list(texts)
        embeddings = self.embedding.embed_documents(texts)
        ids = []

        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
            metadata = metadatas[i] if metadatas else {}
            metadata["text"] = text

            self.collection.insert(doc_id, emb, metadata)
            ids.append(doc_id)

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Search for documents similar to the query."""
        query_embedding = self.embedding.embed_query(query)
        results = self.collection.search(query_embedding, k=k)

        documents = []
        for result in results:
            metadata = dict(result.get("metadata", {}))
            text = metadata.pop("text", "")
            metadata["distance"] = result["distance"]
            metadata["id"] = result["id"]
            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search with relevance scores."""
        query_embedding = self.embedding.embed_query(query)
        results = self.collection.search(query_embedding, k=k)

        docs_with_scores = []
        for result in results:
            metadata = dict(result.get("metadata", {}))
            text = metadata.pop("text", "")
            metadata["id"] = result["id"]
            doc = Document(page_content=text, metadata=metadata)
            docs_with_scores.append((doc, result["distance"]))

        return docs_with_scores

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> LiteVecVectorStore:
        """Create a LiteVecVectorStore from a list of texts."""
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store


if __name__ == "__main__":
    # Demo with mock embedding
    class MockEmbedding(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float(ord(c) % 10) / 10.0 for c in t[:4].ljust(4)] for t in texts]

        def embed_query(self, text: str) -> list[float]:
            return [float(ord(c) % 10) / 10.0 for c in text[:4].ljust(4)]

    store = LiteVecVectorStore(embedding=MockEmbedding(), dimension=4)
    store.add_documents([
        Document(page_content="The cat sat on the mat", metadata={"source": "book1"}),
        Document(page_content="Dogs are loyal animals", metadata={"source": "book2"}),
        Document(page_content="AI is transforming the world", metadata={"source": "article1"}),
    ])

    results = store.similarity_search("cats and dogs", k=2)
    for doc in results:
        print(f"  {doc.page_content} (distance: {doc.metadata.get('distance', 'N/A')})")
