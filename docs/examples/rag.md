# RAG Pipeline Example

Build a complete Retrieval-Augmented Generation (RAG) pipeline with LiteVec.

## Architecture

```
User Query → Embed → LiteVec Search → Top-K Chunks → LLM Prompt → Response
                          ↑
    Documents → Chunk → Embed → LiteVec Insert
```

## Python Implementation

```python
import litevec
from sentence_transformers import SentenceTransformer

# Initialize
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
db = litevec.Database("knowledge_base.lv")
col = db.create_collection("documents", dimension=384)

# Index documents
def index_document(doc_id: str, text: str, metadata: dict):
    """Chunk, embed, and store a document."""
    chunks = chunk_text(text, max_tokens=256, overlap=50)
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        col.insert(
            f"{doc_id}_chunk_{i}",
            embedding,
            {"text": chunk, "source": doc_id, **metadata}
        )

def chunk_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens - overlap):
        chunk = " ".join(words[i:i + max_tokens])
        if chunk:
            chunks.append(chunk)
    return chunks

# Search and generate
def rag_query(question: str, k: int = 5) -> str:
    """Search for relevant chunks and build context."""
    query_embedding = model.encode(question).tolist()
    results = col.search(query_embedding, k=k)
    
    context = "\n\n".join(
        f"[Source: {r['metadata']['source']}]\n{r['metadata']['text']}"
        for r in results
    )
    
    prompt = f"""Answer the question based on the following context:

{context}

Question: {question}
Answer:"""
    
    return prompt  # Send to your LLM of choice

# Usage
index_document("paper1", "LiteVec is an embedded vector database...", {"type": "paper"})
prompt = rag_query("What is LiteVec?")
```

## Rust Implementation

```rust
use litevec::Database;
use serde_json::json;

fn main() -> litevec::Result<()> {
    let db = Database::open("knowledge_base.lv")?;
    let col = db.create_collection("documents", 384)?;

    // Insert pre-computed embeddings
    col.insert("chunk_1", &embeddings[0], json!({
        "text": "LiteVec is an embedded vector database...",
        "source": "README.md"
    }))?;

    // Search
    let results = col.search(&query_embedding, 5).execute()?;
    
    let context: String = results.iter()
        .map(|r| r.metadata["text"].as_str().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n\n");

    println!("Context for LLM:\n{context}");
    Ok(())
}
```

See also: `examples/python/rag/rag_pipeline.py` and `examples/python/ollama/ollama_rag.py` for complete runnable examples.
