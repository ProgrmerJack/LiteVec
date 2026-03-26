//! Semantic Search Example
//!
//! Demonstrates building a simple semantic search engine using LiteVec.
//! In a real application, you'd use an embedding model (e.g., OpenAI, Ollama,
//! or Sentence Transformers) to generate vectors. This example uses synthetic
//! embeddings to show the search workflow without any external dependencies.
//!
//! Run with: cargo run --example semantic_search

use litevec::{Database, Filter};
use serde_json::json;

/// Simulate an embedding by creating a deterministic vector from text.
/// In production, replace this with a real embedding model.
fn fake_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    for (i, byte) in text.bytes().enumerate() {
        v[i % dim] += (byte as f32 - 96.0) * 0.01;
    }
    // Normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn main() -> litevec::Result<()> {
    let dim = 64;

    // Open an in-memory database
    let db = Database::open_memory()?;
    let col = db.create_collection("articles", dim as u32)?;

    // Index a corpus of documents
    let documents = vec![
        ("article-1", "Rust provides memory safety without garbage collection", "tech"),
        ("article-2", "Python is widely used for data science and AI", "tech"),
        ("article-3", "Vector databases enable semantic search", "database"),
        ("article-4", "HNSW is a graph-based approximate nearest neighbor algorithm", "database"),
        ("article-5", "Neural networks learn from labeled training data", "ai"),
        ("article-6", "SQLite is the most deployed database in the world", "database"),
        ("article-7", "Transformers revolutionized natural language processing", "ai"),
        ("article-8", "LiteVec is the SQLite of vector search", "database"),
    ];

    println!("Indexing {} documents...", documents.len());
    for (id, text, category) in &documents {
        let embedding = fake_embed(text, dim);
        col.insert(id, &embedding, json!({"text": text, "category": category}))?;
    }
    println!("Indexed {} documents (dimension={})\n", col.len(), col.dimension());

    // Search 1: Unfiltered semantic search
    let query = "What is the best programming language?";
    let query_vec = fake_embed(query, dim);

    println!("Query: \"{query}\"");
    let results = col.search(&query_vec, 3).execute()?;
    for (i, r) in results.iter().enumerate() {
        println!(
            "  {}. [dist={:.4}] {} — {}",
            i + 1,
            r.distance,
            r.id,
            r.metadata["text"]
        );
    }

    // Search 2: Filtered by category
    let query = "How do databases store data?";
    let query_vec = fake_embed(query, dim);

    println!("\nQuery: \"{query}\" (category=database only)");
    let results = col
        .search(&query_vec, 3)
        .filter(Filter::Eq("category".into(), json!("database")))
        .execute()?;
    for (i, r) in results.iter().enumerate() {
        println!(
            "  {}. [dist={:.4}] {} — {}",
            i + 1,
            r.distance,
            r.id,
            r.metadata["text"]
        );
    }

    // Search 3: Complex filter (category=tech OR category=ai)
    let query = "machine learning with Rust";
    let query_vec = fake_embed(query, dim);

    println!("\nQuery: \"{query}\" (category in [tech, ai])");
    let results = col
        .search(&query_vec, 3)
        .filter(Filter::Or(vec![
            Filter::Eq("category".into(), json!("tech")),
            Filter::Eq("category".into(), json!("ai")),
        ]))
        .execute()?;
    for (i, r) in results.iter().enumerate() {
        println!(
            "  {}. [dist={:.4}] {} — {}",
            i + 1,
            r.distance,
            r.id,
            r.metadata["text"]
        );
    }

    Ok(())
}
