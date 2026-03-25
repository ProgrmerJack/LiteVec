//! LiteVec Quickstart Example
//!
//! Run with: cargo run --example quickstart

use litevec::{Database, Filter};
use serde_json::json;

fn main() -> litevec::Result<()> {
    // Open an in-memory database (or use Database::open("my.lv") for persistence)
    let db = Database::open_memory()?;

    // Create a collection with 3-dimensional vectors
    let col = db.create_collection("documents", 3)?;

    // Insert vectors with metadata
    col.insert(
        "doc1",
        &[1.0, 0.0, 0.0],
        json!({"title": "Introduction to AI", "category": "tech"}),
    )?;
    col.insert(
        "doc2",
        &[0.0, 1.0, 0.0],
        json!({"title": "Machine Learning 101", "category": "tech"}),
    )?;
    col.insert(
        "doc3",
        &[0.0, 0.0, 1.0],
        json!({"title": "Cooking with Rust", "category": "food"}),
    )?;
    col.insert(
        "doc4",
        &[0.7, 0.7, 0.0],
        json!({"title": "Neural Networks Deep Dive", "category": "tech"}),
    )?;

    println!("Inserted {} vectors", col.len());

    // Search for the 3 nearest neighbors
    println!("\n--- Unfiltered Search ---");
    let results = col.search(&[0.9, 0.1, 0.0], 3).execute()?;
    for r in &results {
        println!(
            "  {} (distance: {:.4}) — {}",
            r.id, r.distance, r.metadata["title"]
        );
    }

    // Search with metadata filter
    println!("\n--- Filtered Search (category=tech) ---");
    let filtered = col
        .search(&[0.9, 0.1, 0.0], 3)
        .filter(Filter::Eq("category".into(), json!("tech")))
        .execute()?;
    for r in &filtered {
        println!(
            "  {} (distance: {:.4}) — {}",
            r.id, r.distance, r.metadata["title"]
        );
    }

    // Get a specific vector
    if let Some(record) = col.get("doc1")? {
        println!("\n--- Get doc1 ---");
        println!("  Vector: {:?}", record.vector);
        println!("  Metadata: {}", record.metadata);
    }

    // Delete a vector
    col.delete("doc3")?;
    println!("\nAfter delete: {} vectors remaining", col.len());

    // List collections
    println!("\nCollections: {:?}", db.list_collections());

    println!("\n✅ LiteVec quickstart complete!");
    Ok(())
}
