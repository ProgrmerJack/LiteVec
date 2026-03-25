//! Backup and Restore Example
//!
//! Demonstrates LiteVec's snapshot backup and restore functionality.
//!
//! Run with: cargo run --example backup_restore

use litevec::Database;
use serde_json::json;

fn main() -> litevec::Result<()> {
    let backup_path = std::env::temp_dir().join("litevec_backup_example.snap");

    // Create and populate a database
    println!("--- Creating database with sample data ---");
    {
        let db = Database::open_memory()?;
        let col = db.create_collection("embeddings", 4)?;

        for i in 0..100 {
            let vector = vec![
                (i as f32 / 100.0),
                ((100 - i) as f32 / 100.0),
                ((i * 7 % 100) as f32 / 100.0),
                ((i * 13 % 100) as f32 / 100.0),
            ];
            col.insert(
                &format!("vec_{i}"),
                &vector,
                json!({"index": i, "batch": i / 10}),
            )?;
        }

        println!("  Inserted {} vectors", col.len());

        // Create a backup
        db.create_backup(&backup_path)?;
        println!("  Backup created at: {}", backup_path.display());
    }

    // Inspect the backup
    println!("\n--- Backup Info ---");
    let info = Database::backup_info(&backup_path)?;
    println!("  Version: {}", info.version);
    println!("  Collections: {}", info.num_collections);
    println!("  Total vectors: {}", info.total_vectors);
    for col_info in &info.collections {
        println!(
            "    {} — {} vectors ({}d)",
            col_info.name, col_info.num_vectors, col_info.dimension
        );
    }

    // Restore from backup
    println!("\n--- Restoring from backup ---");
    let restored_db = Database::restore_from_backup(&backup_path)?;
    let col = restored_db.get_collection("embeddings").unwrap();

    println!("  Restored {} vectors", col.len());

    // Verify data
    let record = col.get("vec_42").unwrap().unwrap();
    println!("  vec_42 metadata: {}", record.metadata);

    // Search works on restored data
    let results = col.search(&[0.5, 0.5, 0.5, 0.5], 3).execute()?;
    println!("\n--- Search on restored data ---");
    for r in &results {
        println!("  {} (distance: {:.4})", r.id, r.distance);
    }

    // Cleanup
    let _ = std::fs::remove_file(&backup_path);
    println!("\n✅ Backup and restore complete!");
    Ok(())
}
