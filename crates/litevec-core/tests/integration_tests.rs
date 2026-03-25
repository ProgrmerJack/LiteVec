//! Integration tests for LiteVec core engine.

use litevec_core::{CollectionConfig, Database, DistanceType, Filter, IndexType};
use serde_json::json;

// ──────────────────────────────── Helpers ────────────────────────────────

fn make_memory_db() -> Database {
    Database::open_memory().unwrap()
}

fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
    // Simple deterministic pseudo-random using xorshift
    let mut state = seed.wrapping_add(1);
    (0..dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let bits = (state & 0xFFFF) as f32 / 65535.0;
            bits * 2.0 - 1.0
        })
        .collect()
}

#[allow(dead_code)]
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ──────────────────────────── Full Lifecycle ─────────────────────────────

#[test]
fn test_full_lifecycle_in_memory() {
    let db = make_memory_db();

    // Create collection
    let col = db.create_collection("documents", 64).unwrap();
    assert_eq!(col.name(), "documents");
    assert_eq!(col.dimension(), 64);
    assert!(col.is_empty());

    // Insert vectors
    for i in 0..50 {
        let v = random_vector(64, i);
        col.insert(
            &format!("doc_{i}"),
            &v,
            json!({"category": if i % 2 == 0 { "even" } else { "odd" }, "value": i}),
        )
        .unwrap();
    }
    assert_eq!(col.len(), 50);

    // Search
    let query = random_vector(64, 0); // same as doc_0
    let results = col.search(&query, 5).execute().unwrap();
    assert_eq!(results.len(), 5);
    assert_eq!(results[0].id, "doc_0"); // self-search → perfect match

    // Get
    let record = col.get("doc_0").unwrap().unwrap();
    assert_eq!(record.id, "doc_0");
    assert_eq!(record.metadata["category"], "even");

    // Update metadata
    col.update_metadata("doc_0", json!({"category": "updated"}))
        .unwrap();
    let record = col.get("doc_0").unwrap().unwrap();
    assert_eq!(record.metadata["category"], "updated");

    // Delete
    col.delete("doc_25").unwrap();
    assert_eq!(col.len(), 49);
    assert!(col.get("doc_25").unwrap().is_none());

    // Search still works after delete
    let results = col.search(&query, 5).execute().unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_file_backed_lifecycle() {
    let path = std::env::temp_dir().join(format!("litevec_integ_test_{}.lv", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let snap_path = path.with_extension("lv.snap");
    let _ = std::fs::remove_file(&snap_path);

    {
        let db = Database::open(&path).unwrap();
        let col = db.create_collection("test", 16).unwrap();
        for i in 0..10 {
            let v = random_vector(16, i);
            col.insert(&format!("v{i}"), &v, json!({"idx": i})).unwrap();
        }
        assert_eq!(col.len(), 10);
        db.close().unwrap();
    }

    // File should exist
    assert!(path.exists());

    // Reopen and verify persistence
    {
        let db = Database::open(&path).unwrap();
        let names = db.list_collections();
        assert_eq!(names, vec!["test"]);

        let col = db.get_collection("test").unwrap();
        assert_eq!(col.len(), 10);
        assert_eq!(col.dimension(), 16);

        // Verify we can get individual vectors
        let record = col.get("v0").unwrap();
        assert!(record.is_some());

        // Verify search still works after reload
        let query = random_vector(16, 0);
        let results = col.search(&query, 5).execute().unwrap();
        assert_eq!(results.len(), 5);
    }

    // Cleanup
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&snap_path);
    let wal_path = path.with_extension("lv.wal");
    let _ = std::fs::remove_file(&wal_path);
}

// ────────────────────────── Multiple Collections ─────────────────────────

#[test]
fn test_multiple_collections() {
    let db = make_memory_db();

    let col_a = db.create_collection("images", 512).unwrap();
    let col_b = db.create_collection("text", 384).unwrap();
    let col_c = db.create_collection("audio", 256).unwrap();

    assert_eq!(db.list_collections().len(), 3);

    // Insert into each
    col_a
        .insert("img1", &random_vector(512, 1), json!({}))
        .unwrap();
    col_b
        .insert("txt1", &random_vector(384, 1), json!({}))
        .unwrap();
    col_c
        .insert("aud1", &random_vector(256, 1), json!({}))
        .unwrap();

    // Each has its own data
    assert_eq!(col_a.len(), 1);
    assert_eq!(col_b.len(), 1);
    assert_eq!(col_c.len(), 1);

    // Delete one collection
    db.delete_collection("audio").unwrap();
    assert_eq!(db.list_collections().len(), 2);
    assert!(db.get_collection("audio").is_none());
}

// ───────────────────────────── Metadata Filter ───────────────────────────

#[test]
fn test_search_with_metadata_filter() {
    let db = make_memory_db();
    let col = db.create_collection("products", 8).unwrap();

    for i in 0..20 {
        let v = random_vector(8, i);
        col.insert(
            &format!("p{i}"),
            &v,
            json!({
                "price": (i as f64) * 10.0,
                "category": if i < 10 { "electronics" } else { "books" },
                "in_stock": i % 3 != 0,
            }),
        )
        .unwrap();
    }

    // Filter: category == "electronics"
    let results = col
        .search(&random_vector(8, 5), 20)
        .filter(Filter::Eq("category".into(), json!("electronics")))
        .execute()
        .unwrap();
    for r in &results {
        let rec = col.get(&r.id).unwrap().unwrap();
        assert_eq!(rec.metadata["category"], "electronics");
    }

    // Filter: price >= 100
    let results = col
        .search(&random_vector(8, 0), 20)
        .filter(Filter::Gte("price".into(), 100.0))
        .execute()
        .unwrap();
    for r in &results {
        let rec = col.get(&r.id).unwrap().unwrap();
        let price = rec.metadata["price"].as_f64().unwrap();
        assert!(price >= 100.0, "price {price} should be >= 100");
    }

    // Compound filter: AND(category == books, in_stock == true)
    let results = col
        .search(&random_vector(8, 0), 20)
        .filter(Filter::And(vec![
            Filter::Eq("category".into(), json!("books")),
            Filter::Eq("in_stock".into(), json!(true)),
        ]))
        .execute()
        .unwrap();
    for r in &results {
        let rec = col.get(&r.id).unwrap().unwrap();
        assert_eq!(rec.metadata["category"], "books");
        assert_eq!(rec.metadata["in_stock"], true);
    }
}

// ──────────────────────────── Distance Types ─────────────────────────────

#[test]
fn test_cosine_distance() {
    let db = make_memory_db();
    let config = CollectionConfig {
        dimension: 4,
        distance: DistanceType::Cosine,
        ..CollectionConfig::new(4)
    };
    let col = db
        .create_collection_with_config("cosine_test", config)
        .unwrap();

    // Insert vectors at known angles
    col.insert("north", &[0.0, 1.0, 0.0, 0.0], json!({}))
        .unwrap();
    col.insert("east", &[1.0, 0.0, 0.0, 0.0], json!({}))
        .unwrap();
    col.insert("northeast", &[0.707, 0.707, 0.0, 0.0], json!({}))
        .unwrap();

    // Query pointing north-ish
    let results = col.search(&[0.1, 0.9, 0.0, 0.0], 3).execute().unwrap();
    assert_eq!(results[0].id, "north");
    assert_eq!(results[1].id, "northeast");
}

#[test]
fn test_euclidean_distance() {
    let db = make_memory_db();
    let config = CollectionConfig {
        dimension: 3,
        distance: DistanceType::Euclidean,
        ..CollectionConfig::new(3)
    };
    let col = db
        .create_collection_with_config("euclidean_test", config)
        .unwrap();

    col.insert("origin", &[0.0, 0.0, 0.0], json!({})).unwrap();
    col.insert("near", &[0.1, 0.1, 0.1], json!({})).unwrap();
    col.insert("far", &[10.0, 10.0, 10.0], json!({})).unwrap();

    let results = col.search(&[0.0, 0.0, 0.0], 3).execute().unwrap();
    assert_eq!(results[0].id, "origin");
    assert_eq!(results[1].id, "near");
    assert_eq!(results[2].id, "far");
}

#[test]
fn test_dot_product_distance() {
    let db = make_memory_db();
    let config = CollectionConfig {
        dimension: 3,
        distance: DistanceType::DotProduct,
        ..CollectionConfig::new(3)
    };
    let col = db
        .create_collection_with_config("dot_test", config)
        .unwrap();

    col.insert("aligned", &[1.0, 1.0, 1.0], json!({})).unwrap();
    col.insert("opposite", &[-1.0, -1.0, -1.0], json!({}))
        .unwrap();
    col.insert("orthogonal", &[1.0, -1.0, 0.0], json!({}))
        .unwrap();

    // Query [1,1,1]: aligned should be closest (highest dot product)
    let results = col.search(&[1.0, 1.0, 1.0], 3).execute().unwrap();
    assert_eq!(results[0].id, "aligned");
    assert_eq!(results[2].id, "opposite");
}

// ──────────────────────────── Index Types ─────────────────────────────

#[test]
fn test_flat_index_type() {
    let db = make_memory_db();
    let config = CollectionConfig {
        dimension: 16,
        index: IndexType::Flat,
        ..CollectionConfig::new(16)
    };
    let col = db
        .create_collection_with_config("flat_col", config)
        .unwrap();

    for i in 0..30 {
        col.insert(&format!("v{i}"), &random_vector(16, i), json!({}))
            .unwrap();
    }

    let results = col.search(&random_vector(16, 5), 5).execute().unwrap();
    assert_eq!(results.len(), 5);
    assert_eq!(results[0].id, "v5");
}

#[test]
fn test_hnsw_index_type() {
    let db = make_memory_db();
    let config = CollectionConfig {
        dimension: 32,
        index: IndexType::Hnsw,
        ..CollectionConfig::new(32)
    };
    let col = db
        .create_collection_with_config("hnsw_col", config)
        .unwrap();

    for i in 0..100 {
        col.insert(&format!("v{i}"), &random_vector(32, i), json!({}))
            .unwrap();
    }

    let results = col.search(&random_vector(32, 42), 10).execute().unwrap();
    assert_eq!(results.len(), 10);
    // Self-search should still be the top result
    assert_eq!(results[0].id, "v42");
}

// ──────────────────────────── Edge Cases ─────────────────────────────

#[test]
fn test_empty_collection_search() {
    let db = make_memory_db();
    let col = db.create_collection("empty", 4).unwrap();

    let results = col.search(&[1.0, 0.0, 0.0, 0.0], 10).execute().unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_dimension_mismatch() {
    let db = make_memory_db();
    let col = db.create_collection("dim_test", 4).unwrap();

    // Wrong insert dimension
    let result = col.insert("v1", &[1.0, 2.0], json!({}));
    assert!(result.is_err());

    // Wrong search dimension
    col.insert("v1", &[1.0, 0.0, 0.0, 0.0], json!({})).unwrap();
    let result = col.search(&[1.0, 2.0], 5).execute();
    assert!(result.is_err());
}

#[test]
fn test_duplicate_id_upsert() {
    let db = make_memory_db();
    let col = db.create_collection("upsert_test", 3).unwrap();

    col.insert("v1", &[1.0, 0.0, 0.0], json!({"version": 1}))
        .unwrap();
    assert_eq!(col.len(), 1);

    // Re-insert with same ID → upsert
    col.insert("v1", &[0.0, 1.0, 0.0], json!({"version": 2}))
        .unwrap();
    assert_eq!(col.len(), 1);

    let record = col.get("v1").unwrap().unwrap();
    assert_eq!(record.metadata["version"], 2);
    assert_eq!(record.vector, vec![0.0, 1.0, 0.0]);
}

#[test]
fn test_delete_nonexistent() {
    let db = make_memory_db();
    let col = db.create_collection("test", 4).unwrap();

    // Deleting a non-existent vector should return Ok(false)
    let result = col.delete("nonexistent").unwrap();
    assert!(!result, "deleting nonexistent vector should return false");
}

// ────────────────────────── Batch Operations ─────────────────────────────

#[test]
fn test_batch_insert() {
    let db = make_memory_db();
    let col = db.create_collection("batch", 8).unwrap();

    let items: Vec<(&str, &[f32], serde_json::Value)> = vec![
        ("a", &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], json!({})),
        ("b", &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], json!({})),
        ("c", &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], json!({})),
    ];
    col.insert_batch(&items).unwrap();
    assert_eq!(col.len(), 3);
}

// ──────────────────────── Concurrent Access ──────────────────────────

#[test]
fn test_concurrent_insert_and_search() {
    let db = make_memory_db();
    let col = db.create_collection("concurrent", 16).unwrap();

    // Pre-populate
    for i in 0..50 {
        col.insert(&format!("init_{i}"), &random_vector(16, i), json!({}))
            .unwrap();
    }

    let mut handles = vec![];

    // Writers
    for t in 0..4 {
        let col = col.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0..25 {
                let id = format!("t{t}_v{i}");
                col.insert(&id, &random_vector(16, (t * 100 + i) as u64), json!({}))
                    .unwrap();
            }
        }));
    }

    // Readers
    for _ in 0..4 {
        let col = col.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0..25 {
                let q = random_vector(16, i + 1000);
                let _results = col.search(&q, 5).execute().unwrap();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(col.len(), 150); // 50 init + 4*25 inserts
}

// ──────────────────────── Recall Verification ─────────────────────────

#[test]
fn test_hnsw_recall_vs_flat() {
    let dimension = 64;
    let n = 200;

    // Build both flat and HNSW indexes with same data
    let db = make_memory_db();

    let flat_config = CollectionConfig {
        dimension: dimension as u32,
        index: IndexType::Flat,
        ..CollectionConfig::new(dimension as u32)
    };
    let hnsw_config = CollectionConfig {
        dimension: dimension as u32,
        index: IndexType::Hnsw,
        ..CollectionConfig::new(dimension as u32)
    };

    let flat_col = db
        .create_collection_with_config("flat", flat_config)
        .unwrap();
    let hnsw_col = db
        .create_collection_with_config("hnsw", hnsw_config)
        .unwrap();

    // Same data into both
    for i in 0..n {
        let v = random_vector(dimension, i);
        flat_col.insert(&format!("v{i}"), &v, json!({})).unwrap();
        hnsw_col.insert(&format!("v{i}"), &v, json!({})).unwrap();
    }

    // Run queries and measure recall
    let k = 10;
    let num_queries = 20;
    let mut total_recall = 0.0;

    for q_idx in 0..num_queries {
        let query = random_vector(dimension, 10000 + q_idx);

        let ground_truth: Vec<String> = flat_col
            .search(&query, k)
            .execute()
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();

        let hnsw_results: Vec<String> = hnsw_col
            .search(&query, k)
            .execute()
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();

        let matches = hnsw_results
            .iter()
            .filter(|id| ground_truth.contains(id))
            .count();

        total_recall += matches as f64 / k as f64;
    }

    let avg_recall = total_recall / num_queries as f64;
    assert!(
        avg_recall >= 0.85,
        "Average recall@{k} = {avg_recall:.3}, expected >= 0.85"
    );
}

// ──────────────────── Large-scale Stress Test ─────────────────────────

#[test]
fn test_large_collection() {
    let db = make_memory_db();
    let col = db.create_collection("large", 32).unwrap();

    // Insert 1000 vectors
    for i in 0..1000 {
        let v = random_vector(32, i);
        col.insert(&format!("v{i}"), &v, json!({"idx": i})).unwrap();
    }

    assert_eq!(col.len(), 1000);

    // Search should still work
    let results = col.search(&random_vector(32, 500), 10).execute().unwrap();
    assert_eq!(results.len(), 10);
    assert_eq!(results[0].id, "v500");

    // Delete some
    for i in 0..100 {
        col.delete(&format!("v{i}")).unwrap();
    }
    assert_eq!(col.len(), 900);

    // Search still returns correct results from remaining vectors
    let results = col.search(&random_vector(32, 500), 10).execute().unwrap();
    assert_eq!(results.len(), 10);
    assert_eq!(results[0].id, "v500");
}
