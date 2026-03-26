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

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn normalized_random_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = random_vector(dim, seed);
    normalize(&mut v);
    v
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

// ───────────────────── Crash Recovery (Snapshot) ─────────────────────

#[test]
fn test_crash_recovery_via_drop() {
    let path = std::env::temp_dir().join(format!("litevec_crash_{}.lv", std::process::id()));
    let snap_path = path.with_extension("lv.snap");
    let wal_path = path.with_extension("lv-wal");

    // Cleanup any leftovers
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&snap_path);
    let _ = std::fs::remove_file(&wal_path);

    // Phase 1: Open DB, insert data, drop without explicit close (simulates crash)
    {
        let db = Database::open(&path).unwrap();
        let col = db.create_collection("docs", 8).unwrap();
        for i in 0u64..20 {
            col.insert(
                &format!("d{i}"),
                &random_vector(8, i),
                json!({"idx": i, "tag": format!("item-{i}")}),
            )
            .unwrap();
        }
        assert_eq!(col.len(), 20);
        // Intentional: no db.close() — Drop impl should checkpoint
    }

    // The snapshot should have been written by Drop
    assert!(snap_path.exists(), "snapshot must exist after implicit drop");

    // Phase 2: Reopen — all data should be recovered from the snapshot
    {
        let db = Database::open(&path).unwrap();
        let names = db.list_collections();
        assert_eq!(names, vec!["docs"]);

        let col = db.get_collection("docs").unwrap();
        assert_eq!(col.len(), 20);
        assert_eq!(col.dimension(), 8);

        // Verify individual records survived
        for i in 0u64..20 {
            let rec = col.get(&format!("d{i}")).unwrap();
            assert!(rec.is_some(), "record d{i} must be recovered");
            let rec = rec.unwrap();
            assert_eq!(rec.metadata["idx"], i);
        }

        // Verify search still works after recovery
        let query = random_vector(8, 0);
        let results = col.search(&query, 5).execute().unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].id, "d0");

        db.close().unwrap();
    }

    // Cleanup
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&snap_path);
    let _ = std::fs::remove_file(&wal_path);
}

/// Verify data survives multiple open/drop cycles without explicit close.
#[test]
fn test_repeated_crash_recovery() {
    let path = std::env::temp_dir().join(format!("litevec_multi_crash_{}.lv", std::process::id()));
    let snap_path = path.with_extension("lv.snap");
    let wal_path = path.with_extension("lv-wal");

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&snap_path);
    let _ = std::fs::remove_file(&wal_path);

    // Round 1: create + insert
    {
        let db = Database::open(&path).unwrap();
        let col = db.create_collection("data", 4).unwrap();
        col.insert("a", &[1.0, 0.0, 0.0, 0.0], json!({"round": 1}))
            .unwrap();
        // drop without close
    }

    // Round 2: reopen, verify, add more, drop again
    {
        let db = Database::open(&path).unwrap();
        let col = db.get_collection("data").unwrap();
        assert_eq!(col.len(), 1);
        col.insert("b", &[0.0, 1.0, 0.0, 0.0], json!({"round": 2}))
            .unwrap();
        assert_eq!(col.len(), 2);
        // drop without close
    }

    // Round 3: reopen, verify both records survived both crashes
    {
        let db = Database::open(&path).unwrap();
        let col = db.get_collection("data").unwrap();
        assert_eq!(col.len(), 2);
        let a = col.get("a").unwrap().unwrap();
        assert_eq!(a.metadata["round"], 1);
        let b = col.get("b").unwrap().unwrap();
        assert_eq!(b.metadata["round"], 2);
        db.close().unwrap();
    }

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&snap_path);
    let _ = std::fs::remove_file(&wal_path);
}

// ──────────────────── Large Dataset + Recall ──────────────────────

/// Insert 10,000 vectors, measure HNSW recall@10 against brute-force ground truth.
#[test]
fn test_large_dataset_recall() {
    let n = 10_000u64;
    let dim = 32;
    let k = 10;
    let num_queries = 50;
    let db = make_memory_db();

    let flat_config = CollectionConfig {
        dimension: dim as u32,
        index: IndexType::Flat,
        ..CollectionConfig::new(dim as u32)
    };
    let hnsw_config = CollectionConfig {
        dimension: dim as u32,
        index: IndexType::Hnsw,
        ..CollectionConfig::new(dim as u32)
    };

    let flat_col = db
        .create_collection_with_config("flat_large", flat_config)
        .unwrap();
    let hnsw_col = db
        .create_collection_with_config("hnsw_large", hnsw_config)
        .unwrap();

    // Insert same data into both
    for i in 0..n {
        let v = normalized_random_vector(dim, i);
        flat_col.insert(&format!("v{i}"), &v, json!({})).unwrap();
        hnsw_col.insert(&format!("v{i}"), &v, json!({})).unwrap();
    }

    assert_eq!(flat_col.len(), n as usize);
    assert_eq!(hnsw_col.len(), n as usize);

    // Measure recall
    let mut total_recall = 0.0;
    for q in 0..num_queries {
        let query = normalized_random_vector(dim, 100_000 + q);

        let truth: Vec<String> = flat_col
            .search(&query, k)
            .execute()
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();

        let approx: Vec<String> = hnsw_col
            .search(&query, k)
            .execute()
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();

        let hits = approx.iter().filter(|id| truth.contains(id)).count();
        total_recall += hits as f64 / k as f64;
    }

    let avg_recall = total_recall / num_queries as f64;
    assert!(
        avg_recall >= 0.90,
        "HNSW recall@{k} on {n} vectors = {avg_recall:.3}, expected >= 0.90"
    );
}

// ───────────── Concurrent File-Backed Access ─────────────────────

#[test]
fn test_concurrent_file_backed() {
    let path =
        std::env::temp_dir().join(format!("litevec_concurrent_{}.lv", std::process::id()));
    let snap_path = path.with_extension("lv.snap");
    let wal_path = path.with_extension("lv-wal");

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&snap_path);
    let _ = std::fs::remove_file(&wal_path);

    let db = Database::open(&path).unwrap();
    let col = db.create_collection("shared", 16).unwrap();

    // Seed with initial data
    for i in 0..100 {
        col.insert(&format!("init_{i}"), &random_vector(16, i), json!({}))
            .unwrap();
    }

    let mut handles = vec![];

    // 2 writer threads
    for t in 0..2 {
        let col = col.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0..50 {
                let id = format!("w{t}_{i}");
                col.insert(&id, &random_vector(16, (t * 1000 + i) as u64), json!({}))
                    .unwrap();
            }
        }));
    }

    // 4 reader threads
    for _ in 0..4 {
        let col = col.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0..50 {
                let q = random_vector(16, i + 5000);
                let results = col.search(&q, 5).execute().unwrap();
                assert!(!results.is_empty() || col.is_empty());
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // 100 initial + 2*50 written = 200
    assert_eq!(col.len(), 200);

    // Close and reopen to verify persistence
    db.close().unwrap();

    let db2 = Database::open(&path).unwrap();
    let col2 = db2.get_collection("shared").unwrap();
    assert_eq!(col2.len(), 200);
    db2.close().unwrap();

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&snap_path);
    let _ = std::fs::remove_file(&wal_path);
}

// ─────────────────── All Filter Operators ─────────────────────────

#[test]
fn test_all_filter_operators() {
    let db = make_memory_db();
    let col = db.create_collection("filters", 4).unwrap();

    // Insert 30 items with diverse metadata
    for i in 0u64..30 {
        col.insert(
            &format!("item_{i}"),
            &random_vector(4, i),
            json!({
                "score": i as f64,
                "tier": match i % 3 { 0 => "gold", 1 => "silver", _ => "bronze" },
                "active": i % 2 == 0,
            }),
        )
        .unwrap();
    }

    let q = random_vector(4, 999);

    // Eq
    let results = col
        .search(&q, 30)
        .filter(Filter::Eq("tier".into(), json!("gold")))
        .execute()
        .unwrap();
    assert!(results.iter().all(|r| {
        col.get(&r.id).unwrap().unwrap().metadata["tier"] == "gold"
    }));
    assert_eq!(results.len(), 10); // 0,3,6,...,27

    // Ne
    let results = col
        .search(&q, 30)
        .filter(Filter::Ne("tier".into(), json!("gold")))
        .execute()
        .unwrap();
    assert!(results.iter().all(|r| {
        col.get(&r.id).unwrap().unwrap().metadata["tier"] != "gold"
    }));
    assert_eq!(results.len(), 20);

    // Gt
    let results = col
        .search(&q, 30)
        .filter(Filter::Gt("score".into(), 25.0))
        .execute()
        .unwrap();
    assert!(results.iter().all(|r| {
        col.get(&r.id).unwrap().unwrap().metadata["score"]
            .as_f64()
            .unwrap()
            > 25.0
    }));

    // Lt
    let results = col
        .search(&q, 30)
        .filter(Filter::Lt("score".into(), 5.0))
        .execute()
        .unwrap();
    assert!(results.iter().all(|r| {
        col.get(&r.id).unwrap().unwrap().metadata["score"]
            .as_f64()
            .unwrap()
            < 5.0
    }));

    // Gte
    let results = col
        .search(&q, 30)
        .filter(Filter::Gte("score".into(), 28.0))
        .execute()
        .unwrap();
    assert_eq!(results.len(), 2); // 28, 29

    // Lte
    let results = col
        .search(&q, 30)
        .filter(Filter::Lte("score".into(), 1.0))
        .execute()
        .unwrap();
    assert_eq!(results.len(), 2); // 0, 1

    // In
    let results = col
        .search(&q, 30)
        .filter(Filter::In(
            "tier".into(),
            vec![json!("gold"), json!("silver")],
        ))
        .execute()
        .unwrap();
    assert_eq!(results.len(), 20); // gold(10) + silver(10)

    // Or
    let results = col
        .search(&q, 30)
        .filter(Filter::Or(vec![
            Filter::Eq("tier".into(), json!("bronze")),
            Filter::Gt("score".into(), 27.0),
        ]))
        .execute()
        .unwrap();
    assert!(!results.is_empty());

    // Not
    let results = col
        .search(&q, 30)
        .filter(Filter::Not(Box::new(Filter::Eq(
            "active".into(),
            json!(true),
        ))))
        .execute()
        .unwrap();
    assert!(results.iter().all(|r| {
        col.get(&r.id).unwrap().unwrap().metadata["active"] == false
    }));

    // And (compound)
    let results = col
        .search(&q, 30)
        .filter(Filter::And(vec![
            Filter::Eq("tier".into(), json!("gold")),
            Filter::Gte("score".into(), 15.0),
        ]))
        .execute()
        .unwrap();
    assert!(results.iter().all(|r| {
        let m = &col.get(&r.id).unwrap().unwrap().metadata;
        m["tier"] == "gold" && m["score"].as_f64().unwrap() >= 15.0
    }));
}

// ──────────────────── Full-Text + Hybrid Search ──────────────────────

#[test]
fn test_fulltext_and_hybrid_search_integration() {
    let db = make_memory_db();
    let col = db.create_collection("articles", 4).unwrap();

    let docs = [
        ("d1", "Rust programming language systems", [1.0, 0.0, 0.0, 0.0]),
        ("d2", "Python machine learning data science", [0.0, 1.0, 0.0, 0.0]),
        ("d3", "Rust web framework actix tower", [0.9, 0.1, 0.0, 0.0]),
        ("d4", "JavaScript React frontend development", [0.0, 0.0, 1.0, 0.0]),
        ("d5", "Rust embedded systems microcontroller", [0.8, 0.0, 0.2, 0.0]),
    ];

    for (id, text, vec) in &docs {
        col.insert(id, vec, json!({"text": text})).unwrap();
    }

    // Pure text search — should find Rust-related docs
    let results = col.text_search("Rust programming", 5);
    assert!(!results.is_empty());
    // The top result should mention Rust
    let top = col.get(&results[0].id).unwrap().unwrap();
    let text = top.metadata["text"].as_str().unwrap();
    assert!(text.contains("Rust"), "top text search result should contain 'Rust'");

    // Hybrid search — combines vector similarity with keyword relevance
    let results = col.hybrid_search(&[0.95, 0.05, 0.0, 0.0], "Rust systems", 3).unwrap();
    assert!(!results.is_empty());
    // d1 or d5 should rank high (both Rust + systems related)
    let top_ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        top_ids.contains(&"d1") || top_ids.contains(&"d5"),
        "hybrid search should surface Rust systems docs, got: {top_ids:?}"
    );
}

// ──────────────────── Backup & Restore ──────────────────────

#[test]
fn test_backup_and_restore() {
    let src_path =
        std::env::temp_dir().join(format!("litevec_backup_src_{}.lv", std::process::id()));
    let backup_path =
        std::env::temp_dir().join(format!("litevec_backup_dst_{}.lv", std::process::id()));

    // Clean up
    for p in [&src_path, &backup_path] {
        let _ = std::fs::remove_file(p);
        let _ = std::fs::remove_file(p.with_extension("lv.snap"));
        let _ = std::fs::remove_file(p.with_extension("lv-wal"));
    }

    // Create source DB with data
    {
        let db = Database::open(&src_path).unwrap();
        let col = db.create_collection("important", 8).unwrap();
        for i in 0..50 {
            col.insert(
                &format!("rec_{i}"),
                &random_vector(8, i),
                json!({"value": i}),
            )
            .unwrap();
        }
        db.create_backup(&backup_path).unwrap();
        db.close().unwrap();
    }

    // Restore from backup into a new DB
    {
        let db = Database::restore_from_backup(&backup_path).unwrap();
        let col = db.get_collection("important").unwrap();
        assert_eq!(col.len(), 50);

        // Verify data integrity
        let rec = col.get("rec_0").unwrap().unwrap();
        assert_eq!(rec.metadata["value"], 0);

        // Verify search works on restored data
        let results = col.search(&random_vector(8, 25), 5).execute().unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].id, "rec_25");
    }

    // Check backup_info
    let info = Database::backup_info(&backup_path);
    assert!(info.is_ok());

    // Clean up
    for p in [&src_path, &backup_path] {
        let _ = std::fs::remove_file(p);
        let _ = std::fs::remove_file(p.with_extension("lv.snap"));
        let _ = std::fs::remove_file(p.with_extension("lv-wal"));
    }
}
