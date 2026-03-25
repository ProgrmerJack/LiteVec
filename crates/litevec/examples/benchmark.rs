//! LiteVec benchmark — insert throughput and search latency.
//!
//! Run with: cargo run --release --example benchmark

use litevec::{CollectionConfig, Database, DistanceType, IndexType};
use serde_json::json;
use std::time::Instant;

fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    for _ in 0..dim {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        v.push(((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0);
    }
    v
}

fn main() {
    let dimension = 128;
    let counts = [1_000, 10_000, 50_000];

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              LiteVec Benchmark Suite (dim={dimension})              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!();

    for &n in &counts {
        println!("━━━ {n} vectors (dim={dimension}) ━━━");
        println!();

        // --- HNSW Index ---
        {
            let db = Database::open_memory().unwrap();
            let mut config = CollectionConfig::new(dimension as u32);
            config.index = IndexType::Hnsw;
            config.distance = DistanceType::Cosine;
            let col = db.create_collection_with_config("bench", config).unwrap();

            // Insert benchmark
            let start = Instant::now();
            for i in 0..n {
                let v = random_vector(dimension, i as u64);
                col.insert(&format!("v{i}"), &v, json!({})).unwrap();
            }
            let insert_elapsed = start.elapsed();
            let insert_rate = n as f64 / insert_elapsed.as_secs_f64();

            // Search benchmark (100 queries)
            let num_queries = 100;
            let k = 10;
            let queries: Vec<Vec<f32>> = (0..num_queries)
                .map(|i| random_vector(dimension, 1_000_000 + i))
                .collect();

            let start = Instant::now();
            for q in &queries {
                let _results = col.search(q, k).execute().unwrap();
            }
            let search_elapsed = start.elapsed();
            let avg_search_us = search_elapsed.as_micros() as f64 / num_queries as f64;
            let qps = num_queries as f64 / search_elapsed.as_secs_f64();

            println!("  HNSW Index:");
            println!(
                "    Insert:  {n} vectors in {:.1}ms ({insert_rate:.0} vec/s)",
                insert_elapsed.as_secs_f64() * 1000.0
            );
            println!(
                "    Search:  {num_queries} queries, k={k}, avg {avg_search_us:.0}μs/query ({qps:.0} QPS)"
            );
        }

        // --- Flat Index ---
        if n <= 10_000 {
            let db = Database::open_memory().unwrap();
            let mut config = CollectionConfig::new(dimension as u32);
            config.index = IndexType::Flat;
            config.distance = DistanceType::Cosine;
            let col = db
                .create_collection_with_config("bench_flat", config)
                .unwrap();

            let start = Instant::now();
            for i in 0..n {
                let v = random_vector(dimension, i as u64);
                col.insert(&format!("v{i}"), &v, json!({})).unwrap();
            }
            let insert_elapsed = start.elapsed();

            let num_queries = 100;
            let k = 10;
            let queries: Vec<Vec<f32>> = (0..num_queries)
                .map(|i| random_vector(dimension, 1_000_000 + i))
                .collect();

            let start = Instant::now();
            for q in &queries {
                let _results = col.search(q, k).execute().unwrap();
            }
            let search_elapsed = start.elapsed();
            let avg_search_us = search_elapsed.as_micros() as f64 / num_queries as f64;
            let qps = num_queries as f64 / search_elapsed.as_secs_f64();

            println!("  Flat Index (exact):");
            println!(
                "    Insert:  {n} vectors in {:.1}ms",
                insert_elapsed.as_secs_f64() * 1000.0
            );
            println!(
                "    Search:  {num_queries} queries, k={k}, avg {avg_search_us:.0}μs/query ({qps:.0} QPS)"
            );
        }

        println!();
    }

    // --- Persistence benchmark ---
    println!("━━━ Persistence (10,000 vectors) ━━━");
    {
        let path = std::env::temp_dir().join("litevec_bench.lv");
        let _ = std::fs::remove_file(&path);
        let snap_path = path.with_extension("lv.snap");
        let _ = std::fs::remove_file(&snap_path);

        let db = Database::open(&path).unwrap();
        let col = db.create_collection("bench", dimension as u32).unwrap();
        for i in 0..10_000 {
            let v = random_vector(dimension, i as u64);
            col.insert(&format!("v{i}"), &v, json!({"idx": i})).unwrap();
        }

        let start = Instant::now();
        db.close().unwrap();
        let save_elapsed = start.elapsed();

        let start = Instant::now();
        let db2 = Database::open(&path).unwrap();
        let load_elapsed = start.elapsed();
        let col2 = db2.get_collection("bench").unwrap();
        assert_eq!(col2.len(), 10_000);

        println!("  Save:  {:.1}ms", save_elapsed.as_secs_f64() * 1000.0);
        println!(
            "  Load:  {:.1}ms (including index rebuild)",
            load_elapsed.as_secs_f64() * 1000.0
        );

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&snap_path);
    }

    println!();
    println!("╚══════════════════════════════════════════════════════════════╝");
}
