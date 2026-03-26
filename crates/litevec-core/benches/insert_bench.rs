//! Criterion benchmarks for insert operations.
//!
//! Measures insert throughput for various collection sizes, index types,
//! and vector dimensions.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use litevec_core::Database;
use litevec_core::types::{CollectionConfig, DistanceType, HnswConfig, IndexType};
use serde_json::json;

fn random_vector(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| ((seed * 7 + j * 13) as f32 * 0.1).sin())
        .collect()
}

fn bench_single_insert_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_insert_hnsw");
    group.sample_size(10);

    for &n in &[1_000u64, 5_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("128d", n), &n, |bench, &n| {
            bench.iter(|| {
                let db = Database::open_memory().unwrap();
                let config = CollectionConfig {
                    dimension: 128,
                    distance: DistanceType::Cosine,
                    index: IndexType::Hnsw,
                    hnsw: HnswConfig::default(),
                };
                let col = db.create_collection_with_config("bench", config).unwrap();
                for i in 0..n as usize {
                    let v = random_vector(128, i);
                    col.insert(&format!("v{i}"), &v, json!({})).unwrap();
                }
                black_box(&col);
            });
        });
    }
    group.finish();
}

fn bench_single_insert_flat(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_insert_flat");
    group.sample_size(10);

    for &n in &[1_000u64, 10_000, 50_000] {
        group.bench_with_input(BenchmarkId::new("128d", n), &n, |bench, &n| {
            bench.iter(|| {
                let db = Database::open_memory().unwrap();
                let config = CollectionConfig {
                    dimension: 128,
                    distance: DistanceType::Cosine,
                    index: IndexType::Flat,
                    hnsw: HnswConfig::default(),
                };
                let col = db.create_collection_with_config("bench", config).unwrap();
                for i in 0..n as usize {
                    let v = random_vector(128, i);
                    col.insert(&format!("v{i}"), &v, json!({})).unwrap();
                }
                black_box(&col);
            });
        });
    }
    group.finish();
}

fn bench_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert");
    group.sample_size(10);

    for &n in &[1_000u64, 5_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("128d_hnsw", n), &n, |bench, &n| {
            let items: Vec<(String, Vec<f32>)> = (0..n as usize)
                .map(|i| (format!("v{i}"), random_vector(128, i)))
                .collect();

            bench.iter(|| {
                let db = Database::open_memory().unwrap();
                let config = CollectionConfig {
                    dimension: 128,
                    distance: DistanceType::Cosine,
                    index: IndexType::Hnsw,
                    hnsw: HnswConfig::default(),
                };
                let col = db.create_collection_with_config("bench", config).unwrap();
                let batch: Vec<(&str, &[f32], serde_json::Value)> = items
                    .iter()
                    .map(|(id, v)| (id.as_str(), v.as_slice(), json!({})))
                    .collect();
                col.insert_batch(&batch).unwrap();
                black_box(&col);
            });
        });
    }
    group.finish();
}

fn bench_insert_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_dimensions");
    group.sample_size(10);

    for &dim in &[128u32, 384, 768, 1536] {
        group.bench_with_input(BenchmarkId::new("1000_hnsw", dim), &dim, |bench, &dim| {
            bench.iter(|| {
                let db = Database::open_memory().unwrap();
                let config = CollectionConfig {
                    dimension: dim,
                    distance: DistanceType::Cosine,
                    index: IndexType::Hnsw,
                    hnsw: HnswConfig::default(),
                };
                let col = db.create_collection_with_config("bench", config).unwrap();
                for i in 0..1000 {
                    let v = random_vector(dim as usize, i);
                    col.insert(&format!("v{i}"), &v, json!({})).unwrap();
                }
                black_box(&col);
            });
        });
    }
    group.finish();
}

fn bench_insert_with_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_with_metadata");
    group.sample_size(10);

    let n = 5_000usize;
    group.bench_function("5000_128d_hnsw", |bench| {
        bench.iter(|| {
            let db = Database::open_memory().unwrap();
            let config = CollectionConfig {
                dimension: 128,
                distance: DistanceType::Cosine,
                index: IndexType::Hnsw,
                hnsw: HnswConfig::default(),
            };
            let col = db.create_collection_with_config("bench", config).unwrap();
            for i in 0..n {
                let v = random_vector(128, i);
                col.insert(
                    &format!("v{i}"),
                    &v,
                    json!({"title": format!("Doc {i}"), "category": "test", "score": i as f64 * 0.1}),
                )
                .unwrap();
            }
            black_box(&col);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_single_insert_hnsw,
    bench_single_insert_flat,
    bench_batch_insert,
    bench_insert_dimensions,
    bench_insert_with_metadata,
);
criterion_main!(benches);
