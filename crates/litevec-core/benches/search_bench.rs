//! Criterion benchmarks for insert and search operations.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use litevec_core::Collection;
use litevec_core::types::{CollectionConfig, DistanceType, HnswConfig, IndexType};
use serde_json::json;

fn random_vector(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| ((seed * 7 + j * 13) as f32 * 0.1).sin())
        .collect()
}

fn build_collection(name: &str, dim: u32, n: usize, index_type: IndexType) -> Collection {
    let config = CollectionConfig {
        dimension: dim,
        distance: DistanceType::Cosine,
        index: index_type,
        hnsw: HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
        },
    };
    let col = Collection::new(name, config);
    for i in 0..n {
        let v = random_vector(dim as usize, i);
        col.insert(&format!("v{i}"), &v, json!({"i": i})).unwrap();
    }
    col
}

fn bench_insert_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_hnsw");
    group.sample_size(10);

    for &n in &[1000u64, 5000] {
        group.bench_with_input(BenchmarkId::new("128d", n), &n, |bench, &n| {
            bench.iter(|| {
                let config = CollectionConfig {
                    dimension: 128,
                    distance: DistanceType::Cosine,
                    index: IndexType::Hnsw,
                    hnsw: HnswConfig {
                        m: 16,
                        ef_construction: 200,
                        ef_search: 100,
                    },
                };
                let col = Collection::new("bench", config);
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

fn bench_insert_flat(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_flat");
    group.sample_size(10);

    for &n in &[1000u64, 10_000] {
        group.bench_with_input(BenchmarkId::new("128d", n), &n, |bench, &n| {
            bench.iter(|| {
                let config = CollectionConfig {
                    dimension: 128,
                    distance: DistanceType::Cosine,
                    index: IndexType::Flat,
                    hnsw: HnswConfig::default(),
                };
                let col = Collection::new("bench", config);
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

fn bench_search_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_hnsw");

    for &(n, dim) in &[(1000, 128), (5000, 128), (10000, 128), (1000, 384)] {
        let col = build_collection("bench", dim, n, IndexType::Hnsw);
        let query = random_vector(dim as usize, 999_999);

        group.bench_with_input(BenchmarkId::new(format!("{dim}d"), n), &n, |bench, _| {
            bench.iter(|| {
                let results = col.search(black_box(&query), 10).execute().unwrap();
                black_box(results);
            });
        });
    }
    group.finish();
}

fn bench_search_flat(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_flat");

    for &(n, dim) in &[(1000, 128), (5000, 128), (1000, 384)] {
        let col = build_collection("bench", dim, n, IndexType::Flat);
        let query = random_vector(dim as usize, 999_999);

        group.bench_with_input(BenchmarkId::new(format!("{dim}d"), n), &n, |bench, _| {
            bench.iter(|| {
                let results = col.search(black_box(&query), 10).execute().unwrap();
                black_box(results);
            });
        });
    }
    group.finish();
}

fn bench_search_with_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_filtered");

    let n = 5000;
    let dim = 128u32;
    let col = build_collection("bench", dim, n, IndexType::Hnsw);
    let query = random_vector(dim as usize, 999_999);

    group.bench_function("hnsw_5000_128d_eq_filter", |bench| {
        bench.iter(|| {
            let results = col
                .search(black_box(&query), 10)
                .filter(litevec_core::Filter::Lt {
                    field: "i".to_string(),
                    value: json!(2500),
                })
                .execute()
                .unwrap();
            black_box(results);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_insert_hnsw,
    bench_insert_flat,
    bench_search_hnsw,
    bench_search_flat,
    bench_search_with_filter,
);
criterion_main!(benches);
