//! Criterion benchmarks for distance functions.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use litevec_core::distance::DistanceFn;
use litevec_core::distance::cosine::{CosineDistance, cosine_distance_scalar};
use litevec_core::distance::dot::{DotProductDistance, dot_product_neg_scalar};
use litevec_core::distance::euclidean::{EuclideanDistance, euclidean_distance_sq_scalar};

fn random_vectors(dim: usize, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * 7 + j * 13) as f32 * 0.1).sin())
                .collect()
        })
        .collect()
}

fn bench_cosine_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_scalar");
    for dim in [128, 384, 768, 1536] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| cosine_distance_scalar(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_cosine_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_dispatch");
    let dist = CosineDistance::new();
    for dim in [128, 384, 768, 1536] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| dist.compute(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_euclidean_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_scalar");
    for dim in [128, 384, 768, 1536] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| euclidean_distance_sq_scalar(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_euclidean_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_dispatch");
    let dist = EuclideanDistance::new();
    for dim in [128, 384, 768, 1536] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| dist.compute(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_dot_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_scalar");
    for dim in [128, 384, 768, 1536] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| dot_product_neg_scalar(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_dot_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_dispatch");
    let dist = DotProductDistance::new();
    for dim in [128, 384, 768, 1536] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| dist.compute(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_throughput");
    let dim = 128;
    let vectors = random_vectors(dim, 1000);
    let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).sin()).collect();
    let dist = CosineDistance::new();

    group.bench_function("1000_cosine_128d", |bench| {
        bench.iter(|| {
            for v in &vectors {
                black_box(dist.compute(&query, v));
            }
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_scalar,
    bench_cosine_dispatch,
    bench_euclidean_scalar,
    bench_euclidean_dispatch,
    bench_dot_scalar,
    bench_dot_dispatch,
    bench_throughput,
);
criterion_main!(benches);
