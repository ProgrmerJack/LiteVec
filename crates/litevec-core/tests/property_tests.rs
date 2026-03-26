//! Property-based tests for LiteVec using proptest.
//!
//! These tests verify invariants that must hold for *any* input,
//! not just the specific examples in the integration tests.

use litevec_core::{Database, DistanceType, CollectionConfig};
use proptest::prelude::*;
use serde_json::json;

// ────────────────────── Strategies ──────────────────────────

fn arb_vector(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-1.0f32..1.0f32, dim)
}

fn arb_nonzero_vector(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    arb_vector(dim).prop_filter("vector must be non-zero", |v| {
        v.iter().any(|x| x.abs() > 1e-10)
    })
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

// ──────────────── Distance Function Properties ──────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Cosine distance is always in [0, 2] for any two vectors.
    #[test]
    fn cosine_distance_bounded(
        a in arb_nonzero_vector(16),
        b in arb_nonzero_vector(16),
    ) {
        let db = Database::open_memory().unwrap();
        let config = CollectionConfig {
            dimension: 16,
            distance: DistanceType::Cosine,
            ..CollectionConfig::new(16)
        };
        let col = db.create_collection_with_config("test", config).unwrap();
        col.insert("a", &a, json!({})).unwrap();

        let results = col.search(&b, 1).execute().unwrap();
        prop_assert!(!results.is_empty());
        let d = results[0].distance;
        prop_assert!(d >= -0.01, "cosine distance should be >= 0, got {d}");
        prop_assert!(d <= 2.01, "cosine distance should be <= 2, got {d}");
    }

    /// Euclidean distance is always non-negative.
    #[test]
    fn euclidean_distance_non_negative(
        a in arb_vector(16),
        b in arb_vector(16),
    ) {
        let db = Database::open_memory().unwrap();
        let config = CollectionConfig {
            dimension: 16,
            distance: DistanceType::Euclidean,
            ..CollectionConfig::new(16)
        };
        let col = db.create_collection_with_config("test", config).unwrap();
        col.insert("a", &a, json!({})).unwrap();

        let results = col.search(&b, 1).execute().unwrap();
        prop_assert!(!results.is_empty());
        prop_assert!(
            results[0].distance >= -0.001,
            "euclidean distance must be non-negative, got {}",
            results[0].distance
        );
    }

    /// Searching for a vector that was just inserted should return it as the top-1 result.
    #[test]
    fn insert_then_search_finds_self(v in arb_nonzero_vector(8)) {
        let db = Database::open_memory().unwrap();
        let col = db.create_collection("test", 8).unwrap();

        // Insert the target plus some noise
        col.insert("target", &v, json!({})).unwrap();
        for i in 0..20u64 {
            let noise: Vec<f32> = v.iter().enumerate()
                .map(|(j, x)| x + ((i * 31 + j as u64) % 97) as f32 / 50.0 - 1.0)
                .collect();
            col.insert(&format!("noise_{i}"), &noise, json!({})).unwrap();
        }

        let results = col.search(&v, 1).execute().unwrap();
        prop_assert_eq!(&results[0].id, "target");
    }

    /// Insert → get returns the same vector (within f32 precision).
    #[test]
    fn insert_get_roundtrip(v in arb_vector(8)) {
        let db = Database::open_memory().unwrap();
        let col = db.create_collection("test", 8).unwrap();
        col.insert("x", &v, json!({"hello": "world"})).unwrap();

        let record = col.get("x").unwrap().unwrap();
        for (a, b) in record.vector.iter().zip(v.iter()) {
            prop_assert!((a - b).abs() < 1e-6, "vector mismatch: {a} != {b}");
        }
        prop_assert_eq!(&record.metadata["hello"], "world");
    }

    /// Delete → get returns None.
    #[test]
    fn delete_then_get_returns_none(v in arb_vector(4)) {
        let db = Database::open_memory().unwrap();
        let col = db.create_collection("test", 4).unwrap();
        col.insert("ephemeral", &v, json!({})).unwrap();
        prop_assert!(col.get("ephemeral").unwrap().is_some());

        col.delete("ephemeral").unwrap();
        prop_assert!(col.get("ephemeral").unwrap().is_none());
    }

    /// Identical vectors have ~0 cosine distance.
    #[test]
    fn identical_vectors_zero_cosine_distance(v in arb_nonzero_vector(16)) {
        let nv = normalize(&v);
        let db = Database::open_memory().unwrap();
        let config = CollectionConfig {
            dimension: 16,
            distance: DistanceType::Cosine,
            ..CollectionConfig::new(16)
        };
        let col = db.create_collection_with_config("test", config).unwrap();
        col.insert("v", &nv, json!({})).unwrap();

        let results = col.search(&nv, 1).execute().unwrap();
        prop_assert!(
            results[0].distance < 0.01,
            "identical normalized vectors should have ~0 cosine distance, got {}",
            results[0].distance
        );
    }

    /// Collection length equals number of unique inserts minus deletes.
    #[test]
    fn len_tracks_inserts_and_deletes(
        n_insert in 5u64..30,
        n_delete in 0u64..5,
    ) {
        let db = Database::open_memory().unwrap();
        let col = db.create_collection("test", 4).unwrap();

        for i in 0..n_insert {
            let v: Vec<f32> = vec![i as f32, 0.0, 0.0, 0.0];
            col.insert(&format!("v{i}"), &v, json!({})).unwrap();
        }
        prop_assert_eq!(col.len(), n_insert as usize);

        let to_delete = n_delete.min(n_insert);
        for i in 0..to_delete {
            col.delete(&format!("v{i}")).unwrap();
        }
        prop_assert_eq!(col.len(), (n_insert - to_delete) as usize);
    }

    /// Upsert: re-inserting the same ID doesn't increase collection length.
    #[test]
    fn upsert_preserves_len(
        v1 in arb_vector(4),
        v2 in arb_vector(4),
    ) {
        let db = Database::open_memory().unwrap();
        let col = db.create_collection("test", 4).unwrap();
        col.insert("x", &v1, json!({"v": 1})).unwrap();
        prop_assert_eq!(col.len(), 1);

        col.insert("x", &v2, json!({"v": 2})).unwrap();
        prop_assert_eq!(col.len(), 1);

        let rec = col.get("x").unwrap().unwrap();
        prop_assert_eq!(&rec.metadata["v"], 2);
    }
}
