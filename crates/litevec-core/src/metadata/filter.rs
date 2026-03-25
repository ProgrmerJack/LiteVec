//! Metadata filter engine.
//!
//! Evaluates filters against JSON metadata values. Returns the set of
//! internal IDs that match the filter (pre-filtering approach).

use std::collections::HashSet;

use crate::types::Filter;

use super::store::MetadataStore;

/// Evaluate a filter against all metadata entries, returning matching IDs.
pub fn evaluate_filter(filter: &Filter, store: &MetadataStore) -> HashSet<u64> {
    let mut result = HashSet::new();
    for (id, metadata) in store.iter() {
        if matches_filter(filter, metadata) {
            result.insert(id);
        }
    }
    result
}

/// Check if a single metadata value matches a filter.
pub fn matches_filter(filter: &Filter, metadata: &serde_json::Value) -> bool {
    match filter {
        Filter::Eq(field, value) => metadata.get(field).is_some_and(|v| v == value),

        Filter::Ne(field, value) => metadata.get(field) != Some(value),

        Filter::Gt(field, threshold) => metadata
            .get(field)
            .and_then(|v| v.as_f64())
            .is_some_and(|v| v > *threshold),

        Filter::Gte(field, threshold) => metadata
            .get(field)
            .and_then(|v| v.as_f64())
            .is_some_and(|v| v >= *threshold),

        Filter::Lt(field, threshold) => metadata
            .get(field)
            .and_then(|v| v.as_f64())
            .is_some_and(|v| v < *threshold),

        Filter::Lte(field, threshold) => metadata
            .get(field)
            .and_then(|v| v.as_f64())
            .is_some_and(|v| v <= *threshold),

        Filter::In(field, values) => metadata.get(field).is_some_and(|v| values.contains(v)),

        Filter::Exists(field) => metadata.get(field).is_some(),

        Filter::And(filters) => filters.iter().all(|f| matches_filter(f, metadata)),

        Filter::Or(filters) => filters.iter().any(|f| matches_filter(f, metadata)),

        Filter::Not(inner) => !matches_filter(inner, metadata),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_metadata() -> serde_json::Value {
        json!({
            "category": "science",
            "year": 2024,
            "score": 0.95,
            "tags": ["ai", "ml"],
            "published": true
        })
    }

    #[test]
    fn test_eq() {
        let m = sample_metadata();
        assert!(matches_filter(
            &Filter::Eq("category".into(), json!("science")),
            &m
        ));
        assert!(!matches_filter(
            &Filter::Eq("category".into(), json!("math")),
            &m
        ));
    }

    #[test]
    fn test_ne() {
        let m = sample_metadata();
        assert!(matches_filter(
            &Filter::Ne("category".into(), json!("math")),
            &m
        ));
        assert!(!matches_filter(
            &Filter::Ne("category".into(), json!("science")),
            &m
        ));
    }

    #[test]
    fn test_gt_gte() {
        let m = sample_metadata();
        assert!(matches_filter(&Filter::Gt("year".into(), 2023.0), &m));
        assert!(!matches_filter(&Filter::Gt("year".into(), 2024.0), &m));
        assert!(matches_filter(&Filter::Gte("year".into(), 2024.0), &m));
        assert!(!matches_filter(&Filter::Gte("year".into(), 2025.0), &m));
    }

    #[test]
    fn test_lt_lte() {
        let m = sample_metadata();
        assert!(matches_filter(&Filter::Lt("year".into(), 2025.0), &m));
        assert!(!matches_filter(&Filter::Lt("year".into(), 2024.0), &m));
        assert!(matches_filter(&Filter::Lte("year".into(), 2024.0), &m));
    }

    #[test]
    fn test_in() {
        let m = sample_metadata();
        assert!(matches_filter(
            &Filter::In("category".into(), vec![json!("science"), json!("math")]),
            &m
        ));
        assert!(!matches_filter(
            &Filter::In("category".into(), vec![json!("math"), json!("art")]),
            &m
        ));
    }

    #[test]
    fn test_exists() {
        let m = sample_metadata();
        assert!(matches_filter(&Filter::Exists("category".into()), &m));
        assert!(!matches_filter(&Filter::Exists("nonexistent".into()), &m));
    }

    #[test]
    fn test_and() {
        let m = sample_metadata();
        let filter = Filter::And(vec![
            Filter::Eq("category".into(), json!("science")),
            Filter::Gte("year".into(), 2024.0),
        ]);
        assert!(matches_filter(&filter, &m));

        let filter = Filter::And(vec![
            Filter::Eq("category".into(), json!("science")),
            Filter::Gte("year".into(), 2025.0),
        ]);
        assert!(!matches_filter(&filter, &m));
    }

    #[test]
    fn test_or() {
        let m = sample_metadata();
        let filter = Filter::Or(vec![
            Filter::Eq("category".into(), json!("math")),
            Filter::Gte("year".into(), 2024.0),
        ]);
        assert!(matches_filter(&filter, &m));

        let filter = Filter::Or(vec![
            Filter::Eq("category".into(), json!("math")),
            Filter::Gte("year".into(), 2025.0),
        ]);
        assert!(!matches_filter(&filter, &m));
    }

    #[test]
    fn test_not() {
        let m = sample_metadata();
        let filter = Filter::Not(Box::new(Filter::Eq("category".into(), json!("math"))));
        assert!(matches_filter(&filter, &m));

        let filter = Filter::Not(Box::new(Filter::Eq("category".into(), json!("science"))));
        assert!(!matches_filter(&filter, &m));
    }

    #[test]
    fn test_missing_field() {
        let m = sample_metadata();
        // Gt on missing field → false
        assert!(!matches_filter(&Filter::Gt("missing".into(), 0.0), &m));
        // Eq on missing field → false
        assert!(!matches_filter(
            &Filter::Eq("missing".into(), json!(null)),
            &m
        ));
    }

    #[test]
    fn test_evaluate_filter_on_store() {
        let mut store = MetadataStore::new();
        store.insert(0, json!({"category": "science", "year": 2024}));
        store.insert(1, json!({"category": "math", "year": 2023}));
        store.insert(2, json!({"category": "science", "year": 2022}));

        let filter = Filter::And(vec![
            Filter::Eq("category".into(), json!("science")),
            Filter::Gte("year".into(), 2023.0),
        ]);

        let matching = evaluate_filter(&filter, &store);
        assert!(matching.contains(&0));
        assert!(!matching.contains(&1));
        assert!(!matching.contains(&2));
    }
}
