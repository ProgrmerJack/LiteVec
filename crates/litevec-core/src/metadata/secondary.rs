//! B-tree based secondary indexes on metadata fields.
//!
//! Provides O(log n) equality and range lookups on individual JSON fields,
//! replacing the default linear scan for indexed fields.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

use super::store::MetadataStore;

// ---------------------------------------------------------------------------
// IndexKey – an ordered key wrapping a subset of JSON value types
// ---------------------------------------------------------------------------

/// Ordered key extracted from a JSON value.
///
/// Ordering: Null < Bool(false) < Bool(true) < Integer < Float < String.
/// Floats use [`f64::total_cmp`] for a total order (NaN-safe).
#[derive(Clone, Debug)]
pub enum IndexKey {
    Null,
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

impl IndexKey {
    /// Try to convert a [`serde_json::Value`] into an [`IndexKey`].
    ///
    /// Arrays and objects are not indexable and return `None`.
    pub fn from_value(value: &serde_json::Value) -> Option<Self> {
        match value {
            serde_json::Value::Null => Some(IndexKey::Null),
            serde_json::Value::Bool(b) => Some(IndexKey::Bool(*b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(IndexKey::Integer(i))
                } else {
                    n.as_f64().map(IndexKey::Float)
                }
            }
            serde_json::Value::String(s) => Some(IndexKey::String(s.clone())),
            // Arrays and objects are not indexable.
            _ => None,
        }
    }
}

/// Discriminant rank used for cross-variant ordering.
fn variant_rank(key: &IndexKey) -> u8 {
    match key {
        IndexKey::Null => 0,
        IndexKey::Bool(_) => 1,
        IndexKey::Integer(_) => 2,
        IndexKey::Float(_) => 3,
        IndexKey::String(_) => 4,
    }
}

impl PartialEq for IndexKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for IndexKey {}

impl PartialOrd for IndexKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexKey {
    fn cmp(&self, other: &Self) -> Ordering {
        let rank_ord = variant_rank(self).cmp(&variant_rank(other));
        if rank_ord != Ordering::Equal {
            return rank_ord;
        }
        match (self, other) {
            (IndexKey::Null, IndexKey::Null) => Ordering::Equal,
            (IndexKey::Bool(a), IndexKey::Bool(b)) => a.cmp(b),
            (IndexKey::Integer(a), IndexKey::Integer(b)) => a.cmp(b),
            (IndexKey::Float(a), IndexKey::Float(b)) => a.total_cmp(b),
            (IndexKey::String(a), IndexKey::String(b)) => a.cmp(b),
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// SecondaryIndex – a single field index
// ---------------------------------------------------------------------------

/// A secondary index on a single metadata field.
///
/// Uses [`BTreeMap`] for ordered lookups supporting range queries.
pub struct SecondaryIndex {
    /// Field name this index covers (e.g., `"category"`, `"price"`).
    field: String,
    /// BTreeMap from [`IndexKey`] → `HashSet<u64>` for O(log n) lookups.
    entries: BTreeMap<IndexKey, HashSet<u64>>,
}

impl SecondaryIndex {
    fn new(field: &str) -> Self {
        Self {
            field: field.to_owned(),
            entries: BTreeMap::new(),
        }
    }

    /// Insert an id under the key extracted from `metadata`.
    fn insert(&mut self, id: u64, metadata: &serde_json::Value) {
        if let Some(key) = self.key_from(metadata) {
            self.entries.entry(key).or_default().insert(id);
        }
    }

    /// Remove an id from the key extracted from `metadata`.
    fn remove(&mut self, id: u64, metadata: &serde_json::Value) {
        if let Some(key) = self.key_from(metadata)
            && let Some(ids) = self.entries.get_mut(&key)
        {
            ids.remove(&id);
            if ids.is_empty() {
                self.entries.remove(&key);
            }
        }
    }

    /// Exact-match lookup.
    fn query_eq(&self, value: &serde_json::Value) -> HashSet<u64> {
        IndexKey::from_value(value)
            .and_then(|k| self.entries.get(&k))
            .cloned()
            .unwrap_or_default()
    }

    /// Range lookup. Both bounds are **inclusive** when provided.
    fn query_range(
        &self,
        lower: Option<&serde_json::Value>,
        upper: Option<&serde_json::Value>,
    ) -> HashSet<u64> {
        use std::ops::Bound;

        let lo = match lower.and_then(IndexKey::from_value) {
            Some(k) => Bound::Included(k),
            None => Bound::Unbounded,
        };
        let hi = match upper.and_then(IndexKey::from_value) {
            Some(k) => Bound::Included(k),
            None => Bound::Unbounded,
        };

        let mut result = HashSet::new();
        for ids in self.entries.range((lo, hi)).map(|(_, ids)| ids) {
            result.extend(ids);
        }
        result
    }

    /// Multi-value IN lookup.
    fn query_in(&self, values: &[serde_json::Value]) -> HashSet<u64> {
        let mut result = HashSet::new();
        for v in values {
            if let Some(key) = IndexKey::from_value(v)
                && let Some(ids) = self.entries.get(&key)
            {
                result.extend(ids);
            }
        }
        result
    }

    /// Extract the [`IndexKey`] for our field from a metadata document.
    fn key_from(&self, metadata: &serde_json::Value) -> Option<IndexKey> {
        metadata.get(&self.field).and_then(IndexKey::from_value)
    }
}

// ---------------------------------------------------------------------------
// SecondaryIndexManager – manages indexes across multiple fields
// ---------------------------------------------------------------------------

/// Manages secondary indexes for multiple metadata fields.
pub struct SecondaryIndexManager {
    indexes: HashMap<String, SecondaryIndex>,
}

impl SecondaryIndexManager {
    pub fn new() -> Self {
        Self {
            indexes: HashMap::new(),
        }
    }

    /// Create an index on `field`, scanning `store` to backfill existing data.
    pub fn create_index(&mut self, field: &str, store: &MetadataStore) {
        let mut index = SecondaryIndex::new(field);
        for (id, metadata) in store.iter() {
            index.insert(id, metadata);
        }
        self.indexes.insert(field.to_owned(), index);
    }

    /// Drop the index on `field`.
    pub fn drop_index(&mut self, field: &str) {
        self.indexes.remove(field);
    }

    /// Returns `true` if `field` has an active index.
    pub fn has_index(&self, field: &str) -> bool {
        self.indexes.contains_key(field)
    }

    /// Notify the manager that `metadata` was inserted/updated for `id`.
    pub fn on_insert(&mut self, id: u64, metadata: &serde_json::Value) {
        for index in self.indexes.values_mut() {
            index.insert(id, metadata);
        }
    }

    /// Notify the manager that the entry for `id` (with `metadata`) was removed.
    pub fn on_remove(&mut self, id: u64, metadata: &serde_json::Value) {
        for index in self.indexes.values_mut() {
            index.remove(id, metadata);
        }
    }

    /// Exact-match query. Returns `None` if the field is not indexed.
    pub fn query_eq(&self, field: &str, value: &serde_json::Value) -> Option<HashSet<u64>> {
        self.indexes.get(field).map(|idx| idx.query_eq(value))
    }

    /// Inclusive range query. Returns `None` if the field is not indexed.
    pub fn query_range(
        &self,
        field: &str,
        lower: Option<&serde_json::Value>,
        upper: Option<&serde_json::Value>,
    ) -> Option<HashSet<u64>> {
        self.indexes
            .get(field)
            .map(|idx| idx.query_range(lower, upper))
    }

    /// Multi-value IN query. Returns `None` if the field is not indexed.
    pub fn query_in(&self, field: &str, values: &[serde_json::Value]) -> Option<HashSet<u64>> {
        self.indexes.get(field).map(|idx| idx.query_in(values))
    }

    /// List all currently indexed field names.
    pub fn indexed_fields(&self) -> Vec<String> {
        self.indexes.keys().cloned().collect()
    }
}

impl Default for SecondaryIndexManager {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn populated_store() -> MetadataStore {
        let mut store = MetadataStore::new();
        store.insert(1, json!({"category": "science", "price": 29.99}));
        store.insert(2, json!({"category": "math",    "price": 19.99}));
        store.insert(3, json!({"category": "science", "price": 49.99}));
        store.insert(4, json!({"category": "history", "price": 9.99}));
        store
    }

    // -----------------------------------------------------------------------
    // 1. Create index on string field, query eq
    // -----------------------------------------------------------------------

    #[test]
    fn test_string_field_eq_query() {
        let store = populated_store();
        let mut mgr = SecondaryIndexManager::new();
        mgr.create_index("category", &store);

        let ids = mgr.query_eq("category", &json!("science")).unwrap();
        assert_eq!(ids, HashSet::from([1, 3]));

        let ids = mgr.query_eq("category", &json!("math")).unwrap();
        assert_eq!(ids, HashSet::from([2]));

        let ids = mgr.query_eq("category", &json!("nonexistent")).unwrap();
        assert!(ids.is_empty());
    }

    // -----------------------------------------------------------------------
    // 2. Create index on numeric field, query range (gt, lt, gte, lte)
    // -----------------------------------------------------------------------

    #[test]
    fn test_numeric_range_query() {
        let store = populated_store();
        let mut mgr = SecondaryIndexManager::new();
        mgr.create_index("price", &store);

        // price >= 19.99 AND price <= 29.99  →  ids 1, 2
        let ids = mgr
            .query_range("price", Some(&json!(19.99)), Some(&json!(29.99)))
            .unwrap();
        assert_eq!(ids, HashSet::from([1, 2]));

        // price >= 30.0  →  id 3 (49.99)
        let ids = mgr.query_range("price", Some(&json!(30.0)), None).unwrap();
        assert_eq!(ids, HashSet::from([3]));

        // price <= 10.0  →  id 4 (9.99)
        let ids = mgr.query_range("price", None, Some(&json!(10.0))).unwrap();
        assert_eq!(ids, HashSet::from([4]));

        // all prices
        let ids = mgr.query_range("price", None, None).unwrap();
        assert_eq!(ids, HashSet::from([1, 2, 3, 4]));
    }

    // -----------------------------------------------------------------------
    // 3. Insert/remove keeps index in sync
    // -----------------------------------------------------------------------

    #[test]
    fn test_insert_remove_sync() {
        let store = MetadataStore::new();
        let mut mgr = SecondaryIndexManager::new();
        mgr.create_index("category", &store);

        // Insert
        let meta = json!({"category": "art"});
        mgr.on_insert(10, &meta);
        let ids = mgr.query_eq("category", &json!("art")).unwrap();
        assert_eq!(ids, HashSet::from([10]));

        // Remove
        mgr.on_remove(10, &meta);
        let ids = mgr.query_eq("category", &json!("art")).unwrap();
        assert!(ids.is_empty());
    }

    // -----------------------------------------------------------------------
    // 4. Query on non-indexed field returns None
    // -----------------------------------------------------------------------

    #[test]
    fn test_non_indexed_field_returns_none() {
        let mgr = SecondaryIndexManager::new();
        assert!(mgr.query_eq("whatever", &json!("x")).is_none());
        assert!(mgr.query_range("whatever", None, None).is_none());
        assert!(mgr.query_in("whatever", &[json!("x")]).is_none());
    }

    // -----------------------------------------------------------------------
    // 5. In-query with multiple values
    // -----------------------------------------------------------------------

    #[test]
    fn test_in_query() {
        let store = populated_store();
        let mut mgr = SecondaryIndexManager::new();
        mgr.create_index("category", &store);

        let ids = mgr
            .query_in("category", &[json!("science"), json!("history")])
            .unwrap();
        assert_eq!(ids, HashSet::from([1, 3, 4]));

        // One value doesn't exist
        let ids = mgr
            .query_in("category", &[json!("math"), json!("nope")])
            .unwrap();
        assert_eq!(ids, HashSet::from([2]));
    }

    // -----------------------------------------------------------------------
    // 6. Mixed types in same field
    // -----------------------------------------------------------------------

    #[test]
    fn test_mixed_types() {
        let mut store = MetadataStore::new();
        store.insert(1, json!({"tag": "hello"}));
        store.insert(2, json!({"tag": 42}));
        store.insert(3, json!({"tag": true}));
        store.insert(4, json!({"tag": null}));

        let mut mgr = SecondaryIndexManager::new();
        mgr.create_index("tag", &store);

        assert_eq!(
            mgr.query_eq("tag", &json!("hello")).unwrap(),
            HashSet::from([1])
        );
        assert_eq!(mgr.query_eq("tag", &json!(42)).unwrap(), HashSet::from([2]));
        assert_eq!(
            mgr.query_eq("tag", &json!(true)).unwrap(),
            HashSet::from([3])
        );
        assert_eq!(
            mgr.query_eq("tag", &json!(null)).unwrap(),
            HashSet::from([4])
        );
    }

    // -----------------------------------------------------------------------
    // 7. Null handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_null_handling() {
        let mut store = MetadataStore::new();
        store.insert(1, json!({"val": null}));
        store.insert(2, json!({"val": 1}));
        store.insert(3, json!({"other": "x"})); // val field absent

        let mut mgr = SecondaryIndexManager::new();
        mgr.create_index("val", &store);

        // Explicit null is indexed
        let ids = mgr.query_eq("val", &json!(null)).unwrap();
        assert_eq!(ids, HashSet::from([1]));

        // Missing field is NOT indexed (id 3 should not appear anywhere)
        let all = mgr.query_range("val", None, None).unwrap();
        assert!(!all.contains(&3));
        assert!(all.contains(&1));
        assert!(all.contains(&2));
    }

    // -----------------------------------------------------------------------
    // 8. Empty index queries
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_index_queries() {
        let store = MetadataStore::new();
        let mut mgr = SecondaryIndexManager::new();
        mgr.create_index("field", &store);

        assert!(mgr.query_eq("field", &json!("x")).unwrap().is_empty());
        assert!(mgr.query_range("field", None, None).unwrap().is_empty());
        assert!(mgr.query_in("field", &[json!(1)]).unwrap().is_empty());
    }

    // -----------------------------------------------------------------------
    // 9. Create index on populated store (backfill test)
    // -----------------------------------------------------------------------

    #[test]
    fn test_backfill_on_populated_store() {
        let store = populated_store();
        let mut mgr = SecondaryIndexManager::new();

        // Index created after data is already present.
        mgr.create_index("category", &store);
        mgr.create_index("price", &store);

        let ids = mgr.query_eq("category", &json!("science")).unwrap();
        assert_eq!(ids, HashSet::from([1, 3]));

        let ids = mgr
            .query_range("price", Some(&json!(20.0)), Some(&json!(50.0)))
            .unwrap();
        assert_eq!(ids, HashSet::from([1, 3]));
    }

    // -----------------------------------------------------------------------
    // 10. Drop index
    // -----------------------------------------------------------------------

    #[test]
    fn test_drop_index() {
        let store = populated_store();
        let mut mgr = SecondaryIndexManager::new();
        mgr.create_index("category", &store);

        assert!(mgr.has_index("category"));
        assert!(mgr.query_eq("category", &json!("science")).is_some());

        mgr.drop_index("category");
        assert!(!mgr.has_index("category"));
        assert!(mgr.query_eq("category", &json!("science")).is_none());
    }

    // -----------------------------------------------------------------------
    // IndexKey ordering
    // -----------------------------------------------------------------------

    #[test]
    fn test_index_key_ordering() {
        let keys = vec![
            IndexKey::String("z".into()),
            IndexKey::Null,
            IndexKey::Float(1.5),
            IndexKey::Bool(true),
            IndexKey::Integer(10),
            IndexKey::Bool(false),
            IndexKey::Integer(-1),
        ];
        let mut sorted = keys.clone();
        sorted.sort();

        let expected = vec![
            IndexKey::Null,
            IndexKey::Bool(false),
            IndexKey::Bool(true),
            IndexKey::Integer(-1),
            IndexKey::Integer(10),
            IndexKey::Float(1.5),
            IndexKey::String("z".into()),
        ];
        assert_eq!(sorted, expected);
    }

    #[test]
    fn test_indexed_fields() {
        let store = MetadataStore::new();
        let mut mgr = SecondaryIndexManager::new();
        mgr.create_index("a", &store);
        mgr.create_index("b", &store);

        let mut fields = mgr.indexed_fields();
        fields.sort();
        assert_eq!(fields, vec!["a", "b"]);
    }
}
