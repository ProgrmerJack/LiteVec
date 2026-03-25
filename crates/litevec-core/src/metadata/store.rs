//! JSON metadata storage.

use std::collections::HashMap;

/// In-memory metadata store mapping internal IDs to JSON values.
pub struct MetadataStore {
    entries: HashMap<u64, serde_json::Value>,
}

impl MetadataStore {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: u64, metadata: serde_json::Value) {
        self.entries.insert(id, metadata);
    }

    pub fn get(&self, id: u64) -> Option<&serde_json::Value> {
        self.entries.get(&id)
    }

    pub fn update(&mut self, id: u64, metadata: serde_json::Value) -> bool {
        if let std::collections::hash_map::Entry::Occupied(mut e) = self.entries.entry(id) {
            e.insert(metadata);
            true
        } else {
            false
        }
    }

    pub fn remove(&mut self, id: u64) -> Option<serde_json::Value> {
        self.entries.remove(&id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all (id, metadata) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &serde_json::Value)> {
        self.entries.iter().map(|(&id, v)| (id, v))
    }
}

impl Default for MetadataStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_crud() {
        let mut store = MetadataStore::new();
        assert!(store.is_empty());

        store.insert(0, json!({"title": "Hello"}));
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(0).unwrap(), &json!({"title": "Hello"}));

        store.update(0, json!({"title": "Updated"}));
        assert_eq!(store.get(0).unwrap(), &json!({"title": "Updated"}));

        let removed = store.remove(0);
        assert!(removed.is_some());
        assert!(store.is_empty());
    }

    #[test]
    fn test_get_nonexistent() {
        let store = MetadataStore::new();
        assert!(store.get(42).is_none());
    }
}
