# Rust API Reference

## Database

```rust
use litevec::Database;
```

### Opening

| Method | Description |
|--------|-------------|
| `Database::open(path)` | Open/create file-backed database |
| `Database::open_with_config(path, config)` | Open with custom config |
| `Database::open_memory()` | Create in-memory database |

### Collections

| Method | Description |
|--------|-------------|
| `db.create_collection(name, dimension)` | Create collection |
| `db.create_collection_with_config(name, config)` | Create with config |
| `db.get_collection(name)` | Get existing collection |
| `db.delete_collection(name)` | Delete collection |
| `db.list_collections()` | List all names |

### Persistence

| Method | Description |
|--------|-------------|
| `db.checkpoint()` | Force save to disk |
| `db.close()` | Save and close |
| `db.create_backup(path)` | Create backup snapshot |
| `Database::restore_from_backup(path)` | Restore from backup |
| `Database::backup_info(path)` | Read backup metadata |

## Collection

```rust
use litevec::Collection;
```

### CRUD

| Method | Description |
|--------|-------------|
| `col.insert(id, vector, metadata)` | Insert/upsert vector |
| `col.insert_batch(items)` | Batch insert |
| `col.get(id)` | Get vector by ID |
| `col.delete(id)` | Delete vector |
| `col.update_metadata(id, metadata)` | Update metadata |

### Search

```rust
let results = col.search(&query_vector, k)
    .filter(filter)           // Optional metadata filter
    .ef_search(200)           // Optional HNSW parameter override
    .execute()?;
```

### Info

| Method | Returns |
|--------|---------|
| `col.len()` | Number of vectors |
| `col.is_empty()` | Whether empty |
| `col.dimension()` | Vector dimension |
| `col.name()` | Collection name |
| `col.distance_type()` | Distance metric |

## Filter

```rust
use litevec::Filter;
```

| Variant | Description | Example |
|---------|-------------|---------|
| `Eq(field, value)` | Equal | `Filter::Eq("status".into(), "active".into())` |
| `Ne(field, value)` | Not equal | `Filter::Ne("status".into(), "deleted".into())` |
| `Gt(field, value)` | Greater than | `Filter::Gt("score".into(), json!(0.5))` |
| `Gte(field, value)` | Greater or equal | `Filter::Gte("score".into(), json!(0.5))` |
| `Lt(field, value)` | Less than | `Filter::Lt("price".into(), json!(100))` |
| `Lte(field, value)` | Less or equal | `Filter::Lte("price".into(), json!(100))` |
| `In(field, values)` | In set | `Filter::In("tag".into(), vec![json!("a"), json!("b")])` |
| `NotIn(field, values)` | Not in set | `Filter::NotIn("tag".into(), vec![json!("x")])` |
| `Exists(field)` | Field exists | `Filter::Exists("optional_field".into())` |
| `And(filters)` | All match | `Filter::And(vec![f1, f2])` |
| `Or(filters)` | Any match | `Filter::Or(vec![f1, f2])` |
| `Not(filter)` | Negation | `Filter::Not(Box::new(f))` |

## Types

### SearchResult

```rust
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: serde_json::Value,
}
```

### VectorRecord

```rust
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: serde_json::Value,
}
```

### CollectionConfig

```rust
pub struct CollectionConfig {
    pub dimension: u32,
    pub distance: DistanceType,  // Cosine, Euclidean, DotProduct
    pub index: IndexType,        // Auto, Hnsw, Flat
    pub hnsw: HnswConfig,
}
```

### HnswConfig

```rust
pub struct HnswConfig {
    pub m: usize,                // Default: 16
    pub ef_construction: usize,  // Default: 200
    pub ef_search: usize,        // Default: 100
}
```
