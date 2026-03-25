# Persistence & Backup

LiteVec supports two storage modes and a backup/restore system for disaster recovery.

## Storage Modes

### File-Backed (Persistent)

```rust
let db = Database::open("my_database.lv")?;
```

Data is automatically persisted when the database is closed or dropped. Two files are created:
- `my_database.lv` — Memory-mapped storage pages
- `my_database.lv.snap` — JSON snapshot of all collections

### In-Memory (Ephemeral)

```rust
let db = Database::open_memory()?;
```

All data lives in RAM. Fast, but lost when the process exits. Useful for testing, caching, and serverless environments.

## Automatic Persistence

For file-backed databases, collections are automatically saved:
1. When `db.close()` is called
2. When `db.checkpoint()` is called
3. When the last `Database` reference is dropped

```rust
{
    let db = Database::open("data.lv")?;
    let col = db.create_collection("docs", 384)?;
    col.insert("v1", &vector, json!({}))?;
    // Data is saved automatically when `db` goes out of scope
}

// Reopen — all data is intact
let db = Database::open("data.lv")?;
let col = db.get_collection("docs").unwrap();
assert_eq!(col.len(), 1);
```

## Manual Checkpointing

Force a save without closing the database:

```rust
db.checkpoint()?;
```

## Backup & Restore

### Create a Backup

```rust
// Snapshot all collections to a backup file
db.create_backup("backups/daily_backup.snap")?;
```

### Inspect a Backup

```rust
let info = Database::backup_info("backups/daily_backup.snap")?;
println!("Version: {}", info.version);
println!("Collections: {}", info.num_collections);
println!("Total vectors: {}", info.total_vectors);

for col in &info.collections {
    println!("  {} — {} vectors ({}d)", col.name, col.num_vectors, col.dimension);
}
```

### Restore from Backup

```rust
// Restore into a new in-memory database
let restored = Database::restore_from_backup("backups/daily_backup.snap")?;

// Verify data
let col = restored.get_collection("docs").unwrap();
assert_eq!(col.len(), expected_count);
```

## Snapshot Format

Snapshots use JSON serialization with the following structure:

```json
{
  "version": 1,
  "collections": [
    {
      "name": "documents",
      "config": {
        "dimension": 384,
        "distance": "Cosine",
        "index": "Hnsw",
        "hnsw_m": 16,
        "hnsw_ef_construction": 200,
        "hnsw_ef_search": 100
      },
      "vectors": [
        {
          "id": "doc1",
          "internal_id": 0,
          "vector": [0.1, 0.2, ...],
          "metadata": {"title": "Example"}
        }
      ],
      "next_internal_id": 1
    }
  ]
}
```

## Best Practices

1. **Regular backups**: Call `create_backup()` on a schedule (daily, hourly) depending on your data criticality.
2. **Checkpoint after batch inserts**: After inserting thousands of vectors, call `checkpoint()` to ensure durability.
3. **Test restores**: Periodically verify that backups can be restored successfully.
4. **Offsite copies**: Copy `.snap` files to a different storage medium for disaster recovery.
