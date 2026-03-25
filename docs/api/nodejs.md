# Node.js API Reference

## Installation

```bash
npm install litevec
```

## Database

```javascript
const { Database } = require('litevec');

// File-backed
const db = new Database('my_vectors.lv');

// In-memory
const db = Database.openMemory();
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `new Database(path)` | `Database` | Open file-backed database |
| `Database.openMemory()` | `Database` | Open in-memory database |
| `db.createCollection(name, dimension)` | `Collection` | Create collection |
| `db.getCollection(name)` | `Collection \| null` | Get collection |
| `db.deleteCollection(name)` | `void` | Delete collection |
| `db.listCollections()` | `string[]` | List names |

## Collection

### Insert

```javascript
// Insert with metadata
col.insert('doc1', [0.1, 0.2, 0.3], { title: 'Hello World' });

// Batch insert
col.insertBatch([
  { id: 'doc1', vector: [0.1, 0.2, 0.3], metadata: { title: 'First' } },
  { id: 'doc2', vector: [0.4, 0.5, 0.6], metadata: { title: 'Second' } },
]);
```

### Search

```javascript
const results = col.search([0.1, 0.2, 0.3], 10);

for (const result of results) {
  console.log(`${result.id}: distance=${result.distance.toFixed(4)}`);
  console.log(`  metadata: ${JSON.stringify(result.metadata)}`);
}
```

### Get / Delete

```javascript
const record = col.get('doc1');
// { id: 'doc1', vector: [...], metadata: {...} }

col.delete('doc1');
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `col.len()` | `number` | Number of vectors |
| `col.isEmpty()` | `boolean` | Whether empty |
| `col.dimension()` | `number` | Vector dimension |
| `col.name()` | `string` | Collection name |

## TypeScript Support

TypeScript definitions are included. The package exports:

```typescript
export class Database {
  constructor(path: string);
  static openMemory(): Database;
  createCollection(name: string, dimension: number): Collection;
  getCollection(name: string): Collection | null;
  deleteCollection(name: string): void;
  listCollections(): string[];
}

export class Collection {
  insert(id: string, vector: number[], metadata?: object): void;
  search(query: number[], k: number): SearchResult[];
  get(id: string): VectorRecord | null;
  delete(id: string): boolean;
  len(): number;
  isEmpty(): boolean;
  dimension(): number;
  name(): string;
}

export interface SearchResult {
  id: string;
  distance: number;
  metadata: any;
}

export interface VectorRecord {
  id: string;
  vector: number[];
  metadata: any;
}
```

## Note on Numeric Precision

JavaScript numbers are 64-bit floats (f64). The Node.js bindings automatically convert between f64 (JS) and f32 (LiteVec internal) during insert and search. This means very small differences in vector values may be rounded.
