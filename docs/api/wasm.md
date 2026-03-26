# WebAssembly API Reference

LiteVec compiles to WebAssembly for use in browsers and edge runtimes.

## Building

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for browser
cd crates/litevec-wasm
wasm-pack build --target web

# Build for Node.js
wasm-pack build --target nodejs

# Build for bundler (webpack, etc.)
wasm-pack build --target bundler
```

## Browser Usage

```html
<script type="module">
import init, { WasmDatabase, WasmCollection } from './pkg/litevec_wasm.js';

async function main() {
    await init();

    // Create in-memory database (no file system in browser)
    const db = new WasmDatabase();

    // Create a collection
    const col = db.createCollection('docs', 3);

    // Insert vectors
    col.insert('doc1', new Float32Array([1.0, 0.0, 0.0]), '{"title": "Hello"}');
    col.insert('doc2', new Float32Array([0.0, 1.0, 0.0]), '{"title": "World"}');

    // Search
    const results = col.search(new Float32Array([0.9, 0.1, 0.0]), 2);
    console.log('Results:', results);

    // Cleanup
    col.free();
    db.free();
}

main();
</script>
```

## API

### WasmDatabase

| Method | Description |
|--------|-------------|
| `new WasmDatabase()` | Create in-memory database |
| `db.createCollection(name, dimension)` | Create collection |
| `db.getCollection(name)` | Get existing collection |
| `db.deleteCollection(name)` | Delete collection |
| `db.listCollections()` | List collection names |

### WasmCollection

| Method | Description |
|--------|-------------|
| `col.insert(id, vector, metadataJson)` | Insert vector |
| `col.search(query, k)` | Search k nearest |
| `col.get(id)` | Get vector by ID |
| `col.delete(id)` | Delete vector |
| `col.len()` | Number of vectors |
| `col.isEmpty()` | Whether empty |
| `col.dimension()` | Vector dimension |
| `col.name()` | Collection name |

## Limitations

- **In-memory only** — No file system access in the browser. All data is stored in WASM linear memory.
- **Single-threaded** — No `SharedArrayBuffer` / web workers support yet.
- **Memory limit** — Bound by WASM memory limits (typically 2-4 GB).
- **No SIMD** — Uses scalar distance computations. WASM SIMD can be enabled at build time for supported runtimes.

## Use Cases

- **Client-side semantic search** — Search within user data without server round-trips
- **Offline-first applications** — Vector search works without network connectivity
- **Edge computing** — Deploy to Cloudflare Workers, Deno Deploy, etc.
- **Prototyping** — Quick experiments in the browser console
