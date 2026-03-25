# MCP Server Reference

LiteVec includes an MCP (Model Context Protocol) server that allows AI agents to use vector search as a tool.

## What is MCP?

The [Model Context Protocol](https://modelcontextprotocol.io/) is a standard for connecting AI agents to external tools. LiteVec's MCP server exposes vector database operations as tools that any MCP-compatible agent can call.

## Running the Server

```bash
# Build the MCP server
cargo build --release -p litevec-mcp

# Run with a database file
./target/release/litevec-mcp --db my_vectors.lv

# Run with in-memory database
./target/release/litevec-mcp --memory
```

## Configuration

### VS Code / GitHub Copilot

Add a `.vscode/mcp.json` to your project:

```json
{
  "servers": {
    "litevec": {
      "command": "cargo",
      "args": ["run", "-p", "litevec-mcp", "--", "--memory"]
    }
  }
}
```

Or if you have the binary installed:

```json
{
  "servers": {
    "litevec": {
      "command": "litevec-mcp",
      "args": ["--db", "my_vectors.lv"]
    }
  }
}
```

### Claude Desktop (claude_desktop_config.json)

```json
{
  "mcpServers": {
    "litevec": {
      "command": "litevec-mcp",
      "args": ["--db", "/path/to/vectors.lv"]
    }
  }
}
```

## Available Tools (11)

### litevec_create_collection

Create a new vector collection with a specified dimension.

**Parameters:**
- `name` (string, required) — Collection name (must be unique)
- `dimension` (integer, required) — Vector dimension (e.g., 384, 768, 1536)

### litevec_list_collections

List all collections with their names, dimensions, and vector counts.

**Parameters:** none

### litevec_insert

Insert a single vector with ID and optional metadata (upsert behavior).

**Parameters:**
- `collection` (string, required) — Collection name
- `id` (string, required) — Unique vector ID
- `vector` (number[], required) — Vector values
- `metadata` (object, optional) — JSON metadata

### litevec_batch_insert

Insert multiple vectors at once for bulk operations.

**Parameters:**
- `collection` (string, required) — Collection name
- `items` (array, required) — Array of `{id, vector, metadata?}` objects

### litevec_search

Find the k most similar vectors using HNSW approximate nearest neighbor search. Supports metadata filtering.

**Parameters:**
- `collection` (string, required) — Collection name
- `vector` (number[], required) — Query vector
- `k` (integer, default: 10) — Number of results
- `filter` (object, optional) — Metadata filter. Simple: `{"field": "value"}`. Advanced: `{"field": {"$gt": 5}}`. Logical: `{"$and": [...]}`, `{"$or": [...]}`

### litevec_text_search

Full-text keyword search across metadata string fields using BM25 ranking.

**Parameters:**
- `collection` (string, required) — Collection name
- `query` (string, required) — Text query (keywords)
- `limit` (integer, default: 10) — Maximum results

### litevec_hybrid_search

Combined vector + keyword search using Reciprocal Rank Fusion (RRF). Best of both worlds.

**Parameters:**
- `collection` (string, required) — Collection name
- `vector` (number[], required) — Query embedding vector
- `query` (string, required) — Text query for keyword matching
- `k` (integer, default: 10) — Number of results

### litevec_get

Retrieve a specific vector and its metadata by ID.

**Parameters:**
- `collection` (string, required) — Collection name
- `id` (string, required) — Vector ID

### litevec_delete

Delete a vector by ID.

**Parameters:**
- `collection` (string, required) — Collection name
- `id` (string, required) — Vector ID

### litevec_update_metadata

Update the metadata of an existing vector without changing the vector itself.

**Parameters:**
- `collection` (string, required) — Collection name
- `id` (string, required) — Vector ID
- `metadata` (object, required) — New metadata

### litevec_collection_info

Get detailed info about a specific collection (dimension, count, distance metric).

**Parameters:**
- `collection` (string, required) — Collection name

## Protocol

The MCP server communicates via JSON-RPC 2.0 over stdin/stdout. Each message is a single line of JSON.

### Example Session

```json
→ {"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05"}}
← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","serverInfo":{"name":"litevec-mcp","version":"0.1.0"},"capabilities":{"tools":{}}}}

→ {"jsonrpc":"2.0","method":"tools/call","id":2,"params":{"name":"litevec_create_collection","arguments":{"name":"docs","dimension":384}}}
← {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"Created collection 'docs' (dim=384)"}]}}

→ {"jsonrpc":"2.0","method":"tools/call","id":3,"params":{"name":"litevec_insert","arguments":{"collection":"docs","id":"doc1","vector":[0.1, ...],"metadata":{"title":"Getting Started","text":"LiteVec is..."}}}}
← {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"Inserted vector 'doc1' into 'docs'"}]}}

→ {"jsonrpc":"2.0","method":"tools/call","id":4,"params":{"name":"litevec_text_search","arguments":{"collection":"docs","query":"getting started"}}}
← {"jsonrpc":"2.0","id":4,"result":{"content":[{"type":"text","text":"[{\"id\":\"doc1\",\"score\":1.23,\"metadata\":{...}}]"}]}}
```

## Use Cases

- **AI agent memory** — Give agents persistent memory by storing conversation embeddings
- **RAG tool** — Let agents search a knowledge base during conversations
- **Semantic code search** — Index code embeddings for AI-assisted development
- **Hybrid search** — Combine semantic and keyword search for comprehensive retrieval
