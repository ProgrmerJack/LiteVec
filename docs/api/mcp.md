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

Add LiteVec to your MCP client configuration:

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

### VS Code / Copilot

```json
{
  "mcp": {
    "servers": {
      "litevec": {
        "command": "litevec-mcp",
        "args": ["--memory"]
      }
    }
  }
}
```

## Available Tools

### create_collection

Create a new vector collection.

**Parameters:**
- `name` (string) — Collection name
- `dimension` (number) — Vector dimension

### insert

Insert a vector with metadata.

**Parameters:**
- `collection` (string) — Collection name
- `id` (string) — Unique vector ID
- `vector` (number[]) — Vector values
- `metadata` (object, optional) — JSON metadata

### search

Find nearest neighbors.

**Parameters:**
- `collection` (string) — Collection name
- `query` (number[]) — Query vector
- `k` (number) — Number of results

### get

Retrieve a vector by ID.

**Parameters:**
- `collection` (string) — Collection name
- `id` (string) — Vector ID

### delete

Delete a vector.

**Parameters:**
- `collection` (string) — Collection name
- `id` (string) — Vector ID

### info

Get collection statistics.

**Parameters:**
- `collection` (string) — Collection name

## Protocol

The MCP server communicates via JSON-RPC 2.0 over stdin/stdout. Each message is a single line of JSON.

### Example Session

```json
→ {"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05"}}
← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","serverInfo":{"name":"litevec-mcp","version":"0.1.0"},"capabilities":{"tools":{}}}}

→ {"jsonrpc":"2.0","method":"tools/call","id":2,"params":{"name":"create_collection","arguments":{"name":"docs","dimension":3}}}
← {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"Created collection 'docs' (dim=3)"}]}}
```

## Use Cases

- **AI agent memory** — Give agents persistent memory by storing conversation embeddings
- **RAG tool** — Let agents search a knowledge base during conversations
- **Semantic code search** — Index code embeddings for AI-assisted development
