# LiteVec Documentation

Welcome to the LiteVec documentation — the embedded vector database for AI applications.

## 📖 Guide

- [Getting Started](guide/getting-started.md) — Install and run your first query in under 60 seconds
- [Architecture](guide/architecture.md) — How LiteVec works under the hood
- [Performance Tuning](guide/performance.md) — Optimize for your workload
- [Persistence & Backup](guide/persistence.md) — Data durability and disaster recovery

## 📚 API Reference

- [Rust API](api/rust.md) — Core Rust library
- [Python API](api/python.md) — PyO3 bindings
- [Node.js API](api/nodejs.md) — NAPI-RS bindings
- [C FFI](api/c-ffi.md) — C/C++ foreign function interface
- [WebAssembly](api/wasm.md) — Browser and edge runtimes
- [CLI](api/cli.md) — Command-line tool
- [MCP Server](api/mcp.md) — Model Context Protocol for AI agents

## 🔧 Examples

- [RAG Pipeline](examples/rag.md) — Build a retrieval-augmented generation system
- [Semantic Search](examples/semantic-search.md) — Full-text + vector hybrid search
- [AI Agent Memory](examples/agent-memory.md) — Long-term memory for LLM agents via MCP

## 🏗️ Building from Source

```bash
# Clone the repository
git clone https://github.com/ProgrmerJack/LiteVec.git
cd LiteVec

# Build all crates
cargo build --release

# Run tests
cargo test --workspace

# Build Python bindings
cd bindings/python && maturin develop --release

# Build WASM
cd crates/litevec-wasm && wasm-pack build --target web
```

## License

MIT — see [LICENSE](../LICENSE).
