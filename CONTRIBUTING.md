# Contributing to LiteVec

Thank you for your interest in contributing to LiteVec! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Rust 1.85+ (Edition 2024)
- Python 3.8+ (for Python bindings)
- Node.js 18+ (for Node.js bindings)

### Building

```bash
git clone https://github.com/ProgrmerJack/LiteVec.git
cd LiteVec
cargo build --workspace
cargo test --workspace
```

## How to Contribute

### Reporting Bugs

- Use the [Bug Report](https://github.com/ProgrmerJack/LiteVec/issues/new?template=bug_report.md) issue template
- Include: LiteVec version, OS, Rust version, minimal reproduction steps
- Include the full error message and backtrace if applicable

### Suggesting Features

- Use the [Feature Request](https://github.com/ProgrmerJack/LiteVec/issues/new?template=feature_request.md) issue template
- Describe the use case, not just the feature
- Check existing issues first to avoid duplicates

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Ensure all tests pass: `cargo test --workspace`
5. Ensure clippy is clean: `cargo clippy --workspace -- -D warnings`
6. Ensure formatting is correct: `cargo fmt --all -- --check`
7. Commit with a descriptive message
8. Push and open a PR

### Code Style

- Follow standard Rust conventions (enforced by `rustfmt`)
- All public APIs must have doc comments
- All unsafe code must have `# Safety` documentation
- Aim for zero clippy warnings with `-D warnings`
- Write tests for new functionality

### Testing

```bash
# Run all tests
cargo test --workspace

# Run a specific crate's tests
cargo test -p litevec-core

# Run a specific test
cargo test -p litevec-core -- test_name

# Run clippy
cargo clippy --workspace -- -D warnings

# Check formatting
cargo fmt --all -- --check
```

## Architecture

See [docs/guide/architecture.md](docs/guide/architecture.md) for an overview of the crate structure and design decisions.

### Crate Overview

| Crate | Purpose |
|-------|---------|
| `litevec-core` | Core engine: indexes, storage, collections |
| `litevec` | Public API re-exports |
| `litevec-cli` | Command-line interface |
| `litevec-ffi` | C foreign function interface |
| `litevec-mcp` | MCP server for AI agents |
| `litevec-wasm` | WebAssembly bindings |
| `bindings/python` | Python (PyO3) bindings |
| `bindings/node` | Node.js (NAPI-RS) bindings |

### Adding a New Index

1. Create `crates/litevec-core/src/index/your_index.rs`
2. Implement the `VectorIndex` trait
3. Add `pub mod your_index;` to `index/mod.rs`
4. Include comprehensive tests

### Adding a New Language Binding

1. Create a new crate or binding directory
2. Add it to `Cargo.toml` workspace members
3. Wrap `litevec-core` types with language-specific APIs
4. Add tests and documentation

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
