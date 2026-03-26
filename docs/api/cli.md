# CLI Reference

LiteVec includes a command-line tool for database management and ad-hoc queries.

## Installation

```bash
cargo install litevec-cli
```

Or build from source:

```bash
cargo build --release -p litevec-cli
```

## Commands

### Create Collection

```bash
litevec create mydb.lv --collection documents --dimension 384
```

### Insert Vectors from JSONL

```bash
litevec insert mydb.lv --collection documents --input vectors.jsonl
```

Each line in the JSONL file should be a JSON object:

```json
{"id": "doc1", "vector": [0.1, 0.2, 0.3], "metadata": {"title": "Hello"}}
```

### Search

```bash
litevec search mydb.lv --collection documents --vector '[0.1, 0.2, 0.3]' --k 10
```

### Get Vector

```bash
litevec get mydb.lv --collection documents doc1
```

### Delete Vector

```bash
litevec delete mydb.lv --collection documents doc1
```

### List Collections

```bash
litevec list mydb.lv
```

### Collection Info

```bash
litevec info mydb.lv --collection documents
```

### Serve (HTTP API)

Start a lightweight HTTP server for quick testing and prototyping:

```bash
litevec serve my_vectors.lv --port 8080
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/collections` | List all collections |
| `POST` | `/collections/{name}?dimension=N` | Create collection |
| `POST` | `/collections/{name}/insert` | Insert vector (JSON body) |
| `POST` | `/collections/{name}/search` | Search (JSON body) |
| `GET` | `/collections/{name}/{id}` | Get vector by ID |
| `DELETE` | `/collections/{name}/{id}` | Delete vector |

### Bench (Built-in Benchmarks)

Run built-in benchmarks to measure performance on your hardware:

```bash
litevec bench --dimension 128 --count 10000 --queries 100
```

Output includes insert throughput (vec/s), search latency (μs/query), and filtered search performance.

## Global Options

| Option | Description |
|--------|-------------|
| `--help` | Show help |
| `--version` | Show version |

The database path is a positional argument (first argument) for most commands.
