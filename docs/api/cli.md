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
litevec create --db my_vectors.lv --name documents --dimension 384
```

### Insert Vector

```bash
litevec insert --db my_vectors.lv --collection documents \
  --id doc1 --vector "0.1,0.2,0.3,..." --metadata '{"title":"Hello"}'
```

### Search

```bash
litevec search --db my_vectors.lv --collection documents \
  --query "0.1,0.2,0.3,..." --k 10
```

### Get Vector

```bash
litevec get --db my_vectors.lv --collection documents --id doc1
```

### Delete Vector

```bash
litevec delete --db my_vectors.lv --collection documents --id doc1
```

### List Collections

```bash
litevec list --db my_vectors.lv
```

### Collection Info

```bash
litevec info --db my_vectors.lv --collection documents
```

## Global Options

| Option | Description |
|--------|-------------|
| `--db <PATH>` | Path to database file |
| `--help` | Show help |
| `--version` | Show version |
