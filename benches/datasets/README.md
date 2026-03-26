# Benchmark Datasets

This directory holds standard ANN benchmark datasets for evaluating LiteVec's search performance. The datasets are not committed to the repository due to their size — follow the instructions below to download them.

## Datasets

### SIFT-128 (ANN_SIFT1M)

The most widely used ANN benchmark. 1 million 128-dimensional vectors derived from SIFT image descriptors.

- **Vectors:** 1,000,000
- **Dimensions:** 128
- **Query set:** 10,000 queries
- **Ground truth:** 100 nearest neighbors per query
- **Download:** <http://corpus-texmex.irisa.fr/fvecs/sift.tar.gz>

```bash
wget http://corpus-texmex.irisa.fr/fvecs/sift.tar.gz
tar -xzf sift.tar.gz
```

### GIST-960 (ANN_GIST1M)

High-dimensional stress test. 1 million 960-dimensional GIST descriptors.

- **Vectors:** 1,000,000
- **Dimensions:** 960
- **Query set:** 1,000 queries
- **Ground truth:** 100 nearest neighbors per query
- **Download:** <http://corpus-texmex.irisa.fr/fvecs/gist.tar.gz>

```bash
wget http://corpus-texmex.irisa.fr/fvecs/gist.tar.gz
tar -xzf gist.tar.gz
```

### Custom Random Vectors

For scalability testing, the benchmarks in `crates/litevec-core/benches/` generate random vectors at runtime using deterministic seeds. No download needed.

## File Format

The SIFT and GIST datasets use the `.fvecs` format:

```
<d:int32> <v1:float32> <v2:float32> ... <vd:float32>   (repeated per vector)
```

Each vector is preceded by a 4-byte integer indicating the dimension.

## Running Benchmarks

```bash
# Run all benchmarks (uses generated random vectors)
cargo bench -p litevec-core

# Run specific benchmark group
cargo bench -p litevec-core -- search_hnsw
cargo bench -p litevec-core -- insert
cargo bench -p litevec-core -- distance
```
