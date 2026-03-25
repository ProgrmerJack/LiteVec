"""FAISS benchmark for comparison with LiteVec."""
import time
import numpy as np

try:
    import faiss
except ImportError:
    print("FAISS not installed. pip install faiss-cpu")
    exit(1)

DIM = 128
COUNTS = [1_000, 10_000, 50_000]
NUM_QUERIES = 100
K = 10

print("=" * 60)
print(f"  FAISS Benchmark (dim={DIM})")
print("=" * 60)

for n in COUNTS:
    print(f"\n--- {n} vectors (dim={DIM}) ---\n")
    
    np.random.seed(42)
    data = np.random.randn(n, DIM).astype(np.float32)
    queries = np.random.randn(NUM_QUERIES, DIM).astype(np.float32)
    
    # Normalize for cosine similarity (FAISS uses IP for normalized vectors)
    faiss.normalize_L2(data)
    faiss.normalize_L2(queries)
    
    # --- Flat (exact) ---
    index_flat = faiss.IndexFlatIP(DIM)
    start = time.perf_counter()
    index_flat.add(data)
    insert_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for i in range(NUM_QUERIES):
        index_flat.search(queries[i:i+1], K)
    search_time = time.perf_counter() - start
    avg_us = search_time / NUM_QUERIES * 1_000_000
    qps = NUM_QUERIES / search_time
    
    print(f"  FAISS Flat (exact):")
    print(f"    Insert:  {n} vectors in {insert_time*1000:.1f}ms")
    print(f"    Search:  {NUM_QUERIES} queries, k={K}, avg {avg_us:.0f}μs/query ({qps:.0f} QPS)")
    
    # --- HNSW ---
    index_hnsw = faiss.IndexHNSWFlat(DIM, 16)  # M=16 same as LiteVec
    index_hnsw.hnsw.efConstruction = 200  # Same as LiteVec
    index_hnsw.hnsw.efSearch = 100  # Same as LiteVec
    
    start = time.perf_counter()
    index_hnsw.add(data)
    insert_time = time.perf_counter() - start
    insert_rate = n / insert_time
    
    start = time.perf_counter()
    for i in range(NUM_QUERIES):
        index_hnsw.search(queries[i:i+1], K)
    search_time = time.perf_counter() - start
    avg_us = search_time / NUM_QUERIES * 1_000_000
    qps = NUM_QUERIES / search_time
    
    print(f"  FAISS HNSW (M=16, ef=200/100):")
    print(f"    Insert:  {n} vectors in {insert_time*1000:.1f}ms ({insert_rate:.0f} vec/s)")
    print(f"    Search:  {NUM_QUERIES} queries, k={K}, avg {avg_us:.0f}μs/query ({qps:.0f} QPS)")

print()
print("=" * 60)
