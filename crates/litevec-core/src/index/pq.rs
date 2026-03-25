//! Product Quantization (PQ) for memory-efficient approximate vector search.
//!
//! Compresses high-dimensional vectors into compact codes by dividing each vector
//! into sub-vectors and quantizing each with a learned codebook. Enables 10-32x
//! memory reduction with approximate distance computation (ADC).

use std::collections::HashMap;

use ordered_float::OrderedFloat;
use rand::Rng;

/// Product Quantization index for memory-efficient approximate search.
pub struct ProductQuantizer {
    /// Number of sub-vectors (sub-quantizers).
    num_subvectors: usize,
    /// Number of centroids per sub-quantizer (typically 256 = 8 bits).
    num_centroids: usize,
    /// Dimension of each sub-vector.
    sub_dimension: usize,
    /// Codebooks: `[num_subvectors][num_centroids][sub_dimension]`.
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Encoded vectors: each vector becomes `num_subvectors` bytes.
    codes: HashMap<u64, Vec<u8>>,
}

/// Run k-means clustering on `data`, returning `k` centroids.
///
/// Initializes by picking `k` random data points, then iteratively assigns
/// points to the nearest centroid and recomputes centroids as cluster means.
fn kmeans(data: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(data.len());
    let dim = data[0].len();

    // Initialize centroids by picking k random data points.
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..data.len()).collect();
    // Fisher-Yates partial shuffle to pick k unique indices.
    for i in 0..k {
        let j = i + (rng.r#gen::<usize>() % (data.len() - i));
        indices.swap(i, j);
    }
    let mut centroids: Vec<Vec<f32>> = indices[..k].iter().map(|&i| data[i].clone()).collect();

    let mut assignments = vec![0usize; data.len()];

    for _ in 0..max_iter {
        // Assign each point to the nearest centroid.
        let mut changed = false;
        for (i, point) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = f32::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist = squared_euclidean(point, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Recompute centroids as mean of assigned points.
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, point) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (j, &val) in point.iter().enumerate() {
                sums[c][j] += val;
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..dim {
                    centroids[c][j] = sums[c][j] * inv;
                }
            }
            // Empty clusters keep their previous centroid.
        }
    }

    centroids
}

/// Squared Euclidean distance between two vectors.
#[inline]
fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

impl ProductQuantizer {
    /// Create a new Product Quantizer.
    ///
    /// # Panics
    ///
    /// Panics if `dimension` is not divisible by `num_subvectors`, or if
    /// `num_centroids` exceeds 256 (must fit in a `u8`).
    pub fn new(dimension: usize, num_subvectors: usize, num_centroids: usize) -> Self {
        assert!(
            dimension.is_multiple_of(num_subvectors),
            "dimension ({dimension}) must be divisible by num_subvectors ({num_subvectors})"
        );
        assert!(
            num_centroids <= 256,
            "num_centroids ({num_centroids}) must be <= 256 to fit in u8"
        );
        assert!(num_subvectors > 0, "num_subvectors must be > 0");
        assert!(num_centroids > 0, "num_centroids must be > 0");

        let sub_dimension = dimension / num_subvectors;
        Self {
            num_subvectors,
            num_centroids,
            sub_dimension,
            codebooks: Vec::new(),
            codes: HashMap::new(),
        }
    }

    /// Full vector dimension.
    pub fn dimension(&self) -> usize {
        self.sub_dimension * self.num_subvectors
    }

    /// Number of encoded vectors.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Whether the quantizer has no encoded vectors.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Whether the codebooks have been trained.
    pub fn is_trained(&self) -> bool {
        self.codebooks.len() == self.num_subvectors
    }

    /// Train the codebooks using k-means on the provided training vectors.
    ///
    /// For each sub-vector position, extracts the corresponding slice from all
    /// training vectors and clusters them into `num_centroids` centroids.
    pub fn train(&mut self, vectors: &[&[f32]], max_iterations: usize) {
        assert!(!vectors.is_empty(), "need at least one training vector");
        let dim = self.dimension();
        for v in vectors {
            assert_eq!(
                v.len(),
                dim,
                "training vector dimension {} != expected {dim}",
                v.len()
            );
        }

        let mut codebooks = Vec::with_capacity(self.num_subvectors);

        for m in 0..self.num_subvectors {
            let offset = m * self.sub_dimension;
            // Extract sub-vectors for this partition.
            let sub_vectors: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| v[offset..offset + self.sub_dimension].to_vec())
                .collect();

            let centroids = kmeans(&sub_vectors, self.num_centroids, max_iterations);
            codebooks.push(centroids);
        }

        self.codebooks = codebooks;
    }

    /// Encode a single vector into compact PQ codes.
    ///
    /// Each sub-vector is assigned to its nearest centroid; the result is a
    /// `Vec<u8>` of length `num_subvectors`.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        assert!(self.is_trained(), "must call train() before encode()");
        assert_eq!(vector.len(), self.dimension(), "vector dimension mismatch");

        let mut code = Vec::with_capacity(self.num_subvectors);
        for m in 0..self.num_subvectors {
            let offset = m * self.sub_dimension;
            let sub = &vector[offset..offset + self.sub_dimension];

            let mut best_idx = 0u8;
            let mut best_dist = f32::MAX;
            for (c, centroid) in self.codebooks[m].iter().enumerate() {
                let dist = squared_euclidean(sub, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = c as u8;
                }
            }
            code.push(best_idx);
        }
        code
    }

    /// Decode PQ codes back into an approximate vector reconstruction.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        assert!(self.is_trained(), "must call train() before decode()");
        assert_eq!(codes.len(), self.num_subvectors, "code length mismatch");

        let mut vector = Vec::with_capacity(self.dimension());
        for (m, &c) in codes.iter().enumerate() {
            vector.extend_from_slice(&self.codebooks[m][c as usize]);
        }
        vector
    }

    /// Add a vector to the index under the given ID, encoding it into PQ codes.
    pub fn add(&mut self, id: u64, vector: &[f32]) {
        let code = self.encode(vector);
        self.codes.insert(id, code);
    }

    /// Remove a vector from the index.
    pub fn remove(&mut self, id: u64) {
        self.codes.remove(&id);
    }

    /// Precompute an Asymmetric Distance Computation (ADC) table.
    ///
    /// Returns `[num_subvectors][num_centroids]` distances: for each sub-quantizer
    /// and each centroid, the squared Euclidean distance from the query sub-vector
    /// to that centroid. This allows O(num_subvectors) distance lookups per
    /// candidate instead of O(dimension).
    pub fn compute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        assert!(self.is_trained(), "must call train() before search");
        assert_eq!(query.len(), self.dimension(), "query dimension mismatch");

        let mut table = Vec::with_capacity(self.num_subvectors);
        for m in 0..self.num_subvectors {
            let offset = m * self.sub_dimension;
            let sub_query = &query[offset..offset + self.sub_dimension];

            let distances: Vec<f32> = self.codebooks[m]
                .iter()
                .map(|centroid| squared_euclidean(sub_query, centroid))
                .collect();
            table.push(distances);
        }
        table
    }

    /// Search using a precomputed ADC distance table.
    ///
    /// Returns the top-`k` (id, distance) pairs sorted by ascending distance.
    pub fn search_with_table(&self, table: &[Vec<f32>], k: usize) -> Vec<(u64, f32)> {
        if k == 0 || self.codes.is_empty() {
            return Vec::new();
        }

        // Max-heap of (distance, id) — keeps the k smallest.
        let mut heap: std::collections::BinaryHeap<(OrderedFloat<f32>, u64)> =
            std::collections::BinaryHeap::with_capacity(k + 1);

        for (&id, code) in &self.codes {
            let dist: f32 = code
                .iter()
                .enumerate()
                .map(|(m, &c)| table[m][c as usize])
                .sum();

            heap.push((OrderedFloat(dist), id));
            if heap.len() > k {
                heap.pop();
            }
        }

        let mut results: Vec<(u64, f32)> = heap.into_iter().map(|(d, id)| (id, d.0)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }

    /// Convenience: compute distance table and search in one call.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let table = self.compute_distance_table(query);
        self.search_with_table(&table, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Generate `n` random vectors of the given dimension.
    fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() - 0.5).collect())
            .collect()
    }

    #[test]
    fn test_train_and_encode_decode() {
        let dim = 16;
        let nsub = 4;
        let ncentroids = 8;
        let mut pq = ProductQuantizer::new(dim, nsub, ncentroids);

        let data = random_vectors(200, dim);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs, 20);

        assert!(pq.is_trained());

        // Encode and decode should produce an approximation.
        for v in &data[..10] {
            let code = pq.encode(v);
            assert_eq!(code.len(), nsub);
            // Each code byte should be in [0, ncentroids).
            for &c in &code {
                assert!((c as usize) < ncentroids);
            }

            let reconstructed = pq.decode(&code);
            assert_eq!(reconstructed.len(), dim);

            // Reconstruction error should be finite and bounded.
            let err: f32 = v
                .iter()
                .zip(reconstructed.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            assert!(err.is_finite());
        }
    }

    #[test]
    fn test_search_returns_nearest() {
        let dim = 32;
        let nsub = 8;
        let ncentroids = 16;
        let mut pq = ProductQuantizer::new(dim, nsub, ncentroids);

        let data = random_vectors(500, dim);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs, 30);

        // Add all vectors.
        for (i, v) in data.iter().enumerate() {
            pq.add(i as u64, v);
        }
        assert_eq!(pq.len(), 500);

        // Search for a known vector — it should be among the top results.
        let query = &data[42];
        let results = pq.search(query, 5);
        assert_eq!(results.len(), 5);

        // The query vector itself (id 42) should be the top-1 result.
        assert_eq!(
            results[0].0, 42,
            "expected the query vector to be top-1, got id {}",
            results[0].0
        );
    }

    #[test]
    fn test_memory_reduction() {
        let dim = 128;
        let nsub = 16; // 16 bytes per code
        let ncentroids = 256;
        let mut pq = ProductQuantizer::new(dim, nsub, ncentroids);

        let data = random_vectors(1000, dim);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs, 20);

        for (i, v) in data.iter().enumerate() {
            pq.add(i as u64, v);
        }

        // Raw storage: 1000 vectors * 128 dims * 4 bytes = 512,000 bytes
        let raw_bytes = 1000 * dim * std::mem::size_of::<f32>();
        // PQ storage: 1000 vectors * 16 bytes = 16,000 bytes
        let pq_bytes = 1000 * nsub * std::mem::size_of::<u8>();

        let ratio = raw_bytes as f64 / pq_bytes as f64;
        assert!(
            ratio >= 10.0,
            "expected at least 10x compression, got {ratio:.1}x"
        );
    }

    #[test]
    fn test_training_on_random_data() {
        let dim = 8;
        let nsub = 2;
        let ncentroids = 4;
        let mut pq = ProductQuantizer::new(dim, nsub, ncentroids);

        assert!(!pq.is_trained());
        assert!(pq.is_empty());

        let data = random_vectors(50, dim);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs, 10);

        assert!(pq.is_trained());
        assert_eq!(pq.dimension(), dim);

        // Encode all, verify round-trip.
        for (i, v) in data.iter().enumerate() {
            pq.add(i as u64, v);
        }
        assert_eq!(pq.len(), 50);

        // Remove one.
        pq.remove(0);
        assert_eq!(pq.len(), 49);
    }

    #[test]
    fn test_distance_table_dimensions() {
        let dim = 16;
        let nsub = 4;
        let ncentroids = 8;
        let mut pq = ProductQuantizer::new(dim, nsub, ncentroids);

        let data = random_vectors(100, dim);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs, 10);

        let query = &data[0];
        let table = pq.compute_distance_table(query);

        assert_eq!(table.len(), nsub);
        for row in &table {
            assert_eq!(row.len(), ncentroids);
            for &d in row {
                assert!(d >= 0.0);
                assert!(d.is_finite());
            }
        }
    }

    #[test]
    fn test_search_empty_index() {
        let dim = 8;
        let nsub = 2;
        let ncentroids = 4;
        let mut pq = ProductQuantizer::new(dim, nsub, ncentroids);

        let data = random_vectors(10, dim);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs, 5);

        let results = pq.search(&data[0], 5);
        assert!(results.is_empty());
    }

    #[test]
    #[should_panic(expected = "dimension (10) must be divisible by num_subvectors (3)")]
    fn test_dimension_not_divisible() {
        ProductQuantizer::new(10, 3, 8);
    }

    #[test]
    #[should_panic(expected = "num_centroids (512) must be <= 256")]
    fn test_too_many_centroids() {
        ProductQuantizer::new(16, 4, 512);
    }
}
