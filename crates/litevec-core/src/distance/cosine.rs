//! Cosine similarity distance function.
//!
//! Returns cosine distance = 1.0 - cosine_similarity, so lower = more similar.

use super::DistanceFn;

/// Cosine distance function with automatic SIMD dispatch.
#[derive(Default)]
pub struct CosineDistance {
    _private: (),
}

impl CosineDistance {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DistanceFn for CosineDistance {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { super::simd::cosine_distance_avx2(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { super::simd::cosine_distance_neon(a, b) };
        }

        cosine_distance_scalar(a, b)
    }
}

/// Scalar fallback implementation of cosine distance.
pub fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 1.0;
    }
    1.0 - dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let d = cosine_distance_scalar(&a, &a);
        assert!(
            (d - 0.0).abs() < 1e-6,
            "Identical vectors should have distance ~0, got {d}"
        );
    }

    #[test]
    fn test_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let d = cosine_distance_scalar(&a, &b);
        assert!(
            (d - 1.0).abs() < 1e-6,
            "Orthogonal vectors should have distance ~1, got {d}"
        );
    }

    #[test]
    fn test_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let d = cosine_distance_scalar(&a, &b);
        assert!(
            (d - 2.0).abs() < 1e-6,
            "Opposite vectors should have distance ~2, got {d}"
        );
    }

    #[test]
    fn test_similar_vectors() {
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 0.9];
        let d = cosine_distance_scalar(&a, &b);
        assert!(
            d > 0.0 && d < 0.1,
            "Similar vectors should have small distance, got {d}"
        );
    }

    #[test]
    fn test_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let d = cosine_distance_scalar(&a, &b);
        assert_eq!(d, 1.0, "Zero vector should give distance 1.0");
    }

    #[test]
    fn test_high_dimensional() {
        let n = 384;
        let a: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32).cos()).collect();
        let d = cosine_distance_scalar(&a, &b);
        assert!(
            d >= 0.0 && d <= 2.0,
            "Distance should be in [0, 2], got {d}"
        );
    }

    #[test]
    fn test_scaled_vectors_same_direction() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];
        let d = cosine_distance_scalar(&a, &b);
        assert!(
            (d - 0.0).abs() < 1e-6,
            "Scaled vectors should have distance ~0, got {d}"
        );
    }

    #[test]
    fn test_distance_fn_trait() {
        let f = CosineDistance::new();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let d = f.compute(&a, &b);
        assert!((d - 1.0).abs() < 1e-6);
    }
}
