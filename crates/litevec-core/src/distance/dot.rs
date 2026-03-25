//! Dot product / inner product distance function.
//!
//! Returns the **negated** dot product so that lower = more similar,
//! consistent with other distance types.

use super::DistanceFn;

/// Dot product distance function (negated) with automatic SIMD dispatch.
#[derive(Default)]
pub struct DotProductDistance {
    _private: (),
}

impl DotProductDistance {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DistanceFn for DotProductDistance {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { super::simd::dot_product_neg_avx2(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { super::simd::dot_product_neg_neon(a, b) };
        }

        dot_product_neg_scalar(a, b)
    }
}

/// Scalar fallback implementation of negated dot product.
pub fn dot_product_neg_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }
    -dot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // dot = 4 + 10 + 18 = 32, negated = -32
        let d = dot_product_neg_scalar(&a, &b);
        assert!((d - (-32.0)).abs() < 1e-6, "Expected -32.0, got {d}");
    }

    #[test]
    fn test_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let d = dot_product_neg_scalar(&a, &b);
        assert!(
            (d - 0.0).abs() < 1e-6,
            "Orthogonal vectors should give 0, got {d}"
        );
    }

    #[test]
    fn test_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let d = dot_product_neg_scalar(&a, &b);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_negative_is_more_similar() {
        let a = vec![1.0, 1.0];
        let similar = vec![1.0, 1.0];
        let dissimilar = vec![-1.0, -1.0];
        let d_similar = dot_product_neg_scalar(&a, &similar);
        let d_dissimilar = dot_product_neg_scalar(&a, &dissimilar);
        assert!(
            d_similar < d_dissimilar,
            "Similar should have lower (more negative) distance"
        );
    }

    #[test]
    fn test_high_dimensional() {
        let n = 384;
        let a: Vec<f32> = vec![1.0; n];
        let b: Vec<f32> = vec![2.0; n];
        let d = dot_product_neg_scalar(&a, &b);
        // dot = 384 * 2.0 = 768, negated = -768
        assert!((d - (-768.0)).abs() < 1e-3, "Expected -768.0, got {d}");
    }

    #[test]
    fn test_distance_fn_trait() {
        let f = DotProductDistance::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d = f.compute(&a, &b);
        assert!((d - (-32.0)).abs() < 1e-6);
    }
}
