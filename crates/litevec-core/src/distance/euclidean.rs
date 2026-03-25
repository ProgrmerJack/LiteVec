//! Euclidean (L2) distance function.
//!
//! Returns **squared** L2 distance internally for performance (avoids sqrt).
//! The sqrt is only applied when returning results to the user.

use super::DistanceFn;

/// Euclidean distance function (squared L2) with automatic SIMD dispatch.
#[derive(Default)]
pub struct EuclideanDistance {
    _private: (),
}

impl EuclideanDistance {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DistanceFn for EuclideanDistance {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { super::simd::euclidean_distance_sq_avx2(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { super::simd::euclidean_distance_sq_neon(a, b) };
        }

        euclidean_distance_sq_scalar(a, b)
    }
}

/// Scalar fallback implementation of squared Euclidean distance.
pub fn euclidean_distance_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let d = euclidean_distance_sq_scalar(&a, &a);
        assert!(
            (d - 0.0).abs() < 1e-6,
            "Identical vectors should have distance 0, got {d}"
        );
    }

    #[test]
    fn test_known_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = euclidean_distance_sq_scalar(&a, &b);
        // squared distance = 9 + 16 = 25
        assert!((d - 25.0).abs() < 1e-6, "Expected 25.0, got {d}");
    }

    #[test]
    fn test_unit_distance() {
        let a = vec![0.0];
        let b = vec![1.0];
        let d = euclidean_distance_sq_scalar(&a, &b);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_symmetry() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d1 = euclidean_distance_sq_scalar(&a, &b);
        let d2 = euclidean_distance_sq_scalar(&b, &a);
        assert!((d1 - d2).abs() < 1e-6, "Distance should be symmetric");
    }

    #[test]
    fn test_non_negative() {
        let a = vec![-1.0, -2.0, -3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d = euclidean_distance_sq_scalar(&a, &b);
        assert!(d >= 0.0, "Distance should be non-negative, got {d}");
    }

    #[test]
    fn test_high_dimensional() {
        let n = 384;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
        let d = euclidean_distance_sq_scalar(&a, &b);
        // Each diff = 1.0, so squared distance = n * 1.0 = 384
        assert!((d - 384.0).abs() < 1e-3, "Expected 384.0, got {d}");
    }

    #[test]
    fn test_distance_fn_trait() {
        let f = EuclideanDistance::new();
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = f.compute(&a, &b);
        assert!((d - 25.0).abs() < 1e-6);
    }
}
