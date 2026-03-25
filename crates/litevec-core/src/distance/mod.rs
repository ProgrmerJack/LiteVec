//! Distance functions with SIMD acceleration.
//!
//! Provides cosine, euclidean (L2), and dot product distance functions
//! with automatic runtime dispatch to the fastest SIMD implementation.

pub mod cosine;
pub mod dot;
pub mod euclidean;
pub mod simd;

use crate::types::DistanceType;

/// Trait for computing distance between two vectors.
///
/// All distance functions return a value where **lower = more similar**.
pub trait DistanceFn: Send + Sync {
    /// Compute the distance between vectors `a` and `b`.
    ///
    /// Both slices must have the same length. Panics otherwise.
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Returns the best available distance function for this CPU and distance type.
pub fn get_distance_fn(distance_type: DistanceType) -> Box<dyn DistanceFn> {
    match distance_type {
        DistanceType::Cosine => Box::new(cosine::CosineDistance::new()),
        DistanceType::Euclidean => Box::new(euclidean::EuclideanDistance::new()),
        DistanceType::DotProduct => Box::new(dot::DotProductDistance::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_distance_fn_cosine() {
        let f = get_distance_fn(DistanceType::Cosine);
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((f.compute(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_distance_fn_euclidean() {
        let f = get_distance_fn(DistanceType::Euclidean);
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((f.compute(&a, &b) - 25.0).abs() < 1e-6); // squared L2
    }

    #[test]
    fn test_get_distance_fn_dot_product() {
        let f = get_distance_fn(DistanceType::DotProduct);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // dot = 4+10+18 = 32, negated = -32
        assert!((f.compute(&a, &b) - (-32.0)).abs() < 1e-6);
    }
}
