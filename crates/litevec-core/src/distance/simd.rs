//! SIMD intrinsics abstraction layer.
//!
//! Provides AVX2 (x86_64) and NEON (aarch64) accelerated distance computations.
//! Falls back to scalar if SIMD is not available.

// ── AVX2 implementations (x86_64) ───────────────────────────────────────

/// # Safety
/// Requires x86_64 with AVX2 and FMA support. Caller must verify CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;

        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut dot_acc = _mm256_setzero_ps();
        let mut norm_a_acc = _mm256_setzero_ps();
        let mut norm_b_acc = _mm256_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a_ptr.add(i * 8));
            let vb = _mm256_loadu_ps(b_ptr.add(i * 8));

            dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
            norm_a_acc = _mm256_fmadd_ps(va, va, norm_a_acc);
            norm_b_acc = _mm256_fmadd_ps(vb, vb, norm_b_acc);
        }

        let mut dot = hsum_avx2(dot_acc);
        let mut norm_a = hsum_avx2(norm_a_acc);
        let mut norm_b = hsum_avx2(norm_b_acc);

        let start = chunks * 8;
        for i in 0..remainder {
            let ai = *a_ptr.add(start + i);
            let bi = *b_ptr.add(start + i);
            dot += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            return 1.0;
        }
        1.0 - dot / denom
    }
}

/// # Safety
/// Requires x86_64 with AVX2 and FMA support. Caller must verify CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn euclidean_distance_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;

        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut sum_acc = _mm256_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a_ptr.add(i * 8));
            let vb = _mm256_loadu_ps(b_ptr.add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            sum_acc = _mm256_fmadd_ps(diff, diff, sum_acc);
        }

        let mut sum = hsum_avx2(sum_acc);

        let start = chunks * 8;
        for i in 0..remainder {
            let diff = *a_ptr.add(start + i) - *b_ptr.add(start + i);
            sum += diff * diff;
        }

        sum
    }
}

/// # Safety
/// Requires x86_64 with AVX2 and FMA support. Caller must verify CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_neg_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;

        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut dot_acc = _mm256_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a_ptr.add(i * 8));
            let vb = _mm256_loadu_ps(b_ptr.add(i * 8));
            dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
        }

        let mut dot = hsum_avx2(dot_acc);

        let start = chunks * 8;
        for i in 0..remainder {
            dot += *a_ptr.add(start + i) * *b_ptr.add(start + i);
        }

        -dot
    }
}

/// Horizontal sum of a 256-bit float register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    // All operations here are inherently unsafe via target_feature on the enclosing fn
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(low, high);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

// ── NEON implementations (aarch64) ──────────────────────────────────────

/// Horizontal sum of a 128-bit NEON float register.
///
/// # Safety
/// Requires aarch64 with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hsum_neon(v: std::arch::aarch64::float32x4_t) -> f32 {
    use std::arch::aarch64::*;

    let low = vget_low_f32(v);
    let high = vget_high_f32(v);
    let sum = vpadd_f32(low, high);
    let sum = vpadd_f32(sum, sum);
    vget_lane_f32::<0>(sum)
}

/// # Safety
/// Requires aarch64 with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut dot_acc = vdupq_n_f32(0.0);
    let mut norm_a_acc = vdupq_n_f32(0.0);
    let mut norm_b_acc = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f32(a_ptr.add(i * 4));
        let vb = vld1q_f32(b_ptr.add(i * 4));

        dot_acc = vfmaq_f32(dot_acc, va, vb);
        norm_a_acc = vfmaq_f32(norm_a_acc, va, va);
        norm_b_acc = vfmaq_f32(norm_b_acc, vb, vb);
    }

    let mut dot = hsum_neon(dot_acc);
    let mut norm_a = hsum_neon(norm_a_acc);
    let mut norm_b = hsum_neon(norm_b_acc);

    let start = chunks * 4;
    for i in 0..remainder {
        let ai = *a_ptr.add(start + i);
        let bi = *b_ptr.add(start + i);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 1.0;
    }
    1.0 - dot / denom
}

/// # Safety
/// Requires aarch64 with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn euclidean_distance_sq_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum_acc = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f32(a_ptr.add(i * 4));
        let vb = vld1q_f32(b_ptr.add(i * 4));
        let diff = vsubq_f32(va, vb);
        sum_acc = vfmaq_f32(sum_acc, diff, diff);
    }

    let mut sum = hsum_neon(sum_acc);

    let start = chunks * 4;
    for i in 0..remainder {
        let diff = *a_ptr.add(start + i) - *b_ptr.add(start + i);
        sum += diff * diff;
    }

    sum
}

/// # Safety
/// Requires aarch64 with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_product_neg_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut dot_acc = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f32(a_ptr.add(i * 4));
        let vb = vld1q_f32(b_ptr.add(i * 4));
        dot_acc = vfmaq_f32(dot_acc, va, vb);
    }

    let mut dot = hsum_neon(dot_acc);

    let start = chunks * 4;
    for i in 0..remainder {
        dot += *a_ptr.add(start + i) * *b_ptr.add(start + i);
    }

    -dot
}

#[cfg(test)]
mod tests {
    #[allow(dead_code)]
    /// Use relative epsilon for comparing SIMD vs scalar results.
    fn approx_eq(a: f32, b: f32) -> bool {
        let abs_diff = (a - b).abs();
        let max_val = a.abs().max(b.abs()).max(1.0);
        abs_diff / max_val < 1e-5
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_cosine_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        use super::super::cosine::cosine_distance_scalar;

        let test_dims = [1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 128, 384, 768, 1536];
        for &dim in &test_dims {
            let a: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.1).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.2).cos()).collect();

            let scalar = cosine_distance_scalar(&a, &b);
            let avx2 = unsafe { super::cosine_distance_avx2(&a, &b) };

            assert!(
                approx_eq(scalar, avx2),
                "Cosine mismatch at dim={dim}: scalar={scalar}, avx2={avx2}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_euclidean_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        use super::super::euclidean::euclidean_distance_sq_scalar;

        let test_dims = [1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 128, 384, 768, 1536];
        for &dim in &test_dims {
            let a: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.1).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.2).cos()).collect();

            let scalar = euclidean_distance_sq_scalar(&a, &b);
            let avx2 = unsafe { super::euclidean_distance_sq_avx2(&a, &b) };

            assert!(
                approx_eq(scalar, avx2),
                "Euclidean mismatch at dim={dim}: scalar={scalar}, avx2={avx2}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_dot_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        use super::super::dot::dot_product_neg_scalar;

        let test_dims = [1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 128, 384, 768, 1536];
        for &dim in &test_dims {
            let a: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.1).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.2).cos()).collect();

            let scalar = dot_product_neg_scalar(&a, &b);
            let avx2 = unsafe { super::dot_product_neg_avx2(&a, &b) };

            assert!(
                approx_eq(scalar, avx2),
                "Dot product mismatch at dim={dim}: scalar={scalar}, avx2={avx2}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_cosine_matches_scalar() {
        use super::super::cosine::cosine_distance_scalar;

        let test_dims = [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 128, 384, 768, 1536];
        for &dim in &test_dims {
            let a: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.1).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.2).cos()).collect();

            let scalar = cosine_distance_scalar(&a, &b);
            let neon = unsafe { super::cosine_distance_neon(&a, &b) };

            assert!(
                approx_eq(scalar, neon),
                "Cosine mismatch at dim={dim}: scalar={scalar}, neon={neon}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_euclidean_matches_scalar() {
        use super::super::euclidean::euclidean_distance_sq_scalar;

        let test_dims = [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 128, 384, 768, 1536];
        for &dim in &test_dims {
            let a: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.1).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.2).cos()).collect();

            let scalar = euclidean_distance_sq_scalar(&a, &b);
            let neon = unsafe { super::euclidean_distance_sq_neon(&a, &b) };

            assert!(
                approx_eq(scalar, neon),
                "Euclidean mismatch at dim={dim}: scalar={scalar}, neon={neon}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_matches_scalar() {
        use super::super::dot::dot_product_neg_scalar;

        let test_dims = [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 128, 384, 768, 1536];
        for &dim in &test_dims {
            let a: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.1).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.2).cos()).collect();

            let scalar = dot_product_neg_scalar(&a, &b);
            let neon = unsafe { super::dot_product_neg_neon(&a, &b) };

            assert!(
                approx_eq(scalar, neon),
                "Dot product mismatch at dim={dim}: scalar={scalar}, neon={neon}"
            );
        }
    }
}
