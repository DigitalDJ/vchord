// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::Floating;

impl Floating for f32 {
    #[inline(always)]
    fn reduce_sum_of_abs_x(this: &[f32]) -> f32 {
        reduce_sum_of_abs_x::reduce_sum_of_abs_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_x2(this: &[f32]) -> f32 {
        reduce_sum_of_x2::reduce_sum_of_x2(this)
    }

    #[inline(always)]
    fn reduce_min_max_of_x(this: &[f32]) -> (f32, f32) {
        reduce_min_max_of_x::reduce_min_max_of_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_xy(lhs: &[Self], rhs: &[Self]) -> f32 {
        reduce_sum_of_xy::reduce_sum_of_xy(lhs, rhs)
    }

    #[inline(always)]
    fn reduce_sum_of_d2(lhs: &[Self], rhs: &[Self]) -> f32 {
        reduce_sum_of_d2::reduce_sum_of_d2(lhs, rhs)
    }

    #[inline(always)]
    fn vector_add(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        vector_add::vector_add(lhs, rhs)
    }

    #[inline(always)]
    fn vector_add_inplace(lhs: &mut [f32], rhs: &[f32]) {
        vector_add_inplace::vector_add_inplace(lhs, rhs)
    }

    #[inline(always)]
    fn vector_sub(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        vector_sub::vector_sub(lhs, rhs)
    }

    #[inline(always)]
    fn vector_mul(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        vector_mul::vector_mul(lhs, rhs)
    }

    #[inline(always)]
    fn vector_mul_scalar(lhs: &[f32], rhs: f32) -> Vec<f32> {
        vector_mul_scalar::vector_mul_scalar(lhs, rhs)
    }

    #[inline(always)]
    fn vector_mul_scalar_inplace(lhs: &mut [f32], rhs: f32) {
        vector_mul_scalar_inplace::vector_mul_scalar_inplace(lhs, rhs);
    }

    #[inline(always)]
    fn vector_from_f32(this: &[f32]) -> Vec<f32> {
        this.to_vec()
    }

    #[inline(always)]
    fn vector_to_f32(this: &[f32]) -> Vec<f32> {
        this.to_vec()
    }

    #[inline(always)]
    fn vector_to_f32_borrowed(this: &[Self]) -> impl AsRef<[f32]> {
        this
    }

    #[inline(always)]
    fn vector_abs_inplace(this: &mut [Self]) {
        vector_abs_inplace::vector_abs_inplace(this);
    }
}

// Accumulator counts are the smallest that reach full throughput.

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_abs_x {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_abs_x_v4(this: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        crate::reduce::v4::sum::<4, 1>([this], |s, [x]| _mm512_add_ps(_mm512_abs_ps(x), s))
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_abs_x_v3(this: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        let abs = _mm256_castsi256_ps(_mm256_srli_epi32(_mm256_set1_epi32(-1), 1));
        crate::reduce::v3::sum::<8, 1, _>([this], |s, [x]| _mm256_add_ps(_mm256_and_ps(abs, x), s))
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn reduce_sum_of_abs_x_v2(this: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        let abs = _mm_castsi128_ps(_mm_srli_epi32(_mm_set1_epi32(-1), 1));
        crate::reduce::v2::sum::<8, 1>([this], |s, [x]| _mm_add_ps(_mm_and_ps(abs, x), s))
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_abs_x_a2(this: &[f32]) -> f32 {
        use core::arch::aarch64::*;
        crate::reduce::a2::sum::<4, 1, _>([this], |s, [x]| {
            [
                vaddq_f32(s[0], vabsq_f32(x[0])),
                vaddq_f32(s[1], vabsq_f32(x[1])),
            ]
        })
    }

    #[crate::multiversion(@"v4", @"v3", @"v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_abs_x(this: &[f32]) -> f32 {
        let n = this.len();
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += this[i].abs();
        }
        sum
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_abs_x_test() {
        crate::test_util::check_unary(implementations(), f64::abs);
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_x2 {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_x2_v4(this: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        crate::reduce::v4::sum::<4, 1>([this], |s, [x]| _mm512_fmadd_ps(x, x, s))
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_x2_v3(this: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        crate::reduce::v3::sum::<8, 1, _>([this], |s, [x]| _mm256_fmadd_ps(x, x, s))
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    #[target_feature(enable = "fma")]
    fn reduce_sum_of_x2_v2_fma(this: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        crate::reduce::v2::sum::<8, 1>([this], |s, [x]| _mm_fmadd_ps(x, x, s))
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_x2_a2(this: &[f32]) -> f32 {
        use core::arch::aarch64::*;
        crate::reduce::a2::sum::<4, 1, _>([this], |s, [x]| {
            [vfmaq_f32(s[0], x[0], x[0]), vfmaq_f32(s[1], x[1], x[1])]
        })
    }

    #[crate::multiversion(@"v4", @"v3", @"v2:fma", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_x2(this: &[f32]) -> f32 {
        let n = this.len();
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += this[i] * this[i];
        }
        sum
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x2_test() {
        crate::test_util::check_unary(implementations(), |x| x * x);
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_min_max_of_x {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_min_max_of_x_v4(this: &[f32]) -> (f32, f32) {
        use crate::reduce::v4;
        use core::arch::x86_64::*;
        let init = (
            _mm512_set1_ps(f32::INFINITY),
            _mm512_set1_ps(f32::NEG_INFINITY),
        );
        let merge = |(a, b), (c, d)| (_mm512_min_ps(a, c), _mm512_max_ps(b, d));
        // Repeating an element in the unused lanes leaves min and max unchanged.
        let (min, max) = v4::fold::<_, _, 1, 4>([this], init, merge, |(min, max), [p], len| {
            let x = unsafe { v4::load(p, len, p.read()) };
            (_mm512_min_ps(x, min), _mm512_max_ps(x, max))
        });
        (_mm512_reduce_min_ps(min), _mm512_reduce_max_ps(max))
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_min_max_of_x_v3(this: &[f32]) -> (f32, f32) {
        use crate::emulate::{emulate_mm256_reduce_max_ps, emulate_mm256_reduce_min_ps};
        use crate::reduce::{Lanes, v3};
        use core::arch::x86_64::*;
        let init = (
            _mm256_set1_ps(f32::INFINITY),
            _mm256_set1_ps(f32::NEG_INFINITY),
        );
        let merge = |(a, b), (c, d)| (_mm256_min_ps(a, c), _mm256_max_ps(b, d));
        let (min, max) = v3::fold::<_, _, 1, 4>([this], init, merge, |(min, max), [p], len| {
            let x = unsafe { f32::load_m256(p, len, p.read()) };
            (_mm256_min_ps(x, min), _mm256_max_ps(x, max))
        });
        (
            emulate_mm256_reduce_min_ps(min),
            emulate_mm256_reduce_max_ps(max),
        )
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn reduce_min_max_of_x_v2(this: &[f32]) -> (f32, f32) {
        use crate::emulate::{emulate_mm_reduce_max_ps, emulate_mm_reduce_min_ps};
        use crate::reduce::v2;
        use core::arch::x86_64::*;
        let init = (_mm_set1_ps(f32::INFINITY), _mm_set1_ps(f32::NEG_INFINITY));
        let merge = |(a, b), (c, d)| (_mm_min_ps(a, c), _mm_max_ps(b, d));
        let (min, max) = v2::fold::<_, _, 1, 4>([this], init, merge, |(min, max), [p], len| {
            let x = unsafe { v2::load(p, len, p.read()) };
            (_mm_min_ps(x, min), _mm_max_ps(x, max))
        });
        (emulate_mm_reduce_min_ps(min), emulate_mm_reduce_max_ps(max))
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_min_max_of_x_a2(this: &[f32]) -> (f32, f32) {
        use crate::reduce::{Lanes, a2};
        use core::arch::aarch64::*;
        let init = (
            [vdupq_n_f32(f32::INFINITY); 2],
            [vdupq_n_f32(f32::NEG_INFINITY); 2],
        );
        let min = |x: a2::F32x8, y: a2::F32x8| [vminnmq_f32(x[0], y[0]), vminnmq_f32(x[1], y[1])];
        let max = |x: a2::F32x8, y: a2::F32x8| [vmaxnmq_f32(x[0], y[0]), vmaxnmq_f32(x[1], y[1])];
        let merge = |(a, b), (c, d)| (min(a, c), max(b, d));
        let (lo, hi) = a2::fold::<_, _, 1, 2>([this], init, merge, |(lo, hi), [p], len| {
            let x = unsafe { f32::load_f32x4x2(p, len, p.read()) };
            (min(x, lo), max(x, hi))
        });
        let (lo, hi) = (vminnmq_f32(lo[0], lo[1]), vmaxnmq_f32(hi[0], hi[1]));
        (vminnmvq_f32(lo), vmaxnmvq_f32(hi))
    }

    #[crate::multiversion(@"v4", @"v3", @"v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_min_max_of_x(this: &[f32]) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let n = this.len();
        for i in 0..n {
            min = min.min(this[i]);
            max = max.max(this[i]);
        }
        (min, max)
    }

    /// Includes all-positive and all-negative inputs, where a wrong tail fill would show.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_min_max_of_x_test() {
        for n in crate::test_util::lengths() {
            for (lo, hi) in [(-1.0, 1.0), (1.0, 2.0), (-2.0, -1.0)] {
                let x = crate::test_util::random::<f32>(n, lo, hi);
                let expected = fallback(&x);
                for (name, f) in implementations() {
                    let got = f(&x);
                    assert_eq!(got, expected, "{name}: n = {n} in [{lo}, {hi}]");
                }
            }
        }
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_xy {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_xy_v4(lhs: &[f32], rhs: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        crate::reduce::v4::sum::<4, 2>([lhs, rhs], |s, [x, y]| _mm512_fmadd_ps(x, y, s))
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_xy_v3(lhs: &[f32], rhs: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        crate::reduce::v3::sum::<8, 2, _>([lhs, rhs], |s, [x, y]| _mm256_fmadd_ps(x, y, s))
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    #[target_feature(enable = "fma")]
    fn reduce_sum_of_xy_v2_fma(lhs: &[f32], rhs: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        crate::reduce::v2::sum::<8, 2>([lhs, rhs], |s, [x, y]| _mm_fmadd_ps(x, y, s))
    }

    #[inline]
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[crate::target_cpu(enable = "a3.256")]
    fn reduce_sum_of_xy_a3_256(lhs: &[f32], rhs: &[f32]) -> f32 {
        unsafe extern "C" {
            #[link_name = "fp32_reduce_sum_of_xy_a3_256"]
            unsafe fn f(n: usize, a: *const f32, b: *const f32) -> f32;
        }
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let a = lhs.as_ptr();
        let b = rhs.as_ptr();
        unsafe { f(n, a, b) }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_xy_a2(lhs: &[f32], rhs: &[f32]) -> f32 {
        use core::arch::aarch64::*;
        crate::reduce::a2::sum::<4, 2, _>([lhs, rhs], |s, [x, y]| {
            [vfmaq_f32(s[0], x[0], y[0]), vfmaq_f32(s[1], x[1], y[1])]
        })
    }

    #[crate::multiversion(@"v4", @"v3", @"v2:fma", #[cfg(target_endian = "little")] @"a3.256", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_xy(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += lhs[i] * rhs[i];
        }
        sum
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_test() {
        crate::test_util::check_binary(implementations(), |x, y| x * y);
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_d2 {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_d2_v4(lhs: &[f32], rhs: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        crate::reduce::v4::sum::<4, 2>([lhs, rhs], |s, [x, y]| {
            let d = _mm512_sub_ps(x, y);
            _mm512_fmadd_ps(d, d, s)
        })
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_d2_v3(lhs: &[f32], rhs: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        crate::reduce::v3::sum::<8, 2, _>([lhs, rhs], |s, [x, y]| {
            let d = _mm256_sub_ps(x, y);
            _mm256_fmadd_ps(d, d, s)
        })
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    #[target_feature(enable = "fma")]
    fn reduce_sum_of_d2_v2_fma(lhs: &[f32], rhs: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        crate::reduce::v2::sum::<8, 2>([lhs, rhs], |s, [x, y]| {
            let d = _mm_sub_ps(x, y);
            _mm_fmadd_ps(d, d, s)
        })
    }

    #[inline]
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[crate::target_cpu(enable = "a3.256")]
    fn reduce_sum_of_d2_a3_256(lhs: &[f32], rhs: &[f32]) -> f32 {
        unsafe extern "C" {
            #[link_name = "fp32_reduce_sum_of_d2_a3_256"]
            unsafe fn f(n: usize, a: *const f32, b: *const f32) -> f32;
        }
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let a = lhs.as_ptr();
        let b = rhs.as_ptr();
        unsafe { f(n, a, b) }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_d2_a2(lhs: &[f32], rhs: &[f32]) -> f32 {
        use core::arch::aarch64::*;
        crate::reduce::a2::sum::<4, 2, _>([lhs, rhs], |s, [x, y]| {
            let d = [vsubq_f32(x[0], y[0]), vsubq_f32(x[1], y[1])];
            [vfmaq_f32(s[0], d[0], d[0]), vfmaq_f32(s[1], d[1], d[1])]
        })
    }

    #[crate::multiversion(@"v4", @"v3", @"v2:fma", #[cfg(target_endian = "little")] @"a3.256", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_d2(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let mut sum = 0.0f32;
        for i in 0..n {
            let d = lhs[i] - rhs[i];
            sum += d * d;
        }
        sum
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_d2_test() {
        crate::test_util::check_binary(implementations(), |x, y| (x - y).powi(2));
    }
}

mod vector_add {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_add(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<f32>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] + rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_add_inplace {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_add_inplace(lhs: &mut [f32], rhs: &[f32]) {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        for i in 0..n {
            lhs[i] += rhs[i];
        }
    }
}

mod vector_sub {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_sub(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<f32>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] - rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_mul {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_mul(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<f32>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] * rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_mul_scalar {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_mul_scalar(lhs: &[f32], rhs: f32) -> Vec<f32> {
        let n = lhs.len();
        let mut r = Vec::<f32>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] * rhs);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_mul_scalar_inplace {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_mul_scalar_inplace(lhs: &mut [f32], rhs: f32) {
        let n = lhs.len();
        for i in 0..n {
            lhs[i] *= rhs;
        }
    }
}

mod vector_abs_inplace {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_abs_inplace(this: &mut [f32]) {
        let n = this.len();
        for i in 0..n {
            this[i] = this[i].abs();
        }
    }
}
