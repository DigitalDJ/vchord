// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::emulate::partial;
use crate::f16;

/// Adds `N` accumulators as a tree rather than a chain.
#[inline(always)]
fn reduce_pairwise<T: Copy, const N: usize>(mut x: [T; N], add: impl Fn(T, T) -> T) -> T {
    let mut n = N;
    while n > 1 {
        let h = n / 2;
        for i in 0..h {
            x[i] = add(x[i], x[n - h + i]);
        }
        n -= h;
    }
    x[0]
}

/// Folds equal-length `inputs` `W` elements at a time, through `U` accumulators once `P` unrolled
/// passes fit. `step` gets pointers rather than indices: indexed addressing is slower in these loops.
#[inline(always)]
pub fn fold<T, A: Copy, const K: usize, const W: usize, const U: usize, const P: usize>(
    inputs: [&[T]; K],
    init: A,
    merge: impl Fn(A, A) -> A,
    mut step: impl FnMut(A, [*const T; K], usize) -> A,
) -> A {
    let n = inputs[0].len();
    assert!(inputs.iter().all(|x| x.len() == n));
    let mut at = inputs.map(<[T]>::as_ptr);
    let mut step = |acc, len| {
        let acc = step(acc, at, len);
        at = at.map(|p| unsafe { p.add(len) });
        acc
    };
    let mut i = 0;
    if n >= P * W * U {
        let mut accs = [init; U];
        while n - i >= W * U {
            for acc in &mut accs {
                *acc = step(*acc, W);
            }
            i += W * U;
        }
        // Leftover whole vectors go round-robin to the accumulators, the tail to the last one.
        for acc in &mut accs[..U - 1] {
            if n - i >= W {
                *acc = step(*acc, W);
                i += W;
            }
        }
        if i < n {
            accs[U - 1] = step(accs[U - 1], n - i);
        }
        return reduce_pairwise(accs, merge);
    }
    let mut acc = init;
    while n - i >= W {
        acc = step(acc, W);
        i += W;
    }
    if i < n {
        acc = step(acc, n - i);
    }
    acc
}

/// Element types loaded as f32 lanes; a short load fills the rest with `fill`.
pub trait Lanes: Copy + Default {
    #[cfg(target_arch = "x86_64")]
    unsafe fn load_m512(p: *const Self, len: usize, fill: Self) -> core::arch::x86_64::__m512;

    #[cfg(target_arch = "x86_64")]
    unsafe fn load_m256(p: *const Self, len: usize, fill: Self) -> core::arch::x86_64::__m256;

    #[cfg(target_arch = "aarch64")]
    unsafe fn load_f32x4x2(
        p: *const Self,
        len: usize,
        fill: Self,
    ) -> [core::arch::aarch64::float32x4_t; 2];
}

impl Lanes for f32 {
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load_m512(p: *const f32, len: usize, fill: f32) -> core::arch::x86_64::__m512 {
        unsafe { v4::load(p, len, fill) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load_m256(p: *const f32, len: usize, fill: f32) -> core::arch::x86_64::__m256 {
        use core::arch::x86_64::*;
        unsafe {
            if len == 8 {
                _mm256_loadu_ps(p)
            } else {
                _mm256_loadu_ps(partial::<f32, 8>(p, len, fill).as_ptr())
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn load_f32x4x2(
        p: *const f32,
        len: usize,
        fill: f32,
    ) -> [core::arch::aarch64::float32x4_t; 2] {
        use core::arch::aarch64::*;
        unsafe {
            if len == 8 {
                [vld1q_f32(p), vld1q_f32(p.add(4))]
            } else {
                let x = partial::<f32, 8>(p, len, fill);
                [vld1q_f32(x.as_ptr()), vld1q_f32(x.as_ptr().add(4))]
            }
        }
    }
}

impl Lanes for f16 {
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load_m512(p: *const f16, len: usize, fill: f16) -> core::arch::x86_64::__m512 {
        use core::arch::x86_64::*;
        unsafe {
            if len == 16 {
                _mm512_cvtph_ps(_mm256_loadu_si256(p.cast()))
            } else {
                let mask = _bzhi_u32(0xffff, len as u32) as u16;
                let fill = _mm256_set1_epi16(fill.to_bits() as i16);
                _mm512_cvtph_ps(_mm256_mask_loadu_epi16(fill, mask, p.cast()))
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load_m256(p: *const f16, len: usize, fill: f16) -> core::arch::x86_64::__m256 {
        use core::arch::x86_64::*;
        unsafe {
            if len == 8 {
                _mm256_cvtph_ps(_mm_loadu_si128(p.cast()))
            } else {
                let x = partial::<f16, 8>(p, len, fill);
                _mm256_cvtph_ps(_mm_loadu_si128(x.as_ptr().cast()))
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn load_f32x4x2(
        p: *const f16,
        len: usize,
        fill: f16,
    ) -> [core::arch::aarch64::float32x4_t; 2] {
        use core::arch::aarch64::*;
        unsafe {
            let x = a2::load_f16x8(p, len, fill);
            [vcvt_f32_f16(vget_low_f16(x)), vcvt_high_f32_f16(x)]
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub mod v4 {
    use super::Lanes;
    use core::arch::x86_64::*;

    #[inline(always)]
    pub fn fold<T, A: Copy, const K: usize, const U: usize>(
        inputs: [&[T]; K],
        init: A,
        merge: impl Fn(A, A) -> A,
        step: impl FnMut(A, [*const T; K], usize) -> A,
    ) -> A {
        super::fold::<T, A, K, 16, U, 2>(inputs, init, merge, step)
    }

    #[inline(always)]
    pub unsafe fn load(p: *const f32, len: usize, fill: f32) -> __m512 {
        unsafe {
            if len == 16 {
                _mm512_loadu_ps(p)
            } else {
                let mask = _bzhi_u32(0xffff, len as u32) as u16;
                _mm512_mask_loadu_ps(_mm512_set1_ps(fill), mask, p)
            }
        }
    }

    /// Lane sum of `step` folded over equal-length `inputs`.
    #[inline]
    #[crate::target_cpu(enable = "v4")]
    pub fn sum<const U: usize, const K: usize, T: Lanes>(
        inputs: [&[T]; K],
        step: impl Fn(__m512, [__m512; K]) -> __m512,
    ) -> f32 {
        let add = |x, y| _mm512_add_ps(x, y);
        let sum = fold::<_, _, K, U>(inputs, _mm512_setzero_ps(), add, |s, at, len| {
            step(s, at.map(|p| unsafe { T::load_m512(p, len, T::default()) }))
        });
        _mm512_reduce_add_ps(sum)
    }
}

#[cfg(target_arch = "x86_64")]
pub mod v3 {
    use super::Lanes;
    use crate::emulate::emulate_mm256_reduce_add_ps;
    use core::arch::x86_64::*;

    #[inline(always)]
    pub fn fold<T, A: Copy, const K: usize, const U: usize>(
        inputs: [&[T]; K],
        init: A,
        merge: impl Fn(A, A) -> A,
        step: impl FnMut(A, [*const T; K], usize) -> A,
    ) -> A {
        super::fold::<T, A, K, 8, U, 2>(inputs, init, merge, step)
    }

    /// Lane sum of `step` folded over equal-length `inputs`.
    #[inline]
    #[crate::target_cpu(enable = "v3")]
    pub fn sum<const U: usize, const K: usize, T: Lanes>(
        inputs: [&[T]; K],
        step: impl Fn(__m256, [__m256; K]) -> __m256,
    ) -> f32 {
        let add = |x, y| _mm256_add_ps(x, y);
        let sum = fold::<_, _, K, U>(inputs, _mm256_setzero_ps(), add, |s, at, len| {
            step(s, at.map(|p| unsafe { T::load_m256(p, len, T::default()) }))
        });
        emulate_mm256_reduce_add_ps(sum)
    }
}

#[cfg(target_arch = "x86_64")]
pub mod v2 {
    use super::partial;
    use crate::emulate::emulate_mm_reduce_add_ps;
    use core::arch::x86_64::*;

    #[inline(always)]
    pub fn fold<T, A: Copy, const K: usize, const U: usize>(
        inputs: [&[T]; K],
        init: A,
        merge: impl Fn(A, A) -> A,
        step: impl FnMut(A, [*const T; K], usize) -> A,
    ) -> A {
        super::fold::<T, A, K, 4, U, 2>(inputs, init, merge, step)
    }

    #[inline(always)]
    pub unsafe fn load(p: *const f32, len: usize, fill: f32) -> __m128 {
        unsafe {
            if len == 4 {
                _mm_loadu_ps(p)
            } else {
                _mm_loadu_ps(partial::<f32, 4>(p, len, fill).as_ptr())
            }
        }
    }

    /// Lane sum of `step` folded over equal-length `inputs`.
    #[inline]
    #[crate::target_cpu(enable = "v2")]
    pub fn sum<const U: usize, const K: usize>(
        inputs: [&[f32]; K],
        step: impl Fn(__m128, [__m128; K]) -> __m128,
    ) -> f32 {
        let add = |x, y| _mm_add_ps(x, y);
        let sum = fold::<_, _, K, U>(inputs, _mm_setzero_ps(), add, |s, at, len| {
            step(s, at.map(|p| unsafe { load(p, len, 0.0) }))
        });
        emulate_mm_reduce_add_ps(sum)
    }
}

#[cfg(target_arch = "aarch64")]
pub mod a2 {
    use super::{Lanes, partial};
    use crate::f16;
    use core::arch::aarch64::*;

    #[inline(always)]
    pub fn fold<T, A: Copy, const K: usize, const U: usize>(
        inputs: [&[T]; K],
        init: A,
        merge: impl Fn(A, A) -> A,
        step: impl FnMut(A, [*const T; K], usize) -> A,
    ) -> A {
        super::fold::<T, A, K, 8, U, 1>(inputs, init, merge, step)
    }

    pub type F32x8 = [float32x4_t; 2];

    #[inline(always)]
    pub unsafe fn load_f16x8(p: *const f16, len: usize, fill: f16) -> float16x8_t {
        use crate::emulate::emulate_vreinterpretq_f16_u16;
        unsafe {
            if len == 8 {
                emulate_vreinterpretq_f16_u16(vld1q_u16(p.cast()))
            } else {
                let x = partial::<f16, 8>(p, len, fill);
                emulate_vreinterpretq_f16_u16(vld1q_u16(x.as_ptr().cast()))
            }
        }
    }

    /// Lane sum of `step` folded over equal-length `inputs`.
    #[inline]
    #[crate::target_cpu(enable = "a2")]
    pub fn sum<const U: usize, const K: usize, T: Lanes>(
        inputs: [&[T]; K],
        step: impl Fn(F32x8, [F32x8; K]) -> F32x8,
    ) -> f32 {
        sum_by::<U, K, T, _>(
            inputs,
            |p, len| unsafe { T::load_f32x4x2(p, len, T::default()) },
            step,
        )
    }

    /// `sum` over unconverted halves, for the widening FHM instructions.
    #[inline]
    #[crate::target_cpu(enable = "a2")]
    pub fn sum_f16<const U: usize, const K: usize>(
        inputs: [&[f16]; K],
        step: impl Fn(F32x8, [float16x8_t; K]) -> F32x8,
    ) -> f32 {
        sum_by::<U, K, f16, _>(
            inputs,
            |p, len| unsafe { load_f16x8(p, len, f16::default()) },
            step,
        )
    }

    #[inline(always)]
    fn sum_by<const U: usize, const K: usize, T, V: Copy>(
        inputs: [&[T]; K],
        load: impl Fn(*const T, usize) -> V,
        step: impl Fn(F32x8, [V; K]) -> F32x8,
    ) -> f32 {
        let add = |x: F32x8, y: F32x8| unsafe { [vaddq_f32(x[0], y[0]), vaddq_f32(x[1], y[1])] };
        let zero = unsafe { [vdupq_n_f32(0.0); 2] };
        let sum = fold::<_, _, K, U>(inputs, zero, add, |s, at, len| {
            step(s, at.map(|p| load(p, len)))
        });
        unsafe { vaddvq_f32(vaddq_f32(sum[0], sum[1])) }
    }
}
