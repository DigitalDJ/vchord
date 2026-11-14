// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use simd::Floating;

pub mod bvect;
pub mod rabitq4;
pub mod rabitq8;
pub mod svect;
pub mod vect;

pub trait VectorOwned: Clone + 'static {
    type Borrowed<'a>: VectorBorrowed<Owned = Self>;

    fn as_borrowed(&self) -> Self::Borrowed<'_>;
}

pub trait VectorBorrowed: Copy {
    type Owned: VectorOwned;

    fn own(&self) -> Self::Owned;

    fn dim(&self) -> u32;

    fn norm(&self) -> f32;

    fn operator_dot(self, rhs: Self) -> distance::Distance;

    fn operator_l2s(self, rhs: Self) -> distance::Distance;

    fn operator_cos(self, rhs: Self) -> distance::Distance;

    fn operator_hamming(self, rhs: Self) -> distance::Distance;

    fn operator_jaccard(self, rhs: Self) -> distance::Distance;

    fn function_normalize(&self) -> Self::Owned;

    fn operator_add(&self, rhs: Self) -> Self::Owned;

    fn operator_sub(&self, rhs: Self) -> Self::Owned;

    fn operator_mul(&self, rhs: Self) -> Self::Owned;

    fn operator_and(&self, rhs: Self) -> Self::Owned;

    fn operator_or(&self, rhs: Self) -> Self::Owned;

    fn operator_xor(&self, rhs: Self) -> Self::Owned;
}

/// Cosine distance, rescaling out-of-range norms exactly; a zero vector gives NaN as in pgvector.
#[inline(always)]
pub fn cosine_distance<S: Floating>(
    xy: f32,
    x: &[S],
    y: &[S],
    rescaled_xy: impl FnOnce(&[f32], &[f32]) -> f32,
) -> f32 {
    let (x2, y2) = (S::reduce_sum_of_x2(x), S::reduce_sum_of_x2(y));
    let p = x2 * y2;
    if x2.min(y2) >= MIN_FAST_X2 && p.is_normal() {
        1.0 - xy / p.sqrt()
    } else {
        cosine_distance_rescaled(x, y, rescaled_xy)
    }
}

#[cold]
fn cosine_distance_rescaled<S: Floating>(
    x: &[S],
    y: &[S],
    xy: impl FnOnce(&[f32], &[f32]) -> f32,
) -> f32 {
    let (Some(x), Some(y)) = (rescaled(x), rescaled(y)) else {
        return f32::NAN;
    };
    1.0 - xy(&x, &y) / (f32::reduce_sum_of_x2(&x) * f32::reduce_sum_of_x2(&y)).sqrt()
}

/// Below this, subnormal squares could bias `x2`, so the vector is rescaled.
const MIN_FAST_X2: f32 = 1e-30;

/// The power of two that brings the largest magnitude near 1, or `None` for a zero vector.
fn scale<S: Floating>(values: &[S]) -> Option<f32> {
    let (min, max) = S::reduce_min_max_of_x(values);
    let m = max.max(-min);
    (m > 0.0).then(|| {
        let e = ((m.to_bits() >> 23) as i32 - 127).min(126);
        f32::from_bits(((127 - e) as u32) << 23)
    })
}

fn rescaled<S: Floating>(values: &[S]) -> Option<Vec<f32>> {
    let s = scale(values)?;
    Some(f32::vector_mul_scalar(
        S::vector_to_f32_borrowed(values).as_ref(),
        s,
    ))
}

#[test]
fn cosine_distance_extremes() {
    let cos = |x: &[f32], y: &[f32]| {
        cosine_distance(f32::reduce_sum_of_xy(x, y), x, y, |x, y| {
            f32::reduce_sum_of_xy(x, y)
        })
    };
    let tiny = [1.0e-13f32; 3];
    assert!(cos(&tiny, &tiny).abs() < 1.0e-6);
    let tinier = [1.0e-23f32, 2.0e-23, 0.0];
    assert!(cos(&tinier, &tinier).abs() < 1.0e-6);
    assert!((cos(&tinier, &[2.0, 4.0, 0.0])).abs() < 1.0e-6);
    assert!((cos(&tinier, &[0.0, 0.0, 1.0]) - 1.0).abs() < 1.0e-6);
    let huge = [1.0e19f32, 1.0e19];
    assert!(cos(&huge, &huge).abs() < 1.0e-6);
    let huge_opposite = [-1.0e30f32, -1.0e30];
    let huge_positive = [1.0e30f32, 1.0e30];
    assert!((cos(&huge_positive, &huge_opposite) - 2.0).abs() < 1.0e-6);
    let max = [f32::MAX, f32::MAX, 0.0];
    assert!((cos(&max, &tinier) - (1.0 - 3.0 / 10f32.sqrt())).abs() < 1.0e-6);
    let sub = [1.0e-40f32, -1.0e-40];
    assert!((cos(&sub, &[1.0, -1.0])).abs() < 1.0e-6);
    assert!(cos(&[0.0, 0.0], &[1.0, 0.0]).is_nan());
    assert!(cos(&[0.0, 0.0], &[0.0, 0.0]).is_nan());
    // Subnormal squares: rescaled exactly like the fast path on the scaled-up vector.
    let up = |x: &[f32]| x.iter().map(|&x| x * 2f32.powi(70)).collect::<Vec<_>>();
    let sub = vec![1.1e-21f32; 65535];
    let mut ones = vec![1.0f32; 65535];
    assert_eq!(cos(&sub, &ones).to_bits(), cos(&up(&sub), &ones).to_bits());
    assert!(cos(&sub, &ones).abs() < 1e-4);
    ones[0] = -65533.0;
    assert_eq!(cos(&sub, &ones).to_bits(), cos(&up(&sub), &ones).to_bits());
    assert!((cos(&sub, &ones) - 1.0).abs() < 1e-4);
}
