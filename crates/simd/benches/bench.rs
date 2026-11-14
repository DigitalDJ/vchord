// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

#![allow(unsafe_code)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use std::hint::black_box;

/// Short inputs, common embedding widths and an odd length with a tail.
const SIZES: [usize; 7] = [16, 64, 512, 768, 1024, 1536, 4095];

/// Copies `x` to a cache-line boundary so timings do not depend on the allocator.
fn cache_aligned<T: Copy + Default>(x: &[T]) -> &'static [T] {
    let buffer = vec![T::default(); x.len() + 64].leak();
    let offset = buffer.as_ptr().align_offset(64);
    let r = &mut buffer[offset..offset + x.len()];
    r.copy_from_slice(x);
    r
}

fn random_f32(n: usize) -> &'static [f32] {
    use rand::RngExt;
    let mut rng = rand::rng();
    cache_aligned(
        &(0..n)
            .map(|_| rng.random_range(-1.0..=1.0f32))
            .collect::<Vec<_>>(),
    )
}

fn random_u8(n: usize) -> &'static [u8] {
    use rand::RngExt;
    let mut rng = rand::rng();
    cache_aligned(&(0..n).map(|_| rng.random::<u8>()).collect::<Vec<_>>())
}

fn random_f16(n: usize) -> &'static [simd::f16] {
    use simd::F16;
    let x = random_f32(n).iter().map(|&x| simd::f16::_from_f32(x));
    cache_aligned(&x.collect::<Vec<_>>())
}

fn unary<T: 'static, R>(
    c: &mut Criterion,
    name: &str,
    data: fn(usize) -> &'static [T],
    implementations: Vec<(&str, fn(&[T]) -> R)>,
) {
    let mut group = c.benchmark_group(name);
    for n in SIZES {
        let x = data(n);
        for &(target, f) in &implementations {
            group.bench_with_input(BenchmarkId::new(target, n), x, |b, x| {
                b.iter(|| f(black_box(x)))
            });
        }
    }
    group.finish();
}

fn binary<T: 'static, R>(
    c: &mut Criterion,
    name: &str,
    data: fn(usize) -> &'static [T],
    implementations: Vec<(&str, fn(&[T], &[T]) -> R)>,
) {
    let mut group = c.benchmark_group(name);
    for n in SIZES {
        let (x, y) = (data(n), data(n));
        for &(target, f) in &implementations {
            group.bench_with_input(BenchmarkId::new(target, n), &(x, y), |b, (x, y)| {
                b.iter(|| f(black_box(x), black_box(y)))
            });
        }
    }
    group.finish();
}

fn floating(c: &mut Criterion) {
    use simd::{floating_f16 as h, floating_f32 as s};
    unary(
        c,
        "floating_f32::reduce_sum_of_abs_x",
        random_f32,
        s::reduce_sum_of_abs_x::implementations(),
    );
    unary(
        c,
        "floating_f32::reduce_sum_of_x2",
        random_f32,
        s::reduce_sum_of_x2::implementations(),
    );
    unary(
        c,
        "floating_f32::reduce_min_max_of_x",
        random_f32,
        s::reduce_min_max_of_x::implementations(),
    );
    binary(
        c,
        "floating_f32::reduce_sum_of_xy",
        random_f32,
        s::reduce_sum_of_xy::implementations(),
    );
    binary(
        c,
        "floating_f32::reduce_sum_of_d2",
        random_f32,
        s::reduce_sum_of_d2::implementations(),
    );
    unary(
        c,
        "floating_f16::reduce_sum_of_x2",
        random_f16,
        h::reduce_sum_of_x2::implementations(),
    );
    binary(
        c,
        "floating_f16::reduce_sum_of_xy",
        random_f16,
        h::reduce_sum_of_xy::implementations(),
    );
    binary(
        c,
        "floating_f16::reduce_sum_of_d2",
        random_f16,
        h::reduce_sum_of_d2::implementations(),
    );
}

// RaBitQ's binary and block query LUTs.
fn quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize");
    for (len, n) in [(768, 63.0), (3072, 255.0)] {
        let lut = random_f32(len);
        group.bench_with_input(BenchmarkId::from_parameter(len), lut, |b, lut| {
            b.iter(|| simd::quantize::quantize(black_box(lut), n))
        });
    }
    group.finish();
}

fn byte(c: &mut Criterion) {
    use simd::{byte, halfbyte};
    binary(
        c,
        "byte::reduce_sum_of_xy",
        random_u8,
        byte::reduce_sum_of_xy::implementations(),
    );
    unary(
        c,
        "byte::reduce_sum_of_x",
        random_u8,
        byte::reduce_sum_of_x::implementations(),
    );
    binary(
        c,
        "halfbyte::reduce_sum_of_xy",
        random_u8,
        halfbyte::reduce_sum_of_xy::implementations(),
    );
}

fn column_bench<T: simd::column::Value>(c: &mut Criterion, name: &str, random: impl Fn() -> T) {
    use simd::column::Set;
    for n in [1, 4, 10, 16, 32, 33, 64, 200, 5000] {
        let mut keys = (0..n).map(|_| random()).collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();
        let column = std::array::from_fn::<T, 32, _>(|i| {
            if i % 4 == 0 {
                keys[i % keys.len()]
            } else {
                random()
            }
        });
        if n <= 64 {
            c.bench_function(&format!("column::matches::{name}/{n}"), |b| {
                b.iter(|| T::matches(black_box(&column), black_box(&keys)))
            });
        }
        let set = Set::new(&keys);
        c.bench_function(&format!("column::set::{name}/{n}"), |b| {
            b.iter(|| set.matches(black_box(&column)))
        });
    }
}

fn column_matches(c: &mut Criterion) {
    column_bench(c, "u16", rand::random::<u16>);
    column_bench(c, "u32", rand::random::<u32>);
    column_bench(c, "u64", rand::random::<u64>);
    column_bench(c, "u128", rand::random::<[u64; 2]>);
}

fn column_meets(c: &mut Criterion) {
    use rand::RngExt;
    let mut rng = rand::rng();
    let ranges = std::array::from_fn::<u32, 32, _>(|_| {
        let min = rng.random::<u16>();
        min as u32 | (min.saturating_add(rng.random::<u8>() as u16) as u32) << 16
    });
    let keys = (0..32).map(|_| rng.random::<u16>()).collect::<Vec<_>>();
    for n in [1, 4, 16, 32] {
        let keys = &keys[..n];
        #[cfg(target_arch = "x86_64")]
        if simd::is_cpu_detected!("v4") {
            c.bench_function(&format!("column::meets::v4/{n}"), |b| {
                b.iter(|| unsafe { simd::column::meets::meets_v4(&ranges, keys) })
            });
        }
        #[cfg(target_arch = "x86_64")]
        if simd::is_cpu_detected!("v3") {
            c.bench_function(&format!("column::meets::v3/{n}"), |b| {
                b.iter(|| unsafe { simd::column::meets::meets_v3(&ranges, keys) })
            });
        }
        #[cfg(target_arch = "aarch64")]
        if simd::is_cpu_detected!("a2") {
            c.bench_function(&format!("column::meets::a2/{n}"), |b| {
                b.iter(|| unsafe { simd::column::meets::meets_a2(&ranges, keys) })
            });
        }
        c.bench_function(&format!("column::meets/{n}"), |b| {
            b.iter(|| simd::column::meets(&ranges, keys))
        });
    }
}

criterion_group!(
    benches,
    floating,
    quantize,
    byte,
    column_matches,
    column_meets
);
criterion_main!(benches);
