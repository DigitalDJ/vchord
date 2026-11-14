// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

//! Columns of a block of 32 rows: one value of 2, 4, 8 or 16 bytes per row
//! (`u16`, `u32`, `u64`, `[u64; 2]`), and the rows whose value is in a set.
//! The low 16 bits of the rows' payloads are a column too ([`matches_payload`]).
//! A few values are compared with the whole column at once ([`Value::matches`]);
//! a larger set ([`Set`]) is first tested by the 16-bit folds of the values
//! ([`Value::fold`]) in a bitmap, and only the rows that pass are looked up.

/// A value of a column.
pub trait Value: Copy + Eq + Ord + Default + std::fmt::Debug + 'static {
    /// The rows whose value is one of `keys`, as a bitmask: bit `i` is row
    /// `i`. The cost is linear in the number of keys, so keep it to a few.
    fn matches(column: &[Self; 32], keys: &[Self]) -> u32;
    /// The 16-bit fold of every value of the column ([`Value::fold_one`]).
    fn fold(column: &[Self; 32]) -> [u16; 32];
    /// The exclusive or of the value's 16-bit lanes, the lowest first. A
    /// 16-bit value is its own fold.
    fn fold_one(self) -> u16;
    /// A hash of the value for a table of the given number of bits.
    fn slot(self, bits: u32) -> usize;
    /// The value whose lowest 64 bits are `x`, and the others 0.
    fn from_u64(x: u64) -> Self;
}

macro_rules! value {
    ($t:ty, $m:ident, |$x:ident| fold: $fold:expr, wide: $wide:expr, from: $from:expr) => {
        impl Value for $t {
            #[inline(always)]
            fn matches(column: &[Self; 32], keys: &[Self]) -> u32 {
                $m::matches(column, keys)
            }
            #[inline(always)]
            fn fold(column: &[Self; 32]) -> [u16; 32] {
                $m::fold(column)
            }
            #[inline(always)]
            fn fold_one(self) -> u16 {
                let $x = self;
                $fold
            }
            #[inline(always)]
            fn slot(self, bits: u32) -> usize {
                let $x = self;
                let wide: u64 = $wide;
                (wide.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - bits)) as usize
            }
            #[inline(always)]
            fn from_u64(x: u64) -> Self {
                let $x = x;
                $from
            }
        }
    };
}

value!(u16, u16s, |x| fold: x, wide: x as u64, from: x as u16);
value!(u32, u32s, |x| fold: (x ^ x >> 16) as u16, wide: x as u64, from: x as u32);
value!(u64, u64s, |x| fold: fold_u64(x), wide: x, from: x);
value!([u64; 2], u128s, |x| fold: fold_u64(x[0] ^ x[1]), wide: x[0] ^ x[1].rotate_left(32), from: [x, 0]);

#[inline(always)]
const fn fold_u64(x: u64) -> u16 {
    (x ^ (x >> 16) ^ (x >> 32) ^ (x >> 48)) as u16
}

/// A set of values, for one too large to compare value by value: a bitmap of
/// their folds, and a table of the values (none for 16-bit values, whose
/// folds are the values). The bitmap is inline, so that a set behind one
/// pointer is one load from its bitmap: a set is over 8 KiB.
#[derive(Debug)]
pub struct Set<T> {
    bitmap: [u64; 1024],
    table: Box<[T]>,
    bits: u32,
    /// A value that is not in the set, which marks the free slots.
    free: T,
}

impl<T: Value> Set<T> {
    /// `values`: sorted and deduplicated.
    pub fn new(values: &[T]) -> Self {
        let mut bitmap = [0_u64; 1024];
        for value in values {
            let x = value.fold_one();
            bitmap[x as usize / 64] |= 1 << (x % 64);
        }
        if size_of::<T>() == 2 {
            let table = Box::new([]);
            return Self {
                bitmap,
                table,
                bits: 0,
                free: T::default(),
            };
        }
        let free = (0..)
            .map(T::from_u64)
            .find(|x| values.binary_search(x).is_err())
            .expect("a set of every value");
        // at most half full
        let bits = (2 * values.len())
            .next_power_of_two()
            .trailing_zeros()
            .max(4);
        let mut table = vec![free; 1 << bits].into_boxed_slice();
        let mask = (1 << bits) - 1;
        for &value in values {
            let mut i = value.slot(bits);
            while table[i] != free {
                i = (i + 1) & mask;
            }
            table[i] = value;
        }
        Self {
            bitmap,
            table,
            bits,
            free,
        }
    }

    /// The bitmap of the folds of the values: bit `t` is fold `t`.
    pub fn bitmap(&self) -> &[u64; 1024] {
        &self.bitmap
    }

    /// Whether the set has `x`.
    #[inline(always)]
    pub fn contains(&self, x: T) -> bool {
        let f = x.fold_one();
        if self.bitmap[f as usize / 64] & 1 << (f % 64) == 0 {
            return false;
        }
        self.table.is_empty() || self.lookup(x)
    }

    #[inline(always)]
    fn lookup(&self, x: T) -> bool {
        let mask = (1 << self.bits) - 1;
        let mut i = x.slot(self.bits);
        loop {
            // a free slot first: `x` may be the value that marks them
            let y = self.table[i];
            if y == self.free {
                return false;
            }
            if y == x {
                return true;
            }
            i = (i + 1) & mask;
        }
    }

    /// The rows whose value is in the set, as a bitmask: bit `i` is row `i`.
    /// The cost does not depend on the size of the set, beyond the rows that
    /// pass the bitmap.
    #[inline(always)]
    pub fn matches(&self, column: &[T; 32]) -> u32 {
        let mut mask = matches_bitmap(&T::fold(column), &self.bitmap);
        if self.table.is_empty() {
            return mask;
        }
        let mut rest = mask;
        while rest != 0 {
            let i = rest.trailing_zeros();
            rest &= rest - 1;
            if !self.lookup(column[i as usize]) {
                mask &= !(1 << i);
            }
        }
        mask
    }
}

/// The rows whose 16-bit key, the `i`th of `key`, is in `bitmap`.
#[inline(always)]
fn bits(key: impl Fn(usize) -> usize, bitmap: &[u64; 1024]) -> u32 {
    let mut mask = 0;
    for i in 0..32 {
        let x = key(i);
        mask |= ((bitmap[x / 64] >> (x % 64)) as u32 & 1) << i;
    }
    mask
}

impl Set<u16> {
    /// [`Set::matches`] for the low 16 bits of 32 payloads.
    #[inline(always)]
    pub fn matches_payload(&self, payload: &[u64; 32]) -> u32 {
        matches_bitmap_payload(payload, &self.bitmap)
    }
}

mod matches_bitmap {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn matches_bitmap(keys: &[u16; 32], bitmap: &[u64; 1024]) -> u32 {
        super::bits(|i| keys[i] as usize, bitmap)
    }
}

mod matches_bitmap_payload {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn matches_bitmap_payload(payload: &[u64; 32], bitmap: &[u64; 1024]) -> u32 {
        super::bits(|i| payload[i] as u16 as usize, bitmap)
    }
}

/// The rows whose 16-bit key is in `bitmap` (bit `t` is key `t`), as a
/// bitmask: bit `i` is row `i`.
#[inline(always)]
pub fn matches_bitmap(keys: &[u16; 32], bitmap: &[u64; 1024]) -> u32 {
    matches_bitmap::matches_bitmap(keys, bitmap)
}

/// [`matches_bitmap`] for the low 16 bits of 32 payloads.
#[inline(always)]
pub fn matches_bitmap_payload(payload: &[u64; 32], bitmap: &[u64; 1024]) -> u32 {
    matches_bitmap_payload::matches_bitmap_payload(payload, bitmap)
}

/// [`Value::matches`] for the low 16 bits of 32 payloads, without a copy of
/// them: the same compares of the 16-bit values, after a narrowing load.
#[inline(always)]
pub fn matches_payload(payload: &[u64; 32], keys: &[u16]) -> u32 {
    payload::matches(payload, keys)
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod meets {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn meets_v4(ranges: &[u32; 32], keys: &[u16]) -> u32 {
        use core::arch::x86_64::*;
        // the lanes of the 16-bit `min` (`odd = 0`) or `max` (`odd = 1`) ends of
        // the 32 ranges, in order, among the 64 16-bit lanes of two vectors
        const fn halves(odd: u16) -> [u16; 32] {
            let mut x = [0; 32];
            let mut i = 0;
            while i < 32 {
                x[i] = 2 * i as u16 + odd;
                i += 1;
            }
            x
        }
        static MIN: [u16; 32] = halves(0);
        static MAX: [u16; 32] = halves(1);
        let p = ranges.as_ptr();
        let x0 = unsafe { _mm512_loadu_si512(p.cast()) };
        let x1 = unsafe { _mm512_loadu_si512(p.add(16).cast()) };
        let min = unsafe { _mm512_loadu_si512(MIN.as_ptr().cast()) };
        let max = unsafe { _mm512_loadu_si512(MAX.as_ptr().cast()) };
        let min = _mm512_permutex2var_epi16(x0, min, x1);
        let max = _mm512_permutex2var_epi16(x0, max, x1);
        let mut mask = 0;
        for &key in keys {
            let t = _mm512_set1_epi16(key as i16);
            mask |= _mm512_mask_cmpge_epu16_mask(_mm512_cmple_epu16_mask(min, t), max, t);
        }
        mask
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn meets_v4_test() {
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let (ranges, keys) = sample();
            for n in 0..=keys.len() {
                let specialized = unsafe { meets_v4(&ranges, &keys[..n]) };
                let fallback = fallback(&ranges, &keys[..n]);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn meets_v3(ranges: &[u32; 32], keys: &[u16]) -> u32 {
        use core::arch::x86_64::*;
        /// The ends of 16 ranges, in order, as 16-bit lanes biased by 0x8000,
        /// so that signed comparisons order them.
        #[inline]
        #[crate::target_cpu(enable = "v3")]
        fn narrow(p: &[u32; 16]) -> (__m256i, __m256i) {
            let p = p.as_ptr();
            let x0 = unsafe { _mm256_loadu_si256(p.cast()) };
            let x1 = unsafe { _mm256_loadu_si256(p.add(8).cast()) };
            let low = _mm256_set1_epi32(0xffff);
            let bias = _mm256_set1_epi16(i16::MIN);
            // per 128-bit lane: ranges 0..4 8..12 | 4..8 12..16
            let min = _mm256_packus_epi32(_mm256_and_si256(x0, low), _mm256_and_si256(x1, low));
            let max = _mm256_packus_epi32(_mm256_srli_epi32::<16>(x0), _mm256_srli_epi32::<16>(x1));
            let min = _mm256_permute4x64_epi64(min, 0b11_01_10_00);
            let max = _mm256_permute4x64_epi64(max, 0b11_01_10_00);
            (_mm256_xor_si256(min, bias), _mm256_xor_si256(max, bias))
        }

        let (lo, hi) = ranges.split_at(16);
        let (min0, max0) = narrow(lo.try_into().unwrap());
        let (min1, max1) = narrow(hi.try_into().unwrap());
        let mut miss0 = _mm256_set1_epi16(-1);
        let mut miss1 = _mm256_set1_epi16(-1);
        for &key in keys {
            let t = _mm256_set1_epi16((key ^ 0x8000) as i16);
            let out0 = _mm256_or_si256(_mm256_cmpgt_epi16(min0, t), _mm256_cmpgt_epi16(t, max0));
            let out1 = _mm256_or_si256(_mm256_cmpgt_epi16(min1, t), _mm256_cmpgt_epi16(t, max1));
            miss0 = _mm256_and_si256(miss0, out0);
            miss1 = _mm256_and_si256(miss1, out1);
        }
        // bytes per 64-bit lane: ranges 0..8 16..24 8..16 24..32
        let y = _mm256_packs_epi16(miss0, miss1);
        !(_mm256_movemask_epi8(_mm256_permute4x64_epi64(y, 0b11_01_10_00)) as u32)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn meets_v3_test() {
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let (ranges, keys) = sample();
            for n in 0..=keys.len() {
                let specialized = unsafe { meets_v3(&ranges, &keys[..n]) };
                let fallback = fallback(&ranges, &keys[..n]);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn meets_a2(ranges: &[u32; 32], keys: &[u16]) -> u32 {
        use core::arch::aarch64::*;
        let p = ranges.as_ptr().cast::<u16>();
        // the ends of 8 ranges each, in order
        let x: [uint16x8x2_t; 4] = std::array::from_fn(|i| unsafe { vld2q_u16(p.add(16 * i)) });
        let mut y = [vdupq_n_u16(0); 4];
        for &key in keys {
            let t = vdupq_n_u16(key);
            for i in 0..4 {
                let hit = vandq_u16(vcleq_u16(x[i].0, t), vcgeq_u16(x[i].1, t));
                y[i] = vorrq_u16(y[i], hit);
            }
        }
        let bits = unsafe { vld1q_u16([1_u16, 2, 4, 8, 16, 32, 64, 128].as_ptr()) };
        let mut mask = 0;
        for i in 0..4 {
            mask |= (vaddvq_u16(vandq_u16(y[i], bits)) as u32) << (8 * i);
        }
        mask
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn meets_a2_test() {
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let (ranges, keys) = sample();
            for n in 0..=keys.len() {
                let specialized = unsafe { meets_a2(&ranges, &keys[..n]) };
                let fallback = fallback(&ranges, &keys[..n]);
                assert_eq!(specialized, fallback);
            }
        }
    }

    /// Random ranges, some empty and some of one key, and random keys, about
    /// half of which lie in a range.
    #[cfg(test)]
    fn sample() -> ([u32; 32], Vec<u16>) {
        let ranges: [u32; 32] = std::array::from_fn(|_| {
            let (a, b) = (rand::random::<u16>(), rand::random::<u16>());
            let (min, max) = match rand::random::<u8>() % 4 {
                0 => (a.max(b), a.min(b)),
                1 => (a, a),
                2 => (a, a.saturating_add(b % 256)),
                _ => (a.min(b), a.max(b)),
            };
            min as u32 | (max as u32) << 16
        });
        let keys = (0..20)
            .map(|_| {
                let x = ranges[rand::random::<u32>() as usize % 32];
                match rand::random::<u8>() % 4 {
                    0 => x as u16,
                    1 => (x >> 16) as u16,
                    _ => rand::random(),
                }
            })
            .collect();
        (ranges, keys)
    }

    #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn meets(ranges: &[u32; 32], keys: &[u16]) -> u32 {
        let mut hit = [false; 32];
        for &key in keys {
            for i in 0..32 {
                let (min, max) = (ranges[i] as u16, (ranges[i] >> 16) as u16);
                hit[i] |= min <= key && key <= max;
            }
        }
        let mut mask = 0;
        for i in 0..32 {
            mask |= (hit[i] as u32) << i;
        }
        mask
    }
}

/// The ranges of 16-bit keys, each `min | max << 16` and empty if `min > max`,
/// that hold one of `keys`, as a bitmask: bit `i` is range `i`. The cost is
/// linear in the number of keys, so keep it to a few.
#[inline(always)]
pub fn meets(ranges: &[u32; 32], keys: &[u16]) -> u32 {
    meets::meets(ranges, keys)
}

/// The portable compare of every value with every key, which the dispatchers
/// compile for the targets without a kernel and the tests check the kernels
/// against.
#[inline(always)]
fn matches_portable<T: Value>(column: &[T; 32], keys: &[T]) -> u32 {
    let mut hit = [false; 32];
    for &key in keys {
        for i in 0..32 {
            hit[i] |= column[i] == key;
        }
    }
    let mut mask = 0;
    for i in 0..32 {
        mask |= (hit[i] as u32) << i;
    }
    mask
}

/// Random columns, about a quarter of whose values are one of 20 random keys.
#[cfg(test)]
fn sample<T: Value>(random: impl Fn() -> T) -> ([T; 32], Vec<T>) {
    let keys = (0..20).map(|_| random()).collect::<Vec<_>>();
    let column = std::array::from_fn(|_| {
        if rand::random::<u8>() < 64 {
            keys[rand::random::<u32>() as usize % keys.len()]
        } else {
            random()
        }
    });
    (column, keys)
}

/// Checks a kernel against [`matches_portable`] on random columns.
#[cfg(test)]
fn check<T: Value>(random: impl Fn() -> T, kernel: impl Fn(&[T; 32], &[T]) -> u32) {
    for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
        let (column, keys) = sample(&random);
        for n in 0..=keys.len() {
            assert_eq!(
                kernel(&column, &keys[..n]),
                matches_portable(&column, &keys[..n])
            );
        }
    }
}

/// The rows among 32 16-bit values, in a vector, that equal one of `keys`:
/// the compares of a column of `u16` and of the payloads' low 16 bits.
#[cfg(target_arch = "x86_64")]
macro_rules! compare16_v4 {
    ($x:expr, $keys:expr) => {{
        let x = $x;
        let mut mask = 0;
        for &key in $keys {
            mask |= _mm512_cmpeq_epi16_mask(x, _mm512_set1_epi16(key as i16));
        }
        mask
    }};
}

/// [`compare16_v4`], 16 values in each of two vectors.
#[cfg(target_arch = "x86_64")]
macro_rules! compare16_v3 {
    ($x0:expr, $x1:expr, $keys:expr) => {{
        let (x0, x1) = ($x0, $x1);
        let (mut y0, mut y1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for &key in $keys {
            let k = _mm256_set1_epi16(key as i16);
            y0 = _mm256_or_si256(y0, _mm256_cmpeq_epi16(x0, k));
            y1 = _mm256_or_si256(y1, _mm256_cmpeq_epi16(x1, k));
        }
        // bytes per 64-bit lane: rows 0..8 16..24 8..16 24..32
        let y = _mm256_packs_epi16(y0, y1);
        _mm256_movemask_epi8(_mm256_permute4x64_epi64(y, 0b11_01_10_00)) as u32
    }};
}

/// [`compare16_v4`], 8 values in each of four vectors.
#[cfg(target_arch = "aarch64")]
macro_rules! compare16_a2 {
    ($x:expr, $keys:expr) => {{
        let x: [uint16x8_t; 4] = $x;
        let mut y = [vdupq_n_u16(0); 4];
        for &key in $keys {
            let k = vdupq_n_u16(key);
            for i in 0..4 {
                y[i] = vorrq_u16(y[i], vceqq_u16(x[i], k));
            }
        }
        let bits = unsafe { vld1q_u16([1_u16, 2, 4, 8, 16, 32, 64, 128].as_ptr()) };
        let mut mask = 0;
        for i in 0..4 {
            mask |= (vaddvq_u16(vandq_u16(y[i], bits)) as u32) << (8 * i);
        }
        mask
    }};
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod u16s {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn matches_v4(column: &[u16; 32], keys: &[u16]) -> u32 {
        use core::arch::x86_64::*;
        compare16_v4!(unsafe { _mm512_loadu_si512(column.as_ptr().cast()) }, keys)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn matches_v3(column: &[u16; 32], keys: &[u16]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x0 = unsafe { _mm256_loadu_si256(p.cast()) };
        let x1 = unsafe { _mm256_loadu_si256(p.add(16).cast()) };
        compare16_v3!(x0, x1, keys)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn matches_a2(column: &[u16; 32], keys: &[u16]) -> u32 {
        use core::arch::aarch64::*;
        let p = column.as_ptr();
        let x = std::array::from_fn(|i| unsafe { vld1q_u16(p.add(8 * i)) });
        compare16_a2!(x, keys)
    }

    #[cfg(test)]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn kernels() {
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            super::check(rand::random::<u16>, |c, k| unsafe { matches_v4(c, k) });
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            super::check(rand::random::<u16>, |c, k| unsafe { matches_v3(c, k) });
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            super::check(rand::random::<u16>, |c, k| unsafe { matches_a2(c, k) });
        }
        super::check(rand::random::<u16>, matches);
    }

    #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn matches(column: &[u16; 32], keys: &[u16]) -> u32 {
        super::matches_portable(column, keys)
    }

    #[inline(always)]
    pub fn fold(column: &[u16; 32]) -> [u16; 32] {
        *column
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod payload {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn matches_v4(payload: &[u64; 32], keys: &[u16]) -> u32 {
        use core::arch::x86_64::*;
        let p = payload.as_ptr();
        // the low 16 bits of the 32 payloads, in order
        let x0 = _mm512_cvtepi64_epi16(unsafe { _mm512_loadu_si512(p.cast()) });
        let x1 = _mm512_cvtepi64_epi16(unsafe { _mm512_loadu_si512(p.add(8).cast()) });
        let x2 = _mm512_cvtepi64_epi16(unsafe { _mm512_loadu_si512(p.add(16).cast()) });
        let x3 = _mm512_cvtepi64_epi16(unsafe { _mm512_loadu_si512(p.add(24).cast()) });
        let lo = _mm256_inserti128_si256(_mm256_castsi128_si256(x0), x1, 1);
        let hi = _mm256_inserti128_si256(_mm256_castsi128_si256(x2), x3, 1);
        compare16_v4!(_mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1), keys)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn matches_v3(payload: &[u64; 32], keys: &[u16]) -> u32 {
        use core::arch::x86_64::*;

        /// The low 16 bits of 16 payloads, in order.
        #[inline]
        #[crate::target_cpu(enable = "v3")]
        fn narrow(p: &[u64; 16]) -> __m256i {
            let p = p.as_ptr();
            let low = _mm256_set1_epi64x(0xffff);
            let x0 = _mm256_and_si256(unsafe { _mm256_loadu_si256(p.cast()) }, low);
            let x1 = _mm256_and_si256(unsafe { _mm256_loadu_si256(p.add(4).cast()) }, low);
            let x2 = _mm256_and_si256(unsafe { _mm256_loadu_si256(p.add(8).cast()) }, low);
            let x3 = _mm256_and_si256(unsafe { _mm256_loadu_si256(p.add(12).cast()) }, low);
            // per 128-bit lane: rows 0 1 4 5 8 9 12 13 | 2 3 6 7 10 11 14 15
            let x = _mm256_packus_epi32(_mm256_packus_epi32(x0, x1), _mm256_packus_epi32(x2, x3));
            _mm256_permutevar8x32_epi32(x, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7))
        }

        let (lo, hi) = payload.split_at(16);
        let (x0, x1) = (
            narrow(lo.try_into().unwrap()),
            narrow(hi.try_into().unwrap()),
        );
        compare16_v3!(x0, x1, keys)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn matches_a2(payload: &[u64; 32], keys: &[u16]) -> u32 {
        use core::arch::aarch64::*;
        let p = payload.as_ptr();
        let x = std::array::from_fn(|i| {
            // the low 16 bits of 8 payloads, in order
            let a = vreinterpretq_u16_u64(unsafe { vld1q_u64(p.add(8 * i)) });
            let b = vreinterpretq_u16_u64(unsafe { vld1q_u64(p.add(8 * i + 2)) });
            let c = vreinterpretq_u16_u64(unsafe { vld1q_u64(p.add(8 * i + 4)) });
            let d = vreinterpretq_u16_u64(unsafe { vld1q_u64(p.add(8 * i + 6)) });
            vuzp1q_u16(vuzp1q_u16(a, b), vuzp1q_u16(c, d))
        });
        compare16_a2!(x, keys)
    }

    #[cfg(test)]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn kernels() {
        let random = || {
            let tag = rand::random::<u16>();
            rand::random::<u64>() & !0xffff | tag as u64
        };
        let check = |kernel: &dyn Fn(&[u64; 32], &[u16]) -> u32| {
            for _ in 0..256 {
                let keys = (0..20).map(|_| rand::random::<u16>()).collect::<Vec<_>>();
                let payload: [u64; 32] = std::array::from_fn(|_| match rand::random::<u8>() < 64 {
                    true => random() & !0xffff | keys[rand::random::<u32>() as usize % 20] as u64,
                    false => random(),
                });
                let column = payload.map(|x| x as u16);
                for n in 0..=keys.len() {
                    let expected = super::matches_portable(&column, &keys[..n]);
                    assert_eq!(kernel(&payload, &keys[..n]), expected);
                }
            }
        };
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            check(&|p, k| unsafe { matches_v4(p, k) });
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            check(&|p, k| unsafe { matches_v3(p, k) });
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            check(&|p, k| unsafe { matches_a2(p, k) });
        }
        check(&matches);
    }

    #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn matches(payload: &[u64; 32], keys: &[u16]) -> u32 {
        super::matches_portable(&payload.map(|x| x as u16), keys)
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod u32s {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn matches_v4(column: &[u32; 32], keys: &[u32]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x0 = unsafe { _mm512_loadu_si512(p.cast()) };
        let x1 = unsafe { _mm512_loadu_si512(p.add(16).cast()) };
        let (mut m0, mut m1) = (0, 0);
        for &key in keys {
            let k = _mm512_set1_epi32(key as i32);
            m0 |= _mm512_cmpeq_epi32_mask(x0, k);
            m1 |= _mm512_cmpeq_epi32_mask(x1, k);
        }
        m0 as u32 | (m1 as u32) << 16
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn matches_v3(column: &[u32; 32], keys: &[u32]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x: [__m256i; 4] =
            std::array::from_fn(|i| unsafe { _mm256_loadu_si256(p.add(8 * i).cast()) });
        let mut y = [_mm256_setzero_si256(); 4];
        for &key in keys {
            let k = _mm256_set1_epi32(key as i32);
            for i in 0..4 {
                y[i] = _mm256_or_si256(y[i], _mm256_cmpeq_epi32(x[i], k));
            }
        }
        let mut mask = 0;
        for i in 0..4 {
            mask |= (_mm256_movemask_ps(_mm256_castsi256_ps(y[i])) as u32) << (8 * i);
        }
        mask
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn matches_a2(column: &[u32; 32], keys: &[u32]) -> u32 {
        use core::arch::aarch64::*;
        let p = column.as_ptr();
        let x: [uint32x4_t; 8] = std::array::from_fn(|i| unsafe { vld1q_u32(p.add(4 * i)) });
        let mut y = [vdupq_n_u32(0); 8];
        for &key in keys {
            let k = vdupq_n_u32(key);
            for i in 0..8 {
                y[i] = vorrq_u32(y[i], vceqq_u32(x[i], k));
            }
        }
        let bits = unsafe { vld1q_u32([1_u32, 2, 4, 8].as_ptr()) };
        let mut mask = 0;
        for i in 0..8 {
            mask |= vaddvq_u32(vandq_u32(y[i], bits)) << (4 * i);
        }
        mask
    }

    #[cfg(test)]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn kernels() {
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            super::check(rand::random::<u32>, |c, k| unsafe { matches_v4(c, k) });
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            super::check(rand::random::<u32>, |c, k| unsafe { matches_v3(c, k) });
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            super::check(rand::random::<u32>, |c, k| unsafe { matches_a2(c, k) });
        }
        super::check(rand::random::<u32>, matches);
    }

    #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn matches(column: &[u32; 32], keys: &[u32]) -> u32 {
        super::matches_portable(column, keys)
    }

    #[inline(always)]
    pub fn fold(column: &[u32; 32]) -> [u16; 32] {
        let mut folds = [0; 32];
        for i in 0..32 {
            folds[i] = (column[i] ^ column[i] >> 16) as u16;
        }
        folds
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod u64s {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn matches_v4(column: &[u64; 32], keys: &[u64]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x: [__m512i; 4] =
            std::array::from_fn(|i| unsafe { _mm512_loadu_si512(p.add(8 * i).cast()) });
        let mut m = [0_u8; 4];
        for &key in keys {
            let k = _mm512_set1_epi64(key as i64);
            for i in 0..4 {
                m[i] |= _mm512_cmpeq_epi64_mask(x[i], k);
            }
        }
        u32::from_le_bytes(m)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn matches_v3(column: &[u64; 32], keys: &[u64]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x: [__m256i; 8] =
            std::array::from_fn(|i| unsafe { _mm256_loadu_si256(p.add(4 * i).cast()) });
        let mut y = [_mm256_setzero_si256(); 8];
        for &key in keys {
            let k = _mm256_set1_epi64x(key as i64);
            for i in 0..8 {
                y[i] = _mm256_or_si256(y[i], _mm256_cmpeq_epi64(x[i], k));
            }
        }
        let mut mask = 0;
        for i in 0..8 {
            mask |= (_mm256_movemask_pd(_mm256_castsi256_pd(y[i])) as u32) << (4 * i);
        }
        mask
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn matches_a2(column: &[u64; 32], keys: &[u64]) -> u32 {
        use core::arch::aarch64::*;
        let p = column.as_ptr();
        let x: [uint64x2_t; 16] = std::array::from_fn(|i| unsafe { vld1q_u64(p.add(2 * i)) });
        let mut y = [vdupq_n_u64(0); 16];
        for &key in keys {
            let k = vdupq_n_u64(key);
            for i in 0..16 {
                y[i] = vorrq_u64(y[i], vceqq_u64(x[i], k));
            }
        }
        // two rows' compares in the 32-bit halves of a vector, four rows at a time
        let bits = unsafe { vld1q_u32([1_u32, 2, 4, 8].as_ptr()) };
        let mut mask = 0;
        for i in 0..8 {
            let rows = vcombine_u32(vmovn_u64(y[2 * i]), vmovn_u64(y[2 * i + 1]));
            mask |= vaddvq_u32(vandq_u32(rows, bits)) << (4 * i);
        }
        mask
    }

    #[cfg(test)]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn kernels() {
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            super::check(rand::random::<u64>, |c, k| unsafe { matches_v4(c, k) });
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            super::check(rand::random::<u64>, |c, k| unsafe { matches_v3(c, k) });
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            super::check(rand::random::<u64>, |c, k| unsafe { matches_a2(c, k) });
        }
        super::check(rand::random::<u64>, matches);
    }

    #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn matches(column: &[u64; 32], keys: &[u64]) -> u32 {
        super::matches_portable(column, keys)
    }

    #[inline(always)]
    pub fn fold(column: &[u64; 32]) -> [u16; 32] {
        let mut folds = [0; 32];
        for i in 0..32 {
            let x = column[i] ^ column[i] >> 32;
            folds[i] = (x ^ x >> 16) as u16;
        }
        folds
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod u128s {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn matches_v4(column: &[[u64; 2]; 32], keys: &[[u64; 2]]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x: [__m512i; 8] =
            std::array::from_fn(|i| unsafe { _mm512_loadu_si512(p.add(4 * i).cast()) });
        // the low and the high halves of 8 rows each, in order
        let even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        let odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        let lo: [__m512i; 4] =
            std::array::from_fn(|i| _mm512_permutex2var_epi64(x[2 * i], even, x[2 * i + 1]));
        let hi: [__m512i; 4] =
            std::array::from_fn(|i| _mm512_permutex2var_epi64(x[2 * i], odd, x[2 * i + 1]));
        let mut m = [0_u8; 4];
        for key in keys {
            let (k0, k1) = (
                _mm512_set1_epi64(key[0] as i64),
                _mm512_set1_epi64(key[1] as i64),
            );
            for i in 0..4 {
                m[i] |= _mm512_cmpeq_epi64_mask(lo[i], k0) & _mm512_cmpeq_epi64_mask(hi[i], k1);
            }
        }
        u32::from_le_bytes(m)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn matches_v3(column: &[[u64; 2]; 32], keys: &[[u64; 2]]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x: [__m256i; 16] =
            std::array::from_fn(|i| unsafe { _mm256_loadu_si256(p.add(2 * i).cast()) });
        // the low and the high halves of 4 rows each, rows 0 2 1 3
        let lo: [__m256i; 8] =
            std::array::from_fn(|i| _mm256_unpacklo_epi64(x[2 * i], x[2 * i + 1]));
        let hi: [__m256i; 8] =
            std::array::from_fn(|i| _mm256_unpackhi_epi64(x[2 * i], x[2 * i + 1]));
        let mut y = [_mm256_setzero_si256(); 8];
        for key in keys {
            let (k0, k1) = (
                _mm256_set1_epi64x(key[0] as i64),
                _mm256_set1_epi64x(key[1] as i64),
            );
            for i in 0..8 {
                let both =
                    _mm256_and_si256(_mm256_cmpeq_epi64(lo[i], k0), _mm256_cmpeq_epi64(hi[i], k1));
                y[i] = _mm256_or_si256(y[i], both);
            }
        }
        let mut mask = 0;
        for i in 0..8 {
            // bits: rows 0 2 1 3
            let m = _mm256_movemask_pd(_mm256_castsi256_pd(y[i])) as u32;
            let m = (m & 0b1001) | (m & 0b0010) << 1 | (m & 0b0100) >> 1;
            mask |= m << (4 * i);
        }
        mask
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn matches_a2(column: &[[u64; 2]; 32], keys: &[[u64; 2]]) -> u32 {
        use core::arch::aarch64::*;
        let p = column.as_ptr().cast::<u64>();
        let x: [uint64x2_t; 32] = std::array::from_fn(|i| unsafe { vld1q_u64(p.add(2 * i)) });
        let mut y = [vdupq_n_u32(0); 8];
        for key in keys {
            let k = unsafe { vld1q_u64(key.as_ptr()) };
            for i in 0..8 {
                // the low 32 bits of the halves' compares of four rows, then
                // the rows whose two halves equal the key's
                let a = vcombine_u32(
                    vmovn_u64(vceqq_u64(x[4 * i], k)),
                    vmovn_u64(vceqq_u64(x[4 * i + 1], k)),
                );
                let b = vcombine_u32(
                    vmovn_u64(vceqq_u64(x[4 * i + 2], k)),
                    vmovn_u64(vceqq_u64(x[4 * i + 3], k)),
                );
                let rows = vandq_u32(vuzp1q_u32(a, b), vuzp2q_u32(a, b));
                y[i] = vorrq_u32(y[i], rows);
            }
        }
        let bits = unsafe { vld1q_u32([1_u32, 2, 4, 8].as_ptr()) };
        let mut mask = 0;
        for i in 0..8 {
            mask |= vaddvq_u32(vandq_u32(y[i], bits)) << (4 * i);
        }
        mask
    }

    #[cfg(test)]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn kernels() {
        // halves that often agree with a key's
        let random = || {
            let lo = rand::random::<u64>();
            let hi = if rand::random::<bool>() {
                lo
            } else {
                rand::random()
            };
            [lo, hi]
        };
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            super::check(random, |c, k| unsafe { matches_v4(c, k) });
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            super::check(random, |c, k| unsafe { matches_v3(c, k) });
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            super::check(random, |c, k| unsafe { matches_a2(c, k) });
        }
        super::check(random, matches);
        // a row that matches one half of a key and the other half of another
        let column = std::array::from_fn(|i| [i as u64, 100 + i as u64]);
        let keys = [[3, 0], [0, 103], [5, 105]];
        assert_eq!(matches(&column, &keys), 1 << 5);
    }

    #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn matches(column: &[[u64; 2]; 32], keys: &[[u64; 2]]) -> u32 {
        super::matches_portable(column, keys)
    }

    #[inline(always)]
    pub fn fold(column: &[[u64; 2]; 32]) -> [u16; 32] {
        let mut folds = [0; 32];
        for i in 0..32 {
            let x = column[i][0] ^ column[i][1];
            let x = x ^ x >> 32;
            folds[i] = (x ^ x >> 16) as u16;
        }
        folds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folds<T: Value>(random: impl Fn() -> T) {
        let column: [T; 32] = std::array::from_fn(|_| random());
        let folds = T::fold(&column);
        for i in 0..32 {
            assert_eq!(folds[i], column[i].fold_one());
        }
    }

    #[test]
    fn payload_bitmap() {
        let payload: [u64; 32] = std::array::from_fn(|_| rand::random());
        let mut bitmap = [0_u64; 1024];
        for x in payload.iter().step_by(3) {
            bitmap[*x as u16 as usize / 64] |= 1 << (*x as u16 % 64);
        }
        let column = payload.map(|x| x as u16);
        assert_eq!(
            matches_bitmap_payload(&payload, &bitmap),
            matches_bitmap(&column, &bitmap)
        );
    }

    #[test]
    fn fold() {
        folds(rand::random::<u16>);
        folds(rand::random::<u32>);
        folds(rand::random::<u64>);
        folds(rand::random::<[u64; 2]>);
        // the exclusive or of the 16-bit lanes, at every width
        let x = 0x0123_4567_89ab_cdef_u64;
        let expected = 0x0123 ^ 0x4567 ^ 0x89ab ^ 0xcdef;
        assert_eq!(x.fold_one(), expected);
        assert_eq!([x, 0].fold_one(), expected);
        assert_eq!(((x ^ x >> 32) as u32).fold_one(), expected);
    }

    fn sets<T: Value>(random: impl Fn() -> T) {
        for n in [1, 10, 33, 200, 5000] {
            let mut values = (0..n).map(|_| random()).collect::<Vec<_>>();
            values.sort_unstable();
            values.dedup();
            let set = Set::new(&values);
            for _ in 0..if cfg!(not(miri)) { 64 } else { 1 } {
                let column: [T; 32] = std::array::from_fn(|_| {
                    if rand::random::<bool>() {
                        values[rand::random::<u32>() as usize % values.len()]
                    } else {
                        random()
                    }
                });
                let mask = set.matches(&column);
                for i in 0..32 {
                    let expected = values.binary_search(&column[i]).is_ok();
                    assert_eq!(mask >> i & 1 != 0, expected);
                    assert_eq!(set.contains(column[i]), expected);
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn set() {
        sets(rand::random::<u16>);
        // small values, whose folds collide in few slots
        sets(|| rand::random::<u32>() % 100_000);
        sets(rand::random::<u32>);
        sets(rand::random::<u64>);
        sets(|| [rand::random::<u64>() % 7, rand::random::<u64>() % 7]);
        sets(rand::random::<[u64; 2]>);
        // the value that marks free slots by default is in the set
        let set = Set::new(&[0_u64, 1, 2]);
        assert!(set.contains(0) && set.contains(2) && !set.contains(3));
    }
}
