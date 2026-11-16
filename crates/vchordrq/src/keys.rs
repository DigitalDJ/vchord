// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

//! Each filter key is a column; its hashes and stored forms are on disk and must never change.
//! Block ranges hold whatever the order of the rows, so correctness never depends on sorting.

use crate::tape::Filter;
use crate::tuples::{AppendableTuple, FrozenTuple, MetaTuple, MetaTupleReader, WithReader};
use index::relation::{Page, RelationRead};
use simd::column::{Set, Value};
use std::num::NonZero;
use zerocopy::{FromBytes, Immutable, KnownLayout};

/// The splitmix64 finalizer.
const fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Integers of every width are widened first, so that equality across widths agrees.
pub const fn hash_int(x: i64) -> u64 {
    mix(x as u64 ^ 0xd1b5_4a32_d192_ed03)
}

pub const fn hash_uuid(x: &[u8; 16]) -> u64 {
    let [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p] = *x;
    let lo = u64::from_le_bytes([a, b, c, d, e, f, g, h]);
    let hi = u64::from_le_bytes([i, j, k, l, m, n, o, p]);
    mix(lo ^ mix(hi ^ 0x9e37_79b9_7f4a_7c15))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    W2,
    W4,
    W8,
    W16,
}

impl Width {
    pub const fn bytes(self) -> usize {
        2 << self as usize
    }
    pub fn of_bytes(bytes: usize) -> Option<Self> {
        match bytes {
            2 => Some(Self::W2),
            4 => Some(Self::W4),
            8 => Some(Self::W8),
            16 => Some(Self::W16),
            _ => None,
        }
    }
}

/// A hash folded to `width` by xoring its halves.
pub fn fold(hash: u64, width: Width) -> u64 {
    match width {
        Width::W2 => hash.fold_one() as u64,
        Width::W4 => (hash ^ hash >> 32) & 0xffff_ffff,
        Width::W8 | Width::W16 => hash,
    }
}

/// The key with its sign bit flipped, so that stored forms sort like keys; `None` out of range.
pub const fn raw_int(x: i64, width: Width) -> Option<u64> {
    let bits = 8 * width.bytes() as u32;
    if bits < 64 && (x < -(1 << (bits - 1)) || x >= 1 << (bits - 1)) {
        return None;
    }
    Some((x as u64 ^ 1 << (bits - 1)) & (u64::MAX >> (64 - bits)))
}

/// A range `min | max << 16` that holds no key.
pub const EMPTY_RANGE: u32 = u16::MAX as u32;

/// Whether `bitmap` has a key in `range`.
#[inline(always)]
fn meets_bitmap(bitmap: &[u64; 1024], range: u32) -> bool {
    let (min, max) = (range as u16 as usize, (range >> 16) as usize);
    if min > max {
        return false;
    }
    let (a, b) = (min / 64, max / 64);
    let first = bitmap[a] & u64::MAX << (min % 64);
    let last = bitmap[b] & u64::MAX >> (63 - max % 64);
    if a == b {
        first & last != 0
    } else {
        first != 0 || last != 0 || bitmap[a + 1..b].iter().any(|&x| x != 0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key {
    pub width: Width,
    pub hashed: bool,
    /// Its bytes' offset in a row's columns.
    pub offset: u16,
}

impl Key {
    /// Whether `set` selects exactly its keys' rows: a NULL is stored as zero.
    fn exact(&self, set: &[u128]) -> bool {
        !self.hashed && !set.contains(&0)
    }
    /// The 16-bit sort key; a wide whole key is hashed first, as its bytes need not spread.
    pub fn sort_key(&self, stored: u128) -> u16 {
        if self.width == Width::W2 || self.hashed {
            [stored as u64, (stored >> 64) as u64].fold_one()
        } else {
            hash_uuid(&stored.to_le_bytes()).fold_one()
        }
    }
}

const MAX_KEYS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Keys {
    keys: [Key; MAX_KEYS],
    n: u8,
    columns: u16,
}

impl Default for Keys {
    fn default() -> Self {
        Self::new([])
    }
}

impl Keys {
    pub fn new(keys: impl IntoIterator<Item = (Width, bool)>) -> Self {
        let free = Key {
            width: Width::W2,
            hashed: false,
            offset: 0,
        };
        let mut this = Self {
            keys: [free; MAX_KEYS],
            n: 0,
            columns: 0,
        };
        for (width, hashed) in keys {
            this.keys[this.n as usize] = Key {
                width,
                hashed,
                offset: this.columns,
            };
            this.columns += width.bytes() as u16;
            this.n += 1;
        }
        this
    }
    pub fn as_slice(&self) -> &[Key] {
        &self.keys[..self.n as usize]
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn columns(&self) -> usize {
        self.columns as usize
    }
    /// The sort key of a row's first key.
    pub fn sort_key(&self, columns: Columns<'_>) -> u16 {
        let key = &self.keys[0];
        let bytes = key.width.bytes();
        let mut x = [0; 16];
        x[..bytes].copy_from_slice(columns.get(key.offset, bytes));
        key.sort_key(u128::from_le_bytes(x))
    }
    pub fn range_of_block(&self, payload: &[Option<NonZero<u64>>; 32], columns: &[u8]) -> u32 {
        let rows = (0..32).filter(|&i| payload[i].is_some());
        let folds = rows.map(|i| self.sort_key(Columns::block(columns, i)));
        let (min, max) = folds.fold((u16::MAX, 0), |(a, b), x| (a.min(x), b.max(x)));
        min as u32 | (max as u32) << 16
    }
    /// The columns of a block, each key's 32 values in order, from 32 rows' own.
    pub fn block(&self, rows: &[u8]) -> Vec<u8> {
        let mut block = Vec::with_capacity(rows.len());
        for key in self.as_slice() {
            for row in rows.chunks_exact(self.columns()) {
                block.extend_from_slice(&row[key.offset as usize..][..key.width.bytes()]);
            }
        }
        block
    }
    /// One byte per key: the width in bits 0-1, whether hashed in bit 2.
    pub fn descriptors(&self) -> Vec<u8> {
        let descriptor = |key: &Key| key.width as u8 | (key.hashed as u8) << 2;
        self.as_slice().iter().map(descriptor).collect()
    }
    pub fn of<R: RelationRead>(index: &R) -> Self {
        let meta_guard = index.read(0);
        let meta_bytes = meta_guard.get(1).expect("data corruption");
        Self::of_meta(MetaTuple::deserialize_ref(meta_bytes))
    }
    pub fn of_meta(meta: MetaTupleReader<'_>) -> Self {
        let Some(descriptors) = meta.keys() else {
            return Self::default();
        };
        Self::from_descriptors(descriptors).unwrap_or_else(|| {
            panic!(
                "deserialization: unknown layout of filter keys; {}",
                "after upgrading vchord, please use REINDEX to rebuild the index."
            )
        })
    }
    pub fn from_descriptors(descriptors: &[u8]) -> Option<Self> {
        if descriptors.is_empty() || descriptors.len() > MAX_KEYS {
            return None;
        }
        if descriptors.iter().any(|&x| x >> 3 != 0) {
            return None;
        }
        Some(Self::new(descriptors.iter().map(|&x| {
            let width = Width::of_bytes(2 << (x & 3)).unwrap();
            (width, x & 4 != 0)
        })))
    }
    /// The most column bytes per row for which a block's last part and an appendable tuple fit.
    pub fn most_columns(words: usize, prefetch: usize, freespace: usize) -> usize {
        let block = freespace.saturating_sub(FrozenTuple::estimate_size_0(prefetch, 0, 0)) / 32;
        let row = freespace.saturating_sub(AppendableTuple::estimate_size(prefetch, words, 0));
        block.min(row)
    }
}

/// A row's columns: a block's column-major bytes and the row, or an appendable row's own.
#[derive(Debug, Clone, Copy)]
pub struct Columns<'a> {
    bytes: &'a [u8],
    row: Option<u8>,
}

impl<'a> Columns<'a> {
    pub fn block(bytes: &'a [u8], row: usize) -> Self {
        Self {
            bytes,
            row: Some(row as u8),
        }
    }
    pub fn row(bytes: &'a [u8]) -> Self {
        Self { bytes, row: None }
    }
    #[inline(always)]
    pub fn get(self, offset: u16, bytes: usize) -> &'a [u8] {
        let offset = offset as usize;
        match self.row {
            None => &self.bytes[offset..][..bytes],
            Some(row) => &self.bytes[32 * offset + row as usize * bytes..][..bytes],
        }
    }
    /// Appends the row's columns, as an appendable tuple stores them.
    pub fn extend(self, keys: &Keys, row: &mut Vec<u8>) {
        if self.row.is_none() {
            row.extend_from_slice(&self.bytes[..keys.columns()]);
            return;
        }
        for key in keys.as_slice() {
            row.extend_from_slice(self.get(key.offset, key.width.bytes()));
        }
    }
}

#[derive(Debug)]
enum Values<T> {
    Few(Vec<T>),
    Many(Box<Set<T>>),
}

impl<T: Value> Values<T> {
    /// Above this many keys a [`Set`] is cheaper than comparing key by key.
    const FEW: usize = match (cfg!(target_arch = "x86_64"), size_of::<T>()) {
        (true, 2) => 32,
        (true, 4) => 40,
        (true, 8) => 24,
        (true, _) => 16,
        (false, 2) => 12,
        (false, 4) => 10,
        (false, 8) => 4,
        (false, _) => 3,
    };

    /// `None` for an empty set, which no row matches.
    fn new(mut keys: Vec<T>) -> Option<Self> {
        keys.sort_unstable();
        keys.dedup();
        if keys.is_empty() {
            None
        } else if keys.len() <= Self::FEW {
            Some(Self::Few(keys))
        } else {
            Some(Self::Many(Box::new(Set::new(&keys))))
        }
    }

    #[inline(always)]
    fn matches(&self, column: &[T; 32]) -> u32 {
        match self {
            Self::Few(keys) => T::matches(column, keys),
            Self::Many(set) => set.matches(column),
        }
    }

    #[inline(always)]
    fn contains(&self, x: T) -> bool {
        match self {
            Self::Few(keys) => keys.contains(&x),
            Self::Many(set) => set.contains(x),
        }
    }
}

#[derive(Debug)]
enum Column {
    U16(Values<u16>),
    U32(Values<u32>),
    U64(Values<u64>),
    U128(Values<[u64; 2]>),
}

#[inline(always)]
fn column<T: FromBytes + Immutable + KnownLayout>(columns: &[u8], offset: u16) -> &[T; 32] {
    let bytes = &columns[32 * offset as usize..][..32 * size_of::<T>()];
    <[T; 32]>::ref_from_bytes(bytes).unwrap()
}

impl Column {
    /// Columns are little-endian on disk; compare against `to_le` forms.
    fn new(width: Width, keys: &[u128]) -> Option<Self> {
        Some(match width {
            Width::W2 => Self::U16(Values::new(
                keys.iter().map(|&x| (x as u16).to_le()).collect(),
            )?),
            Width::W4 => Self::U32(Values::new(
                keys.iter().map(|&x| (x as u32).to_le()).collect(),
            )?),
            Width::W8 => Self::U64(Values::new(
                keys.iter().map(|&x| (x as u64).to_le()).collect(),
            )?),
            Width::W16 => Self::U128(Values::new(
                (keys.iter())
                    .map(|&x| [(x as u64).to_le(), ((x >> 64) as u64).to_le()])
                    .collect(),
            )?),
        })
    }

    #[inline(always)]
    fn matches(&self, columns: &[u8], offset: u16) -> u32 {
        match self {
            Self::U16(values) => values.matches(column(columns, offset)),
            Self::U32(values) => values.matches(column(columns, offset)),
            Self::U64(values) => values.matches(column(columns, offset)),
            Self::U128(values) => values.matches(column(columns, offset)),
        }
    }

    #[inline(always)]
    fn contains(&self, columns: Columns<'_>, offset: u16) -> bool {
        let read = |bytes| columns.get(offset, bytes);
        match self {
            Self::U16(values) => values.contains(FromBytes::read_from_bytes(read(2)).unwrap()),
            Self::U32(values) => values.contains(FromBytes::read_from_bytes(read(4)).unwrap()),
            Self::U64(values) => values.contains(FromBytes::read_from_bytes(read(8)).unwrap()),
            Self::U128(values) => values.contains(FromBytes::read_from_bytes(read(16)).unwrap()),
        }
    }
}

#[derive(Debug)]
pub struct KeyFilter {
    /// The sort keys of the first key's set.
    first: Option<Values<u16>>,
    columns: Vec<(u16, Column)>,
    exact: bool,
}

impl KeyFilter {
    /// `sets`: per key, the stored forms a row may hold, or `None` for any; `None` if none can.
    pub fn new(keys: &Keys, sets: impl IntoIterator<Item = Option<Vec<u128>>>) -> Option<Self> {
        let mut filter = Self {
            first: None,
            columns: Vec::new(),
            exact: true,
        };
        for (i, (key, set)) in std::iter::zip(keys.as_slice(), sets).enumerate() {
            let Some(set) = set else { continue };
            filter.exact &= key.exact(&set);
            if i == 0 {
                let keys = set.iter().map(|&x| key.sort_key(x)).collect();
                filter.first = Some(Values::new(keys)?);
            }
            filter
                .columns
                .push((key.offset, Column::new(key.width, &set)?));
        }
        Some(filter)
    }
    /// Whether the rows it selects need no recheck.
    pub fn exact(&self) -> bool {
        self.exact
    }
}

impl Filter for &KeyFilter {
    #[inline(always)]
    fn selects_blocks(self) -> bool {
        self.first.is_some()
    }

    #[inline(always)]
    fn ranges(self, ranges: &[u32; 32]) -> u32 {
        match &self.first {
            Some(Values::Few(keys)) => simd::column::meets(ranges, keys),
            Some(Values::Many(set)) => {
                let mut mask = 0;
                for (i, &range) in ranges.iter().enumerate() {
                    mask |= (meets_bitmap(set.bitmap(), range) as u32) << i;
                }
                mask
            }
            None => u32::MAX,
        }
    }

    #[inline(always)]
    fn block(self, columns: &[u8]) -> u32 {
        let mut mask = u32::MAX;
        for (offset, column) in &self.columns {
            if mask == 0 {
                break;
            }
            mask &= column.matches(columns, *offset);
        }
        mask
    }

    #[inline(always)]
    fn row(self, columns: Columns<'_>) -> bool {
        for (offset, column) in &self.columns {
            if !column.contains(columns, *offset) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::IntoBytes;

    /// On-disk hashes: these values must never change.
    #[test]
    fn golden() {
        let uuid = |s: &str| -> [u8; 16] {
            let s = s.replace('-', "");
            std::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        };
        let hashes = [
            hash_int(0),
            hash_int(1),
            hash_int(-1),
            hash_int(i64::MAX),
            hash_int(i64::MIN),
            hash_uuid(&uuid("00000000-0000-0000-0000-000000000000")),
            hash_uuid(&uuid("00000000-0000-0000-0000-000000000001")),
            hash_uuid(&uuid("ffffffff-ffff-ffff-ffff-ffffffffffff")),
            hash_uuid(&uuid("6f5c8a1e-3b2d-4c7a-9e1f-0a2b3c4d5e6f")),
        ]
        .map(|x| fold(x, Width::W2) as u16);
        assert_eq!(hashes, GOLDEN);
        for x in [
            hash_int(7),
            hash_uuid(&[9; 16]),
            u64::MAX,
            0x0123_4567_89ab_cdef,
        ] {
            assert_eq!(fold(fold(x, Width::W4), Width::W2), fold(x, Width::W2));
            assert_eq!(fold(x, Width::W8), x);
            assert_eq!(fold(x, Width::W4), (x ^ x >> 32) & 0xffff_ffff);
        }
    }

    const GOLDEN: [u16; 9] = [
        55156, 44833, 34830, 26900, 22934, 57732, 51418, 37919, 34892,
    ];

    #[test]
    fn raw() {
        use Width::*;
        assert_eq!(raw_int(0, W2), Some(0x8000));
        assert_eq!(raw_int(-1, W2), Some(0x7fff));
        assert_eq!(raw_int(i16::MIN as i64, W2), Some(0));
        assert_eq!(raw_int(i16::MAX as i64 + 1, W2), None);
        assert_eq!(raw_int(i16::MIN as i64 - 1, W2), None);
        assert_eq!(raw_int(1 << 40, W4), None);
        assert_eq!(raw_int(-5, W8), Some((-5_i64 as u64) ^ 1 << 63));
        let xs = [i32::MIN as i64, -7, -1, 0, 1, 7, i32::MAX as i64];
        assert!(xs.map(|x| raw_int(x, W4).unwrap()).is_sorted());
    }

    #[test]
    fn layouts() {
        use Width::*;
        let keys = Keys::new([(W4, true), (W2, false), (W2, true), (W16, false)]);
        let offsets = keys.as_slice().iter().map(|k| k.offset).collect::<Vec<_>>();
        assert_eq!(offsets, [0, 4, 6, 8]);
        assert_eq!(keys.columns(), 24);
        let descriptors = keys.descriptors();
        assert_eq!(descriptors, [0b101, 0b000, 0b100, 0b011]);
        assert_eq!(Keys::from_descriptors(&descriptors), Some(keys));
        assert_eq!(Keys::from_descriptors(&[]), None);
        assert_eq!(Keys::from_descriptors(&[0b1000]), None);
    }

    fn rows(keys: &Keys, wanted: &[Vec<u128>]) -> Vec<Vec<u8>> {
        (0..32)
            .map(|_| {
                let mut row = Vec::new();
                for (key, wanted) in std::iter::zip(keys.as_slice(), wanted) {
                    let mask = u128::MAX >> (128 - 8 * key.width.bytes());
                    let x = if rand::random::<bool>() {
                        wanted[rand::random::<u32>() as usize % wanted.len()]
                    } else {
                        rand::random::<u128>() & mask
                    };
                    row.extend_from_slice(&x.to_le_bytes()[..key.width.bytes()]);
                }
                row
            })
            .collect()
    }

    fn block(keys: &Keys, rows: &[Vec<u8>]) -> Vec<u64> {
        let mut bytes = Vec::new();
        for key in keys.as_slice() {
            for row in rows {
                bytes.extend_from_slice(&row[key.offset as usize..][..key.width.bytes()]);
            }
        }
        let mut words = vec![0_u64; bytes.len().div_ceil(8)];
        words.as_mut_bytes()[..bytes.len()].copy_from_slice(&bytes);
        words
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn filter() {
        use Width::*;
        let all = [
            Keys::new([(W2, true)]),
            Keys::new([(W4, true), (W2, false), (W16, false)]),
            Keys::new([(W8, false), (W2, true), (W2, true), (W4, false)]),
        ];
        for keys in &all {
            let n = keys.as_slice().len();
            for size in [1, 3, 5, 9, 17, 33, 200, 5000] {
                let wanted = (keys.as_slice().iter())
                    .map(|key| {
                        let mask = u128::MAX >> (128 - 8 * key.width.bytes());
                        (0..size).map(|_| rand::random::<u128>() & mask).collect()
                    })
                    .collect::<Vec<Vec<_>>>();
                for _ in 0..16 {
                    let constrained = rand::random::<u32>() % (1 << n);
                    let sets = (0..n)
                        .map(|i| (constrained >> i & 1 != 0).then(|| wanted[i].clone()))
                        .collect::<Vec<_>>();
                    let filter = KeyFilter::new(keys, sets).unwrap();
                    let rows = rows(keys, &wanted);
                    let words = block(keys, &rows);
                    let columns = &words.as_bytes()[..32 * keys.columns()];
                    let mask = (&filter).block(columns);
                    for (i, row) in rows.iter().enumerate() {
                        let expected = (0..n).all(|k| {
                            let key = keys.as_slice()[k];
                            let bytes = key.width.bytes();
                            let mut x = [0; 16];
                            x[..bytes].copy_from_slice(&row[key.offset as usize..][..bytes]);
                            let x = u128::from_le_bytes(x);
                            constrained >> k & 1 == 0 || wanted[k].contains(&x)
                        });
                        assert_eq!(mask >> i & 1 != 0, expected);
                        assert_eq!((&filter).row(Columns::row(row)), expected);
                        let mut copy = Vec::new();
                        Columns::block(columns, i).extend(keys, &mut copy);
                        assert_eq!(&copy, row);
                    }
                }
            }
        }
        assert!(KeyFilter::new(&all[0], vec![Some(Vec::new())]).is_none());
    }

    #[test]
    fn ranges() {
        use Width::*;
        let range = |min: u16, max: u16| min as u32 | (max as u32) << 16;
        let keys = Keys::new([(W2, true)]);
        let block = |values: &[u16]| {
            let payload =
                std::array::from_fn(|i| (i < values.len()).then(|| NonZero::new(1).unwrap()));
            let columns: [u16; 32] = std::array::from_fn(|i| values.get(i).copied().unwrap_or(0));
            keys.range_of_block(&payload, columns.as_bytes())
        };
        assert_eq!(block(&[]), EMPTY_RANGE);
        assert_eq!(block(&[7]), range(7, 7));
        assert_eq!(block(&[9, 0, 65535, 3]), range(0, 65535));
        let keys = Keys::new([(W4, false), (W2, true)]);
        let values: [u32; 32] = std::array::from_fn(|i| 0x0001_0000 * i as u32 + 5);
        let full = std::array::from_fn(|_| NonZero::new(1));
        let columns = [values.as_bytes(), [0; 64].as_slice()].concat();
        let r = keys.range_of_block(&full, &columns);
        let folds = values.map(|x| keys.as_slice()[0].sort_key(x as u128));
        let (min, max) = (*folds.iter().min().unwrap(), *folds.iter().max().unwrap());
        assert_eq!(r, range(min, max));
        let filter = KeyFilter::new(&keys, vec![Some(vec![values[3] as u128]), None]).unwrap();
        let mut ranges = [EMPTY_RANGE; 32];
        ranges[1] = range(folds[3], folds[3]);
        ranges[2] = range(folds[3].wrapping_add(1), folds[3].wrapping_add(1));
        assert_eq!((&filter).ranges(&ranges), 0b10);
        let filter = KeyFilter::new(&keys, vec![None, Some(vec![1])]).unwrap();
        assert!(!(&filter).selects_blocks());
        let keys = Keys::new([(W2, true)]);
        for n in [1, 40, 200, 5000] {
            let set = (0..n).map(|_| rand::random::<u16>()).collect::<Vec<_>>();
            let filter =
                KeyFilter::new(&keys, vec![Some(set.iter().map(|&x| x as u128).collect())])
                    .unwrap();
            for _ in 0..64 {
                let ranges: [u32; 32] = std::array::from_fn(|i| match i % 4 {
                    0 => EMPTY_RANGE,
                    1 => range(set[i % n], set[i % n]),
                    _ => {
                        let min = rand::random::<u16>();
                        range(min, min.saturating_add(rand::random::<u16>() >> (i % 16)))
                    }
                });
                let expected = (0..32).fold(0, |mask, i| {
                    let (min, max) = (ranges[i] as u16, (ranges[i] >> 16) as u16);
                    let hit = set.iter().any(|&t| min <= t && t <= max);
                    mask | (hit as u32) << i
                });
                assert_eq!((&filter).ranges(&ranges), expected);
            }
        }
    }
}
