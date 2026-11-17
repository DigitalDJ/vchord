// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

//! Stored forms, unsigned integers that order like their keys, and hashes are on disk for good.

use crate::tape::{Filter, Summaries, wide};
use crate::tuples::{AppendableTuple, FrozenTuple, MetaTuple, MetaTupleReader, WithReader};
use index::relation::{Page, RelationRead};
use simd::column::{Set, Value};
use std::cmp::Ordering;
use std::num::NonZero;
use std::ops::Range;
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

/// The bytes as a big-endian integer, which orders as PostgreSQL orders uuids.
pub const fn raw_uuid(x: &[u8; 16]) -> u128 {
    u128::from_be_bytes(*x)
}

/// What every frozen block keeps of a key in the directory, of one size whatever the kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Summary {
    /// The least and the greatest prefix of the block's stored forms: a zone map.
    Bounds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key {
    pub width: Width,
    pub hashed: bool,
    pub summary: Option<Summary>,
    /// Its bytes' offset in a row's columns.
    pub offset: u16,
}

impl Key {
    /// Whether `set` selects exactly its keys' rows: a NULL is stored as zero.
    fn exact(&self, set: &Intervals) -> bool {
        !self.hashed && !set.contains(0)
    }
    #[inline(always)]
    pub fn get(&self, columns: Columns<'_>) -> u128 {
        let bytes = self.width.bytes();
        let mut x = [0; 16];
        x[..bytes].copy_from_slice(columns.get(self.offset, bytes));
        u128::from_le_bytes(x)
    }
    /// The greatest stored form.
    pub const fn max(&self) -> u128 {
        u128::MAX >> (128 - 8 * self.width.bytes())
    }
    /// The shift of a stored form to its prefix of a word, which orders like it.
    const fn shift(&self) -> u32 {
        match self.width {
            Width::W16 => 64,
            _ => 0,
        }
    }
}

const MAX_KEYS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Keys {
    keys: [Key; MAX_KEYS],
    n: u8,
    summaries: u8,
    columns: u16,
}

impl Default for Keys {
    fn default() -> Self {
        Self::new([])
    }
}

impl Keys {
    pub fn new(keys: impl IntoIterator<Item = (Width, bool, Option<Summary>)>) -> Self {
        let free = Key {
            width: Width::W2,
            hashed: false,
            summary: None,
            offset: 0,
        };
        let mut this = Self {
            keys: [free; MAX_KEYS],
            n: 0,
            summaries: 0,
            columns: 0,
        };
        for (width, hashed, summary) in keys {
            this.keys[this.n as usize] = Key {
                width,
                hashed,
                summary,
                offset: this.columns,
            };
            this.columns += width.bytes() as u16;
            this.summaries += summary.is_some() as u8;
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
    /// How many keys every frozen block keeps a summary of.
    pub fn summaries(&self) -> usize {
        self.summaries as usize
    }
    /// Orders rows by their stored forms, the first key first.
    pub fn cmp(&self, a: Columns<'_>, b: Columns<'_>) -> Ordering {
        for key in self.as_slice() {
            match key.get(a).cmp(&key.get(b)) {
                Ordering::Equal => continue,
                x => return x,
            }
        }
        Ordering::Equal
    }
    /// The summaries of a block's live rows, a key's after another's.
    pub fn summarize<'a>(
        &'a self,
        payload: &'a [Option<NonZero<u64>>; 32],
        columns: &'a [u8],
    ) -> impl Iterator<Item = [u64; 2]> + 'a {
        self.as_slice().iter().filter_map(move |key| {
            let rows = (0..32).filter(|&i| payload[i].is_some());
            let stored = rows.map(|i| key.get(Columns::block(columns, i)));
            Some(match key.summary? {
                Summary::Bounds => (stored.map(|x| (x >> key.shift()) as u64))
                    .fold([u64::MAX, 0], |[min, max], x| [min.min(x), max.max(x)]),
            })
        })
    }
    /// The columns of a block: each key's values in order, from the rows' own.
    pub fn block(&self, rows: &[u8]) -> Vec<u8> {
        let mut block = Vec::with_capacity(rows.len());
        for key in self.as_slice() {
            for row in rows.chunks_exact(self.columns()) {
                block.extend_from_slice(&row[key.offset as usize..][..key.width.bytes()]);
            }
        }
        block
    }
    /// A byte per key: its width, whether it is hashed and its summary.
    pub fn descriptors(&self) -> Vec<u8> {
        let summary = |key: &Key| key.summary.map_or(0, |x| 1 + x as u8);
        let descriptor = |key: &Key| key.width as u8 | (key.hashed as u8) << 2 | summary(key) << 3;
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
        if descriptors.iter().any(|&x| x >> 3 > 1) {
            return None;
        }
        Some(Self::new(descriptors.iter().map(|&x| {
            let width = Width::of_bytes(2 << (x & 3)).unwrap();
            (width, x & 4 != 0, (x >> 3 == 1).then_some(Summary::Bounds))
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

/// The stored forms that a key may hold: sorted, disjoint, inclusive intervals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Intervals(Vec<[u128; 2]>);

impl Intervals {
    pub fn new(mut intervals: Vec<[u128; 2]>) -> Self {
        intervals.retain(|x| x[0] <= x[1]);
        intervals.sort_unstable();
        let mut merged: Vec<[u128; 2]> = Vec::with_capacity(intervals.len());
        for x in intervals {
            match merged.last_mut() {
                Some(last) if last[1] >= x[0] => last[1] = last[1].max(x[1]),
                _ => merged.push(x),
            }
        }
        Self(merged)
    }
    #[cfg(test)]
    pub fn points(points: Vec<u128>) -> Self {
        Self::new(points.into_iter().map(|x| [x, x]).collect())
    }
    pub fn intersect(&self, other: &Self) -> Self {
        let (mut i, mut j, mut both) = (0, 0, Vec::new());
        while let (Some(&[a, b]), Some(&[c, d])) = (self.0.get(i), other.0.get(j)) {
            if a.max(c) <= b.min(d) {
                both.push([a.max(c), b.min(d)]);
            }
            if b < d {
                i += 1;
            } else {
                j += 1;
            }
        }
        Self(both)
    }
    fn contains(&self, x: u128) -> bool {
        let i = self.0.partition_point(|y| y[1] < x);
        self.0.get(i).is_some_and(|y| y[0] <= x)
    }
    /// The stored forms, if every interval holds one.
    fn as_points(&self) -> Option<Vec<u128>> {
        self.0.iter().map(|&[a, b]| (a == b).then_some(a)).collect()
    }
    /// The intervals of the stored forms' prefixes, `x >> shift`, merged where they meet.
    fn prefixes(&self, shift: u32) -> Vec<[u64; 2]> {
        let mut prefixes: Vec<[u64; 2]> = Vec::with_capacity(self.0.len());
        for &[a, b] in &self.0 {
            let [a, b] = [(a >> shift) as u64, (b >> shift) as u64];
            match prefixes.last_mut() {
                Some(last) if last[1] >= a => last[1] = last[1].max(b),
                _ => prefixes.push([a, b]),
            }
        }
        prefixes
    }
}

/// Up to this many intervals are tested a word of blocks at a time, more one block at a time.
const FEW_INTERVALS: usize = 8;

/// The blocks of `blocks`, within one word, whose bounds meet one of `set`, a bit each.
#[inline(always)]
fn meets(set: &[[u64; 2]], mins: &[[u32; 2]], maxs: &[[u32; 2]], blocks: Range<usize>) -> u32 {
    if set.len() <= FEW_INTERVALS {
        // the word's bounds as a column; past its blocks, any
        let word = |x: &[[u32; 2]]| -> [u64; 32] {
            let x = &x[blocks.clone()];
            std::array::from_fn(|i| x.get(i).map_or(0, |&w| wide(w).to_le()))
        };
        let (mins, maxs) = (word(mins), word(maxs));
        let mask = set.iter().fold(0, |mask, &[lo, hi]| {
            let below = u64::between(&mins, 0, hi.to_le());
            mask | below & u64::between(&maxs, lo.to_le(), u64::MAX)
        });
        return mask << (blocks.start % 32);
    }
    let (mut mask, mut i, mut last) = (0, 0, 0);
    for b in blocks {
        let (min, max) = (wide(mins[b]), wide(maxs[b]));
        // the intervals before `i` end below `last`, so below `min` while the bounds do not fall
        if min < last {
            i = 0;
        }
        last = min;
        if set.get(i).is_some_and(|x| x[1] < min) {
            i += set[i..].partition_point(|x| x[1] < min);
        }
        mask |= (set.get(i).is_some_and(|x| x[0] <= max) as u32) << (b % 32);
    }
    mask
}

#[derive(Debug)]
enum Values<T> {
    Few(Vec<T>),
    Many(Box<Set<T>>),
    /// Inclusive bounds, as the column stores them.
    Ranges(Vec<[T; 2]>),
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

    /// `form`: a stored form as the column stores it.
    fn new(set: &Intervals, form: impl Fn(u128) -> T) -> Self {
        match set.as_points() {
            Some(points) if points.len() <= Self::FEW => {
                Self::Few(points.into_iter().map(form).collect())
            }
            Some(points) => Self::Many(Box::new(Set::new(
                &points.into_iter().map(form).collect::<Vec<_>>(),
            ))),
            None => Self::Ranges(set.0.iter().map(|&[lo, hi]| [form(lo), form(hi)]).collect()),
        }
    }

    #[inline(always)]
    fn matches(&self, column: &[T; 32]) -> u32 {
        match self {
            Self::Few(keys) => T::matches(column, keys),
            Self::Many(set) => set.matches(column),
            Self::Ranges(ranges) => {
                (ranges.iter()).fold(0, |mask, &[lo, hi]| mask | T::between(column, lo, hi))
            }
        }
    }

    #[inline(always)]
    fn contains(&self, x: T) -> bool {
        match self {
            Self::Few(keys) => keys.contains(&x),
            Self::Many(set) => set.contains(x),
            Self::Ranges(ranges) => {
                let x = x.ordinal();
                (ranges.iter()).any(|&[lo, hi]| lo.ordinal() <= x && x <= hi.ordinal())
            }
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
    fn new(width: Width, set: &Intervals) -> Self {
        match width {
            Width::W2 => Self::U16(Values::new(set, |x| (x as u16).to_le())),
            Width::W4 => Self::U32(Values::new(set, |x| (x as u32).to_le())),
            Width::W8 => Self::U64(Values::new(set, |x| (x as u64).to_le())),
            Width::W16 => Self::U128(Values::new(set, |x| {
                [(x as u64).to_le(), ((x >> 64) as u64).to_le()]
            })),
        }
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
    /// Per constrained summarized key, its place among the summaries and the prefixes to meet.
    blocks: Vec<(usize, Vec<[u64; 2]>)>,
    columns: Vec<(u16, Column)>,
    exact: bool,
}

impl KeyFilter {
    /// `sets`: per key, the stored forms a row may hold, or `None` for any; `None` if none can.
    pub fn new(keys: &Keys, sets: impl IntoIterator<Item = Option<Intervals>>) -> Option<Self> {
        let mut filter = Self {
            blocks: Vec::new(),
            columns: Vec::new(),
            exact: true,
        };
        let mut place = 0;
        for (key, set) in std::iter::zip(keys.as_slice(), sets) {
            place += key.summary.is_some() as usize;
            let Some(set) = set else { continue };
            if set.0.is_empty() {
                return None;
            }
            filter.exact &= key.exact(&set);
            if set.0 == [[0, key.max()]] {
                continue;
            }
            match key.summary {
                Some(Summary::Bounds) => {
                    filter.blocks.push((place - 1, set.prefixes(key.shift())));
                }
                None => {}
            }
            filter
                .columns
                .push((key.offset, Column::new(key.width, &set)));
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
        !self.blocks.is_empty()
    }

    #[inline(always)]
    fn span(self, summaries: Summaries<'_>) -> Range<usize> {
        match self.blocks.first() {
            Some((0, set)) if summaries.sorted() => {
                let (min, max) = (set[0][0], set[set.len() - 1][1]);
                let start = summaries.words(0, 1).partition_point(|&x| wide(x) < min);
                let end = summaries.words(0, 0).partition_point(|&x| wide(x) <= max);
                start..end.max(start)
            }
            _ => 0..summaries.blocks(),
        }
    }

    #[inline(always)]
    fn meets(self, summaries: Summaries<'_>, blocks: Range<usize>) -> u32 {
        let mut mask = u32::MAX;
        for (k, set) in &self.blocks {
            let [first, second] = [0, 1].map(|i| summaries.words(*k, i));
            mask &= meets(set, first, second, blocks.clone());
        }
        mask
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
    use crate::tape::{Layout, SORTED};
    use Summary::Bounds;
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
        let mut uuids = (0..64)
            .map(|_| rand::random::<[u8; 16]>())
            .collect::<Vec<_>>();
        uuids.sort_unstable();
        assert!(uuids.iter().map(raw_uuid).is_sorted());
    }

    #[test]
    fn layouts() {
        use Width::*;
        let keys = Keys::new([
            (W4, true, Some(Bounds)),
            (W2, false, None),
            (W2, true, None),
            (W16, false, Some(Bounds)),
        ]);
        let offsets = keys.as_slice().iter().map(|k| k.offset).collect::<Vec<_>>();
        assert_eq!(offsets, [0, 4, 6, 8]);
        assert_eq!(keys.columns(), 24);
        assert_eq!(keys.summaries(), 2);
        let descriptors = keys.descriptors();
        assert_eq!(descriptors, [0b01101, 0b00000, 0b00100, 0b01011]);
        assert_eq!(Keys::from_descriptors(&descriptors), Some(keys));
        assert_eq!(Keys::from_descriptors(&[]), None);
        assert_eq!(Keys::from_descriptors(&[0b10000]), None);
        assert_eq!(Keys::from_descriptors(&[0b0000, 0b100000]), None);
    }

    fn stored(key: &Key, row: &[u8]) -> u128 {
        key.get(Columns::row(row))
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
        let bytes = keys.block(&rows.concat());
        let mut words = vec![0_u64; bytes.len().div_ceil(8)];
        words.as_mut_bytes()[..bytes.len()].copy_from_slice(&bytes);
        words
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn filter() {
        use Width::*;
        let all = [
            Keys::new([(W2, true, Some(Bounds))]),
            Keys::new([(W4, true, None), (W2, false, None), (W16, false, None)]),
            Keys::new([
                (W8, false, Some(Bounds)),
                (W2, true, Some(Bounds)),
                (W2, true, None),
                (W4, false, Some(Bounds)),
            ]),
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
                    // per key its points, or the interval from its least to its greatest
                    let sets = (0..n)
                        .map(|i| {
                            let (lo, hi) = (wanted[i].iter().min(), wanted[i].iter().max());
                            let set = if rand::random::<bool>() {
                                Intervals::points(wanted[i].clone())
                            } else {
                                Intervals::new(vec![[*lo.unwrap(), *hi.unwrap()]])
                            };
                            (constrained >> i & 1 != 0).then_some(set)
                        })
                        .collect::<Vec<_>>();
                    let filter = KeyFilter::new(keys, sets.clone()).unwrap();
                    let rows = rows(keys, &wanted);
                    let words = block(keys, &rows);
                    let columns = &words.as_bytes()[..32 * keys.columns()];
                    let mask = (&filter).block(columns);
                    for (i, row) in rows.iter().enumerate() {
                        let expected = (0..n).all(|k| {
                            let x = stored(&keys.as_slice()[k], row);
                            constrained >> k & 1 == 0 || sets[k].as_ref().unwrap().contains(x)
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
        assert!(KeyFilter::new(&all[0], vec![Some(Intervals::points(Vec::new()))]).is_none());
        let every = Intervals::new(vec![[0, u16::MAX as u128]]);
        let filter = KeyFilter::new(&all[1], vec![None, Some(every), None]).unwrap();
        assert!(!filter.exact() && (&filter).block(&[0; 32 * 22]) == u32::MAX);
        let filter = KeyFilter::new(
            &all[1],
            vec![None, Some(Intervals::new(vec![[5, 9]])), None],
        );
        assert!(filter.unwrap().exact());
        let filter = KeyFilter::new(&all[1], vec![None, Some(Intervals::points(vec![7])), None]);
        assert!(filter.unwrap().exact());
        let filter = KeyFilter::new(
            &all[1],
            vec![None, Some(Intervals::points(vec![0, 7])), None],
        );
        assert!(!filter.unwrap().exact());
    }

    #[test]
    fn intervals() {
        let set = Intervals::new(vec![[9, 12], [1, 3], [3, 4], [20, 19], [7, 7]]);
        assert_eq!(set.0, [[1, 4], [7, 7], [9, 12]]);
        assert!(set.contains(4) && set.contains(7) && !set.contains(5) && !set.contains(13));
        assert_eq!(set.as_points(), None);
        let points = Intervals::points(vec![12, 3, 7, 3, 30]);
        assert_eq!(points.as_points(), Some(vec![3, 7, 12, 30]));
        assert_eq!(set.intersect(&points), Intervals::points(vec![3, 7, 12]));
        assert!(points.intersect(&Intervals::points(vec![5])).0.is_empty());
        let high = |x: u128| x << 64 | 5;
        let set = Intervals::points(vec![high(1), high(2) - 7, high(2), high(9)]);
        assert_eq!(set.prefixes(64), [[1, 1], [2, 2], [9, 9]]);
    }

    /// Rows sorted by `Keys::cmp`, written as blocks: their summaries and the directory's layout.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn summaries() {
        use Width::*;
        let keys = Keys::new([
            (W16, false, Some(Bounds)),
            (W2, false, Some(Bounds)),
            (W8, true, None),
        ]);
        let owners = (0..12).map(|_| rand::random::<u128>()).collect::<Vec<_>>();
        let mut rows = (0..32 * 40)
            .map(|_| {
                let owner = owners[rand::random::<u32>() as usize % owners.len()];
                let tag = rand::random::<u16>() % 50;
                [
                    &owner.to_le_bytes()[..],
                    &tag.to_le_bytes(),
                    &rand::random::<[u8; 8]>(),
                ]
                .concat()
            })
            .collect::<Vec<_>>();
        rows.sort_by(|a, b| keys.cmp(Columns::row(a), Columns::row(b)));
        let tuple = |row: &Vec<u8>| {
            keys.as_slice()
                .iter()
                .map(|k| stored(k, row))
                .collect::<Vec<_>>()
        };
        assert!(rows.iter().map(tuple).is_sorted());
        let payload = std::array::from_fn(|i| (i != 3).then(|| NonZero::new(1).unwrap()));
        let mut layout = Layout::new(keys.summaries());
        layout.page(7);
        let blocks = rows
            .chunks(32)
            .map(|rows| block(&keys, rows))
            .collect::<Vec<_>>();
        for (j, words) in blocks.iter().enumerate() {
            let columns = &words.as_bytes()[..32 * keys.columns()];
            let zones = keys.summarize(&payload, columns).collect::<Vec<_>>();
            let live = rows[32 * j..][..32]
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != 3);
            let owner = live
                .clone()
                .map(|(_, row)| (stored(&keys.as_slice()[0], row) >> 64) as u64);
            let tag = live.map(|(_, row)| stored(&keys.as_slice()[1], row) as u64);
            assert_eq!(
                zones[0],
                [owner.clone().min().unwrap(), owner.max().unwrap()]
            );
            assert_eq!(zones[1], [tag.clone().min().unwrap(), tag.max().unwrap()]);
            layout.tuple(Some(zones));
        }
        let mut words = layout.ids.clone();
        words.extend(layout.words());
        assert_eq!(words[1], 0);
        assert_eq!(words[2], blocks.len() as u32 | SORTED);
        assert_eq!(words.len(), 3 + 4 * 2 * blocks.len());
        let summaries = Summaries::new(&words[3..], blocks.len(), true);
        for _ in 0..64 {
            let pick = |n: usize| owners[rand::random::<u32>() as usize % n];
            let sets = [
                Some(Intervals::points(
                    (0..1 + rand::random::<u32>() % 3)
                        .map(|_| pick(12))
                        .collect(),
                )),
                (rand::random::<bool>())
                    .then(|| Intervals::points(vec![rand::random::<u128>() % 50])),
                None,
            ];
            let filter = KeyFilter::new(&keys, sets.clone()).unwrap();
            let span = (&filter).span(summaries);
            for (j, words) in blocks.iter().enumerate() {
                let columns = &words.as_bytes()[..32 * keys.columns()];
                let live = (&filter).block(columns) & !(1 << 3) != 0;
                let selected = span.contains(&j) && (&filter).meets(summaries, j..j + 1) != 0;
                assert!(
                    selected || !live,
                    "block {j} holds a row, but is not selected"
                );
            }
        }
    }
}
