// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

//! Filter keys. The key columns of an index after its vector are its filter
//! keys, and every row stores each of them in 2, 4, 8 or 16 bytes ([`Key`]):
//! the value itself when its type has that size, and a hash of it in fewer
//! bytes otherwise. A search with equality keys on some of them reads only
//! the rows whose stored keys a [`KeyFilter`] selects. Hashes are lossy, so
//! the caller rechecks a hashed key on the rows returned.
//!
//! The first key stored in 2 bytes takes the low 16 bits of every payload,
//! which nothing else uses; every other key is a column of each frozen block,
//! the 32 rows' values in order, and a field of each appendable row
//! ([`Keys`]). An index's keys are described in its meta tuple.
//!
//! The hash functions and the stored forms are part of the on-disk format:
//! they must never change.

use crate::tape::Filter;
use crate::tuples::{AppendableTuple, FrozenTuple, MetaTuple, MetaTupleReader, WithReader};
use index::relation::{Page, RelationRead};
use simd::column::{Set, Value};
use std::num::NonZero;
use zerocopy::{FromBytes, Immutable, KnownLayout};

/// The splitmix64 finalizer: every output bit depends on every input bit.
const fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The hash of an integer key. Integers of every width are widened first, so
/// that equality across widths agrees with the hashes.
pub const fn hash_int(x: i64) -> u64 {
    mix(x as u64 ^ 0xd1b5_4a32_d192_ed03)
}

/// The hash of a uuid key, given its 16 bytes in storage order.
pub const fn hash_uuid(x: &[u8; 16]) -> u64 {
    let [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p] = *x;
    let lo = u64::from_le_bytes([a, b, c, d, e, f, g, h]);
    let hi = u64::from_le_bytes([i, j, k, l, m, n, o, p]);
    mix(lo ^ mix(hi ^ 0x9e37_79b9_7f4a_7c15))
}

/// A hash in `bytes` bytes (2, 4 or 8): its halves folded together with an
/// exclusive or until it fits; in 2 bytes, the fold of its 16-bit lanes
/// ([`Value::fold_one`]).
pub fn fold(hash: u64, bytes: usize) -> u64 {
    match bytes {
        8 => hash,
        4 => (hash ^ hash >> 32) & 0xffff_ffff,
        _ => hash.fold_one() as u64,
    }
}

/// The stored form of an integer key of a type of `bytes` bytes, stored
/// whole: its two's complement with the sign bit flipped, so that the stored
/// forms order as the keys do. `None` if the key does not fit the type.
pub const fn raw_int(x: i64, bytes: usize) -> Option<u64> {
    let bits = 8 * bytes as u32;
    if bits < 64 && (x < -(1 << (bits - 1)) || x >= 1 << (bits - 1)) {
        return None;
    }
    Some((x as u64 ^ 1 << (bits - 1)) & (u64::MAX >> (64 - bits)))
}

/// How a key is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key {
    /// 2, 4, 8 or 16.
    pub bytes: usize,
    /// A hash of the key, or the key itself.
    pub hashed: bool,
    /// In the payload, or at an offset of a row's columns.
    pub slot: Slot,
}

impl Key {
    /// Whether the rows whose stored keys are in `set` hold exactly those
    /// keys: those of a key stored whole, except a NULL, which is stored as
    /// zeros like one key (the least integer, the nil uuid).
    pub fn exact(&self, set: &[u128]) -> bool {
        !self.hashed && !set.contains(&0)
    }
    /// The stored form, the low `bytes` bytes of the result in little-endian
    /// order, of a key whose whole form is `raw` and whose hash is `hash`.
    pub fn stored(&self, raw: u128, hash: u64) -> u128 {
        if self.hashed {
            fold(hash, self.bytes) as u128
        } else {
            raw
        }
    }
}

/// Where a key is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    /// The low 16 bits of the payload.
    Payload,
    /// The byte offset of the key in a row's columns.
    Column(usize),
}

/// The filter keys of an index, in the order of its key columns, and where
/// each is stored; none for an index without filter keys.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Keys {
    keys: Vec<Key>,
    /// The bytes of a row's columns.
    columns: usize,
}

impl Keys {
    /// Keys stored as `(bytes, hashed)` each: the first stored in 2 bytes
    /// takes the payload, and the others the columns, in order.
    pub fn new(keys: impl IntoIterator<Item = (usize, bool)>) -> Self {
        let mut payload = false;
        let mut columns = 0;
        let keys = keys
            .into_iter()
            .map(|(bytes, hashed)| {
                assert!(matches!(bytes, 2 | 4 | 8 | 16), "bad width of a filter key");
                let slot = if bytes == 2 && !payload {
                    payload = true;
                    Slot::Payload
                } else {
                    columns += bytes;
                    Slot::Column(columns - bytes)
                };
                Key {
                    bytes,
                    hashed,
                    slot,
                }
            })
            .collect();
        Self { keys, columns }
    }
    pub fn as_slice(&self) -> &[Key] {
        &self.keys
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    /// The bytes of a row's columns; those of a block are 32 times as many.
    pub fn columns(&self) -> usize {
        self.columns
    }
    /// The columns of a block, each key's 32 values in order, of 32 rows'
    /// own (`rows`, a row's after another's).
    pub fn block(&self, rows: &[u8]) -> Vec<u8> {
        let mut block = Vec::with_capacity(rows.len());
        for key in &self.keys {
            if let Slot::Column(offset) = key.slot {
                for row in rows.chunks_exact(self.columns) {
                    block.extend_from_slice(&row[offset..][..key.bytes]);
                }
            }
        }
        block
    }
    /// The keys that a meta tuple without descriptors stands for: one key,
    /// hashed into the payload, as indexes with a filter key column were
    /// written before keys of other widths existed.
    fn single() -> Self {
        Self::new([(2, true)])
    }
    /// The descriptors of the keys in a meta tuple, a byte each: the width
    /// (bits 0 and 1, `2 << x` bytes), whether hashed (bit 2) and whether in
    /// the payload (bit 3). None for [`Keys::single`].
    pub fn descriptors(&self) -> Vec<u8> {
        if *self == Self::single() {
            return Vec::new();
        }
        let descriptor = |key: &Key| {
            let width = key.bytes.trailing_zeros() as u8 - 1;
            width | (key.hashed as u8) << 2 | ((key.slot == Slot::Payload) as u8) << 3
        };
        self.keys.iter().map(descriptor).collect()
    }
    /// The keys of an index; [`Keys::default`] for an index without filter
    /// keys.
    pub fn of<R: RelationRead>(index: &R) -> Self {
        let meta_guard = index.read(0);
        let meta_bytes = meta_guard.get(1).expect("data corruption");
        Self::of_meta(MetaTuple::deserialize_ref(meta_bytes))
    }
    /// The keys of an index, from its meta tuple.
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
    /// The keys that the descriptors of a meta tuple describe; `None` for a
    /// layout that this binary does not know.
    pub fn from_descriptors(descriptors: &[u8]) -> Option<Self> {
        if descriptors.is_empty() {
            return Some(Self::single());
        }
        if descriptors.iter().any(|&x| x >> 4 != 0) {
            return None;
        }
        let keys = Self::new(descriptors.iter().map(|&x| (2 << (x & 3), x & 4 != 0)));
        // the slots must be those that `new` gives
        (keys.descriptors() == descriptors).then_some(keys)
    }
    /// The most bytes of columns that a row may have in an index whose
    /// vectors take `words` words of signs and `prefetch` page ids, on pages
    /// with `freespace` bytes when empty: a block's last part carries its 32
    /// rows' columns, even with no elements, and an appendable tuple a row's,
    /// and each must fit a page.
    pub fn most_columns(words: usize, prefetch: usize, freespace: usize) -> usize {
        let block = freespace.saturating_sub(FrozenTuple::estimate_size_0(prefetch, 0, 0)) / 32;
        let row = freespace.saturating_sub(AppendableTuple::estimate_size(prefetch, words, 0));
        block.min(row)
    }
}

/// The filter key columns of a row: in a frozen block, the block's columns
/// (each key's 32 values in order) and the row; in an appendable tuple, the
/// row's own.
#[derive(Debug, Clone, Copy)]
pub struct Columns<'a> {
    bytes: &'a [u8],
    /// The row in a block, or `None` for a row's own.
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
    /// The stored key at `offset` of a row's columns, of `bytes` bytes.
    #[inline(always)]
    pub fn get(self, offset: usize, bytes: usize) -> &'a [u8] {
        match self.row {
            None => &self.bytes[offset..][..bytes],
            Some(row) => &self.bytes[32 * offset + row as usize * bytes..][..bytes],
        }
    }
    /// Appends the row's columns, as an appendable tuple stores them.
    pub fn extend(self, keys: &Keys, row: &mut Vec<u8>) {
        if self.row.is_none() {
            row.extend_from_slice(&self.bytes[..keys.columns]);
            return;
        }
        for key in &keys.keys {
            if let Slot::Column(offset) = key.slot {
                row.extend_from_slice(self.get(offset, key.bytes));
            }
        }
    }
}

/// A set of stored keys of one width.
#[derive(Debug)]
enum Values<T> {
    /// A few, each compared with a whole block at once.
    Few(Vec<T>),
    /// A set too large to compare key by key.
    Many(Box<Set<T>>),
}

impl<T: Value> Values<T> {
    /// The most keys compared one by one; a larger set is a [`Set`]: where
    /// the cost of comparing meets the set's (`simd::column`'s benchmark), on
    /// a Zen 4 with AVX2 or AVX-512, and on a Cortex-A76 with NEON for other
    /// targets.
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

impl Values<u16> {
    /// The rows whose payload's low 16 bits are in the set.
    #[inline(always)]
    fn matches_payload(&self, payload: &[u64; 32]) -> u32 {
        match self {
            Self::Few(keys) => simd::column::matches_payload(payload, keys),
            Self::Many(set) => set.matches_payload(payload),
        }
    }
}

/// The set of a key in the columns of a block, by its width.
#[derive(Debug)]
enum Column {
    U16(Values<u16>),
    U32(Values<u32>),
    U64(Values<u64>),
    U128(Values<[u64; 2]>),
}

/// The values of a column of a block's columns, at `offset` of a row's.
#[inline(always)]
fn column<T: FromBytes + Immutable + KnownLayout>(columns: &[u8], offset: usize) -> &[T; 32] {
    let bytes = &columns[32 * offset..][..32 * size_of::<T>()];
    <[T; 32]>::ref_from_bytes(bytes).expect("unaligned filter key column")
}

impl Column {
    /// The rows store keys in little-endian order, which is what a column is
    /// read as here: on a big-endian target, the values compared with it are
    /// their byte-swapped forms (`to_le`).
    fn new(bytes: usize, keys: &[u128]) -> Option<Self> {
        Some(match bytes {
            2 => Self::U16(Values::new(
                keys.iter().map(|&x| (x as u16).to_le()).collect(),
            )?),
            4 => Self::U32(Values::new(
                keys.iter().map(|&x| (x as u32).to_le()).collect(),
            )?),
            8 => Self::U64(Values::new(
                keys.iter().map(|&x| (x as u64).to_le()).collect(),
            )?),
            _ => Self::U128(Values::new(
                (keys.iter())
                    .map(|&x| [(x as u64).to_le(), ((x >> 64) as u64).to_le()])
                    .collect(),
            )?),
        })
    }

    #[inline(always)]
    fn matches(&self, columns: &[u8], offset: usize) -> u32 {
        match self {
            Self::U16(values) => values.matches(column(columns, offset)),
            Self::U32(values) => values.matches(column(columns, offset)),
            Self::U64(values) => values.matches(column(columns, offset)),
            Self::U128(values) => values.matches(column(columns, offset)),
        }
    }

    #[inline(always)]
    fn contains(&self, columns: Columns<'_>, offset: usize) -> bool {
        let read = |bytes| columns.get(offset, bytes);
        match self {
            Self::U16(values) => values.contains(FromBytes::read_from_bytes(read(2)).unwrap()),
            Self::U32(values) => values.contains(FromBytes::read_from_bytes(read(4)).unwrap()),
            Self::U64(values) => values.contains(FromBytes::read_from_bytes(read(8)).unwrap()),
            Self::U128(values) => values.contains(FromBytes::read_from_bytes(read(16)).unwrap()),
        }
    }
}

/// The rows a search reads: those whose stored keys are in the sets of the
/// keys it constrains.
#[derive(Debug)]
pub struct KeyFilter {
    /// The set of the key in the payload, if it is constrained.
    payload: Option<Values<u16>>,
    /// The sets of the constrained keys in the columns, at their offsets.
    columns: Vec<(usize, Column)>,
}

impl KeyFilter {
    /// `sets`: per key, the stored forms that a row may hold ([`Key::stored`]),
    /// or `None` for any. `None` if a set is empty, so that no row matches.
    pub fn new(keys: &Keys, sets: Vec<Option<Vec<u128>>>) -> Option<Self> {
        assert_eq!(sets.len(), keys.keys.len());
        let mut filter = Self {
            payload: None,
            columns: Vec::new(),
        };
        for (key, set) in std::iter::zip(&keys.keys, sets) {
            let Some(set) = set else { continue };
            match key.slot {
                Slot::Payload => {
                    let keys = set.iter().map(|&x| x as u16).collect();
                    filter.payload = Some(Values::new(keys)?);
                }
                Slot::Column(offset) => {
                    filter.columns.push((offset, Column::new(key.bytes, &set)?));
                }
            }
        }
        Some(filter)
    }
}

impl Filter for &KeyFilter {
    #[inline(always)]
    fn block(self, payload: &[Option<NonZero<u64>>; 32], columns: &[u8]) -> u32 {
        let mut mask = match &self.payload {
            Some(values) => values.matches_payload(zerocopy::transmute_ref!(payload)),
            None => u32::MAX,
        };
        for (offset, column) in &self.columns {
            if mask == 0 {
                break;
            }
            mask &= column.matches(columns, *offset);
        }
        mask
    }

    #[inline(always)]
    fn row(self, payload: NonZero<u64>, columns: Columns<'_>) -> bool {
        if let Some(values) = &self.payload
            && !values.contains(payload.get() as u16)
        {
            return false;
        }
        (self.columns.iter()).all(|(offset, column)| column.contains(columns, *offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::IntoBytes;

    /// The 16-bit hashes of indexes with a single filter key column before
    /// keys of other widths existed, which are on disk: these values must
    /// never change.
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
        .map(|x| fold(x, 2) as u16);
        assert_eq!(hashes, GOLDEN);
        // folds to every width agree with folding further, and with the
        // kernels' folds
        for x in [
            hash_int(7),
            hash_uuid(&[9; 16]),
            u64::MAX,
            0x0123_4567_89ab_cdef,
        ] {
            assert_eq!(fold(fold(x, 4), 2), fold(x, 2));
            assert_eq!(fold(x, 8), x);
            assert_eq!(fold(x, 4), (x ^ x >> 32) & 0xffff_ffff);
            assert_eq!(fold(x, 2) as u16, x.fold_one());
        }
    }

    const GOLDEN: [u16; 9] = [
        55156, 44833, 34830, 26900, 22934, 57732, 51418, 37919, 34892,
    ];

    /// Two keys with the same 16-bit hash, used by the SQL tests of the
    /// recheck.
    #[test]
    fn collision() {
        assert_eq!(fold(hash_int(1), 2), fold(hash_int(52973), 2));
        let uuid = |x: u128| x.to_be_bytes();
        assert_eq!(
            fold(hash_uuid(&uuid(1)), 2),
            fold(hash_uuid(&uuid(0x145ce)), 2)
        );
    }

    #[test]
    fn raw() {
        assert_eq!(raw_int(0, 2), Some(0x8000));
        assert_eq!(raw_int(-1, 2), Some(0x7fff));
        assert_eq!(raw_int(i16::MIN as i64, 2), Some(0));
        assert_eq!(raw_int(i16::MAX as i64 + 1, 2), None);
        assert_eq!(raw_int(i16::MIN as i64 - 1, 2), None);
        assert_eq!(raw_int(1 << 40, 4), None);
        assert_eq!(raw_int(-5, 8), Some((-5_i64 as u64) ^ 1 << 63));
        // the stored forms order as the keys
        let xs = [i32::MIN as i64, -7, -1, 0, 1, 7, i32::MAX as i64];
        assert!(xs.map(|x| raw_int(x, 4).unwrap()).is_sorted());
    }

    #[test]
    fn layouts() {
        let keys = Keys::new([(4, true), (2, false), (2, true), (16, false)]);
        let slots = keys.as_slice().iter().map(|k| k.slot).collect::<Vec<_>>();
        assert_eq!(
            slots,
            [
                Slot::Column(0),
                Slot::Payload,
                Slot::Column(4),
                Slot::Column(6)
            ]
        );
        assert_eq!(keys.columns(), 22);
        let descriptors = keys.descriptors();
        assert_eq!(descriptors, [0b0101, 0b1000, 0b0100, 0b0011]);
        assert_eq!(Keys::from_descriptors(&descriptors), Some(keys));
        // the single hashed key in the payload has no descriptors
        let single = Keys::new([(2, true)]);
        assert!(single.descriptors().is_empty());
        assert_eq!(Keys::from_descriptors(&[]), Some(single));
        // a raw key in the payload, or two keys, have
        assert_eq!(Keys::new([(2, false)]).descriptors(), [0b1000]);
        assert_eq!(
            Keys::new([(2, true), (2, true)]).descriptors(),
            [0b1100, 0b0100]
        );
        // unknown bits, and slots that `new` does not give
        assert_eq!(Keys::from_descriptors(&[0b1_0000]), None);
        assert_eq!(Keys::from_descriptors(&[0b0100]), None);
        assert_eq!(Keys::from_descriptors(&[0b1101]), None);
        assert_eq!(Keys::from_descriptors(&[0b1100, 0b1100]), None);
    }

    /// The rows of a block, as (payload, row columns), about half of whose
    /// keys are from `wanted`.
    fn rows(keys: &Keys, wanted: &[Vec<u128>]) -> Vec<(NonZero<u64>, Vec<u8>)> {
        (0..32)
            .map(|_| {
                let mut payload = rand::random::<u64>() << 16 | 1 << 16;
                let mut row = Vec::new();
                for (key, wanted) in std::iter::zip(keys.as_slice(), wanted) {
                    let mask = u128::MAX >> (128 - 8 * key.bytes);
                    let x = if rand::random::<bool>() {
                        wanted[rand::random::<u32>() as usize % wanted.len()]
                    } else {
                        rand::random::<u128>() & mask
                    };
                    match key.slot {
                        Slot::Payload => payload |= x as u64,
                        Slot::Column(_) => row.extend_from_slice(&x.to_le_bytes()[..key.bytes]),
                    }
                }
                (NonZero::new(payload).unwrap(), row)
            })
            .collect()
    }

    /// The columns of a block of rows, aligned as on a page.
    fn block(keys: &Keys, rows: &[(NonZero<u64>, Vec<u8>)]) -> Vec<u64> {
        let mut bytes = Vec::new();
        for key in keys.as_slice() {
            if let Slot::Column(offset) = key.slot {
                for (_, row) in rows {
                    bytes.extend_from_slice(&row[offset..][..key.bytes]);
                }
            }
        }
        let mut words = vec![0_u64; bytes.len().div_ceil(8)];
        words.as_mut_bytes()[..bytes.len()].copy_from_slice(&bytes);
        words
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn filter() {
        let all = [
            Keys::new([(2, true)]),
            Keys::new([(4, true), (2, false), (16, false)]),
            Keys::new([(8, false), (2, true), (2, true), (4, false)]),
        ];
        for keys in &all {
            let n = keys.as_slice().len();
            for size in [1, 3, 5, 9, 17, 33, 200, 5000] {
                let wanted = (keys.as_slice().iter())
                    .map(|key| {
                        let mask = u128::MAX >> (128 - 8 * key.bytes);
                        (0..size).map(|_| rand::random::<u128>() & mask).collect()
                    })
                    .collect::<Vec<Vec<_>>>();
                for _ in 0..16 {
                    // any subset of the keys constrained
                    let constrained = rand::random::<u32>() % (1 << n);
                    let sets = (0..n)
                        .map(|i| (constrained >> i & 1 != 0).then(|| wanted[i].clone()))
                        .collect();
                    let filter = KeyFilter::new(keys, sets).unwrap();
                    let rows = rows(keys, &wanted);
                    let words = block(keys, &rows);
                    let columns = &words.as_bytes()[..32 * keys.columns()];
                    let payload = std::array::from_fn(|i| Some(rows[i].0));
                    let mask = (&filter).block(&payload, columns);
                    for (i, (payload, row)) in rows.iter().enumerate() {
                        let expected = (0..n).all(|k| {
                            let key = keys.as_slice()[k];
                            let x = match key.slot {
                                Slot::Payload => payload.get() as u16 as u128,
                                Slot::Column(offset) => {
                                    let mut x = [0; 16];
                                    x[..key.bytes].copy_from_slice(&row[offset..][..key.bytes]);
                                    u128::from_le_bytes(x)
                                }
                            };
                            constrained >> k & 1 == 0 || wanted[k].contains(&x)
                        });
                        assert_eq!(mask >> i & 1 != 0, expected);
                        assert_eq!((&filter).row(*payload, Columns::row(row)), expected);
                        let mut copy = Vec::new();
                        Columns::block(columns, i).extend(keys, &mut copy);
                        assert_eq!(&copy, row);
                    }
                }
            }
        }
        // an empty set: no row
        assert!(KeyFilter::new(&all[0], vec![Some(Vec::new())]).is_none());
    }

    /// The hashes of consecutive keys and of random keys spread evenly: with
    /// 2^20 keys in 2^16 values, every value gets 16 keys on average, and a
    /// perfect hash would give a chi-squared statistic near 65535.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn uniform() {
        fn chi2(hashes: impl Iterator<Item = u64>) -> f64 {
            let mut count = vec![0_u32; 65536];
            let mut n = 0;
            for hash in hashes {
                count[fold(hash, 2) as usize] += 1;
                n += 1;
            }
            let e = n as f64 / 65536.0;
            count.iter().map(|&c| (c as f64 - e).powi(2) / e).sum()
        }
        let n = 1 << 20;
        let ints = chi2((0..n).map(hash_int));
        let uuids = chi2((0..n as u128).map(|x| hash_uuid(&x.to_be_bytes())));
        let random = chi2((0..n).map(|_| hash_uuid(&rand::random())));
        for x in [ints, uuids, random] {
            assert!((63000.0..68000.0).contains(&x), "{x}");
        }
    }
}
