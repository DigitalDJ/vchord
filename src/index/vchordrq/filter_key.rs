// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

//! The filter key columns of a vchordrq index: its key columns after the
//! vector, `USING vchordrq (embedding halfvec_cosine_ops, "ownerId", g)`.
//! Every row stores each key in the bytes that its operator class's `bytes`
//! parameter asks for (`vchordrq::keys`), 2 by default: `"ownerId" uuid_ops
//! (bytes = 16)` stores the uuid itself, and fewer bytes a hash of it. The
//! planner hands the scan `key = x` and `key = ANY(xs)` as scan keys, which
//! the scan turns into the stored keys to read. The executor rechecks the
//! rows of hashed keys, and of whole keys whose wanted keys include the
//! stored form of a NULL.

use crate::index::storage::PostgresRelation;
use crate::index::vchordrq::opclass::support_name;
use pgrx::pg_sys::{self, BuiltinOid, Datum};
use vchordrq::keys::{Key, Keys, Slot, hash_int, hash_uuid, raw_int};

/// The type of a filter key column. Another fixed-width type is a variant
/// with its operator class, size and stored forms ([`Type::stored`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Uuid,
    Int2,
    Int4,
    Int8,
}

impl Type {
    const ALL: [Self; 4] = [Self::Uuid, Self::Int2, Self::Int4, Self::Int8];

    /// The name that support function 1 of the type's operator class returns.
    fn opclass(self) -> &'static str {
        match self {
            Self::Uuid => "vchordrq_uuid_ops",
            Self::Int2 => "vchordrq_int2_ops",
            Self::Int4 => "vchordrq_int4_ops",
            Self::Int8 => "vchordrq_int8_ops",
        }
    }

    /// The size of a value of the type, in bytes.
    pub fn bytes(self) -> usize {
        match self {
            Self::Uuid => 16,
            Self::Int2 => 2,
            Self::Int4 => 4,
            Self::Int8 => 8,
        }
    }

    fn of_oid(oid: pg_sys::Oid) -> Option<Self> {
        match BuiltinOid::from_u32(oid.to_u32()).ok()? {
            BuiltinOid::UUIDOID => Some(Self::Uuid),
            BuiltinOid::INT2OID => Some(Self::Int2),
            BuiltinOid::INT4OID => Some(Self::Int4),
            BuiltinOid::INT8OID => Some(Self::Int8),
            _ => None,
        }
    }

    /// The stored form of `datum`, a value of `self`, in a column of type
    /// `column` stored as `key`. `None` if no value of the column has it: an
    /// `int8` out of the range of an `int4` column stored whole.
    unsafe fn stored(self, datum: Datum, column: Self, key: &Key) -> Option<u128> {
        match self {
            Self::Uuid => {
                let bytes = unsafe { &*datum.cast_mut_ptr::<[u8; 16]>() };
                Some(key.stored(u128::from_le_bytes(*bytes), hash_uuid(bytes)))
            }
            Self::Int2 | Self::Int4 | Self::Int8 => {
                let x = match self {
                    Self::Int2 => datum.value() as i16 as i64,
                    Self::Int4 => datum.value() as i32 as i64,
                    _ => datum.value() as i64,
                };
                let raw = if key.hashed {
                    0
                } else {
                    raw_int(x, column.bytes())? as u128
                };
                Some(key.stored(raw, hash_int(x)))
            }
        }
    }
}

/// The types of the filter key columns of an index, by their operator
/// classes.
unsafe fn types(index_relation: pg_sys::Relation) -> Vec<Type> {
    let n = unsafe { (*(*index_relation).rd_index).indnkeyatts };
    (2..=n)
        .map(|attno| {
            let name = unsafe { support_name(index_relation, attno) };
            let ty = Type::ALL.into_iter().find(|ty| ty.opclass() == name);
            ty.unwrap_or_else(|| pgrx::error!("unknown operator class of a filter key column"))
        })
        .collect()
}

/// The parsed parameters of the operator class of a filter key column.
#[repr(C)]
struct Options {
    vl_len_: i32,
    bytes: i32,
}

/// Rejects a `bytes` that is not 2, 4, 8 or 16, as the parameters are
/// parsed, before an index is created.
#[pgrx::pg_guard]
unsafe extern "C-unwind" fn validate(
    parsed: *mut std::ffi::c_void,
    _values: *mut pg_sys::relopt_value,
    _n: std::ffi::c_int,
) {
    let bytes = unsafe { (*parsed.cast::<Options>()).bytes };
    if !matches!(bytes, 2 | 4 | 8 | 16) {
        pgrx::error!("invalid value for option \"bytes\": {bytes}, not 2, 4, 8 or 16");
    }
}

/// The parameters of the operator class of a filter key column of type `ty`
/// (its options support function): `bytes`, the bytes that a row stores of
/// the key, 2, 4, 8 or 16 and at most the size of `ty`, 2 by default.
pub unsafe fn options(relopts: pgrx::Internal, ty: Type) {
    unsafe {
        let relopts = relopts
            .unwrap()
            .expect("no options")
            .cast_mut_ptr::<pg_sys::local_relopts>();
        pg_sys::init_local_reloptions(relopts, size_of::<Options>());
        pg_sys::add_local_int_reloption(
            relopts,
            c"bytes".as_ptr(),
            c"bytes that a row stores of the key: the key itself if it has that size, a hash of it if fewer"
                .as_ptr(),
            2,
            2,
            ty.bytes() as _,
            std::mem::offset_of!(Options, bytes) as _,
        );
        pg_sys::register_reloptions_validator(relopts, Some(validate));
    }
}

/// What the relation cache entry of an index keeps of its filter keys
/// (`rd_amcache`): plain data, in the entry's memory context.
#[repr(C)]
#[derive(Clone, Copy)]
struct Cached {
    types: [u8; pg_sys::INDEX_MAX_KEYS as usize],
    descriptors: [u8; pg_sys::INDEX_MAX_KEYS as usize],
    /// The number of filter key columns.
    n: u8,
    /// The number of descriptors: `n`, or 0 for a layout without them.
    m: u8,
}

/// The filter key columns of an index: their types, and how their keys are
/// stored.
#[derive(Debug)]
pub struct FilterKeys {
    pub types: Vec<Type>,
    pub keys: Keys,
}

impl FilterKeys {
    /// The filter keys of an index being built, laid out by the parameters
    /// of their operator classes. `None` without filter key columns.
    pub unsafe fn of_options(index_relation: pg_sys::Relation) -> Option<Self> {
        let types = unsafe { types(index_relation) };
        if types.is_empty() {
            return None;
        }
        let options = unsafe { pg_sys::RelationGetIndexAttOptions(index_relation, false) };
        let keys = Keys::new(types.iter().enumerate().map(|(i, ty)| {
            let parsed = unsafe { (*options.add(i + 1)).cast::<Options>() };
            // checked as the parameters were parsed (`validate`)
            let bytes = match parsed.is_null() {
                true => 2,
                false => unsafe { (*parsed).bytes as usize },
            };
            (bytes, bytes < ty.bytes())
        }));
        Some(Self { types, keys })
    }

    /// The filter keys of a built index, laid out by its meta tuple; looked
    /// up once per relation cache entry. `None` without filter key columns.
    pub unsafe fn of_index(index_relation: pg_sys::Relation) -> Option<Self> {
        unsafe {
            if (*(*index_relation).rd_index).indnkeyatts < 2 {
                return None;
            }
            if (*index_relation).rd_amcache.is_null() {
                let types = types(index_relation);
                let index = PostgresRelation::<vchordrq::Opaque>::new(index_relation);
                let keys = Keys::of(&index);
                if keys.as_slice().len() != types.len() {
                    pgrx::error!("the filter keys of the index do not match its columns");
                }
                let descriptors = keys.descriptors();
                let mut cached = Cached {
                    types: [0; _],
                    descriptors: [0; _],
                    n: types.len() as u8,
                    m: descriptors.len() as u8,
                };
                for (i, ty) in types.iter().enumerate() {
                    cached.types[i] = Type::ALL.iter().position(|x| x == ty).unwrap() as u8;
                }
                cached.descriptors[..descriptors.len()].copy_from_slice(&descriptors);
                let memory =
                    pg_sys::MemoryContextAlloc((*index_relation).rd_indexcxt, size_of::<Cached>());
                memory.cast::<Cached>().write(cached);
                (*index_relation).rd_amcache = memory;
            }
            let cached = (*index_relation).rd_amcache.cast::<Cached>().read();
            let types = cached.types[..cached.n as usize].iter();
            let types = types.map(|&i| Type::ALL[i as usize]).collect();
            let descriptors = &cached.descriptors[..cached.m as usize];
            let keys = Keys::from_descriptors(descriptors).expect("checked by `Keys::of`");
            Some(Self { types, keys })
        }
    }

    /// The stored keys of an index row, given the values of its key columns:
    /// the key in the payload, if one is, and the row's columns
    /// (`vchordrq::keys::Columns`). A NULL is stored as zeros.
    pub unsafe fn row(&self, values: *const Datum, is_null: *const bool) -> (Option<u16>, Vec<u8>) {
        let mut payload = None;
        let mut columns = Vec::with_capacity(self.keys.columns());
        for (i, (key, &ty)) in std::iter::zip(self.keys.as_slice(), &self.types).enumerate() {
            let stored = unsafe {
                if is_null.add(i + 1).read() {
                    0
                } else {
                    let stored = ty.stored(values.add(i + 1).read(), ty, key);
                    stored.expect("a key out of the range of its own type")
                }
            };
            match key.slot {
                Slot::Payload => payload = Some(stored as u16),
                Slot::Column(_) => columns.extend_from_slice(&stored.to_le_bytes()[..key.bytes]),
            }
        }
        (payload, columns)
    }

    /// The stored keys that a scan key on filter key column `i` accepts:
    /// one for `= x`, one per element for `= ANY(xs)`, none for a NULL or for
    /// a value that no row can have; sorted and deduplicated.
    pub unsafe fn of_scan_key(&self, i: usize, scan_key: &pg_sys::ScanKeyData) -> Vec<u128> {
        if scan_key.sk_flags & pg_sys::SK_ISNULL as i32 != 0 {
            return Vec::new();
        }
        let (column, key) = (self.types[i], &self.keys.as_slice()[i]);
        // the right type of the operator, which differs for `int4 = int8`
        let ty = Type::of_oid(scan_key.sk_subtype).unwrap_or(column);
        if scan_key.sk_flags & pg_sys::SK_SEARCHARRAY as i32 == 0 {
            let stored = unsafe { ty.stored(scan_key.sk_argument, column, key) };
            return stored.into_iter().collect();
        }
        unsafe {
            let datum = scan_key.sk_argument.cast_mut_ptr::<pg_sys::varlena>();
            let array = pg_sys::pg_detoast_datum(datum).cast::<pg_sys::ArrayType>();
            let elemtype = (*array).elemtype;
            let (mut typlen, mut typbyval, mut typalign) = (0, false, 0);
            pg_sys::get_typlenbyvalalign(elemtype, &mut typlen, &mut typbyval, &mut typalign);
            let mut elements = std::ptr::null_mut();
            let mut nulls = std::ptr::null_mut();
            let mut n = 0;
            pg_sys::deconstruct_array(
                array,
                elemtype,
                typlen as _,
                typbyval,
                typalign,
                &mut elements,
                &mut nulls,
                &mut n,
            );
            let mut stored = (0..n as usize)
                .filter(|&i| !nulls.add(i).read())
                .filter_map(|i| ty.stored(elements.add(i).read(), column, key))
                .collect::<Vec<_>>();
            stored.sort_unstable();
            stored.dedup();
            pg_sys::pfree(elements.cast());
            pg_sys::pfree(nulls.cast());
            if array.cast() != datum {
                pg_sys::pfree(array.cast());
            }
            stored
        }
    }
}
