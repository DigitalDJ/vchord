// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

//! The filter key columns of a vchordrq index: their stored keys and scan keys.

use crate::index::storage::PostgresRelation;
use pgrx::pg_sys::{self, BuiltinOid, Datum};
use vchordrq::keys::{
    Condition, Intervals, Key, Keys, Summary, Width, fold, hash_int, hash_uuid, raw_int, raw_uuid,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Uuid,
    Int2,
    Int4,
    Int8,
}

impl Type {
    pub fn width(self) -> Width {
        match self {
            Self::Uuid => Width::W16,
            Self::Int2 => Width::W2,
            Self::Int4 => Width::W4,
            Self::Int8 => Width::W8,
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

    /// Where `datum`, of type `self`, lies among the stored forms of `key`.
    unsafe fn bound(self, datum: Datum, key: &Key) -> Bound {
        match self {
            Self::Uuid => {
                let bytes = unsafe { &*datum.cast_mut_ptr::<[u8; 16]>() };
                Bound::At(if key.hashed {
                    fold(hash_uuid(bytes), key.width) as u128
                } else {
                    raw_uuid(bytes)
                })
            }
            Self::Int2 | Self::Int4 | Self::Int8 => {
                let x = match self {
                    Self::Int2 => datum.value() as i16 as i64,
                    Self::Int4 => datum.value() as i32 as i64,
                    _ => datum.value() as i64,
                };
                if key.hashed {
                    return Bound::At(fold(hash_int(x), key.width) as u128);
                }
                match raw_int(x, key.width) {
                    Some(stored) => Bound::At(stored as u128),
                    None if x < 0 => Bound::Below,
                    None => Bound::Above,
                }
            }
        }
    }
}

/// Where a constant lies among the stored forms of a key, which order like the key.
#[derive(Debug, Clone, Copy)]
enum Bound {
    Below,
    At(u128),
    Above,
}

/// The stored forms, up to `max`, that a scan key of `strategy` accepts against `bound`.
fn interval(strategy: u16, bound: Bound, max: u128) -> Option<[u128; 2]> {
    use Bound::*;
    const LESS: u16 = pg_sys::BTLessStrategyNumber as u16;
    const LESS_EQUAL: u16 = pg_sys::BTLessEqualStrategyNumber as u16;
    const EQUAL: u16 = pg_sys::BTEqualStrategyNumber as u16;
    const GREATER_EQUAL: u16 = pg_sys::BTGreaterEqualStrategyNumber as u16;
    const GREATER: u16 = pg_sys::BTGreaterStrategyNumber as u16;
    match (strategy, bound) {
        (LESS, At(x)) => x.checked_sub(1).map(|x| [0, x]),
        (LESS_EQUAL, At(x)) => Some([0, x]),
        (EQUAL, At(x)) => Some([x, x]),
        (GREATER_EQUAL, At(x)) => Some([x, max]),
        (GREATER, At(x)) => (x < max).then(|| [x + 1, max]),
        (LESS | LESS_EQUAL, Above) | (GREATER_EQUAL | GREATER, Below) => Some([0, max]),
        _ => None,
    }
}

unsafe fn types(index_relation: pg_sys::Relation) -> impl Iterator<Item = Type> {
    let n = unsafe { (*(*index_relation).rd_index).indnkeyatts } as usize;
    (1..n).map(move |i| {
        let oid = unsafe { *(*index_relation).rd_opcintype.add(i) };
        Type::of_oid(oid)
            .unwrap_or_else(|| pgrx::error!("unknown operator class of a filter key column"))
    })
}

#[repr(C)]
struct Options {
    vl_len_: i32,
    bytes: i32,
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn validate(
    parsed: *mut std::ffi::c_void,
    _values: *mut pg_sys::relopt_value,
    _n: std::ffi::c_int,
) {
    let bytes = unsafe { (*parsed.cast::<Options>()).bytes };
    if Width::of_bytes(bytes as usize).is_none() {
        pgrx::error!("invalid value for option \"bytes\": {bytes}, not 2, 4 or 8");
    }
}

/// The options support function of a `*_hash_ops` filter key column: `bytes`, at most half the key.
pub unsafe fn options(relopts: pgrx::Internal, ty: Type) {
    let most = ty.width().bytes() / 2;
    unsafe {
        let relopts = relopts
            .unwrap()
            .expect("no options")
            .cast_mut_ptr::<pg_sys::local_relopts>();
        pg_sys::init_local_reloptions(relopts, size_of::<Options>());
        pg_sys::add_local_int_reloption(
            relopts,
            c"bytes".as_ptr(),
            c"bytes of the hash that a row stores of the key".as_ptr(),
            most.min(4) as _,
            2,
            most as _,
            std::mem::offset_of!(Options, bytes) as _,
        );
        pg_sys::register_reloptions_validator(relopts, Some(validate));
    }
}

/// The filter keys of an index being built; `zone_maps`: how many leading keys get block bounds.
pub unsafe fn of_options(index_relation: pg_sys::Relation, zone_maps: usize) -> Keys {
    let options = unsafe { pg_sys::RelationGetIndexAttOptions(index_relation, false) };
    Keys::new(unsafe { types(index_relation) }.enumerate().map(|(i, ty)| {
        let hash = unsafe { options.add(i + 1).read().cast::<Options>().as_ref() };
        let width = hash.map_or(ty.width(), |x| Width::of_bytes(x.bytes as usize).unwrap());
        (
            width,
            hash.is_some(),
            (i < zone_maps).then_some(Summary::Bounds),
        )
    }))
}

pub struct FilterKeys<'a> {
    pub keys: &'a Keys,
    index_relation: pg_sys::Relation,
}

impl<'a> FilterKeys<'a> {
    pub fn new(index_relation: pg_sys::Relation, keys: &'a Keys) -> Self {
        Self {
            keys,
            index_relation,
        }
    }

    /// The keys are decoded once per relation cache entry, into `rd_amcache`.
    pub unsafe fn of_index(index_relation: pg_sys::Relation) -> Self {
        unsafe {
            if (*index_relation).rd_amcache.is_null() {
                let keys = if (*(*index_relation).rd_index).indnkeyatts < 2 {
                    Keys::default()
                } else {
                    let index = PostgresRelation::<vchordrq::Opaque>::new(index_relation);
                    let keys = Keys::of(&index);
                    if keys.as_slice().len() != types(index_relation).count() {
                        pgrx::error!("the filter keys of the index do not match its columns");
                    }
                    keys
                };
                let memory =
                    pg_sys::MemoryContextAlloc((*index_relation).rd_indexcxt, size_of::<Keys>());
                memory.cast::<Keys>().write(keys);
                (*index_relation).rd_amcache = memory;
            }
            Self {
                keys: &*(*index_relation).rd_amcache.cast::<Keys>(),
                index_relation,
            }
        }
    }

    fn ty(&self, i: usize) -> Type {
        let oid = unsafe { *(*self.index_relation).rd_opcintype.add(i + 1) };
        Type::of_oid(oid).unwrap()
    }

    /// The row's columns: a NULL is stored as zeros and marked in the NULL mask.
    pub unsafe fn row(&self, values: *const Datum, is_null: *const bool) -> Vec<u8> {
        let mut columns = Vec::with_capacity(self.keys.columns());
        let mut nulls = 0_u32;
        for (i, key) in self.keys.as_slice().iter().enumerate() {
            let stored = unsafe {
                if is_null.add(i + 1).read() {
                    nulls |= 1 << i;
                    0
                } else {
                    match self.ty(i).bound(values.add(i + 1).read(), key) {
                        Bound::At(stored) => stored,
                        _ => unreachable!("a key out of the range of its own type"),
                    }
                }
            };
            columns.extend_from_slice(&stored.to_le_bytes()[..key.width.bytes()]);
        }
        if !self.keys.is_empty() {
            columns.extend_from_slice(&nulls.to_le_bytes());
        }
        columns
    }

    /// What a scan key on filter key column `i` accepts; with `ANY(array)`, what an element does.
    pub unsafe fn of_scan_key(&self, i: usize, scan_key: &pg_sys::ScanKeyData) -> Condition {
        let (column, key) = (self.ty(i), &self.keys.as_slice()[i]);
        let flag = |x: u32| scan_key.sk_flags & x as i32 != 0;
        let null = flag(pg_sys::SK_SEARCHNULL);
        if flag(pg_sys::SK_ISNULL) {
            let values = flag(pg_sys::SK_SEARCHNOTNULL).then_some([0, key.max()]);
            let values = Intervals::new(values.into_iter().collect());
            return Condition { values, null };
        }
        let values = unsafe { self.values(column, key, scan_key) };
        Condition { values, null }
    }

    unsafe fn values(&self, column: Type, key: &Key, scan_key: &pg_sys::ScanKeyData) -> Intervals {
        // the right type of the operator, which differs for `int4 = int8`
        let ty = Type::of_oid(scan_key.sk_subtype).unwrap_or(column);
        let accepts = |datum| {
            interval(
                scan_key.sk_strategy,
                unsafe { ty.bound(datum, key) },
                key.max(),
            )
        };
        if scan_key.sk_flags & pg_sys::SK_SEARCHARRAY as i32 == 0 {
            return Intervals::new(accepts(scan_key.sk_argument).into_iter().collect());
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
            let accepted = (0..n as usize)
                .filter(|&i| !nulls.add(i).read())
                .filter_map(|i| accepts(elements.add(i).read()))
                .collect::<Vec<_>>();
            pg_sys::pfree(elements.cast());
            pg_sys::pfree(nulls.cast());
            if array.cast() != datum {
                pg_sys::pfree(array.cast());
            }
            Intervals::new(accepted)
        }
    }
}
