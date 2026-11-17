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
    Intervals, Key, Keys, Summary, Width, fold, hash_int, hash_uuid, raw_int, raw_uuid,
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

    /// The stored form of `datum` (of type `self`) under `key`; `None` if no row can hold it.
    unsafe fn stored(self, datum: Datum, key: &Key) -> Option<u128> {
        match self {
            Self::Uuid => {
                let bytes = unsafe { &*datum.cast_mut_ptr::<[u8; 16]>() };
                Some(if key.hashed {
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
                Some(if key.hashed {
                    fold(hash_int(x), key.width) as u128
                } else {
                    raw_int(x, key.width)? as u128
                })
            }
        }
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
        pgrx::error!("invalid value for option \"bytes\": {bytes}, not 2, 4, 8 or 16");
    }
}

/// The options support function of a filter key column of type `ty`.
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
            c"bytes that a row stores of the key: the key itself by default, a hash of it if fewer"
                .as_ptr(),
            ty.width().bytes() as _,
            2,
            ty.width().bytes() as _,
            std::mem::offset_of!(Options, bytes) as _,
        );
        pg_sys::register_reloptions_validator(relopts, Some(validate));
    }
}

/// The filter keys of an index being built; `zone_maps`: how many leading keys get block bounds.
pub unsafe fn of_options(index_relation: pg_sys::Relation, zone_maps: usize) -> Keys {
    let options = unsafe { pg_sys::RelationGetIndexAttOptions(index_relation, false) };
    Keys::new(unsafe { types(index_relation) }.enumerate().map(|(i, ty)| {
        let bytes = unsafe { (*(*options.add(i + 1)).cast::<Options>()).bytes };
        let width = Width::of_bytes(bytes as usize).unwrap();
        (
            width,
            width != ty.width(),
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

    /// The row's columns; a NULL is stored as zeros.
    pub unsafe fn row(&self, values: *const Datum, is_null: *const bool) -> Vec<u8> {
        let mut columns = Vec::with_capacity(self.keys.columns());
        for (i, key) in self.keys.as_slice().iter().enumerate() {
            let stored = unsafe {
                if is_null.add(i + 1).read() {
                    0
                } else {
                    let stored = self.ty(i).stored(values.add(i + 1).read(), key);
                    stored.expect("a key out of the range of its own type")
                }
            };
            columns.extend_from_slice(&stored.to_le_bytes()[..key.width.bytes()]);
        }
        columns
    }

    /// The stored forms that a scan key on filter key column `i` accepts.
    pub unsafe fn of_scan_key(&self, i: usize, scan_key: &pg_sys::ScanKeyData) -> Intervals {
        if scan_key.sk_flags & pg_sys::SK_ISNULL as i32 != 0 {
            return Intervals::points(Vec::new());
        }
        let (column, key) = (self.ty(i), &self.keys.as_slice()[i]);
        // the right type of the operator, which differs for `int4 = int8`
        let ty = Type::of_oid(scan_key.sk_subtype).unwrap_or(column);
        if scan_key.sk_flags & pg_sys::SK_SEARCHARRAY as i32 == 0 {
            let stored = unsafe { ty.stored(scan_key.sk_argument, key) };
            return Intervals::points(stored.into_iter().collect());
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
            let stored = (0..n as usize)
                .filter(|&i| !nulls.add(i).read())
                .filter_map(|i| ty.stored(elements.add(i).read(), key))
                .collect::<Vec<_>>();
            pg_sys::pfree(elements.cast());
            pg_sys::pfree(nulls.cast());
            if array.cast() != datum {
                pg_sys::pfree(array.cast());
            }
            Intervals::points(stored)
        }
    }
}
