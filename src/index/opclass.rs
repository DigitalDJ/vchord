// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_vector_l2_ops() -> String {
    "vchordg_vector_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_vector_cosine_ops() -> String {
    "vchordg_vector_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_vector_ip_ops() -> String {
    "vchordg_vector_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_halfvec_l2_ops() -> String {
    "vchordg_halfvec_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_halfvec_cosine_ops() -> String {
    "vchordg_halfvec_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_halfvec_ip_ops() -> String {
    "vchordg_halfvec_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_rabitq8_l2_ops() -> String {
    "vchordg_rabitq8_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_rabitq8_cosine_ops() -> String {
    "vchordg_rabitq8_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_rabitq8_ip_ops() -> String {
    "vchordg_rabitq8_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_rabitq4_l2_ops() -> String {
    "vchordg_rabitq4_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_rabitq4_cosine_ops() -> String {
    "vchordg_rabitq4_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_rabitq4_ip_ops() -> String {
    "vchordg_rabitq4_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_l2_ops() -> String {
    "vchordrq_vector_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_ip_ops() -> String {
    "vchordrq_vector_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_cosine_ops() -> String {
    "vchordrq_vector_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_l2_ops() -> String {
    "vchordrq_halfvec_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_ip_ops() -> String {
    "vchordrq_halfvec_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_cosine_ops() -> String {
    "vchordrq_halfvec_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq8_l2_ops() -> String {
    "vchordrq_rabitq8_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq8_ip_ops() -> String {
    "vchordrq_rabitq8_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq8_cosine_ops() -> String {
    "vchordrq_rabitq8_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq4_l2_ops() -> String {
    "vchordrq_rabitq4_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq4_ip_ops() -> String {
    "vchordrq_rabitq4_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq4_cosine_ops() -> String {
    "vchordrq_rabitq4_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_maxsim_ops() -> String {
    "vchordrq_vector_maxsim_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_maxsim_ops() -> String {
    "vchordrq_halfvec_maxsim_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq8_maxsim_ops() -> String {
    "vchordrq_rabitq8_maxsim_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq4_maxsim_ops() -> String {
    "vchordrq_rabitq4_maxsim_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_uuid_ops() -> String {
    "vchordrq_uuid_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_int2_ops() -> String {
    "vchordrq_int2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_int4_ops() -> String {
    "vchordrq_int4_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_int8_ops() -> String {
    "vchordrq_int8_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_uuid_options(relopts: pgrx::Internal) {
    use crate::index::vchordrq::filter_key::{Type, options};
    unsafe { options(relopts, Type::Uuid) }
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_int2_options(relopts: pgrx::Internal) {
    use crate::index::vchordrq::filter_key::{Type, options};
    unsafe { options(relopts, Type::Int2) }
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_int4_options(relopts: pgrx::Internal) {
    use crate::index::vchordrq::filter_key::{Type, options};
    unsafe { options(relopts, Type::Int4) }
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_int8_options(relopts: pgrx::Internal) {
    use crate::index::vchordrq::filter_key::{Type, options};
    unsafe { options(relopts, Type::Int8) }
}

pub struct Sphere<T> {
    pub center: T,
    pub radius: f32,
}
