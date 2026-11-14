// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use distance::Distance;
use index_accessor::{
    Accessor1, Accessor2, ApproximateL2S, ByteDistanceAccessor, DefaultWithDimension,
    DistanceAccessor, Dot, HalfbyteDistanceAccessor, L2S, RAccess,
};
use rabitq::bit::binary::BinaryLut;
use rabitq::bit::block::{BlockLut, STEP};
use rabitq::bit::{CodeMetadata, Residual};
use simd::{Floating, f16};
use std::fmt::Debug;
use std::marker::PhantomData;
use vector::rabitq4::{Rabitq4Borrowed, Rabitq4Owned};
use vector::rabitq8::{Rabitq8Borrowed, Rabitq8Owned};
use vector::vect::{VectBorrowed, VectOwned};
use vector::{VectorBorrowed, VectorOwned};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[derive(Debug)]
pub struct BlockAccessor<F>([u32; 32], F);

impl<F: Call<u32, CodeMetadata, f32>>
    Accessor2<[u8; 16], [u8; 16], (&[[f32; 32]; 4], &[f32; 32]), ()> for BlockAccessor<F>
{
    type Output = [F::Output; 32];

    #[inline(always)]
    fn push(&mut self, input: &[[u8; 16]], target: &[[u8; 16]]) {
        use std::iter::zip;

        for (input, target) in zip(input.chunks(STEP), target.chunks(STEP)) {
            let delta = simd::fast_scan::scan(input, target);
            simd::fast_scan::accu(&mut self.0, &delta);
        }
    }

    #[inline(always)]
    fn finish(mut self, (metadata, delta): (&[[f32; 32]; 4], &[f32; 32]), (): ()) -> Self::Output {
        std::array::from_fn(|i| {
            (self.1).call(
                self.0[i],
                CodeMetadata {
                    dis_u_2: metadata[0][i],
                    factor_cnt: metadata[1][i],
                    factor_ip: metadata[2][i],
                    factor_err: metadata[3][i],
                },
                delta[i],
            )
        })
    }
}

#[derive(Debug, Clone)]
pub struct CloneAccessor<V: Vector>(Vec<V::Element>);

impl<V: Vector> Default for CloneAccessor<V> {
    #[inline(always)]
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<V: Vector> Accessor1<V::Element, (V::Metadata, u32)> for CloneAccessor<V> {
    type Output = V;

    #[inline(always)]
    fn push(&mut self, input: &[V::Element]) {
        self.0.extend(input);
    }

    #[inline(always)]
    fn finish(self, (metadata, dim): (V::Metadata, u32)) -> Self::Output {
        V::pack(dim, self.0, metadata)
    }
}

pub trait Vector: VectorOwned {
    type Element: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;

    type Metadata: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;

    fn split(vector: Self::Borrowed<'_>) -> (Vec<&[Self::Element]>, Self::Metadata);

    fn count(dim: u32) -> u32;

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata);

    fn pack(dim: u32, elements: Vec<Self::Element>, metadata: Self::Metadata) -> Self;

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut;

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut);

    fn code(vector: Self::Borrowed<'_>) -> rabitq::bit::Code;

    fn squared_norm(vector: Self::Borrowed<'_>) -> f32;
}

impl Vector for VectOwned<f32> {
    type Metadata = ();

    type Element = f32;

    fn split(vector: Self::Borrowed<'_>) -> (Vec<&[f32]>, ()) {
        let vector = vector.slice();
        (
            match vector.len() {
                0 => unreachable!(),
                1..=960 => vec![vector],
                961..=1280 => vec![&vector[..640], &vector[640..]],
                1281.. => vector.chunks(1920).collect(),
            },
            (),
        )
    }

    fn count(dim: u32) -> u32 {
        match dim {
            0 => unreachable!(),
            1..=960 => 1,
            961..=1280 => 2,
            1281.. => dim.div_ceil(1920),
        }
    }

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn pack(_: u32, elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
        VectOwned::new(elements)
    }

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut {
        rabitq::bit::block::preprocess(vector.slice())
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut) {
        rabitq::bit::preprocess(vector.slice())
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::bit::Code {
        rabitq::bit::code(vector.slice())
    }

    fn squared_norm(vector: Self::Borrowed<'_>) -> f32 {
        f32::reduce_sum_of_x2(vector.slice())
    }
}

impl Vector for VectOwned<f16> {
    type Metadata = ();

    type Element = f16;

    fn split(vector: Self::Borrowed<'_>) -> (Vec<&[f16]>, ()) {
        let vector = vector.slice();
        (
            match vector.len() {
                0 => unreachable!(),
                1..=1920 => vec![vector],
                1921..=2560 => vec![&vector[..1280], &vector[1280..]],
                2561.. => vector.chunks(3840).collect(),
            },
            (),
        )
    }

    fn count(dim: u32) -> u32 {
        match dim {
            0 => unreachable!(),
            1..=1920 => 1,
            1921..=2560 => 2,
            2561.. => dim.div_ceil(3840),
        }
    }

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn pack(_: u32, elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
        VectOwned::new(elements)
    }

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut {
        rabitq::bit::block::preprocess(&f16::vector_to_f32(vector.slice()))
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut) {
        rabitq::bit::preprocess(&f16::vector_to_f32(vector.slice()))
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::bit::Code {
        rabitq::bit::code(&f16::vector_to_f32(vector.slice()))
    }

    fn squared_norm(vector: Self::Borrowed<'_>) -> f32 {
        f16::reduce_sum_of_x2(vector.slice())
    }
}

impl Vector for Rabitq8Owned {
    type Metadata = [f32; 4];

    type Element = u8;

    fn split(vector: Self::Borrowed<'_>) -> (Vec<&[u8]>, [f32; 4]) {
        (
            match vector.packed_code().len() {
                0 => unreachable!(),
                1..=3840 => vec![vector.packed_code()],
                3841..=5120 => vec![&vector.packed_code()[..1280], &vector.packed_code()[1280..]],
                5121.. => vector.packed_code().chunks(7680).collect(),
            },
            [
                vector.sum_of_x2(),
                vector.norm_of_lattice(),
                vector.sum_of_code(),
                vector.sum_of_abs_x(),
            ],
        )
    }

    fn count(dim: u32) -> u32 {
        match dim {
            0 => unreachable!(),
            1..=3840 => 1,
            3841..=5120 => 2,
            5121.. => dim.div_ceil(7680),
        }
    }

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (
            vector.packed_code(),
            [
                vector.sum_of_x2(),
                vector.norm_of_lattice(),
                vector.sum_of_code(),
                vector.sum_of_abs_x(),
            ],
        )
    }

    fn pack(dim: u32, elements: Vec<Self::Element>, [_0, _1, _2, _3]: Self::Metadata) -> Self {
        Rabitq8Owned::new(dim, _0, _1, _2, _3, elements)
    }

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut {
        let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
        let mut result = Vec::with_capacity(vector.dim() as _);
        for c in vector.unpacked_code() {
            let base = -0.5 * ((1 << 8) - 1) as f32;
            result.push((base + c as f32) * scale);
        }
        rabitq::bit::block::preprocess(&result)
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut) {
        let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
        let mut result = Vec::with_capacity(vector.dim() as _);
        for c in vector.unpacked_code() {
            let base = -0.5 * ((1 << 8) - 1) as f32;
            result.push((base + c as f32) * scale);
        }
        rabitq::bit::preprocess(&result)
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::bit::Code {
        let n = vector.dim();
        let sum_of_abs_x = vector.sum_of_abs_x();
        let sum_of_x_2 = vector.sum_of_x2();
        (
            // Shares the guards of `rabitq::bit::metadata`: a zero vector
            // (`0 / 0`) and a 1-dimensional one (`0 / 0` in the error factor)
            // would otherwise store NaN factors, which sort every candidate
            // with them last.
            rabitq::bit::metadata(n as usize, sum_of_abs_x, sum_of_x_2, {
                let cnt_pos = vector.unpacked_code().filter(|&x| x >= 128).count();
                let cnt_neg = vector.unpacked_code().filter(|&x| x <= 127).count();
                cnt_pos as f32 - cnt_neg as f32
            }),
            {
                let vector = vector.unpacked_code();
                let mut signs = Vec::new();
                for x in vector {
                    signs.push(x >= 128);
                }
                signs
            },
        )
    }

    fn squared_norm(vector: Self::Borrowed<'_>) -> f32 {
        vector.sum_of_x2()
    }
}

impl Vector for Rabitq4Owned {
    type Metadata = [f32; 4];

    type Element = u8;

    fn split(vector: Self::Borrowed<'_>) -> (Vec<&[u8]>, [f32; 4]) {
        (
            match vector.packed_code().len() {
                0 => unreachable!(),
                1..=3840 => vec![vector.packed_code()],
                3841..=5120 => vec![&vector.packed_code()[..1280], &vector.packed_code()[1280..]],
                5121.. => vector.packed_code().chunks(7680).collect(),
            },
            [
                vector.sum_of_x2(),
                vector.norm_of_lattice(),
                vector.sum_of_code(),
                vector.sum_of_abs_x(),
            ],
        )
    }

    fn count(dim: u32) -> u32 {
        match dim {
            0 => unreachable!(),
            1..=7680 => 1,
            7681..=10240 => 2,
            10241.. => dim.div_ceil(15360),
        }
    }

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (
            vector.packed_code(),
            [
                vector.sum_of_x2(),
                vector.norm_of_lattice(),
                vector.sum_of_code(),
                vector.sum_of_abs_x(),
            ],
        )
    }

    fn pack(dim: u32, elements: Vec<Self::Element>, [_0, _1, _2, _3]: Self::Metadata) -> Self {
        Rabitq4Owned::new(dim, _0, _1, _2, _3, elements)
    }

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut {
        let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
        let mut result = Vec::with_capacity(vector.dim() as _);
        for c in vector.unpacked_code() {
            let base = -0.5 * ((1 << 4) - 1) as f32;
            result.push((base + c as f32) * scale);
        }
        rabitq::bit::block::preprocess(&result)
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut) {
        let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
        let mut result = Vec::with_capacity(vector.dim() as _);
        for c in vector.unpacked_code() {
            let base = -0.5 * ((1 << 4) - 1) as f32;
            result.push((base + c as f32) * scale);
        }
        rabitq::bit::preprocess(&result)
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::bit::Code {
        let n = vector.dim();
        let sum_of_abs_x = vector.sum_of_abs_x();
        let sum_of_x_2 = vector.sum_of_x2();
        (
            // Shares the guards of `rabitq::bit::metadata`: a zero vector
            // (`0 / 0`) and a 1-dimensional one (`0 / 0` in the error factor)
            // would otherwise store NaN factors, which sort every candidate
            // with them last.
            rabitq::bit::metadata(n as usize, sum_of_abs_x, sum_of_x_2, {
                let cnt_pos = vector.unpacked_code().filter(|&x| x >= 8).count();
                let cnt_neg = vector.unpacked_code().filter(|&x| x <= 7).count();
                cnt_pos as f32 - cnt_neg as f32
            }),
            {
                let vector = vector.unpacked_code();
                let mut signs = Vec::new();
                for x in vector {
                    signs.push(x >= 8);
                }
                signs
            },
        )
    }

    fn squared_norm(vector: Self::Borrowed<'_>) -> f32 {
        vector.sum_of_x2()
    }
}

pub trait Operator: 'static + Debug + Copy {
    type Vector: Vector;

    type DistanceAccessor: DefaultWithDimension
        + Accessor2<
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Metadata,
            <Self::Vector as Vector>::Metadata,
            Output = Distance,
        >;

    /// Query-centroid distances used only to choose which centroids to probe
    /// (and which list an inserted vector joins). It may be approximate.
    ///
    /// It must never become `dis_f` for `block_access` / `binary_access` at a
    /// level whose output reaches the reranker: in residual mode `dis_f` is
    /// added to every candidate's lower bound with no error budget (a vector
    /// equal to its centroid has `err = 0`), and the reranker emits a result
    /// once no unvisited lower bound is smaller.
    type RoutingAccessor: DefaultWithDimension
        + Accessor2<
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Metadata,
            <Self::Vector as Vector>::Metadata,
            Output = Distance,
        >;

    /// Whether `RoutingAccessor` returns exactly what `DistanceAccessor`
    /// returns on this CPU. If not, residual search recomputes the selected
    /// centroids' distances with `DistanceAccessor` before using them as
    /// `dis_f`.
    fn routing_is_exact() -> bool;

    /// What `block_access` and `binary_access` need of a residual list, from
    /// its centroid's `dis_f` and `norm`. With `centering`, the query is
    /// centered on its projection on the centroid, `lambda c`, instead of the
    /// centroid `c`; the L2 formula cannot use it, as it would need the exact
    /// `<r, c>`, which is not stored.
    fn residual(lut: &BlockLut, dis_f: f32, norm: f32, centering: bool) -> Residual;

    fn block_access(
        lut: &BlockLut,
        residual: Option<Residual>,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>;

    fn binary_access(
        lut: &BinaryLut,
        residual: Option<Residual>,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32);

    fn build(
        vector: <Self::Vector as VectorOwned>::Borrowed<'_>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32);
}

#[derive(Debug)]
pub struct Op<V, D>(PhantomData<fn(V) -> V>, PhantomData<fn(D) -> D>);

impl<V, D> Clone for Op<V, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V, D> Copy for Op<V, D> {}

impl Operator for Op<VectOwned<f32>, L2S> {
    type Vector = VectOwned<f32>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f32>, L2S>;

    type RoutingAccessor = Self::DistanceAccessor;

    fn routing_is_exact() -> bool {
        true
    }

    fn residual(_lut: &BlockLut, dis_f: f32, _norm: f32, _centering: bool) -> Residual {
        Residual::l2(dis_f)
    }

    fn block_access(
        lut: &BlockLut,
        residual: Option<Residual>,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |value, code, delta| match residual {
                None => rabitq::bit::block::half_process_l2s(value, code, lut.0),
                Some(residual) => rabitq::bit::block::half_process_l2s_residual(
                    value, code, lut.0, residual, delta,
                ),
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        residual: Option<Residual>,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        move |metadata: [f32; 4], elements: &[u64], delta: f32| {
            let value = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            match residual {
                None => rabitq::bit::binary::half_process_l2s(value, code, lut.0),
                Some(residual) => rabitq::bit::binary::half_process_l2s_residual(
                    value, code, lut.0, residual, delta,
                ),
            }
        }
    }

    fn build(
        vector: VectBorrowed<'_, f32>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if let Some(centroid) = centroid {
            let residual = VectOwned::new(f32::vector_sub(vector.slice(), centroid.slice()));
            let code = Self::Vector::code(residual.as_borrowed());
            let delta = if code.0.dis_u_2 == 0.0 {
                // The vector equals its centroid: the residual is zero, so it has
                // no direction and contributes nothing (the formula below is 0/0).
                0.0
            } else {
                use std::iter::zip;
                let dim = vector.dim();
                let t = zip(&code.1, centroid.slice())
                    .map(|(&sign, &num)| std::hint::select_unpredictable(sign, num, -num))
                    .sum::<f32>()
                    / (dim as f32).sqrt();
                let sum_of_x_2 = code.0.dis_u_2;
                let sum_of_abs_x = sum_of_x_2 / code.0.factor_ip;
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (dim as f32).sqrt();
                2.0 * dis_u * t / x_0
            };
            (code, delta)
        } else {
            let code = Self::Vector::code(vector);
            let delta = 0.0;
            (code, delta)
        }
    }
}

impl Operator for Op<VectOwned<f32>, Dot> {
    type Vector = VectOwned<f32>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f32>, Dot>;

    type RoutingAccessor = Self::DistanceAccessor;

    fn routing_is_exact() -> bool {
        true
    }

    fn residual(lut: &BlockLut, dis_f: f32, norm: f32, centering: bool) -> Residual {
        let lambda = if centering {
            rabitq::bit::projection(dis_f, norm)
        } else {
            1.0
        };
        Residual::dot(lut, dis_f, norm, lambda)
    }

    fn block_access(
        lut: &BlockLut,
        residual: Option<Residual>,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |sum, code, delta| match residual {
                None => rabitq::bit::block::half_process_dot(sum, code, lut.0),
                Some(residual) => {
                    rabitq::bit::block::half_process_dot_residual(sum, code, lut.0, residual, delta)
                }
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        residual: Option<Residual>,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        move |metadata: [f32; 4], elements: &[u64], delta: f32| {
            let sum = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            match residual {
                None => rabitq::bit::binary::half_process_dot(sum, code, lut.0),
                Some(residual) => rabitq::bit::binary::half_process_dot_residual(
                    sum, code, lut.0, residual, delta,
                ),
            }
        }
    }

    fn build(
        vector: VectBorrowed<'_, f32>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if let Some(centroid) = centroid {
            let residual = VectOwned::new(f32::vector_sub(vector.slice(), centroid.slice()));
            let code = Self::Vector::code(residual.as_borrowed());
            let delta = if code.0.dis_u_2 == 0.0 {
                // The vector equals its centroid: the residual is zero, so it has
                // no direction and contributes nothing (the formula below is 0/0).
                0.0
            } else {
                use std::iter::zip;
                let dim = vector.dim();
                let t = zip(&code.1, centroid.slice())
                    .map(|(&sign, &num)| std::hint::select_unpredictable(sign, num, -num))
                    .sum::<f32>()
                    / (dim as f32).sqrt();
                let sum_of_x_2 = code.0.dis_u_2;
                let sum_of_abs_x = sum_of_x_2 / code.0.factor_ip;
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (dim as f32).sqrt();
                dis_u * t / x_0 - f32::reduce_sum_of_xy(residual.slice(), centroid.slice())
            };
            (code, delta)
        } else {
            let code = Self::Vector::code(vector);
            let delta = 0.0;
            (code, delta)
        }
    }
}

impl Operator for Op<VectOwned<f16>, L2S> {
    type Vector = VectOwned<f16>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f16>, L2S>;

    type RoutingAccessor = DistanceAccessor<VectOwned<f16>, ApproximateL2S>;

    fn routing_is_exact() -> bool {
        f16::reduce_sum_of_d2_approx_is_exact()
    }

    fn residual(_lut: &BlockLut, dis_f: f32, _norm: f32, _centering: bool) -> Residual {
        Residual::l2(dis_f)
    }

    fn block_access(
        lut: &BlockLut,
        residual: Option<Residual>,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |value, code, delta| match residual {
                None => rabitq::bit::block::half_process_l2s(value, code, lut.0),
                Some(residual) => rabitq::bit::block::half_process_l2s_residual(
                    value, code, lut.0, residual, delta,
                ),
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        residual: Option<Residual>,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        move |metadata: [f32; 4], elements: &[u64], delta: f32| {
            let value = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            match residual {
                None => rabitq::bit::binary::half_process_l2s(value, code, lut.0),
                Some(residual) => rabitq::bit::binary::half_process_l2s_residual(
                    value, code, lut.0, residual, delta,
                ),
            }
        }
    }

    fn build(
        vector: VectBorrowed<'_, f16>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if let Some(centroid) = centroid {
            let residual = VectOwned::new(f16::vector_sub(vector.slice(), centroid.slice()));
            let code = Self::Vector::code(residual.as_borrowed());
            let delta = if code.0.dis_u_2 == 0.0 {
                // The vector equals its centroid: the residual is zero, so it has
                // no direction and contributes nothing (the formula below is 0/0).
                0.0
            } else {
                use std::iter::zip;
                let dim = vector.dim();
                let t = zip(&code.1, centroid.slice())
                    .map(|(&sign, &num)| std::hint::select_unpredictable(sign, num, -num))
                    .map(simd::F16::_to_f32)
                    .sum::<f32>()
                    / (dim as f32).sqrt();
                let sum_of_x_2 = code.0.dis_u_2;
                let sum_of_abs_x = sum_of_x_2 / code.0.factor_ip;
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (dim as f32).sqrt();
                2.0 * dis_u * t / x_0
            };
            (code, delta)
        } else {
            let code = Self::Vector::code(vector);
            let delta = 0.0;
            (code, delta)
        }
    }
}

impl Operator for Op<VectOwned<f16>, Dot> {
    type Vector = VectOwned<f16>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f16>, Dot>;

    type RoutingAccessor = Self::DistanceAccessor;

    fn routing_is_exact() -> bool {
        true
    }

    fn residual(lut: &BlockLut, dis_f: f32, norm: f32, centering: bool) -> Residual {
        let lambda = if centering {
            rabitq::bit::projection(dis_f, norm)
        } else {
            1.0
        };
        Residual::dot(lut, dis_f, norm, lambda)
    }

    fn block_access(
        lut: &BlockLut,
        residual: Option<Residual>,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |sum, code, delta| match residual {
                None => rabitq::bit::block::half_process_dot(sum, code, lut.0),
                Some(residual) => {
                    rabitq::bit::block::half_process_dot_residual(sum, code, lut.0, residual, delta)
                }
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        residual: Option<Residual>,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        move |metadata: [f32; 4], elements: &[u64], delta: f32| {
            let sum = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            match residual {
                None => rabitq::bit::binary::half_process_dot(sum, code, lut.0),
                Some(residual) => rabitq::bit::binary::half_process_dot_residual(
                    sum, code, lut.0, residual, delta,
                ),
            }
        }
    }

    fn build(
        vector: VectBorrowed<'_, f16>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if let Some(centroid) = centroid {
            let residual = VectOwned::new(f16::vector_sub(vector.slice(), centroid.slice()));
            let code = Self::Vector::code(residual.as_borrowed());
            let delta = if code.0.dis_u_2 == 0.0 {
                // The vector equals its centroid: the residual is zero, so it has
                // no direction and contributes nothing (the formula below is 0/0).
                0.0
            } else {
                use std::iter::zip;
                let dim = vector.dim();
                let t = zip(&code.1, centroid.slice())
                    .map(|(&sign, &num)| std::hint::select_unpredictable(sign, num, -num))
                    .map(simd::F16::_to_f32)
                    .sum::<f32>()
                    / (dim as f32).sqrt();
                let sum_of_x_2 = code.0.dis_u_2;
                let sum_of_abs_x = sum_of_x_2 / code.0.factor_ip;
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (dim as f32).sqrt();
                dis_u * t / x_0 - f16::reduce_sum_of_xy(residual.slice(), centroid.slice())
            };
            (code, delta)
        } else {
            let code = Self::Vector::code(vector);
            let delta = 0.0;
            (code, delta)
        }
    }
}

impl Operator for Op<Rabitq8Owned, L2S> {
    type Vector = Rabitq8Owned;

    type DistanceAccessor = ByteDistanceAccessor<Rabitq8Owned, L2S>;

    type RoutingAccessor = Self::DistanceAccessor;

    fn routing_is_exact() -> bool {
        true
    }

    fn residual(_: &BlockLut, _: f32, _: f32, _: bool) -> Residual {
        unreachable!("residual quantization is not supported for rabitq8")
    }

    fn block_access(
        lut: &BlockLut,
        residual: Option<Residual>,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        assert!(residual.is_none());
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |value, code, _delta| {
                rabitq::bit::block::half_process_l2s(value, code, lut.0)
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        residual: Option<Residual>,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        assert!(residual.is_none());
        move |metadata: [f32; 4], elements: &[u64], _delta: f32| {
            let value = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            rabitq::bit::binary::half_process_l2s(value, code, lut.0)
        }
    }

    fn build(
        vector: Rabitq8Borrowed<'_>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if centroid.is_some() {
            unimplemented!();
        }
        (Self::Vector::code(vector), 0.0)
    }
}

impl Operator for Op<Rabitq8Owned, Dot> {
    type Vector = Rabitq8Owned;

    type DistanceAccessor = ByteDistanceAccessor<Rabitq8Owned, Dot>;

    type RoutingAccessor = Self::DistanceAccessor;

    fn routing_is_exact() -> bool {
        true
    }

    fn residual(_: &BlockLut, _: f32, _: f32, _: bool) -> Residual {
        unreachable!("residual quantization is not supported for rabitq8")
    }

    fn block_access(
        lut: &BlockLut,
        residual: Option<Residual>,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        assert!(residual.is_none());
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |sum, code, _delta| {
                rabitq::bit::block::half_process_dot(sum, code, lut.0)
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        residual: Option<Residual>,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        assert!(residual.is_none());
        move |metadata: [f32; 4], elements: &[u64], _delta: f32| {
            let sum = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            rabitq::bit::binary::half_process_dot(sum, code, lut.0)
        }
    }

    fn build(
        vector: Rabitq8Borrowed<'_>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if centroid.is_some() {
            unimplemented!();
        }
        (Self::Vector::code(vector), 0.0)
    }
}

impl Operator for Op<Rabitq4Owned, L2S> {
    type Vector = Rabitq4Owned;

    type DistanceAccessor = HalfbyteDistanceAccessor<Rabitq4Owned, L2S>;

    type RoutingAccessor = Self::DistanceAccessor;

    fn routing_is_exact() -> bool {
        true
    }

    fn residual(_: &BlockLut, _: f32, _: f32, _: bool) -> Residual {
        unreachable!("residual quantization is not supported for rabitq4")
    }

    fn block_access(
        lut: &BlockLut,
        residual: Option<Residual>,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        assert!(residual.is_none());
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |value, code, _delta| {
                rabitq::bit::block::half_process_l2s(value, code, lut.0)
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        residual: Option<Residual>,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        assert!(residual.is_none());
        move |metadata: [f32; 4], elements: &[u64], _delta: f32| {
            let value = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            rabitq::bit::binary::half_process_l2s(value, code, lut.0)
        }
    }

    fn build(
        vector: Rabitq4Borrowed<'_>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if centroid.is_some() {
            unimplemented!();
        }
        (Self::Vector::code(vector), 0.0)
    }
}

impl Operator for Op<Rabitq4Owned, Dot> {
    type Vector = Rabitq4Owned;

    type DistanceAccessor = HalfbyteDistanceAccessor<Rabitq4Owned, Dot>;

    type RoutingAccessor = Self::DistanceAccessor;

    fn routing_is_exact() -> bool {
        true
    }

    fn residual(_: &BlockLut, _: f32, _: f32, _: bool) -> Residual {
        unreachable!("residual quantization is not supported for rabitq4")
    }

    fn block_access(
        lut: &BlockLut,
        residual: Option<Residual>,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        assert!(residual.is_none());
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |sum, code, _delta| {
                rabitq::bit::block::half_process_dot(sum, code, lut.0)
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        residual: Option<Residual>,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        assert!(residual.is_none());
        move |metadata: [f32; 4], elements: &[u64], _delta: f32| {
            let sum = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            rabitq::bit::binary::half_process_dot(sum, code, lut.0)
        }
    }

    fn build(
        vector: Rabitq4Borrowed<'_>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if centroid.is_some() {
            unimplemented!();
        }
        (Self::Vector::code(vector), 0.0)
    }
}

pub trait Call<A, B, C> {
    type Output;

    fn call(&mut self, a: A, b: B, c: C) -> Self::Output;
}

impl<A, B, C, F: Fn(A, B, C) -> R, R> Call<A, B, C> for F {
    type Output = R;

    #[inline(always)]
    fn call(&mut self, a: A, b: B, c: C) -> R {
        (self)(a, b, c)
    }
}

#[cfg(test)]
mod tests {
    use super::Vector;
    use simd::Floating;
    use vector::VectorOwned;
    use vector::rabitq4::Rabitq4Owned;
    use vector::rabitq8::Rabitq8Owned;

    fn rabitq8(x: &[f32]) -> Rabitq8Owned {
        let (metadata, elements) = rabitq::byte::ugly_code(x);
        Rabitq8Owned::new(
            x.len() as u32,
            metadata.dis_u_2,
            metadata.norm_of_lattice,
            metadata.sum_of_code,
            f32::reduce_sum_of_abs_x(x),
            rabitq::byte::pack_code(&elements),
        )
    }

    fn rabitq4(x: &[f32]) -> Rabitq4Owned {
        let (metadata, elements) = rabitq::halfbyte::ugly_code(x);
        Rabitq4Owned::new(
            x.len() as u32,
            metadata.dis_u_2,
            metadata.norm_of_lattice,
            metadata.sum_of_code,
            f32::reduce_sum_of_abs_x(x),
            rabitq::halfbyte::pack_code(&elements),
        )
    }

    fn assert_finite(m: rabitq::bit::CodeMetadata, what: &str) {
        assert!(m.dis_u_2.is_finite(), "{what}: dis_u_2 = {}", m.dis_u_2);
        assert!(
            m.factor_cnt.is_finite(),
            "{what}: factor_cnt = {}",
            m.factor_cnt
        );
        assert!(
            m.factor_ip.is_finite(),
            "{what}: factor_ip = {}",
            m.factor_ip
        );
        assert!(
            m.factor_err.is_finite(),
            "{what}: factor_err = {}",
            m.factor_err
        );
    }

    // Zero and 1-dimensional vectors produced NaN factors (#23), and equal
    // magnitudes can round `1 / x_0^2 - 1` below zero.
    #[test]
    fn degenerate_codes_have_finite_factors() {
        let inputs: Vec<(&str, Vec<f32>)> = vec![
            ("zero 1", vec![0.0]),
            ("zero 8", vec![0.0; 8]),
            ("zero 100", vec![0.0; 100]),
            ("dim 1 positive", vec![3.0]),
            ("dim 1 negative", vec![-0.25]),
            ("equal magnitude", vec![1.0; 64]),
            (
                "equal magnitude signs",
                (0..65)
                    .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
                    .collect(),
            ),
        ];
        for (what, x) in inputs {
            let v8 = rabitq8(&x);
            assert_finite(<Rabitq8Owned as Vector>::code(v8.as_borrowed()).0, what);
            let v4 = rabitq4(&x);
            assert_finite(<Rabitq4Owned as Vector>::code(v4.as_borrowed()).0, what);
        }
    }

    fn gaussian(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        let mut uniform = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f64 + 0.5) / (1u64 << 24) as f64
        };
        move || {
            let (u, v) = (uniform(), uniform());
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
        }
    }

    fn normalized(x: Vec<f32>) -> Vec<f32> {
        let norm = f32::reduce_sum_of_x2(&x).sqrt();
        x.into_iter().map(|x| x / norm).collect()
    }

    /// Error statistics of `(rough, err)` against exact values.
    #[derive(Debug, Default)]
    struct Errors {
        n: u32,
        sum_of_e_2: f64,
        failures: u32,
    }

    impl Errors {
        fn push(&mut self, exact: f32, (rough, err): (f32, f32)) {
            self.n += 1;
            self.sum_of_e_2 += ((rough - exact) as f64).powi(2);
            self.failures += u32::from(rough - 1.9 * err > exact);
        }
        fn rms(&self) -> f64 {
            (self.sum_of_e_2 / self.n as f64).sqrt()
        }
        fn failure_rate(&self) -> f64 {
            self.failures as f64 / self.n as f64
        }
    }

    /// The residual dot product estimates of the block (frozen) and binary
    /// (appendable) paths, with the query centered on the centroid and on its
    /// projection on the centroid, against exact distances. The rows are unit
    /// vectors around a spherical centroid in a cone, as from CLIP; residuals
    /// are isotropic or skewed (decaying variances, rotated like the index
    /// does). "Same" queries are drawn like the rows; "cross" queries are
    /// nearly orthogonal to the cone, as text queries against images are.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn projected_centering() {
        use super::{Op, Operator};
        use index_accessor::{Accessor1, Dot};
        use vector::vect::{VectBorrowed, VectOwned};
        type O = Op<VectOwned<f32>, Dot>;
        let d = 256;
        let mut rng = gaussian(7);
        let axis = normalized((0..d).map(|_| rng()).collect());
        let direction = {
            let u = normalized((0..d).map(|_| rng()).collect());
            normalized((0..d).map(|i| 0.8 * axis[i] + 0.6 * u[i]).collect())
        };
        for skewed in [false, true] {
            let scale = |i: usize| {
                if skewed {
                    (-(i as f32) / 16.0).exp()
                } else {
                    1.0
                }
            };
            let mut sample = || {
                let g = normalized((0..d).map(|i| scale(i) * rng()).collect());
                let g = if skewed {
                    rabitq::rotate::rotate(&g)
                } else {
                    g
                };
                normalized((0..d).map(|i| direction[i] + 0.9 * g[i]).collect())
            };
            let rows = (0..512).map(|_| sample()).collect::<Vec<_>>();
            let same = (0..32).map(|_| sample()).collect::<Vec<_>>();
            let c = normalized((0..d).map(|i| rows.iter().map(|x| x[i]).sum()).collect());
            let cross = (0..32)
                .map(|_| normalized((0..d).map(|i| 0.25 * axis[i] + rng() / 16.0).collect()))
                .collect::<Vec<_>>();
            let norm = f32::reduce_sum_of_x2(&c).sqrt();
            let codes = rows
                .iter()
                .map(|x| O::build(VectBorrowed::new(x), Some(VectOwned::new(c.clone()))))
                .collect::<Vec<_>>();
            let blocks = codes
                .chunks(32)
                .map(|chunk| {
                    let u4 = chunk
                        .iter()
                        .map(|((_, signs), _)| rabitq::packing::pack_to_u4(signs))
                        .collect::<Vec<_>>();
                    let elements = rabitq::packing::pack(std::array::from_fn(|i| &u4[i][..]));
                    let metadata: [[f32; 32]; 4] = std::array::from_fn(|k| {
                        std::array::from_fn(|i| chunk[i].0.0.into_array()[k])
                    });
                    let delta: [f32; 32] = std::array::from_fn(|i| chunk[i].1);
                    (elements, metadata, delta)
                })
                .collect::<Vec<_>>();
            for (queries, what) in [(&same, "same"), (&cross, "cross")] {
                let mut errors: [[Errors; 2]; 2] = Default::default();
                for q in queries {
                    let lut = <VectOwned<f32> as Vector>::preprocess(VectBorrowed::new(q));
                    let dis_f = -f32::reduce_sum_of_xy(q, &c);
                    for (errors, centering) in errors.iter_mut().zip([false, true]) {
                        let residual = Some(O::residual(&lut.0, dis_f, norm, centering));
                        let exact = |i: usize| -f32::reduce_sum_of_xy(q, &rows[i]);
                        for (b, (elements, metadata, delta)) in blocks.iter().enumerate() {
                            let mut access = O::block_access(&lut.0, residual);
                            access.push(elements);
                            for (i, value) in
                                access.finish((metadata, delta)).into_iter().enumerate()
                            {
                                errors[0].push(exact(32 * b + i), value);
                            }
                        }
                        let mut access = O::binary_access(&lut.1, residual);
                        for (i, ((metadata, signs), delta)) in codes.iter().enumerate() {
                            let elements = rabitq::bit::binary::pack_code(signs);
                            let value = access(metadata.into_array(), &elements, *delta);
                            errors[1].push(exact(i), value);
                        }
                    }
                    // `|q - lambda c|` is smallest at the projection.
                    let bound = |centering: bool| {
                        let residual = Some(O::residual(&lut.0, dis_f, norm, centering));
                        let mut access = O::binary_access(&lut.1, residual);
                        let ((metadata, signs), delta) = &codes[0];
                        let elements = rabitq::bit::binary::pack_code(signs);
                        access(metadata.into_array(), &elements, *delta).1
                    };
                    assert!(bound(true) <= bound(false) * (1.0 + 1e-6), "{what}");
                }
                // The bound was conservative and fails slightly more often:
                // 2.0% to 2.6% of rows here at the centroid, 2.7% to 3.1% at
                // the projection. The error shrinks by about 20% for cross
                // queries and by about 6% for same queries.
                let [centroid, projected] = errors;
                for (centroid, projected) in centroid.iter().zip(&projected) {
                    assert!(projected.failure_rate() <= 0.05, "{what}: {projected:?}");
                    let ratio = projected.rms() / centroid.rms();
                    let limit = if what == "cross" { 0.85 } else { 1.0 };
                    assert!(ratio <= limit, "{what}: {ratio}");
                }
            }
        }
    }
}
