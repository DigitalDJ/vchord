// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::closure_lifetime_binder::{id_0, id_1, id_2};
use crate::linked_vec::LinkedVec;
use crate::operator::*;
use crate::tape::{by_directory, by_next};
use crate::tuples::*;
use crate::{Opaque, centroids, tape};
use always_equal::AlwaysEqual;
use dary_heap::QuaternaryHeap as Heap;
use distance::Distance;
use index::bump::Bump;
use index::fetch::BorrowedIter;
use index::packed::{PackedRefMut4, PackedRefMut8};
use index::prefetcher::{Prefetcher, PrefetcherHeapFamily, PrefetcherSequenceFamily, Sequence};
use index::relation::{Page, RelationRead};
use index_accessor::{DefaultWithDimension, FunctionalAccessor, LAccess};
use rabitq::bit::binary::BinaryLut;
use rabitq::bit::block::BlockLut;
use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::num::NonZero;
use std::rc::Rc;
use vector::{VectorBorrowed, VectorOwned};

type Extra1<'b> = &'b mut (u32, f32, u16, BorrowedIter<'b>);

/// Where a centroid's vector is stored: its head and the pages it spans.
/// Kept so its exact distance can be recomputed after approximate routing.
type Location<'b> = Option<(u16, BorrowedIter<'b>)>;

/// (distance, norm, first, location); until it is read, the distance may be routing's estimate.
type List<'b> = (
    Reverse<Distance>,
    AlwaysEqual<f32>,
    AlwaysEqual<u32>,
    AlwaysEqual<Location<'b>>,
);

pub type Candidate<'b> = (
    (Reverse<Distance>, AlwaysEqual<()>),
    AlwaysEqual<PackedRefMut4<'b, (NonZero<u64>, u16, BorrowedIter<'b>)>>,
);

pub type MaxsimCandidate<'b> = (
    (Reverse<Distance>, AlwaysEqual<Distance>),
    AlwaysEqual<PackedRefMut8<'b, (NonZero<u64>, u16, BorrowedIter<'b>)>>,
);

#[derive(Debug)]
pub struct SearchParams {
    pub probes: Vec<u32>,
    pub epsilon: f32,
}

struct Searcher<'b, R, O, V, B, H1, H0> {
    index: &'b R,
    bump: &'b B,
    vector: V,
    dim: u32,
    is_residual: bool,
    /// Bottom first.
    cells: Vec<u32>,
    root: List<'b>,
    lut: (BlockLut, BinaryLut),
    epsilon: f32,
    prefetch_h1_vectors: H1,
    prefetch_h0_tuples: H0,
    _operator: PhantomData<fn() -> O>,
}

struct Route<'b, I> {
    lists: Vec<List<'b>>,
    /// The other lists of the bottom level, nearest first, if `sorted`.
    rest: I,
}

impl<'b, R, O, V, B, H1, H0> Searcher<'b, R, O, V, B, H1, H0>
where
    R: RelationRead,
    R::Page: Page<Opaque = Opaque>,
    O: Operator,
    V: Borrow<O::Vector> + Clone,
    B: Bump,
    H1: PrefetcherHeapFamily<'b, R>,
    H0: PrefetcherSequenceFamily<'b, R>,
{
    fn new(
        index: &'b R,
        query: V,
        params: &SearchParams,
        bump: &'b B,
        prefetch_h1_vectors: H1,
        prefetch_h0_tuples: H0,
    ) -> Self {
        let vector = query.borrow().as_borrowed();
        let meta_guard = index.read(0);
        let meta_bytes = meta_guard.get(1).expect("data corruption");
        let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
        let dim = meta_tuple.dim();
        let is_residual = meta_tuple.is_residual();
        let height_of_root = meta_tuple.height_of_root();
        let cells = meta_tuple.cells().to_vec();
        assert_eq!(dim, vector.dim(), "unmatched dimensions");
        if height_of_root as usize != 1 + params.probes.len() {
            panic!(
                "usage: need {} probes, but {} probes provided",
                height_of_root - 1,
                params.probes.len()
            );
        }
        debug_assert_eq!(cells[(height_of_root - 1) as usize], 1);
        let norm = meta_tuple.centroid_norm();
        let first = meta_tuple.first();
        let root = if is_residual {
            let prefetch =
                BorrowedIter::from_slice(meta_tuple.centroid_prefetch(), |x| bump.alloc_slice(x));
            let head = meta_tuple.centroid_head();
            let distance = centroids::read::<R, O, _>(
                prefetch.map(|id| index.read(id)),
                head,
                LAccess::new(
                    O::Vector::unpack(vector),
                    O::RoutingAccessor::default_with_dimension(dim),
                ),
            );
            (
                Reverse(distance),
                AlwaysEqual(norm),
                AlwaysEqual(first),
                AlwaysEqual(Some((head, prefetch))),
            )
        } else {
            // fast path
            (
                Reverse(Distance::ZERO),
                AlwaysEqual(norm),
                AlwaysEqual(first),
                AlwaysEqual(None),
            )
        };
        drop(meta_guard);
        let lut = O::Vector::preprocess(vector);
        Self {
            index,
            bump,
            vector: query,
            dim,
            is_residual,
            cells,
            root,
            lut,
            epsilon: params.epsilon,
            prefetch_h1_vectors,
            prefetch_h0_tuples,
            _operator: PhantomData,
        }
    }

    fn route(
        &mut self,
        probes: &[u32],
        sorted: bool,
    ) -> Route<'b, impl Iterator<Item = List<'b>> + use<'b, R, O, V, B, H1, H0>> {
        let Self {
            ref vector,
            index,
            bump,
            dim,
            is_residual,
            ref cells,
            root,
            ref lut,
            epsilon,
            ref mut prefetch_h1_vectors,
            ..
        } = *self;
        let mut step = |lists: Vec<List<'b>>| {
            let mut results = LinkedVec::<(_, AlwaysEqual<Extra1<'b>>)>::new();
            for (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first), _) in lists {
                let residual =
                    is_residual.then(|| O::routing_residual(&lut.0, dis_f.to_f32(), norm));
                tape::read_h1_tape::<R, _, _>(
                    by_next(index, first),
                    || O::block_access(&lut.0, residual),
                    |(rough, err), head, norm, first, prefetch| {
                        let lowerbound = Distance::from_f32(rough - err * epsilon);
                        results.push((
                            Reverse(lowerbound),
                            AlwaysEqual(bump.alloc((
                                first,
                                norm,
                                head,
                                BorrowedIter::from_slice(prefetch, |x| bump.alloc_slice(x)),
                            ))),
                        ));
                    },
                );
            }
            let mut heap = prefetch_h1_vectors.prefetch(results.into_vec());
            let mut cache = BinaryHeap::<(_, _, _, _)>::new();
            let vector = vector.clone();
            std::iter::from_fn(move || {
                while let Some((
                    (Reverse(_), AlwaysEqual(&mut (first, norm, head, pages))),
                    prefetch,
                )) = heap.next_if(|(d, _)| Some(*d) > cache.peek().map(|(d, ..)| *d))
                {
                    let distance = centroids::read::<R, O, _>(
                        prefetch,
                        head,
                        LAccess::new(
                            O::Vector::unpack(vector.borrow().as_borrowed()),
                            O::RoutingAccessor::default_with_dimension(dim),
                        ),
                    );
                    cache.push((
                        Reverse(distance),
                        AlwaysEqual(norm),
                        AlwaysEqual(first),
                        AlwaysEqual(Some((head, pages))),
                    ));
                }
                cache.pop()
            })
        };
        let levels = probes.len();
        let mut lists = vec![root];
        let mut rest = None;
        for (i, (&probes, &cells)) in
            std::iter::zip(probes, cells[..levels].iter().rev()).enumerate()
        {
            let partial_scan = probes < cells;
            if partial_scan || is_residual || (sorted && i + 1 == levels) {
                lists = rest.insert(step(lists)).take(probes as _).collect();
            } else {
                // fast path
                let mut results = LinkedVec::new();
                for (Reverse(_), AlwaysEqual(_), AlwaysEqual(first), _) in lists {
                    tape::read_h1_tape::<R, _, _>(
                        by_next(index, first),
                        || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _])),
                        |(), _, norm, first, _| {
                            results.push((
                                Reverse(Distance::ZERO),
                                AlwaysEqual(norm),
                                AlwaysEqual(first),
                                AlwaysEqual(None),
                            ));
                        },
                    );
                }
                lists = results.into_vec();
            }
        }
        Route {
            lists,
            rest: rest.into_iter().flatten(),
        }
    }

    /// Returns how many rows the list holds.
    fn read_list(
        &mut self,
        (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first), AlwaysEqual(location)): List<'b>,
        mut callback: impl for<'a> FnMut((f32, f32), u16, NonZero<u64>, &'a [u32]),
    ) -> u64 {
        let Self {
            index,
            ref vector,
            dim,
            is_residual,
            ref lut,
            ref mut prefetch_h0_tuples,
            ..
        } = *self;
        // Between lists nothing is locked (#28).
        index::interrupt::check();
        // Residual bounds add `dis_f` with no error budget, so routing's estimate will not do.
        let dis_f = if is_residual && !O::routing_is_exact() {
            let (head, prefetch) = location.expect("internal: residual probe without location");
            centroids::read::<R, O, _>(
                prefetch.map(|id| index.read(id)),
                head,
                LAccess::new(
                    O::Vector::unpack(vector.borrow().as_borrowed()),
                    O::DistanceAccessor::default_with_dimension(dim),
                ),
            )
        } else {
            dis_f
        }
        .to_f32();
        let residual = is_residual.then(|| O::residual(&lut.0, dis_f, norm));
        let jump_guard = index.read(first);
        let jump_bytes = jump_guard.get(1).expect("data corruption");
        let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
        let block_access = || O::block_access(&lut.0, residual);
        if prefetch_h0_tuples.is_not_plain() {
            let directory =
                tape::read_directory_tape::<R>(by_next(index, jump_tuple.directory_first()));
            tape::read_frozen_tape::<R, _, _>(
                by_directory(prefetch_h0_tuples, directory),
                block_access,
                &mut callback,
            );
        } else {
            tape::read_frozen_tape::<R, _, _>(
                by_next(index, jump_tuple.frozen_first()),
                block_access,
                &mut callback,
            );
        }
        tape::read_appendable_tape::<R, _>(
            by_next(index, jump_tuple.appendable_first()),
            O::binary_access(&lut.1, residual),
            &mut callback,
        );
        jump_tuple.tuples()
    }

    fn read(&mut self, lists: Vec<List<'b>>) -> Vec<Candidate<'b>> {
        let (bump, epsilon) = (self.bump, self.epsilon);
        let mut results = LinkedVec::new();
        for list in lists {
            self.read_list(
                list,
                id_2(|(rough, err), head, payload, prefetch| {
                    let lowerbound = Distance::from_f32(rough - err * epsilon);
                    results.push((
                        (Reverse(lowerbound), AlwaysEqual(())),
                        AlwaysEqual(PackedRefMut4(bump.alloc((
                            payload,
                            head,
                            BorrowedIter::from_slice(prefetch, |x| bump.alloc_slice(x)),
                        )))),
                    ));
                }),
            );
        }
        results.into_vec()
    }
}

pub struct Candidates<'b> {
    heap: Heap<Candidate<'b>>,
}

impl<'b> Candidates<'b> {
    fn new(mut results: Vec<Candidate<'b>>, bound: Option<Distance>) -> Self {
        if let Some(bound) = bound {
            results.retain(|((Reverse(lowerbound), _), _)| *lowerbound < bound);
        }
        Self {
            heap: Heap::from(results),
        }
    }
}

impl<'b> Sequence for Candidates<'b> {
    type Item = Candidate<'b>;
    type Inner = std::vec::IntoIter<Candidate<'b>>;

    #[inline]
    fn next(&mut self) -> Option<Candidate<'b>> {
        self.heap.pop()
    }

    #[inline]
    fn peek(&mut self) -> Option<&Candidate<'b>> {
        self.heap.peek()
    }

    fn into_inner(self) -> Self::Inner {
        self.heap.into_vec().into_iter()
    }
}

/// Drops candidates whose lower bound reaches `bound`; a larger `epsilon` misses fewer rows.
pub fn default_search<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    vector: Rc<O::Vector>,
    bound: Option<Distance>,
    params: SearchParams,
    bump: &'b impl Bump,
    prefetch_h1_vectors: impl PrefetcherHeapFamily<'b, R>,
    prefetch_h0_tuples: impl PrefetcherSequenceFamily<'b, R>,
) -> Candidates<'b>
where
    R::Page: Page<Opaque = Opaque>,
{
    let mut searcher = Searcher::<_, O, _, _, _, _>::new(
        index,
        vector,
        &params,
        bump,
        prefetch_h1_vectors,
        prefetch_h0_tuples,
    );
    let Route { lists, .. } = searcher.route(&params.probes, false);
    Candidates::new(searcher.read(lists), bound)
}

pub fn maxsim_search<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    vector: &O::Vector,
    params: &SearchParams,
    mut threshold: u32,
    bump: &'b impl Bump,
    prefetch_h1_vectors: impl PrefetcherHeapFamily<'b, R>,
    prefetch_h0_tuples: impl PrefetcherSequenceFamily<'b, R>,
) -> (Vec<MaxsimCandidate<'b>>, Distance)
where
    R::Page: Page<Opaque = Opaque>,
{
    let mut searcher = Searcher::<_, O, _, _, _, _>::new(
        index,
        vector,
        params,
        bump,
        prefetch_h1_vectors,
        prefetch_h0_tuples,
    );
    let epsilon = params.epsilon;
    let Route { lists, rest, .. } = searcher.route(&params.probes, threshold != 0);
    let mut results = LinkedVec::<(_, AlwaysEqual<_>)>::new();
    for list in lists {
        let tuples = searcher.read_list(
            list,
            id_2(|(rough, err), head, payload, prefetch| {
                let lowerbound = Distance::from_f32(rough - err * epsilon);
                let rough = Distance::from_f32(rough);
                results.push((
                    (Reverse(lowerbound), AlwaysEqual(rough)),
                    AlwaysEqual(PackedRefMut8(bump.alloc((
                        payload,
                        head,
                        BorrowedIter::from_slice(prefetch, |x| bump.alloc_slice(x)),
                    )))),
                ));
            }),
        );
        threshold = threshold.saturating_sub(tuples.min(u32::MAX as _) as _);
    }
    let mut estimation_by_threshold = Distance::NEG_INFINITY;
    // Remaining routing distances only estimate missing MaxSim terms; they are
    // not lower bounds and never reach the reranker.
    for (Reverse(distance), AlwaysEqual(_), AlwaysEqual(first), _) in rest {
        if threshold == 0 {
            break;
        }
        index::interrupt::check();
        let jump_guard = index.read(first);
        let jump_bytes = jump_guard.get(1).expect("data corruption");
        let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
        threshold = threshold.saturating_sub(jump_tuple.tuples().min(u32::MAX as _) as _);
        estimation_by_threshold = distance;
    }
    (results.into_vec(), estimation_by_threshold)
}
