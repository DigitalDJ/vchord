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

/// A list: (distance, norm, first, location). Until the list is read, the
/// distance is a routing distance and may be approximate.
type List<'b> = (
    Reverse<Distance>,
    AlwaysEqual<f32>,
    AlwaysEqual<u32>,
    AlwaysEqual<Location<'b>>,
);

/// A candidate row: the lower bound of its distance and where it is stored.
pub type Candidate<'b> = (
    (Reverse<Distance>, AlwaysEqual<()>),
    AlwaysEqual<PackedRefMut4<'b, (NonZero<u64>, u16, BorrowedIter<'b>)>>,
);

/// A candidate row of a MaxSim search: the lower bound of its distance, its
/// estimated distance and where it is stored.
pub type MaxsimCandidate<'b> = (
    (Reverse<Distance>, AlwaysEqual<Distance>),
    AlwaysEqual<PackedRefMut8<'b, (NonZero<u64>, u16, BorrowedIter<'b>)>>,
);

/// How a search chooses lists and which of their rows it returns.
#[derive(Debug)]
pub struct SearchParams {
    /// How many of the nearest lists to read at each level below the root.
    pub probes: Vec<u32>,
    pub epsilon: f32,
    /// Center residual estimates on the query's projection on each list's
    /// centroid ([`Operator::residual`]).
    pub projected_centering: bool,
}

/// One search of an index for a vector `V` (an `O::Vector`, owned or shared):
/// what routing and reading lists share.
struct Searcher<'b, R, O, V, B, H1, H0> {
    index: &'b R,
    bump: &'b B,
    vector: V,
    dim: u32,
    is_residual: bool,
    /// The number of lists at each level, bottom first.
    cells: Vec<u32>,
    root: List<'b>,
    lut: (BlockLut, BinaryLut),
    epsilon: f32,
    projected_centering: bool,
    prefetch_h1_vectors: H1,
    prefetch_h0_tuples: H0,
    _operator: PhantomData<fn() -> O>,
}

/// The lists that routing selects at the bottom level.
struct Route<'b, I> {
    /// The lists that the probes select, nearest first.
    lists: Vec<List<'b>>,
    /// The other lists, nearest first, if routing was asked to sort them.
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
            projected_centering: params.projected_centering,
            prefetch_h1_vectors,
            prefetch_h0_tuples,
            _operator: PhantomData,
        }
    }

    /// The lists that `probes` selects, nearest first at each level. With
    /// `sorted`, the rest of the bottom level follows them in order.
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
                // Routing keeps the query centered on the centroid.
                let residual =
                    is_residual.then(|| O::residual(&lut.0, dis_f.to_f32(), norm, false));
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

    /// Reads the rows of `list`: its frozen tape, then its appendable tape.
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
            projected_centering,
            ref mut prefetch_h0_tuples,
            ..
        } = *self;
        // Between lists nothing is locked (#28).
        index::interrupt::check();
        // The list was chosen with `O::RoutingAccessor`, which may be
        // approximate. In residual mode its distance is reused as `dis_f`,
        // which the residual formulas add to every row's lower bound with no
        // error budget, so recompute it with `O::DistanceAccessor`; routing has
        // just read its pages. Non-residual formulas ignore `dis_f`, and the
        // fast path never computes it.
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
        // With `projected_centering`, the query is centered on its projection
        // on this list's centroid. Routing keeps the centroid, so the same
        // lists are probed either way.
        let residual = is_residual.then(|| O::residual(&lut.0, dis_f, norm, projected_centering));
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

    /// The candidates in `lists`.
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

/// The candidates of a search, nearest lower bound first.
pub struct Candidates<'b> {
    heap: Heap<Candidate<'b>>,
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

/// The search keeps `vector`, which the caller may share with its reranker.
pub fn default_search<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    vector: Rc<O::Vector>,
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
    Candidates {
        heap: Heap::from(searcher.read(lists)),
    }
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
