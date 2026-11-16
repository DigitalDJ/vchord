// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::closure_lifetime_binder::id_4;
use crate::operator::*;
use crate::tuples::{MetaTuple, WithReader};
use crate::{RerankMethod, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use index::fetch::BorrowedIter;
use index::packed::PackedRefMut;
use index::prefetcher::Prefetcher;
use index::relation::{Page, RelationRead};
use index_accessor::{Accessor2, DefaultWithDimension, LTryAccess};
use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::marker::PhantomData;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

type Result = (Reverse<Distance>, AlwaysEqual<NonZero<u64>>);

pub fn how(index: &impl RelationRead) -> RerankMethod {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    if rerank_in_heap {
        RerankMethod::Heap
    } else {
        RerankMethod::Index
    }
}

/// The table that returned rows are fetched from, read ahead of them.
pub trait Table {
    /// The most rows to read ahead; zero reads none.
    fn window(&self) -> usize;
    fn prefetch(&mut self, payload: NonZero<u64>);
}

impl Table for () {
    fn window(&self) -> usize {
        0
    }

    fn prefetch(&mut self, _: NonZero<u64>) {}
}

pub struct Reranker<T, F, P, W, H> {
    prefetcher: P,
    cache: BinaryHeap<Result>,
    /// The smallest cached rows, in order, with their table pages read ahead.
    ahead: VecDeque<Result>,
    returned: usize,
    f: F,
    table: H,
    _phantom: PhantomData<fn(T, W) -> (T, W)>,
}

impl<'b, T, F, P, W, H> Iterator for Reranker<T, F, P, W, H>
where
    F: FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut<T = (NonZero<u64>, u16, BorrowedIter<'b>)>,
    H: Table,
{
    type Item = (Distance, NonZero<u64>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // The previous candidate's guards are released here (#28).
            index::interrupt::check();
            let top = self.ahead.front().or(self.cache.peek()).map(|(d, ..)| *d);
            let Some(((_, AlwaysEqual(mut w)), prefetch)) =
                self.prefetcher.next_if(|((d, _), ..)| Some(*d) > top)
            else {
                // With nothing cached, every candidate was returned: the prefetcher may read more.
                if top.is_none() && self.prefetcher.refill() {
                    continue;
                }
                break;
            };
            let &mut (payload, head, ..) = w.get_mut();
            if let Some(distance) = (self.f)(payload, prefetch, head) {
                self.push((Reverse(distance), AlwaysEqual(payload)));
            };
        }
        let row = self.ahead.pop_front().or_else(|| self.cache.pop())?;
        self.returned += 1;
        while self.ahead.len() < self.window()
            && let Some(row) = self.cache.pop()
        {
            self.table.prefetch(row.1.0);
            self.ahead.push_back(row);
        }
        let (Reverse(distance), AlwaysEqual(payload)) = row;
        Some((distance, payload))
    }
}

impl<T, F, P, W, H: Table> Reranker<T, F, P, W, H> {
    /// Rows to read ahead, doubling with the rows returned so that a scan ended early reads few.
    fn window(&self) -> usize {
        (1 << (self.returned / 2 + 1).ilog2()).min(self.table.window())
    }

    fn push(&mut self, row: Result) {
        let i = self.ahead.partition_point(|x| x.0 >= row.0);
        if i < self.window() {
            self.table.prefetch(row.1.0);
            self.ahead.insert(i, row);
            if self.ahead.len() > self.window() {
                self.cache.extend(self.ahead.pop_back());
            }
        } else {
            self.cache.push(row);
        }
    }
}

impl<T, F, P, W, H> Reranker<T, F, P, W, H> {
    pub fn finish(self) -> (P, impl Iterator<Item = Result>) {
        (self.prefetcher, self.ahead.into_iter().chain(self.cache))
    }
}

pub fn rerank_index<
    'b,
    O: Operator,
    T,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut<T = (NonZero<u64>, u16, BorrowedIter<'b>)>,
>(
    vector: impl Borrow<O::Vector>,
    prefetcher: P,
    table: impl Table,
) -> Reranker<T, impl FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>, P, W, impl Table> {
    let dim = vector.borrow().as_borrowed().dim();
    Reranker {
        prefetcher,
        cache: BinaryHeap::new(),
        ahead: VecDeque::new(),
        returned: 0,
        table,
        f: id_4::<_, P, _, _, _>(move |payload, prefetch, head| {
            vectors::read::<P::R, O, _>(
                prefetch,
                head,
                payload,
                LTryAccess::new(
                    O::Vector::unpack(vector.borrow().as_borrowed()),
                    O::DistanceAccessor::default_with_dimension(dim),
                ),
            )
        }),
        _phantom: PhantomData,
    }
}

pub fn rerank_heap<
    'b,
    O: Operator,
    T,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut<T = (NonZero<u64>, u16, BorrowedIter<'b>)>,
>(
    vector: impl Borrow<O::Vector>,
    prefetcher: P,
    mut fetch: impl FnMut(NonZero<u64>) -> Option<O::Vector> + 'b,
) -> Reranker<T, impl FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>, P, W, ()> {
    let dim = vector.borrow().as_borrowed().dim();
    Reranker {
        prefetcher,
        cache: BinaryHeap::new(),
        ahead: VecDeque::new(),
        returned: 0,
        table: (),
        f: id_4::<_, P, _, _, _>(move |payload, _, _| {
            let unpack = O::Vector::unpack(vector.borrow().as_borrowed());
            let vector = fetch(payload)?;
            let vector = O::Vector::unpack(vector.as_borrowed());
            let mut accessor = O::DistanceAccessor::default_with_dimension(dim);
            accessor.push(unpack.0, vector.0);
            let distance = accessor.finish(unpack.1, vector.1);
            Some(distance)
        }),
        _phantom: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use index::packed::PackedRefMut4;
    use index::prefetcher::PlainPrefetcher;
    use index::test_support::MemoryRelation;
    use std::cell::RefCell;

    struct Recorder<'a>(usize, &'a RefCell<Vec<NonZero<u64>>>);

    impl Table for Recorder<'_> {
        fn window(&self) -> usize {
            self.0
        }

        fn prefetch(&mut self, payload: NonZero<u64>) {
            self.1.borrow_mut().push(payload);
        }
    }

    #[test]
    fn reading_ahead_keeps_the_order() {
        let index = MemoryRelation::<crate::Opaque>::new([]);
        // (lower bound, exact distance); the fourth row's distance is below its bound
        let rows = [
            (0.1, 0.3),
            (0.2, 0.25),
            (0.3, 0.9),
            (0.35, 0.33),
            (0.4, 0.5),
            (0.6, 0.7),
        ];
        let run = |table: Recorder| {
            let candidates = rows.iter().enumerate().map(|(i, &(bound, _))| {
                let payload = NonZero::new(i as u64 + 1).unwrap();
                let w = Box::leak(Box::new((
                    payload,
                    0,
                    BorrowedIter::from_slice(&[], |_| &[]),
                )));
                (
                    (Reverse(Distance::from_f32(bound)), AlwaysEqual(())),
                    AlwaysEqual(PackedRefMut4(w)),
                )
            });
            let prefetcher = PlainPrefetcher::new(&index, candidates.collect::<BinaryHeap<_>>());
            let reranker = Reranker {
                prefetcher,
                cache: BinaryHeap::new(),
                ahead: VecDeque::new(),
                returned: 0,
                table,
                f: |payload: NonZero<u64>, _, _| {
                    Some(Distance::from_f32(rows[payload.get() as usize - 1].1))
                },
                _phantom: PhantomData::<fn((), PackedRefMut4<_>) -> ((), PackedRefMut4<_>)>,
            };
            reranker
                .map(|(_, payload)| payload.get())
                .collect::<Vec<_>>()
        };
        let (none, one, four) = Default::default();
        let expected = run(Recorder(0, &none));
        assert_eq!(expected, [2, 1, 4, 5, 6, 3]);
        assert_eq!(run(Recorder(1, &one)), expected);
        assert_eq!(run(Recorder(4, &four)), expected);
        let read = |x: RefCell<Vec<NonZero<u64>>>| {
            x.take().into_iter().map(NonZero::get).collect::<Vec<_>>()
        };
        assert!(read(none).is_empty());
        // the window ramps from one row, so the first rows are read as with a window of one
        assert_eq!(read(one), [1, 2, 1, 3, 4, 3, 5, 3, 6, 3]);
        assert_eq!(read(four), [1, 2, 1, 3, 4, 5, 6]);
    }
}
