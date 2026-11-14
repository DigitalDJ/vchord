// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::closure_lifetime_binder::{id_0, id_1, id_2, id_3};
use crate::keys::{Columns, Keys};
use crate::operator::{Operator, Vector};
use crate::tape_writer::{DirectoryTapeWriter, FrozenTapeWriter};
use crate::tuples::*;
use crate::{Branch, Opaque, freepages, tape};
use index::prefetcher::PrefetcherSequenceFamily;
use index::relation::{
    Page, PageGuard, Relation, RelationRead, RelationReadTypes, RelationWrite, RelationWriteTypes,
};
use index_accessor::FunctionalAccessor;
use rabitq::packing::unpack;
use std::cell::RefCell;
use std::num::NonZero;

pub trait MaintainChooser {
    fn choose(&mut self, i: usize) -> bool;
    /// Whether a chosen list whose compaction would reproduce its tapes
    /// unchanged may be skipped (VACUUM); parallel build compaction always
    /// rewrites.
    fn skip_unchanged(&self) -> bool {
        false
    }
}

pub struct Maintain {
    pub number_of_formerly_allocated_pages: usize,
    pub number_of_freshly_allocated_pages: usize,
    pub number_of_freed_pages: usize,
    /// Live index tuples in the lists that were chosen and compacted.
    pub number_of_tuples: u64,
}

/// `memory`: the bytes that a list of an index with filter keys may take to
/// be sorted by its first keys (`crate::keys`); a larger list is written in
/// the order it is read.
pub fn maintain<'b, R: RelationRead + RelationWrite, O: Operator>(
    index: &'b R,
    mut prefetch_h0_tuples: impl PrefetcherSequenceFamily<'b, R>,
    chooser: &mut impl MaintainChooser,
    memory: usize,
    check: impl Fn(),
) -> Maintain
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dim = meta_tuple.dim();
    let height_of_root = meta_tuple.height_of_root();
    let freepages_first = meta_tuple.freepages_first();
    let keys = Keys::of_meta(meta_tuple);
    let keyed = !keys.is_empty();

    type State = Vec<u32>;
    let mut state: State = vec![meta_tuple.first()];

    drop(meta_guard);

    let step = |state: State| {
        let mut results = Vec::new();
        for first in state {
            tape::read_h1_tape::<R, _, _>(
                tape::by_next(index, first),
                || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _])),
                |(), _, _, first, _| results.push(first),
            );
            // `check` sleeps for cost-based delay and checks interrupts, which
            // LWLocks hold off: call it only where no buffer lock is held (#30).
            check();
        }
        results
    };

    for _ in (1..height_of_root).rev() {
        state = step(state);
    }

    struct Buffers {
        pages: Vec<u32>,
        number_of_formerly_allocated_pages: usize,
        number_of_freshly_allocated_pages: usize,
    }

    let buffers = RefCell::new(Buffers {
        pages: Vec::new(),
        number_of_formerly_allocated_pages: 0,
        number_of_freshly_allocated_pages: 0,
    });

    let mut number_of_tuples = 0_u64;
    let mut number_of_freed_pages = 0_usize;

    for (idx, first) in state.into_iter().enumerate() {
        if !chooser.choose(idx) {
            continue;
        }

        if chooser.skip_unchanged()
            && let Some(tuples) = inspect_unchanged::<R>(index, first)
        {
            number_of_tuples += tuples;
            check();
            continue;
        }

        let mut jump_guard = index.write(first, false);
        let jump_bytes = jump_guard.get_mut(1).expect("data corruption");
        let mut jump_tuple = JumpTuple::deserialize_mut(jump_bytes);

        let hooked_index = RelationHooked(index, {
            id_3(|index: &R, opaque: Opaque, tracking_freespace: bool| {
                if !tracking_freespace {
                    let mut buffers = buffers.borrow_mut();
                    if let Some(id) = buffers.pages.pop() {
                        drop(buffers);
                        let mut guard = index.write(id, false);
                        guard.clear(opaque);
                        guard
                    } else if let Some(mut guard) = freepages::alloc(index, freepages_first) {
                        buffers.number_of_formerly_allocated_pages += 1;
                        drop(buffers);
                        guard.clear(opaque);
                        guard
                    } else {
                        buffers.number_of_freshly_allocated_pages += 1;
                        drop(buffers);
                        index.extend(opaque, false)
                    }
                } else {
                    index.extend(opaque, true)
                }
            })
        });

        let prefetch = O::Vector::count(dim) as _;
        let mut tape = FrozenTapeWriter::create(&hooked_index, prefetch, &keys, false);

        let mut trace_directory = Vec::new();
        let mut trace_forzen = Vec::new();
        let mut trace_appendable = Vec::new();

        let mut tuples = 0_u64;
        // An index with filter keys writes the rows ordered by their first
        // keys (`crate::keys`), unless they take more than `memory`.
        let mut rows = Rows::new(dim as _);
        let mut sorting = keyed;
        let mut row = Vec::new();
        let mut callback = id_2(
            |(code, delta): (_, _), head, payload, prefetch: &[_], columns| {
                if sorting {
                    rows.push((code, delta), head, payload, prefetch, columns, &keys);
                    if rows.bytes() > memory {
                        for i in 0..rows.len() {
                            let (branch, columns) = rows.branch(i);
                            tape.push(branch, columns);
                        }
                        rows = Rows::new(dim as _);
                        sorting = false;
                    }
                } else {
                    let branch = Branch {
                        code,
                        delta,
                        prefetch: prefetch.to_vec(),
                        head,
                        norm: 0.0,
                        extra: payload,
                    };
                    row.clear();
                    columns.extend(&keys, &mut row);
                    tape.push(branch, &row);
                }
                tuples += 1;
            },
        );
        let directory = tape::read_directory_tape::<R>(
            tape::by_next(index, *jump_tuple.directory_first())
                .inspect(|guard| trace_directory.push(guard.id())),
        );
        tape::read_frozen_tape::<R, _, _>(
            tape::by_directory(&mut prefetch_h0_tuples, directory)
                .inspect(|guard| trace_forzen.push(guard.id())),
            tape::All,
            || {
                FunctionalAccessor::new(
                    Vec::<[u8; 16]>::new(),
                    Vec::<[u8; 16]>::extend_from_slice,
                    id_1(
                        |elements: Vec<_>, (metadata, delta): (&[[f32; 32]; 4], &[f32; 32])| {
                            let unpacked = unpack(&elements);
                            std::array::from_fn(|i| {
                                let f = |&x| [x & 1 != 0, x & 2 != 0, x & 4 != 0, x & 8 != 0];
                                let signs = unpacked[i].iter().flat_map(f).collect::<Vec<_>>();
                                (
                                    (
                                        rabitq::bit::CodeMetadata {
                                            dis_u_2: metadata[0][i],
                                            factor_cnt: metadata[1][i],
                                            factor_ip: metadata[2][i],
                                            factor_err: metadata[3][i],
                                        },
                                        signs,
                                    ),
                                    delta[i],
                                )
                            })
                        },
                    ),
                )
            },
            (),
            &mut callback,
        );
        tape::read_appendable_tape::<R, _>(
            tape::by_next(index, *jump_tuple.appendable_first())
                .inspect(|guard| trace_appendable.push(guard.id())),
            |metadata, elements, delta| (unpack_code(metadata, elements, dim as _), delta),
            (),
            &mut callback,
        );

        for i in rows.sorted() {
            let (branch, columns) = rows.branch(i as _);
            tape.push(branch, columns);
        }
        if keyed {
            // the reader of page ids stopped at the layout, before its last pages
            trace_directory = tape::by_next(index, *jump_tuple.directory_first())
                .map(|guard| guard.id())
                .collect();
        }

        let (frozen_tape, branches, columns) = tape.into_inner();

        let mut appendable_tape = tape::TapeWriter::create(&hooked_index, false);

        let width = keys.columns();
        for (i, branch) in branches.into_iter().enumerate() {
            appendable_tape.push(AppendableTuple {
                metadata: [
                    branch.code.0.dis_u_2,
                    branch.code.0.factor_cnt,
                    branch.code.0.factor_ip,
                    branch.code.0.factor_err,
                ],
                elements: rabitq::bit::binary::pack_code(&branch.code.1),
                delta: branch.delta,
                prefetch: branch.prefetch,
                head: branch.head,
                payload: Some(branch.extra),
                columns: columns[i * width..][..width].to_vec(),
            });
        }

        let frozen_first = { frozen_tape }.first();

        let mut directory = tape::Layout::default();
        for guard in tape::by_next(index, frozen_first) {
            directory.page(guard.id());
            if keyed {
                for i in 1..=guard.len() {
                    let bytes = guard.get(i).expect("data corruption");
                    directory.tuple(match FrozenTuple::deserialize_ref(bytes) {
                        FrozenTupleReader::_0(x) => {
                            Some(keys.range_of_block(x.payload(), x.columns()))
                        }
                        FrozenTupleReader::_1(_) => None,
                    });
                }
            }
        }

        let mut directory_tape = DirectoryTapeWriter::create(&hooked_index, false);
        directory_tape.push(directory.ids.as_slice());
        if keyed {
            directory_tape.push_layout(&directory.words());
        }
        let directory_tape = directory_tape.into_inner();

        *jump_tuple.directory_first() = { directory_tape }.first();
        *jump_tuple.frozen_first() = frozen_first;
        *jump_tuple.appendable_first() = { appendable_tape }.first();
        *jump_tuple.tuples() = tuples;
        number_of_tuples += tuples;

        // Publish the new tapes (dropping the guard WAL-logs the jump page),
        // then free this list's retired pages right away instead of keeping
        // them in memory until every list is done, so a cancellation at the
        // `check` below cannot lose them (#30). Later lists allocate them
        // back from the free list.
        drop(jump_guard);

        for id in trace_directory
            .into_iter()
            .chain(trace_forzen)
            .chain(trace_appendable)
        {
            freepages::free(index, freepages_first, id);
            number_of_freed_pages += 1;
        }
        check();
    }

    let buffers = RefCell::into_inner(buffers);
    for id in buffers.pages.iter().copied() {
        freepages::free(index, freepages_first, id);
        number_of_freed_pages += 1;
        check();
    }

    Maintain {
        number_of_formerly_allocated_pages: buffers.number_of_formerly_allocated_pages,
        number_of_freshly_allocated_pages: buffers.number_of_freshly_allocated_pages,
        number_of_freed_pages,
        number_of_tuples,
    }
}

/// Decides, under the list's jump page share lock, whether compacting the
/// list would reproduce its tapes unchanged: no frozen or appendable entry is
/// deleted (raw payload slots are inspected; the tape readers skip deleted
/// entries) and the appendable tape holds fewer tuples than one frozen block,
/// so compaction would write the same frozen blocks and the same remainder.
/// In an index with filter keys it would sort that remainder into the frozen
/// blocks;
/// that can wait, and the list keeps its layout, or its lack of one.
/// Returns the live tuple count of an unchanged list (#24).
fn inspect_unchanged<R: RelationRead>(index: &R, first: u32) -> Option<u64>
where
    R::Page: Page<Opaque = Opaque>,
{
    let jump_guard = index.read(first);
    let jump_bytes = jump_guard.get(1).expect("data corruption");
    let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
    let mut frozen = 0_u64;
    let directory =
        tape::read_directory_tape::<R>(tape::by_next(index, jump_tuple.directory_first()))
            .collect::<Vec<_>>();
    for id in directory {
        let guard = index.read(id);
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            if let FrozenTupleReader::_0(tuple) = FrozenTuple::deserialize_ref(bytes) {
                if tuple.payload().iter().any(Option::is_none) {
                    return None;
                }
                frozen += 32;
            }
        }
    }
    let mut appendable = 0_u64;
    for guard in tape::by_next(index, jump_tuple.appendable_first()) {
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            // a deleted entry: the list needs compacting
            AppendableTuple::deserialize_ref(bytes).payload()?;
            appendable += 1;
        }
        if appendable >= 32 {
            return None;
        }
    }
    drop(jump_guard);
    Some(frozen + appendable)
}

/// The rows of a list of an index with filter keys, packed, to be written
/// ordered by their first keys.
struct Rows {
    dim: usize,
    /// per row: code metadata, delta, head, payload, the fold of its first key
    fixed: Vec<([f32; 4], f32, u16, NonZero<u64>, u16)>,
    /// per row: the signs of the code, a bit each
    signs: Vec<u64>,
    /// per row: the same number of pages
    prefetch: Vec<u32>,
    /// per row: the filter key columns, as an appendable tuple stores them
    columns: Vec<u8>,
}

impl Rows {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            fixed: Vec::new(),
            signs: Vec::new(),
            prefetch: Vec::new(),
            columns: Vec::new(),
        }
    }
    fn len(&self) -> usize {
        self.fixed.len()
    }
    /// The bytes held, with those to sort.
    fn bytes(&self) -> usize {
        let fixed = size_of::<([f32; 4], f32, u16, NonZero<u64>, u16)>() + size_of::<u32>();
        let words = self.signs.len() * size_of::<u64>() + self.prefetch.len() * 4;
        self.len() * fixed + words + self.columns.len()
    }
    fn push(
        &mut self,
        (code, delta): (rabitq::bit::Code, f32),
        head: u16,
        payload: NonZero<u64>,
        prefetch: &[u32],
        columns: Columns<'_>,
        keys: &Keys,
    ) {
        let m = code.0;
        let metadata = [m.dis_u_2, m.factor_cnt, m.factor_ip, m.factor_err];
        let key = keys.sort_key(payload, columns);
        self.fixed.push((metadata, delta, head, payload, key));
        // a frozen code is padded to a multiple of 4 signs, with zeros
        let signs = rabitq::bit::binary::pack_code(&code.1[..self.dim]);
        self.signs.extend_from_slice(&signs);
        self.prefetch.extend_from_slice(prefetch);
        columns.extend(keys, &mut self.columns);
    }
    /// The rows in the order of their first keys, stable.
    fn sorted(&self) -> Vec<u32> {
        let mut order = (0..self.len() as u32).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&i| (self.fixed[i as usize].4, i));
        order
    }
    /// A row, with its filter key columns.
    fn branch(&self, i: usize) -> (Branch<NonZero<u64>>, &[u8]) {
        let (metadata, delta, head, payload, _) = self.fixed[i];
        let words = self.dim.div_ceil(64);
        let n = self.prefetch.len() / self.len();
        let width = self.columns.len() / self.len();
        let branch = Branch {
            code: unpack_code(metadata, &self.signs[i * words..][..words], self.dim),
            delta,
            prefetch: self.prefetch[i * n..][..n].to_vec(),
            head,
            norm: 0.0,
            extra: payload,
        };
        (branch, &self.columns[i * width..][..width])
    }
}

/// A code from its metadata and its first `dim` signs, packed in words
/// (`rabitq::bit::binary::pack_code`), as an appendable tuple stores them.
fn unpack_code(metadata: [f32; 4], words: &[u64], dim: usize) -> rabitq::bit::Code {
    let signs = words
        .iter()
        .flat_map(|x| std::array::from_fn::<_, 64, _>(|j| x & (1 << j) != 0))
        .take(dim)
        .collect();
    (rabitq::bit::CodeMetadata::from_array(metadata), signs)
}

#[derive(Clone)]
struct RelationHooked<'b, R, E>(&'b R, E);

impl<'b, R, E> Relation for RelationHooked<'b, R, E>
where
    R: Relation,
    E: Clone,
{
    type Page = R::Page;
}

impl<'b, R, E> RelationReadTypes for RelationHooked<'b, R, E>
where
    R: RelationRead,
    E: Clone,
{
    type ReadGuard<'a> = R::ReadGuard<'a>;
}

impl<'b, R, E> RelationRead for RelationHooked<'b, R, E>
where
    R: RelationRead,
    E: Clone,
{
    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        self.0.read(id)
    }
}

impl<'b, R, E> RelationWriteTypes for RelationHooked<'b, R, E>
where
    R: RelationWrite,
    E: Clone + for<'a> Fn(&'a R, <Self::Page as Page>::Opaque, bool) -> R::WriteGuard<'a>,
{
    type WriteGuard<'a> = R::WriteGuard<'a>;
}

impl<'b, R, E> RelationWrite for RelationHooked<'b, R, E>
where
    R: RelationWrite,
    E: Clone + for<'a> Fn(&'a R, <Self::Page as Page>::Opaque, bool) -> R::WriteGuard<'a>,
{
    fn commit_pair(&self, first: Self::WriteGuard<'_>, second: Self::WriteGuard<'_>) {
        self.0.commit_pair(first, second);
    }

    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        self.0.write(id, tracking_freespace)
    }

    fn extend(
        &self,
        opaque: <Self::Page as Page>::Opaque,
        tracking_freespace: bool,
    ) -> Self::WriteGuard<'_> {
        (self.1)(self.0, opaque, tracking_freespace)
    }

    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>> {
        self.0.search(freespace)
    }
}
