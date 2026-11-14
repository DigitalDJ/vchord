// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::keys::Columns;
use crate::tuples::*;
use crate::{Opaque, freepages};
use index::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use index::relation::{Page, PageGuard, RelationRead, RelationWrite};
use index_accessor::Accessor1;
use std::marker::PhantomData;
use std::num::NonZero;

/// Writes a new tape page by page.
///
/// When the head page is full, it is linked to a freshly extended page and its
/// WAL record is written before the new page's record. So the tape must stay
/// unreachable from readers until the writer is dropped: either the index is
/// still being built, or (in `maintain`) the tape is published by updating the
/// jump tuple only after every writer of it has been dropped. Appending to a
/// reachable tape goes through [`append`] instead.
pub struct TapeWriter<'a, R, T>
where
    R: RelationWrite + 'a,
{
    head: R::WriteGuard<'a>,
    first: u32,
    index: &'a R,
    tracking_freespace: bool,
    _phantom: PhantomData<fn(T) -> T>,
}

impl<'a, R, T> TapeWriter<'a, R, T>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
{
    pub fn create(index: &'a R, tracking_freespace: bool) -> Self {
        let mut head = index.extend(
            Opaque {
                next: u32::MAX,
                skip: u32::MAX,
            },
            tracking_freespace,
        );
        head.get_opaque_mut().skip = head.id();
        let first = head.id();
        Self {
            head,
            first,
            index,
            tracking_freespace,
            _phantom: PhantomData,
        }
    }
    pub fn first(&self) -> u32 {
        self.first
    }
    pub fn freespace(&self) -> u16 {
        self.head.freespace()
    }
    pub fn tape_move(&mut self) {
        if self.head.len() == 0 {
            panic!("implementation: a clear page cannot accommodate a single tuple");
        }
        let next = self.index.extend(
            Opaque {
                next: u32::MAX,
                skip: u32::MAX,
            },
            self.tracking_freespace,
        );
        self.head.get_opaque_mut().next = next.id();
        self.head = next;
    }
}

impl<'a, R, T> TapeWriter<'a, R, T>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
    T: Tuple,
{
    pub fn push(&mut self, x: T) -> (u32, u16) {
        let bytes = T::serialize(&x);
        if let Some(i) = self.head.alloc(&bytes) {
            (self.head.id(), i)
        } else {
            let next = self.index.extend(
                Opaque {
                    next: u32::MAX,
                    skip: u32::MAX,
                },
                self.tracking_freespace,
            );
            self.head.get_opaque_mut().next = next.id();
            self.head = next;
            if let Some(i) = self.head.alloc(&bytes) {
                (self.head.id(), i)
            } else {
                panic!("implementation: a free page cannot accommodate a single tuple")
            }
        }
    }
    pub fn tape_put(&mut self, x: T) -> (u32, u16) {
        let bytes = T::serialize(&x);
        if let Some(i) = self.head.alloc(&bytes) {
            (self.head.id(), i)
        } else {
            panic!("implementation: a free page cannot accommodate a single tuple")
        }
    }
}

pub fn read_directory_tape<'b, R>(
    iter: impl Iterator<Item = R::ReadGuard<'b>>,
) -> impl Iterator<Item = u32>
where
    R: RelationRead + 'b,
{
    use std::pin::Pin;
    use std::ptr::NonNull;

    #[pin_project::pin_project]
    struct State<'b, R: RelationRead + 'b, I> {
        slice: NonNull<[u32]>,
        #[pin]
        now: Option<(R::ReadGuard<'b>, u16)>,
        iter: I,
    }

    impl<'b, R: RelationRead + 'b, I: Iterator<Item = R::ReadGuard<'b>>> State<'b, R, I> {
        fn init(self: Pin<&mut Self>) {
            let mut this = self.project();
            let now = this.iter.next().map(|guard| (guard, 0));
            this.now.set(now);
        }

        fn next(mut self: Pin<&mut Self>) -> Option<u32> {
            loop {
                let mut this = self.as_mut().project();
                // Safety: If the slice is not empty, the function will return immediately,
                // so the guard will not be moved or dropped and the slice remains valid. If
                // the slice is empty, a pointer is trivially never dangling, so it's safe
                // to use.
                #[allow(unsafe_code)]
                if let Some((first, more)) = unsafe { this.slice.as_ref() }.split_first() {
                    *this.slice = more.into();
                    return Some(*first);
                }
                // Safety: `guard` is never moved in this block
                #[allow(unsafe_code)]
                if let Some((guard, i)) = unsafe { this.now.as_mut().get_unchecked_mut() } {
                    if *i < guard.len() {
                        *i += 1;
                        let bytes = guard.get(*i).expect("data corruption");
                        let tuple = DirectoryTuple::deserialize_ref(bytes);
                        let elements = match tuple {
                            DirectoryTupleReader::_0(tuple) => Some(tuple.elements()),
                            DirectoryTupleReader::_1(tuple) => Some(tuple.elements()),
                            // the layout of an index with filter keys, after every page id
                            DirectoryTupleReader::_2(_) => None,
                        };
                        if let Some(elements) = elements {
                            *this.slice = elements.into();
                            continue;
                        }
                    } else {
                        let now = this.iter.next().map(|guard| (guard, 0));
                        this.now.set(now);
                        continue;
                    }
                }
                this.now.set(None);
                return None;
            }
        }
    }

    let mut state = Box::pin(State::<'b, R, _> {
        slice: NonNull::from(&mut []),
        now: None,
        iter,
    });

    impl<'b, R: RelationRead + 'b, I: Iterator<Item = R::ReadGuard<'b>>> Iterator
        for Pin<Box<State<'b, R, I>>>
    {
        type Item = u32;

        fn next(&mut self) -> Option<u32> {
            self.as_mut().next()
        }
    }

    state.as_mut().init();

    state
}

pub fn by_directory<'b, R>(
    p: &mut impl PrefetcherSequenceFamily<'b, R>,
    iter: impl Iterator<Item = u32>,
) -> impl Iterator<Item = R::ReadGuard<'b>>
where
    R: RelationRead + 'b,
{
    let mut t = p.prefetch(iter.peekable());
    std::iter::from_fn(move || {
        let (_, mut x) = t.next()?;
        let ret = x.next().expect("should be at least one element");
        assert!(x.next().is_none(), "should be at most one element");
        Some(ret)
    })
}

pub fn by_next<'b, R>(index: &'b R, first: u32) -> impl Iterator<Item = R::ReadGuard<'b>>
where
    R: RelationRead + 'b,
    R::Page: Page<Opaque = Opaque>,
{
    let mut current = first;
    std::iter::from_fn(move || {
        if current != u32::MAX {
            let guard = index.read(current);
            current = guard.get_opaque().next;
            Some(guard)
        } else {
            None
        }
    })
}

pub fn read_h1_tape<'b, R, A, T>(
    iter: impl Iterator<Item = R::ReadGuard<'b>>,
    accessor: impl Fn() -> A,
    mut callback: impl for<'a> FnMut(T, u16, f32, u32, &'a [u32]),
) where
    R: RelationRead + 'b,
    A: for<'a> Accessor1<[u8; 16], (&'a [[f32; 32]; 4], &'a [f32; 32]), Output = [T; 32]>,
{
    let mut x = None;
    for guard in iter {
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            let tuple = H1Tuple::deserialize_ref(bytes);
            match tuple {
                H1TupleReader::_0(tuple) => {
                    let mut x = x.take().unwrap_or_else(&accessor);
                    x.push(tuple.elements());
                    let values = x.finish((tuple.metadata(), tuple.delta()));
                    let prefetch = tuple.prefetch();
                    let flattened = prefetch.as_flattened();
                    let step = prefetch.len();
                    for (j, value) in values.into_iter().enumerate() {
                        if j < tuple.len() as usize {
                            callback(
                                value,
                                tuple.head()[j],
                                tuple.norm()[j],
                                tuple.first()[j],
                                &flattened[j * step..][..step],
                            );
                        }
                    }
                }
                H1TupleReader::_1(tuple) => {
                    x.get_or_insert_with(&accessor).push(tuple.elements());
                }
            }
        }
    }
}

/// Selects the rows a tape reader passes on; `()` selects every row.
pub trait Filter: Copy {
    /// Whether to choose the blocks of a frozen tape to read by the ranges
    /// of their first filter keys ([`plan`]).
    fn selects_blocks(self) -> bool;
    /// The blocks of a frozen tape to read, given the ranges of the 16-bit
    /// keys of their rows' first filter keys (`crate::keys::Key::sort_key`),
    /// as a bitmask: bit `i` is block `i`.
    fn ranges(self, ranges: &[u32; 32]) -> u32;
    /// The rows of a frozen block to pass on, given their payloads and the
    /// block's filter key columns (`crate::keys`), as a bitmask: bit `i` is
    /// row `i`.
    fn block(self, payload: &[Option<NonZero<u64>>; 32], columns: &[u8]) -> u32;
    /// Whether to pass on an appendable row.
    fn row(self, payload: NonZero<u64>, columns: Columns<'_>) -> bool;
}

impl Filter for () {
    #[inline(always)]
    fn selects_blocks(self) -> bool {
        false
    }

    #[inline(always)]
    fn ranges(self, _: &[u32; 32]) -> u32 {
        u32::MAX
    }

    #[inline(always)]
    fn block(self, _: &[Option<NonZero<u64>>; 32], _: &[u8]) -> u32 {
        u32::MAX
    }

    #[inline(always)]
    fn row(self, _: NonZero<u64>, _: Columns<'_>) -> bool {
        true
    }
}

/// The blocks of a frozen tape that a reader reads, over the pages it is
/// given.
pub trait Blocks {
    /// The index of the block that the first tuple of the next page belongs
    /// to.
    fn page(&mut self) -> u32;
    /// Whether to read block `b`.
    fn contains(&self, b: u32) -> bool;
}

/// Every block, over every page of the tape.
pub struct All;

impl Blocks for All {
    #[inline(always)]
    fn page(&mut self) -> u32 {
        0
    }

    #[inline(always)]
    fn contains(&self, _: u32) -> bool {
        true
    }
}

/// The blocks that a [`plan`] selects, over the pages it keeps.
pub struct Selected<'a> {
    firsts: std::slice::Iter<'a, u32>,
    masks: &'a [u32],
}

impl<'a> Selected<'a> {
    /// `firsts`: the first block of each page; `masks`: the selected blocks,
    /// as a bitmask.
    pub fn new(firsts: &'a [u32], masks: &'a [u32]) -> Self {
        Self {
            firsts: firsts.iter(),
            masks,
        }
    }
}

impl Blocks for Selected<'_> {
    #[inline(always)]
    fn page(&mut self) -> u32 {
        *self
            .firsts
            .next()
            .expect("internal: a page without its first block")
    }

    #[inline(always)]
    fn contains(&self, b: u32) -> bool {
        self.masks[b as usize / 32] >> (b % 32) & 1 != 0
    }
}

/// Reads the rows of a frozen tape that `blocks` and `filter` select. The
/// other blocks are neither decoded nor passed on; `pages` must hold every
/// part of a selected block.
pub fn read_frozen_tape<'b, R, A, T>(
    pages: impl Iterator<Item = R::ReadGuard<'b>>,
    mut blocks: impl Blocks,
    accessor: impl Fn() -> A,
    filter: impl Filter,
    mut callback: impl for<'a> FnMut(T, u16, NonZero<u64>, &'a [u32], Columns<'a>),
) where
    R: RelationRead + 'b,
    A: for<'a> Accessor1<[u8; 16], (&'a [[f32; 32]; 4], &'a [f32; 32]), Output = [T; 32]>,
{
    let mut x = None;
    for guard in pages {
        let mut b = blocks.page();
        for i in 1..=guard.len() {
            let selected = blocks.contains(b);
            let bytes = guard.get(i).expect("data corruption");
            let tuple = FrozenTuple::deserialize_ref(bytes);
            match tuple {
                FrozenTupleReader::_0(tuple) => {
                    b += 1;
                    let mask = if selected {
                        filter.block(tuple.payload(), tuple.columns())
                    } else {
                        0
                    };
                    if mask == 0 {
                        // Skip the block, with the parts of it pushed so far.
                        x = None;
                        continue;
                    }
                    let columns = tuple.columns();
                    let mut x = x.take().unwrap_or_else(&accessor);
                    x.push(tuple.elements());
                    let values = x.finish((tuple.metadata(), tuple.delta()));
                    let prefetch = tuple.prefetch();
                    let flattened = prefetch.as_flattened();
                    let step = prefetch.len();
                    for (j, value) in values.into_iter().enumerate() {
                        if mask & (1 << j) != 0
                            && let Some(payload) = tuple.payload()[j]
                        {
                            callback(
                                value,
                                tuple.head()[j],
                                payload,
                                &flattened[j * step..][..step],
                                Columns::block(columns, j),
                            );
                        }
                    }
                }
                FrozenTupleReader::_1(tuple) => {
                    if selected {
                        x.get_or_insert_with(&accessor).push(tuple.elements());
                    }
                }
            }
        }
    }
}

/// The bit of a page's first block that tells that the page begins with a
/// part of a block begun on an earlier page (`crate::keys`).
pub const CONTINUED: u32 = 1 << 31;

/// The page ids and the layout of a frozen tape (`DirectoryTuple`), as its
/// pages and tuples are read in order.
#[derive(Debug, Default)]
pub struct Layout {
    pub ids: Vec<u32>,
    firsts: Vec<u32>,
    ranges: Vec<u32>,
    continued: bool,
}

impl Layout {
    pub fn page(&mut self, id: u32) {
        let continued = if self.continued { CONTINUED } else { 0 };
        self.ids.push(id);
        self.firsts.push(self.ranges.len() as u32 | continued);
    }
    /// A tuple of the last page: a block's last with the range of its rows' keys
    /// (`FrozenTuple::_0`), or a part before it (`FrozenTuple::_1`).
    pub fn tuple(&mut self, range: Option<u32>) {
        self.continued = range.is_none();
        self.ranges.extend(range);
    }
    /// The words that follow the page ids in the directory.
    pub fn words(&self) -> Vec<u32> {
        [self.firsts.as_slice(), self.ranges.as_slice()].concat()
    }
}

/// Reads the words of a directory into `words`: the ids of the pages of the
/// frozen tape, then, with `layout`, its layout if it has one. Returns how
/// many pages that is.
pub fn read_directory_words<'b, R>(
    iter: impl Iterator<Item = R::ReadGuard<'b>>,
    words: &mut Vec<u32>,
    layout: bool,
) -> usize
where
    R: RelationRead + 'b,
{
    words.clear();
    let mut pages = 0;
    for guard in iter {
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            let (elements, is_id) = match DirectoryTuple::deserialize_ref(bytes) {
                DirectoryTupleReader::_0(tuple) => (tuple.elements(), true),
                DirectoryTupleReader::_1(tuple) => (tuple.elements(), true),
                DirectoryTupleReader::_2(tuple) => (tuple.elements(), false),
            };
            if !is_id && !layout {
                // the layout follows every page id
                return pages;
            }
            words.extend_from_slice(elements);
            pages += if is_id { elements.len() } else { 0 };
        }
    }
    pages
}

/// Plans a filtered read of a frozen tape from the words of its directory:
/// `pages` page ids, then its layout if it has one. `None` if every page is to
/// be read: there is no layout, or `filter` selects every block. Otherwise,
/// rewrites `words` so that it starts with the ids of the pages to read, the
/// words from `pages` on hold the first block of each of them, and the words
/// from `2 * pages` on hold the selected blocks as a bitmask, and returns how
/// many pages that is.
pub fn plan(words: &mut [u32], pages: usize, filter: impl Filter) -> Option<usize> {
    let (ids, layout) = words.split_at_mut(pages);
    let (firsts, ranges) = layout.split_at_mut_checked(pages)?;
    let blocks = ranges.len();
    let masks = select_blocks(filter, ranges);
    if masks.iter().map(|x| x.count_ones() as usize).sum::<usize>() == blocks {
        return None;
    }
    Some(select_pages(ids, firsts, masks, blocks))
}

/// Replaces the ranges of blocks with the bitmask of the blocks that
/// `filter` selects, one word per 32 blocks, and returns that.
fn select_blocks(filter: impl Filter, ranges: &mut [u32]) -> &[u32] {
    let n = ranges.len().div_ceil(32);
    for i in 0..n {
        // word `i` is written once the chunk `i` that holds it is read
        let chunk = &ranges[32 * i..ranges.len().min(32 * (i + 1))];
        let len = chunk.len();
        let mut padded = [crate::keys::EMPTY_RANGE; 32];
        padded[..len].copy_from_slice(chunk);
        ranges[i] = filter.ranges(&padded) & u32::MAX >> (32 - len);
    }
    &ranges[..n]
}

/// Keeps, in place, the pages that hold a part of a block that `masks`
/// selects, and their first blocks without [`CONTINUED`]; returns how many.
fn select_pages(ids: &mut [u32], firsts: &mut [u32], masks: &[u32], blocks: usize) -> usize {
    let n = ids.len();
    let mut kept = 0;
    for i in 0..n {
        let first = firsts[i] & !CONTINUED;
        // the blocks with a part on page `i` are `first..end`
        let end = match firsts.get(i + 1) {
            Some(&next) => (next & !CONTINUED) + (next >> 31),
            None => blocks as u32,
        };
        if (first..end).any(|j| masks[j as usize / 32] >> (j % 32) & 1 != 0) {
            ids[kept] = ids[i];
            firsts[kept] = first;
            kept += 1;
        }
    }
    kept
}

pub fn read_appendable_tape<'b, R, T>(
    iter: impl Iterator<Item = R::ReadGuard<'b>>,
    mut access: impl for<'a> FnMut([f32; 4], &'a [u64], f32) -> T,
    filter: impl Filter,
    mut callback: impl for<'a> FnMut(T, u16, NonZero<u64>, &'a [u32], Columns<'a>),
) where
    R: RelationRead + 'b,
{
    for guard in iter {
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            let tuple = AppendableTuple::deserialize_ref(bytes);
            let columns = Columns::row(tuple.columns());
            if let Some(payload) = tuple.payload()
                && filter.row(payload, columns)
            {
                let value = access(tuple.metadata(), tuple.elements(), tuple.delta());
                callback(value, tuple.head(), payload, tuple.prefetch(), columns);
            }
        }
    }
}

#[allow(clippy::collapsible_else_if)]
pub fn append<R: RelationRead + RelationWrite>(
    index: &R,
    first: u32,
    bytes: &[u8],
    tracking_freespace: bool,
    freepages_first: Option<u32>,
) -> (u32, u16)
where
    R::Page: Page<Opaque = Opaque>,
{
    assert!(!tracking_freespace || freepages_first.is_none());
    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = index.read(current);
        if read.freespace() as usize >= bytes.len() || read.get_opaque().next == u32::MAX {
            drop(read);
            let mut write = index.write(current, tracking_freespace);
            if write.get_opaque().next == u32::MAX {
                if let Some(i) = write.alloc(bytes) {
                    return (current, i);
                }
                let mut extend = {
                    if let Some(freepages_first) = freepages_first {
                        if let Some(mut guard) = freepages::alloc(index, freepages_first) {
                            guard.clear(Opaque {
                                next: u32::MAX,
                                skip: u32::MAX,
                            });
                            guard
                        } else {
                            index.extend(
                                Opaque {
                                    next: u32::MAX,
                                    skip: u32::MAX,
                                },
                                tracking_freespace,
                            )
                        }
                    } else {
                        index.extend(
                            Opaque {
                                next: u32::MAX,
                                skip: u32::MAX,
                            },
                            tracking_freespace,
                        )
                    }
                };
                let fresh = extend.id();
                let Some(i) = extend.alloc(bytes) else {
                    panic!("implementation: a clear page cannot accommodate a single tuple");
                };
                if tracking_freespace {
                    // FSM allocation bypasses the tail lock. Publish the new
                    // vector page and its incoming link in one WAL record,
                    // before making its free space available to other writers.
                    write.get_opaque_mut().next = fresh;
                    index.commit_pair(write, extend);
                } else {
                    // These pages are not allocated through the FSM. Log the
                    // new page before exposing it through the locked tail.
                    drop(extend);
                    write.get_opaque_mut().next = fresh;
                    drop(write);
                }
                let mut past = index.write(first, tracking_freespace);
                past.get_opaque_mut().skip = fresh.max(past.get_opaque().skip);
                return (fresh, i);
            }
            if current == first && write.get_opaque().skip != first {
                current = write.get_opaque().skip;
            } else {
                current = write.get_opaque().next;
            }
        } else {
            if current == first && read.get_opaque().skip != first {
                current = read.get_opaque().skip;
            } else {
                current = read.get_opaque().next;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::KeyFilter;

    /// A filter of a single 2-byte key.
    fn filter(keys: Vec<u16>) -> KeyFilter {
        let set = keys.into_iter().map(u128::from).collect();
        KeyFilter::new(&crate::keys::Keys::new([(2, true)]), vec![Some(set)]).unwrap()
    }

    /// A random frozen tape whose blocks span pages: per page, its tuples as
    /// (block, whether it is the block's last), and the blocks' ranges,
    /// sorted as a build writes them.
    fn tape() -> (Vec<Vec<(usize, bool)>>, Vec<u32>) {
        let blocks = rand::random::<u32>() as usize % 100;
        let mut pages = vec![Vec::new()];
        for j in 0..blocks {
            let n = 1 + rand::random::<u32>() as usize % 4;
            for k in 0..n {
                if !pages.last().unwrap().is_empty() && rand::random::<u8>() % 3 == 0 {
                    pages.push(Vec::new());
                }
                pages.last_mut().unwrap().push((j, k == n - 1));
            }
        }
        let mut tags = (0..2 * blocks)
            .map(|_| rand::random::<u16>() % 1000)
            .collect::<Vec<_>>();
        tags.sort_unstable();
        let ranges = (0..blocks)
            .map(|j| tags[2 * j] as u32 | (tags[2 * j + 1] as u32) << 16)
            .collect();
        (pages, ranges)
    }

    /// The directory words of a tape, as `maintain` writes them.
    fn directory(pages: &[Vec<(usize, bool)>], ranges: &[u32], layout: bool) -> Vec<u32> {
        let mut x = Layout::default();
        for (i, page) in pages.iter().enumerate() {
            x.page(7000 + i as u32);
            for &(j, last) in page {
                x.tuple(last.then(|| ranges[j]));
            }
        }
        let mut words = x.ids.clone();
        if layout {
            words.extend(x.words());
        }
        words
    }

    /// The layout selects the blocks whose range meets a wanted tag and
    /// exactly the pages that hold their parts, and a reader of those pages
    /// tells every tuple's block.
    #[test]
    fn layout() {
        for _ in 0..256 {
            let (pages, ranges) = tape();
            let wanted = (0..1 + rand::random::<u32>() % 4)
                .map(|_| rand::random::<u16>() % 1000)
                .collect::<Vec<_>>();
            let filter = filter(wanted.clone());
            let meets = |r: u32| {
                let (min, max) = (r as u16, (r >> 16) as u16);
                wanted.iter().any(|&t| min <= t && t <= max)
            };
            let mut words = directory(&pages, &ranges, true);
            let n = pages.len();
            let Some(kept) = plan(&mut words, n, &filter) else {
                assert!(ranges.iter().all(|&r| meets(r)));
                continue;
            };
            let (ids, layout) = words.split_at(n);
            let (firsts, masks) = layout.split_at(n);
            for (j, &r) in ranges.iter().enumerate() {
                assert_eq!(masks[j / 32] >> (j % 32) & 1 != 0, meets(r));
            }
            let expected = (0..n)
                .filter(|&i| pages[i].iter().any(|&(j, _)| meets(ranges[j])))
                .map(|i| 7000 + i as u32)
                .collect::<Vec<_>>();
            assert_eq!(&ids[..kept], expected);
            for (&id, &first) in std::iter::zip(&ids[..kept], &firsts[..kept]) {
                let mut j = first as usize;
                for &(block, last) in &pages[(id - 7000) as usize] {
                    assert_eq!(j, block);
                    j += last as usize;
                }
            }
        }
    }

    /// Every page is read without a layout, or when every block is selected.
    #[test]
    fn layout_all() {
        for _ in 0..64 {
            let (pages, ranges) = tape();
            let n = pages.len();
            let some = filter(vec![rand::random::<u16>() % 1000]);
            let mut words = directory(&pages, &ranges, false);
            assert_eq!(plan(&mut words, n, &some), None);
            let all = filter((0..1000).collect());
            let mut words = directory(&pages, &ranges, true);
            assert_eq!(plan(&mut words, n, &all), None);
        }
    }
}
