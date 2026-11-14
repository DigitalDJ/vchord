// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::keys::Keys;
use crate::tape::TapeWriter;
use crate::tuples::*;
use crate::{Branch, Opaque};
use index::relation::{Page, RelationWrite};
use rabitq::packing::{any_pack, padding_pack};
use std::num::NonZero;

pub struct DirectoryTapeWriter<'a, R>
where
    R: RelationWrite + 'a,
{
    tape: TapeWriter<'a, R, DirectoryTuple>,
}

impl<'a, R> DirectoryTapeWriter<'a, R>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
{
    pub fn create(index: &'a R, tracking_freespace: bool) -> Self {
        Self {
            tape: TapeWriter::create(index, tracking_freespace),
        }
    }
    pub fn push(&mut self, branch: &[u32]) {
        put(
            &mut self.tape,
            branch,
            DirectoryTuple::estimate_size_0,
            DirectoryTuple::fit_1,
            |elements| DirectoryTuple::_0 {
                elements: elements.to_vec(),
            },
            |elements| DirectoryTuple::_1 {
                elements: elements.to_vec(),
            },
        );
    }
    /// Pushes the layout of the tape whose pages were pushed (`DirectoryTuple`).
    pub fn push_layout(&mut self, layout: &[u32]) {
        let f = |elements: &[u32]| DirectoryTuple::_2 {
            elements: elements.to_vec(),
        };
        let (size, fit) = (DirectoryTuple::estimate_size_0, DirectoryTuple::fit_1);
        put(&mut self.tape, layout, size, fit, f, f);
    }
    pub fn into_inner(self) -> TapeWriter<'a, R, DirectoryTuple> {
        self.tape
    }
}

pub struct H1TapeWriter<'a, R>
where
    R: RelationWrite + 'a,
{
    tape: TapeWriter<'a, R, H1Tuple>,
    branches: Vec<Branch<u32>>,
    prefetch: usize,
}

impl<'a, R> H1TapeWriter<'a, R>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
{
    pub fn create(index: &'a R, prefetch: usize, tracking_freespace: bool) -> Self {
        Self {
            tape: TapeWriter::create(index, tracking_freespace),
            branches: Vec::new(),
            prefetch,
        }
    }
    pub fn push(&mut self, branch: Branch<u32>) {
        self.branches.push(branch);
        if let Ok(chunk) = <&[_; 32]>::try_from(self.branches.as_slice()) {
            let elements =
                padding_pack(chunk.iter().map(|x| rabitq::packing::pack_to_u4(&x.code.1)));
            let prefetch = self.prefetch;
            put(
                &mut self.tape,
                &elements,
                |n| H1Tuple::estimate_size_0(prefetch, n),
                |freespace| H1Tuple::fit_1(prefetch, freespace),
                |elements| H1Tuple::_0 {
                    metadata: [
                        chunk.each_ref().map(|x| x.code.0.dis_u_2),
                        chunk.each_ref().map(|x| x.code.0.factor_cnt),
                        chunk.each_ref().map(|x| x.code.0.factor_ip),
                        chunk.each_ref().map(|x| x.code.0.factor_err),
                    ],
                    delta: chunk.each_ref().map(|x| x.delta),
                    prefetch: fix_good(chunk.each_ref().map(|x| x.prefetch.as_slice())),
                    norm: chunk.each_ref().map(|x| x.norm),
                    head: chunk.each_ref().map(|x| x.head),
                    first: chunk.each_ref().map(|x| x.extra),
                    len: chunk.len() as _,
                    elements: elements.to_vec(),
                },
                |elements| H1Tuple::_1 {
                    elements: elements.to_vec(),
                },
            );
            self.branches.clear();
        }
    }
    pub fn into_inner(self) -> (TapeWriter<'a, R, H1Tuple>, Vec<Branch<u32>>) {
        (self.tape, self.branches)
    }
    pub fn flush(tape: &mut TapeWriter<'_, R, H1Tuple>, prefetch: usize, chunk: Vec<Branch<u32>>) {
        if chunk.is_empty() {
            return;
        }
        let elements = padding_pack(chunk.iter().map(|x| rabitq::packing::pack_to_u4(&x.code.1)));
        put(
            tape,
            &elements,
            |n| H1Tuple::estimate_size_0(prefetch, n),
            |freespace| H1Tuple::fit_1(prefetch, freespace),
            |elements| H1Tuple::_0 {
                metadata: [
                    any_pack(chunk.iter().map(|x| x.code.0.dis_u_2)),
                    any_pack(chunk.iter().map(|x| x.code.0.factor_cnt)),
                    any_pack(chunk.iter().map(|x| x.code.0.factor_ip)),
                    any_pack(chunk.iter().map(|x| x.code.0.factor_err)),
                ],
                delta: any_pack(chunk.iter().map(|x| x.delta)),
                prefetch: fix_bad(chunk.iter().map(|x| x.prefetch.as_slice())),
                head: any_pack(chunk.iter().map(|x| x.head)),
                norm: any_pack(chunk.iter().map(|x| x.norm)),
                first: any_pack(chunk.iter().map(|x| x.extra)),
                len: chunk.len() as _,
                elements: elements.to_vec(),
            },
            |elements| H1Tuple::_1 {
                elements: elements.to_vec(),
            },
        );
    }
}

pub struct FrozenTapeWriter<'a, R>
where
    R: RelationWrite + 'a,
{
    tape: TapeWriter<'a, R, FrozenTuple>,
    branches: Vec<Branch<NonZero<u64>>>,
    prefetch: usize,
    keys: &'a Keys,
    /// The filter key columns of the branches, a row's after another's.
    columns: Vec<u8>,
}

impl<'a, R> FrozenTapeWriter<'a, R>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
{
    /// `keys`: the filter keys of the index (`crate::keys`).
    pub fn create(index: &'a R, prefetch: usize, keys: &'a Keys, tracking_freespace: bool) -> Self {
        Self {
            tape: TapeWriter::create(index, tracking_freespace),
            branches: Vec::new(),
            prefetch,
            keys,
            columns: Vec::new(),
        }
    }
    /// `columns`: the branch's filter key columns, as an appendable tuple
    /// stores them.
    pub fn push(&mut self, branch: Branch<NonZero<u64>>, columns: &[u8]) {
        self.branches.push(branch);
        self.columns
            .extend_from_slice(&columns[..self.keys.columns()]);
        if let Ok(chunk) = <&[_; 32]>::try_from(self.branches.as_slice()) {
            let elements =
                padding_pack(chunk.iter().map(|x| rabitq::packing::pack_to_u4(&x.code.1)));
            let columns = self.keys.block(&self.columns);
            let (prefetch, width) = (self.prefetch, columns.len());
            put(
                &mut self.tape,
                &elements,
                |n| FrozenTuple::estimate_size_0(prefetch, n, width),
                |freespace| FrozenTuple::fit_1(prefetch, freespace),
                |elements| FrozenTuple::_0 {
                    metadata: [
                        chunk.each_ref().map(|x| x.code.0.dis_u_2),
                        chunk.each_ref().map(|x| x.code.0.factor_cnt),
                        chunk.each_ref().map(|x| x.code.0.factor_ip),
                        chunk.each_ref().map(|x| x.code.0.factor_err),
                    ],
                    delta: chunk.each_ref().map(|x| x.delta),
                    prefetch: fix_good(chunk.each_ref().map(|x| x.prefetch.as_slice())),
                    head: chunk.each_ref().map(|x| x.head),
                    payload: chunk.each_ref().map(|x| Some(x.extra)),
                    elements: elements.to_vec(),
                    columns,
                },
                |elements| FrozenTuple::_1 {
                    elements: elements.to_vec(),
                },
            );
            self.branches.clear();
            self.columns.clear();
        }
    }
    /// The tape, and the branches of a block not filled, with their filter
    /// key columns, a row's after another's.
    pub fn into_inner(
        self,
    ) -> (
        TapeWriter<'a, R, FrozenTuple>,
        Vec<Branch<NonZero<u64>>>,
        Vec<u8>,
    ) {
        (self.tape, self.branches, self.columns)
    }
}

/// Writes a tuple of `elements` that may not fit the head page whole: the
/// tuple (`whole`, with the elements left) if it fits, or else as many
/// elements as fit, in a part (`part`) of their own. A part without elements
/// would only take the room of the page's tail: the tuple goes on on the next
/// page instead.
fn put<R, T, E>(
    tape: &mut TapeWriter<'_, R, T>,
    elements: &[E],
    size: impl Fn(usize) -> usize,
    fit: impl Fn(u16) -> Option<usize>,
    whole: impl FnOnce(&[E]) -> T,
    part: impl Fn(&[E]) -> T,
) where
    R: RelationWrite,
    R::Page: Page<Opaque = Opaque>,
    T: Tuple,
{
    let mut remain = elements;
    loop {
        let freespace = tape.freespace();
        if size(remain.len()) <= freespace as usize {
            tape.tape_put(whole(remain));
            return;
        }
        match fit(freespace).map(|w| w.min(remain.len())) {
            Some(w) if w > 0 => {
                let (left, right) = remain.split_at(w);
                tape.tape_put(part(left));
                remain = right;
            }
            _ => tape.tape_move(),
        }
    }
}

fn fix_good<'a>(into_iter: impl IntoIterator<Item = &'a [u32]>) -> Vec<[u32; 32]> {
    let slices = into_iter.into_iter().collect::<Vec<_>>();
    if slices.len() != 32 {
        panic!("too many or too few slices");
    }
    let min = slices.iter().map(|x| x.len()).min().unwrap_or_default();
    let max = slices.iter().map(|x| x.len()).max().unwrap_or_default();
    if min != max {
        panic!("the number of pages for prefetching is not a constant");
    }
    let flattened = slices.into_iter().flatten().copied().collect::<Vec<_>>();
    let (arrays, remainder) = flattened.as_chunks::<32>();
    assert!(remainder.is_empty());
    arrays.to_vec()
}

fn fix_bad<'a>(into_iter: impl IntoIterator<Item = &'a [u32]>) -> Vec<[u32; 32]> {
    let mut slices = into_iter.into_iter().collect::<Vec<_>>();
    if slices.len() > 32 {
        panic!("too many slices");
    }
    let min = slices.iter().map(|x| x.len()).min().unwrap_or_default();
    let max = slices.iter().map(|x| x.len()).max().unwrap_or_default();
    if min != max {
        panic!("the number of pages for prefetching is not a constant");
    }
    let temp = vec![u32::MAX; max];
    slices.resize(32, temp.as_slice());
    let flattened = slices.into_iter().flatten().copied().collect::<Vec<_>>();
    let (arrays, remainder) = flattened.as_chunks::<32>();
    assert!(remainder.is_empty());
    arrays.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use index::relation::{
        PageGuard, Relation, RelationRead, RelationReadTypes, RelationWriteTypes,
    };
    use std::cell::{Ref, RefCell, RefMut};
    use std::ops::{Deref, DerefMut};

    /// A page that counts its free space as a PostgreSQL page of 8 KiB does,
    /// with an 8-byte special space.
    struct MemoryPage {
        opaque: Opaque,
        items: Vec<Option<Vec<u64>>>,
    }

    impl Page for MemoryPage {
        type Opaque = Opaque;
        fn get_opaque(&self) -> &Opaque {
            &self.opaque
        }
        fn get_opaque_mut(&mut self) -> &mut Opaque {
            &mut self.opaque
        }
        fn len(&self) -> u16 {
            self.items.len() as u16
        }
        fn get(&self, i: u16) -> Option<&[u8]> {
            use zerocopy::IntoBytes;
            self.items
                .get(i as usize - 1)?
                .as_deref()
                .map(|x| x.as_bytes())
        }
        fn get_mut(&mut self, i: u16) -> Option<&mut [u8]> {
            use zerocopy::IntoBytes;
            let item = self.items.get_mut(i as usize - 1)?.as_deref_mut();
            item.map(|x| x.as_mut_bytes())
        }
        fn alloc(&mut self, data: &[u8]) -> Option<u16> {
            use zerocopy::IntoBytes;
            if data.len() > self.freespace() as usize {
                return None;
            }
            let mut item = vec![0_u64; data.len().div_ceil(8)];
            item.as_mut_bytes()[..data.len()].copy_from_slice(data);
            self.items.push(Some(item));
            Some(self.len())
        }
        fn free(&mut self, i: u16) {
            self.items[i as usize - 1] = None;
        }
        fn freespace(&self) -> u16 {
            let lower = 24 + 4 * self.items.len();
            let used = self
                .items
                .iter()
                .flatten()
                .map(|x| 8 * x.len())
                .sum::<usize>();
            (8192 - 8 - used).saturating_sub(lower + 4) as u16
        }
        fn clear(&mut self, opaque: Opaque) {
            self.opaque = opaque;
            self.items.clear();
        }
    }

    struct Guard<T>(u32, T);

    impl<T> PageGuard for Guard<T> {
        fn id(&self) -> u32 {
            self.0
        }
    }

    impl<T: Deref<Target = MemoryPage>> Deref for Guard<T> {
        type Target = MemoryPage;
        fn deref(&self) -> &MemoryPage {
            &self.1
        }
    }

    impl<T: DerefMut<Target = MemoryPage>> DerefMut for Guard<T> {
        fn deref_mut(&mut self) -> &mut MemoryPage {
            &mut self.1
        }
    }

    /// Pages in boxes, which stay put as more are added and are never
    /// removed.
    #[allow(clippy::vec_box)]
    #[derive(Default)]
    struct MemoryRelation(RefCell<Vec<Box<RefCell<MemoryPage>>>>);

    impl MemoryRelation {
        fn page(&self, id: u32) -> &RefCell<MemoryPage> {
            let page: *const RefCell<MemoryPage> = &*self.0.borrow()[id as usize];
            // Safety: the box lives as long as the relation
            #[allow(unsafe_code)]
            unsafe {
                &*page
            }
        }
    }

    impl Relation for MemoryRelation {
        type Page = MemoryPage;
    }

    impl RelationReadTypes for MemoryRelation {
        type ReadGuard<'b> = Guard<Ref<'b, MemoryPage>>;
    }

    impl RelationRead for MemoryRelation {
        fn read(&self, id: u32) -> Self::ReadGuard<'_> {
            Guard(id, self.page(id).borrow())
        }
    }

    impl RelationWriteTypes for MemoryRelation {
        type WriteGuard<'b> = Guard<RefMut<'b, MemoryPage>>;
    }

    impl RelationWrite for MemoryRelation {
        fn write(&self, id: u32, _: bool) -> Self::WriteGuard<'_> {
            Guard(id, self.page(id).borrow_mut())
        }
        fn extend(&self, opaque: Opaque, _: bool) -> Self::WriteGuard<'_> {
            let page = MemoryPage {
                opaque,
                items: Vec::new(),
            };
            let id = {
                let mut pages = self.0.borrow_mut();
                pages.push(Box::new(RefCell::new(page)));
                pages.len() as u32 - 1
            };
            self.write(id, false)
        }
        fn search(&self, _: usize) -> Option<Self::WriteGuard<'_>> {
            None
        }
        fn commit_pair(&self, _: Self::WriteGuard<'_>, _: Self::WriteGuard<'_>) {}
    }

    /// A branch of `dim` signs.
    fn branch<T>(i: usize, dim: usize, extra: T) -> Branch<T> {
        Branch {
            code: (
                rabitq::bit::CodeMetadata::from_array([1.0; 4]),
                (0..dim).map(|j| (i + j) % 3 == 0).collect(),
            ),
            delta: 0.0,
            prefetch: vec![i as u32],
            head: 0,
            norm: 0.0,
            extra,
        }
    }

    /// Every part of a frozen block carries elements: a page's tail too small
    /// for the block's last part (`_0`) is left empty, not filled with empty
    /// parts (`_1`), and every row reads back.
    #[test]
    fn frozen_parts() {
        for dim in [64, 768, 1000, 1536, 3000] {
            let relation = MemoryRelation::default();
            let keys = Keys::new([]);
            let mut tape = FrozenTapeWriter::create(&relation, 1, &keys, false);
            let n = 32 * 40;
            for i in 0..n {
                tape.push(branch(i, dim, NonZero::new(1 + i as u64).unwrap()), &[]);
            }
            let first = tape.into_inner().0.first();
            let mut rows = 0;
            for guard in crate::tape::by_next(&relation, first) {
                for i in 1..=guard.len() {
                    match FrozenTuple::deserialize_ref(guard.get(i).unwrap()) {
                        FrozenTupleReader::_0(tuple) => rows += tuple.payload().len(),
                        FrozenTupleReader::_1(tuple) => {
                            assert!(!tuple.elements().is_empty(), "dim {dim}: an empty part")
                        }
                    }
                }
            }
            assert_eq!(rows, n);
        }
    }

    /// The same for the tuples of upper levels, whole blocks and a last one
    /// of fewer rows (`flush`).
    #[test]
    fn h1_parts() {
        for dim in [64, 768, 1000, 1536, 3000] {
            let relation = MemoryRelation::default();
            let mut tape = H1TapeWriter::create(&relation, 1, false);
            let n = 32 * 40 + 7;
            for i in 0..n {
                tape.push(branch(i, dim, i as u32));
            }
            let (mut tape, chunk) = tape.into_inner();
            H1TapeWriter::flush(&mut tape, 1, chunk);
            let first = { tape }.first();
            let mut rows = 0;
            for guard in crate::tape::by_next(&relation, first) {
                for i in 1..=guard.len() {
                    match H1Tuple::deserialize_ref(guard.get(i).unwrap()) {
                        H1TupleReader::_0(tuple) => rows += tuple.len() as usize,
                        H1TupleReader::_1(tuple) => {
                            assert!(!tuple.elements().is_empty(), "dim {dim}: an empty part")
                        }
                    }
                }
            }
            assert_eq!(rows, n);
        }
    }

    /// And for directories, whose parts all have the header of the last, so
    /// that the last fits wherever a part with its elements does.
    #[test]
    fn directory_parts() {
        for n in [1, 100, 2040, 2041, 5000, 20000] {
            let relation = MemoryRelation::default();
            let mut tape = DirectoryTapeWriter::create(&relation, false);
            let ids = (0..n).collect::<Vec<u32>>();
            tape.push(&ids);
            tape.push(&ids);
            tape.push_layout(&ids);
            let first = tape.into_inner().first();
            let mut read = Vec::new();
            for guard in crate::tape::by_next(&relation, first) {
                for i in 1..=guard.len() {
                    let elements = match DirectoryTuple::deserialize_ref(guard.get(i).unwrap()) {
                        DirectoryTupleReader::_0(tuple) => tuple.elements(),
                        DirectoryTupleReader::_1(tuple) | DirectoryTupleReader::_2(tuple) => {
                            assert!(!tuple.elements().is_empty(), "{n}: an empty part");
                            tuple.elements()
                        }
                    };
                    read.extend_from_slice(elements);
                }
            }
            assert_eq!(
                read,
                [ids.as_slice(), ids.as_slice(), ids.as_slice()].concat()
            );
        }
    }
}
