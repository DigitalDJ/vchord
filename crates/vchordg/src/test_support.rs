// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

// Real serialized graph pages with deterministic interleaving at the write-lock boundary.
use crate::Opaque;
use crate::operator::Op;
use crate::tuples::*;
use distance::Distance;
use index::relation::{
    Page, PageGuard, Relation, RelationRead, RelationReadTypes, RelationWrite, RelationWriteTypes,
};
use index::test_support::{MemoryPage, MemoryRelation};
use index_accessor::L2S;
use std::cell::{Cell, RefCell};
use std::num::{NonZero, Wrapping};
use vector::vect::VectOwned;

pub(crate) type TestOp = Op<VectOwned<f32>, L2S>;
pub(crate) struct TestRelation {
    pages: MemoryRelation<Opaque>,
    pub inject: Cell<bool>,
    pub conflicts: Cell<usize>,
    /// `(first, second)` of every `commit_pair`, in call order.
    pub pairs: RefCell<Vec<(u32, u32)>>,
}
impl TestRelation {
    fn new(pages: Vec<MemoryPage<Opaque>>, inject: bool) -> Self {
        Self {
            pages: MemoryRelation::new(pages),
            inject: Cell::new(inject),
            conflicts: Cell::new(0),
            pairs: RefCell::new(Vec::new()),
        }
    }
    pub(crate) fn with_full_page(self, id: u32) -> Self {
        self.pages.page(id).borrow_mut().full = true;
        self
    }
}
impl Relation for TestRelation {
    type Page = MemoryPage<Opaque>;
}
impl RelationReadTypes for TestRelation {
    type ReadGuard<'b> = <MemoryRelation<Opaque> as RelationReadTypes>::ReadGuard<'b>;
}
impl RelationRead for TestRelation {
    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        self.pages.read(id)
    }
}
impl RelationWriteTypes for TestRelation {
    type WriteGuard<'b> = <MemoryRelation<Opaque> as RelationWriteTypes>::WriteGuard<'b>;
}
impl RelationWrite for TestRelation {
    fn commit_pair(&self, first: Self::WriteGuard<'_>, second: Self::WriteGuard<'_>) {
        self.pairs.borrow_mut().push((first.id(), second.id()));
    }

    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        let mut page = self.pages.write(id, tracking_freespace);
        // Emulate intervening successful list writes before taking the write
        // lock: the observed list version is now stale. No thread scheduling.
        if self.inject.get() && id == 2 {
            let VectorTupleWriter::_0(mut tuple) =
                VectorTuple::<VectOwned<f32>>::deserialize_mut(page.get_mut(1).unwrap())
            else {
                panic!()
            };
            *tuple.version() += 1;
            self.conflicts.set(self.conflicts.get() + 1);
        }
        page
    }
    fn extend(&self, opaque: Opaque, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        self.pages.extend(opaque, tracking_freespace)
    }
    fn search(&self, _: usize) -> Option<Self::WriteGuard<'_>> {
        None
    }
}

fn page(link: u32, next: u32, items: Vec<Vec<u8>>) -> MemoryPage<Opaque> {
    let mut page = MemoryPage::new(Opaque { link, next });
    for item in items {
        page.alloc(&item).unwrap();
    }
    page
}
pub(crate) fn fixture(inject: bool) -> TestRelation {
    let meta = MetaTuple {
        dim: 2,
        bits: 1,
        m: 4,
        alpha: vec![1.2],
        ef_construction: 64,
        beam_construction: 1,
        start: OptionPointer::some((1, 1)),
        skip: 1,
    }
    .serialize();
    let vertices = (1..=2)
        .map(|i| {
            VertexTuple {
                metadata: [0.; 3],
                elements: vec![0],
                payload: NonZero::new(i),
                pointers: vec![Pointer::new((i as u32 + 1, 1))],
            }
            .serialize()
        })
        .collect();
    let u = VectorTuple::<VectOwned<f32>>::_0 {
        payload: NonZero::new(1),
        elements: vec![0., 0.],
        metadata: (),
        neighbours: vec![OptionNeighbour::NONE; 4],
        version: Wrapping(0),
    }
    .serialize();
    let t = VectorTuple::<VectOwned<f32>>::_0 {
        payload: NonZero::new(2),
        elements: vec![1., 0.],
        metadata: (),
        neighbours: vec![
            OptionNeighbour::some((1, 1), Distance::from_f32(1.)),
            OptionNeighbour::NONE,
            OptionNeighbour::NONE,
            OptionNeighbour::NONE,
        ],
        version: Wrapping(0),
    }
    .serialize();
    TestRelation::new(
        vec![
            page(1, u32::MAX, vec![meta]),
            page(2, u32::MAX, vertices),
            page(u32::MAX, 3, vec![u]),
            page(u32::MAX, u32::MAX, vec![t]),
        ],
        inject,
    )
}

/// One real vector page per vertex; IDs are (1, position + 1).
pub(crate) fn graph(vertices: &[(bool, &[u16])], m: u32) -> TestRelation {
    let meta = MetaTuple {
        dim: 2,
        bits: 1,
        m,
        alpha: vec![1.2],
        ef_construction: 64,
        beam_construction: 1,
        start: OptionPointer::some((1, 1)),
        skip: 1,
    }
    .serialize();
    let mut vector_pages = Vec::new();
    let mut vertex_tuples = Vec::new();
    for (n, &(live, edges)) in vertices.iter().enumerate() {
        let payload = live.then(|| NonZero::new(n as u64 + 1).unwrap());
        let mut neighbours = edges
            .iter()
            .map(|&v| {
                OptionNeighbour::some(
                    (1, v),
                    Distance::from_f32((n as f32 + 1.0 - v as f32).powi(2)),
                )
            })
            .collect::<Vec<_>>();
        assert!(neighbours.len() <= m as usize);
        neighbours.resize(m as usize, OptionNeighbour::NONE);
        let vector = VectorTuple::<VectOwned<f32>>::_0 {
            payload,
            elements: vec![n as f32, 1.0],
            metadata: (),
            neighbours,
            version: Wrapping(0),
        }
        .serialize();
        vector_pages.push(page(
            u32::MAX,
            if n + 1 == vertices.len() {
                u32::MAX
            } else {
                n as u32 + 3
            },
            vec![vector],
        ));
        vertex_tuples.push(
            VertexTuple {
                metadata: [0.; 3],
                elements: vec![0],
                payload,
                pointers: vec![Pointer::new((n as u32 + 2, 1))],
            }
            .serialize(),
        );
    }
    let mut pages = vec![
        page(1, u32::MAX, vec![meta]),
        page(2, u32::MAX, vertex_tuples),
    ];
    pages.extend(vector_pages);
    TestRelation::new(pages, false)
}
