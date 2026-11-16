// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

/// Calls `$search` with `$args` and the prefetchers that `$io` asks for.
macro_rules! prefetched {
    ($index:expr, $io:expr, $search:path, ($($args:expr),* $(,)?)) => {{
        use $crate::index::scanners::Io;
        use $crate::index::vchordrq::dispatch::*;
        let index = $index;
        let vectors = MakeH1PlainPrefetcher { index };
        match $io {
            Io::Plain => $search($($args,)* vectors, MakeH0PlainPrefetcher { index }),
            Io::Simple => $search($($args,)* vectors, MakeH0SimplePrefetcher { index }),
            Io::Stream => {
                let hints = ::index::relation::Hints::default().full(true);
                $search($($args,)* vectors, MakeH0StreamPrefetcher { index, hints })
            }
        }
    }};
}

mod default;
mod maxsim;

use crate::index::scanners::Io;
use simd::f16;
use vchordrq::types::OwnedVector;
use vector::rabitq4::Rabitq4Owned;
use vector::rabitq8::Rabitq8Owned;
use vector::vect::VectOwned;

pub use default::DefaultBuilder;
pub use maxsim::MaxsimBuilder;

#[derive(Debug)]
pub struct SearchOptions {
    pub search: vchordrq::SearchParams,
    pub max_scan_tuples: Option<u32>,
    pub maxsim_refine: u32,
    pub maxsim_threshold: u32,
    pub io_search: Io,
    pub io_rerank: Io,
    pub prefilter: bool,
    pub table: crate::index::fetcher::TablePrefetch,
}

/// An opfamily's own vector type, taken out of an `OwnedVector`.
pub trait FromOwned {
    fn from_owned(vector: OwnedVector) -> Self;
}

impl FromOwned for VectOwned<f32> {
    fn from_owned(vector: OwnedVector) -> Self {
        let OwnedVector::Vecf32(vector) = vector else {
            unreachable!()
        };
        vector
    }
}

impl FromOwned for VectOwned<f16> {
    fn from_owned(vector: OwnedVector) -> Self {
        let OwnedVector::Vecf16(vector) = vector else {
            unreachable!()
        };
        vector
    }
}

impl FromOwned for Rabitq8Owned {
    fn from_owned(vector: OwnedVector) -> Self {
        let OwnedVector::Rabitq8(vector) = vector else {
            unreachable!()
        };
        vector
    }
}

impl FromOwned for Rabitq4Owned {
    fn from_owned(vector: OwnedVector) -> Self {
        let OwnedVector::Rabitq4(vector) = vector else {
            unreachable!()
        };
        vector
    }
}
