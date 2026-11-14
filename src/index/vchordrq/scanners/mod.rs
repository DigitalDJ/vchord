// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

/// Calls `$search` with `$args` and then the prefetchers of vectors and of
/// tuples that `$io` asks for.
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
}
