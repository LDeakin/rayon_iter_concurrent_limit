//! Benchmark [`ConcurrentLimit::concurrent_limit`].
//!
//! Every group runs the same workload three ways:
//! - `concurrent_limit`: the approach introduced in 0.3.0.
//! - `chunked`: the approach used by version 0.2.0 of this crate.
//!   It splits the iterator with [`IndexedParallelIterator::chunks`] and applies the operation
//!   sequentially within each chunk.
//! - `unlimited`: plain [`rayon`], with no concurrency limit at all.
#![allow(missing_docs)]

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
    Throughput,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::ConcurrentLimit;

/// The concurrency limiting approach used by version 0.2.0 of this crate, for comparison.
mod chunked {
    use rayon::iter::{Chunks, IndexedParallelIterator};

    pub fn subdivide<I: IndexedParallelIterator>(
        concurrent_limit: usize,
        iterator: I,
    ) -> Chunks<I> {
        let chunk_size = iterator.len().div_ceil(concurrent_limit).max(1);
        iterator.chunks(chunk_size)
    }
}

/// Concurrency limits to sweep.
const LIMITS: [usize; 2] = [4, 64];

#[inline]
fn cheap(i: usize) -> usize {
    i.wrapping_mul(2654435761) >> 3
}

/// Register the `chunked` and `concurrent_limit` arms for every limit in `limits`.
///
/// All arms share `len` as their [`BenchmarkId`] parameter, including the `unlimited` baseline
/// registered by the caller, so criterion lines them up as one comparison rather than as
/// unrelated series.
fn sweep_limits<R>(
    group: &mut BenchmarkGroup<WallTime>,
    len: usize,
    limits: impl IntoIterator<Item = usize>,
    chunked: impl Fn(usize) -> R,
    limited: impl Fn(usize) -> R,
) {
    for limit in limits {
        group.bench_function(BenchmarkId::new(format!("chunked/{limit}"), len), |b| {
            b.iter(|| chunked(limit));
        });
        group.bench_function(
            BenchmarkId::new(format!("concurrent_limit/{limit}"), len),
            |b| b.iter(|| limited(limit)),
        );
    }
}

/// `for_each` over cheap items: measures the per-item overhead of limiting concurrency.
///
/// NOTE: the chunked approach materialises a [`Vec`] per chunk.
fn for_each(c: &mut Criterion) {
    let mut group = c.benchmark_group("for_each");
    for len in [1_000usize, 1_000_000] {
        group.throughput(Throughput::Elements(len as u64));

        group.bench_function(BenchmarkId::new("unlimited", len), |b| {
            b.iter(|| {
                (0..len).into_par_iter().for_each(|i| {
                    std::hint::black_box(cheap(i));
                });
            });
        });

        sweep_limits(
            &mut group,
            len,
            LIMITS,
            |limit| {
                chunked::subdivide(limit, (0..len).into_par_iter()).for_each(|chunk| {
                    chunk.into_iter().for_each(|i| {
                        std::hint::black_box(cheap(i));
                    });
                });
            },
            |limit| {
                (0..len)
                    .into_par_iter()
                    .concurrent_limit(limit)
                    .for_each(|i| {
                        std::hint::black_box(cheap(i));
                    });
            },
        );
    }
    group.finish();
}

/// `map` followed by `collect`: measures the cost of losing the indexed iterator.
///
/// The chunked approach flattens with `flat_map_iter`, which yields an *unindexed* iterator and
/// `collect` goes through rayon's linked-list collect.
/// The exact split stays indexed and collects straight into a preallocated [`Vec`].
fn map_collect(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_collect");
    for len in [1_000usize, 1_000_000] {
        group.throughput(Throughput::Elements(len as u64));

        group.bench_function(BenchmarkId::new("unlimited", len), |b| {
            b.iter(|| (0..len).into_par_iter().map(cheap).collect::<Vec<usize>>());
        });

        sweep_limits(
            &mut group,
            len,
            LIMITS,
            |limit| {
                chunked::subdivide(limit, (0..len).into_par_iter())
                    .flat_map_iter(|chunk| chunk.into_iter().map(cheap))
                    .collect::<Vec<usize>>()
            },
            |limit| {
                (0..len)
                    .into_par_iter()
                    .concurrent_limit(limit)
                    .map(cheap)
                    .collect::<Vec<usize>>()
            },
        );
    }
    group.finish();
}

/// `any` with a match early in the iterator: measures short-circuiting.
///
/// The exact-split driver checks [`rayon::iter::plumbing::Consumer::full`] before splitting and
/// abandons the remaining work early.
/// The chunked approach materialises a whole chunk before the predicate sees its first item.
fn any_short_circuit(c: &mut Criterion) {
    const LEN: usize = 1_000_000;
    // Inside the first work item for every limit under test, so all arms have the same amount of
    // work to abandon.
    const NEEDLE: usize = LEN / 1000;

    let mut group = c.benchmark_group("any_short_circuit");

    group.bench_function(BenchmarkId::new("unlimited", LEN), |b| {
        b.iter(|| (0..LEN).into_par_iter().any(|i| i == NEEDLE));
    });

    sweep_limits(
        &mut group,
        LEN,
        LIMITS,
        |limit| {
            chunked::subdivide(limit, (0..LEN).into_par_iter())
                .any(|chunk| chunk.into_iter().any(|i| i == NEEDLE))
        },
        |limit| {
            (0..LEN)
                .into_par_iter()
                .concurrent_limit(limit)
                .any(|i| i == NEEDLE)
        },
    );
    group.finish();
}

/// The motivating use case: an memory intensive operation that is itself parallel.
fn nested_parallelism(c: &mut Criterion) {
    const LEN: usize = 64;
    const INNER: usize = 4096;

    let op = |i: usize| -> usize {
        let alloc = vec![i; INNER];
        alloc.into_par_iter().sum::<usize>()
    };

    let mut group = c.benchmark_group("nested_parallelism");
    group.throughput(Throughput::Elements(LEN as u64));

    group.bench_function(BenchmarkId::new("unlimited", LEN), |b| {
        b.iter(|| (0..LEN).into_par_iter().map(op).sum::<usize>());
    });

    sweep_limits(
        &mut group,
        LEN,
        [2usize, 8],
        |limit| {
            chunked::subdivide(limit, (0..LEN).into_par_iter())
                .flat_map_iter(|chunk| chunk.into_iter().map(op))
                .sum::<usize>()
        },
        |limit| {
            (0..LEN)
                .into_par_iter()
                .concurrent_limit(limit)
                .map(op)
                .sum::<usize>()
        },
    );
    group.finish();
}

criterion_group!(
    benches,
    for_each,
    map_collect,
    any_short_circuit,
    nested_parallelism
);
criterion_main!(benches);
