//! Benchmark [`ConcurrentLimit::concurrent_limit`] against the chunking approach it replaced.
//!
//! Every group runs the same workload three ways where a baseline makes sense:
//! - `chunked`: the approach used by version 0.2.0 of this crate, reproduced in the [`chunked`]
//!   module below. It splits the iterator with [`IndexedParallelIterator::chunks`] and applies the
//!   operation sequentially within each chunk.
//! - `concurrent_limit`: the exact-split driver of version 0.3.0.
//! - `unlimited`: plain [`rayon`], with no concurrency limit at all. This is the floor that a
//!   limiting strategy pays overhead against — it does not bound concurrency.
#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::ConcurrentLimit;

/// The concurrency limiting approach used by version 0.2.0 of this crate, for comparison.
mod chunked {
    use rayon::iter::{Chunks, IndexedParallelIterator};

    /// Version 0.2.0's `iter_subdivide`, which every `iter_concurrent_limit!` arm called.
    pub fn subdivide<I: IndexedParallelIterator>(
        concurrent_limit: usize,
        iterator: I,
    ) -> Chunks<I> {
        if concurrent_limit == 0 {
            iterator.chunks(1)
        } else {
            let chunk_size = iterator.len().div_ceil(concurrent_limit).max(1);
            iterator.chunks(chunk_size)
        }
    }
}

/// Concurrency limits to sweep.
const LIMITS: [usize; 2] = [4, 64];

#[inline]
fn cheap(i: usize) -> usize {
    std::hint::black_box(i).wrapping_mul(2654435761) >> 3
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

        for limit in LIMITS {
            let id = format!("{len}/{limit}");
            group.bench_function(BenchmarkId::new("chunked", &id), |b| {
                b.iter(|| {
                    chunked::subdivide(limit, (0..len).into_par_iter()).for_each(|chunk| {
                        chunk.into_iter().for_each(|i| {
                            std::hint::black_box(cheap(i));
                        });
                    });
                });
            });
            group.bench_function(BenchmarkId::new("concurrent_limit", &id), |b| {
                b.iter(|| {
                    (0..len)
                        .into_par_iter()
                        .concurrent_limit(limit)
                        .for_each(|i| {
                            std::hint::black_box(cheap(i));
                        });
                });
            });
        }
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

        for limit in LIMITS {
            let id = format!("{len}/{limit}");
            group.bench_function(BenchmarkId::new("chunked", &id), |b| {
                b.iter(|| {
                    chunked::subdivide(limit, (0..len).into_par_iter())
                        .flat_map_iter(|chunk| chunk.into_iter().map(cheap))
                        .collect::<Vec<usize>>()
                });
            });
            group.bench_function(BenchmarkId::new("concurrent_limit", &id), |b| {
                b.iter(|| {
                    (0..len)
                        .into_par_iter()
                        .concurrent_limit(limit)
                        .map(cheap)
                        .collect::<Vec<usize>>()
                });
            });
        }
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

    group.bench_function("unlimited", |b| {
        b.iter(|| (0..LEN).into_par_iter().any(|i| i == NEEDLE));
    });

    for limit in LIMITS {
        group.bench_function(BenchmarkId::new("chunked", limit), |b| {
            b.iter(|| {
                chunked::subdivide(limit, (0..LEN).into_par_iter())
                    .any(|chunk| chunk.into_iter().any(|i| i == NEEDLE))
            });
        });
        group.bench_function(BenchmarkId::new("concurrent_limit", limit), |b| {
            b.iter(|| {
                (0..LEN)
                    .into_par_iter()
                    .concurrent_limit(limit)
                    .any(|i| i == NEEDLE)
            });
        });
    }
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

    group.bench_function("unlimited", |b| {
        b.iter(|| (0..LEN).into_par_iter().map(op).sum::<usize>());
    });

    for limit in [2usize, 8] {
        group.bench_function(BenchmarkId::new("chunked", limit), |b| {
            b.iter(|| {
                chunked::subdivide(limit, (0..LEN).into_par_iter())
                    .flat_map_iter(|chunk| chunk.into_iter().map(op))
                    .sum::<usize>()
            });
        });
        group.bench_function(BenchmarkId::new("concurrent_limit", limit), |b| {
            b.iter(|| {
                (0..LEN)
                    .into_par_iter()
                    .concurrent_limit(limit)
                    .map(op)
                    .sum::<usize>()
            });
        });
    }
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
