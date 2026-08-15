//! `concurrent_limit` must reach *exactly* the requested concurrency, including for limits that
//! do not evenly divide the iterator length, and must otherwise behave like the unlimited chain.
//!
//! A dedicated thread pool makes these assertions independent of the host's core count. The
//! operations sleep rather than spin, so the threads do not need dedicated cores.
//!
//! Per-method coverage, including that parallelism *within* a limited operation is unrestricted,
//! lives in `concurrent_limit.rs`.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::ConcurrentLimit;

use common::{pool, Concurrency};

const LEN: usize = 10;
/// Longer than the 10ms used elsewhere: these tests assert the concurrency *exactly*, so an
/// operation must not finish before its peers have all started.
const DUR: core::time::Duration = core::time::Duration::from_millis(20);

/// Run one execution of the tracked operation.
fn record(concurrency: &Concurrency) {
    concurrency.record(|| std::thread::sleep(DUR));
}

/// Consuming the limited iterator directly reaches the limit exactly, for every limit.
#[test]
fn consuming_is_exact() {
    pool().install(|| {
        for limit in 1..=LEN {
            let concurrency = Concurrency::default();
            (0..LEN)
                .into_par_iter()
                .concurrent_limit(limit)
                .for_each(|_| record(&concurrency));
            assert_eq!(concurrency.max(), limit, "for_each with limit {limit}");
        }
    });
}

/// Chaining an adaptor before consuming also reaches the limit exactly, for every limit.
#[test]
fn adapted_is_exact() {
    pool().install(|| {
        for limit in 1..=LEN {
            let concurrency = Concurrency::default();
            let output = (0..LEN)
                .into_par_iter()
                .concurrent_limit(limit)
                .map(|i| {
                    record(&concurrency);
                    i * 2
                })
                .collect::<Vec<_>>();
            assert_eq!(output, (0..LEN).map(|i| i * 2).collect::<Vec<_>>());
            assert_eq!(concurrency.max(), limit, "map with limit {limit}");
        }
    });
}

/// Short-circuiting operations must still stop early once the answer is known.
#[test]
fn short_circuit_still_applies() {
    let visited = AtomicUsize::new(0);
    let any = (0..1000).into_par_iter().concurrent_limit(4).any(|i| {
        visited.fetch_add(1, Ordering::SeqCst);
        i == 0
    });
    assert!(any);
    assert!(
        visited.load(Ordering::SeqCst) < 1000,
        "any should not visit every item"
    );

    let visited = AtomicUsize::new(0);
    let all = (0..1000).into_par_iter().concurrent_limit(4).all(|i| {
        visited.fetch_add(1, Ordering::SeqCst);
        i != 0
    });
    assert!(!all);
    assert!(
        visited.load(Ordering::SeqCst) < 1000,
        "all should not visit every item"
    );

    let visited = AtomicUsize::new(0);
    let result = (0..1000)
        .into_par_iter()
        .concurrent_limit(4)
        .try_for_each(|i| {
            visited.fetch_add(1, Ordering::SeqCst);
            if i == 0 {
                Err(std::io::Error::other("stop"))
            } else {
                Ok(())
            }
        });
    assert!(result.is_err());
    assert!(
        visited.load(Ordering::SeqCst) < 1000,
        "try_for_each should not visit every item"
    );
}

/// The limit composes with rayon methods of every shape.
#[test]
fn composes_with_any_method() {
    // `try_for_each` with an `Option` output, i.e. a `Try` type other than `Result`.
    let found = (0..100)
        .into_par_iter()
        .concurrent_limit(2)
        .try_for_each(|i| if i < 100 { Some(()) } else { None });
    assert_eq!(found, Some(()));

    // A fold/reduce.
    let sum = (0..100usize)
        .into_par_iter()
        .concurrent_limit(3)
        .fold(|| 0usize, |acc, i| acc + i)
        .sum::<usize>();
    assert_eq!(sum, (0..100).sum::<usize>());

    // A limit of zero applies no limit, but must still produce correct results.
    let sum = (0..100usize)
        .into_par_iter()
        .concurrent_limit(0)
        .sum::<usize>();
    assert_eq!(sum, (0..100).sum::<usize>());
}

/// The output is indexed, so indexed methods are available downstream.
#[test]
fn output_is_indexed() {
    let mut output = Vec::new();
    (0..LEN)
        .into_par_iter()
        .concurrent_limit(2)
        .map(|i| i * 2)
        .collect_into_vec(&mut output);
    assert_eq!(output, (0..LEN).map(|i| i * 2).collect::<Vec<_>>());

    let zipped = (0..LEN)
        .into_par_iter()
        .concurrent_limit(2)
        .map(|i| i * 2)
        .zip((0..LEN).into_par_iter())
        .collect::<Vec<_>>();
    assert_eq!(zipped, (0..LEN).map(|i| (i * 2, i)).collect::<Vec<_>>());

    let enumerated = (0..LEN)
        .into_par_iter()
        .concurrent_limit(2)
        .update(|i| *i *= 2)
        .enumerate()
        .rev()
        .collect::<Vec<_>>();
    assert_eq!(
        enumerated,
        (0..LEN).map(|i| (i, i * 2)).rev().collect::<Vec<_>>()
    );
}

/// When a downstream *indexed* adaptor takes the producer, the exact split cannot be applied, but
/// the limit must still be respected as an upper bound via the `with_min_len` fallback.
#[test]
fn limit_is_upper_bound_through_indexed_chaining() {
    pool().install(|| {
        for limit in 1..=8usize {
            let concurrency = Concurrency::default();
            let output = (0..LEN)
                .into_par_iter()
                .concurrent_limit(limit)
                .map(|i| {
                    record(&concurrency);
                    i
                })
                .zip((0..LEN).into_par_iter())
                .collect::<Vec<_>>();
            assert_eq!(output.len(), LEN);
            let max = concurrency.max();
            assert!(
                max <= limit,
                "zip chained after a limit of {limit} ran {max} concurrently"
            );
        }
    });
}

/// An upstream `with_min_len` forbids work items smaller than the minimum, so it caps the number
/// of work items at `len / min_len`. The limit must not split through it.
#[test]
fn upstream_with_min_len_caps_the_limit() {
    const N: usize = 32;
    pool().install(|| {
        for (min_len, expected) in [(16usize, 2usize), (8, 4), (4, 8), (1, 12)] {
            let concurrency = Concurrency::default();
            (0..N)
                .into_par_iter()
                .with_min_len(min_len)
                .concurrent_limit(12)
                .for_each(|_| record(&concurrency));
            assert_eq!(
                concurrency.max(),
                expected,
                "with_min_len({min_len}) then a limit of 12 over {N} items"
            );
        }
    });
}

/// A second `concurrent_limit` cannot *raise* a tighter limit applied earlier in the chain: the
/// operations between the two calls must keep the tighter one.
#[test]
fn a_later_limit_cannot_relax_an_earlier_one() {
    const N: usize = 16;
    pool().install(|| {
        // Tighter first: both operations must stay at the tighter limit of 2.
        let first = Concurrency::default();
        let second = Concurrency::default();
        let output = (0..N)
            .into_par_iter()
            .concurrent_limit(2)
            .map(|i| {
                record(&first);
                i
            })
            .concurrent_limit(8)
            .map(|i| {
                record(&second);
                i
            })
            .collect::<Vec<_>>();
        assert_eq!(output, (0..N).collect::<Vec<_>>());
        assert_eq!(first.max(), 2, "operation before the relaxed limit");
        assert_eq!(second.max(), 2, "operation after the relaxed limit");

        // Tighter last: the tighter limit applies to both, as it always has.
        let first = Concurrency::default();
        let second = Concurrency::default();
        (0..N)
            .into_par_iter()
            .concurrent_limit(8)
            .map(|i| {
                record(&first);
                i
            })
            .concurrent_limit(2)
            .for_each(|_| record(&second));
        assert_eq!(first.max(), 2, "operation before the tightened limit");
        assert_eq!(second.max(), 2, "operation after the tightened limit");
    });
}

/// Every item must be visited exactly once, for every limit.
#[test]
fn every_item_visited_once() {
    for limit in 0..=13usize {
        let counts: Vec<AtomicUsize> = (0..LEN).map(|_| AtomicUsize::new(0)).collect();
        (0..LEN)
            .into_par_iter()
            .concurrent_limit(limit)
            .for_each(|i| {
                counts[i].fetch_add(1, Ordering::SeqCst);
            });
        for (i, count) in counts.iter().enumerate() {
            assert_eq!(count.load(Ordering::SeqCst), 1, "item {i}, limit {limit}");
        }
    }
}

/// An empty iterator must not panic for any limit.
#[test]
fn empty_iterator() {
    for limit in 0..=4usize {
        let empty: Vec<usize> = Vec::new();
        empty
            .into_par_iter()
            .concurrent_limit(limit)
            .for_each(|_| unreachable!());
        assert!(!(0..0)
            .into_par_iter()
            .concurrent_limit(limit)
            .any(|_: usize| unreachable!()));
        assert!((0..0)
            .into_par_iter()
            .concurrent_limit(limit)
            .all(|_: usize| unreachable!()));
    }
}
