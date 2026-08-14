//! The `concurrent_limit` adaptor must limit the concurrency of each chained rayon method, while
//! leaving parallelism *within* the chained operation unrestricted.
//!
//! Each test runs in a dedicated thread pool, so the assertions hold regardless of the host's core
//! count and of what else is running alongside them.

mod common;

use core::time;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::ConcurrentLimit;

use common::{pool, Concurrency as Gauge};

const LEN: usize = 10;
const DUR: time::Duration = time::Duration::from_millis(10);

/// The concurrency limits to exercise for each method.
const LIMITS: &[usize] = &[1, 2, 4];

/// Tracks the concurrency of an operation and of parallel work nested inside it.
#[derive(Default)]
struct Concurrency {
    outer: Gauge,
    inner: Gauge,
}

impl Concurrency {
    /// Simulate an expensive operation which itself uses rayon.
    fn record(&self) {
        self.outer.record(|| {
            std::thread::sleep(DUR);
            (0..LEN)
                .into_par_iter()
                .for_each(|_| self.inner.record(|| std::thread::sleep(DUR)));
        });
    }

    /// The operation must have reached the limit exactly, and nested parallel work must not have
    /// been restricted by it.
    fn assert_limited_to(&self, concurrent_limit: usize) {
        assert_eq!(
            self.outer.max(),
            concurrent_limit,
            "concurrency of the limited operation"
        );
        assert!(
            self.inner.max() > concurrent_limit,
            "nested parallel work should not be limited (reached {}, limit {concurrent_limit})",
            self.inner.max()
        );
    }
}

/// Exercise `chain` at every limit, in a thread pool dedicated to this test.
fn each_limit(chain: impl Fn(usize, &Concurrency) + Sync) {
    pool().install(|| {
        for &limit in LIMITS {
            let concurrency = Concurrency::default();
            chain(limit, &concurrency);
            concurrency.assert_limited_to(limit);
        }
    });
}

#[test]
fn map() {
    each_limit(|limit, concurrency| {
        let output = (0..LEN)
            .into_par_iter()
            .concurrent_limit(limit)
            .map(|i| {
                concurrency.record();
                i * 2
            })
            .collect::<Vec<_>>();
        assert_eq!(output, (0..LEN).map(|i| i * 2).collect::<Vec<_>>());
    });
}

#[test]
fn for_each() {
    each_limit(|limit, concurrency| {
        (0..LEN)
            .into_par_iter()
            .concurrent_limit(limit)
            .for_each(|_| concurrency.record());
    });
}

#[test]
fn try_for_each() {
    each_limit(|limit, concurrency| {
        let result = (0..LEN)
            .into_par_iter()
            .concurrent_limit(limit)
            .try_for_each(|_| -> Result<(), std::io::Error> {
                concurrency.record();
                Ok(())
            });
        assert!(result.is_ok());
    });
}

#[test]
fn filter() {
    each_limit(|limit, concurrency| {
        let output = (0..LEN)
            .into_par_iter()
            .concurrent_limit(limit)
            .filter(|i| {
                concurrency.record();
                i % 2 == 0
            })
            .collect::<Vec<_>>();
        assert_eq!(output, (0..LEN).filter(|i| i % 2 == 0).collect::<Vec<_>>());
    });
}

#[test]
fn filter_map() {
    each_limit(|limit, concurrency| {
        let output = (0..LEN)
            .into_par_iter()
            .concurrent_limit(limit)
            .filter_map(|i| {
                concurrency.record();
                (i % 2 == 0).then_some(i)
            })
            .collect::<Vec<_>>();
        assert_eq!(output, (0..LEN).filter(|i| i % 2 == 0).collect::<Vec<_>>());
    });
}

#[test]
fn update() {
    each_limit(|limit, concurrency| {
        let output = (0..LEN)
            .into_par_iter()
            .concurrent_limit(limit)
            .update(|i| {
                concurrency.record();
                *i *= 2;
            })
            .collect::<Vec<_>>();
        assert_eq!(output, (0..LEN).map(|i| i * 2).collect::<Vec<_>>());
    });
}

#[test]
fn any() {
    each_limit(|limit, concurrency| {
        // A predicate that never matches, so no work item exits early.
        let any = (0..LEN).into_par_iter().concurrent_limit(limit).any(|_| {
            concurrency.record();
            false
        });
        assert!(!any);
    });
}

#[test]
fn all() {
    each_limit(|limit, concurrency| {
        // A predicate that always holds, so no work item exits early.
        let all = (0..LEN).into_par_iter().concurrent_limit(limit).all(|_| {
            concurrency.record();
            true
        });
        assert!(all);
    });
}
