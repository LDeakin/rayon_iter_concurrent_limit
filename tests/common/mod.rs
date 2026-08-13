use std::sync::atomic::{AtomicUsize, Ordering};

/// The size of the [`pool`] used by the concurrency tests.
///
/// Comfortably above the largest limit under test, so that nested parallel work has room to exceed
/// the limit applied to its parent operation.
const THREADS: usize = 16;

/// A dedicated thread pool for a single test.
///
/// Concurrency assertions are otherwise at the mercy of the host's core count and of whatever else
/// is running: the global pool has one thread per core, so on a 2-core CI runner an operation
/// cannot reach a limit of 4 no matter how the iterator is split. A fixed-size pool makes the
/// assertions depend only on this crate.
///
/// The operations under test sleep rather than spin, so these threads do not need dedicated cores
/// and the pool can safely be larger than the machine.
pub fn pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(THREADS)
        .build()
        .unwrap()
}

/// Tracks the maximum number of concurrent executions of an operation.
#[derive(Default)]
pub struct Concurrency {
    active: AtomicUsize,
    max: AtomicUsize,
}

impl Concurrency {
    /// Run `f` as one execution of the tracked operation.
    pub fn record<T>(&self, f: impl FnOnce() -> T) -> T {
        self.active.fetch_add(1, Ordering::SeqCst);
        let output = f();
        let active = self.active.fetch_sub(1, Ordering::SeqCst);
        self.max.fetch_max(active, Ordering::SeqCst);
        output
    }

    /// The maximum number of executions observed running concurrently.
    pub fn max(&self) -> usize {
        self.max.load(Ordering::SeqCst)
    }
}
