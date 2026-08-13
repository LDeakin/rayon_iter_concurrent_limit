//! Compare concurrency limiting by chunking against the exact-split driver used by this crate.
//!
//! Chunking materialises each chunk into a `Vec` and rounds the chunk *size* up, so the number of
//! chunks — and therefore the achieved concurrency — can fall short of the requested limit. This
//! is how earlier versions of this crate limited concurrency. The exact split allocates nothing
//! and reaches the limit exactly.
//!
//! Run with `cargo run --release --example concurrency_comparison`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// A global allocator which records the allocation count and peak live bytes.
struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
        PEAK.fetch_max(live, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn reset() {
    ALLOCS.store(0, Ordering::SeqCst);
    LIVE.store(0, Ordering::SeqCst);
    PEAK.store(0, Ordering::SeqCst);
}

/// Allocations and peak live bytes during `f`.
fn measure_allocation(f: impl FnOnce()) -> (usize, usize) {
    reset();
    f();
    (ALLOCS.load(Ordering::SeqCst), PEAK.load(Ordering::SeqCst))
}

/// Concurrency limiting by chunking, for comparison.
///
/// The chunk *size* is rounded up, which rounds the chunk *count* — and so the achieved
/// concurrency — down.
fn for_each_chunked<I, OP>(concurrent_limit: usize, iterator: I, op: OP)
where
    I: IndexedParallelIterator,
    OP: Fn(I::Item) + Sync + Send,
{
    let chunk_size = if concurrent_limit == 0 {
        1
    } else {
        iterator.len().div_ceil(concurrent_limit).max(1)
    };
    iterator
        .chunks(chunk_size)
        .flat_map_iter(|chunk| chunk)
        .for_each(op);
}

/// The maximum number of concurrent executions observed while running `f`.
fn measure_concurrency(f: impl FnOnce(&(dyn Fn() + Sync))) -> usize {
    let active = AtomicUsize::new(0);
    let max = AtomicUsize::new(0);
    f(&|| {
        active.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(20));
        let n = active.fetch_sub(1, Ordering::SeqCst);
        max.fetch_max(n, Ordering::SeqCst);
    });
    max.load(Ordering::SeqCst)
}

fn main() {
    // Warm the thread pool so its allocations are not attributed to the measurements.
    (0..1000).into_par_iter().for_each(|_| {});
    let sink = AtomicUsize::new(0);

    println!("== allocation while consuming (0..len) with a concurrency limit ==");
    println!(
        "{:>9} {:>7} {:>16} {:>14} {:>16} {:>14}",
        "len", "limit", "chunked allocs", "chunked peak", "exact allocs", "exact peak"
    );
    for len in [10_000usize, 1_000_000] {
        for limit in [2usize, 3, 8] {
            let (chunk_allocs, chunk_peak) = measure_allocation(|| {
                for_each_chunked(limit, (0..len).into_par_iter(), |i| {
                    sink.fetch_add(i & 1, Ordering::Relaxed);
                });
            });
            let (exact_allocs, exact_peak) = measure_allocation(|| {
                (0..len)
                    .into_par_iter()
                    .concurrent_limit(limit)
                    .for_each(|i| {
                        sink.fetch_add(i & 1, Ordering::Relaxed);
                    });
            });
            println!(
                "{len:>9} {limit:>7} {chunk_allocs:>16} {chunk_peak:>14} {exact_allocs:>16} {exact_peak:>14}"
            );
        }
    }

    println!("\n== max concurrent executions ({} items) ==", 10);
    println!("{:>7} {:>18} {:>18}", "limit", "chunked", "exact");
    for limit in 1..=8usize {
        let chunked = measure_concurrency(|record| {
            for_each_chunked(limit, (0..10).into_par_iter(), |_| record());
        });
        let exact = measure_concurrency(|record| {
            (0..10)
                .into_par_iter()
                .concurrent_limit(limit)
                .for_each(|_| record());
        });
        let flag = if chunked < limit {
            "  <-- undershoots"
        } else {
            ""
        };
        println!("{limit:>7} {chunked:>18} {exact:>18}{flag}");
    }

    println!("\n== wall time, 2M cheap items ==");
    for limit in [2usize, 4, 8] {
        let start = Instant::now();
        for_each_chunked(limit, (0..2_000_000usize).into_par_iter(), |i| {
            sink.fetch_add(i & 1, Ordering::Relaxed);
        });
        let chunked = start.elapsed();

        let start = Instant::now();
        (0..2_000_000usize)
            .into_par_iter()
            .concurrent_limit(limit)
            .for_each(|i| {
                sink.fetch_add(i & 1, Ordering::Relaxed);
            });
        let exact = start.elapsed();

        println!("limit {limit}: chunked {chunked:>10.2?}   exact {exact:>10.2?}");
    }

    // Keep the operations from being optimised away.
    assert_ne!(sink.load(Ordering::Relaxed), 0);
}
