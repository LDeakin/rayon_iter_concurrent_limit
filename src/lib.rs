//! Limit the concurrency of a rayon parallel iterator.
//!
//! # Example
//! The [`concurrent_limit`](ConcurrentLimit::concurrent_limit) method of the
//! [`ConcurrentLimit`] extension trait limits the concurrency of everything chained after it.
//! ```rust
//! use rayon::iter::{IntoParallelIterator, ParallelIterator};
//! use rayon_iter_concurrent_limit::ConcurrentLimit;
//! const N: usize = 1000;
//! let output = (0..100)
//!     .into_par_iter()
//!     .concurrent_limit(2) // limits everything chained after it
//!     .map(|i| {
//!         let alloc = vec![i; N]; // max of 2 concurrent allocations
//!         alloc.into_par_iter().sum::<usize>() // runs on all threads
//!     })
//!     .map(|alloc_sum| {
//!         alloc_sum / N // max of 2 concurrent executions
//!     })
//!     .collect::<Vec<usize>>();
//! assert_eq!(output, (0..100).collect::<Vec<usize>>());
//! ```
//!
//! # Motivation
//! Consider this example:
//! ```rust
//! use rayon::iter::{IntoParallelIterator, ParallelIterator};
//! let op = |_: usize| {
//!     // operation involving a large allocation
//! };
//! (0..100).into_par_iter().for_each(op);
//! ```
//! In this case, it may be necessary to limit the number of concurrent executions of `op` due to memory constraints.
//! The number of threads could be limited with [`rayon::ThreadPool::install`](rayon::ThreadPool::install) like so:
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use rayon::iter::{IntoParallelIterator, ParallelIterator};
//! # let op = |_: usize| {};
//! let thread_pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
//! thread_pool.install(|| {
//!     (0..100).into_par_iter().for_each(op);
//! });
//! # Ok(())
//! # }
//! ```
//!
//! However, this constrains more than intended and has a footgun.
//! Any parallel operations within `op` use the same thread-limited pool, and the iterator must be consumed inside the [`install`](rayon::ThreadPool::install) scope or it will not use that pool at all.
//! Calling [`install`](rayon::ThreadPool::install) internally with a different pool avoids the first problem but introduces a worse one: `op` can then yield, so multiple instances of `op` may run concurrently on a single thread, as detailed [here](https://docs.rs/rayon/1.8.1/rayon/struct.ThreadPool.html#warning-execution-order) in the [`install`](rayon::ThreadPool::install) documentation.
//!
//! # How it works
//! This crate provides [`ConcurrentLimit`], an extension trait implemented for every [`rayon::iter::IndexedParallelIterator`].
//! Its single method, [`concurrent_limit`](ConcurrentLimit::concurrent_limit), limits the concurrency of every subsequent method in the chain, while parallel operations *within* the supplied function continue to use the whole thread pool.
//!
//! Concurrency is limited by reducing the number of work items available to [`rayon`], so that a chained operation runs *sequentially* within a work item but in *parallel* across them.
//! [`concurrent_limit`](ConcurrentLimit::concurrent_limit) splits the iterator into exactly `concurrent_limit` pieces of near-equal size.
//! Nothing is allocated and items are consumed lazily.
//!
//! Reaching the limit *exactly* requires this crate to drive the iterator itself, via [`rayon::iter::plumbing`].
//! [`rayon`]'s own driver always splits a producer at its midpoint and decides whether to split with a boolean, which rounds the number of work items to a power of two; splitting proportionally to a target piece count instead hits any piece count exactly.
//!
//! [`ConcurrencyLimited`] is an [`IndexedParallelIterator`], so indexed methods such as [`zip`](IndexedParallelIterator::zip), [`enumerate`](IndexedParallelIterator::enumerate), and [`collect_into_vec`](IndexedParallelIterator::collect_into_vec) remain available.
//! Whether the limit stays *exact* depends on what is chained after it:
//! - Adaptors that merely wrap the consumer — [`map`](rayon::iter::ParallelIterator::map), [`filter`](rayon::iter::ParallelIterator::filter), [`filter_map`](rayon::iter::ParallelIterator::filter_map), [`update`](rayon::iter::ParallelIterator::update), [`inspect`](rayon::iter::ParallelIterator::inspect), [`cloned`](rayon::iter::ParallelIterator::cloned), [`copied`](rayon::iter::ParallelIterator::copied) — leave the split to [`ConcurrencyLimited`], so the limit stays exact.
//!   So do the methods that consume the iterator, such as [`for_each`](rayon::iter::ParallelIterator::for_each), [`collect`](rayon::iter::ParallelIterator::collect), [`collect_into_vec`](IndexedParallelIterator::collect_into_vec), and [`reduce`](rayon::iter::ParallelIterator::reduce).
//! - Adaptors that need a [`Producer`](rayon::iter::plumbing::Producer) of their own — [`zip`](IndexedParallelIterator::zip), [`enumerate`](IndexedParallelIterator::enumerate), [`rev`](IndexedParallelIterator::rev), [`skip`](IndexedParallelIterator::skip), [`take`](IndexedParallelIterator::take), [`step_by`](IndexedParallelIterator::step_by), [`chunks`](IndexedParallelIterator::chunks) — take over the splitting.
//!   The limit then degrades from exact to an upper bound, applied as an [`IndexedParallelIterator::with_min_len`] hint: [`rayon`] never splits below the minimum length, but because it splits by halving, the concurrency can undershoot the limit (as low as approximately half of it).
//!
//! # Interaction with `with_min/max_len` or a second `concurrent_limit`
//! A minimum length set further up the chain is a hard floor on the size of a work item, so it caps the number of work items at `len / min_len`.
//! [`concurrent_limit`](ConcurrentLimit::concurrent_limit) honours that floor, which means **the tightest constraint in the chain wins** and the limit is always an upper bound, never an override:
//! ```rust
//! # use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
//! # use rayon_iter_concurrent_limit::ConcurrentLimit;
//! // 64 items, at least 32 per work item, so at most 2 work items — not 16.
//! (0..64)
//!     .into_par_iter()
//!     .with_min_len(32)
//!     .concurrent_limit(16)
//!     .for_each(|_i| {
//!         // at most 2 concurrent executions
//!     });
//! ```
//! The same rule governs two [`concurrent_limit`](ConcurrentLimit::concurrent_limit) calls in one chain.
//! The first one degrades to its [`with_min_len`](IndexedParallelIterator::with_min_len) fallback (an adaptor downstream of it took the producer), so the operations *between* the two calls keep the tighter of the two limits:
//! ```rust
//! # use rayon::iter::{IntoParallelIterator, ParallelIterator};
//! # use rayon_iter_concurrent_limit::ConcurrentLimit;
//! let _output = (0..64)
//!     .into_par_iter()
//!     .concurrent_limit(2)
//!     .map(|i| {
//!         i // at most 2 concurrent executions, not 8
//!     })
//!     .concurrent_limit(8)
//!     .map(|i| {
//!         i // also at most 2: the tighter limit upstream still applies
//!     })
//!     .collect::<Vec<usize>>();
//! ```
//! Relaxing a limit part-way through a chain is therefore not possible; split the chain into two separate iterators instead.
//! Note that [`with_max_len`](IndexedParallelIterator::with_max_len) is *not* honoured, since asking for smaller work items is the opposite of what [`concurrent_limit`](ConcurrentLimit::concurrent_limit) is for.
//!
//! # Alternatives
//! Chunking the iterator is the closest equivalent without this crate, and is what earlier versions of this crate did:
//! ```rust
//! # use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
//! # const N: usize = 1000;
//! # let op = |i: usize| -> usize {
//! #     let alloc = vec![i; N];
//! #     alloc.into_par_iter().sum::<usize>()
//! # };
//! let sum_iter = (0..100)
//!     .into_par_iter()
//!     .chunks(100_usize.div_ceil(2))
//!     .flat_map_iter(|chunk| chunk)
//!     .map(op);
//! # let output = sum_iter
//! #     .map(|alloc_sum| -> usize {
//! #         alloc_sum / N
//! #     })
//! #     .collect::<Vec<usize>>();
//! # assert_eq!(output, (0..100).collect::<Vec<usize>>());
//! ```
//! That form allocates a [`Vec`] per chunk, produces an unindexed iterator, and can fall short of the requested concurrency because the chunk *size* is rounded up.
//!
//! [`IndexedParallelIterator::by_uniform_blocks`](https://docs.rs/rayon/latest/rayon/iter/trait.IndexedParallelIterator.html#method.by_uniform_blocks) (rayon 1.9.0) can also bound the number of concurrent executions of an operation.
//! `iterator.by_uniform_blocks(limit)` processes blocks of `limit` items sequentially, with parallelism within each block.
//! However, every block ends with a synchronisation point, so a single slow item stalls the entire pipeline at each block boundary.
//! The approach of this crate lets each of the `limit` concurrent streams proceed independently, which suits expensive operations with variable cost.

#![warn(unused_variables)]
#![warn(dead_code)]
#![deny(missing_docs)]

mod concurrency_limited;

use rayon::iter::IndexedParallelIterator;

pub use concurrency_limited::ConcurrencyLimited;

/// An extension trait which limits the concurrency of an iterator chain.
///
/// It is implemented for every [`rayon::iter::IndexedParallelIterator`].
///
/// The [crate root documentation](crate) explains the motivation for this approach, provides further details on the underlying implementation, and details its limitations.
pub trait ConcurrentLimit: IndexedParallelIterator {
    /// Limit the concurrency of every subsequent method in the iterator chain.
    ///
    /// This splits the iterator into exactly `concurrent_limit` work items, so at most
    /// `concurrent_limit` executions of any operation chained from it run concurrently.
    ///
    /// - A `concurrent_limit` of zero applies no limit, and neither does one exceeding the length of
    ///   the iterator, since there are never more work items than items.
    /// - A `concurrent_limit` of one runs chained operations sequentially, in a single [`rayon`] work item.
    /// - Parallel rayon methods executed *within* a chained operation continue to use the whole
    ///   thread pool (the global one, unless another has been installed; see [`rayon::ThreadPool`]).
    /// - The limit is an upper bound, never an override: a tighter constraint already imposed
    ///   upstream — an earlier `concurrent_limit`, or a
    ///   [`with_min_len`](IndexedParallelIterator::with_min_len) — still applies. See
    ///   [Interaction with `with_min_len` and with a second `concurrent_limit`](crate#interaction-with-with_min_len-and-with-a-second-concurrent_limit).
    ///
    /// The output is an [`IndexedParallelIterator`], so indexed methods such as [`zip`](IndexedParallelIterator::zip)
    /// and [`collect_into_vec`](IndexedParallelIterator::collect_into_vec) remain available.
    /// Whether the limit stays exact or degrades to an upper bound depends on what is chained after
    /// it; see [How it works](crate#how-it-works).
    ///
    /// # Examples
    /// The limit applies to any [`rayon`] method, such as
    /// [`try_for_each`](rayon::iter::ParallelIterator::try_for_each):
    /// ```rust
    /// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
    /// # use rayon_iter_concurrent_limit::ConcurrentLimit;
    /// let found = (0..100)
    ///     .into_par_iter()
    ///     .concurrent_limit(2)
    ///     .try_for_each(|i| if i < 100 { Some(()) } else { None });
    /// assert_eq!(found, Some(()));
    /// ```
    ///
    /// Indexed methods remain available on the output:
    /// ```rust
    /// # use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
    /// # use rayon_iter_concurrent_limit::ConcurrentLimit;
    /// let mut output = Vec::new();
    /// (0..100)
    ///     .into_par_iter()
    ///     .concurrent_limit(2)
    ///     .map(|i| i * 2)
    ///     .collect_into_vec(&mut output);
    /// assert_eq!(output, (0..100).map(|i| i * 2).collect::<Vec<usize>>());
    /// ```
    fn concurrent_limit(self, concurrent_limit: usize) -> ConcurrencyLimited<Self> {
        ConcurrencyLimited::new(concurrent_limit, self)
    }
}

impl<I: IndexedParallelIterator> ConcurrentLimit for I {}
