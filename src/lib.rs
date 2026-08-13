//! Limit the concurrency of a rayon parallel iterator.
//!
//! # Example
//! The [`concurrent_limit`](ParallelIteratorConcurrentLimit::concurrent_limit) method of the
//! [`ParallelIteratorConcurrentLimit`] extension trait limits the concurrency of everything chained after it.
//! ```rust
//! use rayon::iter::{IntoParallelIterator, ParallelIterator};
//! use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;
//! const N: usize = 1000;
//! let output = (0..100)
//!     .into_par_iter()
//!     .concurrent_limit(2)
//!     .map(|i| {
//!         let alloc = vec![i; N]; // max 2 concurrent allocations in this example
//!         alloc.into_par_iter().sum::<usize>() // runs on all threads
//!     })
//!     .map(|alloc_sum| {
//!         alloc_sum / N // also runs with a max of 2 concurrent executions
//!     })
//!     .collect::<Vec<usize>>();
//! assert_eq!(output, (0..100).collect::<Vec<usize>>());
//! ```
//! The closest equivalent without this crate is to chunk the iterator, which is what earlier
//! versions of this crate did:
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
//! That form allocates a [`Vec`] per chunk, produces an unindexed iterator, and can fall short of
//! the requested concurrency because the chunk *size* is rounded up.
//! [`concurrent_limit`](ParallelIteratorConcurrentLimit::concurrent_limit) has none of those
//! drawbacks; see [Implementation](#implementation).
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
//! However, this has some limitations and footguns:
//! - Any parallel operations within `op` will use the same thread-limited thread pool, unless [`install`](rayon::ThreadPool::install) is called internally with a different thread pool.
//! - If [`install`](rayon::ThreadPool::install) is called internally, `op` can yield and multiple instances of `op` may run concurrently on a thread. This is detailed [here](https://docs.rs/rayon/1.8.1/rayon/struct.ThreadPool.html#warning-execution-order) in the [`install`](rayon::ThreadPool::install) documentation.
//! - An iterator must be consumed in the [`install`](rayon::ThreadPool::install) scope of a [`ThreadPool`](rayon::ThreadPool), otherwise it will not use that thread pool.
//!
//! # Solution
//! This crate provides [`ParallelIteratorConcurrentLimit`], an extension trait for [`rayon::iter::IndexedParallelIterator`].
//! Its single method, [`concurrent_limit`](ParallelIteratorConcurrentLimit::concurrent_limit), limits the concurrency of every subsequent method in the chain, while parallel operations *within* the supplied function continue to use the whole thread pool.
//!
//! ### Implementation
//! Concurrency is limited by reducing the number of work items available to [`rayon`], so that a chained operation runs *sequentially* within a work item but in *parallel* across them.
//! [`concurrent_limit`](ParallelIteratorConcurrentLimit::concurrent_limit) splits the iterator into exactly `concurrent_limit` pieces of near-equal size.
//! Nothing is allocated and items are consumed lazily.
//!
//! Reaching the limit *exactly* requires this crate to drive the iterator itself, via [`rayon::iter::plumbing`].
//! [`rayon`]'s own driver always splits a producer at its midpoint and decides whether to split with a boolean, which rounds the number of work items to a power of two; splitting proportionally to a target piece count instead hits any piece count exactly.
//!
//! [`ConcurrentLimit`] is an [`IndexedParallelIterator`], so indexed methods such as [`zip`](IndexedParallelIterator::zip), [`enumerate`](IndexedParallelIterator::enumerate), and [`collect_into_vec`](IndexedParallelIterator::collect_into_vec) remain available.
//! Whether the limit stays *exact* depends on what is chained after it:
//! - Adaptors that merely wrap the consumer — [`map`](rayon::iter::ParallelIterator::map), [`filter`](rayon::iter::ParallelIterator::filter), [`filter_map`](rayon::iter::ParallelIterator::filter_map), [`update`](rayon::iter::ParallelIterator::update), [`inspect`](rayon::iter::ParallelIterator::inspect), [`cloned`](rayon::iter::ParallelIterator::cloned), [`copied`](rayon::iter::ParallelIterator::copied) — leave the split to [`ConcurrentLimit`], so the limit stays exact.
//!   So do the methods that consume the iterator, such as [`for_each`](rayon::iter::ParallelIterator::for_each), [`collect`](rayon::iter::ParallelIterator::collect), [`collect_into_vec`](IndexedParallelIterator::collect_into_vec), and [`reduce`](rayon::iter::ParallelIterator::reduce).
//! - Adaptors that need a [`Producer`](rayon::iter::plumbing::Producer) of their own — [`zip`](IndexedParallelIterator::zip), [`enumerate`](IndexedParallelIterator::enumerate), [`rev`](IndexedParallelIterator::rev), [`skip`](IndexedParallelIterator::skip), [`take`](IndexedParallelIterator::take), [`step_by`](IndexedParallelIterator::step_by), [`chunks`](IndexedParallelIterator::chunks) — take over the splitting.
//!   The limit then degrades from exact to an upper bound, applied as an [`IndexedParallelIterator::with_min_len`] hint: [`rayon`] never splits below the minimum length, but because it splits by halving, the concurrency can undershoot the limit (as low as approximately half of it).
//!
//! ### Limitations
//! - [`concurrent_limit`](ParallelIteratorConcurrentLimit::concurrent_limit) is only available on iterators implementing [`rayon::iter::IndexedParallelIterator`].
//!
//! ### Alternatives
//! [`IndexedParallelIterator::by_uniform_blocks`](https://docs.rs/rayon/latest/rayon/iter/trait.IndexedParallelIterator.html#method.by_uniform_blocks) (rayon 1.9.0) can also bound the number of concurrent executions of an operation.
//! `iterator.by_uniform_blocks(limit)` processes blocks of `limit` items sequentially, with parallelism within each block.
//! However, every block ends with a synchronisation point, so a single slow item stalls the entire pipeline at each block boundary.
//! The approach of this crate lets each of the `limit` concurrent streams proceed independently, which suits expensive operations with variable cost.

#![warn(unused_variables)]
#![warn(dead_code)]
#![deny(missing_docs)]

mod split;

use rayon::iter::IndexedParallelIterator;

pub use split::ConcurrentLimit;

/// An extension trait for [`rayon::iter::IndexedParallelIterator`] which limits the concurrency of an iterator chain.
///
/// The [crate root documentation](crate) explains the motivation for this approach, provides further details on the underlying implementation, and details its limitations.
pub trait ParallelIteratorConcurrentLimit: IndexedParallelIterator {
    /// Limit the concurrency of every subsequent method in the iterator chain.
    ///
    /// This splits the iterator into exactly `concurrent_limit` work items, so at most
    /// `concurrent_limit` executions of any operation chained from it run concurrently.
    ///
    /// - A `concurrent_limit` of zero applies no limit.
    /// - A `concurrent_limit` of one runs chained operations sequentially, in a single [`rayon`] work item.
    /// - Parallel rayon methods executed *within* a chained operation continue to use the whole
    ///   thread pool (the global one, unless another has been installed; see [`rayon::ThreadPool`]).
    ///
    /// The output is an [`IndexedParallelIterator`], so indexed methods such as [`zip`](IndexedParallelIterator::zip)
    /// and [`collect_into_vec`](IndexedParallelIterator::collect_into_vec) remain available.
    /// The limit stays exact through consumer-wrapping adaptors such as [`map`](rayon::iter::ParallelIterator::map)
    /// and [`filter`](rayon::iter::ParallelIterator::filter), but chaining an adaptor that needs a producer of
    /// its own — [`zip`](IndexedParallelIterator::zip), [`enumerate`](IndexedParallelIterator::enumerate),
    /// [`rev`](IndexedParallelIterator::rev), and similar — hands over the splitting, and the limit
    /// degrades from exact to an upper bound.
    /// See [Implementation](crate#implementation) for the full rule.
    ///
    /// # Examples
    /// ```rust
    /// use rayon::iter::{IntoParallelIterator, ParallelIterator};
    /// use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;
    /// let sum = (0..100)
    ///     .into_par_iter()
    ///     .concurrent_limit(2)
    ///     .map(|i| {
    ///         let alloc = vec![i; 1000]; // max 2 concurrent allocations
    ///         alloc.into_par_iter().sum::<usize>() // runs on all threads
    ///     })
    ///     .sum::<usize>();
    /// assert_eq!(sum, (0..100).map(|i| i * 1000).sum::<usize>());
    /// ```
    ///
    /// The limit applies to any [`rayon`] method, such as
    /// [`try_for_each`](rayon::iter::ParallelIterator::try_for_each):
    /// ```rust
    /// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
    /// # use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;
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
    /// # use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;
    /// let mut output = Vec::new();
    /// (0..100)
    ///     .into_par_iter()
    ///     .concurrent_limit(2)
    ///     .map(|i| i * 2)
    ///     .collect_into_vec(&mut output);
    /// assert_eq!(output, (0..100).map(|i| i * 2).collect::<Vec<usize>>());
    /// ```
    fn concurrent_limit(self, concurrent_limit: usize) -> ConcurrentLimit<Self> {
        ConcurrentLimit::new(concurrent_limit, self)
    }
}

impl<I: IndexedParallelIterator> ParallelIteratorConcurrentLimit for I {}
