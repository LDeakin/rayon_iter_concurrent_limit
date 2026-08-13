//! Limit the concurrency of an individual rayon parallel iterator method.
//!
//! # Example
//! This example demonstrates a concurrency-limited `map` with the
//! [`map_concurrent_limit`](ParallelIteratorConcurrentLimit::map_concurrent_limit) method of the [`ParallelIteratorConcurrentLimit`] extension trait.
//! `map` is one of many methods with concurrency-limited variants.
//! ```rust
//! use rayon::iter::{IntoParallelIterator, ParallelIterator};
//! use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;
//! const N: usize = 1000;
//! let sum_iter = (0..100).into_par_iter().map_concurrent_limit(2, |i| {
//!     let alloc = vec![i; N]; // max 2 concurrent allocations in this example
//!     alloc.into_par_iter().sum::<usize>() // runs on all threads
//! });
//! let output = sum_iter
//!     .map(|alloc_sum| {
//!         alloc_sum / N // runs with limited concurrency, since it is chained from map_concurrent_limit
//!     })
//!     .collect::<Vec<usize>>();
//! assert_eq!(output, (0..100).into_iter().collect::<Vec<usize>>());
//! ```
//! The equivalent `sum_iter` expression using [`iter_subdivide`] is:
//! ```rust
//! # use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
//! # use rayon_iter_concurrent_limit::iter_subdivide;
//! # const N: usize = 1000;
//! # let op = |i: usize| -> usize {
//! #     let alloc = vec![i; N]; // max 2 concurrent allocations in this example
//! #     alloc.into_par_iter().sum::<usize>() // runs on all threads
//! # };
//! let sum_iter = iter_subdivide(2, (0..100).into_par_iter())
//!     .flat_map_iter(|chunk| chunk)
//!     .map(op);
//! # let output = sum_iter
//! #     .map(|alloc_sum| -> usize {
//! #         alloc_sum / N
//! #     })
//! #     .collect::<Vec<usize>>();
//! # assert_eq!(output, (0..100).into_iter().collect::<Vec<usize>>());
//! ```
//! The equivalent expression without using functionality in this crate is:
//! ```rust
//! # use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
//! # const N: usize = 1000;
//! # let op = |i: usize| -> usize {
//! #     let alloc = vec![i; N]; // max 2 concurrent allocations in this example
//! #     alloc.into_par_iter().sum::<usize>() // runs on all threads
//! # };
//! let sum_iter = (0..100)
//!     .into_par_iter()
//!     .chunks((100 + 2 - 1) / 2)
//!     .flat_map_iter(|chunk| chunk)
//!     .map(op);
//! # let output = sum_iter
//! #     .map(|alloc_sum| -> usize {
//! #         alloc_sum / N
//! #     })
//! #     .collect::<Vec<usize>>();
//! # assert_eq!(output, (0..100).into_iter().collect::<Vec<usize>>());
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
//! However, this has some limitations and footguns:
//! - Any parallel operations within `op` will use the same thread-limited thread pool, unless [`install`](rayon::ThreadPool::install) is called internally with a different thread pool.
//! - If [`install`](rayon::ThreadPool::install) is called internally, `op` can yield and multiple instances of `op` may run concurrently on a thread. This is detailed [here](https://docs.rs/rayon/1.8.1/rayon/struct.ThreadPool.html#warning-execution-order) in the [`install`](rayon::ThreadPool::install) documentation.
//! - An iterator must be consumed in the [`install`](rayon::ThreadPool::install) scope of a [`ThreadPool`](rayon::ThreadPool), otherwise it will not use that thread pool.
//!
//! # Solution
//! This crate provides [`ParallelIteratorConcurrentLimit`], an extension trait for [`rayon::iter::IndexedParallelIterator`] which adds variants of many [`ParallelIterator`] methods with a `_concurrent_limit` suffix and a `concurrent_limit` parameter.
//! These methods execute their operands (operations, predicates, etc.) with limited concurrency.
//!
//! The deprecated [`iter_concurrent_limit`] macro provides the same functionality.
//!
//! ### Implementation
//! Concurrency is limited by calling [`IndexedParallelIterator::chunks`] on the parallel iterator (using the [`iter_subdivide`] method) to reduce the number of work items for [`rayon`].
//! Internally, the [`iter_subdivide`] method calculates the chunk size as `iterator.len().div_ceil(concurrent_limit)`.
//! The function passed to a concurrency-limited method is called sequentially on the items in each chunk, but in parallel over the chunks.
//! Methods with an iterator output (e.g. [`map_concurrent_limit`](ParallelIteratorConcurrentLimit::map_concurrent_limit)) flatten their output iterator, and subsequent chained iterator methods continue to run with limited concurrency.
//!
//! ### Limitations
//! - The concurrency-limited methods are only available on iterators implementing [`rayon::iter::IndexedParallelIterator`].
//! - Only a subset of relevant [`ParallelIterator`]/[`IndexedParallelIterator`] methods currently have concurrency-limited variants.
//! - [`try_for_each_concurrent_limit`](ParallelIteratorConcurrentLimit::try_for_each_concurrent_limit) only supports [`Result`] operation outputs (unlike [`rayon::iter::ParallelIterator::try_for_each`], which also supports [`Option`] and [`std::ops::ControlFlow`]), because the `Try` trait of [`rayon`] is private.
//!
//! ### Alternatives
//! [`IndexedParallelIterator::by_uniform_blocks`](https://docs.rs/rayon/latest/rayon/iter/trait.IndexedParallelIterator.html#method.by_uniform_blocks) (rayon 1.9.0) can also bound the number of concurrent executions of an operation.
//! `iterator.by_uniform_blocks(limit)` processes blocks of `limit` items sequentially (with parallelism within each block), avoiding the per-chunk allocation of the approach in this crate.
//! However, every block ends with a synchronisation point, so a single slow item stalls the entire pipeline at each block boundary.
//! The chunking approach of this crate lets each of the `limit` concurrent streams proceed independently, which suits expensive operations with variable cost.

#![warn(unused_variables)]
#![warn(dead_code)]
#![deny(missing_docs)]

use rayon::iter::{Chunks, IndexedParallelIterator, ParallelIterator};

/// Subdivide a [`rayon::iter::IndexedParallelIterator`] into `num_chunks` chunks.
///
/// This returns the output of the [`IndexedParallelIterator::chunks`] function with a chunk size calculated according to:
/// ```rust
/// # use rayon::iter::IntoParallelIterator;
/// # use rayon::iter::IndexedParallelIterator;
/// # let num_chunks: usize = 1;
/// # let iterator = (0..1).into_par_iter();
/// iterator.len().div_ceil(num_chunks)
/// # ;
/// ```
/// If `num_chunks` is zero, then there will be one chunk per iterator item.
///
/// If `num_chunks` does not evenly divide the iterator length, the last chunk will be smaller than the rest.
///
/// This method is used internally by the [`ParallelIteratorConcurrentLimit`] methods and the [`iter_concurrent_limit`] macro.
pub fn iter_subdivide<I: IndexedParallelIterator>(num_chunks: usize, iterator: I) -> Chunks<I> {
    if num_chunks == 0 {
        iterator.chunks(1)
    } else {
        let chunk_size = std::cmp::max(iterator.len().div_ceil(num_chunks), 1);
        iterator.chunks(chunk_size)
    }
}

/// Subdivide `iterator` into `concurrent_limit` chunks and flatten it.
///
/// Chunks are the unit of work for rayon, so at most `concurrent_limit` chunks are processed concurrently.
/// Chained (unindexed) iterator methods are applied sequentially to the items within each chunk.
fn iter_limited<I: IndexedParallelIterator>(
    concurrent_limit: usize,
    iterator: I,
) -> impl ParallelIterator<Item = I::Item> {
    iter_subdivide(concurrent_limit, iterator).flat_map_iter(|chunk| chunk)
}

/// An extension trait for [`rayon::iter::IndexedParallelIterator`] adding variants of [`ParallelIterator`] methods with a limit on the number of concurrent executions of the supplied function.
///
/// Concurrent executions are limited by chunking the iterator to reduce the number of work items.
/// The [crate root documentation](crate) explains the motivation for this approach, provides further details on the underlying implementation, and details its limitations.
///
/// # Arguments
/// Each method takes a `concurrent_limit` parameter before the function parameter of the underlying [`ParallelIterator`] method:
/// - `concurrent_limit` is a [`usize`] specifying the maximum concurrent executions of the supplied function.
///   - A `concurrent_limit` of zero means no concurrent limit. Methods which consume the iterator (e.g. [`for_each_concurrent_limit`](ParallelIteratorConcurrentLimit::for_each_concurrent_limit)) will skip internal chunking in this case.
///   - A `concurrent_limit` of one executes the function sequentially in a single [`rayon`] work item, and the entire iterator is collected into a single chunk.
/// - The function is called *sequentially* on the items in each chunk, but in *parallel* over the chunks, with the number of concurrent executions upper bounded by the `concurrent_limit`.
///   - Parallel rayon methods executed in the function will implicitly utilise the global thread pool unless an alternative thread pool has been installed (see [`rayon::ThreadPool`]).
pub trait ParallelIteratorConcurrentLimit: IndexedParallelIterator {
    /// Execute `op` on each item of this iterator, with a limit on the number of concurrent executions of `op`.
    ///
    /// This is a concurrency-limited variant of [`rayon::iter::ParallelIterator::for_each`].
    ///
    /// # Examples
    /// ```rust
    /// use rayon::iter::{IntoParallelIterator, ParallelIterator};
    /// use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;
    /// (0..10).into_par_iter().for_each_concurrent_limit(2, |i| {
    ///     let alloc = vec![i; 1000]; // limited concurrency
    ///     alloc.into_par_iter().for_each(|_j| {}); // runs on all threads
    /// });
    /// ```
    fn for_each_concurrent_limit<OP>(self, concurrent_limit: usize, op: OP)
    where
        OP: Fn(Self::Item) + Sync + Send;

    /// Execute the fallible `op` on each item of this iterator, with a limit on the number of concurrent executions of `op`.
    ///
    /// This is a concurrency-limited variant of [`rayon::iter::ParallelIterator::try_for_each`].
    ///
    /// Unlike [`try_for_each`](rayon::iter::ParallelIterator::try_for_each), `op` must return a [`Result`] (rather than any `Try` type), because the `Try` trait of [`rayon`] is private.
    ///
    /// # Examples
    /// ```rust
    /// use rayon::iter::{IntoParallelIterator, ParallelIterator};
    /// use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;
    /// let result = (0..10).into_par_iter().try_for_each_concurrent_limit(2, |i| {
    ///     let alloc = vec![i; 1000]; // limited concurrency
    ///     alloc.into_par_iter().for_each(|_j| {}); // runs on all threads
    ///     Ok::<(), std::io::Error>(())
    /// });
    /// assert!(result.is_ok());
    /// ```
    fn try_for_each_concurrent_limit<OP, E>(self, concurrent_limit: usize, op: OP) -> Result<(), E>
    where
        OP: Fn(Self::Item) -> Result<(), E> + Sync + Send,
        E: Send;

    /// Apply `map_op` to each item of this iterator, with a limit on the number of concurrent executions of `map_op`.
    ///
    /// This is a concurrency-limited variant of [`rayon::iter::ParallelIterator::map`].
    ///
    /// The output iterator is unindexed, and methods chained from it continue to run with limited concurrency.
    ///
    /// # Examples
    /// ```rust
    /// use rayon::iter::{IntoParallelIterator, ParallelIterator};
    /// use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;
    /// let sum = (0..100)
    ///     .into_par_iter()
    ///     .map_concurrent_limit(2, |i| {
    ///         let alloc = vec![i; 1000]; // limited concurrency
    ///         alloc.into_par_iter().sum::<usize>() // runs on all threads
    ///     })
    ///     .sum::<usize>();
    /// assert_eq!(sum, (0..100).map(|i| i * 1000).sum::<usize>());
    /// ```
    fn map_concurrent_limit<OP, R>(
        self,
        concurrent_limit: usize,
        map_op: OP,
    ) -> impl ParallelIterator<Item = R>
    where
        OP: Fn(Self::Item) -> R + Sync + Send,
        R: Send;

    /// Mutate each item of this iterator with `update_op`, with a limit on the number of concurrent executions of `update_op`.
    ///
    /// This is a concurrency-limited variant of [`rayon::iter::ParallelIterator::update`].
    ///
    /// The output iterator is unindexed, and methods chained from it continue to run with limited concurrency.
    ///
    /// # Examples
    /// ```rust
    /// use rayon::iter::{IntoParallelIterator, ParallelIterator};
    /// use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;
    /// let doubled = (0..100)
    ///     .into_par_iter()
    ///     .update_concurrent_limit(2, |i| {
    ///         // ... do work with limited concurrency
    ///         *i *= 2;
    ///     })
    ///     .collect::<Vec<usize>>();
    /// assert_eq!(doubled, (0..100).map(|i| i * 2).collect::<Vec<usize>>());
    /// ```
    fn update_concurrent_limit<OP>(
        self,
        concurrent_limit: usize,
        update_op: OP,
    ) -> impl ParallelIterator<Item = Self::Item>
    where
        OP: Fn(&mut Self::Item) + Sync + Send;

    /// Keep the items of this iterator satisfying `filter_op`, with a limit on the number of concurrent executions of `filter_op`.
    ///
    /// This is a concurrency-limited variant of [`rayon::iter::ParallelIterator::filter`].
    ///
    /// The output iterator is unindexed, and methods chained from it continue to run with limited concurrency.
    ///
    /// # Examples
    /// ```rust
    /// use rayon::iter::{IntoParallelIterator, ParallelIterator};
    /// use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;
    /// let even = (0..100)
    ///     .into_par_iter()
    ///     .filter_concurrent_limit(2, |i| {
    ///         // ... do work with limited concurrency
    ///         i % 2 == 0
    ///     })
    ///     .collect::<Vec<usize>>();
    /// assert_eq!(even, (0..100).filter(|i| i % 2 == 0).collect::<Vec<usize>>());
    /// ```
    fn filter_concurrent_limit<P>(
        self,
        concurrent_limit: usize,
        filter_op: P,
    ) -> impl ParallelIterator<Item = Self::Item>
    where
        P: Fn(&Self::Item) -> bool + Sync + Send;

    /// Apply the filtering `filter_op` to each item of this iterator, with a limit on the number of concurrent executions of `filter_op`.
    ///
    /// This is a concurrency-limited variant of [`rayon::iter::ParallelIterator::filter_map`].
    ///
    /// The output iterator is unindexed, and methods chained from it continue to run with limited concurrency.
    ///
    /// # Examples
    /// ```rust
    /// use rayon::iter::{IntoParallelIterator, ParallelIterator};
    /// use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;
    /// let even_doubled = (0..100)
    ///     .into_par_iter()
    ///     .filter_map_concurrent_limit(2, |i| {
    ///         // ... do work with limited concurrency
    ///         if i % 2 == 0 { Some(i * 2) } else { None }
    ///     })
    ///     .collect::<Vec<usize>>();
    /// assert_eq!(
    ///     even_doubled,
    ///     (0..100).filter_map(|i| if i % 2 == 0 { Some(i * 2) } else { None }).collect::<Vec<usize>>()
    /// );
    /// ```
    fn filter_map_concurrent_limit<P, R>(
        self,
        concurrent_limit: usize,
        filter_op: P,
    ) -> impl ParallelIterator<Item = R>
    where
        P: Fn(Self::Item) -> Option<R> + Sync + Send,
        R: Send;

    /// Search for items of this iterator matching `predicate`, with a limit on the number of concurrent executions of `predicate`.
    ///
    /// This is a concurrency-limited variant of [`rayon::iter::ParallelIterator::any`].
    ///
    /// # Examples
    /// ```rust
    /// use rayon::iter::{IntoParallelIterator, ParallelIterator};
    /// use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;
    /// let any_eq_50 = (0..100).into_par_iter().any_concurrent_limit(2, |i| {
    ///     // ... do work with limited concurrency
    ///     i == 50
    /// });
    /// assert!(any_eq_50);
    /// ```
    fn any_concurrent_limit<P>(self, concurrent_limit: usize, predicate: P) -> bool
    where
        P: Fn(Self::Item) -> bool + Sync + Send;

    /// Test that all items of this iterator match `predicate`, with a limit on the number of concurrent executions of `predicate`.
    ///
    /// This is a concurrency-limited variant of [`rayon::iter::ParallelIterator::all`].
    ///
    /// # Examples
    /// ```rust
    /// use rayon::iter::{IntoParallelIterator, ParallelIterator};
    /// use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;
    /// let all_lt_100 = (0..100).into_par_iter().all_concurrent_limit(2, |i| {
    ///     // ... do work with limited concurrency
    ///     i < 100
    /// });
    /// assert!(all_lt_100);
    /// ```
    fn all_concurrent_limit<P>(self, concurrent_limit: usize, predicate: P) -> bool
    where
        P: Fn(Self::Item) -> bool + Sync + Send;
}

impl<I: IndexedParallelIterator> ParallelIteratorConcurrentLimit for I {
    fn for_each_concurrent_limit<OP>(self, concurrent_limit: usize, op: OP)
    where
        OP: Fn(Self::Item) + Sync + Send,
    {
        if concurrent_limit == 0 {
            self.for_each(op)
        } else {
            iter_limited(concurrent_limit, self).for_each(op)
        }
    }

    fn try_for_each_concurrent_limit<OP, E>(self, concurrent_limit: usize, op: OP) -> Result<(), E>
    where
        OP: Fn(Self::Item) -> Result<(), E> + Sync + Send,
        E: Send,
    {
        if concurrent_limit == 0 {
            self.try_for_each(op)
        } else {
            iter_limited(concurrent_limit, self).try_for_each(op)
        }
    }

    fn map_concurrent_limit<OP, R>(
        self,
        concurrent_limit: usize,
        map_op: OP,
    ) -> impl ParallelIterator<Item = R>
    where
        OP: Fn(Self::Item) -> R + Sync + Send,
        R: Send,
    {
        iter_limited(concurrent_limit, self).map(map_op)
    }

    fn update_concurrent_limit<OP>(
        self,
        concurrent_limit: usize,
        update_op: OP,
    ) -> impl ParallelIterator<Item = Self::Item>
    where
        OP: Fn(&mut Self::Item) + Sync + Send,
    {
        iter_limited(concurrent_limit, self).update(update_op)
    }

    fn filter_concurrent_limit<P>(
        self,
        concurrent_limit: usize,
        filter_op: P,
    ) -> impl ParallelIterator<Item = Self::Item>
    where
        P: Fn(&Self::Item) -> bool + Sync + Send,
    {
        iter_limited(concurrent_limit, self).filter(filter_op)
    }

    fn filter_map_concurrent_limit<P, R>(
        self,
        concurrent_limit: usize,
        filter_op: P,
    ) -> impl ParallelIterator<Item = R>
    where
        P: Fn(Self::Item) -> Option<R> + Sync + Send,
        R: Send,
    {
        iter_limited(concurrent_limit, self).filter_map(filter_op)
    }

    fn any_concurrent_limit<P>(self, concurrent_limit: usize, predicate: P) -> bool
    where
        P: Fn(Self::Item) -> bool + Sync + Send,
    {
        if concurrent_limit == 0 {
            self.any(predicate)
        } else {
            iter_limited(concurrent_limit, self).any(predicate)
        }
    }

    fn all_concurrent_limit<P>(self, concurrent_limit: usize, predicate: P) -> bool
    where
        P: Fn(Self::Item) -> bool + Sync + Send,
    {
        if concurrent_limit == 0 {
            self.all(predicate)
        } else {
            iter_limited(concurrent_limit, self).all(predicate)
        }
    }
}

/// Apply a method on a [`rayon::iter::IndexedParallelIterator`] with a limit on the number of concurrent executions of the function passed to the method.
///
/// **Deprecated**: use the methods of the [`ParallelIteratorConcurrentLimit`] extension trait instead, such as [`map_concurrent_limit`](ParallelIteratorConcurrentLimit::map_concurrent_limit).
///
/// Concurrent executions are limited by chunking the iterator to reduce the number of work items.
/// The [crate root documentation](crate) explains the motivation for this approach, provides further details on the underlying implementation of the macro, and details its limitations.
///
/// # Arguments
/// The macro arguments are `(concurrent_limit, iterator, method, function)`:
/// - `concurrent_limit` is a [`usize`] specifying the maximum concurrent executions of `function`.
///   - A `concurrent_limit` of zero means no concurrent limit. Some methods will skip internal chunking in this case.
/// - `iterator` implements [`std::iter::IntoIterator`] and [`rayon::iter::IntoParallelIterator`]
///   - The parallel iterator must implement [`rayon::iter::IndexedParallelIterator`].
/// - `method` is the name of a supported iterator method:
///   - Only methods which call a supplied function are supported.
///   - Operations without a function (e.g. min, max) will not allocate and there is little benefit in setting a concurrent limit for such methods.
///   - Not every iterator method matching this criteria is currently supported.
/// - `function` is a function compatible with `method`, such as an operation, predicate, etc.
///   - The function is called *sequentially* on the items in each chunk, but in *parallel* over the chunks, with the number of concurrent executions upper bounded by the `concurrent_limit`.
///   - Parallel rayon methods executed in the function will implicitly utilise the global thread pool unless an alternative thread pool has been installed (see [`rayon::ThreadPool`]).
///
/// # Examples
/// ### for_each
/// ```rust
/// # #![allow(deprecated)]
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: usize| {
///     let alloc = vec![i; 1000]; // limited concurrency
///     alloc.into_par_iter().for_each(|_j| {}); // runs on all threads
/// };
/// iter_concurrent_limit!(2, 0..10, for_each, op);
/// ```
///
/// ### try_for_each
/// ```rust
/// # #![allow(deprecated)]
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: usize| {
///     let alloc = vec![i; 1000]; // limited concurrency
///     alloc.into_par_iter().for_each(|_j| {}); // runs on all threads
///     Ok::<(), std::io::Error>(())
/// };
/// iter_concurrent_limit!(2, 0..10, try_for_each, op)?;
/// # Ok(())
/// # }
/// ```
///
/// ### map
/// ```rust
/// # #![allow(deprecated)]
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: usize| {
///     let alloc = vec![i; 1000]; // limited concurrency
///     alloc.into_par_iter().sum::<usize>() // runs on all threads
/// };
/// let sum =
///     iter_concurrent_limit!(2, 0..100, map, op)
///     .sum::<usize>();
/// assert_eq!(sum, (0..100).into_iter().map(op).sum::<usize>());
/// ```
///
/// ### filter
/// ```rust
/// # #![allow(deprecated)]
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: &usize| -> bool {
///     // ... do work with limited concurrency
///     i % 2 == 0
/// };
/// let even =
///     iter_concurrent_limit!(2, 0..100, filter, op)
///     .collect::<Vec<usize>>();
/// assert_eq!(even, (0..100).into_iter().filter(op).collect::<Vec<usize>>());
/// ```
///
/// ### filter_map
/// ```rust
/// # #![allow(deprecated)]
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: usize| -> Option<usize> {
///     // ... do work with limited concurrency
///     if i % 2 == 0 { Some(i * 2) } else { None }
/// };
/// let even_doubled =
///     iter_concurrent_limit!(2, 0..100, filter_map, op)
///     .collect::<Vec<usize>>();
/// assert_eq!(even_doubled, (0..100).into_iter().filter_map(op).collect::<Vec<usize>>());
/// ```
///
/// ### any
/// ```rust
/// # #![allow(deprecated)]
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: usize| -> bool {
///     // ... do work with limited concurrency
///     i == 50
/// };
/// let any_eq_50 = iter_concurrent_limit!(2, 0..100, any, op);
/// assert_eq!(any_eq_50, (0..100).into_iter().any(op));
/// ```
///
/// ### all
/// ```rust
/// # #![allow(deprecated)]
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: usize| -> bool {
///     // ... do work with limited concurrency
///     i == 50
/// };
/// let all_eq_50 = iter_concurrent_limit!(2, 0..100, all, op);
/// assert_eq!(all_eq_50, (0..100).into_iter().all(op));
/// ```
///
#[deprecated(
    since = "0.3.0",
    note = "use the methods of the ParallelIteratorConcurrentLimit trait instead"
)]
#[macro_export]
macro_rules! iter_concurrent_limit {
    ( $concurrent_limit:expr, $iterator:expr, for_each, $op:expr ) => {{
        let concurrent_limit = $concurrent_limit;
        let op = $op;
        if concurrent_limit == 1 {
            $iterator.into_iter().for_each(op)
        } else {
            $crate::ParallelIteratorConcurrentLimit::for_each_concurrent_limit(
                $iterator.into_par_iter(),
                concurrent_limit,
                op,
            )
        }
    }};
    ( $concurrent_limit:expr, $iterator:expr, try_for_each, $op:expr ) => {{
        // Not delegated to try_for_each_concurrent_limit, which only supports Result outputs.
        let concurrent_limit = $concurrent_limit;
        let op = $op;
        if concurrent_limit == 0 {
            $iterator.into_par_iter().try_for_each(op)
        } else if concurrent_limit == 1 {
            $iterator.into_iter().try_for_each(op)
        } else {
            let chunks = $crate::iter_subdivide(concurrent_limit, $iterator.into_par_iter());
            chunks.try_for_each(|chunk| chunk.into_iter().try_for_each(op))
        }
    }};
    ( $concurrent_limit:expr, $iterator:expr, map, $map_op:expr ) => {{
        $crate::ParallelIteratorConcurrentLimit::map_concurrent_limit(
            $iterator.into_par_iter(),
            $concurrent_limit,
            $map_op,
        )
    }};
    ( $concurrent_limit:expr, $iterator:expr, update, $update_op:expr ) => {{
        $crate::ParallelIteratorConcurrentLimit::update_concurrent_limit(
            $iterator.into_par_iter(),
            $concurrent_limit,
            $update_op,
        )
    }};
    ( $concurrent_limit:expr, $iterator:expr, filter, $filter_op:expr ) => {{
        $crate::ParallelIteratorConcurrentLimit::filter_concurrent_limit(
            $iterator.into_par_iter(),
            $concurrent_limit,
            $filter_op,
        )
    }};
    ( $concurrent_limit:expr, $iterator:expr, filter_map, $filter_op:expr ) => {{
        $crate::ParallelIteratorConcurrentLimit::filter_map_concurrent_limit(
            $iterator.into_par_iter(),
            $concurrent_limit,
            $filter_op,
        )
    }};
    ( $concurrent_limit:expr, $iterator:expr, any, $predicate:expr ) => {{
        let concurrent_limit = $concurrent_limit;
        let predicate = $predicate;
        if concurrent_limit == 1 {
            $iterator.into_iter().any(predicate)
        } else {
            $crate::ParallelIteratorConcurrentLimit::any_concurrent_limit(
                $iterator.into_par_iter(),
                concurrent_limit,
                predicate,
            )
        }
    }};
    ( $concurrent_limit:expr, $iterator:expr, all, $predicate:expr ) => {{
        let concurrent_limit = $concurrent_limit;
        let predicate = $predicate;
        if concurrent_limit == 1 {
            $iterator.into_iter().all(predicate)
        } else {
            $crate::ParallelIteratorConcurrentLimit::all_concurrent_limit(
                $iterator.into_par_iter(),
                concurrent_limit,
                predicate,
            )
        }
    }};
    ( $concurrent_limit:expr, $iterator:expr, $method:ident, $predicate:expr ) => {{
        std::compile_error!("This macro does not support the requested method");
    }};
}
