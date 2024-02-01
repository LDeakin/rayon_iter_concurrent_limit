//! Limit the concurrency of an individual rayon parallel iterator method with a convenient macro.
//!
//! # Example
//! ```rust
//! # use rayon::iter::{IntoParallelIterator, ParallelIterator};
//! # use rayon_iter_concurrent_limit::iter_concurrent_limit;
//! const N: usize = 1000;
//! let output = iter_concurrent_limit!(2, (0..100).into_par_iter(), map, |i: usize| {
//!     let alloc = vec![i; N];              // max of 2 concurrent allocations
//!     alloc.into_par_iter().sum::<usize>() // runs on all threads
//! }).map(|alloc_sum| -> usize {
//!     alloc_sum / N                        // runs on all threads
//! }).collect::<Vec<usize>>();
//! assert_eq!(output, (0..100).into_iter().collect::<Vec<usize>>());
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
//! The number of threads could be limited with [`rayon::ThreadPool::install`] like so:
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
//! - If [`install`](rayon::ThreadPool::install) is called internally, `op` can yield and multiple instances of `op` may run concurrently on each thread. This is detailed [here](https://docs.rs/rayon/1.8.1/rayon/struct.ThreadPool.html#warning-execution-order) in the [`install`](rayon::ThreadPool::install) documentation.
//! - An iterator must be consumed in the [`install`](rayon::ThreadPool::install) scope of a [`ThreadPool`](rayon::ThreadPool), otherwise it will not use that thread pool.
//!
//! # Solution
//! This crate provides [`iter_concurrent_limit`], a macro that enables many [`rayon::iter::ParallelIterator`] methods to execute their operands with limited concurrency.
//!
//! The [Examples](crate::iter_concurrent_limit#examples) section of [`iter_concurrent_limit`] has various usage examples.
//!
//! ### Implementation
//! The macro limits concurrency by calling [`IndexedParallelIterator::chunks`] on the parallel iterator (using the [`chunks_concurrent_limit`] method) to reduce the number of work items for [`rayon`].
//! Internally, the [`chunks_concurrent_limit`] method calculates the chunk size as `iterator.len().ceiling_div(concurrent_limit)`.
//! The function passed to the macro is called sequentially on the items in each chunk, but in parallel over the chunks.
//! The output of the function is flattened for methods with an iterator output, like `map` and `filter`.
//!
//! ### Limitations
//! - Iterators passed to [`iter_concurrent_limit`] must implement [`IndexedParallelIterator`].
//! - Not all relevant [`ParallelIterator`](rayon::iter::ParallelIterator) methods are currently supported.
// TODO: - Methods which rely on thread-local initialisation (e.g. [`rayon::iter::ParallelIterator::map_init`]) will not function identically when run though [`iter_concurrent_limit`].

#![warn(unused_variables)]
#![warn(dead_code)]
#![deny(missing_docs)]

use rayon::iter::{Chunks, IndexedParallelIterator};

/// Split a [`rayon::iter::IndexedParallelIterator`] into `concurrent_limit` chunks.
///
/// This method is used internally by the [`iter_concurrent_limit`] macro.
pub fn chunks_concurrent_limit<I: IndexedParallelIterator>(
    iterator: I,
    concurrent_limit: usize,
) -> Chunks<I> {
    let chunk_size = (iterator.len() + concurrent_limit - 1) / concurrent_limit;
    iterator.chunks(chunk_size)
}

// TODO: Support more methods
/// Apply a method on a [`rayon::iter::IndexedParallelIterator`] with a limit on the number of concurrent executions of the function passed to the method.
///
/// Concurrent executions are limited by chunking the iterator to reduce the number of work items.
/// The [crate root documentation](crate) explains the motivation for this approach, provides further details on the underlying implementation of the macro, and details its limitations.
///
/// # Arguments
/// The macro arguments are `(concurrent_limit, parallel_iterator, method, function)`:
/// - `concurrent_limit` is a [`usize`] specifying the maximum concurrent executions of `function`.
/// - `parallel_iterator` is an iterator implementing [`rayon::iter::IndexedParallelIterator`].
/// - `method` is the name of a supported [`ParallelIterator`](rayon::iter::ParallelIterator)/[`IndexedParallelIterator`] method:
///   - Only methods which call a supplied function are supported.
///   - Operations without a function (e.g. min, max) will not allocate and there is little benefit in setting a concurrent limit for such methods.
///   - Not every [`ParallelIterator`](rayon::iter::ParallelIterator)/[`IndexedParallelIterator`] method matching this criteria is currently supported.
/// - `function` is a function compatible with `method`, such as an operation, predicate, etc.
///   - The function is called *sequentially* on the items in each chunk, but in *parallel* over the chunks, with the number of concurrent executions constrained by the `concurrent_limit`.
///   - Parallel rayon methods executed in the function will implicitly utilise the global thread pool unless an alternative thread pool has been installed (see [`rayon::ThreadPool`]).
///
/// # Examples
/// ### for_each
/// ```rust
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// iter_concurrent_limit!(2, (0..10).into_par_iter(), for_each, |i: usize| {
///     let alloc = vec![i; 1000]; // max of 2 concurrent allocations
///     alloc.into_par_iter().for_each(|_j| {}); // runs on all threads
/// });
/// ```
///
/// ### try_for_each
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// iter_concurrent_limit!(2, (0..10).into_par_iter(), try_for_each, |i: usize| {
///     let alloc = vec![i; 1000]; // max of 2 concurrent allocations
///     alloc.into_par_iter().for_each(|_j| {}); // runs on all threads
///     Ok::<(), std::io::Error>(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// ### map
/// ```rust
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// const N: usize = 1000;
/// let sum = iter_concurrent_limit!(2, (0..100).into_par_iter(), map, |i: usize| {
///     let alloc = vec![i; 1000]; // max of 2 concurrent allocations
///     alloc.into_par_iter().sum::<usize>() // runs on all threads
/// }).sum::<usize>();
/// assert_eq!(sum, (0..100).into_iter().map(|i| i * 1000).sum::<usize>());
/// ```
///
/// ### filter
/// ```rust
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// const N: usize = 1000;
/// let even = iter_concurrent_limit!(2, (0..100).into_par_iter(), filter, |i: &usize| -> bool {
///     // .. do work with limited concurrency
///     i % 2 == 0
/// }).collect::<Vec<usize>>();
/// assert_eq!(even, (0..100).into_iter().filter(|i| i % 2 == 0).collect::<Vec<usize>>());
/// ```
///
/// ### filter_map
/// ```rust
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// const N: usize = 1000;
/// let even = iter_concurrent_limit!(2, (0..100).into_par_iter(), filter_map, |i: usize| -> Option<usize> {
///     // .. do work with limited concurrency
///     if i % 2 == 0 { Some(i * 2) } else { None }
/// }).collect::<Vec<usize>>();
/// assert_eq!(even, (0..100).into_iter().filter_map(|i| if i % 2 == 0 { Some(i * 2) } else { None }).collect::<Vec<usize>>());
/// ```
///
/// ### any
/// ```rust
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// const N: usize = 1000;
/// let any_eq_50 = iter_concurrent_limit!(2, (0..100).into_par_iter(), any, |i: usize| -> bool {
///     // .. do work with limited concurrency
///     i == 50
/// });
/// assert_eq!(any_eq_50, (0..100).into_iter().any(|i| i == 50));
/// ```
///
/// ### all
/// ```rust
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// const N: usize = 1000;
/// let all_lt_100 = iter_concurrent_limit!(2, (0..100).into_par_iter(), all, |i: usize| -> bool {
///     // .. do work with limited concurrency
///     i < 100
/// });
/// assert_eq!(all_lt_100, (0..100).into_iter().any(|i| i < 100));
/// ```
///
#[macro_export]
macro_rules! iter_concurrent_limit {
    ( $concurrent_limit:expr, $iterator:expr, for_each, $op:expr ) => {{
        let chunks = $crate::chunks_concurrent_limit($iterator, $concurrent_limit);
        chunks.for_each(|chunk| chunk.into_iter().for_each($op))
    }};
    // TODO: for_each_with?
    // TODO: for_each_init?
    ( $concurrent_limit:expr, $iterator:expr, try_for_each, $op:expr ) => {{
        let chunks = $crate::chunks_concurrent_limit($iterator, $concurrent_limit);
        chunks.try_for_each(|chunk| chunk.into_iter().try_for_each($op))
    }};
    // TODO: try_for_each_with?
    // TODO: try_for_each_init?
    ( $concurrent_limit:expr, $iterator:expr, map, $map_op:expr ) => {{
        let chunks = $crate::chunks_concurrent_limit($iterator, $concurrent_limit);
        chunks.flat_map_iter(|chunk| chunk.into_iter().map($map_op))
    }};
    // TODO: map_with?
    // TODO: map_init?
    // IGNORE: inspect
    ( $concurrent_limit:expr, $iterator:expr, update, $update_op:expr ) => {{
        let chunks = $crate::chunks_concurrent_limit($iterator, $concurrent_limit);
        chunks.flat_map_iter(|chunk| {
            chunk.into_iter().map(|mut item| {
                $update_op(&mut item);
                item
            })
        })
    }};
    ( $concurrent_limit:expr, $iterator:expr, filter, $filter_op:expr ) => {{
        let chunks = $crate::chunks_concurrent_limit($iterator, $concurrent_limit);
        chunks.flat_map_iter(|chunk| chunk.into_iter().filter($filter_op))
    }};
    ( $concurrent_limit:expr, $iterator:expr, filter_map, $filter_op:expr ) => {{
        let chunks = $crate::chunks_concurrent_limit($iterator, $concurrent_limit);
        chunks.flat_map_iter(|chunk| chunk.into_iter().filter_map($filter_op))
    }};
    // TODO: flat_map?
    // ( $concurrent_limit:expr, $iterator:expr, flat_map, $map_op:expr ) => {{
    //     let chunks = $crate::chunks_concurrent_limit($iterator, $concurrent_limit);
    //     chunks.flat_map_iter(|chunk| chunk.into_iter().map($map_op))
    // }};
    // TODO: flat_map_iter?
    // TODO: reduce?
    // TODO: reduce_with?
    // TODO: try_reduce?
    // TODO: try_reduce_with?
    // TODO: fold?
    // TODO: fold_with?
    // TODO: try_fold?
    // TODO: try_fold_with?
    // ( $concurrent_limit:expr, $iterator:expr, max_by_key, $f:expr ) => {{
    //     let chunks = $crate::chunks_concurrent_limit($iterator, $concurrent_limit);
    //     chunks
    //         .flat_map(|chunk| chunk.into_iter().max_by_key($f))
    //         .max_by_key($f)
    // }};
    // ( $concurrent_limit:expr, $iterator:expr, min_by_key, $f:expr ) => {{
    //     let chunks = chunks_concurrent_limit($iterator, $concurrent_limit);
    //     chunks
    //         .flat_map(|chunk| chunk.into_iter().min_by_key($f))
    //         .min_by_key($f)
    // }};
    // TODO: find_any?
    // TODO: find_first?
    // TODO: find_last?
    // TODO: find_map_any?
    // TODO: find_map_first?
    // TODO: find_map_last?
    ( $concurrent_limit:expr, $iterator:expr, any, $predicate:expr ) => {{
        let chunks = $crate::chunks_concurrent_limit($iterator, $concurrent_limit);
        chunks.any(|chunk| chunk.into_iter().any($predicate))
    }};
    ( $concurrent_limit:expr, $iterator:expr, all, $predicate:expr ) => {{
        let chunks = $crate::chunks_concurrent_limit($iterator, $concurrent_limit);
        chunks.all(|chunk| chunk.into_iter().all($predicate))
    }};
    // TODO: partition?
    // TODO: partition_map?
    // TODO: take_any_while?
    // TODO: skip_any_while?
    // TODO: IndexedParallelIterator zip, zip_eq, fold_chunks, fold_chunks_with, cmp, partial_cmp, position_any, position_first, position_last, positions?
    ( $concurrent_limit:expr, $iterator:expr, $method:ident, $predicate:expr ) => {{
        std::compile_error!("This macro does not support the requested method");
    }};
}
