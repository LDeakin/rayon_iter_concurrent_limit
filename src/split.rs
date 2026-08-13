//! The [`ConcurrentLimit`] adaptor and its exact-split driver.
//!
//! See the [Implementation](crate#implementation) section of the crate documentation for why an
//! exact split is needed and what it buys over chunking. In short: [`Producer::split_at`] accepts
//! an arbitrary index, so splitting proportionally to a target piece count yields exactly `n`
//! pieces for any `n`, where [`rayon`]'s own driver only ever halves.

use rayon::iter::plumbing::{
    Consumer, Folder, Producer, ProducerCallback, Reducer, UnindexedConsumer,
};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

/// A parallel iterator which yields the items of its base iterator, split into exactly
/// `concurrent_limit` work items.
///
/// This struct is created by the [`concurrent_limit`] method on
/// [`ParallelIteratorConcurrentLimit`]. See its documentation for more.
///
/// [`concurrent_limit`]: crate::ParallelIteratorConcurrentLimit::concurrent_limit
/// [`ParallelIteratorConcurrentLimit`]: crate::ParallelIteratorConcurrentLimit
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Debug, Clone)]
pub struct ConcurrentLimit<I> {
    base: I,
    concurrent_limit: usize,
}

impl<I> ConcurrentLimit<I> {
    pub(crate) fn new(concurrent_limit: usize, base: I) -> Self {
        Self {
            base,
            concurrent_limit,
        }
    }
}

impl<I: IndexedParallelIterator> ParallelIterator for ConcurrentLimit<I> {
    type Item = I::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        self.drive(consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        self.base.opt_len()
    }
}

impl<I: IndexedParallelIterator> IndexedParallelIterator for ConcurrentLimit<I> {
    fn len(&self) -> usize {
        self.base.len()
    }

    /// Drive the iterator ourselves, splitting into exactly `concurrent_limit` pieces.
    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        if self.concurrent_limit == 0 {
            return self.base.drive(consumer);
        }
        let len = self.base.len();
        // `len` may be zero, so this cannot be a `clamp(1, len)`: that would panic on `min > max`.
        let num_pieces = self.concurrent_limit.min(len).max(1);
        self.base.with_producer(Callback {
            len,
            num_pieces,
            consumer,
        })
    }

    /// Hand the producer to a downstream adaptor that needs one of its own (`zip`, `enumerate`,
    /// `rev`, ...).
    ///
    /// That adaptor drives the split in this case, so the exact split above cannot be applied.
    /// Fall back to [`IndexedParallelIterator::with_min_len`], which still bounds the number of
    /// work items (rayon never splits below the minimum length) but can undershoot it, since
    /// rayon halves rather than splitting proportionally.
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        if self.concurrent_limit == 0 {
            return self.base.with_producer(callback);
        }
        let min_len = self.base.len().div_ceil(self.concurrent_limit).max(1);
        self.base.with_min_len(min_len).with_producer(callback)
    }
}

struct Callback<C> {
    len: usize,
    num_pieces: usize,
    consumer: C,
}

impl<T, C: Consumer<T>> ProducerCallback<T> for Callback<C> {
    type Output = C::Result;

    fn callback<P>(self, producer: P) -> Self::Output
    where
        P: Producer<Item = T>,
    {
        exact_split(producer, self.len, self.num_pieces, self.consumer)
    }
}

/// Recursively halve the *piece count*, splitting the producer proportionally.
///
/// This mirrors [`rayon::iter::plumbing::bridge_producer_consumer`], except that the split point
/// follows the target piece count instead of always being the midpoint, and the recursion stops
/// at a fixed depth instead of consulting an adaptive splitter.
fn exact_split<P, C>(producer: P, len: usize, num_pieces: usize, consumer: C) -> C::Result
where
    P: Producer,
    C: Consumer<P::Item>,
{
    if consumer.full() {
        // A short-circuiting consumer (e.g. `any`) already has its answer.
        consumer.into_folder().complete()
    } else if num_pieces <= 1 || len <= 1 {
        producer.fold_with(consumer.into_folder()).complete()
    } else {
        let left_pieces = num_pieces / 2;
        // Widened to avoid overflowing the product for very long iterators.
        let mid = ((len as u128 * left_pieces as u128) / num_pieces as u128) as usize;
        let mid = mid.clamp(1, len - 1);

        let (left_producer, right_producer) = producer.split_at(mid);
        let (left_consumer, right_consumer, reducer) = consumer.split_at(mid);
        let (left, right) = rayon::join(
            || exact_split(left_producer, mid, left_pieces, left_consumer),
            || {
                exact_split(
                    right_producer,
                    len - mid,
                    num_pieces - left_pieces,
                    right_consumer,
                )
            },
        );
        reducer.reduce(left, right)
    }
}
