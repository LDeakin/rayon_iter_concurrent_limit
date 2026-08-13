# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
 - Add the `ParallelIteratorConcurrentLimit` extension trait for `rayon::iter::IndexedParallelIterator`, with a single method: `concurrent_limit`
   - `iterator.concurrent_limit(n)` limits the concurrency of *every* method chained after it, and composes with any `rayon` method
   - Example: `iter_concurrent_limit!(2, 0..100, map, op)` becomes `(0..100).into_par_iter().concurrent_limit(2).map(op)`
   - Unlike the macro, closure signatures do not have to be made explicit
 - Add the `ConcurrentLimit` parallel iterator adaptor returned by `concurrent_limit`
   - It is an `IndexedParallelIterator`, so indexed methods (`zip`, `enumerate`, `collect_into_vec`, ...) remain available
   - It splits the iterator into exactly `concurrent_limit` work items using a custom driver built on `rayon::iter::plumbing`, so the concurrency limit is reached exactly and nothing is allocated
   - The limit stays exact through consumer-wrapping adaptors (`map`, `filter`, `filter_map`, `update`, ...), but chaining an adaptor that needs a producer of its own (`zip`, `enumerate`, `rev`, ...) hands over the splitting, and the limit degrades from exact to an upper bound
 - Add the `concurrency_comparison` example, which measures allocation and achieved concurrency for chunking versus the exact split

### Removed
 - **Breaking**: Remove the `iter_concurrent_limit!` macro, superseded by `ParallelIteratorConcurrentLimit::concurrent_limit`
   - The macro limited concurrency by chunking. That materialised the entire iterator into `concurrent_limit` `Vec`s (8 MB for a 1M-element iterator), produced an unindexed iterator, and could fall short of the requested concurrency, since rounding the chunk *size* up rounds the chunk *count* down (e.g. 10 items with a limit of 8 gave 5 chunks, so only 5 concurrent executions)
 - **Breaking**: Remove `iter_subdivide`, which existed to implement the chunking of the `iter_concurrent_limit!` macro
   - Use `concurrent_limit` to limit concurrency. To chunk an iterator for its own sake, call `rayon`'s `IndexedParallelIterator::chunks` directly: `iter_subdivide(n, iterator)` is `iterator.chunks(iterator.len().div_ceil(n).max(1))`

### Changed
 - **Breaking**: Bump the MSRV to 1.75 (from 1.63)
 - Remove some needless parentheses around ranges in docs and tests

## [0.2.0] - 2024-02-29

### Changed
 - **Breaking**: The iterator argument in the `iter_concurrent_limit!` macro now expects an expression implementing `rayon::iter::IntoParallelIterator`
   - Example: use `iter_concurrent_limit!(n, (0..100), method, op)` instead of `iter_concurrent_limit!(n, (0..100).into_par_iter(), method, op)`
   - The `into_par_iter()` iterator must implement `rayon::iter::IndexedParallelIterator`
   - Some iterator methods require the iterator argument to also implement `std::iter::IntoIterator` for fast paths
 - Add fast paths in `iter_concurrent_limit!` if `concurrent_limit` is 1 for methods: `for_each`, `try_for_each`, `any`, `all`
   - These paths do not start additional `rayon` work items

## [0.1.0] - 2024-02-19

### Fixed
 - Remove incorrect panics docs for `iter_subdivide`

## [0.1.0-alpha4] - 2024-02-18

### Changed
 - Minor documentation improvements
 - Add fast paths avoiding chunking for various methods in `iter_concurrent_limit` if `concurrent_limit` is zero

### Fixed
 - Fixed `iter_subdivide` if supplied with an empty iterator
 - Handle `num_chunks` of zero in `iter_subdivide`

## [0.1.0-alpha3] - 2024-02-05

### Changed
 - **Breaking**: rename `chunks_concurrent_limit` to `iter_subdivide`
 - Various documentation improvements

## [0.1.0-alpha2] - 2024-02-04

### Added
 - Add `chunks_concurrent_limit_map` tests
 - Add doc example with equivalent code to macro

### Changed
 - **Breaking**: Swapped the argument order of `iterator` and `concurrent_limit` in `chunks_concurrent_limit`

## [0.1.0-alpha] - 2024-02-04

### Added
 - Initial alpha release for review

[unreleased]: https://github.com/LDeakin/rayon_iter_concurrent_limit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/LDeakin/rayon_iter_concurrent_limit/releases/tag/v0.2.0
[0.1.0]: https://github.com/LDeakin/rayon_iter_concurrent_limit/releases/tag/v0.1.0
[0.1.0-alpha4]: https://github.com/LDeakin/rayon_iter_concurrent_limit/releases/tag/v0.1.0-alpha4
[0.1.0-alpha3]: https://github.com/LDeakin/rayon_iter_concurrent_limit/releases/tag/v0.1.0-alpha3
[0.1.0-alpha2]: https://github.com/LDeakin/rayon_iter_concurrent_limit/releases/tag/v0.1.0-alpha2
[0.1.0-alpha]: https://github.com/LDeakin/rayon_iter_concurrent_limit/releases/tag/v0.1.0-alpha
