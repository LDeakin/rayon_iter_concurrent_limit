mod common;

use common::{pool, Concurrency};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::ParallelIteratorConcurrentLimit;

const DUR: core::time::Duration = core::time::Duration::from_millis(10);

/// The example from `README.md`, instrumented to check the concurrency it claims.
#[test]
fn readme() {
    pool().install(readme_example);
}

fn readme_example() {
    let alloc_sum = Concurrency::default();
    let div = Concurrency::default();

    let concurrent_limit = 2;
    const N: usize = 1000;
    let output = (0..100)
        .into_par_iter()
        .concurrent_limit(concurrent_limit) // limits everything chained after it
        .map(|i| {
            alloc_sum.record(|| {
                let alloc = vec![i; N]; // max of 2 concurrent allocations
                std::thread::sleep(DUR);
                alloc.into_par_iter().sum::<usize>() // runs on all threads
            })
        })
        .map(|alloc_sum| {
            div.record(|| {
                std::thread::sleep(DUR);
                alloc_sum / N // max of 2 concurrent executions
            })
        })
        .collect::<Vec<usize>>();
    assert_eq!(output, (0..100).collect::<Vec<usize>>());
    assert_eq!(alloc_sum.max(), concurrent_limit);
    assert_eq!(div.max(), concurrent_limit);
}
