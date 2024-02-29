mod common;
use core::time;
use std::sync::atomic::AtomicUsize;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use rayon_iter_concurrent_limit::iter_concurrent_limit;

use common::{calc_active_operations, incr_active_operations};

const DUR: time::Duration = core::time::Duration::from_millis(10);

fn iter_concurrent_limit_update(concurrent_limit: usize) {
    let threads_active = AtomicUsize::new(0);
    let threads_active_max = AtomicUsize::new(0);
    let threads_active_inner = AtomicUsize::new(0);
    let threads_active_inner_max = AtomicUsize::new(0);
    let output = iter_concurrent_limit!(concurrent_limit, 0..10, update, |i: &mut usize| {
        incr_active_operations(&threads_active);
        std::thread::sleep(DUR);
        (0..10).into_par_iter().for_each(|_| {
            incr_active_operations(&threads_active_inner);
            std::thread::sleep(DUR);
            calc_active_operations(&threads_active_inner, &threads_active_inner_max);
        });
        calc_active_operations(&threads_active, &threads_active_max);
        *i = *i * 2;
    })
    .collect::<Vec<_>>();
    assert_eq!(
        output,
        (0..10).into_iter().map(|i| i * 2).collect::<Vec<_>>()
    );
    assert_eq!(threads_active_max.into_inner(), concurrent_limit);
    if cfg!(not(feature = "ci")) {
        assert!(threads_active_inner_max.into_inner() > concurrent_limit);
    }
}

#[test]
fn iter_concurrent_limit_update_1() {
    iter_concurrent_limit_update(1);
}

#[test]
fn iter_concurrent_limit_update_2() {
    iter_concurrent_limit_update(2);
}

#[cfg_attr(feature = "ci", ignore)]
#[test]
fn iter_concurrent_limit_update_4() {
    iter_concurrent_limit_update(4);
}
