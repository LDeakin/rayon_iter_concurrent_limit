default:
    @just --list

# Format the code. Pass `--check` to verify without writing.
fmt *ARGS:
    cargo fmt --all -- {{ ARGS }}

# Type check every target
check:
    cargo check --all-targets

# Clippy, warnings denied
lint:
    cargo clippy --all-targets -- -D warnings

# Build the documentation and its dependencies, warnings denied. Pass `--open` to view it in a browser.
doc *ARGS:
    RUSTDOCFLAGS="-D warnings" cargo doc {{ ARGS }}

# Run benchmarks. Pass `-- map_collect` to filter.
bench *ARGS:
    cargo bench --bench concurrent_limit {{ ARGS }}

# Run the tests.
test:
    cargo nextest run --all-targets
    cargo test --doc

# Everything CI runs
ci: (fmt "--check") check lint (doc "--no-deps") test
