# List recipes
default:
    @just --list

# One-time: install tools, enable git hooks
setup:
    cargo binstall -y cargo-nextest cargo-deny cargo-machete cargo-llvm-cov cargo-mutants cargo-hack taplo-cli typos-cli sccache
    git config core.hooksPath .githooks

# Everything CI runs, in order (also the pre-push hook)
ci: fmt-check typos lint test doc deny machete

fmt:
    cargo fmt --all
    RUST_LOG=warn taplo fmt

fmt-check:
    cargo fmt --all --check
    RUST_LOG=warn taplo fmt --check

typos:
    typos

lint:
    cargo lint

test *args:
    cargo nextest run --workspace --locked --no-tests=warn {{args}}  # drop --no-tests once tests exist
    cargo test --workspace --doc --locked

doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked

deny:
    cargo deny check

machete:
    cargo machete

# Every feature combination compiles and passes clippy
hack:
    cargo hack clippy --workspace --all-targets --feature-powerset -- -D warnings

cov:
    cargo llvm-cov nextest --workspace --html --open

# Mutation testing: finds code the tests don't actually check
mutants *args:
    cargo mutants --workspace --test-tool nextest {{args}}

# UB detection for unsafe code (needs nightly + miri component)
miri *args:
    cargo +nightly miri nextest run --workspace {{args}}

bench *args:
    cargo bench --workspace {{args}}

cache-stats:
    sccache --show-stats

# Fetch pinned upstream LemonGraph into reference/
reference:
    ./scripts/fetch-reference.sh
