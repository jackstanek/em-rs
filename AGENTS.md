# AGENTS.md

`em-rs` is a dependency-free Rust library implementing the generic
expectation-maximization (EM) algorithm. All library code lives in
`src/lib.rs`; `examples/` holds two runnable demo models. Single `main`
branch, no CI.

## Build & verify

- `cargo build` — the real check for library changes.
- There are **no unit tests**: `cargo test` executes 0 tests (lib + doc).
  The examples are the only end-to-end coverage, so treat them as smoke
  tests and keep them runnable:
  - `cargo run --example bern_mixture`
  - `cargo run --example gmm`
  Both must print `Converged: true`.

## Formatting / linting

- The committed tree is **not** `rustfmt`-clean (`cargo fmt --check` reports
  import-order diffs). Match the surrounding file style; do not run
  `cargo fmt` over the whole tree — it produces churn diffs.
- `cargo clippy` passes except one pre-existing warning in `src/lib.rs`
  (`delta.map_or(true, ...)` → `is_none_or`). Keep new code warning-free
  without "fixing" unrelated lines.

## Architecture

- The only extension point is the `Model` trait in `src/lib.rs`:
  - `zero_stats` — empty sufficient-statistics object shaped to the params
  - `accumulate` — E-step for **one datum**: fold its stats into `acc` and
    return `log p(datum | params)`
  - `merge` — combine stats partitions
  - `maximize` — M-step: argmax of Q over aggregate stats → new params
- `em(model, data, initial_params, tol, max_iter)` loops: accumulate over all
  data → `maximize` → stop when the log-likelihood delta is under `tol`.
  Returns `Fit { params, log_lik, converged }`; `log_lik` includes the
  initial value, and `converged` is false if `max_iter` is exhausted.
- `debug_assert!` in `em()` (debug builds only) fires if log-likelihood
  decreases — almost always a bug in a model's `accumulate` (e.g. missing
  log-sum-exp). Verify with default debug builds (`cargo run`); it compiles
  out of `--release`.
- `examples/gmm.rs` is the fuller reference `Model` implementation; the
  `bern_mixture.rs` `merge` is a no-op stub (fine in the single-threaded
  examples, wrong for real use).

## Install / roadmap

- Not published to crates.io; README says to install via
  `cargo add --git 'https://github.com/jackstanek/em-rs.git'`.
- README roadmap (rayon multi-threaded E-step, built-in GMM/HMM/PCFG
  models) is unimplemented.
