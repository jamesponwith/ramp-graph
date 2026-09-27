# ramp-graph

Rust rewrite of LemonGraph (NSA): a log-based transactional graph DB (nodes/edges/properties)
backed by a single LMDB file, with historical views by log position, a query language (LGQL),
and a REST service. Goals: faster, fix upstream bugs, keep the core model.

## Upstream reference
- Run `./scripts/fetch-reference.sh` to get the pinned upstream source at `reference/lemongraph/`.
- It is gitignored and **read-only** — never edit it, never copy it into tracked files wholesale.
- Map: C core in `lib/lemongraph.c` + `lib/db.c`; Python API `LemonGraph/__init__.py`;
  query language `LemonGraph/MatchLGQL.py`; REST server `LemonGraph/server/`, `LemonGraph/httpd.py`;
  REST docs `RESTAPI`; tests `test.py`; benchmark `bench.py`.
- When porting behavior, cite the upstream file:line in the PR description, not in code comments.

## Workflow
- First time: `just setup` (installs tools, enables `.githooks`: pre-commit = fmt/typos/clippy, pre-push = `just ci`).
- `just ci` must pass before anything is done (fmt, taplo, typos, clippy `-D warnings`, nextest, doctests,
  rustdoc, cargo-deny, machete). Deeper checks: `just cov`, `just mutants`, `just miri`, `just hack`.
- Toolchain pinned in `rust-toolchain.toml`; lints live in `[workspace.lints]` in the root `Cargo.toml` —
  every crate uses `[lints] workspace = true`. Add deps to `[workspace.dependencies]`, then `dep.workspace = true`.
- Lints are the spec, not suggestions. Fix the code; don't silence it. When an exception is genuinely right,
  use `#[expect(lint, reason = "...")]` on the narrowest item — never `#[allow]`, never crate-wide.
- Every `unsafe` block needs a `// SAFETY:` comment; `unsafe` code gets a Miri-run test.
- No `unwrap`/`expect`/`panic`/indexing/`as` casts in non-test code: return errors, use `get`, `TryFrom`.
