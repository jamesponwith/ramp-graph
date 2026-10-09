# ramp-graph spec

Rust rewrite of LemonGraph (upstream pinned at `f7363c6`, see `scripts/fetch-reference.sh`).
Keep the model, fix the bugs, go faster. Upstream refs are `file:line` in `reference/lemongraph/`.

## Model (unchanged from upstream)

- One LMDB file per graph. Every change appends one **log entry** with the next `LogId` (from 1):
  Node `(type, value)`, Edge `(type, value, src, tgt)`, Property `(parent, key, value)`, Deletion `(target)`.
- Records are never removed. Ending one (delete, or a new value for a property) sets its `next` to the ending entry's ID.
  A deletion cascades to properties (recursively) and, for a node, its edges. Every cascaded record gets `next = deletion ID`.
  The cascade writes no extra log entries.
- **Historical view** `before = Some(b)`: an entry is visible iff `id < b && (next == 0 || next >= b)`.
  `before >= next_id` is the same as `None` (live).
- Identity (uniqueness among live entries): node `(type, value)`; edge `(src, tgt, type, value)`; prop `(parent, key)`.
  `parent == 0` is the graph itself. Properties can be attached to properties.
- Strings (types, values, keys) are interned byte strings. `StrId 0` is the empty string.
- Non-logged `kv` store: `(domain, key) → value`, no history.
- Streaming seed-set expansion is the main use case. A client keeps a log bookmark and queries `[pos, ∞)` for newly matching patterns.

## Storage layout (`crates/ramp-graph`)

Keys are tuples of order-preserving varints (`varint.rs`: a length byte, then minimal big-endian bytes). Index values are empty.

| table | key → value |
|---|---|
| `log` | `[id]` → `tag, next, fields…` |
| `scalar` | `StrId` (u64 BE) → bytes |
| `scalar_idx` | `fnv64(bytes) ++ StrId` (BE) → '' (collisions re-checked against `scalar`) |
| `node_idx` | `[type, val, id]` |
| `edge_idx` | `[type, val, src, tgt, id]` |
| `prop_idx` | `[parent, key, id]` |
| `srcnode_idx` / `tgtnode_idx` | `[node, type, edge]` |
| `txnlog` | `[end]` → `[nodes, edges]`: live counts after each write txn; `end` = first ID after it |
| `kv` | `[domain] ++ key` → value |

Lookup at view `b`: seek to the largest index key `< prefix ++ [b]`, check the prefix, then check visibility.
Counts at view `b`: take the `txnlog` entry with the greatest `end ≤ b`, then replay the log from `end` to `b`.

### Deliberate deviations from upstream

- **Not byte-compatible** with upstream files. If we need to migrate, write an importer that replays the upstream `log` table.
- `txnlog` is keyed by `end`. This replaces the custom `magic_txnlog_cmp` comparator (`lib/lemongraph.c:117`), so no custom comparator is needed.
- String index uses FNV-1a 64 with prefix scan instead of crc32 + DUPSORT.
- **Map size: the file plus 1 GiB of headroom**, as in upstream (`lib/db.c:344-404`). Before each write txn the map grows if less than 1 GiB is free. So one write txn can add at most about 1 GiB, as upstream.
  - LMDB resizes only while no txn of the env is active in the process. Every txn holds a `Ticket` on a counter + condvar (upstream `db->txns`). A resize waits for the count to reach zero and blocks new txns meanwhile.
  - A thread must therefore not start a write txn while it holds another txn on the same graph.
  - History: PR #2 shipped a fixed 1 TiB map. With a 48-bit address space (128 TiB) that allowed only ~127 open graphs per process, and creates beyond that failed with ENOMEM.
- One `Txn` type, read or write. Writes in a read txn return `GraphError::ReadOnly`.
  Resolving an existing node/edge works read-only, which `test.py:242` relies on.
- Nested write txns via `Txn::nested`. Dropping the child aborts it; committing folds its count deltas into the parent.
- No `assert`-abort on storage errors: everything returns `Result`.

## Upstream bugs: fixed in core

- Node-deletion count replay subtracted **every** live edge in the graph instead of the node's cascaded edges (`lib/lemongraph.c:1646-1653`).
  Now it counts the node's edges whose `next == deletion`.
- Deleting an already-deleted entry re-cascaded and double-decremented counts. Now it returns `NotFound`.
- Edges could reference non-nodes or dead nodes, and properties could hang off dead parents (no validation). Both are now validated.
- `Both`-direction edge iteration yielded a self-loop twice (`lib/lemongraph.c:1286`). Now yielded once.
- Lookup by ID returned deleted objects with no way to tell. `Entry.next` is exposed and `live_at()` checks it.

## Upstream bugs: to fix in later layers

**LGQL / streaming** — all fixed in `lgql.rs`/`query.rs` (regression tests in `tests.rs`):

**LGQL** (`MatchLGQL.py`)
- Hex literals never parse; NUM is tried first (`:42,55`).
- Floats are truncated to int (`:136`).
- `\/` in a regex is broken (`:25`).
- Two `~` tests on the same key intersect into "cannot match" (`:85`). They should AND.
- `true == 1` conflation.
- Numeric trailer aliases skip inferred slots: `n()-n()-n()` numbers its slots 1, 2, 4 (`:373`).
- Cross-type range comparisons raise inside evaluation and abort the whole query.
- `ID=` seeds out of range raise KeyError → 500.

**Streaming** (`query.py`)
- Edge trigger shares one `seen` set across src and tgt, so the target is missed (`:105`).
- Property triggers have no dedup (`:242`).
- Deletions never trigger.
- `KeyError` is swallowed for the whole entry (`:153`).
- `start > stop` scans the whole tail.

**REST/collection**
- First-chunk errors become UnboundLocalError 500s (`server/__init__.py:236`).
- PUT upload is broken (`:511`).
- `Accept`/`Content-Type` parameters are rejected.
- Partial `send`.
- No body-read timeout.
- Crawl emits duplicates.
- `mktime` is applied to a UTC tuple.
- Every graph open does an index write txn.
- DELETE/write race.
- Lock keys use salted `hash()`.
- Auth is skipped for empty files and when no `user` is sent.
- `/status` has no permission check.
- `view?style=` allows path traversal.
- `/exec` is arbitrary code execution. **Drop it.**

**Python helpers**
- Fifo rebase is broken.
- `_uints_string` decodes the wrong buffer.
- Indexer `pack('=i', crc32)` overflows.
- `updates()` crashes on deletions.

## LGQL (`lgql.rs` parser, `query.rs` executor; upstream grammar `MatchLGQL.py:9-36,220-428`)

```
query   := obj { link obj } [ ',' trailer ]
link    := '-' | '->' | '<-' | '<->'
obj     := '@'* ('n'|'N'|'e'|'E') [':' alias {',' alias}] '(' [test {',' test}] ')'
           '@' = omit from result; upper case = slot need not be unique in the chain
test    := keypath [op value]          bare keypath = exists
keypath := key {'.' key}               key = bareword | 'str' | "str"
op      := = != ~ !~ : !: < <= > >=    value or [list]; regex /re/imsx; types boolean|string|number|array|object
trailer := (index | alias) '(' tests ')' {',' …}   merges tests into the referenced slot(s)
```
- Adjacent same-kind objects get an inferred slot of the other kind (`n()->n()` ≡ `n()->@e()->n()`).
- Seed slot choice, best rank first:
  1. edge `ID=`
  2. node `ID=`
  3. node type + value
  4. edge type + value
  5. node type
  6. edge type
  7. any node
  8. any edge
- Expansion walks both directions from the seed with uniqueness filtering.
- Streaming: a chain is emitted when a test on some slot flips false → true at a log entry in `[start, stop]`, or when a newly created object satisfies a static test. The chain is evaluated at `entry.id + 1`.

### Implementation decisions
- **Property values are msgpack** (`value.rs`), as upstream's server stores them. Decoding is strict: one value and no trailing bytes.
  Raw-byte properties resolve to nothing, so they fail every test.
- Node/edge `type` and `value` are read as UTF-8 strings (lossy). So `n(value=1)` never matches the value `"1"`, as upstream.
- **Native keys implemented:**
  - both: `ID`, `type`, `value`, `typeID`, `valueID`
  - nodes: `edge_count`, `inbound_count`, `outbound_count`
  - edges: `srcID`, `tgtID`, `src.*`, `tgt.*`
- **List-valued native keys** are also implemented:
  - `edges`, `edgeIDs`, `inbound`, `inboundIDs`, `outbound`, `outboundIDs`, `neighbors` and `neighborIDs` are ID lists. Upstream yields objects for the non-`IDs` forms, and LGQL could only test those for existence.
  - `neighbor_count` is a number. `neighbor_types` is `{type: count}`.
  - Neighbours exclude self and are distinct.
- **Filters are a plain AND.** Upstream's set "munging" was only an optimisation, and it caused the `~`-intersection bug.
  Seed choice still reads `ID=`/`type=`/`value=` sets.
- **Comparisons:**
  - `1 == 1.0`; booleans are never numbers.
  - Ranges compare number with number or string with string; any other pair is false (upstream crashed).
  - Non-strings fail both `~` and `!~`.
- Results go to a callback `sink(pattern_idx, chain) -> keep_going`. Results are not buffered, and the callback provides the limit.
- **Streaming candidates per log entry:**
  - a new node
  - a new edge, plus its endpoints
  - a property's parent, plus the parent's edges if any edge slot tests `src.*` or `tgt.*`
  - a deletion: the deleted edge's endpoints, the deleted property's parent, or the deleted node's neighbours

  A candidate is reported when a slot matches at `x+1` but did not at `x`. Chains are deduplicated within each entry.
  Re-matching after an unmatch is reported again, as upstream.

## REST (`crates/ramp-server`; upstream `RESTAPI`, `server/__init__.py`)

`ramp-server [-i ip] [-p port] [dir]`: a single process. axum is used only as the transport. One fallback handler reads the body (1 GiB cap, 60 s timeout) and runs synchronous routing (`api.rs`) on a blocking thread, because LMDB txns are thread-bound.
- `store.rs`: one open `Graph` per UUID (LMDB forbids double opens).
  - At most `-n` (default 64) stay open while idle, closed least recently used first. A graph a request holds is never closed.
  - Each open graph costs 3 fds and ~2 MiB of RAM: LMDB preallocates a 2 MiB write-txn dirty list per env.
  - The in-memory index (status + graph props) is rebuilt at startup by opening every `<uuid>.db`.
- `input.rs`: `as_dict`/`format_edge` rendering, POST-body application with upstream `merge_values`, and the depth/cost adapters (scanning forward, so cascades reach a fixpoint).

**Wire-compatible:**
- endpoints, JSON/msgpack by `Accept`, stream framing (JSON array / concatenated msgpack)
- `X-lg-maxID`/`X-lg-updates`/`x-lg-sync`
- error body `{code, reason, message}`
- the dump and listing shapes
- seeds in kv `lg.seeds`, REST kv in `lg.restobjs`
- reserved-key dropping, cost rules

**Deliberate differences:**
- **Removed:**
  - `/graph/exec`, `/graph/<uuid>/exec` (RCE).
  - `/view`, `/static`, `/favicon.ico` (UI assets; `/view` had path traversal).
  - Gzip request bodies.
  - `-s`/`-m` nosync flags and the sync daemon: commits are durable.
- **Permissions:**
  - A graph with a `roles` property requires `user` (and a matching `role` if given) on **every** endpoint, `/status` included.
  - Graphs without `roles` are open.
  - Listings show exactly the graphs the caller may open.
  - `user`/`role` are unauthenticated claims: deploy behind something that sets them.
- **Status codes:**
  - Unknown paths are 404 (upstream 400).
  - Missing `type`/`value`, bad IDs and bad numbers are 409/400, not 500.
  - `/kv/<uuid>/<missing>` is 404 (upstream 500).
  - `Content-Type` parameters (`; charset=utf-8`) are accepted.
- **Behaviour:**
  - `PUT /reset/<uuid>` rebuilds in place in one txn (atomic). The file does not shrink.
  - Streaming results render each chain as of the log entry that matched (upstream's documented "frozen" semantics).
  - Crawl emits each node/edge once.
  - `created_after`/`created_before` accept `YYYY-MM-DD[THH:MM[:SS[.f]]][Z]` (UTC) only.
- **Streaming:** bodies up to 1 MiB are buffered and sent with `Content-Length`; errors before that still get their status. Larger bodies stream in 64 KiB chunks from the open read txn.
  - Applies to dumps, crawl, d3, query results, listings and seeds.
  - The channel to the client is bounded, so a slow client stalls its handler rather than growing memory.
  - An error, or the client going away, mid-stream cuts the connection.
- **Ceilings (ponytail):**
  - Crawl collects its subgraph before writing.
  - The index is rebuilt at startup: 0.22s for 1000 graphs, so persisting it is not worth it.
  - Uploads are held in memory (1 GiB cap).

## Roadmap

1. ✅ Storage core: nodes, edges, props, delete cascade, history, counts, kv, nested txns, reset, snapshot.
2. ✅ Value layer: msgpack property values (`value.rs`), `fifo_push`/`fifo_pop`/`fifo_len` on the kv store, `log(start, end)` scan, `update_id`.
   - No sset: it is kv with empty values (`kv_put(m, b"")`/`kv_get`/`kv_del`/`kv_iter`).
   - No upstream `uint`/`uints` serializers; the varints are internal.
   - Fifo fixes: indexes are read from storage on each push, so two handles never clobber each other; no broken rebase.
3. ✅ LGQL parser + ad-hoc executor + streaming `mquery`. Edge-type pushdown and per-query key resolution done (PRs #4, #5).
4. ✅ REST server (`ramp-server`): all upstream endpoints except exec/UI, collection index, permissions, depth/cost adapters.
5. ✅ Benchmark `crates/ramp-graph/benches/insert.rs` (`just bench`): a port of upstream `bench.py`.

   **Results.** ramp-graph measured 2026-10-08 at `134cb2f` (after the LMDB map-growth change), i5-1135G7 laptop, idle (load avg 0.09). Upstream numbers are from 2026-10-07 on the same machine, built from the pinned rev on CPython 3.14 (no PyPy available). Three interleaved rounds, best of each (spread ≤3%):

   | phase | upstream | ramp-graph | speedup |
   |---|---|---|---|
   | 1M nodes | 150k/s | 584k/s | 3.9× |
   | 1M props | 375k/s | 542k/s | 1.4× |
   | 1M edges | 110k/s | 229k/s | 2.1× |
   | commit | — | 69ms | |
   | file | 293 MiB | 300 MiB | ≈ |

   Growing the map before each write txn costs nothing measurable: every phase is within noise of the fixed-1-TiB numbers from 2026-10-07 (573k/s, 536k/s, 220k/s).

   **Write-path optimisations.** Next log/string IDs are cached per write txn, as upstream does (nested txns inherit and fold back the cache). Each string is looked up once per insert.

   **Query benchmark** `crates/ramp-graph/benches/query.rs`. 1M nodes with one property each, plus 1M deterministic edges. Upstream ran an identical Python mirror on CPython 3.14, on the same idle machine. Result counts match exactly:

   | query | upstream | ramp-graph | speedup |
   |---|---|---|---|
   | 10k point lookups | 0.689s | 0.047s | 15× |
   | `n(type="node3")` (200k) | 0.787s | 0.089s | 8.8× |
   | `n(prop2="value2")` (full scan) | 3.957s | 0.671s | 5.9× |
   | `e(type="edge3")` (200k) | 0.787s | 0.089s | 8.8× |
   | `n(type="node1")->e()->n()` | 5.479s | 0.344s | 16× |
   | `n(type="node1")-e(type="edge2")-n()` (200k) | not measured | 0.528s | |
   | `n(type="node1")-n()-n()` (400k) | 30.0s | 2.07s | 14× |
   | node count at a mid-txn view | 8ms | 9ms | ≈ |

   The typed-edge query was added in PR #4 after the upstream mirror was run; its A/B against the untyped expansion is in that PR (0.720s → 0.630s at the time).

   A count at a mid-txn view replays the log from the last txn boundary, as upstream does. That is slow only for a view inside one huge transaction.

   **Not yet measured:**
   - REST throughput
   - a profile of the props phase: per-insert parent-liveness check, msgpack round trip
