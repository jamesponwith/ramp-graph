# ramp-graph

A fast, log-structured, transactional graph database in Rust. It rewrites the NSA's
[LemonGraph](https://github.com/NationalSecurityAgency/lemongraph): the same model, its bugs fixed, and much faster.

- **Log-based history.** Every change appends a log entry, and nothing is overwritten, so you can query the graph as it stood at any log position.
- **One LMDB file per graph.** ACID transactions, including nested ones, and no server needed when you embed it.
- **LGQL**, LemonGraph's pattern language (`n(type="person")->e(type="knows")->n()`). Run it ad hoc, or as standing queries that stream new matches as the graph grows.
- **REST server** compatible with upstream's API, plus `/expand` for steered expansion.
- **Steered expansion** (`ramp-expand`) explores sources too large to copy within a fetch budget, with a pluggable policy deciding what to expand next.

## Performance

1M nodes, 1M properties, 1M edges on an i5-1135G7 laptop. Every number is reproduced by a bench or script in this repo.

| workload | upstream LemonGraph | ramp-graph | speedup |
|---|---|---|---|
| node inserts | 150k/s | 780k/s | 5.2× |
| edge inserts (`edge_batch`) | 110k/s | 845k/s | 7.7× |
| 10k point lookups | 0.689 s | 0.048 s | 14× |
| `n(type="node1")->e()->n()` | 5.479 s | 0.317 s | 17× |
| `n(type="node1")-n()-n()` (400k paths) | 30.0 s | 2.04 s | 15× |

ramp-graph is also ahead of [LadybugDB](https://github.com/LadybugDB/ladybug) (the community fork of Kùzu) on every query benchmarked, at 1 thread and at 8. On the real ICIJ Offshore Leaks graph (2M nodes), guided expansion links its sample tasks using 5.4× fewer edges than breadth-first search at the median.
The full scorecard, with the caveats, is in [`docs/spec.md`](docs/spec.md).

## Quick start

```bash
cargo run --release -p ramp-server -- -p 8000 graphs/
```

Create a graph:

```bash
curl -si -X POST localhost:8000/graph -H 'Content-Type: application/json' -d '{
  "nodes": [{"type": "person", "value": "alice"}, {"type": "person", "value": "bob"}],
  "edges": [{"type": "knows", "value": "",
             "src": {"type": "person", "value": "alice"},
             "tgt": {"type": "person", "value": "bob"}}]
}'
# HTTP/1.1 201 Created
# location: /graph/<uuid>
```

Query it with LGQL:

```bash
curl -sG localhost:8000/graph/<uuid> --data-urlencode 'q=n(type="person")->e(type="knows")->n()'
```

## Layout

| crate | what it is |
|---|---|
| [`ramp-graph`](crates/ramp-graph) | storage engine, LGQL parser and executor, in-memory projection for parallel queries |
| [`ramp-server`](crates/ramp-server) | REST service (axum), upstream-compatible |
| [`ramp-expand`](crates/ramp-expand) | budgeted, policy-steered graph expansion with standing queries |

Design docs: [`docs/spec.md`](docs/spec.md) covers the architecture, deviations from upstream, the bugs fixed and the benchmarks. [`docs/expansion.md`](docs/expansion.md) covers steered expansion.

## Development

```bash
just setup   # tools + git hooks
just ci      # fmt, typos, clippy -D warnings, nextest, doctests, rustdoc, cargo-deny, machete
just bench   # benchmarks
```

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
