# Steered graph expansion

`crates/ramp-expand`: expanding a graph into a world too large to copy, with a System 1
model (Jev, or any fast policy) deciding what to expand, and standing queries watching
the graph as it grows.

## The problem

Expansion starts from seeds (people, accounts, selectors) and pulls their neighbours from
outside sources into the graph, then their neighbours, and so on. Every fetch costs
something real: API quota, latency, money, analyst attention downstream. The world is
heavy-tailed. An employer has 100k employees and a VPN exit 100k accounts. One careless
expansion of such a hub floods the graph with entities that say almost nothing about the
seeds. This is the noise-to-signal problem in the request, and hardcoding depth and
breadth is the usual answer:

- **Depth limits** (`bfs(depth=3)`) cut paths that are one hop longer than guessed, and
  still expand every hub within reach.
- **Degree caps** (`rules(cap=1000)`) skip hubs, including the one moderately shared IP
  the answer runs through.
- **Every task wants different limits.** A link between two people, a watchlist alert,
  and an open-ended map each need their own settings, retuned for each world.

So the task is a **budgeted search** where the policy decides two things per step: which
discovered entity to expand next, and how much of it to take. The policy should be
conditioned on the task's intent (the prompt) and on live evidence from the graph.

## Architecture

```
            ┌───────────── Task (compiled from the prompt) ──────────────┐
            │ seeds · goal · relation/domain interest · goal shape ·     │
            │ standing queries · probes · budget                         │
            └──────────────────────────┬─────────────────────────────────┘
                                       ▼
  Source ──peek (counts per relation, free)──►  Expander  ◄──decide (batched)──  Policy
         ◄─fetch(id, relation, offset, limit)─  frontier                          (Jev)
                                                   │
                     one write txn per step ───────┤──────── mquery over the step's
                     (edge_batch, log, history)    │         new log entries
                                                   ▼         (standing queries)
                                       ramp-graph: LMDB log  ──►  projection
                                       (write-optimised)          (read-optimised,
                                                                   advanced per commit)
                                                   ▲
                       probes: LGQL rooted at the candidate about to be paid for
```

- **`Source`** is the world: an API, a corpus, a feed. `peek` returns per-relation edge
  counts without fetching, which most selector services can give. `fetch` pages one
  relation at a time. `fetch_where` returns only neighbours matching a `Filter`
  (below), if the source can filter (`can_filter`; the default is no).
- **`Task`** is the intent, and it is what a model is conditioned on:
  - interest per relation and per domain ("a financial link": accounts, logins and
    emails matter; employers don't);
  - the goal (connect two seeds, alert, explore);
  - the **shape** of a hit (person → account → ip → account → person), standing queries,
    probe templates;
  - a budget in fetched edges. There is no depth and no breadth.
- **`Candidate`** is what the policy sees about each discovered entity. It carries the
  source's counts, what is already fetched, hops and route cost from each side, what is
  known in the graph (degree, neighbours per domain), watched neighbours, standing-query
  matches, and probe hits.
- **`Policy::decide`** takes a batch of candidates and returns a priority and a
  `take: [(relation, how many)]` for each. Batches let a model run batched inference. A
  `Box<dyn Policy>` or `&mut P` plugs in unchanged.
- **The loop** pops the best candidate, runs its probes against the live graph, and lets
  the policy revise or defer before paying. Then it fetches what was asked, writes it in
  one transaction with `edge_batch`, and runs the standing queries over just the new log
  entries. Finally it re-scores only the entities whose situation changed, in one batch.

### Read-optimised next to write-optimised: yes

Expansion is a constant mix of small writes, one transaction per step, and reads: probes,
standing queries, and features. That is the workload the incrementally maintained
projection exists for (PR #26). The log stays the write-optimised source of truth. It is
durable, and it keeps history, so every entity carries the record of when and from what
it was expanded. The projection serves the policy's live reads at in-memory speed, and
each commit advances it by only that step's log entries. On the explore scenario (2M
world, 18.6k steps, a probe before each), identical decisions either way:

| probes read | µs per probe | probe time | whole run |
|---|---|---|---|
| projection | **79** | 1.5 s | **2.5 s** |
| LMDB | 366 | 6.8 s | 7.9 s |

Standing queries (`mquery`) read the log, because they are about what changed. They stay
cheap per step, because a step's new entries are few and `mquery` skips entries that
cannot change a match (PR #19).

## The System 1 seam, and the stand-in

`policy::Guided` is not Jev. It is a baseline built only from features and intent, with
no depth or breadth constants, to show that the seam carries enough information:

- **Cost, not hops.** Passing through an entity with `d` edges costs `ln(1 + d)`. A
  candidate's priority is its cheapest known route from a seed plus the cost of what is
  left to fetch of it. This is uniform-cost search in a "specificity" metric, the graph
  version of discounting very common words. It finds the least hub-heavy paths first, is
  bidirectional when there are two seeds, and needs no depth limit.
- **Intent.** It fetches only relations the task cares about. Cost is divided by domain
  interest, by signal (watched neighbours, alerts, probe hits), and by **shape fit**: an
  entity whose domain fits its distance from the seed, with more of the shape beyond it.
- **Breadth from budget.** It takes up to 1/64 of the remaining budget per relation per
  visit. A hub is taken in chunks, and gets cheaper to finish as it shrinks.

A model replaces `Guided` behind the same trait. It sees the same candidates plus the
prompt, and the expansion log is its training data: every decision and what came of it
is in the graph's history.

## Scenarios and results

`cargo bench -p ramp-expand --bench expand -- 2000000 400000`: a synthetic world of 2M
entities and 5.4M edges, across seven domains and seven relations, with per-domain
power-law degrees. The biggest employer and VPN exit each have about 104k edges. Signal
is planted in the world; a run stops when all of it has been written, or at the budget.
The table gives fetched edges to recover the signal ("fails" = budget spent without it).

| scenario | what it tests | BFS | rules cap 100 | rules cap 1k | rules cap 10k | Guided | Guided + pushdown |
|---|---|---|---|---|---|---|---|
| path-5hop | two people linked via account–ip(184)–account–email; both at the top employer | fails | 108 | 108 | 108 | **69** | 69 |
| path-7hop | seven hops across five domains, through a device shared by 485 accounts | fails | 12,850 | 67,117 | fails | **4,593** | 4,593 |
| pivot-busy | one-sided: from a watched person through an ip shared by ~4,800 accounts to a watched one; the seed's other accounts sit behind VPN hubs | fails | fails | fails | 132,391 | 185,743 | **42** (source scanned 3.0M) |
| alert | standing query: a watch-A person shares an account with a watch-B person; 0.3% of people are watch-B decoys | 104,192 | 158 | 158 | 158 | 204 | **2** (source scanned 110) |

Rules and BFS run at the depth each scenario needs; they were handed that.

Reading:
- **No fixed setting works across tasks.** Each rules setting fails at least one
  scenario. The one that solves pivot-busy (cap 10k) fails path-7hop outright. BFS fails
  three of four and floods the fourth.
- **Guided, with no knobs, is the only policy that solves all four.** On the paths it is
  cheapest outright: 1.6× under the best rules on the 5-hop, 2.8× on the 7-hop. On the
  two one-sided tasks it costs 1.3–1.4× the best-tuned rules.
- **pivot-busy is a needle in a haystack** without pushdown: nothing distinguishes the
  target's account from the ~4,800 others behind the busy IP until it has been expanded,
  so any policy pays for the haystack. With the target pushed down to the source, it
  costs 42 fetched edges. The alert fired through the 104k-account VPN exit, where the
  target also has an account (a genuine match of the standing query, hence 1/2 of the
  planted edges), and the source scanned 3.0M entities answering the semi-joins.
- "1st link" in the bench output shows the trap in connect tasks: both seeds share the
  top employer, so a meaningless hub link exists from the first step. Signal recovery,
  not connectivity, is the measure.

### What building it taught

- **Bidirectional search makes a hub in the middle harmless.** A first version of the
  busy-IP scenario linked two seeds through it. Every policy solved it cheaply: each side
  meets the busy IP from its own quiet accounts, and it never has to be expanded. A hub
  only hurts when it must be *expanded*, which is why pivot-busy is one-sided.
- **Chunking is not limiting.** Sampling a hub in chunks and re-queueing it at the same
  priority swallowed it whole, chunk after chunk. Pricing what is left to fetch fixed it.
- **Standing queries need selectivity-ordered evaluation.** A 5-node standing query fired
  on every new edge into a 100k-account VPN exit, and walked the whole hub before
  checking the selective `watch='S'` end. That is quadratic. The executor now fills a
  pattern towards the nearest selective slot first (any filter beyond a bare `type=`),
  in `query.rs`, which helps every query engine-wide.

## Predicate pushdown

A hub on the way to a hit is only expensive when it has to be fetched whole. A source that
can filter takes a `Filter`:

- `Attr { key, value }`: the neighbour has a property, such as a watchlist entry;
- `Via { rel, then }`: the neighbour has, by `rel`, a neighbour matching `then`, a
  semi-join one hop further out.

`Take` carries an optional filter. The expander uses `fetch_where` when the source can
filter, and an ordinary fetch when it cannot, so a policy never depends on support.
`Guided { pushdown: true }` composes filters from the task's intent: a **target** (what
the far end of a hit looks like) and the hit's **shape**. For a candidate on the shape
with more of it beyond, it wraps the target in one `Via` per remaining step:

- seed (person): accounts *that log in from an IP that has an account owned by a
  watch-T person*;
- IP: accounts *owned by a watch-T person*.

It then fetches only that relation, filtered. Two things keep the cost model honest:
- crossing an entity through a filter costs `ln(1 + what the filter returned)`, not
  `ln(1 + degree)`, since the filtered hop is as specific as its result;
- the policy prices a filtered expansion as a few edges only when the source reports it
  can filter.

Pushdown moves cost from fetching to the source. Both the 42 and the 2 above are fetched
edges; the source scanned 3.0M and 110 entities to answer. That is a good trade when
fetches are the scarce resource (quota, latency, downstream noise), and an indexed
source answers such semi-joins far cheaper than a scan. Path tasks name no target, so
they run as before. A reachability filter ("leads to seed B within k") is possible, but
few sources could answer it.

## On a real graph: ICIJ Offshore Leaks

`Guided` was built against our own generator, so the scenarios above share its
assumptions. `benches/icij.rs` loads the ICIJ Offshore Leaks database: officers,
entities, intermediaries, addresses and others, 2.0M nodes and 2.9M distinct edges. Its
13 relationship types are grouped into six relations: `officer_of`, `intermediary_of`,
`registered_address`, `same_as`, `similar` and `connected_to`. Its hubs are real: an
address with 37k links, an officer with 36k, intermediaries with 35k and 15k.
`Builder::empty` and `Builder::entity` load any graph into a `World` the same way.

Nothing is planted, so the bench samples tasks from the data (seeded, 50 per kind,
budget 200k):
- **connect:** two officers 4–6 hops apart. A run stops when they are joined. Besides
  the edges fetched, it scores the joining path: its cost (`ln(1 + degree)` summed over
  the entities it crosses) against the cheapest path in the whole graph, and the widest
  hub it crosses.
- **alert:** from an officer with at least 50 co-officers, a co-officer from the
  country rarest among them, as a standing query
  (`n(officer, value=seed)-n(entity)-n(officer, country=X)`). The shape is
  officer → entity → officer, and the target is `country = X`.

`cargo bench -p ramp-expand --bench icij -- DIR 50`:

| connect | solved | fetched (median) | p90 | joined by the cheapest path |
|---|---|---|---|---|
| bfs(depth=3) | 50/50 | 548 | 8,329 | 36/50 |
| rules cap 100 | 22/50 | — | — | 21/22 |
| rules cap 1k | 40/50 | 479 | — | 32/40 |
| rules cap 10k | 47/50 | 548 | 8,329 | 35/47 |
| **guided** | **50/50** | **101** | **2,196** | **46/50** |

| alert | solved | fetched (median) | p90 |
|---|---|---|---|
| bfs(depth=2) | 50/50 | 115 | 470 |
| rules cap 100 / 1k / 10k | 30 / 47 / 50 of 50 | 124 / 115 / 115 | — / 470 / 470 |
| guided | 50/50 | 120 | 450 |
| **guided+pushdown** | **50/50** | **4** | **4** |

What holds and what does not:
- **The cost model holds.** On connect tasks `Guided` is the only policy that solves
  every task, with 5.4× fewer edges than BFS at the median and 3.8× fewer at p90. It
  also finds the cheapest path more often (46/50 against 36/50). Degree caps fail
  between a fifth and more than half of the tasks: real paths do run through hubs (the
  cheapest path's widest hub has a median degree of 668).
- **The synthetic world overstated BFS's failure.** Offshore officers mostly have a
  handful of links, so BFS at depth 3 stays within budget here. Its cost shows in the
  tail, not in failures.
- **Without pushdown, alerts are a wash.** An officer's two-hop neighbourhood is small,
  so nothing beats fetching it (115–120 edges). With the target pushed down, every alert
  costs 4 edges.
- **Widest hub is not what the cost model minimises.** A path through one wide hub can
  beat one through several medium ones: `Guided`'s joining paths cross hubs as wide as
  BFS's (median 651 against 668). Penalising the widest hub, not only the sum, is a knob
  worth trying if analysts read paths that way.

## Concurrent fetches

A real source answers in 50–200 ms, so a run that waits for each fetch before choosing
the next pays one round trip per step: path-7hop's 624 steps would be minutes.
`Expander::parallel(k)` makes each round pop up to `k` candidates and fetch them
concurrently. Then it writes and re-scores them in order, and runs the standing queries
once.
- **Budget:** each pick's limits are reserved from the remaining budget when it is
  popped, so concurrent fetches cannot overspend it.
- **Speculation:** a round's extra picks are made before the first one's results are
  known. An extra far behind the best would probably not have been chosen once those
  results came in. So a round only takes candidates within the policy's
  `Policy::spread` of its best priority. For `Guided` that is 1 nat, a route at most
  ≈2.7× dearer. Without the bound, pivot-busy at `k = 16` spent 11,097 edges instead of
  42: the extra picks were chunks of the VPN hubs that the leading filtered fetch was
  there to avoid.

`World::set_latency` makes each fetch sleep, standing in for the round trip; peeks stay
free, since selector services return counts in bulk. At 2 ms per fetch:

| scenario | policy | k | fetched | rounds | wall | speedup |
|---|---|---|---|---|---|---|
| path-5hop | guided | 1 / 4 / 16 | 69 / 84 / 67 | 16 / 6 / 3 | 74 / 37 / 20 ms | 1 / 2.0 / **3.8×** |
| path-7hop | guided | 1 / 4 / 16 | 4,593 / 4,618 / 4,728 | 624 / 157 / 42 | 2,908 / 1,113 / 369 ms | 1 / 2.6 / **7.9×** |
| pivot-busy | guided+pushdown | 1 / 4 / 16 | 42 / 43 / 43 | 13 / 6 / 6 | ≈100 ms | 1.0× |
| alert | guided+pushdown | 1 / 4 / 16 | 2 / 32 / 32 | 2 / 2 / 2 | 4–5 ms | 1.0× |

- **Path tasks gain the most,** because their search is broad: 7.9× fewer round trips
  for 3% more edges.
- **The targeted tasks do not speed up.** They finish in a handful of rounds, and their
  time goes to the source evaluating filters (3M entities scanned on pivot-busy), not
  to waiting. At `k ≥ 4`, alert fetches 30 extra edges: siblings that ride along in the
  second round.

A step's own relation calls still run one after another; issuing them concurrently too
is the next cut, where a step takes several relations. `/expand` takes `parallel`, and
`spread` for a remote policy, whose priority scale only the client knows.

## Steering from outside: `/expand`

Jev runs outside the process, so `ramp-server` exposes the `Policy` seam over HTTP. The
client pulls work and needs no server of its own. The JSON is `ramp_expand::wire`.

| request | does |
|---|---|
| `POST /expand` | starts a run: `{"scenario", "policy"?, "budget"?, "entities"?, "seed"?, "cap"?, "timeout"?, "parallel"?, "spread"?}`. Returns `{"id", "graph", "task"}`: the task with domain and relation names, and a new graph of the collection the run writes into |
| `GET /expand/<id>?wait=s` | long-polls (default 25 s): `{"state": "ask", "seq", "context", "candidates"}`, `{"state": "done", "report"}`, or `{"state": "running"}` |
| `POST /expand/<id>` | `{"seq", "decisions": [{"priority", "take": [{"rel", "limit", "filter"?}]}]}`, one per candidate; 409 for a batch that is not open, 400 for the wrong count |
| `GET /expand/<id>/trace` | the labelled decision trace, JSON lines, once done |
| `DELETE /expand/<id>` | stops and forgets the run; its graph stays |

- **`policy`** is `remote` by default: the client decides every batch, including the
  re-decision after a probe. `guided`, `pushdown`, `bfs` and `rules` run in-process,
  for baselines and for generating traces.
- **A batch unanswered for `timeout` seconds (default 60) stops the run,** so a crashed
  client does not hold a thread forever.
- **The graph is ordinary.** While the run writes, the client can query it with LGQL,
  run its own standing queries from a log position (`GET /graph/<uuid>?q=...&start=n`),
  and probe what a candidate's neighbourhood looks like before deciding.
- **The world is the synthetic one** with its planted scenarios. That is a gym: ground
  truth is known, so traces are labelled. A real source behind the same endpoint is
  what the "real dataset" item below needs.

### The decision trace

Every run records each step in `Report::trace`: the candidate as the policy saw it, the
decision carried out, the budget spent before, and the edges the step added. A step's
**hits** are how many of those edges ended up in a hit:
- in a standing-query match, credited along the matched chain (consecutive matched
  entities, the written edges between them);
- or in ground truth the caller supplies (a scenario's planted signal).

A step is also on the **route** (`Report::route`) when its candidate leads to a hit:
the endpoints of every hit edge and, back through the steps that discovered them, each
entity whose fetch found the next. So a step that fetched the way to a hit, but not the
hit itself, is labelled too.

`wire::trace` writes it as JSON lines, one per step, with `hits` and `route`. Both
benches write one file per run with `RAMP_TRACE=dir` (`benches/icij.rs` labels a
connect run by the joining path it found, an alert run by the standing query's match),
and the server serves it per run. On five ICIJ tasks of each kind, the route adds
steps that only found the way: 5 of Guided's 103 connect steps beyond the 25 with
hits, 3 of pushdown's 13 alert steps beyond the 10.

## Next

1. **A learned policy.** Train Jev on the traces above, from both worlds: the plan is in [`jev.md`](jev.md).

Projection rebuilds no longer stall a long expansion: a graph the server opens
rebuilds in the background (see `docs/spec.md`).
