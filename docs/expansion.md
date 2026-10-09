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

## Next

1. **A learned policy.** Train Jev on logged expansions: the candidates, the decision,
   and whether what was fetched ended up in a hit.
2. **Pipelined fetches.** Real sources have latency. Fetch the top-k candidates
   concurrently while writing and re-scoring the previous step.
3. **Background projection rebuilds,** so a long expansion never pays one inline.
