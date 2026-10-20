# Training Jev to steer expansion

The plan for bead ramp-o8i: a learned policy behind the `Policy` seam
([`expansion.md`](expansion.md)), built on Jev.

## What Jev is, and what training means here

Jev is TypeSafe's System One model (`jev-latest`, `POST
https://api.typesafe.ai/v1/systemone`). It takes a state (text or JSON) and a map of
typed questions and returns calibrated answers in one parallel pass:

- **noul:** a yes/no probability.
- **choice:** one of up to 255 options, with probabilities.
- **score:** a level on a 2–10 level rubric, with probabilities.

TypeSafe quotes 70–500 ms per call and $0.042 per million input tokens. Its weights
are fixed: the API has no fine-tuning.

So we train the policy **around** Jev, the way TypeSafe's own "autoresearch feature
discovery" cookbook does:
- Jev answers atomic questions about each candidate, and the probabilities become
  features.
- A gradient-boosted ranker combines those features with the numeric candidate features
  into a priority.
- An LLM proposes, revises and drops questions over several rounds, and a change is kept
  only if held-out ranking improves.

Jev's edge over `Guided` is text. `Guided` sees counts and costs. Jev can read what
`Guided` cannot:
- names, jurisdictions, company types and service providers;
- addresses;
- relationship labels ("nominee director of");
- the analyst's prompt.

It can then judge whether an entity is a specific party or a mass-registration service.

## What is missing today

1. **Batches, not steps.** A trace holds the candidate each step expanded. A ranker needs
   every candidate of the batch the policy chose from.
2. **Labels for what was not expanded.** `hits` and `route` only label executed steps.
   Each offered candidate needs an oracle label:
   - **Synthetic:** on a planted signal route.
   - **ICIJ connect:** on a full-graph cheapest path between the seeds, or one within ε
     of its cost.
   - **ICIJ alert:** the shared entity or the target co-officer.
3. **Text.** `World` keeps one attribute (`country`). The ICIJ CSVs carry names,
   addresses, jurisdictions, company types, statuses, service providers and relationship
   labels. None of it reaches a policy.
4. **A prompt.** `Task` states intent as numbers. Jev conditions on words, so tasks need
   `prompt` text, generated from the task template for benchmarks.
5. **ICIJ over `/expand`.** The server runs only the synthetic world. Running a Python
   policy online on ICIJ needs `/expand` to serve ICIJ tasks too.

## Phases

| phase | where | delivers | done when |
|---|---|---|---|
| P0 data | Rust | Per-batch logging (`RAMP_BATCHES=dir`: context, every candidate, decisions, oracle label per candidate). Entity text through `Source`/`wire`. `Task.prompt`. ICIJ loader and task sampler moved from the bench into `ramp_expand::icij`, and served by `/expand` (`world: "icij"`). | A dataset from ≥200 ICIJ tasks per kind plus the synthetic scenarios, with batch counts and positive rates reported. |
| P1 baseline ranker | `jev/` (Python, uv) | Loader into ranking groups (one group per batch); numeric features; a CatBoost model (see P1 results: a classifier, not `YetiRank`). Offline NDCG@k and recall@k against the oracle, with `Guided`'s own ordering scored the same way. Train/test split by task and seed entity. | A numeric-only ranker matches or beats `Guided` offline. Otherwise the data or labels are wrong: fix before adding Jev. |
| P2 Jev features | `jev/` | TypeSafe client (stdlib HTTP: TypeSafe ships no Python SDK; `TYPESAFE_API_KEY`, backoff on 429/529/5xx, 8 concurrent). Question set v0 by hand, 10 questions about the entity alone (see P2 results). One request per entity carrying every question. SQLite cache keyed by (model, question, entity text), so each entity is asked each question once. | Ablation of numeric vs numeric+Jev on held-out tasks. Determinism checked by re-asking 100 cached items. |
| P3 autoresearch | `jev/` | Claude proposes, revises and drops questions from feedback (worst batches, importances, metric history), over 5 rounds. A refit costs no API calls, so rejected changes are free. Bootstrap CI on the held-out change. | The final question set beats v0 on held-out NDCG with a CI excluding zero. |
| P4 serving | `jev/` | `jev serve`: an `/expand` client that long-polls and scores each batch (cached Jev answers + numeric features → priority). Breadth starts with `Guided`'s rule (≤1/64 of remaining budget per relation per visit). Newly discovered entities are prefetched while the batch waits; `parallel(k)` and `spread` hide latency. | It runs every held-out task on both worlds without timeouts. |
| P5 evaluation | both | The ICIJ and scenario tables with Jev added: solved, fetched p50/p90, cheapest-path rate, widest hub, wall time, Jev calls and cost per run. | Ship if it solves at least as many as `Guided`, fetches fewer at p50 and p90 on ICIJ connect, and does not regress on the synthetic scenarios, at a stated cost per run. Otherwise report why. |
| P6 later | | Learned breadth (take limits). On-policy data: run the learned policy, relabel with the oracle, retrain (DAgger). Route credit as a reward weight. | |

## P1 results

`cd jev && uv run python -m jev DIR` trains on the batches of the bench's
`RAMP_BATCHES` dir. The dataset is 200 ICIJ tasks per kind with every policy: 2.36M
candidates in 177k batches. The split is 85 test tasks (whole groups of tasks that
share a seed). Loading takes 37 s and training 30 s.

- **Features.** 46 numbers per candidate: domain, goal, peek and edges left per
  relation, graph neighbours per domain, hops and route cost from the near and far
  side, watched, alerts, probes, the relevant edges left, the spent budget, side
  balance, batch size, and the candidate's cost rank in its batch.
- **Training groups are cut to 64 rows** (every positive, then sampled negatives). A
  hub's expansion offers up to 37k candidates in one batch, and 1.05M of the 1.33M
  rows in mixed-label batches sit in batches over 256. Uncut, `YetiRank` ran for over
  half an hour in 16 GB and swapped.
- **Scored the way the budget pays.** Besides NDCG@5, recall@5 and top-1, each test
  batch is scored by the edges ranked ahead of its first positive.
  - A `YetiRank` ranker won top-1 against `Guided`'s own order on connect (0.72
    against 0.14). But it spent more edges before the first hit (188 against 110 on
    average). It learned which entities lie on the path, not what they cost.
  - `Guided`'s top-1 misses are mostly cheap dead ends: a leaf with one edge left,
    ranked over the connector.
- **So the model is a calibrated classifier,** and a candidate's priority is its chance
  of being on the path per edge: `log p − log(1 + relevant edges left)`. No weight is
  tuned.

Test tasks, each batch ordered by the model against the policy that ran:

| batches | groups | top-1 | edges ahead (mean) | p90 | model − policy, 95% CI |
|---|---|---|---|---|---|
| connect, `Guided` | 239 | 0.14 → 0.31 | 110 → 84 | 255 → 148 | [−70, +26] |
| alert, `Guided` | 85 | 0.87 → 0.77 | 7.8 → 8.7 | 28 → 21 | [−2.6, +4.7] |

**The gate holds: the model matches `Guided`.** Both intervals include zero, and the
connect point estimates favour the model. It is not yet a clear win, which is what the
text features (P2) are for.

On pushdown batches the comparison means little: pushdown prices a filtered fetch as a
few edges, and the model prices the unfiltered relation.

## P2 results

`uv run python -m jev.ask DIR [N]` asks question set v0 (`jev/src/jev/questions.py`)
about every entity in `DIR/entities.jsonl`, or a seeded sample of N, into
`DIR/jev.sqlite`. `python -m jev DIR` then fits numeric and numeric + Jev and reports
both.

- **Questions.** Ten, each about the entity alone, so they are asked once per entity,
  not per task:
  - nominee or agent;
  - service provider;
  - mass-registration address;
  - hub-likeness (a 4-level score);
  - natural person;
  - specific party;
  - secrecy jurisdiction;
  - shell;
  - still active;
  - notable or politically exposed.

  Relevance to the analyst's prompt would need a call per (task, candidate) pair, so it
  waits for P3.
- **Asking.** 333,988 entities at about 64 calls/s, 66 minutes in all. Each call is about
  600 input tokens and 170 output tokens: about 200M input tokens, roughly $8.4 at the
  quoted rate, plus output tokens (not priced in the docs).
- **Determinism.** Jev is not deterministic. Re-asking 100 cached entities moved an
  answer by at most 0.10–0.12, at p99 by 0.06, and 36% of answers came back identical.
  That is small next to the spread across entities, and the cache keeps experiments
  reproducible.

Test tasks against the same split as P1. "edges ahead" is the mean edges ranked before
the first positive:

| batches | top-1: policy / numeric / +Jev | edges ahead: policy / numeric / +Jev | +Jev − numeric, 95% CI |
|---|---|---|---|
| connect, `Guided` | 0.14 / 0.31 / 0.37 | 110 / 84 / 77 | [−34, +18] |
| alert, `Guided` | 0.87 / 0.77 / 0.86 | 7.8 / 8.7 / 5.9 | [−7.7, +1.4] |

- **Jev's answers help on every metric.** On connect, NDCG@5 goes 0.57 → 0.60 and p90
  edges ahead 148 → 119. On alert, the model with Jev beats `Guided`'s own order on
  edges ahead.
- **Not yet significant.** Both intervals include zero. With 239 and 85 test batches the
  interval is wide, so P3 needs either more test tasks or larger gains.
- **Read with care: these are single fits.** P3 found that refitting one feature set
  under another seed moves the score as much as these gains (see P3 results).
- **Importance.** Four Jev answers make the top 12: `secrecy` 6.4, `hub` 6.3,
  `notable` 4.8 and `person` 2.9. Route `cost` stays first.
- **Weak questions.** For company entities, `shell` and `secrecy` are almost constant
  (above 0.5 for all of them), and `mass_address` is above 0.5 for 85–100% of each
  domain. P3 starts by revising those.

## P3 results

P3 had a budget of $5–10, so it skipped the five-round loop the plan describes, which
would ask every question of every entity each round. Everything is in
`uv run python -m jev.research DIR <command>`. Question choices were made on a
validation split carved out of the training tasks, by a second salted `data.split`.
The test tasks were scored once, at the end.

- **Objective.** The mean of `log(1 + edges ahead)` over `Guided` batches of both
  kinds. The log keeps one hub batch from outweighing the rest.
- **Spend.** $2.08 in all. The cache's `spent` table is a ledger of every call:
  - $1.28 for the validation round;
  - $0.72 for the test pairs;
  - about $0.08 of answers lost when a run was stopped before it committed them.

  Output tokens are charged at the input rate, since TypeSafe does not publish an
  output price.
- **Spend guard.** `client.answers` estimates its cost first and refuses a budget the
  estimate exceeds. Its workers stop starting calls once billing passes the budget.
  Answers come from `jev-1.13.0`, and a different model behind `jev-latest` stops the
  run.

### Seed noise sets the bar

Refitting one feature set under another seed moves the validation loss by about
±0.2. A drop-one study over the ten v0 questions measured only that noise: every
difference fell inside it. Averaging four seeds cuts the noise to about ±0.05, so every
later comparison uses four-seed averages, and a gain under 0.1 does not count.

### Prompt relevance

All ten v0 questions are about the entity alone. Three questions about the candidate
in the light of one task (`research.RELEVANCE`: `tie` to the seeds, part of the
`link` asked for, `fits` what the task names) were asked once per (task, candidate)
pair. The state was the prompt, the seeds' text and the candidate's text.

To keep the cost down, training covers 8k batches sampled from 250 training tasks (30k
pairs), with the 64-row cut fixed in advance so that every seed trains on rows that
were asked. Every variant trains on the same rows. Losses, lower is better:

| | validation | test | test connect | test alert |
|---|---|---|---|---|
| `Guided`'s own order | — | 1.87 | 2.35 | 0.50 |
| numeric | 1.92 | 1.65 | 2.01 | 0.66 |
| v0 | 1.73 | 1.57 | 1.92 | 0.57 |
| numeric + relevance | 1.84 | 1.64 | 2.02 | 0.57 |
| v0 + relevance | **1.63** | 1.58 | 1.97 | **0.48** |

- **Relevance did not hold up on test.** On validation, v0 + relevance beat v0 by
  0.10, with an interval of [−0.17, −0.02]. On test it ties v0, with an interval of
  [−0.07, +0.08]. It helps on alert and costs a little on connect. The P3 gate (beat
  v0 with an interval excluding zero) is **not met**.
- **Jev against numeric: suggestive.** v0 + relevance minus numeric is [−0.19, +0.04]
  on test, and v0 beats numeric on both splits.
- **The learned models beat `Guided`.** v0 + relevance minus `Guided` is
  [−0.42, −0.15], all of it on connect. On alert, `Guided` ties the best model.
- **What next.** The entity questions (v0) carry the Jev gain we can measure. More
  question rounds would chase gains under the noise this evaluation can detect. Two
  things would help before any more spend:
  - more test tasks, from the free bench;
  - a less noisy fit, such as more seeds or a larger cap.

## A larger held-out set

`RAMP_SEED=8` on the bench samples another 300 tasks per kind (`~/.cache/ramp-jev/seed8`).
596 of them share no seed entity with the original 400 tasks. `uv run python -m
jev.holdout TRAIN_DIR TEST_DIR [BUDGET]` trains on every original task and scores
those 596: 5,112 `Guided` batches, against 324 in the P1–P3 test split.

- **Per task, not per batch.** A long task has hundreds of correlated batches. Metrics
  are averaged per task first, and intervals resample tasks. Per batch, the connect
  comparison flips sign.
- **v0 needs a spend.** 39.5k entities in the scored rows are not in the cache,
  estimated at $1.22. A partial v0 column would bias the comparison, so v0 is skipped
  until they are asked.

Numeric (four-seed average) against `Guided`'s own order, per-task means:

| | tasks | log1p(edges ahead): `Guided` / numeric | numeric − `Guided`, 95% CI | edges ahead: `Guided` / numeric | top-1 |
|---|---|---|---|---|---|
| connect | 296 | 2.35 / 1.82 | [−0.60, −0.45] | 143 / 166 | 0.16 / 0.40 |
| alert | 297 | 0.59 / 0.61 | [−0.06, +0.09] | 15.8 / 13.7 | 0.85 / 0.81 |

- **Connect: the model wins the typical batch and loses the tail.** On the log scale it
  is clearly better. But in mean edges, which is what the budget pays, it is worse (166
  against 143), because its misses are expensive.
- **Alert: a tie.**
- **So the P1 cost correction is not finished.** `log p − log1p(left)` still lets
  high-confidence, expensive candidates rank first.

### The cost fix

The 64-row cut keeps every positive and samples the negatives, which inflates `p` most
in big hub batches, where candidates are expensive. `rank.fit(reweight=True)` weighs
each kept negative by its batch's negatives over those kept. The fix was tuned on a
third set: `RAMP_SEED=9`, 592 tasks that share no seed with seed 7 or seed 8
(`holdout --sweep`, numeric only, free). Per-task means of edges ahead:

| validation (seed 9) | `Guided` | P1 setting | `alpha` 1.5 | `alpha` 2 | reweight | reweight, `alpha` 2 |
|---|---|---|---|---|---|---|
| connect | 166 | 213 | 222 | 229 | **138** | 151 |
| alert | 35.1 | 45.6 | 42.7 | 42.2 | 43.8 | 36.8 |

A stronger cost term (`alpha` > 1) makes connect worse. Reweighting fixes it: on
validation, connect beats `Guided` with an interval of [−45, −13]. Reweighting with
`alpha` 1 is kept, so no knob is added.

On the seed-8 test tasks, the change helps less:

| test (seed 8) | `Guided` | P1 setting | reweight | reweight − `Guided`, 95% CI |
|---|---|---|---|---|
| connect | 143 | 166 | 159 | [−53, +140] |
| alert | 15.8 | 13.7 | 13.1 | [−7.0, +0.8] |

- **Mean edges is dominated by a few long tasks.** The connect interval spans almost
  200 edges, so on mean edges the model and `Guided` are tied on test, and the
  reweighting gain over P1 is within noise.
- **The log scale.** Reweighting is still best there: 1.81 against 2.35 for `Guided`.

### v0 on the larger test set

38,124 entities in the scored seed-8 rows were not cached. Asking them cost $1.31 on
the ledger, in two capped runs: $1.25 and then $0.06. Per-task means, with each model
a four-seed average:

| test (seed 8) | `Guided` | numeric | v0 | v0 P1 setting | v0 − numeric, 95% CI |
|---|---|---|---|---|---|
| connect, edges ahead | 143 | 159 | 151 | 163 | [−21, +6] |
| connect, log1p(ahead) | 2.35 | 1.81 | **1.70** | 1.74 | |
| alert, edges ahead | 15.8 | 13.1 | **12.7** | 14.7 | [−3.5, +2.7] |
| alert, log1p(ahead) | 0.59 | 0.58 | 0.59 | 0.61 | |

- **v0 with reweighting is the best model on every measure but one.** Jev's entity
  answers and the cost fix each help, and they add up. The exception is connect mean
  edges, where `Guided` still leads, 143 against 151, inside a wide tail interval.
- **The Jev gain is not significant, even with 296 tasks per kind.** v0 − numeric on
  mean edges includes zero for both kinds, and the point estimates favour v0.
- **For P4, serve v0 with reweighting.** Its answers are cached for 372k entities, so
  serving them is free.

## P4 results

`jev.serve` is the client. `train DIR` fits the four-seed v0 model, reweighted, on every
seed-7 task and saves it beside the cache (5 minutes, no API calls). `run URL TASKS_DIR
--scenarios` starts each task as a `remote` run on `ramp-server -x icij_dir` and
long-polls it. Each batch gets the training features (`data.batch`, shared with the
loader) and the cached v0 answers about its candidates, and is answered with priority
`log p − log(1 + relevant edges left)` and `Guided`'s breadth rule. An entity the cache
lacks scores with NaN answers. `--budget` (off by default) pays to ask it first, from
one cap shared by the whole run. Scoring a batch takes about 2 ms; a round trip with the
server's step is about 28 ms.

Every seed-8 task and the five synthetic scenarios, `parallel` 1, $0 spent:

| | v0 served: solved | fetched p50 / p90 | `Guided` (bench, same tasks) |
|---|---|---|---|
| ICIJ connect | 300/300 | 188 / 2,306 | 300/300, 190 / 2,910 |
| ICIJ alert | 298/300 | 101 / 450 | 300/300, 100 / 450 |
| path-5hop | solved | 409 | 202 |
| path-7hop | not in 100k | | 2,222 |
| pivot-busy | not in 100k | | 31,407 |
| alert (synthetic) | solved | 243 | 150 |
| explore | runs to budget, as `Guided` does | | |

- **The gate is met: no batch went unanswered past the 60 s timeout in 605 runs.**
- **On ICIJ, v0 matches `Guided`'s median and cuts its connect p90 by a fifth.** It
  misses two alerts that `Guided` solves, each after spending the full 200k budget.
- **On the synthetic world it regresses.** The model saw only ICIJ batches, and two of
  five scenarios go unsolved. P5 measures this properly; the fix to try first is
  training on the synthetic batches too.
- **Breadth, prefetch and `parallel` are as planned, but minimal.** Uncached entities are
  asked before their batch is answered, not in the background, since the cache covers
  ICIJ; `--parallel k` and `--spread` pass through to the server untested beyond 1.

## P5 results

**Training on both worlds.** The synthetic bench takes `RAMP_SEED` for its world and,
with `RAMP_BATCHES`, writes `tasks.jsonl` and only the policy tables. Worlds 1–8 at the
server's defaults (200k entities, 100k budget) give the synthetic batches, with NaN Jev
answers since the world has no text. `serve train DIR --syn DIR…` fits v0 on them and on
seed 7's ICIJ batches: 651k batches, 11 minutes.

**The test.** The 589 seed-8 ICIJ tasks that share no seed with seed 7 or seed 9, and the four scenarios with a signal on
unseen worlds 101–110. Both policies ran through the same client and server
(`serve run --policy guided` for the baseline), `parallel` 1, with no late batch and $0
spent. An unsolved run counts as infinite.

| | `Guided`: solved | fetched p50 / p90 | v0: solved | fetched p50 / p90 | v0 / `Guided`, paired geo-mean, 95% CI |
|---|---|---|---|---|---|
| ICIJ connect | 296/296 | 190 / 3,034 | 296/296 | **167 / 1,984** | **0.84 [0.76, 0.93]** |
| ICIJ alert | 293/293 | 102 / 450 | 293/293 | 101 / 450 | 0.99 [0.96, 1.03] |
| syn alert | 10/10 | 147 / 190 | 10/10 | 181 / 310 | 1.36 [0.82, 2.25] |
| syn path-5hop | 10/10 | 173 / 268 | 10/10 | 74 / 353 | 0.66 [0.40, 1.11] |
| syn path-7hop | 10/10 | 3,792 / 4,565 | 10/10 | **638 / 1,539** | **0.14 [0.08, 0.23]** |
| syn pivot-busy | 8/10 | 64,800 / ∞ | **10/10** | **29,743 / 35,206** | **0.41 [0.27, 0.59]** |

On ICIJ connect, `Guided` finds the cheapest path in 270 of 296 runs (mean cost 1.010×
the cheapest) and v0 in 205 (1.036×), at the same widest hub (812 and 815). Wall p50 is
1.3 s and 0.7 s.

- **The gate is met.** v0 solves every task `Guided` solves, plus two pivot-busy worlds.
  It fetches fewer edges at p50 and p90 on ICIJ connect, 16% fewer per task with an
  interval that excludes zero. It costs $0 per run while the cache covers the tasks.
- **The synthetic batches fixed the regression.** Trained on ICIJ alone, v0 left
  path-7hop and pivot-busy unsolved (P4). With worlds 1–8 it beats `Guided` on both by
  2.5–7× on unseen worlds. The ICIJ numbers did not suffer: connect went from
  188 / 2,306 in P4 to 167 / 1,984 here, on nearly the same tasks.
- **Synthetic alert is a tie, not a regression.** On the 10 worlds above it looked worse:
  1.36× the edges, with an interval spanning 1. On 100 more unseen worlds (111–210,
  `serve run --only alert`), v0 fetches 150 / 244 at p50 / p90 against `Guided`'s
  165 / 256: paired 0.94× [0.80, 1.14], fewer edges in 57 worlds, more in 43. With
  pushdown both take 2 / 3 edges in every world (see Pushdown).
- **v0 trades path quality for edges.** It stops at the first join, like `Guided`, but
  finds the cheapest path a quarter less often. Where analysts want the most specific
  link, that is a cost.
- **The cache lacks 23% of v0's connect candidates,** since v0 walks where `Guided`'s
  traces did not. They score with NaN answers. `serve run --uncached FILE` lists them
  with no spend. The rerun reproduced the table above exactly. It found 35,902
  entities with text, median 228 characters: about $1.07 at the input rate, or about
  $1.25 at the rate the ledger has billed for v0 so far. One pass is not the end of it:
  with answers, v0 walks elsewhere and meets new uncached entities.
- **Jev's share of the win is small, so that spend waits.** The same model without Jev
  answers (`train --numeric`), served on the same 589 tasks:

  | ICIJ, paired geo-mean of edges fetched, 95% CI | connect | alert |
  |---|---|---|
  | numeric / `Guided` | 0.88 [0.80, 0.95] | 0.99 [0.96, 1.02] |
  | v0 / `Guided` | 0.84 [0.76, 0.93] | 0.99 [0.96, 1.03] |
  | v0 / numeric | 0.96 [0.87, 1.06] | 1.00 [0.97, 1.04] |

  Numeric's connect p50 / p90 is 191 / 2,151 (v0: 167 / 1,984). It finds the cheapest
  path in 199 of 296 runs (v0: 205), with a widest hub of 738 (v0: 815). Most of the
  gain over `Guided` is the learned cost model; Jev adds a few percent that 296 tasks
  cannot resolve. Filling the missing 23% for about $1.25 could not show a gain this
  test would detect, so it was not bought. A larger test set or a more decisive
  question set comes first.

### Pushdown

`serve run --pushdown` gives v0 `Guided`'s pushdown. A candidate on the task's shape,
with more of it beyond, fetches only the relation to the next step, filtered by
semi-joins to the target. That fetch is priced as one edge left, so its priority is
`log p − ln 2`. Only tasks that name a target change: ICIJ alert, and synthetic alert
and pivot-busy. Same tasks as above, against `Guided` with pushdown, $0 spent:

| | `Guided`+pushdown: solved, fetched p50 / p90 | v0+pushdown | paired geo-mean, 95% CI |
|---|---|---|---|
| ICIJ alert | 293/293, 4 / 4 | 293/293, **2 / 3** | **0.71 [0.68, 0.74]**, fewer in 159, more in 1 |
| syn alert | 10/10, 2 / 2 | 10/10, 2 / 2 | 1.00 |
| syn pivot-busy | 10/10, 60 / 128 | 10/10, **7 / 18** | **0.13 [0.10, 0.19]** |

Connect and the path scenarios are unchanged: they name no target. Pushdown also
removes the one synthetic case where v0 was behind: synthetic alert drops from 181 to
2 edges, level with `Guided`. The learned order gets more from pushdown than `Guided`'s
does. Both take the same filtered fetches, but v0 takes the ones that pay first.

## P6: on-policy data (DAgger)

`Guided`'s traces show v0 only the states `Guided` reaches. Served, v0 goes elsewhere:
23% of its connect candidates are entities the traces never offered. DAgger trains on
the states the learned policy reaches, labelled by the oracle:
- **`/expand` serves labelled batches.** A run started with `"batches": true` keeps every
  policy call and labels each candidate with the benches' oracle: near-cheapest path
  for ICIJ connect, shared with the target for alert, a signal endpoint for a scenario
  (`GET /expand/<id>/batches`).
- **`serve run --collect DIR`** saves them per run, with `tasks.jsonl` and the entity
  text, in the loader's format. **`serve train … --syn DIR…`** takes those dirs too,
  with cached answers where there is text.

**Round 1.** v0 with pushdown ran on the training tasks only: the 400 seed-7 ICIJ tasks
and the four scenarios on worlds 1–8. That gave 432 runs and 39 MB. `dagger` is v0
retrained on the `Guided` batches plus these. On the P5 test sets with pushdown, paired
against v0, $0 spent:

| | v0: fetched p50 / p90 | `dagger` | `dagger` / v0, geo-mean, 95% CI |
|---|---|---|---|
| ICIJ connect | 167 / 1,984 | 139 / 2,123 | 0.93 [0.85, 1.01] |
| ICIJ alert | 2 / 3 | 2 / 3 | 0.98 [0.97, 1.00] |
| syn path-5hop | 74 / 353 | 66 / 517 | 0.72 [0.51, 1.01] |
| syn path-7hop | 638 / 1,539 | 575 / 1,368 | 0.73 [0.45, 1.24] |
| syn pivot-busy | 7 / 18 | 7 / 19 | 0.87 [0.70, 1.01] |
| syn alert | 2 / 2 | 2 / 2 | 1.00 |

Every kind leans towards `dagger`, and none is significant. Both solve every task. On
connect, `dagger` lowers the median (139 against 167) and raises p90 (2,123 against
1,984); mean edges are level (0.99). Round 2 collects with `dagger` and trains on both
rounds.

## Costs and risks

- **Cost.** Every one of the 334k entities is a candidate somewhere. At about 600 input
  tokens per call (the questions dominate the short entity text), a full pass of the
  10 v0 questions is about $8.4 in input tokens, plus output (P2). A revised question
  re-asks only itself, but for every entity. The cache makes reruns free.
- **Latency.** One call per uncached entity, 70–500 ms. Per-entity caching and
  prefetching leave most batches with no call. Rate limits are undocumented, so the
  client backs off.
- **Determinism.** Jev is not deterministic: a re-ask moves answers by up to about 0.1
  (P2). The cache makes experiments reproducible.
- **Model drift.** The cache is keyed by `jev-latest`, and its answers are from
  `jev-1.13.0`. The client stops when the name resolves to another version, so cached
  and new answers are never mixed.
- **Oracle bias.** One cheapest path is one answer, so positives use every path within
  ε of the best cost. The metric stays "least hub-heavy", as in the ICIJ tables.
- **Leakage.** Splits are by task and by seed entity, never by batch.
- **Data.** ICIJ is public (ODbL), so sending it is fine. A private source later gets
  the redaction ops already applies before calling TypeSafe.
- **Key.** `TYPESAFE_API_KEY` comes from the environment and is never committed.
