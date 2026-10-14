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
| P1 baseline ranker | `jev/` (Python, uv) | Loader into ranking groups (one group per batch); numeric features; CatBoost `YetiRank`. Offline NDCG@k and recall@k against the oracle, with `Guided`'s own ordering scored the same way. Train/test split by task and seed entity. | A numeric-only ranker matches or beats `Guided` offline. Otherwise the data or labels are wrong: fix before adding Jev. |
| P2 Jev features | `jev/` | TypeSafe client (Python SDK, `TYPESAFE_API_KEY`, backoff on 429/529, 8 concurrent). Question set v0 by hand, ~10 questions: hub-likeness, nominee or agent, mass-registration address, secrecy jurisdiction, relevance to the prompt. One request per entity carrying every question. Disk cache keyed by (model, entity text, question): entity text does not change, so each entity is asked once. | Ablation of numeric vs numeric+Jev on held-out tasks. Determinism checked by re-asking 100 cached items. |
| P3 autoresearch | `jev/` | Claude proposes, revises and drops questions from feedback (worst batches, importances, metric history), over 5 rounds. A refit costs no API calls, so rejected changes are free. Bootstrap CI on the held-out change. | The final question set beats v0 on held-out NDCG with a CI excluding zero. |
| P4 serving | `jev/` | `jev serve`: an `/expand` client that long-polls and scores each batch (cached Jev answers + numeric features → priority). Breadth starts with `Guided`'s rule (≤1/64 of remaining budget per relation per visit). Newly discovered entities are prefetched while the batch waits; `parallel(k)` and `spread` hide latency. | It runs every held-out task on both worlds without timeouts. |
| P5 evaluation | both | The ICIJ and scenario tables with Jev added: solved, fetched p50/p90, cheapest-path rate, widest hub, wall time, Jev calls and cost per run. | Ship if it solves at least as many as `Guided`, fetches fewer at p50 and p90 on ICIJ connect, and does not regress on the synthetic scenarios, at a stated cost per run. Otherwise report why. |
| P6 later | | Learned breadth (take limits). On-policy data: run the learned policy, relabel with the oracle, retrain (DAgger). Route credit as a reward weight. | |

## Costs and risks

- **Cost.** About 100k distinct frontier entities × ~300 input tokens ≈ 30M tokens per
  full dataset build, ≈ $1.30. The cache makes reruns free.
- **Latency.** One call per uncached entity, 70–500 ms. Per-entity caching and
  prefetching leave most batches with no call. Rate limits are undocumented, so the
  client backs off.
- **Determinism.** Assumed and checked in P2. The cache makes experiments reproducible
  either way.
- **Oracle bias.** One cheapest path is one answer, so positives use every path within
  ε of the best cost. The metric stays "least hub-heavy", as in the ICIJ tables.
- **Leakage.** Splits are by task and by seed entity, never by batch.
- **Data.** ICIJ is public (ODbL), so sending it is fine. A private source later gets
  the redaction ops already applies before calling TypeSafe.
- **Key.** `TYPESAFE_API_KEY` comes from the environment and is never committed.
