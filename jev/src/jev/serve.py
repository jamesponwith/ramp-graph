"""`jev serve` (P4, P5): steer `/expand` runs with the v0 model.

`uv run python -m jev.serve train DIR [--syn DIR...] [--model NAME] [--numeric]`: fit the four-seed
v0 model, reweighted, on every task of DIR (cached answers only, no API calls) and of
each synthetic batches dir (`benches/expand.rs` with `RAMP_BATCHES`, no text, so NaN
answers), and save it beside the cache.

`uv run python -m jev.serve run URL [TASKS_DIR] [options]`: run each ICIJ task of
TASKS_DIR's `tasks.jsonl` that shares no seed with an `--exclude` dir over the server at
URL (started with `-x icij_dir`), and with `--seeds` every synthetic scenario with a
signal on each of those worlds. `--policy guided` (or any server policy) runs the
baseline through the same loop. `--pushdown` gives v0 `Guided`'s pushdown (see `pushdown`). Each batch is scored from its numeric
features and the cached Jev answers about its candidates: priority is
`log p - log(1 + relevant edges left)`, in nats like `Guided`'s, so `--spread 1`
means the same. Breadth is `Guided`'s rule: every relevant relation, up to 1/64 of the
remaining budget (at least 16) per visit. An entity the cache lacks gets NaN features,
unless `--budget` dollars (shared by the whole run, off by default) pays to ask it
before its batch is answered. One JSON line per run goes to `--out`.

`uv run python -m jev.serve table A.jsonl B.jsonl...`: the P5 table of those runs.
"""

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from . import client, data, questions, rank
from .holdout import FITS, answers
from .research import CACHE, loaded

MODELS = CACHE / "serve"
# The scenarios with a planted signal: `explore` has none, so it is never solved.
SCENARIOS = ("path-5hop", "path-7hop", "pivot-busy", "alert")
# What a candidate not worth expanding gets: JSON has no -inf.
SKIP = -1e300


def with_answers(dir: Path, d: data.Data) -> data.Data:
    """`d` with its cached v0 answers as columns: NaN where the cache lacks them, and
    everywhere if `dir` has no entity text (the synthetic world)."""
    if (dir / "entities.jsonl").exists():
        cols = answers(dir, d, np.ones(len(d.y), bool), 0.0)
    else:
        cols = np.full((len(d.y), len(questions.NAMES)), np.nan, np.float32)
    return replace(d, x=np.hstack([d.x, cols]))


def credit(d: data.Data, slack: Path, strength: float) -> np.ndarray:
    """Route credit, one per row: `exp(-strength * slack)` for a positive whose task has
    a row in `slack` (`benches/icij.rs` with `RAMP_SLACK`), so a candidate on the
    cheapest path counts more than one on a path 10% dearer; 1 for every other row."""
    by_task = {
        (r := json.loads(line))["run"]: {int(i): s for i, s in r["slack"].items()}
        for line in slack.read_text().splitlines()
    }
    task = np.array(d.task)[d.group]
    out = np.ones(len(d.y))
    for r in np.flatnonzero(d.y == 1):
        s = by_task.get(task[r], {}).get(int(d.id[r]))
        if s is not None:
            out[r] = math.exp(-strength * s)
    return out


def train(
    dir: Path,
    more: list[Path],
    name: str,
    numeric: bool = False,
    slack: Path | None = None,
    strength: float = 0.0,
) -> None:
    """`dir` is the original ICIJ batches (pickled); `more` are synthetic batch dirs and
    on-policy dirs (`run --collect`), of either world. With `slack`, positives get
    route credit (see `credit`)."""
    parts = [with_answers(dir, loaded(dir))]
    for m in more:
        parts.append(with_answers(m, data.load(m, ("connect", *SCENARIOS))))
    d = data.concat(parts)
    print(f"  {len(d.run):,} batches, {len(d.y):,} candidates", flush=True)
    every = np.ones(len(d.y), bool)
    names = data.NAMES + ([] if numeric else questions.NAMES)
    d = replace(d, x=d.x[:, : len(names)])
    weight = credit(d, slack, strength) if slack else None
    if weight is not None:
        print(f"  route credit: {np.mean(weight[d.y == 1] < 1):.0%} of positives below 1")
    MODELS.mkdir(exist_ok=True)
    for seed in range(FITS):
        m = rank.fit(d, every, names, seed=seed, reweight=True, credit=weight)
        m.save_model(str(MODELS / f"{name}.{seed}.cbm"))
        print(f"  saved {name}.{seed}", flush=True)


def call(method: str, url: str, body: dict | None = None) -> dict:
    raw = fetch(method, url, body)
    return json.loads(raw) if raw else {}


def fetch(method: str, url: str, body: dict | None = None) -> bytes:
    req = urllib.request.Request(
        url,
        None if body is None else json.dumps(body).encode(),
        {"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def joining(task: dict, a: int, b: int) -> int | None:
    """The relation joining domains `a` and `b`, either way round."""
    for i, r in enumerate(task["relations"]):
        if (r["from"], r["to"]) in ((a, b), (b, a)):
            return i
    return None


def pushdown(task: dict, c: dict) -> tuple[int, dict] | None:
    """`Guided`'s filtered fetch from `c` towards the task's target: for a candidate on
    the task's shape with more of it beyond, the relation to its next step and the
    target wrapped in one semi-join per step after that. `None` if the task names no
    target or the source cannot filter, the candidate is off-shape, or that relation
    was already fetched filtered."""
    if not task["filters"] or task["target"] is None:
        return None
    at = min(h for h in c["hops"] if h is not None)
    for shape in task["shape"]:
        if at + 1 >= len(shape) or shape[at] != c["domain"]:
            continue
        f, ok = task["target"], True
        for i in reversed(range(at + 1, len(shape) - 1)):
            via = joining(task, shape[i], shape[i + 1])
            if via is None:
                ok = False
                break
            f = {"via": via, "then": f}
        rel = joining(task, c["domain"], shape[at + 1])
        if ok and rel is not None and not c["pushed"][rel]:
            return rel, f
    return None


def decide(models: list, task: dict, ask: dict, got: dict, push: bool = False) -> list[dict]:
    """One decision per candidate of `ask`. With `push`, a candidate `pushdown` applies
    to fetches only that filtered relation, priced as one edge left, as `Guided`
    prices it: what comes back is what can complete a hit."""
    cands, ctx = ask["candidates"], ask["context"]
    if not cands:
        return []
    ids = np.array([c["id"] for c in cands])
    x = np.hstack(
        [
            np.asarray(data.batch(cands, ctx, task), np.float32),
            questions.features(ids, got),
        ]
    )
    # A numeric-only model reads only the leading columns.
    p = sum(m.predict_proba(x[:, : len(m.feature_names_)])[:, 1] for m in models) / len(models)
    logp = np.log(p)
    score = logp - x[:, rank.LEFT]
    interest = task["relation_interest"]
    breadth = max((ctx["budget"] - ctx["fetched"]) // 64, 16)
    out = []
    for c, s, lp in zip(cands, score, logp):
        rem = [max(a - b, 0) for a, b in zip(c["peek"], c["fetched"])]
        take = [
            {"rel": r, "limit": min(n, breadth)}
            for r, n in enumerate(rem)
            if n > 0 and interest[r] > 0
        ]
        if not take or task["domain_interest"][c["domain"]] <= 0:
            out.append({"priority": SKIP})
        elif push and (pd := pushdown(task, c)):
            rel, f = pd
            filtered = {"rel": rel, "limit": breadth, "filter": f}
            out.append({"priority": float(lp - math.log(2)), "take": [filtered]})
        else:
            out.append({"priority": float(s), "take": take})
    return out


def run(
    url: str,
    req: dict,
    models: list,
    budget: float,
    spent0: float,
    push: bool = False,
    collect: Path | None = None,
) -> dict:
    """One run to its end: the report, the batches decided, and whether one was late.
    With `collect`, its oracle-labelled batches go to `collect/<name>.v0.jsonl`, and
    the row carries its task and the entity text it saw."""
    cache = client.Cache(CACHE / "jev.sqlite")
    started = time.time()
    r = call("POST", f"{url}/expand", {**req, "batches": collect is not None})
    run_id, task = r["id"], r["task"]
    batches, late, seen, known, uncached, saw = 0, False, 0, 0, {}, {}
    try:
        while True:
            a = call("GET", f"{url}/expand/{run_id}?wait=25")
            if a["state"] == "running":
                continue
            if a["state"] == "done":
                break
            texts = {c["id"]: c["text"] for c in a["candidates"] if c.get("text")}
            if collect:
                saw.update((c["id"], c.get("text", "")) for c in a["candidates"])
            left = budget - (cache.spent() - spent0)
            try:
                got = client.answers(texts, questions.QUESTIONS, cache, max(left, 0.0))
            except client.OverBudget:
                got = client.answers(texts, questions.QUESTIONS, cache)
            for c in a["candidates"]:
                got.setdefault(c["id"], dict.fromkeys(questions.QUESTIONS))
            uncached.update((i, t) for i, t in texts.items() if None in got[i].values())
            seen += len(a["candidates"])
            known += sum(None not in v.values() for v in got.values())
            body = {"seq": a["seq"], "decisions": decide(models, task, a, got, push)}
            try:
                call("POST", f"{url}/expand/{run_id}", body)
            except urllib.error.HTTPError as e:
                # 409: the server stopped waiting for this batch.
                if e.code != 409:
                    raise
                late = True
            batches += 1
        if collect:
            raw = fetch("GET", f"{url}/expand/{run_id}/batches")
            (collect / f"{req['name']}.v0.jsonl").write_bytes(raw)
    finally:
        call("DELETE", f"{url}/expand/{run_id}")
    rep = a["report"]
    path = rep.get("path") or {}
    return {
        **req,
        "solved": rep["done_at"] is not None,
        "cost/best": path["cost"] / path["best"] if path.get("best") else None,
        "widest": path.get("widest"),
        "answered": known / seen if seen else None,
        "uncached": uncached,
        "collected": {"run": req["name"], "task": task, "texts": saw} if collect else None,
        "fetched": rep["fetched"],
        "rounds": rep["rounds"],
        "batches": batches,
        "late": late,
        "wall": time.time() - started,
        "report": rep,
    }


def kind(r: dict) -> str:
    return r.get("goal") or f"syn {r['scenario']}"


def summary(rows: list[dict], label: str = "") -> None:
    """Per kind: solved, fetched p50/p90 (an unsolved run counts as infinite, as in the
    benches), cheapest-path rate, cost over the cheapest, widest hub p50, wall p50,
    batches answered late, and the share of candidates with every Jev answer."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(kind(r), []).append(r)
    for k, rs in sorted(by.items()):
        f = [r["fetched"] if r["solved"] else math.inf for r in rs]
        p50, p90 = (np.percentile(f, q, method="higher") for q in (50, 90))
        ratio = [r["cost/best"] for r in rs if r["solved"] and r.get("cost/best")]
        hub = [r["widest"] for r in rs if r["solved"] and r.get("widest") is not None]
        line = (
            f"  {label:8} {k:15} solved {sum(r['solved'] for r in rs):3}/{len(rs):<3}"
            f" fetched p50 {p50:>8,.0f} p90 {p90:>8,.0f}"
        )
        if ratio:
            cheapest = sum(x < 1 + 1e-9 for x in ratio)
            line += f"  cheapest {cheapest}/{len(ratio)} cost/best {np.mean(ratio):.3f}"
            line += f" widest {np.median(hub):,.0f}"
        line += f"  wall p50 {np.median([r['wall'] for r in rs]):.1f}s"
        line += f"  late {sum(r['late'] for r in rs)}"
        if rs[0].get("answered") is not None:
            line += f"  answered {np.mean([r['answered'] for r in rs]):.0%}"
        print(line)


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["train", "run", "table"])
    a.add_argument("target", help="train: batches DIR; run: server URL; table: a run file")
    a.add_argument("tasks", type=Path, nargs="*", help="run: ICIJ tasks DIR; table: more")
    a.add_argument("--syn", type=Path, nargs="*", default=[], help="train: more batch dirs")
    a.add_argument("--collect", type=Path, help="run: save on-policy batches here")
    a.add_argument("--slack", type=Path, help="train: slack.jsonl, for route credit")
    a.add_argument("--credit", type=float, default=0.0, help="train: credit strength")
    a.add_argument("--model", default="v0")
    a.add_argument("--numeric", action="store_true", help="train: without Jev answers")
    a.add_argument("--policy", default="remote")
    a.add_argument("--pushdown", action="store_true", help="run: push targets down")
    a.add_argument("--exclude", type=Path, nargs="*", default=[])
    a.add_argument("--seeds", type=int, nargs="*", default=[], help="synthetic worlds")
    a.add_argument("--only", nargs="*", default=SCENARIOS, help="run: these scenarios")
    a.add_argument("--limit", type=int)
    a.add_argument("--budget", type=float, default=0.0)
    a.add_argument("--parallel", type=int, default=1)
    a.add_argument("--spread", type=float, default=1.0)
    a.add_argument("--runs", type=int, default=4, help="concurrent runs")
    a.add_argument("--out", type=Path, default=CACHE / "serve.jsonl")
    a.add_argument("--uncached", type=Path, help="run: where to list what the cache lacks")
    args = a.parse_args()
    if args.cmd == "train":
        train(Path(args.target), args.syn, args.model, args.numeric, args.slack, args.credit)
        return
    if args.cmd == "table":
        for f in [Path(args.target), *args.tasks]:
            summary([json.loads(line) for line in f.read_text().splitlines()], f.stem)
        return
    models = []
    for seed in range(FITS if args.policy == "remote" else 0):
        m = CatBoostClassifier()
        m.load_model(str(MODELS / f"{args.model}.{seed}.cbm"))
        models.append(m)
    common = {"policy": args.policy, "parallel": args.parallel, "spread": args.spread}
    reqs = []
    seen = {s for d in args.exclude for t in data.read_tasks(d).values() for s in t["seeds"]}
    for dir in args.tasks:
        tasks = data.read_tasks(dir).items()
        fresh = [(n, t) for n, t in tasks if not seen & set(t["seeds"])]
        for n, t in fresh[: args.limit]:
            reqs.append(
                {
                    **common,
                    "name": n,
                    "world": "icij",
                    "goal": t["goal"],
                    "seeds": t["seeds"],
                    "budget": t["budget"],
                }
            )
    reqs += [
        {**common, "name": f"{s}.s{w}", "scenario": s, "seed": w}
        for w in args.seeds
        for s in args.only
    ]
    spent0 = client.Cache(CACHE / "jev.sqlite").spent()
    rows, lacking, collected = [], {}, []
    if args.collect:
        args.collect.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.runs) as pool, args.out.open("w") as out:
        for n, row in enumerate(
            pool.map(
                lambda r: run(
                    args.target, r, models, args.budget, spent0, args.pushdown, args.collect
                ),
                reqs,
            ),
            1,
        ):
            lacking.update(row.pop("uncached"))
            if (c := row.pop("collected")) is not None:
                collected.append(c)
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            if n % 50 == 0 or n == len(reqs):
                print(f"{n}/{len(reqs)} runs", flush=True)
                summary(rows)
    if args.collect:
        tasks = "".join(json.dumps({"run": c["run"], "task": c["task"]}) + "\n" for c in collected)
        (args.collect / "tasks.jsonl").write_text(tasks)
        texts = {i: t for c in collected for i, t in c["texts"].items()}
        if any(texts.values()):
            ents = "".join(json.dumps({"id": i, "text": t}) + "\n" for i, t in texts.items())
            (args.collect / "entities.jsonl").write_text(ents)
        print(f"collected {len(collected)} runs, {len(texts):,} entities")
    if args.uncached:
        todo = {i: list(questions.QUESTIONS) for i in lacking}
        args.uncached.write_text(
            "".join(json.dumps({"id": i, "text": t}) + "\n" for i, t in lacking.items())
        )
        print(
            f"{len(lacking):,} entities with text lack answers: about "
            f"${client.estimate(todo, lacking):.2f} to ask (input rate)"
        )
    print(f"Jev spend this run: ${client.Cache(CACHE / 'jev.sqlite').spent() - spent0:.2f}")


if __name__ == "__main__":
    main()
