"""`jev serve` (P4, P5): steer `/expand` runs with the v0 model.

`uv run python -m jev.serve train DIR [--syn DIR...] [--model NAME] [--numeric]`: fit the four-seed
v0 model, reweighted, on every task of DIR (cached answers only, no API calls) and of
each synthetic batches dir (`benches/expand.rs` with `RAMP_BATCHES`, no text, so NaN
answers), and save it beside the cache.

`uv run python -m jev.serve run URL [TASKS_DIR] [options]`: run each ICIJ task of
TASKS_DIR's `tasks.jsonl` that shares no seed with an `--exclude` dir over the server at
URL (started with `-x icij_dir`), and with `--seeds` every synthetic scenario with a
signal on each of those worlds. `--policy guided` (or any server policy) runs the
baseline through the same loop. Each batch is scored from its numeric
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


def train(dir: Path, syn: list[Path], name: str, numeric: bool = False) -> None:
    d = loaded(dir)
    parts = [replace(d, x=np.hstack([d.x, answers(dir, d, np.ones(len(d.y), bool), 0.0)]))]
    for s in syn:
        e = data.load(s, SCENARIOS)
        nan = np.full((len(e.y), len(questions.NAMES)), np.nan, np.float32)
        parts.append(replace(e, x=np.hstack([e.x, nan])))
    d = data.concat(parts)
    print(f"  {len(d.run):,} batches, {len(d.y):,} candidates", flush=True)
    every = np.ones(len(d.y), bool)
    names = data.NAMES + ([] if numeric else questions.NAMES)
    d = replace(d, x=d.x[:, : len(names)])
    MODELS.mkdir(exist_ok=True)
    for seed in range(FITS):
        m = rank.fit(d, every, names, seed=seed, reweight=True)
        m.save_model(str(MODELS / f"{name}.{seed}.cbm"))
        print(f"  saved {name}.{seed}", flush=True)


def call(method: str, url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        url,
        None if body is None else json.dumps(body).encode(),
        {"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def decide(models: list, task: dict, ask: dict, got: dict) -> list[dict]:
    """One decision per candidate of `ask`."""
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
    score = np.log(p) - x[:, rank.LEFT]
    interest = task["relation_interest"]
    breadth = max((ctx["budget"] - ctx["fetched"]) // 64, 16)
    out = []
    for c, s in zip(cands, score):
        rem = [max(a - b, 0) for a, b in zip(c["peek"], c["fetched"])]
        take = [
            {"rel": r, "limit": min(n, breadth)}
            for r, n in enumerate(rem)
            if n > 0 and interest[r] > 0
        ]
        if not take or task["domain_interest"][c["domain"]] <= 0:
            out.append({"priority": SKIP})
        else:
            out.append({"priority": float(s), "take": take})
    return out


def run(url: str, req: dict, models: list, budget: float, spent0: float) -> dict:
    """One run to its end: the report, the batches decided, and whether one was late."""
    cache = client.Cache(CACHE / "jev.sqlite")
    started = time.time()
    r = call("POST", f"{url}/expand", req)
    run_id, task = r["id"], r["task"]
    batches, late, seen, known, uncached = 0, False, 0, 0, {}
    try:
        while True:
            a = call("GET", f"{url}/expand/{run_id}?wait=25")
            if a["state"] == "running":
                continue
            if a["state"] == "done":
                break
            texts = {c["id"]: c["text"] for c in a["candidates"] if c.get("text")}
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
            body = {"seq": a["seq"], "decisions": decide(models, task, a, got)}
            try:
                call("POST", f"{url}/expand/{run_id}", body)
            except urllib.error.HTTPError as e:
                # 409: the server stopped waiting for this batch.
                if e.code != 409:
                    raise
                late = True
            batches += 1
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
    a.add_argument("--syn", type=Path, nargs="*", default=[])
    a.add_argument("--model", default="v0")
    a.add_argument("--numeric", action="store_true", help="train: without Jev answers")
    a.add_argument("--policy", default="remote")
    a.add_argument("--exclude", type=Path, nargs="*", default=[])
    a.add_argument("--seeds", type=int, nargs="*", default=[], help="synthetic worlds")
    a.add_argument("--limit", type=int)
    a.add_argument("--budget", type=float, default=0.0)
    a.add_argument("--parallel", type=int, default=1)
    a.add_argument("--spread", type=float, default=1.0)
    a.add_argument("--runs", type=int, default=4, help="concurrent runs")
    a.add_argument("--out", type=Path, default=CACHE / "serve.jsonl")
    a.add_argument("--uncached", type=Path, help="run: where to list what the cache lacks")
    args = a.parse_args()
    if args.cmd == "train":
        train(Path(args.target), args.syn, args.model, args.numeric)
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
        fresh = [t for t in data.read_tasks(dir).values() if not seen & set(t["seeds"])]
        for t in fresh[: args.limit]:
            reqs.append(
                {
                    **common,
                    "world": "icij",
                    "goal": t["goal"],
                    "seeds": t["seeds"],
                    "budget": t["budget"],
                }
            )
    reqs += [{**common, "scenario": s, "seed": w} for w in args.seeds for s in SCENARIOS]
    spent0 = client.Cache(CACHE / "jev.sqlite").spent()
    rows, lacking = [], {}
    with ThreadPoolExecutor(args.runs) as pool, args.out.open("w") as out:
        for n, row in enumerate(
            pool.map(lambda r: run(args.target, r, models, args.budget, spent0), reqs), 1
        ):
            lacking.update(row.pop("uncached"))
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            if n % 50 == 0 or n == len(reqs):
                print(f"{n}/{len(reqs)} runs", flush=True)
                summary(rows)
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
