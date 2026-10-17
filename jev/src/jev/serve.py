"""`jev serve` (P4): steer `/expand` runs with the v0 model.

`uv run python -m jev.serve train DIR`: fit the four-seed v0 model, reweighted, on
every task of DIR (cached answers only, no API calls), and save it beside the cache.

`uv run python -m jev.serve run URL [TASKS_DIR] [options]`: run each ICIJ task of
TASKS_DIR's `tasks.jsonl` over the server at URL (started with `-x icij_dir`), and with
`--scenarios` every synthetic scenario too. Each batch is scored from its numeric
features and the cached Jev answers about its candidates: priority is
`log p - log(1 + relevant edges left)`, in nats like `Guided`'s, so `--spread 1`
means the same. Breadth is `Guided`'s rule: every relevant relation, up to 1/64 of the
remaining budget (at least 16) per visit. An entity the cache lacks gets NaN features,
unless `--budget` dollars (shared by the whole run, off by default) pays to ask it
before its batch is answered. One JSON line per run goes to `--out`.
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
SCENARIOS = ("path-5hop", "path-7hop", "pivot-busy", "alert", "explore")
# What a candidate not worth expanding gets: JSON has no -inf.
SKIP = -1e300


def train(dir: Path) -> None:
    d = loaded(dir)
    every = np.ones(len(d.y), bool)
    x = np.hstack([d.x, answers(dir, d, every, 0.0)])
    names = data.NAMES + questions.NAMES
    MODELS.mkdir(exist_ok=True)
    for seed in range(FITS):
        m = rank.fit(replace(d, x=x), every, names, seed=seed, reweight=True)
        m.save_model(str(MODELS / f"v0.{seed}.cbm"))
        print(f"  saved v0.{seed}", flush=True)


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
    p = sum(m.predict_proba(x)[:, 1] for m in models) / len(models)
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
    batches, late = 0, False
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
    solved = rep["done_at"] is not None
    return {
        **req,
        "solved": solved,
        "fetched": rep["fetched"],
        "rounds": rep["rounds"],
        "batches": batches,
        "late": late,
        "wall": time.time() - started,
        "report": rep,
    }


def summary(rows: list[dict]) -> None:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r.get("goal") or f"syn {r['scenario']}", []).append(r)
    for k, rs in by.items():
        f = [r["fetched"] for r in rs if r["solved"]]
        p50, p90 = (np.percentile(f, q) if f else math.nan for q in (50, 90))
        print(
            f"  {k:10} solved {sum(r['solved'] for r in rs)}/{len(rs)}  fetched p50 {p50:,.0f}"
            f" p90 {p90:,.0f}  late {sum(r['late'] for r in rs)}"
            f"  wall {sum(r['wall'] for r in rs):.0f}s"
        )


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["train", "run"])
    a.add_argument("target", help="train: batches DIR; run: server URL")
    a.add_argument("tasks", type=Path, nargs="?")
    a.add_argument("--scenarios", action="store_true")
    a.add_argument("--limit", type=int)
    a.add_argument("--budget", type=float, default=0.0)
    a.add_argument("--parallel", type=int, default=1)
    a.add_argument("--spread", type=float, default=1.0)
    a.add_argument("--runs", type=int, default=4, help="concurrent runs")
    a.add_argument("--out", type=Path, default=CACHE / "serve.jsonl")
    args = a.parse_args()
    if args.cmd == "train":
        train(Path(args.target))
        return
    models = []
    for seed in range(FITS):
        m = CatBoostClassifier()
        m.load_model(str(MODELS / f"v0.{seed}.cbm"))
        models.append(m)
    common = {"policy": "remote", "parallel": args.parallel, "spread": args.spread}
    reqs = []
    if args.tasks:
        for t in list(data.read_tasks(args.tasks).values())[: args.limit]:
            reqs.append(
                {
                    **common,
                    "world": "icij",
                    "goal": t["goal"],
                    "seeds": t["seeds"],
                    "budget": t["budget"],
                }
            )
    if args.scenarios:
        reqs += [{**common, "scenario": s} for s in SCENARIOS]
    spent0 = client.Cache(CACHE / "jev.sqlite").spent()
    rows = []
    with ThreadPoolExecutor(args.runs) as pool, args.out.open("w") as out:
        for n, row in enumerate(
            pool.map(lambda r: run(args.target, r, models, args.budget, spent0), reqs), 1
        ):
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            if n % 50 == 0 or n == len(reqs):
                print(f"{n}/{len(reqs)} runs", flush=True)
                summary(rows)
    print(f"Jev spend this run: ${client.Cache(CACHE / 'jev.sqlite').spent() - spent0:.2f}")


if __name__ == "__main__":
    main()
