"""`uv run python -m jev.holdout TRAIN_DIR TEST_DIR [options]`: train on every task of
TRAIN_DIR and score on the tasks of TEST_DIR (another bench seed, `RAMP_SEED`) that
share no seed entity with TRAIN_DIR or any `--exclude` dir.

- `--sweep`: numeric only, for each cost setting (`reweight` off and on, `alpha` 1 to 4):
  the free tuning run, on a validation seed.
- Otherwise, numeric and v0 at `--alpha`/`--reweight` against the P1 setting and
  `Guided`'s own order. v0 answers come from the cache. Entities it lacks are asked
  only within `--budget` dollars, and v0 is skipped while any are missing.

Each model is a four-seed average. Batches of one task are correlated, and long tasks
have hundreds, so every metric is averaged per task first, and intervals resample
tasks. `ahead` (edges before the first positive) is what the budget pays.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import client, data, questions, rank
from .research import CACHE, loaded

FITS = 4


def answers(dir: Path, d: data.Data, rows: np.ndarray, budget: float) -> np.ndarray:
    """v0 columns for `d`, asking (within `budget`) only about the entities in `rows`;
    other rows read the cache and may stay NaN."""
    texts = data.texts(dir)
    got = client.answers(
        {int(i): texts[int(i)] for i in np.unique(d.id[rows])},
        questions.QUESTIONS,
        client.Cache(CACHE / "jev.sqlite"),
        budget,
    )
    cols = np.full((len(d.y), len(questions.NAMES)), np.nan, np.float32)
    cols[rows] = questions.features(d.id[rows], got)
    return cols


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("train", type=Path)
    a.add_argument("test", type=Path)
    a.add_argument("--exclude", type=Path, nargs="*", default=[])
    a.add_argument("--budget", type=float, default=0.0)
    a.add_argument("--alpha", type=float, default=1.0)
    a.add_argument("--reweight", action="store_true")
    a.add_argument("--sweep", action="store_true")
    args = a.parse_args()
    tr, te = loaded(args.train), data.load(args.test)
    seen = {
        s
        for dir in [args.train, *args.exclude]
        for t in data.read_tasks(dir).values()
        for s in t["seeds"]
    }
    fresh = {n for n, t in data.read_tasks(args.test).items() if not seen & set(t["seeds"])}
    pos = np.bincount(te.group, te.y, len(te.run))
    size = np.bincount(te.group, minlength=len(te.run))
    scored = np.flatnonzero(
        np.isin(te.task, list(fresh))
        & np.array([r.endswith(".guided") for r in te.run])
        & (pos > 0)
        & (pos < size)
    )
    m = rank.rows(te, scored)
    print(f"{len(fresh)} fresh tasks, {len(scored)} scored batches", flush=True)
    kind = np.array([r.split(".")[0] for r in te.run])[scored]
    task_of = np.unique(np.array(te.task)[scored], return_inverse=True)[1]
    left = np.expm1(te.x[m, rank.LEFT])
    every = np.ones(len(tr.y), bool)

    def by_task(v: np.ndarray, sel: np.ndarray) -> np.ndarray:
        t = task_of[sel]
        u = np.unique(t)
        return np.bincount(t, v[sel])[u] / np.bincount(t)[u]

    def p_of(xtr: np.ndarray, xte: np.ndarray, names: list[str], reweight: bool) -> np.ndarray:
        p = np.zeros(m.sum())
        for seed in range(FITS):
            model = rank.fit(replace(tr, x=xtr), every, names, seed=seed, reweight=reweight)
            p += model.predict_proba(xte[m])[:, 1] / FITS
        return p

    def ahead(score: np.ndarray) -> np.ndarray:
        return rank.per_group(score, te.y[m], te.group[m], 5, left)["ahead"]

    out = {"guided": ahead(te.prior[m])}
    cost = te.x[m, rank.LEFT]
    if args.sweep:
        for reweight in (False, True):
            p = p_of(tr.x, te.x, data.NAMES, reweight)
            for alpha in (1.0, 1.5, 2.0, 3.0, 4.0):
                out[f"{'rw' if reweight else 'p'} a{alpha:g}"] = ahead(np.log(p) - alpha * cost)
            print(f"  fitted reweight={reweight}", flush=True)
    else:
        jte = answers(args.test, te, m, args.budget)
        covered = np.mean(~np.isnan(jte[m, 0]))
        print(f"v0 answers for {covered:.2%} of scored rows")
        variants = [("numeric", tr.x, te.x, data.NAMES)]
        if covered == 1:
            jtr = answers(args.train, tr, every, 0.0)
            both = data.NAMES + questions.NAMES
            variants.append(("v0", np.hstack([tr.x, jtr]), np.hstack([te.x, jte]), both))
        else:
            print("  v0 skipped: a partial column would bias it (--budget asks the rest)")
        for name, xtr, xte, names in variants:
            if args.alpha != 1 or args.reweight:
                out[f"{name} P1"] = ahead(np.log(p_of(xtr, xte, names, False)) - cost)
            p = p_of(xtr, xte, names, args.reweight)
            out[name] = ahead(np.log(p) - args.alpha * cost)
            print(f"  fitted {name}", flush=True)

    width = max(len(n) for n in out) + 2
    print(" " * 16 + "".join(f"{n:>{width}}" for n in out))
    for k in ("connect", "alert"):
        sel = kind == k
        print(f"{k} ({sel.sum()} batches, {len(np.unique(task_of[sel]))} tasks; per-task means)")
        for metric, f in (("log1p(ahead)", np.log1p), ("ahead", lambda v: v)):
            means = (by_task(f(r), sel).mean() for r in out.values())
            print(f"  {metric:14}" + "".join(f"{v:{width}.3f}" for v in means))
        for name in list(out)[1:]:
            lo, hi = rank.bootstrap(by_task(out[name] - out["guided"], sel))
            print(f"  ahead, {name} - guided: 95% CI [{lo:+.1f}, {hi:+.1f}]")
        if "v0" in out:
            lo, hi = rank.bootstrap(by_task(out["v0"] - out["numeric"], sel))
            print(f"  ahead, v0 - numeric: 95% CI [{lo:+.1f}, {hi:+.1f}]")


main()
