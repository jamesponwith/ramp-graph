"""`uv run python -m jev.holdout TRAIN_DIR TEST_DIR [BUDGET]`: train on every task of
TRAIN_DIR and score on the tasks of TEST_DIR (another bench seed, `RAMP_SEED`) that
share no seed entity with them: a larger held-out set than the 85 test tasks of one
dataset's split.

v0 answers come from the cache. Entities it lacks are asked only within BUDGET
dollars (default 0, and then they stay NaN). Numeric and v0 are each a four-seed
average, against `Guided`'s own order on the same batches.
"""

import sys
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
    train_dir, test_dir = Path(sys.argv[1]), Path(sys.argv[2])
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    tr, te = loaded(train_dir), data.load(test_dir)
    seen = {s for t in data.read_tasks(train_dir).values() for s in t["seeds"]}
    fresh = {name for name, t in data.read_tasks(test_dir).items() if not seen & set(t["seeds"])}
    pos = np.bincount(te.group, te.y, len(te.run))
    size = np.bincount(te.group, minlength=len(te.run))
    scored = np.flatnonzero(
        np.isin(te.task, list(fresh))
        & np.array([r.endswith(".guided") for r in te.run])
        & (pos > 0)
        & (pos < size)
    )
    m = rank.rows(te, scored)
    print(f"{len(fresh)} fresh test tasks, {len(scored)} scored batches")
    jte = answers(test_dir, te, m, budget)
    covered = np.mean(~np.isnan(jte[m, 0]))
    print(f"v0 answers for {covered:.2%} of scored test rows")
    every = np.ones(len(tr.y), bool)
    variants = [("numeric", tr.x, te.x, data.NAMES)]
    if covered == 1:
        jtr = answers(train_dir, tr, every, 0.0)
        both = data.NAMES + questions.NAMES
        variants.append(("v0", np.hstack([tr.x, jtr]), np.hstack([te.x, jte]), both))
    else:
        print("  v0 skipped: a partial column would bias it (pass a BUDGET to ask the rest)")

    kind = np.array([r.split(".")[0] for r in te.run])[scored]
    # Batches of one task are correlated, and long tasks have hundreds: every metric is
    # averaged per task first, and intervals resample tasks.
    task_of = np.unique(np.array(te.task)[scored], return_inverse=True)[1]

    def by_task(v: np.ndarray, sel: np.ndarray) -> np.ndarray:
        t = task_of[sel]
        return np.bincount(t, v[sel])[np.unique(t)] / np.bincount(t)[np.unique(t)]

    left = np.expm1(te.x[m, rank.LEFT])

    def per(score: np.ndarray) -> dict[str, np.ndarray]:
        return rank.per_group(score, te.y[m], te.group[m], 5, left)

    out = {"guided": per(te.prior[m])}
    for name, xtr, xte, names in variants:
        p = np.zeros(m.sum())
        for seed in range(FITS):
            model = rank.fit(replace(tr, x=xtr), every, names, seed=seed)
            p += model.predict_proba(xte[m])[:, 1] / FITS
        out[name] = per(np.log(p) - te.x[m, rank.LEFT])
        print(f"  fitted {name}", flush=True)

    print(" " * 18 + "".join(f"{n:>12}" for n in out))
    for k in ("connect", "alert"):
        sel = kind == k
        print(f"{k} ({sel.sum()} batches, {len(np.unique(task_of[sel]))} tasks; per-task means)")
        for metric, f in (
            ("log1p(ahead)", lambda r: np.log1p(r["ahead"])),
            ("ahead", lambda r: r["ahead"]),
            ("top1", lambda r: r["top1"]),
        ):
            means = (by_task(f(r), sel).mean() for r in out.values())
            print(f"  {metric:16}" + "".join(f"{v:12.3f}" for v in means))
        pairs = [("numeric", "guided"), ("v0", "guided"), ("v0", "numeric")]
        for a, b in (p for p in pairs if p[0] in out):
            diff = np.log1p(out[a]["ahead"]) - np.log1p(out[b]["ahead"])
            lo, hi = rank.bootstrap(by_task(diff, sel))
            print(f"  log1p(ahead), {a} - {b}: 95% CI [{lo:+.3f}, {hi:+.3f}]")


main()
