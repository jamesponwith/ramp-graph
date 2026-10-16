"""Question research (P3), on a validation split carved from the training tasks: the
test tasks are not looked at until the final question set is chosen.

`uv run python -m jev.research DIR drop`: refit without each cached question in turn
(no API calls) and report what each one is worth on validation. `seeds`: refit numeric
and v0 under four seeds, for the noise floor. `ensembles`: two disjoint four-seed
averages of each. `relevance BUDGET`: ask per-task relevance questions (see there); `final BUDGET`: the
same, scored once on the test tasks.

The objective is the mean of log(1 + edges ahead of the first positive) over
validation batches of both kinds: what a policy pays for a wrong order, without one
hub batch outweighing the rest.
"""

import pickle
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import client, data, questions, rank

CACHE = Path.home() / ".cache" / "ramp-jev"


def loaded(dir: Path) -> data.Data:
    """`data.load(dir)`, pickled beside the answer cache after the first time."""
    path = CACHE / "data.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    d = data.load(dir)
    path.write_bytes(pickle.dumps(d))
    return d


class Bench:
    """Train and validation rows, and the score of a feature set on them. `final`
    scores the test tasks instead of validation: once, for the chosen set."""

    def __init__(self, dir: Path, final: bool = False):
        self.d = loaded(dir)
        d = self.d
        tasks = data.read_tasks(dir)
        test = data.split(tasks)
        val = data.split({t: v for t, v in tasks.items() if t not in test}, salt="val")
        task = np.array(d.task)
        groups = np.arange(len(d.run))
        self.train = rank.rows(d, groups[~np.isin(task, list(test | val))])
        pos = np.bincount(d.group, d.y, len(d.run))
        size = np.bincount(d.group, minlength=len(d.run))
        guided = np.array([r.endswith(".guided") for r in d.run])
        scored = test if final else val
        self.val = groups[np.isin(task, list(scored)) & guided & (pos > 0) & (pos < size)]
        self.kinds = np.array([r.split(".")[0] for r in d.run])
        texts = data.texts(dir)
        self.texts = {int(i): texts[int(i)] for i in np.unique(d.id)}
        print(f"{len(scored)} {'test' if final else 'validation'} tasks, {len(self.val)} batches")

    def jev(self, qs: dict[str, dict]) -> np.ndarray:
        """Cached answers to `qs` as columns (NaN where not asked)."""
        got = client.answers(self.texts, qs, client.Cache(CACHE / "jev.sqlite"))
        return questions.features(self.d.id, got, qs)

    def score(
        self, cols: np.ndarray, names: list[str], seed: int = 0, fits: int = 1
    ) -> dict[str, np.ndarray]:
        """Per validation batch, `log1p(edges ahead)` and kind, for numeric + `cols`, with
        the probability averaged over `fits` seeds."""
        d = self.d
        x = np.hstack([d.x, cols]) if cols.size else d.x
        m = rank.rows(d, self.val)
        p = np.zeros(m.sum())
        for k in range(fits):
            model = rank.fit(replace(d, x=x), self.train, data.NAMES + names, seed=seed + k)
            p += model.predict_proba(x[m])[:, 1] / fits
        left = np.expm1(d.x[m, rank.LEFT])
        out = rank.per_group(np.log(p) - d.x[m, rank.LEFT], d.y[m], d.group[m], 5, left)
        return {"loss": np.log1p(out["ahead"]), "kind": self.kinds[self.val]}


def report(name: str, s: dict, base: dict | None) -> None:
    by = "  ".join(f"{k} {s['loss'][s['kind'] == k].mean():.3f}" for k in ("connect", "alert"))
    line = f"  {name:16} loss {s['loss'].mean():.3f} ({by})"
    if base is not None:
        lo, hi = rank.bootstrap(s["loss"] - base["loss"])
        line += f"  vs v0 [{lo:+.3f}, {hi:+.3f}]"
    print(line, flush=True)


def drop(dir: Path) -> None:
    started = time.time()
    b = Bench(dir)
    qs = questions.QUESTIONS
    cols = b.jev(qs)
    print(f"ready in {time.time() - started:.0f}s")
    report("numeric", b.score(np.empty((len(b.d.y), 0)), []), None)
    base = b.score(cols, questions.NAMES)
    report("v0", base, None)
    for j, q in enumerate(qs):
        keep = [k for k in range(len(qs)) if k != j]
        names = [questions.NAMES[k] for k in keep]
        report(f"v0 - {q}", b.score(cols[:, keep], names), base)


def seeds(dir: Path) -> None:
    """How much the score moves between fits of one feature set: the noise floor."""
    b = Bench(dir)
    cols = b.jev(questions.QUESTIONS)
    for seed in range(4):
        report(f"numeric seed {seed}", b.score(cols[:, :0], [], seed), None)
        report(f"v0 seed {seed}", b.score(cols, questions.NAMES, seed), None)


def ensembles(dir: Path) -> None:
    """Two disjoint 4-seed averages per feature set: does averaging tame the noise?"""
    b = Bench(dir)
    cols = b.jev(questions.QUESTIONS)
    for seed in (0, 4):
        report(f"numeric x4 @{seed}", b.score(cols[:, :0], [], seed, 4), None)
        report(f"v0 x4 @{seed}", b.score(cols, questions.NAMES, seed, 4), None)


# Questions about a candidate in the light of one task: its prompt and seeds.
RELEVANCE: dict[str, dict] = {
    "tie": {
        "type": "score",
        "instructions": "How closely is the candidate tied to the seed parties of the "
        "task, on the evidence of its own record?",
        "criteria": [
            "no visible tie",
            "same jurisdiction, region or service provider only",
            "shares a name, family name, address or company with a seed",
        ],
    },
    "link": {
        "type": "noul",
        "instructions": "Is the candidate likely to be part of the specific link the "
        "task asks for, rather than a mass registration agent, shared address or "
        "unrelated party?",
    },
    "fits": {
        "type": "noul",
        "instructions": "Does the candidate match what the task names: its people, "
        "companies or countries?",
    },
}


def relevance(dir: Path, final: bool = False) -> None:
    """Ask `RELEVANCE` about every validation (task, candidate) pair and a seeded
    sample of training batches' pairs, within `sys.argv[3]` dollars; then compare
    four-seed averages of numeric, v0 and v0 + relevance, all trained on the sampled
    batches. `final`: the same on the test tasks, against `Guided`'s own order too."""
    budget, train_pairs = float(sys.argv[3]), 30_000
    b = Bench(dir, final)
    d = b.d
    tasks = data.read_tasks(dir)
    task = np.array(d.task)[d.group]
    pos = np.bincount(d.group, d.y, len(d.run))
    size = np.bincount(d.group, minlength=len(d.run))
    mixed = ((pos > 0) & (pos < size))[d.group]
    # The 64-row cut is fixed here, so every fit and seed trains on the asked rows.
    train = rank.capped(d, b.train & mixed, 64, 0)
    # Whole batches drawn from every training task, until `train_pairs` pairs.
    groups = np.unique(d.group[train])
    random.Random(0).shuffle(groups)
    keep, seen = [], set()
    for g in groups:
        if len(seen) >= train_pairs:
            break
        keep.append(g)
        rows = train & (d.group == g)
        seen.update(zip(task[rows], d.id[rows].tolist()))
    b.train = train & np.isin(d.group, keep)
    val = rank.rows(d, b.val)
    asked = b.train | val
    pairs = sorted(set(zip(task[asked], d.id[asked].tolist())))
    print(
        f"{len(keep)} training batches from {len(set(task[b.train]))} tasks; {len(pairs):,} pairs"
    )

    def state(t: str, i: int) -> str:
        seeds = "\n".join(f"- {b.texts.get(s, '')[:300]}" for s in tasks[t]["seeds"])
        return f"Task: {tasks[t]['prompt']}\nSeeds:\n{seeds}\nCandidate: {b.texts[i]}"

    states = {n: state(t, i) for n, (t, i) in enumerate(pairs)}
    got = client.answers(states, RELEVANCE, client.Cache(CACHE / "jev.sqlite"), budget)
    by_pair = {p: [questions.value(got[n][q]) for q in RELEVANCE] for n, p in enumerate(pairs)}
    rel = np.full((len(d.y), len(RELEVANCE)), np.nan, np.float32)
    for r in np.flatnonzero(asked):
        rel[r] = by_pair[task[r], int(d.id[r])]
    names_rel = [f"rel_{q}" for q in RELEVANCE]
    v0 = b.jev(questions.QUESTIONS)
    base = b.score(v0, questions.NAMES, 0, 4)
    numeric = b.score(v0[:, :0], [], 0, 4)
    if final:
        left = np.expm1(d.x[val, rank.LEFT])
        out = rank.per_group(d.prior[val], d.y[val], d.group[val], 5, left)
        guided = {"loss": np.log1p(out["ahead"]), "kind": b.kinds[b.val]}
        report("guided", guided, base)
    report("numeric x4", numeric, base)
    report("v0 x4", base, None)
    report("numeric+rel x4", b.score(rel, names_rel, 0, 4), base)
    both = b.score(np.hstack([v0, rel]), questions.NAMES + names_rel, 0, 4)
    report("v0+rel x4", both, base)
    if final:
        for name, other in (("numeric", numeric), ("guided", guided)):
            lo, hi = rank.bootstrap(both["loss"] - other["loss"])
            print(f"  v0+rel - {name}: 95% CI [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    commands = {
        "drop": drop,
        "seeds": seeds,
        "ensembles": ensembles,
        "relevance": relevance,
        "final": lambda dir: relevance(dir, final=True),
    }
    commands[sys.argv[2]](Path(sys.argv[1]))
