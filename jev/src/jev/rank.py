"""The numeric-only policy model (P1), and how a batch ordering is scored.

A candidate's priority is the chance it is on the oracle's path per edge it costs:
`log p - log(1 + relevant edges left)`, with `p` from a CatBoost classifier. A ranker
trained on labels alone learns which entities lie on the path, not what they cost to
expand, and orders expensive connectors before cheap ones: better top-1, more edges
spent before the first hit. A calibrated probability divided by cost fixes that
without a tuned weight.
"""

import numpy as np
from catboost import CatBoostClassifier, Pool

from .data import NAMES, Data

LEFT = NAMES.index("log_relevant")


def per_group(
    score: np.ndarray, y: np.ndarray, group: np.ndarray, k: int, cost: np.ndarray
) -> dict[str, np.ndarray]:
    """Per group with a positive and a negative: NDCG@k, recall@k (positives in the
    top k over min(k, positives)), top-1 (the first pick is positive), and the edges
    (`cost` per candidate) ranked ahead of the first positive, which is what a policy
    pays for a wrong order. Ties break against the score: a positive tied with a
    negative ranks below it."""
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    ends = np.r_[starts[1:], len(group)]
    out: dict[str, list[float]] = {f"ndcg@{k}": [], f"recall@{k}": [], "top1": [], "ahead": []}
    disc = 1 / np.log2(np.arange(2, k + 2))
    for a, b in zip(starts, ends):
        ys = y[a:b]
        pos = int(ys.sum())
        if pos == 0 or pos == len(ys):
            continue
        order = np.lexsort((ys, -score[a:b]))
        first = int(np.argmax(ys[order]))
        got = ys[order[:k]]
        out[f"ndcg@{k}"].append((got * disc[: len(got)]).sum() / disc[: min(pos, k)].sum())
        out[f"recall@{k}"].append(got.sum() / min(pos, k))
        out["top1"].append(float(got[0]))
        out["ahead"].append(float(cost[a:b][order[:first]].sum()))
    return {name: np.asarray(v) for name, v in out.items()}


def bootstrap(diff: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean of `diff`, resampling groups."""
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, (n, len(diff))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rows(d: Data, groups: np.ndarray) -> np.ndarray:
    """Row mask for the given group numbers."""
    return np.isin(d.group, groups)


def capped(d: Data, m: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """`m` with each group cut to `cap` rows: its positives first, then a random
    sample of its negatives. A hub's expansion offers tens of thousands of candidates
    in one batch; uncut, those few batches would be most of the training set."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(m)
    # Within each group: positives, then negatives in random order.
    idx = idx[np.lexsort((rng.random(len(idx)), -d.y[idx], d.group[idx]))]
    g = d.group[idx]
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    rank_in = np.arange(len(idx)) - np.repeat(starts, np.diff(np.r_[starts, len(idx)]))
    out = np.zeros_like(m)
    out[idx[rank_in < cap]] = True
    return out


def fit(
    d: Data, train: np.ndarray, iterations: int = 500, cap: int = 64, seed: int = 0
) -> CatBoostClassifier:
    """A classifier on the rows `train` (a mask), each group cut to `cap` rows."""
    m = capped(d, train, cap, seed)
    model = CatBoostClassifier(
        iterations=iterations, random_seed=seed, verbose=False, allow_writing_files=False
    )
    model.fit(Pool(d.x[m], label=d.y[m], feature_names=NAMES))
    return model


def priority(model: CatBoostClassifier, x: np.ndarray) -> np.ndarray:
    """Chance on the path per edge: `log p - log(1 + relevant edges left)`."""
    p = model.predict_proba(Pool(x, feature_names=NAMES))[:, 1]
    return np.log(p) - x[:, LEFT]
