"""`uv run python -m jev DIR [K]`: train the numeric-only model on the training tasks'
batches (every policy) and compare its ordering with the policy's own on the test
tasks' batches, per kind and policy. `ahead` is the edges ranked before the first
positive; its interval is the paired 95% bootstrap of model minus policy."""

import sys
import time
from pathlib import Path

import numpy as np

from . import data, rank


def main() -> None:
    dir = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    started = time.time()
    d = data.load(dir)
    test = data.split(data.read_tasks(dir))
    is_test = np.array([t in test for t in d.task])
    print(
        f"{len(d.y):,} candidates in {len(d.run):,} batches, {d.y.mean():.1%} positive; "
        f"{len(test)} test tasks; loaded in {time.time() - started:.0f}s"
    )
    groups = np.arange(len(d.run))
    model = rank.fit(d, rank.rows(d, groups[~is_test]))
    print(f"trained in {time.time() - started:.0f}s")
    for kind in ("connect", "alert"):
        for policy in ("guided", "guided_pushdown"):
            pick = groups[
                is_test & np.array([r.startswith(kind) and r.endswith(f".{policy}") for r in d.run])
            ]
            m = rank.rows(d, pick)
            if not m.any():
                continue
            y, g, x = d.y[m], d.group[m], d.x[m]
            left = np.expm1(x[:, rank.LEFT])
            base = rank.per_group(d.prior[m], y, g, k, left)
            ours = rank.per_group(rank.priority(model, x), y, g, k, left)
            n = len(base["top1"])
            print(f"\n{kind} test, {policy} batches ({n} with mixed labels)")
            if n == 0:
                continue
            for name in base:
                print(
                    f"  {name:9} {policy:>16} {base[name].mean():8.3f}   model {ours[name].mean():8.3f}"
                )
            lo, hi = rank.bootstrap(ours["ahead"] - base["ahead"])
            print(
                f"  ahead p90 {np.percentile(base['ahead'], 90):25.1f}   model "
                f"{np.percentile(ours['ahead'], 90):8.1f}\n"
                f"  ahead, model - {policy}: 95% CI [{lo:.1f}, {hi:.1f}]"
            )
    imp = model.get_feature_importance(type="PredictionValuesChange")
    top = sorted(zip(imp, data.NAMES), reverse=True)[:10]
    print("\ntop features:", ", ".join(f"{n} {v:.1f}" for v, n in top))


main()
