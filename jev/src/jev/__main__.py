"""`uv run python -m jev DIR [K]`: train the numeric-only model on the training tasks'
batches (every policy) and compare its ordering with the policy's own on the test
tasks' batches, per kind and policy. `ahead` is the edges ranked before the first
positive; its interval is the paired 95% bootstrap of model minus policy.

With Jev answers cached in `DIR/jev.sqlite` (`python -m jev.ask DIR`), it also fits
numeric + Jev and reports it the same way: the P2 ablation."""

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import client, data, questions, rank


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
    train = rank.rows(d, groups[~is_test])
    models = {"numeric": (rank.fit(d, train), d.x)}
    if (dir / "jev.sqlite").exists():
        texts = data.texts(dir)
        got = client.answers(
            {int(i): texts[int(i)] for i in np.unique(d.id)},
            questions.QUESTIONS,
            client.Cache(dir / "jev.sqlite"),
            asking=False,
        )
        x = np.hstack([d.x, questions.features(d.id, got)])
        print(f"Jev answers for {np.mean(~np.isnan(x[:, -1])):.1%} of rows")
        both = data.NAMES + questions.NAMES
        models["jev"] = (rank.fit(replace(d, x=x), train, both), x)
    print(f"trained in {time.time() - started:.0f}s")
    for kind in ("connect", "alert"):
        for policy in ("guided", "guided_pushdown"):
            pick = groups[
                is_test & np.array([r.startswith(kind) and r.endswith(f".{policy}") for r in d.run])
            ]
            m = rank.rows(d, pick)
            if not m.any():
                continue
            y, g = d.y[m], d.group[m]
            left = np.expm1(d.x[m, rank.LEFT])
            out = {policy: rank.per_group(d.prior[m], y, g, k, left)}
            for name, (model, x) in models.items():
                out[name] = rank.per_group(rank.priority(model, x[m]), y, g, k, left)
            n = len(out[policy]["top1"])
            print(f"\n{kind} test, {policy} batches ({n} with mixed labels)")
            if n == 0:
                continue
            print(" " * 11 + "".join(f"{name:>16}" for name in out))
            for metric in out[policy]:
                print(f"  {metric:9}" + "".join(f"{r[metric].mean():16.3f}" for r in out.values()))
            print(
                "  ahead p90"
                + "".join(f"{np.percentile(r['ahead'], 90):16.1f}" for r in out.values())
            )
            pairs = [(name, policy) for name in models] + [("jev", "numeric")] * ("jev" in models)
            for a, b in pairs:
                lo, hi = rank.bootstrap(out[a]["ahead"] - out[b]["ahead"])
                print(f"  ahead, {a} - {b}: 95% CI [{lo:.1f}, {hi:.1f}]")
    for name, (model, _) in models.items():
        imp = model.get_feature_importance(type="PredictionValuesChange")
        top = sorted(zip(imp, model.feature_names_), reverse=True)[:12]
        print(f"\n{name} top features:", ", ".join(f"{n} {v:.1f}" for v, n in top))


main()
