import numpy as np

from jev import data, rank


def test_metrics_and_ties():
    y = np.array([0, 1, 0, 0, 1, 1])
    group = np.array([0, 0, 0, 1, 1, 1])
    perfect = np.array([0, 1, 0, 0, 1, 1], float)
    cost = np.array([5, 1, 7, 2, 1, 1], float)
    m = rank.per_group(perfect, y, group, 1, cost)
    assert len(m["top1"]) == 2 and m["top1"].all() and (m["ndcg@1"] == 1).all()
    assert not m["ahead"].any()
    # All tied: the negatives rank first.
    tied = rank.per_group(np.zeros(6), y, group, 1, cost)
    assert not tied["top1"].any()
    assert tied["ahead"].tolist() == [12, 2]
    lo, hi = rank.bootstrap(np.array([1.0, 1.0, 1.0]))
    assert lo == hi == 1.0


def test_split_keeps_shared_seeds_together():
    tasks = {f"connect.{i}": {"seeds": [i, 1000 + i // 3]} for i in range(200)}
    test = data.split(tasks)
    assert 0 < len(test) < 200
    for a in tasks:
        for b in tasks:
            if set(tasks[a]["seeds"]) & set(tasks[b]["seeds"]):
                assert (a in test) == (b in test)


def test_capped_keeps_positives():
    n = 300
    d = data.Data(
        x=np.zeros((n, 1), np.float32),
        y=(np.arange(n) % 50 == 0).astype(np.int8),
        group=np.repeat([0, 1], [250, 50]),
        prior=np.zeros(n),
        run=["a", "b"],
        task=["a", "b"],
        id=np.arange(n),
    )
    m = rank.capped(d, np.ones(n, bool), 8, 0)
    assert m[:250].sum() == 8 and m[250:].sum() == 8
    assert (d.y[m] == 1).sum() == 6  # every positive: 5 in group 0, 1 in group 1


def test_weights_undo_the_cut():
    n = 300
    d = data.Data(
        x=np.zeros((n, 1), np.float32),
        y=(np.arange(n) % 50 == 0).astype(np.int8),
        group=np.repeat([0, 1], [250, 50]),
        prior=np.zeros(n),
        run=["a", "b"],
        task=["a", "b"],
        id=np.arange(n),
    )
    every = np.ones(n, bool)
    m = rank.capped(d, every, 64, 0)
    w = rank.weights(d, every, m)
    y, g = d.y[m], d.group[m]
    assert (w[y == 1] == 1).all()
    # Group 0: 245 negatives, 59 kept; group 1 is under the cap and keeps all 49.
    assert np.allclose(w[(y == 0) & (g == 0)], 245 / 59)
    assert (w[(y == 0) & (g == 1)] == 1).all()
    assert np.isclose(w[g == 0].sum(), 250)


def test_concat_shifts_groups():
    def two(n):
        return data.Data(
            x=np.zeros((n, 1), np.float32),
            y=np.zeros(n, np.int8),
            group=np.repeat([0, 1], [1, n - 1]),
            prior=np.zeros(n),
            run=["a", "b"],
            task=["a", "b"],
            id=np.arange(n),
        )

    d = data.concat([two(3), two(4)])
    assert d.group.tolist() == [0, 1, 1, 2, 3, 3, 3]
    assert d.run == ["a", "b", "a", "b"] and len(d.id) == 7


def test_credit_weighs_positives_only():
    n = 64
    d = data.Data(
        x=np.arange(n, dtype=np.float32)[:, None],
        y=(np.arange(n) % 2).astype(np.int8),
        group=np.zeros(n, np.int64),
        prior=np.zeros(n),
        run=["a"],
        task=["a"],
        id=np.arange(n),
    )
    every = np.ones(n, bool)
    low = rank.fit(d, every, ["x"], iterations=20, credit=np.full(n, 0.1))
    base = rank.fit(d, every, ["x"], iterations=20)
    # Down-weighted positives pull the predicted rate down.
    assert low.predict_proba(d.x)[:, 1].mean() < base.predict_proba(d.x)[:, 1].mean()
