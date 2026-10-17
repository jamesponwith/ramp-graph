import numpy as np

from jev import questions, serve


class Half:
    def predict_proba(self, x):
        return np.full((len(x), 2), 0.5)


def cand(id, peek, fetched=None, domain=0):
    return {
        "id": id,
        "domain": domain,
        "peek": peek,
        "fetched": fetched or [0] * 8,
        "hops": [1, None],
        "cost": [1.0, None],
        "known": 0,
        "known_by_domain": [0] * 8,
        "watched": 0,
        "is_watched": False,
        "alerts": 0,
        "probes": None,
        "pushed": [False] * 8,
    }


def test_breadth_skip_and_cost():
    task = {
        "goal": "connect",
        "relation_interest": [1, 0] + [1] * 6,
        "domain_interest": [1, 0] + [1] * 6,
    }
    ask = {
        "context": {"fetched": 0, "budget": 6400, "by_side": [1, 0], "steps": 3},
        "candidates": [
            cand(1, [500, 7] + [0] * 6),  # hub: capped at 6400 / 64
            cand(2, [3, 0] + [0] * 6),  # cheap
            cand(3, [0, 9] + [0] * 6),  # only an uninteresting relation
            cand(4, [5] + [0] * 7, domain=1),  # uninteresting domain
        ],
    }
    got = {i: dict.fromkeys(questions.QUESTIONS) for i in range(1, 5)}
    hub, cheap, rel, dom = serve.decide([Half()], task, ask, got)
    assert hub["take"] == [{"rel": 0, "limit": 100}]
    assert cheap["take"] == [{"rel": 0, "limit": 3}]
    assert hub["priority"] < cheap["priority"]
    assert "take" not in rel and "take" not in dom
