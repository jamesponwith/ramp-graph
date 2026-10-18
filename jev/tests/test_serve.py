import numpy as np

from jev import data, questions, serve


class Half:
    feature_names_ = data.NAMES + questions.NAMES

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


def test_pushdown_filters_towards_the_target():
    # officer -0- entity -0- officer; the target is an officer's country.
    task = {
        "goal": "alert",
        "relation_interest": [1] * 8,
        "domain_interest": [1] * 8,
        "relations": [{"from": 1, "to": 0}],
        "shape": [[1, 0, 1]],
        "target": {"attr": "country", "value": "BMU"},
        "filters": True,
    }
    seed = cand(1, [40] + [0] * 7, domain=1)
    seed["hops"] = [0, None]
    company = cand(2, [9] + [0] * 7, domain=0)
    ask = {
        "context": {"fetched": 0, "budget": 6400, "by_side": [1, 0], "steps": 1},
        "candidates": [seed, company],
    }
    got = {i: dict.fromkeys(questions.QUESTIONS) for i in (1, 2)}
    plain = serve.decide([Half()], task, ask, got)
    pushed = serve.decide([Half()], task, ask, got, push=True)
    # From the seed: officer_of filtered by a semi-join to a target officer.
    assert pushed[0]["take"] == [
        {"rel": 0, "limit": 100, "filter": {"via": 0, "then": task["target"]}}
    ]
    # From the company, one step from the end: the target itself.
    assert pushed[1]["take"] == [{"rel": 0, "limit": 100, "filter": task["target"]}]
    assert pushed[0]["priority"] > plain[0]["priority"]
    # Already fetched filtered: back to the plain fetch.
    seed["pushed"] = [True] + [False] * 7
    assert serve.decide([Half()], task, ask, got, push=True)[0] == plain[0]
