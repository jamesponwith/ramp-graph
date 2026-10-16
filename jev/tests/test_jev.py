import math

import numpy as np

from jev import client, questions


def test_answers_ask_once_then_cache(tmp_path, monkeypatch):
    calls = []

    def ask(text, qs, model):
        calls.append(sorted(qs))
        return {q: {"type": "noul", "noul": 0.25} for q in qs}

    monkeypatch.setattr(client, "ask", ask)
    qs = {"a": {"type": "noul", "instructions": "A?"}, "b": {"type": "noul", "instructions": "B?"}}
    cache = client.Cache(tmp_path / "c.sqlite")
    assert client.answers({1: "x", 2: "y"}, qs, cache)[2]["b"]["noul"] == 0.25
    assert calls == [["a", "b"], ["a", "b"]]
    # A revised question re-asks only itself; a cache-only read of new text asks nothing.
    qs["b"] = {"type": "noul", "instructions": "B, revised?"}
    client.answers({1: "x"}, qs, cache)
    assert calls[2:] == [["b"]]
    assert client.answers({3: "z"}, qs, cache, asking=False)[3]["a"] is None
    assert len(calls) == 3


def test_features_scale_scores_and_mark_missing():
    score = {"type": "score", "score": 1.5, "legend": {"0": "", "1": "", "2": "", "3": ""}}
    assert questions.value(score) == 0.5
    answers = {7: {q: {"type": "noul", "noul": 0.9} for q in questions.QUESTIONS}}
    answers[8] = dict.fromkeys(questions.QUESTIONS)
    x = questions.features(np.array([8, 7, 7]), answers)
    assert x.shape == (3, len(questions.NAMES))
    assert all(math.isnan(v) for v in x[0]) and np.allclose(x[1:], 0.9)
