import math

import numpy as np
import pytest

from jev import client, questions


def fake(calls, model=client.PINNED):
    def ask(text, qs, _model):
        calls.append(sorted(qs))
        answers = {q: {"type": "noul", "noul": 0.25} for q in qs}
        usage = {"input_tokens": 1_000_000, "output_tokens": 0}
        return {"model": model, "answers": answers, "usage": usage}

    return ask


def test_answers_ask_once_then_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(client, "ask", fake(calls))
    qs = {"a": {"type": "noul", "instructions": "A?"}, "b": {"type": "noul", "instructions": "B?"}}
    cache = client.Cache(tmp_path / "c.sqlite")
    assert client.answers({1: "x", 2: "y"}, qs, cache, 1.0)[2]["b"]["noul"] == 0.25
    assert calls == [["a", "b"], ["a", "b"]]
    assert cache.spent() == pytest.approx(2 * 0.042)
    # A revised question re-asks only itself; with no budget, nothing is asked.
    qs["b"] = {"type": "noul", "instructions": "B, revised?"}
    client.answers({1: "x"}, qs, cache, 1.0)
    assert calls[2:] == [["b"]]
    assert client.answers({3: "z"}, qs, cache)[3]["a"] is None
    assert len(calls) == 3


def test_answers_keep_to_the_budget_and_the_model(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(client, "ask", fake(calls))
    qs = {"a": {"type": "noul", "instructions": "A?"}}
    cache = client.Cache(tmp_path / "c.sqlite")
    texts = {i: "x" * 100 for i in range(100)}
    # The estimate alone refuses, before any call.
    with pytest.raises(client.OverBudget):
        client.answers(texts, qs, cache, client.estimate({0: ["a"]}, texts) * 50)
    assert calls == []
    # Billed usage (0.042 a call here) stops the run past the budget.
    with pytest.raises(client.OverBudget):
        client.answers(texts, qs, cache, 0.1, workers=1)
    assert len(calls) == 3
    assert len(cache.get([client.key(client.MODEL, qs["a"], "x" * 100)])) == 1
    monkeypatch.setattr(client, "ask", fake(calls, "jev-9"))
    with pytest.raises(RuntimeError, match="jev-9"):
        client.answers({7: "new"}, qs, cache, 1.0)


def test_features_scale_scores_and_mark_missing():
    score = {"type": "score", "score": 1.5, "legend": {"0": "", "1": "", "2": "", "3": ""}}
    assert questions.value(score) == 0.5
    answers = {7: {q: {"type": "noul", "noul": 0.9} for q in questions.QUESTIONS}}
    answers[8] = dict.fromkeys(questions.QUESTIONS)
    x = questions.features(np.array([8, 7, 7]), answers)
    assert x.shape == (3, len(questions.NAMES))
    assert all(math.isnan(v) for v in x[0]) and np.allclose(x[1:], 0.9)
