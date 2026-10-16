"""TypeSafe System One client with a disk cache.

One request per entity carries every question it lacks. An answer is cached under
(model, question, entity text), so each entity is asked each question once, and a
revised question re-asks only itself.

Spending is capped: a call to `answers` estimates its cost first and refuses a budget
it would exceed, stops once the usage it is billed passes the budget, and keeps a
ledger of every call's cost in the cache. Answers come from one model version
(`PINNED`); a different one behind `jev-latest` stops the run rather than mix them in.
"""

import hashlib
import json
import os
import random
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
RETRY = {429, 529, 500, 502, 503, 504}
PINNED = "jev-1.13.0"
# $ per token, input as quoted. Output is unpriced in the docs; charged the same here.
PRICE = 0.042e-6
# Tokens per call: fixed overhead, per question in and out (measured), per text character.
OVERHEAD, PER_QUESTION, PER_CHAR = 330, 30, 0.3


class OverBudget(Exception):
    pass


def key(model: str, question: dict, text: str) -> str:
    q = json.dumps(question, sort_keys=True)
    return hashlib.sha256(f"{model}\0{q}\0{text}".encode()).hexdigest()


def ask(text: str, questions: dict[str, dict], model: str = MODEL, tries: int = 8) -> dict:
    """One call: the response (`answers`, `model`, `usage`). Retries 429/529/5xx with
    jittered backoff."""
    body = json.dumps({"model": model, "state": text, "questions": questions}).encode()
    req = urllib.request.Request(
        URL,
        body,
        {
            "Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    for n in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in RETRY or n == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if n == tries - 1:
                raise
        time.sleep(min(2**n, 30) * (0.5 + random.random()))
    raise AssertionError("unreachable")


class Cache:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.execute("create table if not exists a (k text primary key, v text)")
        self.db.execute("create table if not exists spent (at real, dollars real)")

    def spent(self) -> float:
        """Dollars billed through this cache, all runs since the ledger began (P3)."""
        return self.db.execute("select coalesce(sum(dollars), 0) from spent").fetchone()[0]

    def charge(self, usage: dict) -> float:
        dollars = (usage["input_tokens"] + usage["output_tokens"]) * PRICE
        self.db.execute("insert into spent values (?, ?)", (time.time(), dollars))
        return dollars

    def get(self, keys: list[str]) -> dict[str, dict]:
        out = {}
        for i in range(0, len(keys), 900):
            part = keys[i : i + 900]
            rows = self.db.execute(
                f"select k, v from a where k in ({','.join('?' * len(part))})", part
            )
            out.update((k, json.loads(v)) for k, v in rows)
        return out

    def put(self, items: Iterable[tuple[str, dict]]) -> None:
        """Uncommitted: callers commit in batches, since a commit per answer is slower
        than the API."""
        self.db.executemany(
            "insert or replace into a values (?, ?)", ((k, json.dumps(v)) for k, v in items)
        )


def estimate(todo: dict[int, list[str]], texts: dict[int, str]) -> float:
    """Dollars to ask `todo`, from the measured token counts."""
    tokens = sum(
        OVERHEAD + PER_QUESTION * len(qs) + PER_CHAR * len(texts[i]) for i, qs in todo.items()
    )
    return tokens * PRICE


def answers(
    texts: dict[int, str],
    questions: dict[str, dict],
    cache: Cache,
    budget: float = 0.0,
    model: str = MODEL,
    workers: int = 8,
    progress: int = 1000,
) -> dict[int, dict[str, dict | None]]:
    """`{entity id: {question id: answer}}`, asking Jev only for what the cache lacks,
    within `budget` dollars. With no budget it asks nothing, and what the cache lacks
    is `None`."""
    keys = {(i, q): key(model, d, t) for i, t in texts.items() for q, d in questions.items()}
    have = cache.get(list(keys.values()))
    todo: dict[int, list[str]] = {}
    for (i, q), k in keys.items():
        if budget > 0 and k not in have:
            todo.setdefault(i, []).append(q)
    if todo:
        cost = estimate(todo, texts)
        print(f"  asking {len(todo):,} entities, about ${cost:.2f} of ${budget:.2f}")
        if cost > budget:
            raise OverBudget(f"about ${cost:.2f} needed, ${budget:.2f} allowed")

    # Workers stop starting calls once billing passes the budget, so a run overshoots by
    # at most the calls in flight.
    lock, billed = threading.Lock(), [0.0]

    def one(item: tuple[int, list[str]]) -> tuple[int, list[str], dict | None]:
        i, qs = item
        with lock:
            if billed[0] > budget:
                return i, qs, None
        out = ask(texts[i], {q: questions[q] for q in qs}, model)
        with lock:
            billed[0] += (out["usage"]["input_tokens"] + out["usage"]["output_tokens"]) * PRICE
        return i, qs, out

    started = time.time()
    pool = ThreadPoolExecutor(workers)
    try:
        for n, (i, qs, out) in enumerate(pool.map(one, todo.items()), 1):
            if out is None:
                raise OverBudget(f"stopped at ${billed[0]:.2f} after {n - 1:,} entities")
            cache.charge(out["usage"])
            if out["model"] != PINNED:
                raise RuntimeError(f"{model} is now {out['model']}, cached answers are {PINNED}")
            got = [(keys[i, q], out["answers"][q]) for q in qs]
            cache.put(got)
            have.update(got)
            if n % 200 == 0:
                cache.db.commit()
            if n % progress == 0:
                rate = n / (time.time() - started)
                print(f"  {n:,}/{len(todo):,} cached, {rate:.0f}/s; ${billed[0]:.2f} billed")
    finally:
        cache.db.commit()
        pool.shutdown(cancel_futures=True)
    return {i: {q: have.get(keys[i, q]) for q in questions} for i in texts}
