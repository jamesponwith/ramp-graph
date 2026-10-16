"""TypeSafe System One client with a disk cache.

One request per entity carries every question it lacks. An answer is cached under
(model, question, entity text), so each entity is asked each question once, and a
revised question re-asks only itself.
"""

import hashlib
import json
import os
import random
import sqlite3
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
RETRY = {429, 529, 500, 502, 503, 504}


def key(model: str, question: dict, text: str) -> str:
    q = json.dumps(question, sort_keys=True)
    return hashlib.sha256(f"{model}\0{q}\0{text}".encode()).hexdigest()


def ask(text: str, questions: dict[str, dict], model: str = MODEL, tries: int = 8) -> dict:
    """One call: `{question id: answer}`. Retries 429/529/5xx with jittered backoff."""
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
                return json.load(r)["answers"]
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
        self.db.executemany(
            "insert or replace into a values (?, ?)", ((k, json.dumps(v)) for k, v in items)
        )
        self.db.commit()


def answers(
    texts: dict[int, str],
    questions: dict[str, dict],
    cache: Cache,
    model: str = MODEL,
    workers: int = 8,
    progress: int = 1000,
    asking: bool = True,
) -> dict[int, dict[str, dict | None]]:
    """`{entity id: {question id: answer}}`, asking Jev only for what the cache lacks.
    Not `asking`, what it lacks is `None`."""
    keys = {(i, q): key(model, d, t) for i, t in texts.items() for q, d in questions.items()}
    have = cache.get(list(keys.values()))
    todo: dict[int, list[str]] = {}
    for (i, q), k in keys.items():
        if asking and k not in have:
            todo.setdefault(i, []).append(q)

    def one(item: tuple[int, list[str]]) -> list[tuple[str, dict]]:
        i, qs = item
        got = ask(texts[i], {q: questions[q] for q in qs}, model)
        return [(keys[i, q], got[q]) for q in qs]

    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for n, got in enumerate(pool.map(one, todo.items()), 1):
            cache.put(got)
            have.update(got)
            if n % progress == 0:
                print(f"  {n:,}/{len(todo):,} asked, {n / (time.time() - started):.0f}/s")
    return {i: {q: have.get(keys[i, q]) for q in questions} for i in texts}
