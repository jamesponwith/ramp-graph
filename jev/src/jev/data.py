"""Decision batches (`RAMP_BATCHES=dir` from `benches/icij.rs`) as ranking groups.

One group per policy call: every candidate offered, its oracle label, and the
priority the policy that ran gave it. Features are what a candidate and its
context say in numbers; text comes later (P2).
"""

import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REL = 8
DOM = 8
# Hops and costs for a side that has not reached the candidate.
FAR = 1e3

NAMES = (
    ["domain", "goal"]
    + [f"peek{r}" for r in range(REL)]
    + [f"left{r}" for r in range(REL)]
    + [f"known_dom{d}" for d in range(DOM)]
    + [
        "log_degree",
        "log_left",
        "log_relevant",
        "hops",
        "hops_far",
        "two_sided",
        "cost",
        "cost_far",
        "known",
        "watched",
        "is_watched",
        "alerts",
        "probes",
        "pushed",
        "domain_interest",
        "spent",
        "side_balance",
        "steps",
        "batch",
        "rank_cost",
    ]
)


@dataclass
class Data:
    x: np.ndarray  # features, float32, one row per candidate
    y: np.ndarray  # oracle label, 0/1
    group: np.ndarray  # batch number, contiguous
    prior: np.ndarray  # the generating policy's priority (-inf: not expanded)
    run: list[str]  # per group: "<kind>.<task>.<policy>"
    task: list[str]  # per group: "<kind>.<task>"
    id: np.ndarray  # entity id, one per row


def features(c: dict, ctx: dict, task: dict, size: int) -> list[float]:
    """One candidate's numbers; `rank_cost` is filled per batch afterwards."""
    peek, fetched = c["peek"], c["fetched"]
    left = [max(p - f, 0) for p, f in zip(peek, fetched)]
    interest = task["relation_interest"]
    hops = sorted(FAR if h is None else h for h in c["hops"])
    cost = sorted(FAR if k is None else k for k in c["cost"])
    budget = max(ctx["budget"], 1)
    sides = ctx["by_side"]
    return [
        c["domain"],
        0 if task["goal"] == "connect" else 1,
        *peek,
        *left,
        *c["known_by_domain"],
        math.log1p(sum(peek)),
        math.log1p(sum(left)),
        math.log1p(sum(i * n for i, n in zip(interest, left))),
        hops[0],
        hops[1],
        float(hops[1] < FAR),
        cost[0],
        cost[1],
        c["known"],
        c["watched"],
        float(c["is_watched"]),
        c["alerts"],
        -1 if c["probes"] is None else c["probes"],
        sum(c["pushed"]),
        task["domain_interest"][c["domain"]],
        ctx["fetched"] / budget,
        (sides[0] - sides[1]) / max(sides[0] + sides[1], 1),
        math.log1p(ctx["steps"]),
        math.log1p(size),
        0.0,
    ]


def batch(cands: list[dict], ctx: dict, task: dict) -> list[list[float]]:
    """Feature rows for one policy call."""
    rows = [features(c, ctx, task, len(cands)) for c in cands]
    # The candidate's place in the batch by route cost: Guided's main signal, relative
    # to what it competes with.
    order = np.argsort(np.argsort([r[NAMES.index("cost")] for r in rows]))
    for r, o in zip(rows, order):
        r[-1] = o / max(len(rows) - 1, 1)
    return rows


def priority(d: dict) -> float:
    return d["priority"] if d["take"] else -math.inf


def load(dir: Path, kinds: tuple[str, ...] = ("connect", "alert")) -> Data:
    """Every batch file in `dir` of the given kinds, with `tasks.jsonl`."""
    tasks = read_tasks(dir)
    x, y, group, prior, run, task, ids = [], [], [], [], [], [], []
    for f in sorted(dir.glob("*.jsonl")):
        parts = f.name.split(".")
        if len(parts) < 4 or parts[0] not in kinds:
            continue
        name = f"{parts[0]}.{parts[1]}"
        t = tasks[name]
        for line in f.read_text().splitlines():
            b = json.loads(line)
            cands = b["candidates"]
            g = len(run)
            x += batch(cands, b["context"], t)
            y += [int(c["label"]) for c in cands]
            prior += [priority(d) for d in b["decisions"]]
            group += [g] * len(cands)
            ids += [c["id"] for c in cands]
            run.append(f"{name}.{'.'.join(parts[2:-1])}")
            task.append(name)
    return Data(
        np.asarray(x, np.float32),
        np.asarray(y, np.int8),
        np.asarray(group, np.int64),
        np.asarray(prior, np.float64),
        run,
        task,
        np.asarray(ids, np.int64),
    )


def split(tasks: dict[str, dict], test: float = 0.2, salt: str = "") -> set[str]:
    """Test tasks: whole components of tasks that share a seed, so no seed entity is
    in both sides. Deterministic; another `salt` draws another split."""
    parent = {name: name for name in tasks}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    by_seed: dict[int, str] = {}
    for name, t in tasks.items():
        for s in t["seeds"]:
            if s in by_seed:
                parent[find(name)] = find(by_seed[s])
            else:
                by_seed[s] = name
    cut = int(test * 2**32)
    return {n for n in tasks if zlib.crc32((salt + find(n)).encode()) < cut}


def read_tasks(dir: Path) -> dict[str, dict]:
    return {
        (row := json.loads(line))["run"]: row["task"]
        for line in (dir / "tasks.jsonl").read_text().splitlines()
    }


def texts(dir: Path) -> dict[int, str]:
    """Entity text by id, from `entities.jsonl`."""
    return {
        (e := json.loads(line))["id"]: e["text"]
        for line in (dir / "entities.jsonl").read_text().splitlines()
    }
