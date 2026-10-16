"""`uv run python -m jev.ask DIR BUDGET [N]`: ask question set v0 about the entities in
`DIR/entities.jsonl` (a seeded sample of N, or all), caching in `DIR/jev.sqlite`, for
at most BUDGET dollars."""

import random
import sys
import time
from pathlib import Path

from . import client, data
from .questions import QUESTIONS


def main() -> None:
    dir, budget = Path(sys.argv[1]), float(sys.argv[2])
    all = data.texts(dir)
    ids = sorted(all)
    if len(sys.argv) > 3:
        ids = random.Random(0).sample(ids, int(sys.argv[3]))
    cache = client.Cache(dir / "jev.sqlite")
    started = time.time()
    got = client.answers({i: all[i] for i in ids}, QUESTIONS, cache, budget)
    print(
        f"{len(got):,} entities answered in {time.time() - started:.0f}s; "
        f"${cache.spent():.2f} in the ledger (kept since P3)"
    )


main()
