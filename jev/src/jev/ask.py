"""`uv run python -m jev.ask DIR [N]`: ask question set v0 about the entities in
`DIR/entities.jsonl` (a seeded sample of N, or all), caching in `DIR/jev.sqlite`.
Then re-asks 100 cached entities fresh and reports how far the answers moved."""

import random
import sys
import time
from pathlib import Path

from . import client, data
from .questions import QUESTIONS, value


def main() -> None:
    dir = Path(sys.argv[1])
    all = data.texts(dir)
    ids = sorted(all)
    if len(sys.argv) > 2:
        ids = random.Random(0).sample(ids, int(sys.argv[2]))
    cache = client.Cache(dir / "jev.sqlite")
    started = time.time()
    got = client.answers({i: all[i] for i in ids}, QUESTIONS, cache)
    print(f"{len(got):,} entities answered in {time.time() - started:.0f}s")
    moved = []
    for i in random.Random(1).sample(ids, min(100, len(ids))):
        fresh = client.ask(all[i], QUESTIONS)
        moved += [abs(value(fresh[q]) - value(got[i][q])) for q in QUESTIONS]
    moved.sort()
    print(
        f"re-asked {len(moved) // len(QUESTIONS)}: |change| max {moved[-1]:.4f}, "
        f"p99 {moved[int(0.99 * (len(moved) - 1))]:.4f}, "
        f"{sum(m == 0 for m in moved) / len(moved):.0%} identical"
    )


main()
