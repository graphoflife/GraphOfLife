"""
How evenly every world already read shares its tokens: the Gini coefficient and the richest
tenth's share (research/atlas/profile.py `evenness`), added to each row of the readings that
at.py and profile.py wrote, from the very checkpoint the row was read from.

The author (2026-10-19): the share of agents holding three tokens or fewer only stood in for a
world too even — almost everyone holding a few. These two say it directly, draw no line and do
not grow with the world. Only the tokens are read, so this is quick; a row whose checkpoint has
since moved on (a run continued past the reading) is left without them and said so.

    python3 research/atlas/evenness.py            fills at1000/at2000/at10000/profile.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import profile as P  # noqa: E402  (research/atlas/profile.py)

READINGS = ("at1000", "at2000", "at10000", "profile")


def checkpoint(row) -> str:
    """The file a row was read from: a pilot's ckpt_<at>, a lab run's archive at <at> or its last checkpoint."""
    at = int(row["at"])
    if "/" in row["id"]:
        return os.path.join(P.PILOTS, row["id"], f"ckpt_{at:06d}.npz")
    archived = os.path.join(P.RUNS, row["id"], "archive", f"world_{at:06d}.npz")
    return archived if os.path.exists(archived) else os.path.join(P.RUNS, row["id"], "checkpoint.npz")


def fill(name: str) -> None:
    path = os.path.join(P.PILOTS, "atlas", f"{name}.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)
    filled, missing = 0, []
    for row in data["worlds"]:
        if row.get("dead") or "error" in row or "at" not in row:
            continue
        try:
            with np.load(checkpoint(row), allow_pickle=False) as z:
                if int(z["counters"][2]) != int(row["at"]):
                    missing.append(row["id"])
                    continue
                row.update(P.evenness(z["tokens"]))
                filled += 1
        except (OSError, KeyError, ValueError):
            missing.append(row["id"])
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, indent=1)
    os.replace(path + ".tmp", path)
    print(f"{name}: {filled} worlds filled, {len(missing)} not"
          + (f" ({', '.join(missing[:6])}{' …' if len(missing) > 6 else ''})" if missing else ""), flush=True)


if __name__ == "__main__":
    for reading in READINGS:
        fill(reading)
