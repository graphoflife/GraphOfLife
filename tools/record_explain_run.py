#!/usr/bin/env python3
"""
record_explain_run.py -- record the small world the Explanation walks through.

The Explanation shows one real run, stepped through a stage at a time: the
observation, the births, the cull, the staking, the conquest, the pruning, the
mutation. A recorded frame is far too coarse for that — it is written once a
phase, so all of those land in the same frame and cannot be told apart. The
engine offers a snapshot at each stage instead, through the `on_step` hook it
leaves switched off; this turns it on and writes down what comes back.

    python3 tools/record_explain_run.py

The window was not picked by hand. Every mechanic has to appear at least once
or the Explanation would be describing something the reader cannot see, and the
world has to stay small enough to follow one agent through it. A search over
seeds and starting points settled on seed 5 from iteration 129, which runs
between 27 and 42 agents and contains, across ten iterations, 59 births, 32
handovers, 95 revolutions, 193 conquests, 169 pruned links and 27 starvations.
Its first iteration also loses a piece to the largest-component rule in both
phases, which is the one thing hardest to catch by chance.

Nothing here belongs to the Explanation in particular. What comes out is a list
of stages, each a graph plus a bag of marks naming who did what, which is the
same shape the Viewer could produce from a live run's own decision records.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GraphOfLifeSimple import new_world           # noqa: E402
from gol_config import SimConfig                  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "web", "data", "explain-run.json")

# Chosen by tools/record_explain_run.py --search; see the note above.
SEED = 5
SKIP = 128            # iterations computed and thrown away
KEEP = 10
TOKENS = 500
AGENTS = 40


def config(seed: int) -> SimConfig:
    return SimConfig(total_tokens=TOKENS, n_nodes=AGENTS, k_neighbors=4,
                     hidden_layers=[24, 16], message_amount=3,
                     random_input_amount=2, seed=seed)


def record(seed: int, skip: int, keep: int, out: str) -> None:
    world = new_world(config(seed))
    for _ in range(skip):
        world.step()

    stages: list = []
    world.on_step = stages.append
    for _ in range(keep):
        world.step()
    world.on_step = None

    # Positions are worked out in the browser by the same force layout the rest
    # of the site uses, so nothing about the geometry is decided here.
    payload = {
        "seed": seed,
        "from_iteration": skip + 1,
        "iterations": keep,
        "tokens": TOKENS,
        "stages": stages,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    order = []
    for stage in stages:
        if stage["step"] not in order:
            order.append(stage["step"])
    sizes = [len(s["ids"]) for s in stages]
    print(f"wrote {out}")
    print(f"  {len(stages)} stages over {keep} iterations, "
          f"{len(order)} distinct: {', '.join(order)}")
    print(f"  {min(sizes)}-{max(sizes)} agents, {os.path.getsize(out) / 1e3:.0f} KB")


def search(first: int, last: int) -> None:
    """Look for a window where every mechanic fires and the world stays small."""
    needed = ("births", "handovers", "revolutions", "conquests", "pruned", "starved")
    for seed in range(first, last + 1):
        world = new_world(config(seed))
        rows = []
        for i in range(150):
            repro, game = world.step(record_decisions=True)
            d1 = repro.get("decisions") or {}
            d2 = game.get("decisions") or {}
            winners = d2.get("winners") or []
            rows.append({
                "it": i + 1, "agents": game["summary"]["nodes"],
                "births": len(d1.get("births") or []),
                "handovers": sum(len(b.get("handed_over") or [])
                                 for b in (d1.get("births") or [])),
                "revolutions": sum(1 for x in winners if x.get("revolt", 0) > 0),
                "conquests": sum(1 for x in winners if x["winner"] != x["node"]),
                "pruned": len(d2.get("pruned_edges") or []),
                "starved": repro["cleanup"]["starved"] + game["cleanup"]["starved"],
                "orphaned": repro["cleanup"]["orphaned"] + game["cleanup"]["orphaned"],
            })
            if game["summary"]["nodes"] < 8:
                break

        for start in range(49, max(50, len(rows) - 10)):
            window = rows[start:start + 10]
            if len(window) < 10:
                continue
            if window[0]["orphaned"] == 0:
                continue
            if any(sum(r[k] for r in window) == 0 for k in needed):
                continue
            sizes = [r["agents"] for r in window]
            if min(sizes) < 14 or max(sizes) > 55:
                continue
            totals = {k: sum(r[k] for r in window) for k in needed}
            print(f"  seed {seed:2d} from {window[0]['it']:4d}  "
                  f"{min(sizes)}-{max(sizes)} agents  {totals}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip", type=int, default=SKIP)
    parser.add_argument("--keep", type=int, default=KEEP)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--search", nargs=2, type=int, metavar=("FIRST", "LAST"),
                        help="look for windows over a range of seeds instead")
    args = parser.parse_args()

    if args.search:
        search(*args.search)
    else:
        record(args.seed, args.skip, args.keep, args.out)


if __name__ == "__main__":
    main()
