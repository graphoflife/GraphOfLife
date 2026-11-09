"""
Round 3's chosen worlds played again (the author, 2026-11-08: "shortlist and
new seeds, but with 100,000-token simulations; more simulations are good than
just eleven"). Writes the lab's plan for it: every chosen world of round 3 a
condition with its own drawn settings, exactly as the sampler drew them (the
allocation too, for the draws played a token at a time), at a new size, with
new seeds, and starting with as many founders a token as it did.

A world of round 3 kept, as a world's settings are kept, only what differs from
B1 (`spec.lab["set"]`), and its founders (`n_nodes`: 0 for one founder per 100
tokens, else a number at 30,000 tokens); at another size the founders are
scaled to keep one founder per as many tokens.

    python3 research/screen/followup.py E43 --worlds s3026 s4002w ... --seeds 4301..4302 --tokens 100000
        writes book/experiments/E43.json (the thesis is written into it by hand, before any run)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import gol_plan  # noqa: E402

#: The rounds a chosen world can come from, and how its name says which (a token at a time: w, s).
ROUNDS = {"": "E40", "w": "E41", "s": "E42"}


def world_of(label: str) -> gol_plan.RunSpec:
    """The run a short name names: s3026 for Experiment 40's, s4002w and s4002s for the draws played a token at a time."""
    m = re.fullmatch(r"s(\d+)\s*([ws]?)", label.strip())
    if not m:
        raise SystemExit(f"{label!r} is not a world's short name (s3026, s4002w, s4002s)")
    seed, tag = int(m.group(1)), m.group(2)
    found = [s for s in gol_plan.experiment_runs(ROUNDS[tag]) if s.lab["seed"] == seed]
    if not found:
        raise SystemExit(f"no world {label} in {ROUNDS[tag]}")
    return found[0]


def condition(spec: gol_plan.RunSpec, tokens: int) -> Dict[str, Any]:
    """One chosen world as a condition: its drawn settings, and its founders a token at the new size."""
    out: Dict[str, Any] = {"name": f"s{spec.lab['seed']}" + {"E41": " weakest first", "E42": " strongest first"}.get(
        spec.experiments[0], ""), "set": dict(spec.lab["set"])}
    founders = int(spec.lab["world"].get("n_nodes") or 0)
    if founders:
        out["world"] = {"n_nodes": round(founders * tokens / int(spec.lab["world"]["total_tokens"]))}
    return out


def plan(name: str, labels: List[str], seeds: str, tokens: int, iterations: int, archive_every: int) -> Dict[str, Any]:
    return {"id": name, "title": "Round 3's best worlds, again: new seeds and more tokens",
            "thesis": {"claim": "", "why": "", "confirm": "", "refute": ""},
            "runs": {"baseline": "B1", "world": {"total_tokens": tokens}, "seeds": seeds, "iterations": iterations,
                     "record": {"heavy_every": 25, "checkpoint_every": 100, "archive_every": archive_every},
                     "conditions": [condition(world_of(label), tokens) for label in labels]}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name")
    ap.add_argument("--worlds", nargs="+", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--tokens", type=int, default=100000)
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--archive-every", type=int, default=250)
    a = ap.parse_args()
    out = plan(a.name, a.worlds, a.seeds, a.tokens, a.iterations, a.archive_every)
    path = os.path.join(ROOT, "book", "experiments", f"{a.name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {path}: {len(out['runs']['conditions'])} worlds, seeds {a.seeds}")


if __name__ == "__main__":
    main()
