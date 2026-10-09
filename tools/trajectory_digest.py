#!/usr/bin/env python3
"""
trajectory_digest.py -- a fingerprint of what an engine does, to compare two.

A change meant to leave every run exactly as it was is checked by running the
old engine and the new one over the same short worlds and comparing what they
recorded. The files cannot be compared directly: gzip writes a timestamp into
every frame and zip into every checkpoint, so two identical histories are
never identical files. The fingerprint is taken of the content instead: every
frame as canonical JSON, then the arrays of the final world's checkpoint,
hashed in order.

    python3 tools/trajectory_digest.py                  # the engine in this checkout
    python3 tools/trajectory_digest.py --engine DIR     # any other

An older engine is one `git archive <commit> | tar -x -C DIR` away. Two engines
agree when every line they print agrees.

`--company` steps each world in turn with a second one in the same process,
the way the server runs simulations side by side, and says whether each still
records what it records alone.

`--stats` fingerprints the statistics as well: every frame's row as a run's
stats.jsonl would hold it (gol_series.frame_stats, heavy on every frame, and
the family count), so a change to the statistics code is caught as surely as
a change to the simulation. Those modules are then loaded from the engine too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Small worlds, one per thing a change could quietly break: each brain kind,
#: every mechanic switched on, the book's baseline, the frozen defaults, the
#: other way of sharing out the dead, and the control that never reads its
#: inputs. Every setting is written out rather than taken from what the form
#: offers, which moves: a fingerprint has to change only when the engine does.
EVERY_MECHANIC = dict(allow_gifting=True, prune_after="reproduction", inactive_window="iteration")
CASES: List[Tuple[str, Callable[[Any], Any], int]] = [
    ("float, every mechanic", lambda C: C(total_tokens=2000, **EVERY_MECHANIC, seed=101), 0),
    ("float16, baseline B1", lambda C: C(total_tokens=2000, brain_kind="float16",
                                         message_amount=30, mutation_probability=0.2,
                                         seed=102), 0),
    ("binary, every mechanic", lambda C: C(total_tokens=2000, brain_kind="binary",
                                           **C.BRAIN_PRESETS["binary"], **EVERY_MECHANIC,
                                           seed=103), 0),
    ("gol-1", lambda C: C(total_tokens=2000, seed=104), 0),
    ("gol-1 by_tokens, no prepass", lambda C: C(total_tokens=2000, redistribution="by_tokens",
                                                message_prepass=False, seed=105), 0),
    ("noise control", lambda C: C(total_tokens=2000, random_decisions=True, seed=106), 0),
    # The book's own worlds, at their size and past their youth's first turns.
    ("float16, B1 at 10,000 tokens", lambda C: C(total_tokens=10000, brain_kind="float16",
                                                 message_amount=30, mutation_probability=0.2,
                                                 seed=1), 60),
]

#: The statistics, which `--stats` also loads from the engine directory.
STATS_MODULES = ("gol_series", "gol_spectral", "gol_lightning", "gol_record")


def load_engine(directory: str, stats: bool = False):
    """Import the engine that lives in `directory`, and make sure it is that one."""
    import importlib
    directory = os.path.abspath(directory)
    sys.dont_write_bytecode = True          # leave an unpacked commit as it was
    sys.path.insert(0, directory)
    names = ("gol_config", "GraphOfLifeSimple") + (STATS_MODULES if stats else ())
    modules = {name: importlib.import_module(name) for name in names}
    for module in modules.values():
        if os.path.dirname(os.path.abspath(module.__file__)) != directory:
            raise SystemExit(f"{module.__name__} came from {module.__file__}, not {directory}")
    return modules["gol_config"].SimConfig, modules["GraphOfLifeSimple"], modules


def frame_hash(frame: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(frame, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def world_hash(world: Any) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(world.to_checkpoint().items()):
        digest.update(f"{key}|{value.dtype.str}|{value.shape}|".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def history(engine: Any, worlds: List[Any], iterations: int,
            keep: bool = False) -> Tuple[List[List[str]], List[List[Dict[str, Any]]]]:
    """Step the worlds in turn and return each one's frame hashes, and the frames if asked."""
    hashes: List[List[str]] = [[] for _ in worlds]
    kept: List[List[Dict[str, Any]]] = [[] for _ in worlds]
    for _ in range(iterations):
        for i, world in enumerate(worlds):
            for frame in world.step(record_decisions=True):
                hashes[i].append(frame_hash(frame))
                if keep:
                    kept[i].append(frame)
    return hashes, kept


def stats_digest(modules: Dict[str, Any], frames: List[Dict[str, Any]]) -> str:
    """Every frame's row as a run's stats.jsonl would hold it, heavy every time."""
    series, record = modules["gol_series"], modules["gol_record"]
    families = series._CladeWindow()
    previous = None
    digest_ = hashlib.sha256()
    for index, frame in enumerate(frames):
        row = series.frame_stats(frame, previous, True)
        row["cladesInWindow"] = families.count(frame, index)
        digest_.update(json.dumps(record._finite(row), sort_keys=True,
                                  separators=(",", ":")).encode())
        previous = frame
    return digest_.hexdigest()[:16]


def digest(frames: List[str], world: Any) -> str:
    return hashlib.sha256(("".join(frames) + world_hash(world)).encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", default=HERE, help="directory holding the engine")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--company", action="store_true",
                        help="also step each world beside a second one in this process")
    parser.add_argument("--stats", action="store_true",
                        help="also fingerprint every frame's statistics")
    args = parser.parse_args()

    Config, engine, modules = load_engine(args.engine, args.stats)
    for name, build, iterations in CASES:
        iterations = iterations or args.iterations
        world = engine.new_world(build(Config))
        hashes, frames = history(engine, [world], iterations, keep=args.stats)
        frames_alone = hashes[0]
        line = f"{digest(frames_alone, world)}  {name:<30} {world.G.number_of_nodes():>5} agents"
        if args.stats:
            line += f"   stats {stats_digest(modules, frames[0])}"

        if args.company:
            world = engine.new_world(build(Config))
            other = engine.new_world(Config(total_tokens=2000, seed=999))
            together = history(engine, [world, other], iterations)[0][0]
            first = next((i for i, (a, b) in enumerate(zip(frames_alone, together)) if a != b), None)
            line += ("   in company: the same" if first is None
                     else f"   in company: differs from frame {first}")
        print(line, flush=True)


if __name__ == "__main__":
    main()
