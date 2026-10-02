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
CASES: List[Tuple[str, Callable[[Any], Any]]] = [
    ("float, every mechanic", lambda C: C(total_tokens=2000, **EVERY_MECHANIC, seed=101)),
    ("float16, baseline B1", lambda C: C(total_tokens=2000, brain_kind="float16",
                                         message_amount=30, mutation_probability=0.2,
                                         seed=102)),
    ("binary, every mechanic", lambda C: C(total_tokens=2000, brain_kind="binary",
                                           **C.BRAIN_PRESETS["binary"], **EVERY_MECHANIC,
                                           seed=103)),
    ("gol-1", lambda C: C(total_tokens=2000, seed=104)),
    ("gol-1 by_tokens, no prepass", lambda C: C(total_tokens=2000, redistribution="by_tokens",
                                                message_prepass=False, seed=105)),
    ("noise control", lambda C: C(total_tokens=2000, random_decisions=True, seed=106)),
]


def load_engine(directory: str):
    """Import the engine that lives in `directory`, and make sure it is that one."""
    directory = os.path.abspath(directory)
    sys.dont_write_bytecode = True          # leave an unpacked commit as it was
    sys.path.insert(0, directory)
    import gol_config                       # noqa: E402
    import GraphOfLifeSimple                # noqa: E402
    for module in (gol_config, GraphOfLifeSimple):
        if os.path.dirname(os.path.abspath(module.__file__)) != directory:
            raise SystemExit(f"{module.__name__} came from {module.__file__}, not {directory}")
    return gol_config.SimConfig, GraphOfLifeSimple


def frame_hash(frame: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(frame, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def world_hash(world: Any) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(world.to_checkpoint().items()):
        digest.update(f"{key}|{value.dtype.str}|{value.shape}|".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def history(engine: Any, worlds: List[Any], iterations: int) -> List[List[str]]:
    """Step the worlds in turn and return each one's frame hashes."""
    hashes: List[List[str]] = [[] for _ in worlds]
    for _ in range(iterations):
        for i, world in enumerate(worlds):
            hashes[i].extend(frame_hash(f) for f in world.step(record_decisions=True))
    return hashes


def digest(frames: List[str], world: Any) -> str:
    return hashlib.sha256(("".join(frames) + world_hash(world)).encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", default=HERE, help="directory holding the engine")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--company", action="store_true",
                        help="also step each world beside a second one in this process")
    args = parser.parse_args()

    Config, engine = load_engine(args.engine)
    for name, build in CASES:
        world = engine.new_world(build(Config))
        frames = history(engine, [world], args.iterations)[0]
        line = f"{digest(frames, world)}  {name:<30} {world.G.number_of_nodes():>5} agents"

        if args.company:
            alone = frames
            world = engine.new_world(build(Config))
            other = engine.new_world(Config(total_tokens=2000, seed=999))
            together = history(engine, [world, other], args.iterations)[0]
            first = next((i for i, (a, b) in enumerate(zip(alone, together)) if a != b), None)
            line += ("   in company: the same" if first is None
                     else f"   in company: differs from frame {first}")
        print(line, flush=True)


if __name__ == "__main__":
    main()
