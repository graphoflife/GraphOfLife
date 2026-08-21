#!/usr/bin/env python3
"""
record_home_run.py -- record the world that plays behind the front page.

The landing page shows a real run of the engine. It does not compute one: a
hundred iterations of a world this size takes minutes in native Python, and in
the browser it would have to fetch a Python runtime first, so a visitor would
be looking at an empty page for the length of it. On a machine with
gol_server.py running it would be worse than slow — asking the backend for a
world would leave a stray run on disk every time the front page was opened.

So the run happens here, once, and the site ships the result.

    python3 tools/record_home_run.py

The first iterations are computed and thrown away. A world's opening moves are
its least interesting: a ring lattice with the tokens spread evenly over it,
before anything has had the chance to become unlike anything else. What gets
recorded is what the run looks like once it has found its shape.

Only what the renderer reads is kept -- who exists, what they hold, who their
parent was, and who they are joined to. No brains, no messages, no decisions,
no statistics.

Even that is too much written plainly, so it is packed. Agent ids reach seven
digits by the time a run has been going a while, and every edge carries two of
them; as JSON that is about 0.74 MB for a frame of twenty thousand agents. The
same frame here is 0.17 MB, from three things:

  * ids are sorted and stored as gaps between neighbours. The gaps have a
    median of 3, so a seven-digit number becomes one byte.
  * every number is a varint, so small ones cost one byte instead of four.
  * edges are stored as an adjacency list of gaps rather than as pairs, which
    halves the number of indices written and makes each one small.

Storing each frame as a difference from the one before was measured and mostly
abandoned: agents change by 8% per iteration and would compress well, but the
edges churn by 46%, and the edges are most of the file.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GraphOfLifeSimple import new_world           # noqa: E402
from gol_config import SimConfig                  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "web", "data", "home-run.bin")


MAGIC = b"GOLH"
VERSION = 1


def _varint(value: int, out: bytearray) -> None:
    """Seven bits at a time, low first, top bit set while more follow."""
    if value < 0:
        raise ValueError(f"varints are unsigned; got {value}")
    while True:
        chunk = value & 0x7F
        value >>= 7
        out.append(chunk | (0x80 if value else 0))
        if not value:
            return


def _zigzag(value: int) -> int:
    """Signed to unsigned, keeping small magnitudes small either way."""
    return (value << 1) if value >= 0 else ((-value) << 1) - 1


def _pack_frame(ids, tokens, edges, parents, out: bytearray) -> None:
    """One frame, in the format web/js/home.js reads."""
    n = len(ids)
    _varint(n, out)

    previous = 0
    for agent in ids:
        _varint(agent - previous, out)
        previous = agent

    for value in tokens:
        _varint(max(0, value), out)

    # A parent as the step from the child's own position, zigzagged so a
    # parent just before or just after costs one byte. Zero means none, so the
    # encoded value is shifted up by one to leave that spare.
    for i, parent in enumerate(parents):
        _varint(0 if parent < 0 else _zigzag(parent - i) + 1, out)

    # Adjacency, upper triangle only: each edge is written once, from its lower
    # endpoint, as the gap from whatever was written before it.
    for i in range(n):
        neighbours = edges[i]
        _varint(len(neighbours), out)
        previous = i
        for j in neighbours:
            _varint(j - previous, out)
            previous = j


def _unpack(blob: bytes):
    """The decoder, used to check the encoder against itself before writing."""
    at = 0

    def take():
        nonlocal at
        shift = 0
        value = 0
        while True:
            byte = blob[at]
            at += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
            shift += 7

    assert blob[:4] == MAGIC, "not a home recording"
    assert blob[4] == VERSION, "unknown version"
    at = 5
    frames = []
    for _ in range(take()):
        n = take()
        ids, previous = [], 0
        for _ in range(n):
            previous += take()
            ids.append(previous)
        tokens = [take() for _ in range(n)]
        parents = []
        for i in range(n):
            raw = take()
            if raw == 0:
                parents.append(-1)
            else:
                raw -= 1
                delta = (raw >> 1) if raw % 2 == 0 else -((raw + 1) >> 1)
                parents.append(i + delta)
        pairs = []
        for i in range(n):
            count = take()
            previous = i
            for _ in range(count):
                previous += take()
                pairs.append((i, previous))
        frames.append({"ids": ids, "tokens": tokens, "parents": parents, "edges": pairs})
    return frames


def record(tokens_total: int, warmup: int, keep: int, seed: int, out: str) -> None:
    cfg = SimConfig(total_tokens=tokens_total, allow_rewire=False, seed=seed)
    world = new_world(cfg)

    started = time.perf_counter()
    for i in range(warmup):
        world.step()
        if (i + 1) % 10 == 0:
            print(f"  warm-up {i + 1}/{warmup}  "
                  f"{len(world.G):,} agents  {time.perf_counter() - started:.0f}s",
                  flush=True)

    print(f"warm-up done in {time.perf_counter() - started:.0f}s; recording {keep}", flush=True)

    kept = []
    for i in range(keep):
        # One frame per iteration rather than both phases. The pair differ by a
        # single phase, which at a frame a second is a flicker rather than a
        # development, and keeping one halves the file.
        frame = world.step()[-1]
        raw_ids = [int(v) for v in (frame.get("ids") or range(len(frame["tokens"])))]

        # Sorted, because the packing stores ids as the gaps between them and
        # gaps only stay small if they are in order. Everything else is
        # reordered to match.
        order = sorted(range(len(raw_ids)), key=lambda k: raw_ids[k])
        ids = [raw_ids[k] for k in order]
        tokens = [int(frame["tokens"][k]) for k in order]
        at = {agent: k for k, agent in enumerate(ids)}

        parents = [-1] * len(ids)
        if frame.get("parent_ids"):
            for position, k in enumerate(order):
                parent = frame["parent_ids"][k]
                if parent is not None:
                    parents[position] = at.get(int(parent), -1)

        # Upper triangle: every edge written once, from its lower endpoint.
        adjacency = [[] for _ in ids]
        for a, b in frame["edges"]:
            i_a, i_b = at.get(int(a)), at.get(int(b))
            if i_a is None or i_b is None or i_a == i_b:
                continue
            low, high = (i_a, i_b) if i_a < i_b else (i_b, i_a)
            adjacency[low].append(high)
        for neighbours in adjacency:
            neighbours.sort()

        kept.append((ids, tokens, adjacency, parents))
        if (i + 1) % 10 == 0:
            print(f"  recorded {i + 1}/{keep}  {len(ids):,} agents  "
                  f"{sum(len(a) for a in adjacency):,} edges", flush=True)

    blob = bytearray(MAGIC)
    blob.append(VERSION)
    _varint(len(kept), blob)
    for ids, tokens, adjacency, parents in kept:
        _pack_frame(ids, tokens, adjacency, parents, blob)
    blob = bytes(blob)

    # Read it back before writing it out. A packed file that decodes to
    # something other than what went in is a bug nobody would see until the
    # front page looked subtly wrong, so the encoder is checked against its own
    # decoder here rather than trusted.
    restored = _unpack(blob)
    assert len(restored) == len(kept), "frame count changed in packing"
    for (ids, tokens, adjacency, parents), back in zip(kept, restored):
        assert back["ids"] == ids, "ids did not survive packing"
        assert back["tokens"] == tokens, "tokens did not survive packing"
        assert back["parents"] == parents, "parents did not survive packing"
        expected = [(i, j) for i, row in enumerate(adjacency) for j in row]
        assert back["edges"] == expected, "edges did not survive packing"
    print("packed file decodes back to exactly what went in")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as handle:
        handle.write(blob)

    # A companion the page never reads, so what was recorded is written down
    # somewhere a person can look at it.
    with open(os.path.splitext(out)[0] + ".json", "w") as handle:
        json.dump({"recorded": time.strftime("%Y-%m-%d"), "format": "GOLH v1",
                   "config": {"total_tokens": tokens_total, "allow_rewire": False,
                              "seed": seed, "warmup": warmup, "frames": len(kept)}},
                  handle, indent=2)

    agents = [len(f[0]) for f in kept]
    edges = [sum(len(a) for a in f[2]) for f in kept]
    print(f"\nwrote {out}")
    print(f"  {len(kept)} frames, {min(agents):,}-{max(agents):,} agents, "
          f"{min(edges):,}-{max(edges):,} edges")
    print(f"  {len(blob) / 1e6:.2f} MB  ({len(blob) / max(1, len(kept)) / 1e3:.0f} KB a frame)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=100000)
    parser.add_argument("--warmup", type=int, default=50,
                        help="iterations computed and discarded")
    parser.add_argument("--keep", type=int, default=50,
                        help="iterations recorded, played back and forth")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    record(args.tokens, args.warmup, args.keep, args.seed, args.out)


if __name__ == "__main__":
    main()
