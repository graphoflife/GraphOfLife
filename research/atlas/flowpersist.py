"""
How long the flow's modules hold together, in lab worlds of every kind.

The modules of one game are those of the map equation (web/js/flowmodules.js,
through tools/flowmodules_cli.js). Two measures, each set beside a null that
keeps the modules' sizes and deals their agents at random:

- **Lives.** Modules followed from game to game as the Research view follows
  them (best overlap first, above its floor of 0.3), against the same frames'
  modules with their agents shuffled (`--shuffle`): the share of modules
  followed for at least 2, 5, 10, 20 and 40 games.
- **Likeness.** No following at all: how much one game's partition says about
  another's, `lag` games later, as normalised mutual information over the
  agents in both, against the same with the later partition's agents dealt at
  random among modules of its sizes. Its chance level is not zero when modules
  are small, which is why the null is measured rather than assumed.

    python3 research/atlas/flowpersist.py [RUN ...]

Writes GraphOfLifeRuns/.pilots/atlas/flowpersist/<run>.json.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "atlas"))

from flowmodules import CLI, GAMES, START, cut  # noqa: E402  the same window as the first pass

OUT = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "atlas", "flowpersist")
LAGS = (1, 2, 5, 10, 20, 40)
LIVES = (2, 5, 10, 20, 40)
SHUFFLES = 3
#: Both baselines and the rules that changed most in how a world plays.
WORLDS = (["B1-10000-s001", "B1-10000-s002", "B1-10000-s003",
           "B2-10000-s001", "B2-10000-s002", "B2-10000-s003",
           "B1-100000-s001", "B1-100000-s002",
           "B1-10000-a8525d-s001", "B1-10000-a8525d-s002",
           "B2-10000-bad270-s001", "B2-10000-bad270-s002",
           "B2-10000-bf74a9-s001", "B2-10000-fe3009-s001",
           "B2-5000-s001", "B2-5000-s002", "B2-5000-71fc89-s002", "B2-5000-71fc89-s003"])


def frames_of(run: str) -> Tuple[int, List[Dict[str, Any]]]:
    """The game frames of the first pass's window: from 950, or the last GAMES of a shorter run."""
    import book_data as D
    last = int(json.load(open(os.path.join(ROOT, "GraphOfLifeRuns", run, "meta.json"))).get("iteration", 0))
    start = START if last >= START + GAMES else max(0, last - GAMES)
    frames = []
    for t in range(start, start + GAMES):
        try:
            frames.append(cut(D.frame_at(run, t, 2)))
        except Exception:
            break
    return start, frames


def cli(frames: List[Dict[str, Any]], args: List[str]) -> Dict[str, Any]:
    text = "\n".join(json.dumps(f, separators=(",", ":")) for f in frames)
    done = subprocess.run(["node", CLI] + args, input=text, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"flowmodules_cli.js {' '.join(args)}: {done.stderr.strip()[-600:]}")
    return json.loads(done.stdout)


def lived(lives: Dict[str, int], at_least: int) -> float:
    total = sum(lives.values())
    return sum(n for life, n in lives.items() if int(life) >= at_least) / total if total else float("nan")


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised mutual information of two labellings of the same agents: 2 I / (H_a + H_b)."""
    n = len(a)
    if n == 0:
        return float("nan")
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    joint = np.unique(ia.astype(np.int64) * (ib.max() + 1) + ib, return_counts=True)[1] / n
    pa = np.bincount(ia) / n
    pb = np.bincount(ib) / n
    h = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())
    ha, hb, hab = h(pa), h(pb), h(joint)
    return 2.0 * (ha + hb - hab) / (ha + hb) if ha + hb > 0 else 1.0


def likeness(parts: List[Dict[str, Any]], rng: np.random.RandomState) -> Dict[str, Dict[str, float]]:
    """Mean likeness of partitions lag games apart, and the same with the later one dealt at random."""
    out = {}
    for lag in LAGS:
        real, null = [], []
        for t in range(len(parts) - lag):
            a, b = parts[t], parts[t + lag]
            where_b = {agent: i for i, agent in enumerate(b["ids"])}
            pairs = [(la, b["module"][where_b[agent]]) for agent, la in zip(a["ids"], a["module"])
                     if la >= 0 and agent in where_b and b["module"][where_b[agent]] >= 0]
            if len(pairs) < 2:
                continue
            la = np.array([p[0] for p in pairs])
            lb = np.array([p[1] for p in pairs])
            real.append(nmi(la, lb))
            null.append(float(np.mean([nmi(la, rng.permutation(lb)) for _ in range(SHUFFLES)])))
        if real:
            out[str(lag)] = {"real": float(np.mean(real)), "null": float(np.mean(null)), "pairs": len(real)}
    return out


def measure(run: str) -> Dict[str, Any]:
    start, frames = frames_of(run)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "partitions.jsonl")
        real = cli(frames, ["--partitions", path])
        parts = [json.loads(line) for line in open(path) if line.strip()]
    dealt = cli(frames, ["--shuffle", "1"])
    result = {
        "id": run, "family": re.sub(r"-s\d+$", "", run), "from": start, "games": len(frames),
        "agents": float(np.mean(real["agentsPerFrame"])) if real["agentsPerFrame"] else None,
        "modules": real["summary"]["meanModules"] if real["summary"] else None,
        "compression": real["summary"]["compression"] if real["summary"] else None,
        "mean_size": (sum(int(k) * v for k, v in real["sizes"].items()) / sum(real["sizes"].values())
                      if real["sizes"] else None),
        "lives": {str(L): {"real": lived(real["lives"], L), "shuffled": lived(dealt["lives"], L)} for L in LIVES},
        "likeness": likeness(parts, np.random.RandomState(1)),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, f"{run}.json"), "w"), indent=1)
    return result


def main():
    runs = sys.argv[1:] or WORLDS
    for run in runs:
        if os.path.exists(os.path.join(OUT, f"{run}.json")):
            continue
        try:
            r = measure(run)
        except Exception as e:  # noqa: BLE001 - one world that cannot be read stops no other
            print(f"{run}: not measured ({type(e).__name__}: {e})", flush=True)
            continue
        like = r["likeness"]
        print(f"{run}: {r['modules']:.0f} modules of {r['mean_size']:.1f}; lives >= 10 "
              f"{r['lives']['10']['real']:.3f} (shuffled {r['lives']['10']['shuffled']:.3f}); likeness lag 1 "
              f"{like.get('1', {}).get('real', float('nan')):.2f} (null {like.get('1', {}).get('null', float('nan')):.2f}), "
              f"lag 40 {like.get('40', {}).get('real', float('nan')):.2f} "
              f"(null {like.get('40', {}).get('null', float('nan')):.2f})", flush=True)
    print("flow persistence measured", flush=True)


if __name__ == "__main__":
    main()
