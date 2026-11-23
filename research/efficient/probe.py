"""
Brains asked the same questions: what each decides in a fixed set of real situations.

A probe is one real situation: what an agent of a kept world saw of itself and its neighbours (the engine's
own input matrix), and the tokens it held. A brain's behaviour on a probe is what it would decide there:

  child   how many tokens it gives a child (the engine's reading of the two child-share rows, rounded
          down; less than `min_child` is no child)
  first   where its first token goes in the game: the candidate its stake row scores highest (0 = its own
          node). Experiment 48's worlds place their stakes a token at a time, so this is the first move of
          every agent's game, read with the board still empty.
  home    whether that first token stays at home

Two brains are compared by the share of probes on which they decide differently. Nothing here plays a world.

    research/efficient/probe.py is a library: B1, B2 and A1 use it.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GOL_RUNS_DIR", "/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns")
sys.path.insert(0, ROOT)

import gol_store  # noqa: E402
from gol_config import SimConfig  # noqa: E402
from GraphOfLife import GraphOfLife  # noqa: E402

RUNS = {"s4003": "B1-10000-085a32-s4801", "s4002": "B1-10000-f511a5-s4801", "s4008": "B1-10000-1cfaa4-s4801"}


def config(run_id: str) -> SimConfig:
    return SimConfig.from_dict(gol_store.load_meta(run_id)["config"])


def world_at(run_id: str, iteration: int) -> Tuple[Any, Dict[str, np.ndarray], SimConfig]:
    cfg = config(run_id)
    blob = gol_store.load_archive(run_id, iteration)
    return GraphOfLife.from_checkpoint(blob, cfg), blob, cfg


def probes(run_id: str, iteration: int, count: int, seed: int = 1) -> List[Dict[str, Any]]:
    """`count` situations of the kept world: the input matrix of an agent drawn at random, and its tokens."""
    world, _, _ = world_at(run_id, iteration)
    rng = np.random.RandomState(seed)
    world.rng = np.random.RandomState(seed + 1)          # the noise inputs, the same every time
    f = world._precompute_features()
    ids = sorted(u for u in world.G.nodes() if world.tokens.get(u, 0) >= 1)
    out = []
    for u in rng.choice(ids, size=min(count, len(ids)), replace=False):
        u = int(u)
        X = world._inputs(u, [u] + list(f[1][u]), f[0], f[2], f[3], f[4], f[5])
        out.append({"X": X, "tokens": int(world.tokens[u]), "candidates": X.shape[1]})
    return out


def layers(blob: Dict[str, np.ndarray], rows: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """The weights and biases of the agents `rows` of a kept world, one stack a layer."""
    n = int(blob["n_layers"][0])
    return ([blob[f"W{i}"][rows].astype(np.float64) for i in range(n)],
            [blob[f"b{i}"][rows].astype(np.float64) for i in range(n)])


def stack(brains: List[Any]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """The same of brain objects of the engine."""
    n = len(brains[0].weights)
    return ([np.stack([b.weights[i] for b in brains]).astype(np.float64) for i in range(n)],
            [np.stack([b.biases[i] for b in brains]).astype(np.float64) for i in range(n)])


def forward(W: List[np.ndarray], b: List[np.ndarray], X: np.ndarray, activation: str) -> np.ndarray:
    """Every brain of the stack on one input matrix: (brains, outputs, candidates), as Brain.forward computes it."""
    squash = np.tanh if activation == "tanh" else (lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))))
    a = X[None, :, :]
    for i in range(len(W)):
        z = W[i] @ a + b[i]
        a = z if i == len(W) - 1 else squash(z)
    return a


def behaviour(W: List[np.ndarray], b: List[np.ndarray], asked: List[Dict[str, Any]], cfg: SimConfig) -> Dict[str, np.ndarray]:
    """What each brain decides on each probe: arrays (brains, probes) of `child`, `first` and `home`."""
    from GraphOfLife import build_heads
    heads = build_heads(cfg)
    n = W[0].shape[0]
    child = np.zeros((n, len(asked)), dtype=np.int32)
    first = np.zeros((n, len(asked)), dtype=np.int32)
    for j, p in enumerate(asked):
        Y = forward(W, b, p["X"], cfg.hidden_activation)
        frac = Y[:, heads["REPRO_FRACTION"], :].mean(axis=2)                 # (brains, 2)
        va, vb = np.maximum(0.0, frac[:, 0]), np.maximum(0.0, frac[:, 1])
        share = np.where(va + vb > 0, va / np.where(va + vb > 0, va + vb, 1.0), 0.5)
        tokens = np.floor(share * p["tokens"]).astype(np.int32)
        child[:, j] = np.where(tokens >= cfg.min_child, tokens, 0)
        first[:, j] = np.argmax(Y[:, heads["BLOTTO"], :], axis=1)
    return {"child": child, "first": first, "home": (first == 0)}


def differ(a: Dict[str, np.ndarray], i: int, b: Dict[str, np.ndarray], j: int) -> Dict[str, float]:
    """The share of probes on which brain i of a and brain j of b decide differently, by decision and for either."""
    c = a["child"][i] != b["child"][j]
    f = a["first"][i] != b["first"][j]
    return {"child": float(c.mean()), "first": float(f.mean()), "any": float((c | f).mean())}


def mutant(brain: Any, cfg: SimConfig, rng: np.random.RandomState) -> Any:
    """A copy changed once by the engine's own mutation."""
    clone = brain.copy()
    clone.cfg = dataclasses.replace(cfg, mutation_probability=1.0)
    assert clone.mutate(-1, rng)
    clone.cfg = cfg
    return clone


def nudged(brain: Any, cfg: SimConfig, k: int, rng: np.random.RandomState) -> Any:
    """A copy with exactly k of its numbers changed by the engine's noise (its size for that layer), nothing redrawn."""
    clone = brain.copy()
    parts = clone.weights + clone.biases
    sizes = np.array([p.size for p in parts])
    scale = [cfg.mutation_noise_std / np.sqrt(w.shape[1]) for w in clone.weights] * 2
    for which in rng.choice(len(parts), size=k, p=sizes / sizes.sum()):
        p = parts[which]
        at = tuple(int(rng.randint(s)) for s in p.shape)
        p[at] = p.dtype.type(float(p[at]) + rng.normal(0.0, scale[which]))
    return clone
