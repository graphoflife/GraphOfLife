# -*- coding: utf-8 -*-
"""
How big should a brain be? Round 3 of the random screen (Experiments 40 to 42)
drew every world's brain at random: three, four or five hidden layers of 16, 32
or 50 tanh units, nine shapes in all. This page reads those worlds again for
what the shape did. No world is played.

Two kinds of reading:

  - a probe of the brains themselves (`probe`, kept per run): in each living
    world, at its end, what a sample of agents see is rebuilt exactly as the
    engine builds it, and shown to the agents' own brains, to brains of every
    earlier kept world (one each 250 iterations, from the founders on), to
    fresh founders' brains, and to founders' brains taken to mutation's balance
    (`balance`, `drifted`: what mutation alone would make of them). How much of
    a small change in the inputs reaches the outputs is the brain's gain (What
    the brains are like); whether the stakes move when the neighbours look
    richer, the noise is drawn again or the messages fall silent says whether
    the decisions follow; and one change of a brain is set against a
    stranger's brain;
  - what the screen already measured of every world (research/screen/health.json
    and effects.json: whether it died, its health, how it evolved), compared
    across depths and widths by shuffling, with the settings already known to
    matter for each outcome taken into account (`shape_test`).

A world of ten thousand agents keeps ten thousand brains, and a probe of a few
hundred needs none of the others: a kept world is rebuilt here with every
agent's tokens, links and messages, and the brains of the sampled agents only
(`kept_world`).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import book_data as D
import book_figures as F
import gol_store as store
from book_chapters.common import BLUE, CYAN, GREEN, GREY, ORANGE, VIOLET, YELLOW
from book_chapters.inner import MESSAGE_BLOCKS, _decide, rows_of
from book_figures import chapter, describe, dots, line, recipe

#: Agents probed per world, and the size of the small change every input is moved by
#: (as on What the brains are like, so the gains compare).
AGENTS = 300
NUDGE = 0.1
#: Seeds of the probe's random choices: the agents, the nudges, the fresh noise, the founders,
#: the brains of each kept world and the changes of a mutation.
SEED_AGENTS, SEED_NUDGE, SEED_NOISE, SEED_FOUNDERS, SEED_ERA, SEED_MUTATION = 53, 5301, 5302, 5303, 5304, 5305


# ---------------------------------------------------------------------------
# A kept world, with the brains of a few of its agents
# ---------------------------------------------------------------------------

class _Unread:
    """
    Stands in for a stack of every agent's weights that is not read: it has the
    stack's shape, which the engine checks, and hands each agent an empty array.
    """

    def __init__(self, shape: Tuple[int, ...], dtype: Any) -> None:
        self.shape = shape
        self.dtype = dtype

    def __getitem__(self, i: int) -> np.ndarray:
        return np.empty((0, 0), dtype=self.dtype)


def _is_layer(key: str) -> bool:
    return key[0] in "Wb" and key[1:].isdigit()


def _sample(tokens: np.ndarray, count: int, seed: int) -> np.ndarray:
    """`count` agents holding a token or more, drawn at random (all of them if there are fewer)."""
    holding = np.flatnonzero(tokens > 0)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(holding, min(count, holding.size), replace=False))


def _layers(z: Any, picked: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """The weights and biases of the picked agents only, one layer at a time."""
    n = int(z["n_layers"][0])
    return [z[f"W{i}"][picked] for i in range(n)], [z[f"b{i}"][picked] for i in range(n)]


def kept_world(run_id: str, path: str, count: int, seed: int):
    """
    The world kept in `path` (a run's checkpoint or one of its archive), rebuilt
    with every agent's tokens, links and messages, and with the brains of
    `count` agents holding a token or more, drawn at random; every other brain
    is left empty. Returns the world and the ids of the agents drawn.
    """
    from GraphOfLifeSimple import GraphOfLife
    cfg = store.load_config(run_id)
    with np.load(path) as z:
        blob = {k: z[k] for k in z.files if not _is_layer(k)}
        picked = _sample(blob["tokens"], count, seed)
        weights, biases = _layers(z, picked)
        n = len(blob["ids"])
        for i, (w, b) in enumerate(zip(weights, biases)):
            blob[f"W{i}"] = _Unread((n,) + w.shape[1:], w.dtype)
            blob[f"b{i}"] = _Unread((n,) + b.shape[1:], b.dtype)
    world = GraphOfLife.from_checkpoint(blob, cfg)
    ids = [int(blob["ids"][i]) for i in picked]
    for j, u in enumerate(ids):
        world.brains[u].weights = [w[j] for w in weights]
        world.brains[u].biases = [b[j] for b in biases]
    return world, ids


def kept_brains(run_id: str, iteration: int, count: int, seed: int) -> List[Any]:
    """
    The brains of `count` agents of the world kept at `iteration`, drawn at random
    among those holding a token or more — again, with repeats, where fewer hold one.
    """
    from GraphOfLifeSimple import make_brain
    cfg = store.load_config(run_id)
    with np.load(store.archive_path(run_id, iteration)) as z:
        picked = _sample(z["tokens"], count, seed)
        weights, biases = _layers(z, picked)
    order = np.resize(np.arange(len(picked)), count)
    brains = []
    for j in order:
        brain = make_brain(cfg, 0)
        brain.weights = [w[j] for w in weights]
        brain.biases = [b[j] for b in biases]
        brains.append(brain)
    return brains


# ---------------------------------------------------------------------------
# What a brain does with a small change
# ---------------------------------------------------------------------------

def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def _squash(cfg):
    from GraphOfLifeSimple import _sigmoid
    return np.tanh if cfg.hidden_activation == "tanh" else _sigmoid


def through_layers(brain: Any, X: np.ndarray, eps: np.ndarray, squash) -> Tuple[List[float], List[float]]:
    """
    A small change followed layer by layer: after each layer, the root-mean-square
    change over the root-mean-square change before it; and how large each layer's
    sums are (their root mean square), which says how far into the flat ends of
    the squashing they reach.
    """
    a, b = X.astype(float), (X + eps).astype(float)
    ratios, sums = [], []
    last = len(brain.weights) - 1
    for i, (W, bias) in enumerate(zip(brain.weights, brain.biases)):
        za = W.astype(float) @ a + bias.astype(float)
        zb = W.astype(float) @ b + bias.astype(float)
        a2, b2 = (za, zb) if i == last else (squash(za), squash(zb))
        ratios.append(_rms(b2 - a2) / max(1e-300, _rms(b - a)))
        sums.append(_rms(za))
        a, b = a2, b2
    return ratios, sums


def weight_scale(brain: Any) -> List[float]:
    """Each layer's weights as a multiple of a founder's: the root mean square times √(fan-in), 1 at birth."""
    return [_rms(W.astype(float)) * math.sqrt(W.shape[1]) for W in brain.weights]


def balance(cfg: Any) -> Tuple[float, float]:
    """
    Where mutation alone would bring a brain's numbers after very many changes,
    with nothing selected (How a brain changes): each change jitters a share s
    of the numbers by a normal amount of standard deviation σ/√k, and with
    chance s redraws a share s of a matrix from the founders' 1/√k (a bias from
    σ/√k). The variance v of a weight stops moving where the two balance,
        v = (1 − s²)(v + s σ²/k) + s²/k,  so  v = (1 + (1 − s²) σ²/s) / k,
    and that of a bias at σ²/k · (1 + (1 − s²)/s). Returned as multiples: the
    weights' scale over a founder's, √(v k), and the biases' σ² (1 + (1 − s²)/s)
    to be divided by k.
    """
    s, sigma = float(cfg.mutation_sparsity), float(cfg.mutation_noise_std)
    if s <= 0:
        return 1.0, 0.0
    return math.sqrt(1 + (1 - s * s) * sigma ** 2 / s), sigma ** 2 * (1 + (1 - s * s) / s)


def drifted(cfg: Any, rng: np.random.RandomState) -> Any:
    """A founder's brain taken to mutation's balance: its weights scaled up, its biases drawn at their spread."""
    from GraphOfLifeSimple import make_brain
    brain = make_brain(cfg, 0, rng)
    scale, bias = balance(cfg)
    for i, (W, b) in enumerate(zip(brain.weights, brain.biases)):
        brain.weights[i] = (W.astype(float) * scale).astype(brain.dtype)
        brain.biases[i] = rng.normal(0.0, math.sqrt(bias / W.shape[1]), size=b.shape).astype(brain.dtype)
    return brain


def balance_check(settings: Sequence[Tuple[float, float]], brains: int = 8,
                  seed: int = SEED_MUTATION) -> Dict[str, Dict[str, float]]:
    """
    The balance worked out above, against the engine's own changes: for each (σ, s), founders' brains of
    three layers of 16 changed again and again, with nothing selected, for eight times as many changes as
    the balance takes to settle (8/s²); their weights' size as a multiple of a founder's, and the balance's.
    No world is played: a brain is only changed.
    """
    from gol_config import SimConfig
    from GraphOfLifeSimple import make_brain
    out = {}
    for sigma, s in settings:
        cfg = SimConfig(hidden_layers=[16, 16, 16], hidden_activation="tanh", mutation_probability=1.0,
                        mutation_noise_std=sigma, mutation_sparsity=s)
        rng = np.random.RandomState(seed)
        sizes = []
        for _ in range(brains):
            brain = make_brain(cfg, 0, rng)
            for i in range(int(8 / (s * s))):
                brain.mutate(i + 1, rng)
            sizes.append(float(np.mean(weight_scale(brain))))
        out[f"sigma {sigma}, s {s}"] = {"changed": float(np.mean(sizes)), "balance": balance(cfg)[0]}
    return out


def mutated(brain: Any, cfg: Any, rng: np.random.RandomState) -> Any:
    """A copy of the brain after one change, as How a brain changes makes it (the change made for sure)."""
    copy = brain.copy()
    copy.cfg = dataclasses.replace(cfg, mutation_probability=1.0)
    copy.mutate(1, rng)
    return copy


def _variants(world: Any, X: np.ndarray, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    The inputs with one thing about them different: every neighbour twice as rich
    as the agent sees it; the random numbers drawn again; every message read set
    to 0. A world without noise or messages has no such variant.
    """
    cfg = world.cfg
    out = {}
    its = rows_of(cfg, ("its_tokens",)).start
    richer = X.copy()
    richer[its, 1:] = np.log1p(2.0 * np.expm1(richer[its, 1:]))
    out["richer"] = richer
    if cfg.random_input_amount > 0:
        fresh = X.copy()
        noise = rows_of(cfg, ("noise",))
        fresh[noise] = rng.uniform(-2.0, 2.0, fresh[noise].shape)
        out["noise"] = fresh
    if cfg.message_amount > 0:
        silent = X.copy()
        silent[rows_of(cfg, MESSAGE_BLOCKS)] = 0.0
        out["messages"] = silent
    return out


def _response(brain: Any, heads: Dict[str, Any], X: np.ndarray, eps: np.ndarray, variants: Dict[str, np.ndarray],
              tokens: int, squash) -> Dict[str, Any]:
    """Everything the probe asks of one brain on one agent's inputs."""
    Y, Ye = brain.forward(X), brain.forward(X + eps)
    d, e = Ye - Y, _rms(eps)
    decide = heads["MESSAGE_START"]
    ratios, sums = through_layers(brain, X, eps, squash)
    positive, spread, alloc, share, _ = _decide(brain, heads, X, tokens)
    out = {"gain": _rms(d) / e, "gain_decide": _rms(d[:decide]) / e, "gain_score": _rms(d[heads["BLOTTO"]]) / e,
           "layers": ratios, "sums": sums, "scale": weight_scale(brain), "even": positive == 0,
           "spread": spread, "child": math.floor(share * tokens) >= 1, "home": float(alloc[0])}
    for name, Z in variants.items():
        _, _, moved, _, _ = _decide(brain, heads, Z, tokens)
        out[f"tv_{name}"] = 0.5 * float(np.abs(alloc - moved).sum())
    return out


def _stake_change(brain_a: Any, brain_b: Any, heads: Dict[str, Any], X: np.ndarray, tokens: int) -> Tuple[float, float]:
    """How differently two brains decide on the same inputs: their outputs' root-mean-square difference
    in the rows that make decisions, and the share of the stake placed elsewhere."""
    _, _, a, _, _ = _decide(brain_a, heads, X, tokens)
    _, _, b, _, _ = _decide(brain_b, heads, X, tokens)
    decide = heads["MESSAGE_START"]
    outputs = _rms(brain_a.forward(X)[:decide] - brain_b.forward(X)[:decide])
    return outputs, 0.5 * float(np.abs(a - b).sum())


def _summary(rows: List[Dict[str, Any]], variants: Sequence[str]) -> Dict[str, Any]:
    """One kept world's brains, summed up over the agents probed."""
    def col(key):
        return np.array([r[key] for r in rows], dtype=float)
    out: Dict[str, Any] = {}
    for key in ("gain", "gain_decide", "gain_score"):
        q1, q2, q3 = np.quantile(col(key), [0.25, 0.5, 0.75])
        out[key], out[f"{key}_quartiles"] = float(q2), [float(q1), float(q3)]
    out.update(layers=np.median([r["layers"] for r in rows], axis=0).tolist(),
               sums=np.median([r["sums"] for r in rows], axis=0).tolist(),
               scale=np.median([r["scale"] for r in rows], axis=0).tolist(),
               even=float(col("even").mean()), spread=float(col("spread").mean()),
               child=float(col("child").mean()), home=float(col("home").mean()))
    for name in variants:
        tv = col(f"tv_{name}")
        out[f"changed_{name}"] = float((tv > 0).mean())
        out[f"moved_{name}"] = float(tv.mean())
    return out


@D.measure("brainsize")
def probe(run_id: str) -> Dict[str, Any]:
    """
    How strongly a world's brains respond, at its end: for AGENTS agents holding a
    token or more, their inputs exactly as the engine builds them, and on those
    inputs their own brains, the brains of every world kept along the way (drawn
    at random, one per agent probed), and fresh founders' brains. Also how much
    one change of a brain moves its decisions, against a stranger's brain.
    """
    from GraphOfLifeSimple import brain_shape, make_brain
    cfg = store.load_config(run_id)
    world, picked = kept_world(run_id, store.checkpoint_path(run_id), AGENTS, SEED_AGENTS)
    log_deg, neighs, q_tok, q_deg, log_tok, at_risk = world._precompute_features()
    heads, squash = world.heads, _squash(cfg)
    nudges, fresh = np.random.default_rng(SEED_NUDGE), np.random.default_rng(SEED_NOISE)
    seen = []
    for u in picked:
        targets = [u] + list(neighs[u])
        X = world._inputs(u, targets, log_deg, q_tok, q_deg, log_tok, at_risk)
        seen.append((X, nudges.normal(0.0, NUDGE, X.shape), _variants(world, X, fresh), int(world.tokens[u])))
    variants = sorted(seen[0][2]) if seen else []

    def respond(brains: Sequence[Any]) -> Dict[str, Any]:
        return _summary([_response(b, heads, X, eps, var, tokens, squash)
                         for b, (X, eps, var, tokens) in zip(brains, seen)], variants)

    def change(brains: Sequence[Any], rng: np.random.RandomState) -> Dict[str, float]:
        """
        One change of each brain, and a stranger's brain, on the same inputs: the stranger is the agent
        half the sample away, in the order of their ids, so that it was born far from it in time.
        """
        mut, other = [], []
        for j, (brain, (X, _, _, tokens)) in enumerate(zip(brains, seen)):
            mut.append(_stake_change(brain, mutated(brain, cfg, rng), heads, X, tokens))
            other.append(_stake_change(brain, brains[(j + len(brains) // 2) % len(brains)], heads, X, tokens))
        mut, other = np.array(mut), np.array(other)
        return {"mutation_outputs": float(np.median(mut[:, 0])), "stranger_outputs": float(np.median(other[:, 0])),
                "mutation_stake": float(mut[:, 1].mean()), "stranger_stake": float(other[:, 1].mean())}

    own = [world.brains[u] for u in picked]
    born = np.random.RandomState(SEED_FOUNDERS)
    founders = [make_brain(cfg, 0, born) for _ in picked]
    drift = [drifted(cfg, born) for _ in picked]
    eras = [t for t in store.archived(run_id) if t < world.iteration]
    out = {"iteration": int(world.iteration), "agents": len(world.tokens), "probed": len(picked),
           "shape": list(cfg.hidden_layers), "activation": cfg.hidden_activation, "inputs": cfg.n_inputs(),
           "outputs": cfg.n_outputs(), "weights": int(brain_shape(cfg)["weights"]), "variants": variants,
           "balance": balance(cfg)[0], "evolved": respond(own), "founders": respond(founders),
           "drift": respond(drift), "eras": {}}
    changes = np.random.RandomState(SEED_MUTATION)
    for who, brains in (("evolved", own), ("founders", founders), ("drift", drift)):
        out[who].update(change(brains, changes))
    for t in eras:
        out["eras"][str(t)] = respond(kept_brains(run_id, t, len(picked), SEED_ERA + t))
    return out


# ---------------------------------------------------------------------------
# Round 3's worlds, by the shape of their brains
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEALTH = os.path.join(ROOT, "research", "screen", "health.json")
EFFECTS = os.path.join(ROOT, "research", "screen", "effects.json")
MINDS = os.path.join(ROOT, "book", "results", "minds.json")
ROUNDS = ("E40", "E41", "E42")
DEPTHS, WIDTHS = (3, 4, 5), (16, 32, 50)
#: Shuffles per test, and their seed.
SHUFFLES, SEED_TESTS = 20000, 53
#: A setting is known to matter for an outcome when its own shuffling test there (How the drawn settings
#: shape a world) had a chance below this; a test of the brain's shape takes such settings into account.
KNOWN = 0.05
#: A finding: a false discovery rate below this, as on How the drawn settings shape a world.
FINDING = 0.1
#: What the screen measured of every world, in the order the page shows them: (key, what it is).
OUTCOMES = [("died", "died"), ("health", "health"), ("c_decentralised", "decentralised"),
            ("c_resilient", "resilient"), ("c_space", "almost three-dimensional"), ("c_diverse", "diverse"),
            ("c_uneven", "not too even"), ("c_culls", "few tokens culled"), ("heredity", "heredity index"),
            ("generations", "generations in 3,000 iterations"),
            ("population", "effective population (log10)"), ("novelty", "late novelty of the line of descent"),
            ("fertility", "fertility passed on")]

RUNS = ("Round 3 of the random screen: Experiment 40's 84 worlds, each a draw of its own, and the 16 of "
        "Experiments 41 and 42, eight draws played twice with the stakes placed a token at a time "
        "(30,000 tokens, 3,000 iterations; [A screen of random worlds](a-screen-of-random-worlds.md))")
DATA_PROBE = ("each living world's checkpoint, `GraphOfLifeRuns/<run>/checkpoint.npz` (the world after its last "
              "iteration), and its archive, `GraphOfLifeRuns/<run>/archive/world_<iteration>.npz` (the whole world "
              "every 250 iterations, from the founders on), read by `book_chapters.brainsize.kept_world` with the "
              "brains of the agents probed only")
DATA_SCREEN = ("each world's drawn settings (its plan, `book/experiments/E40.json`, drawn by `gol_screen.py`) and "
               "what the screen measured of it: whether it died, and for the living its health "
               "(`research/screen/health.json`) and how it evolved (`research/screen/effects.json`, from each "
               "world's dossier)")
PROBE_STEP = (f"For {AGENTS} agents drawn at random among those holding a token or more (seed {SEED_AGENTS}), their "
              "inputs exactly as the engine builds them for a look at their candidates "
              "(`World._precompute_features`, `World._inputs`).")


def _shape(depth: int, width: int) -> str:
    return f"{depth} × {width}"


def round_three() -> List[Dict[str, Any]]:
    """Every world of round 3: its run, experiment, settings and brain, whether it lived, and what was measured."""
    import gol_plan
    health = {w["id"]: w for w in json.load(open(HEALTH))["worlds"]}
    measured = {w["id"]: w for w in json.load(open(EFFECTS))["worlds"]}
    out = []
    for experiment in ROUNDS:
        for spec in gol_plan.experiment_runs(experiment):
            h, cfg = health[spec.run_id], spec.config
            out.append({**measured.get(spec.run_id, {}), "id": spec.run_id, "experiment": experiment,
                        "seed": int(spec.lab["seed"]), "alive": bool(h["alive"]), "config": cfg,
                        "died": not h["alive"], "depth": len(cfg["hidden_layers"]),
                        "width": int(cfg["hidden_layers"][0]),
                        "health": h["health"]["product"] if h.get("health") else None,
                        "mean_agents": (h.get("averages") or {}).get("agents")})
    return out


def _ranks(x: Sequence[float]) -> np.ndarray:
    """Ranks from 1, ties sharing the mean of theirs."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    r = np.empty(x.size)
    r[order] = np.arange(1, x.size + 1)
    for v in np.unique(x):
        tie = x == v
        if tie.sum() > 1:
            r[tie] = r[tie].mean()
    return r


def _indicators(values: Sequence[Any]) -> np.ndarray:
    """A setting's values as columns: one per value but the first, 1 where a world has it."""
    labels = [json.dumps(v) for v in values]
    levels = sorted(set(labels))[1:]
    return np.array([[label == level for level in levels] for label in labels], float).reshape(len(labels), -1)


def shape_test(y: Sequence[float], factor: Sequence[float], others: Sequence[Sequence[Any]],
               rng: np.random.RandomState) -> Tuple[float, float]:
    """
    Whether an outcome rises or falls with the brain's depth (or width), beyond chance, with other settings
    taken into account: the outcome's ranks, less what a least-squares fit on those settings' values (as
    columns of 0 and 1) accounts for, correlated with the ranks of the factor. The chance is the share of
    SHUFFLES shuffles of the factor among the worlds with a correlation at least as far from 0. The shape was
    drawn apart from every other setting, so any shuffle is a draw that could have happened.
    """
    r = _ranks(y)
    C = np.column_stack([np.ones(r.size)] + [_indicators(v) for v in others])
    coef, *_ = np.linalg.lstsq(C, r, rcond=None)
    rest = r - C @ coef
    rest -= rest.mean()
    f = _ranks(factor)
    f -= f.mean()
    norm = math.sqrt(float(rest @ rest) * float(f @ f))
    if norm == 0:
        return 0.0, 1.0
    observed = float(rest @ f) / norm
    null = f[np.argsort(rng.rand(SHUFFLES, f.size), axis=1)] @ rest / norm
    return observed, float((1 + (np.abs(null) >= abs(observed) - 1e-12).sum()) / (1 + SHUFFLES))


def false_discovery(chances: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg: the false discovery rate at which each chance would be called a finding."""
    m = len(chances)
    order = np.argsort(chances)
    q, running = np.empty(m), 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, chances[order[i]] * m / (i + 1))
        q[order[i]] = running
    return q.tolist()


def known_settings() -> Dict[str, List[str]]:
    """For every outcome, the settings already known to matter for it (How the drawn settings shape a world)."""
    tests = json.load(open(EFFECTS))["tests"]
    return {t["outcome"]: [r["setting"] for r in t["rows"] if r["chance"] < KNOWN and r["setting"] != "hidden_layers"]
            for t in tests}


def by_shape_tests(ws: List[Dict[str, Any]], outcomes: Sequence[Tuple[str, str]],
                   known: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Every outcome against depth and against width, over Experiment 40's worlds, with what else is known."""
    rng = np.random.RandomState(SEED_TESTS)
    rows = []
    for key, name in outcomes:
        among = [w for w in ws if w["experiment"] == "E40" and (key == "died" or w["alive"])
                 and w.get(key) is not None]
        y = [float(w[key]) for w in among]
        settings = known.get(key, [])
        row = {"outcome": key, "name": name, "worlds": len(among), "known": settings}
        for factor, other, levels in (("depth", "width", DEPTHS), ("width", "depth", WIDTHS)):
            others = [[w["config"][s] for w in among] for s in settings] + [[w[other] for w in among]]
            r, chance = shape_test(y, [w[factor] for w in among], others, rng)
            plain, plain_chance = shape_test(y, [w[factor] for w in among], [], rng)
            values = {lv: [v for v, w in zip(y, among) if w[factor] == lv] for lv in levels}
            row[factor] = {"r": r, "chance": chance, "plain_r": plain, "plain_chance": plain_chance,
                           "by": {str(lv): {"worlds": len(v), "median": float(np.median(v)) if v else None,
                                            "mean": float(np.mean(v)) if v else None} for lv, v in values.items()}}
        rows.append(row)
    for factor in ("depth", "width"):
        for row, q in zip(rows, false_discovery([r[factor]["chance"] for r in rows])):
            row[factor]["fdr"] = q
    return rows


def cost_of(run_id: str, agents: float, iterations: int) -> Optional[float]:
    """
    About what a run cost per agent and iteration, in microseconds of processor time: the seconds of every
    session of the run (its provenance), over its iterations times its mean number of agents.
    """
    path = os.path.join(store.run_dir(run_id), "provenance.json")
    if not os.path.exists(path) or not agents:
        return None
    with open(path) as f:
        sessions = json.load(f).get("sessions") or []
    cpu = sum(s.get("cpu") or 0 for s in sessions)
    return 1e6 * cpu / (agents * iterations) if cpu else None


def _side_by_side(groups: Sequence[Sequence[Sequence[float]]], colours: Sequence[int],
                  labels: Sequence[Optional[str]], seed: int = 7) -> List[Dict[str, Any]]:
    """
    Dots of two or three kinds side by side in every category, each with a short bar at its median; the
    dots are spread sideways at random only so that they do not hide each other.
    """
    out: List[Dict[str, Any]] = []
    offsets = np.linspace(-0.25, 0.25, len(groups)) if len(groups) > 1 else [0.0]
    for g, (by_category, colour, label, off) in enumerate(zip(groups, colours, labels, offsets)):
        named = False
        for k, values in enumerate(by_category):
            v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
            if not v.size:
                continue
            out.append({"label": None if named else label, "y": v.tolist(), "points": True, "size": 5,
                        "x": F.jitter(k + off, v.size, seed + 31 * g + k, 0.06).tolist(),
                        "colour": colour, "alpha": 0.8})
            med = float(np.median(v))
            out.append({"label": None, "x": [k + off - 0.1, k + off + 0.1], "y": [med, med], "colour": colour,
                        "width": 3})
            named = True
    return out


def _chance(c: float) -> str:
    return f"{c:.3f}" if c >= 0.001 else "< 0.001"


@chapter
def brainsize(ch: F.Chapter) -> None:
    if not (os.path.exists(HEALTH) and os.path.exists(EFFECTS)):
        return
    ws = round_three()
    living = [w for w in ws if w["alive"]]
    probes = {w["id"]: probe(w["id"]) for w in living}
    shapes = [(d, wd) for d in DEPTHS for wd in WIDTHS]
    cats = [_shape(*s) for s in shapes]
    minds = json.load(open(MINDS))["numbers"]
    b1 = {"gain": minds["gain"]["evolved"]["median"], "founders": minds["gain"]["founder"]["median"],
          **{k: minds[f"changed_{k}"][0]["median"] for k in ("richer", "noise", "messages")}}
    ch.number("b1", b1)

    def of(shape, kind, key, among=living):
        return [probes[w["id"]][kind][key] for w in among if (w["depth"], w["width"]) == shape]

    # ---- the nine shapes: how many worlds, what a brain holds, what it costs ----
    e40 = [w for w in ws if w["experiment"] == "E40"]
    for w in living:
        w["cost"] = cost_of(w["id"], w["mean_agents"], 3000) if w["experiment"] == "E40" else None
    rows = []
    for (d, wd), name in zip(shapes, cats):
        drawn = [w for w in e40 if (w["depth"], w["width"]) == (d, wd)]
        alive = [w for w in living if (w["depth"], w["width"]) == (d, wd)]
        weights = [probes[w["id"]]["weights"] for w in alive]
        cost = [w["cost"] for w in alive if w.get("cost")]
        rows.append([name, f"{sum(w['alive'] for w in drawn)} of {len(drawn)}",
                     f"{sum(w['experiment'] != 'E40' for w in alive)}",
                     f"{int(np.median(weights)):,}" if weights else "—",
                     f"{np.median(of((d, wd), 'founders', 'gain')):.2f}" if alive else "—",
                     f"{np.median(of((d, wd), 'evolved', 'gain')):.2f}" if alive else "—",
                     f"{np.median(cost):.0f}" if cost else "—"])
    ch.number("table:shapes", {"head": ["hidden layers × units", "lived, of the worlds drawn (Experiment 40)",
                                        "more living worlds (Experiments 41 and 42)", "numbers in a brain (median)",
                                        "founders' gain", "evolved gain",
                                        "processor time per agent and iteration, µs (median)"],
                               "rows": rows})
    weights_all = [probes[w["id"]]["weights"] for w in living]
    ch.number("weights", {_shape(*s): describe([probes[w["id"]]["weights"] for w in living
                                                if (w["depth"], w["width"]) == s]) for s in shapes}
              | {"all": describe(weights_all)})
    costs = [(probes[w["id"]]["weights"], w["cost"], w["depth"]) for w in living if w.get("cost")]
    slope, level = np.polyfit(np.log([c[0] for c in costs]), np.log([c[1] for c in costs]), 1)
    ends = [min(c[0] for c in costs), max(c[0] for c in costs)]
    depth_colour = {3: YELLOW, 4: GREEN, 5: VIOLET}
    ch.figure(
        "cost", title="What a brain costs",
        x={"label": "numbers in a brain (logarithmic)", "log": True, "min": 1000, "max": 30000},
        y={"label": "processor time per agent and iteration, µs (logarithmic)", "log": True, "min": 200,
           "max": 4000},
        series=[{"label": f"{d} hidden layers", "x": [c[0] for c in costs if c[2] == d],
                 "y": [c[1] for c in costs if c[2] == d], "points": True, "size": 6, "colour": depth_colour[d],
                 "alpha": 0.85} for d in DEPTHS]
        + [line(ends, [math.exp(level) * e ** slope for e in ends], f"fit: time ∝ size^{slope:.2f}", GREY,
                dash=[4, 4])],
        caption="One dot per living world of Experiment 40: the number of weights and biases in each of its "
                "brains, against the processor time its run took per agent and iteration — the seconds of every "
                "session of the run, over 3,000 iterations times the world's mean number of agents. The dashed "
                "line is a straight line fitted to the logarithms of both.",
        recipe=recipe(RUNS, "each run's `GraphOfLifeRuns/<run>/provenance.json` (the processor seconds of each of its "
                            "sessions), its settings, and its mean number of agents from iteration 100 on "
                            "(`research/screen/health.json`)",
                      ["A brain's size: its weights and biases, `GraphOfLifeSimple.brain_shape`.",
                       "Time: the sum of `cpu` over the sessions, in microseconds, over 3,000 × the mean agents "
                       "(`cost_of`).",
                       "Least squares of ln(time) on ln(size)."]))
    ch.number("cost", {"by_depth": {d: describe([w["cost"] for w in living if w.get("cost") and w["depth"] == d])
                                    for d in DEPTHS},
                       "by_shape": {_shape(*s): describe([w["cost"] for w in living if w.get("cost")
                                                          and (w["depth"], w["width"]) == s]) for s in shapes},
                       "elasticity": float(slope), "worlds": len(costs),
                       "eightfold": float(8 ** slope)})

    # ---- a change in, a change out ----
    gain_f = [of(s, "founders", "gain") for s in shapes]
    gain_e = [of(s, "evolved", "gain") for s in shapes]
    ch.figure(
        "gain", title="Every brain of round 3 responds",
        x={"label": "hidden layers × units in each", "categories": cats},
        y={"label": "output change ÷ input change (logarithmic)", "log": True, "min": 3e-4, "max": 5},
        series=_side_by_side([gain_f, gain_e], [GREY, BLUE], ["founders' brains", "evolved brains"]),
        guides=[{"axis": "y", "at": b1["gain"], "label": f"B1, evolved: {b1['gain']:.4f}"},
                {"axis": "y", "at": b1["founders"], "label": f"B1, founders: {b1['founders']:.4f}"},
                {"axis": "y", "at": 1.0, "label": "passed on whole"}],
        caption=f"Each dot is a living world of round 3, placed by its brains' shape: the median, over {AGENTS} of its "
                "agents, of how much a small change of every input reaches the outputs — every input moved by a "
                f"normal amount of standard deviation {NUDGE}, and the root-mean-square change of the outputs over "
                "that of the inputs, the brain's gain. Blue: the agents' own brains at the end of the world; grey: "
                "fresh founders' brains of the same shape (weights normal with standard deviation 1/√(fan-in), "
                "biases 0) shown exactly the same inputs; the bar is the median. The dashed lines are B1's five "
                "sigmoid layers ([What the brains are like](what-the-brains-are-like.md)).",
        recipe=recipe(RUNS, DATA_PROBE, [
            PROBE_STEP,
            f"Draw ε, normal with standard deviation {NUDGE} (seed {SEED_NUDGE}), the shape of each agent's input "
            "matrix X; the gain is RMS(forward(X + ε) − forward(X)) / RMS(ε), for its own brain and for a fresh "
            f"founder's (seed {SEED_FOUNDERS}).",
            "The median over the agents of a world (`book_chapters.brainsize.probe`)."]))
    ch.number("gain", {
        "founders": describe([g for gs in gain_f for g in gs]), "evolved": describe([g for gs in gain_e for g in gs]),
        "founders_by_depth": {d: describe([p["founders"]["gain"] for w, p in ((w, probes[w["id"]]) for w in living)
                                           if w["depth"] == d]) for d in DEPTHS},
        "evolved_by_depth": {d: describe([p["evolved"]["gain"] for w, p in ((w, probes[w["id"]]) for w in living)
                                          if w["depth"] == d]) for d in DEPTHS},
        "founders_by_width": {wd: describe([probes[w["id"]]["founders"]["gain"] for w in living if w["width"] == wd])
                              for wd in WIDTHS},
        "evolved_by_width": {wd: describe([probes[w["id"]]["evolved"]["gain"] for w in living if w["width"] == wd])
                             for wd in WIDTHS},
        "ratio": describe([probes[w["id"]]["evolved"]["gain"] / probes[w["id"]]["founders"]["gain"] for w in living]),
        "score": {k: describe([probes[w["id"]][k]["gain_score"] for w in living]) for k in ("founders", "evolved")},
        "decide": {k: describe([probes[w["id"]][k]["gain_decide"] for w in living]) for k in ("founders", "evolved")},
        "worlds": len(living)})

    # ---- layer by layer ----
    panels = []
    layers_out = {}
    for d in DEPTHS:
        among = [probes[w["id"]] for w in living if w["depth"] == d]
        names = [str(i) for i in range(1, d + 1)] + ["out"]
        f = np.median([p["founders"]["layers"] for p in among], axis=0)
        e = np.median([p["evolved"]["layers"] for p in among], axis=0)
        layers_out[d] = {"founders": f.tolist(), "evolved": e.tolist(), "worlds": len(among),
                         "founders_product": float(np.prod(f)), "evolved_product": float(np.prod(e)),
                         "founders_sums": np.median([p["founders"]["sums"] for p in among], axis=0).tolist(),
                         "evolved_sums": np.median([p["evolved"]["sums"] for p in among], axis=0).tolist()}
        panels.append(dict(title=f"{d} hidden layers ({len(among)} worlds)",
                           x={"label": "layer", "categories": names},
                           y={"label": "change after ÷ change before", "min": 0, "max": 1.6},
                           series=[F.bars(names, f, GREY, "founders' brains", (-0.38, 0.0)),
                                   F.bars(names, e, BLUE, "evolved brains", (0.0, 0.38))],
                           guides=[{"axis": "y", "at": 1.0, "label": "unchanged"},
                                   {"axis": "y", "at": 0.25, "label": "a sigmoid's ¼"}],
                           legend=d == DEPTHS[0]))
    ch.grid("layers", panels, columns=3, title="Layer by layer",
            caption="How a small change of the inputs grows or shrinks through each layer of the brain: the "
                    "root-mean-square change after the layer over that before it, the median over the agents probed "
                    "of a world and then over the living worlds of each depth. Every hidden layer (1 to 5) ends in "
                    "tanh; the output layer (out) is linear. Grey: fresh founders' brains; blue: the agents' own "
                    "brains at the end. The lower dashed line is the most a sigmoid layer could pass on.",
            recipe=recipe(RUNS, DATA_PROBE, [
                PROBE_STEP,
                "With ε as for the gain, follow X and X + ε through the brain one layer at a time, and divide the "
                "root-mean-square difference after each layer by that before it (`through_layers`)."]))
    ch.number("layers", layers_out)

    # ---- do the decisions follow? ----
    placed = [w for w in living if w["config"].get("allocation", "full") == "full"]
    kinds = [("founders", "founders", GREY), ("drift", "balance", CYAN), ("evolved", "evolved", BLUE)]
    variants = [("richer", "Neighbours richer"), ("noise", "Noise drawn again"), ("messages", "No messages")]
    panels, changed = [], {}
    for key, title in variants:
        among = [probes[w["id"]] for w in placed if key in probes[w["id"]]["variants"]]
        groups = [[p[k][f"changed_{key}"] for p in among] for k, _, _ in kinds]
        changed[key] = {k: describe(g) for (k, _, _), g in zip(kinds, groups)} | {"worlds": len(among)}
        changed[key]["moved"] = {k: describe([p[k][f"moved_{key}"] for p in among]) for k, _, _ in kinds}
        panels.append(dict(title=f"{title} ({len(among)} worlds)",
                           x={"label": "", "categories": [label for _, label, _ in kinds]},
                           y={"label": "share of agents whose stakes change", "min": 0, "max": 1},
                           series=dots([label for _, label, _ in kinds], groups, [c for _, _, c in kinds]),
                           guides=[{"axis": "y", "at": b1[key], "label": f"B1: {b1[key]:.1%}"}],
                           legend=False))
    ch.grid("decisions", panels, columns=3, title="What changes a decision",
            caption="One dot per living world of Experiment 40, whose agents place all their stakes at once: the "
                    "share of the agents probed whose stakes would differ at all if one thing about their inputs "
                    "were different. Left: every neighbour's tokens, as the agent sees them, doubled; middle: the "
                    "random numbers of every column drawn again (only in worlds whose brains read random numbers); "
                    "right: every message read set to 0 (only in worlds with messages). Grey: fresh founders' "
                    "brains; cyan: founders' brains taken to mutation's balance, where its changes alone would bring "
                    "them, with nothing selected (see the text); blue: the agents' own brains; the bar is the "
                    "median. The dashed line is B1's evolved brains.",
            recipe=recipe(RUNS, DATA_PROBE, [
                PROBE_STEP,
                "Each brain's stakes by the engine's rules (`_apportion`, or all on the best-scored candidate), "
                "once on X and once on X with one thing changed: row `its_tokens` of every neighbour's column "
                "replaced by ln(1 + 2·(eˣ − 1)); the noise rows drawn again, uniform on (−2, 2) (seed "
                f"{SEED_NOISE}); the message rows set to 0.",
                "Count the agents whose stakes differ in any candidate."]))
    ch.number("changed", changed)
    ch.number("decisions", {k: {"even": describe([probes[w["id"]][k]["even"] for w in placed]),
                                "spread": describe([probes[w["id"]][k]["spread"] for w in placed]),
                                "child": describe([probes[w["id"]][k]["child"] for w in placed])}
                            for k, _, _ in kinds} | {"worlds": len(placed)})

    # ---- what evolution did to the gain, and how much of it mutation alone does ----
    eras = sorted({int(t) for w in living for t in probes[w["id"]]["eras"]})
    times = eras + [3000]
    table = []
    for w in living:
        p = probes[w["id"]]
        start = p["eras"]["0"]["gain"]
        table.append([p["eras"][str(t)]["gain"] / start if str(t) in p["eras"] else np.nan for t in eras]
                     + [p["evolved"]["gain"] / start])
    band = {k: v for k, v in F.quantile_band(table, times).items() if k not in ("outerLo", "outerHi")}
    drift_ratio = [probes[w["id"]]["drift"]["gain"] / probes[w["id"]]["eras"]["0"]["gain"] for w in living]
    scale_e = [float(np.median(probes[w["id"]]["evolved"]["scale"])) for w in living]
    scale_m = [probes[w["id"]]["balance"] for w in living]
    colours = {3: YELLOW, 4: GREEN, 5: VIOLET}
    ge = [probes[w["id"]]["evolved"]["gain"] for w in living]
    gd = [probes[w["id"]]["drift"]["gain"] for w in living]
    scatter, spread = [], []
    for d in DEPTHS:
        idx = [i for i, w in enumerate(living) if w["depth"] == d]
        scatter.append({"label": f"{d} hidden layers", "x": [gd[i] for i in idx], "y": [ge[i] for i in idx],
                        "points": True, "size": 6, "colour": colours[d], "alpha": 0.85})
        spread.append({"label": None, "x": [scale_m[i] for i in idx], "y": [scale_e[i] for i in idx],
                       "points": True, "size": 6, "colour": colours[d], "alpha": 0.85})
    top = max(ge + gd) * 1.05
    ch.grid("evolution", [
        dict(title="Through a world's life", x={"label": "iteration of the brains", "min": 0, "max": 3000},
             y={"label": "gain ÷ the founders' gain", "min": 0, "max": 3},
             series=[{**band, "label": None, "colour": BLUE, "width": 2}],
             guides=[{"axis": "y", "at": 1.0, "label": "founders"},
                     {"axis": "y", "at": float(np.median(drift_ratio)),
                      "label": f"mutation alone: {np.median(drift_ratio):.2f}"}],
             legend=False),
        dict(title="Against mutation alone", x={"label": "gain at mutation's balance", "min": 0, "max": top},
             y={"label": "gain of the evolved brains", "min": 0, "max": top},
             series=scatter + [line([0, top], [0, top], None, GREY, dash=[4, 4])], legend=True),
        dict(title="The weights' size", x={"label": "balance (× a founder's)", "min": 0.9, "max": 2.0},
             y={"label": "evolved weights (× a founder's)", "min": 0.9, "max": 2.0},
             series=spread + [line([0.9, 2.0], [0.9, 2.0], None, GREY, dash=[4, 4])], legend=False)],
        columns=3, title="What evolution did to the gain",
        caption="Left: the brains of every world kept along the way (one every 250 iterations, the founders first, "
                "the agents' own at the end) shown the same inputs, those of the world's end, and their gain over "
                "the founders' gain; the line is the median of the 82 living worlds, the band holds the middle half "
                "of them. The upper dashed line is the median gain, over the founders', of founders' brains taken to "
                "mutation's balance: where its changes alone would bring them after very many, with nothing "
                "selected. Middle: one dot per world, the gain of its evolved brains against that of brains at its "
                "mutation's balance; on the dashed line the two are equal. Right: each world's evolved weights, the "
                "median over layers of their size as a multiple of a founder's, against the size mutation alone "
                "would bring them to.",
            recipe=recipe(RUNS, DATA_PROBE, [
                PROBE_STEP,
                f"For every kept world of the run, {AGENTS} brains of agents holding a token, drawn at random (seed "
                f"{SEED_ERA} + the iteration), each shown one agent's inputs; their gain as for the figure above.",
                "Mutation's balance (`balance`): a change jitters a share s of the numbers by a normal amount of "
                "standard deviation σ/√k and, with chance s, redraws a share s of a matrix at the founders' 1/√k; "
                "the variance of a weight settles at (1 + (1 − s²)σ²/s)/k, and of a bias at σ²(1 + (1 − s²)/s)/k. "
                "A founder's weights scaled to it, and biases drawn at it (`drifted`).",
                "The weights' size: the root mean square of a layer's weights times √(fan-in), the median over the "
                "layers and the agents."]))
    ch.number("evolution", {
        "by_era": {str(t): describe(col) for t, col in zip(times, np.array(table).T.tolist())},
        "drift_ratio": describe(drift_ratio), "evolved_ratio": describe([r[-1] for r in table]),
        "over_drift": describe([e / d for e, d in zip(ge, gd)]),
        "above_drift": int(sum(e > d for e, d in zip(ge, gd))), "worlds": len(living),
        "scale_evolved": describe(scale_e), "scale_balance": describe(scale_m),
        "scale_over_balance": describe([e / m for e, m in zip(scale_e, scale_m)]),
        "scale_above": int(sum(e > m for e, m in zip(scale_e, scale_m))),
        "correlation": float(np.corrcoef(gd, ge)[0, 1]),
        "excess_correlation": float(np.corrcoef([e / m for e, m in zip(scale_e, scale_m)],
                                                [e / d for e, d in zip(ge, gd)])[0, 1]),
        "evolved_by_sigma": {str(sigma): describe([probes[w["id"]]["evolved"]["gain"] for w in living
                                                   if float(w["config"]["mutation_noise_std"]) == sigma])
                             for sigma in sorted({float(w["config"]["mutation_noise_std"]) for w in living})},
        "above_one": int(sum(g > 1 for g in ge))})
    ch.number("balance_check", balance_check(sorted({(float(w["config"]["mutation_noise_std"]),
                                                      float(w["config"]["mutation_sparsity"])) for w in living})))

    # ---- one change, against a stranger ----
    def relative(kind, key, among):
        """One change's difference over a stranger's, per world, by shape."""
        return [[probes[w["id"]][kind][f"mutation_{key}"] / probes[w["id"]][kind][f"stranger_{key}"]
                 for w in among if (w["depth"], w["width"]) == s and probes[w["id"]][kind][f"stranger_{key}"] > 0]
                for s in shapes]
    outputs = {k: relative(k, "outputs", living) for k in ("founders", "evolved")}
    stake = {k: relative(k, "stake", placed) for k in ("founders", "evolved")}

    def by_depth(groups):
        return [[v for s, g in zip(shapes, groups) if s[0] == d for v in g] for d in DEPTHS]
    depths = [f"{d} hidden layers" for d in DEPTHS]
    top_change = max(v for k in outputs for g in outputs[k] for v in g) * 1.05
    ch.figure(
        "change", title="How far one change moves a brain",
        x={"label": "", "categories": depths},
        y={"label": "one change ÷ a stranger's brain", "min": 0, "max": top_change},
        series=_side_by_side([by_depth(outputs["founders"]), by_depth(outputs["evolved"])], [GREY, BLUE],
                             ["founders' brains", "evolved brains"]),
        caption="One dot per living world, by the depth of its brains: how much one change of a brain ([How a "
                "brain changes](rules-in-detail.md#how-a-brain-changes), made for sure, with the world's own settings) moves the outputs that "
                "make decisions, over how much the brain of a stranger differs from it, on the same inputs — the "
                "root-mean-square differences, the median over the agents probed. Grey: fresh founders' brains, "
                "each against another founder's; blue: the agents' own brains at the end, each against another "
                "agent's of the same world; the bar is the median. The dots high above the rest are the worlds "
                "whose changes touch three numbers in ten (a mutation sparsity of 0.3).",
        recipe=recipe(RUNS, DATA_PROBE, [
            PROBE_STEP,
            f"Each brain changed once with the world's own mutation settings (seed {SEED_MUTATION}; `mutated`), and "
            "both shown the agent's inputs; the stranger is the agent probed half the sample away in the order of "
            "their ids, so that it was born far from it in time.",
            "The ratio of the two differences, per world (`_stake_change`)."]))
    flat = lambda groups: [v for g in groups for v in g]

    def settings_of(w):
        return float(w["config"]["mutation_noise_std"]), float(w["config"]["mutation_sparsity"])
    usual = [w for w in living if settings_of(w) == (0.2, 0.1)]
    ch.number("change_settings", {
        f"sigma {sigma}, s {s}": {k: describe([probes[w["id"]][k]["mutation_outputs"]
                                               / probes[w["id"]][k]["stranger_outputs"]
                                               for w in living if settings_of(w) == (sigma, s)])
                                  for k in ("founders", "evolved")}
        for sigma, s in sorted({settings_of(w) for w in living})}
        | {"usual_by_depth": {d: {k: describe([probes[w["id"]][k]["mutation_outputs"]
                                               / probes[w["id"]][k]["stranger_outputs"]
                                               for w in usual if w["depth"] == d]) for k in ("founders", "evolved")}
                              for d in DEPTHS}})
    ch.number("change", {"outputs": {k: describe(flat(v)) for k, v in outputs.items()},
                         "stake": {k: describe(flat(v)) for k, v in stake.items()},
                         "by_depth": {k: {d: describe(flat(g for s, g in zip(shapes, outputs[k]) if s[0] == d))
                                          for d in DEPTHS} for k in outputs},
                         "by_width": {k: {wd: describe(flat(g for s, g in zip(shapes, outputs[k]) if s[1] == wd))
                                          for wd in WIDTHS} for k in outputs}})

    # ---- does the shape matter for the worlds? ----
    known = known_settings()
    tests = by_shape_tests(ws, OUTCOMES, known)
    ch.number("tests", tests)
    levels = [(f"{d} layers" if d == DEPTHS[0] else str(d), "depth", d) for d in DEPTHS] + \
             [(f"{wd} units" if wd == WIDTHS[0] else str(wd), "width", wd) for wd in WIDTHS]
    names = [name for name, _, _ in levels]
    shown = [("died", "Died", "share of the worlds"), ("health", "Health", "health"),
             ("heredity", "Heredity", "heredity index"), ("generations", "Generations", "in 3,000 iterations"),
             ("population", "Effective population", "log10"),
             ("novelty", "Late novelty", "steps above the noise")]
    panels = []
    for key, head, label in shown:
        among = [w for w in e40 if (key == "died" or w["alive"]) and w.get(key) is not None]
        if key == "died":
            rates = [float(np.mean([w["died"] for w in among if w[f] == v])) for _, f, v in levels]
            series = [F.bars(names, rates, BLUE)]
            y = {"label": label, "min": 0, "max": 0.5}
        else:
            groups = [[w[key] for w in among if w[f] == v] for _, f, v in levels]
            series = dots(names, groups, [ORANGE] * 3 + [GREEN] * 3)
            y = {"label": label}
        panels.append(dict(title=head, x={"label": "", "categories": names}, y=y, series=series, legend=False))
    ch.grid("worlds", panels, columns=3, title="What the shape did to the worlds",
            caption="Experiment 40's worlds by the depth of their brains (orange: 3, 4 or 5 hidden layers) and by "
                    "their width (green: 16, 32 or 50 units a layer): the share that died, of all 84; and for the 70 "
                    "living worlds, one dot each with a bar at the median, their health (the six criteria "
                    "multiplied, as their geometric average), how much children act like their parents, how many "
                    "generations they lived in 3,000 iterations, their effective population, and how far their line "
                    "of descent kept moving into new behaviour ([Candidates for open-ended evolution]"
                    "(candidates-for-open-ended-evolution.md#how-each-pillar-is-read)). The table below gives, for "
                    "each, the chance by shuffling of a trend with depth or with width at least as strong, with the "
                    "settings already known to matter for that outcome taken into account.",
            recipe=recipe(RUNS, DATA_SCREEN, [
                "For each outcome, the settings whose own shuffling test on it had a chance below "
                f"{KNOWN} ([How the drawn settings shape a world](how-the-drawn-settings-shape-a-world.md)), "
                "and the other of depth and width.",
                "Rank the outcome over the worlds; take away a least-squares fit of the ranks on those settings' "
                "values (one column of 0 and 1 per value); correlate what is left with the rank of depth (or width).",
                f"The chance: the share of {SHUFFLES:,} shuffles of depth (or width) among the worlds (seed "
                f"{SEED_TESTS}) giving a correlation at least as far from 0 (`shape_test`)."]))

    def by(row, factor):
        values = row[factor]["by"]
        fmt = (lambda v: f"{v['mean']:.0%}") if row["outcome"] == "died" else (
            lambda v: f"{v['median']:,.0f}" if row["outcome"] == "generations" else f"{v['median']:.2f}")
        return " / ".join(fmt(v) for v in values.values())
    ch.number("table:tests", {
        "head": ["outcome", "3 / 4 / 5 layers", "chance (false discovery rate)", "16 / 32 / 50 units",
                 "chance (false discovery rate)", "taken into account"],
        "rows": [[r["name"] if r["outcome"] != "died" else "died (the share)", by(r, "depth"),
                  f"{_chance(r['depth']['chance'])} ({r['depth']['fdr']:.2f})", by(r, "width"),
                  f"{_chance(r['width']['chance'])} ({r['width']['fdr']:.2f})",
                  ", ".join(f"`{s}`" for s in r["known"]) or "—"] for r in tests]})
    ch.number("findings", [f"{r['outcome']}/{f}" for r in tests for f in ("depth", "width") if r[f]["fdr"] < FINDING])

    # ---- the brains' own numbers, by depth and width ----
    rng = np.random.RandomState(SEED_TESTS + 1)
    brain_tests = {}
    for kind in ("founders", "evolved", "drift"):
        among = [w for w in living if w["experiment"] == "E40"]
        y = [probes[w["id"]][kind]["gain"] for w in among]
        for factor, other in (("depth", "width"), ("width", "depth")):
            r, chance = shape_test(y, [w[factor] for w in among], [[w[other] for w in among]], rng)
            brain_tests[f"{kind}_{factor}"] = {"r": r, "chance": chance, "worlds": len(among)}
    ch.number("brain_tests", brain_tests)
