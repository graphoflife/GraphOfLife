# -*- coding: utf-8 -*-
"""
Assembly theory: how many steps to make a thing, and how many copies.

Assembly theory (Marshall et al. 2021; Sharma et al. 2023) measures an object
by the fewest joining steps that build it from basic blocks, reusing what was
built before (its assembly index), and an ensemble by how many copies it holds
of objects that take many steps: A = Σ e^{a_i} (n_i − 1) / N_T. Its home
ground is molecules, which are graphs; here the objects are the agents' ego
networks — an agent, its neighbours and the links among them — and the block
is one link (book_assembly.py: the exact index of a graph, its canonical form,
and the checks against what is known).

The worlds, each read after its last game:

- B1 at 100,000 tokens (Experiment 10, all thirty worlds, iteration 1,499);
- B2 at 10,000 tokens (Experiment 16, its ten worlds of B2, iteration 1,999);
- the five healthiest worlds of round 3 by the product (Experiments 40 and
  42), the densest world of round 3 (s3094) and its star (s3020), iteration
  2,999;

each beside its random twin (the same degrees, wired at random: seed 23, as
on Measuring space), and four networks built with no history at all: a
triangular lattice, a cubic lattice, a random tree and a random space. Two
worlds are also read through their lives, every 100 or 250 iterations.

Each world's shapes are kept beside the run, .book/<run>.assembly.json (and
.assembly-life.json); the controls are made again each time, from fixed seeds.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import book_assembly as AT
import book_data as D
import book_figures as F
import gol_store as store
from book_chapters.common import BLUE, CYAN, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW
from book_figures import chapter, recipe, runs_of
from book_graph import degree_preserving, graph

#: Ego networks of at most this many links get the exact index, whatever their copies.
CAP = 16
#: Steps of the exact search a larger shape gets when two or more agents share it (its index
#: counts in A only then); beyond them, its index is given as bounds.
WORK = 20_000
#: The random twin's seed, as on Measuring space.
TWIN_SEED = 23
#: The size of the networks built with no history, and their seed.
CONTROL_AGENTS = 10_000
CONTROL_SEED = 23
#: The random space's mean degree: about the mean of the worlds read (3.5 to 4.6 in B1 and round 3).
SPACE_DEGREE = 4.0

B1 = ("E10", "baseline")
B2 = ("E16", "B2")
ROUND3 = ["B1-30000-b82ad0-s3042", "B1-30000-8526cf-s4006", "B1-30000-af1e28-s3018",
          "B1-30000-625bc7-s3080", "B1-30000-38aee9-s3057"]
DENSE = "B1-30000-4ac946-s3094"
STAR = "B1-30000-511930-s3020"
#: Worlds read through their lives, and how often.
LIVES = [("B1-100000-s001", 100), ("B1-30000-b82ad0-s3042", 250)]
#: Worlds whose genotypes' depths are read, and how often.
DEPTHS = [("B1-100000-s001", 100), ("B2-10000-s001", 100), ("B1-30000-b82ad0-s3042", 250)]
#: The chance that a brain offered a change takes it, in B1 and B2 (mutation_probability).
CHANGE = 0.2

_ASSEMBLER = AT.Assembler()


# ---------------------------------------------------------------------------
# One network's ego networks
# ---------------------------------------------------------------------------

def shapes_of(adj: Dict[int, set]) -> Dict[str, Any]:
    """
    Every agent's ego network sorted into shapes, and each shape's links, the
    centre's degree, its copies and its assembly index (lo, hi; equal when
    exact). A shape of at most CAP links gets the exact index; a larger one
    gets it when at least two agents share it and the search ends within
    WORK units of work, and bounds otherwise (book_assembly.ego_index).

    An ego network is a set of agents and the links among them. Two agents
    linked to each other and to the same others (twins) have one and the same
    ego network, the same agents and links; it is counted once, as one object,
    not as two copies (`same`).
    """
    shapes = AT.Shapes(CAP)
    isolated = same = 0
    seen = set()
    for agent in adj:
        closed = frozenset(adj[agent]) | {agent}
        if closed in seen:
            same += 1
            continue
        seen.add(closed)
        links = AT.ego_links(adj, agent)
        if links:
            shapes.add(links)
        else:
            isolated += 1
    rows = []
    for key, copies in shapes.copies.items():
        nodes, m, links = shapes.example[key]
        shared = copies > 1
        lo, hi = AT.ego_index(_ASSEMBLER, key, nodes, m, links if shared else None, WORK if shared else None)
        rows.append([m, nodes - 1, copies, lo, hi])
    rows.sort()
    return {"objects": sum(shapes.copies.values()), "isolated": isolated, "same": same, "shapes": rows}


def summary(snap: Dict[str, Any]) -> Dict[str, Any]:
    """What the page reads off one network's shapes."""
    rows, total = snap["shapes"], snap["objects"]
    exact = [r for r in rows if r[3] == r[4]]
    covered = sum(r[2] for r in exact)
    floor = sum(r[2] for r in exact if r[3] == AT.chain_length(r[0]))
    shared_rows = [r for r in rows if r[2] > 1]           # a shape with one copy adds nothing to A

    def e(x: float) -> float:
        return math.exp(x) if x < 700 else math.inf

    a_lo = sum(e(r[3]) * (r[2] - 1) for r in shared_rows) / total
    a_hi = sum(e(r[4]) * (r[2] - 1) for r in shared_rows) / total
    by_size = sum(e(AT.chain_range(r[0])[0]) * (r[2] - 1) for r in shared_rows) / total
    shared = [r for r in exact if r[2] > 1]
    return {
        "objects": total, "isolated": snap["isolated"], "same": snap.get("same", 0), "shapes": len(rows),
        "exact": covered / total, "A": a_lo, "A_hi": a_hi, "A_floor": by_size,
        "mean_index": sum(r[3] * r[2] for r in exact) / covered if covered else 0.0,
        "mean_links": sum(r[0] * r[2] for r in rows) / total,
        "on_floor": floor / covered if covered else 0.0,
        "above_floor": covered - floor,
        "shared": sum(r[2] for r in rows if r[2] > 1) / total,
        "top_shared": max((r[3] for r in shared), default=0),
        "top_ten": max((r[3] for r in exact if r[2] >= 10), default=0),
        "largest_shared": max((r[0] for r in rows if r[2] > 1), default=0),
        "inexact_shared": sum(r[2] for r in rows if r[2] > 1 and r[3] != r[4]),
        "largest_inexact": max((r[0] for r in rows if r[2] > 1 and r[3] != r[4]), default=0),
    }


def _key(run_id: str, t: Optional[int] = None) -> Dict[str, Any]:
    return {"iteration": store.load_meta(run_id).get("iteration"), "t": t, "cap": CAP, "work": WORK,
            "twin": TWIN_SEED, "version": 2}


@D.measure("assembly", field="key", want=_key)
def assembly_of(run_id: str, t: Optional[int] = None) -> Dict[str, Any]:
    """A world's ego networks after the game of iteration t (its last, unless given), and its twin's."""
    t = D.last_iteration(run_id) if t is None else t
    adj = graph(D.frame_at(run_id, t, 2))
    return {"t": t, "world": shapes_of(adj), "twin": shapes_of(degree_preserving(adj, TWIN_SEED))}


def _life_key(run_id: str, every: int = 100) -> Dict[str, Any]:
    return {**_key(run_id), "every": every}


@D.measure("assembly-life", field="key", want=_life_key)
def assembly_life(run_id: str, every: int = 100) -> Dict[str, Any]:
    """The same, after the games of iterations 0, every, 2·every, … and the last: summaries only."""
    last = D.last_iteration(run_id)
    out = []
    for t in sorted(set(range(0, last + 1, every)) | {last}):
        adj = graph(D.frame_at(run_id, t, 2))
        out.append({"t": t, "world": summary(shapes_of(adj)),
                    "twin": summary(shapes_of(degree_preserving(adj, TWIN_SEED)))})
    return {"life": out}


def who_shares(run_id: str) -> Dict[str, Dict[str, int]]:
    """
    For the pairs of agents whose ego networks have the same shape, after the
    world's last game: how they stand to each other. Twins have the same
    neighbours (not linked to each other: linked twins share one ego network,
    counted once, as shapes_of counts it); otherwise they are neighbours, two
    steps apart, or farther. And whether one is the other's parent. By the
    shape's size: 6 to CAP links, and more.
    """
    t = D.last_iteration(run_id)
    frame = D.frame_at(run_id, t, 2)
    adj = graph(frame)
    parent = dict(zip(frame["ids"], frame["parent_ids"]))
    shapes = AT.Shapes(CAP)
    owners: Dict[Tuple, List[int]] = defaultdict(list)
    seen = set()
    same = Counter()
    for agent in adj:
        closed = frozenset(adj[agent]) | {agent}
        if closed in seen:
            same["linked twins counted once"] += 1
            same["of them a parent and its child"] += sum(1 for b in adj[agent] if adj[b] | {b} == closed
                                                          and (parent.get(agent) == b or parent.get(b) == agent)) > 0
            continue
        seen.add(closed)
        links = AT.ego_links(adj, agent)
        if links:
            owners[shapes.add(links)].append(agent)
    out: Dict[str, Counter] = {"small": Counter(), "large": Counter(), "same": same}
    for key, agents in owners.items():
        size = shapes.example[key][1]
        if len(agents) < 2 or size < 6:
            continue
        tally = out["large" if size > CAP else "small"]
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                if adj[a] | {a} == adj[b] | {b} or adj[a] == adj[b]:
                    tally["twins"] += 1
                elif b in adj[a]:
                    tally["neighbours"] += 1
                elif adj[a] & adj[b]:
                    tally["two apart"] += 1
                else:
                    tally["farther"] += 1
                if parent.get(a) == b or parent.get(b) == a:
                    tally["parent and child"] += 1
                tally["pairs"] += 1
    return {k: dict(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Brains: a genotype's path in the family tree
# ---------------------------------------------------------------------------

def _depth_key(run_id: str, every: int = 100) -> Dict[str, Any]:
    return {"iteration": store.load_meta(run_id).get("iteration"), "every": every, "version": 1}


@D.measure("genotype-depth", field="key", want=_depth_key)
def genotype_depth(run_id: str, every: int = 100) -> Dict[str, Any]:
    """
    Every genotype's depth, the changes on its line since its founder (the
    length of the path the family tree records: with one recorded parent, the
    only path it shows), read from every frame in order; and, after the games
    of every `every`th iteration and the last, the depths and copies of the
    genotypes the living carry, with the assembly A they would give if a
    genotype's index were its depth (as log10 A, for A overflows).
    """
    from gol_series import NO_PARENT
    # Depths by genotype number (numbers are handed out in order, a parent's before its child's):
    # an array, for a world may have made tens of millions of genotypes.
    depth = np.full(1 << 20, -1, dtype=np.int32)
    unseen = made = 0
    last = D.last_iteration(run_id)
    out = []
    for index in range(store.count_frames(run_id)):
        frame = store.read_frame(run_id, index)
        brains = np.asarray(frame["brain_ids"], dtype=np.int64)
        parents = np.asarray(frame["parent_brain_ids"], dtype=np.int64)
        if brains.size and brains.max() >= depth.size:
            grown = np.full(max(int(brains.max()) + 1, 2 * depth.size), -1, dtype=np.int32)
            grown[:depth.size] = depth
            depth = grown
        new = depth[brains] < 0
        if new.any():
            order = np.argsort(brains[new], kind="stable")
            for g, p in zip(brains[new][order].tolist(), parents[new][order].tolist()):
                if depth[g] >= 0:
                    continue
                if p == NO_PARENT:
                    depth[g] = 0
                else:
                    if depth[p] < 0:
                        unseen += 1
                    depth[g] = max(int(depth[p]), 0) + 1
                made += 1
        t = frame["iteration"]
        if frame["phase"] != 2 or (t % every and t != last):
            continue
        kinds, counts = np.unique(brains, return_counts=True)
        ds = depth[brains].astype(float)
        shared = [(int(depth[g]), int(n)) for g, n in zip(kinds, counts) if n > 1]
        point = {"t": t, "agents": int(ds.size), "genotypes": int(kinds.size),
                 "median_depth": float(np.median(ds)) if ds.size else 0.0,
                 "max_depth": float(ds.max()) if ds.size else 0.0,
                 "shared": sum(n for _, n in shared) / ds.size if ds.size else 0.0,
                 "most_copies": int(counts.max()) if counts.size else 0}
        if shared:
            top = max(d for d, _ in shared)
            terms = [math.exp(d - top) * (n - 1) for d, n in shared]
            point["top_shared_depth"] = top
            point["log10_A"] = (top + math.log(sum(terms)) - math.log(ds.size)) / math.log(10)
            point["top_term"] = max(terms) / sum(terms)
        out.append(point)
    return {"points": out, "parents_unseen": unseen, "genotypes": made}


# ---------------------------------------------------------------------------
# Networks with no history
# ---------------------------------------------------------------------------

def _adjacency(n: int, pairs) -> Dict[int, set]:
    adj: Dict[int, set] = {i: set() for i in range(n)}
    for a, b in pairs:
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def triangular_lattice(side: int) -> Dict[int, set]:
    """A triangular lattice on a torus: six neighbours each, and every ego network a wheel of six spokes."""
    at = lambda i, j: (i % side) * side + j % side  # noqa: E731
    pairs = [(at(i, j), at(i + di, j + dj)) for i in range(side) for j in range(side)
             for di, dj in ((1, 0), (0, 1), (1, -1))]
    return _adjacency(side * side, pairs)


def cubic_lattice(side: int) -> Dict[int, set]:
    """A cubic lattice on a three-dimensional torus: six neighbours each, and every ego network a star of six."""
    at = lambda i, j, k: ((i % side) * side + j % side) * side + k % side  # noqa: E731
    pairs = [(at(i, j, k), at(i + d[0], j + d[1], k + d[2])) for i in range(side) for j in range(side)
             for k in range(side) for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    return _adjacency(side ** 3, pairs)


def random_tree(n: int, seed: int) -> Dict[int, set]:
    """A random tree grown one node at a time, each joined to an earlier node drawn evenly: every ego network a star."""
    rng = random.Random(seed)
    return _adjacency(n, [(i, rng.randrange(i)) for i in range(1, n)])


def random_space(n: int, degree: float, seed: int) -> Dict[int, set]:
    """
    Points thrown evenly into a cube that wraps round on itself (a three-
    dimensional torus), each joined to every point nearer than the radius at
    which the mean degree is `degree`: local structure, triangles included,
    made by chance and geometry alone.
    """
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 3))
    radius = (3 * degree / (4 * math.pi * n)) ** (1 / 3)
    cells = max(1, int(1 / radius))
    grid: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i, p in enumerate(pts):
        grid[tuple((p * cells).astype(int) % cells)].append(i)
    pairs = []
    for (cx, cy, cz), members in grid.items():
        near = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    near.update(grid.get(((cx + dx) % cells, (cy + dy) % cells, (cz + dz) % cells), ()))
        near = sorted(near)
        for i in members:
            d = np.abs(pts[near] - pts[i])
            d = np.minimum(d, 1 - d)
            close = np.sqrt((d ** 2).sum(axis=1)) < radius
            pairs += [(i, j) for j, c in zip(near, close) if c and j > i]
    return _adjacency(n, pairs)


def controls() -> Dict[str, Dict[str, Any]]:
    """The four networks with no history, each with its twin."""
    side2 = round(math.sqrt(CONTROL_AGENTS))
    side3 = round(CONTROL_AGENTS ** (1 / 3))
    nets = {"lattice": triangular_lattice(side2), "cubic": cubic_lattice(side3),
            "tree": random_tree(CONTROL_AGENTS, CONTROL_SEED),
            "space": random_space(CONTROL_AGENTS, SPACE_DEGREE, CONTROL_SEED)}
    return {name: {"world": shapes_of(adj), "twin": shapes_of(degree_preserving(adj, TWIN_SEED)),
                   "agents": len(adj), "links": sum(map(len, adj.values())) // 2}
            for name, adj in nets.items()}


CONTROL_WORDS = {"lattice": "triangular lattice", "cubic": "cubic lattice", "tree": "random tree",
                 "space": "random space"}


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def _runs(experiment: str, condition: str) -> List[str]:
    return [s.run_id for s in runs_of(experiment, condition)]


def _short(run_id: str) -> str:
    return run_id.rsplit("-", 1)[-1]


def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def _rounded(x: float, digits: int = 2) -> float:
    return float(f"{x:.{digits}g}") if x and abs(x) < 10 else round(x, 1)


WORLDS_TEXT = ("the thirty worlds of B1 at 100,000 tokens (Experiment 10, `B1-100000-s001` … `-s030`) after "
               "the game of iteration 1,499; the ten worlds of B2 at 10,000 tokens (Experiment 16, "
               "`B2-10000-s001` … `-s010`) after the game of iteration 1,999; and seven worlds of round 3 of "
               "the screen, 30,000 tokens each, after the game of iteration 2,999: the five healthiest by the "
               "product (`B1-30000-b82ad0-s3042`, `B1-30000-8526cf-s4006` played strongest first, "
               "`B1-30000-af1e28-s3018`, `B1-30000-625bc7-s3080`, `B1-30000-38aee9-s3057`), the densest "
               "(`B1-30000-4ac946-s3094`) and the star (`B1-30000-511930-s3020`)")
DATA_TEXT = ("each run's last frame, `GraphOfLifeRuns/<run>/frames/`, read with `book_data.frame_at(run, t, 2)` "
             "(the agents and their connections after the game); the shapes are kept in "
             "`GraphOfLifeRuns/.book/<run>.assembly.json`")
EGO_STEPS = [
    "For every agent, its ego network: its links to its neighbours, and the links among those neighbours "
    "(`book_assembly.ego_links`).",
    "Sort the ego networks into shapes, one for each class of isomorphic graphs (`book_assembly.Shapes`: a "
    "canonical form, as nauty makes one, with twins skipped; above 16 links a Weisfeiler–Lehman hash first). "
    "A shape's copy number is how many agents have it.",
    f"Each shape's assembly index (`book_assembly.Assembler`): the fewest joins that build it from single links, "
    f"reusing anything built, found by a search from the shape down, deepening from the addition-chain bound "
    f"l(m) of its m links. Exact for every shape of at most {CAP} links; for a larger shape that two or more "
    f"agents share, exact when the search ends within {WORK:,} units of work, and otherwise an interval; for a larger "
    "star, l(d).",
    "The twin: the same agents with as many links each, wired at random (`book_graph.degree_preserving`, seed "
    f"{TWIN_SEED}), read the same way.",
]


@chapter
def assembly(ch: F.Chapter) -> None:
    b1 = {r: assembly_of(r) for r in _runs(*B1)}
    b2 = {r: assembly_of(r) for r in _runs(*B2)}
    r3 = {r: assembly_of(r) for r in ROUND3}
    dense, star = assembly_of(DENSE), assembly_of(STAR)
    made = controls()
    groups = {"B1": b1, "B2": b2, "round 3": r3, "dense": {DENSE: dense}, "star": {STAR: star}}
    sums = {g: {r: {"world": summary(v["world"]), "twin": summary(v["twin"])} for r, v in worlds.items()}
            for g, worlds in groups.items()}
    ctl = {name: {"world": summary(v["world"]), "twin": summary(v["twin"]), "agents": v["agents"],
                  "links": v["links"]} for name, v in made.items()}

    # ---- the floor: index against links, every shape read ----
    seen: Dict[Tuple[int, int], List[int]] = defaultdict(lambda: [0, 0])     # (links, index) -> [shapes, agents]
    unsure = 0
    for worlds in groups.values():
        for v in worlds.values():
            for side in ("world", "twin"):
                for m, d, n, lo, hi in v[side]["shapes"]:
                    if lo == hi:
                        seen[(m, lo)][0] += 1
                        seen[(m, lo)][1] += n
                    else:
                        unsure += 1
    for v in made.values():
        for side in ("world", "twin"):
            for m, d, n, lo, hi in v[side]["shapes"]:
                if lo == hi:
                    seen[(m, lo)][0] += 1
                    seen[(m, lo)][1] += n
    top_m = 40
    on = [(m, a) for (m, a) in seen if a == AT.chain_length(m) and m <= top_m]
    off = [(m, a) for (m, a) in seen if a != AT.chain_length(m) and m <= top_m]
    stair_x, stair_y = [], []
    for m in range(1, top_m + 1):
        stair_x += [m - 0.5, m + 0.5]
        stair_y += [AT.chain_length(m), AT.chain_length(m)]
    shapes_on = sum(seen[k][0] for k in seen if k[1] == AT.chain_length(k[0]))
    shapes_off = sum(seen[k][0] for k in seen if k[1] != AT.chain_length(k[0]))
    agents_on = sum(seen[k][1] for k in seen if k[1] == AT.chain_length(k[0]))
    agents_off = sum(seen[k][1] for k in seen if k[1] != AT.chain_length(k[0]))
    ch.figure(
        "floor", title="The index of an ego network is set by how many links it has",
        x={"label": "links of the ego network, m", "min": 0, "max": top_m + 1},
        y={"label": "assembly index", "min": 0, "max": 12},
        series=[F.line(stair_x, stair_y, "l(m): the fewest joins m links allow", GREY, width=1.5),
                F.line([1, top_m], [0, top_m - 1], "m − 1: one link at a time", GREY, dash=[4, 3], width=1),
                {"label": "a shape at the floor", "x": [m for m, _ in on], "y": [a for _, a in on],
                 "points": True, "size": 6, "colour": BLUE, "alpha": 0.9},
                {"label": "a shape above it", "x": [m for m, _ in off], "y": [a for _, a in off],
                 "points": True, "size": 7, "colour": ORANGE, "alpha": 0.95}],
        height=300,
        caption=(f"Every shape of ego network whose index is known exactly, in all the worlds read, their twins and "
                 f"the four networks with no history: a dot for each pair of links and index that occurs (up to "
                 f"{top_m} links). The grey steps are l(m), the shortest addition chain of the number of links: no "
                 f"shape can be built in fewer joins, since each join at most adds two sizes already made. The "
                 f"dashed line is m − 1, a link at a time, the most any shape needs. Blue: shapes whose index is "
                 f"exactly l(m), the floor; orange: shapes above it. Of {shapes_on + shapes_off:,} shapes counted "
                 f"(a shape found in two networks counts twice), {shapes_off:,} lie above the floor; they are "
                 f"the ego networks of {agents_off:,} of {agents_on + agents_off:,} agents."),
        recipe=recipe(f"{WORLDS_TEXT[0].upper() + WORLDS_TEXT[1:]}; and four networks built from seed "
                      f"{CONTROL_SEED} with no history (see the figure of the worlds).", DATA_TEXT,
                      EGO_STEPS + ["Mark each pair (links, index) that some shape with an exact index has: blue if "
                                   "the index is l(links), orange if it is more."]))
    ch.number("floor", {"shapes_on": shapes_on, "shapes_off": shapes_off, "agents_on": agents_on,
                        "agents_off": agents_off, "unsure_shapes": unsure,
                        "off": sorted([list(k) + seen[k] for k in seen if k[1] != AT.chain_length(k[0])])})

    # ---- AT's picture: copies against index ----
    def copies_series(snap: Dict[str, Any], label: str, colour: int, offset: float, seed: int) -> Dict[str, Any]:
        rows = [r for r in snap["shapes"] if r[3] == r[4]]
        rng = np.random.default_rng(seed)
        return {"label": label, "x": [r[3] + offset + rng.uniform(-0.12, 0.12) for r in rows],
                "y": [r[2] for r in rows], "points": True, "size": 5, "colour": colour, "alpha": 0.7}

    first_b1, first_b2 = _runs(*B1)[0], _runs(*B2)[0]
    panels = [(f"B1 at 100,000 tokens, world {_short(first_b1)}", b1[first_b1]),
              (f"B2 at 10,000 tokens, world {_short(first_b2)}", b2[first_b2]),
              (f"round 3, world {_short(ROUND3[0])}", r3[ROUND3[0]]),
              ("a random space (no history)", {"world": made["space"]["world"], "twin": made["space"]["twin"]})]
    charts = []
    for k, (title, v) in enumerate(panels):
        charts.append({"title": title, "x": {"label": "assembly index", "min": -0.5, "max": 9.5},
                       "y": {"label": "copies", "log": True, "min": 0.8, "max": 20000},
                       "series": [copies_series(v["world"], "the network's shapes", ORANGE, -0.18, 10 + k),
                                  copies_series(v["twin"], "its twin's shapes", BLUE, 0.18, 20 + k)],
                       "height": 220, "legend": k == 0})
    ch.grid("copies", charts, columns=2,
            title="Copies fall with the index, in the worlds as in their twins",
            caption=("Assembly theory's own picture: one dot for every shape of ego network, at its assembly index "
                     "(across) and its copy number, the agents whose ego network has that shape (up, on a "
                     "logarithmic scale). Orange: the network's shapes, a world's or the random space's; blue: its random twin's, the same agents "
                     "with as many links each, wired at random. The dots are spread a little sideways at random so "
                     "that they do not hide each other; only shapes whose index is known exactly are drawn. "
                     "Assembly theory reads many copies of a shape of high index as the mark of selection; a "
                     "shape with one copy adds nothing to the assembly A."),
            recipe=recipe(f"`{first_b1}`, `{first_b2}` and `{ROUND3[0]}` after their last games (iterations "
                          f"1,499, 1,999 and 2,999), and a random space of {CONTROL_AGENTS:,} points (seed "
                          f"{CONTROL_SEED}).", DATA_TEXT, EGO_STEPS + ["One dot per shape: its index and its copies."]))

    # ---- A in every world and its twin ----
    cats = ["B1", "B2", "round 3", "dense", "star", "triangles", "cubes", "tree", "space"]
    world_vals: List[List[float]] = []
    twin_vals: List[List[float]] = []
    for g in ("B1", "B2", "round 3", "dense", "star"):
        world_vals.append([s["world"]["A"] for s in sums[g].values()])
        twin_vals.append([s["twin"]["A"] for s in sums[g].values()])
    for name in ("lattice", "cubic", "tree", "space"):
        world_vals.append([ctl[name]["world"]["A"]])
        twin_vals.append([ctl[name]["twin"]["A"]])
    series = []
    for k, (w, t) in enumerate(zip(world_vals, twin_vals)):
        rng = np.random.default_rng(30 + k)
        series.append({"label": "the network" if k == 0 else None, "x": (k - 0.17 + rng.uniform(-0.08, 0.08, len(w))).tolist(),
                       "y": w, "points": True, "size": 6, "colour": ORANGE, "alpha": 0.85})
        series.append({"label": "its random twin" if k == 0 else None, "x": (k + 0.17 + rng.uniform(-0.08, 0.08, len(t))).tolist(),
                       "y": t, "points": True, "size": 6, "colour": BLUE, "alpha": 0.85})
        for vals, dx, c in ((w, -0.17, ORANGE), (t, 0.17, BLUE)):
            med = _median(vals)
            series.append({"label": None, "x": [k + dx - 0.13, k + dx + 0.13], "y": [med, med], "colour": c, "width": 3})
    ch.figure(
        "worlds", title="The assembly of a world is about that of its twin",
        x={"label": "", "categories": cats}, y={"label": "assembly A of the ego networks", "log": True,
                                                "min": 0.8, "max": 1000},
        series=series, height=300,
        caption=("The assembly A = Σ e^{a_i} (n_i − 1) / N_T of each network's ego networks, a_i the index of shape "
                 "i, n_i its copies and N_T the agents with at least one link. Orange: the network; blue: its random "
                 "twin. A dot per world; the bars are the medians. B1: the thirty worlds at 100,000 tokens; B2: the "
                 "ten at 10,000; round 3: its five healthiest worlds; dense: its densest world, s3094; star: "
                 "its world held together by one hub. The last four are built with no history, from seed "
                 f"{CONTROL_SEED}. Triangles: a triangular lattice on a torus, {CONTROL_AGENTS:,} nodes; cubes: a "
                 f"cubic lattice on a three-dimensional torus, {round(CONTROL_AGENTS ** (1 / 3)) ** 3:,} nodes; "
                 f"tree: a random tree of {CONTROL_AGENTS:,} nodes; space: a random space, {CONTROL_AGENTS:,} "
                 f"points in a cube that wraps round, joined when close enough for {SPACE_DEGREE:g} links each on "
                 "average. Where a shared shape's index is only bounded, A is drawn at its lower bound."),
        recipe=recipe(f"{WORLDS_TEXT[0].upper() + WORLDS_TEXT[1:]}.", DATA_TEXT,
                      EGO_STEPS + ["A = Σ e^{a_i} (n_i − 1) / N_T over the shapes.",
                                   "The networks with no history are made by `book_chapters/assembly.py` "
                                   "(`triangular_lattice`, `cubic_lattice`, `random_tree`, `random_space`) and "
                                   "read the same way."]))

    # ---- A through a life ----
    lives = {run: assembly_life(run, every)["life"] for run, every in LIVES}
    charts = []
    for k, (run, every) in enumerate(LIVES):
        life = lives[run]
        ts = [p["t"] for p in life]
        charts.append({"title": ("B1, world " if run.startswith("B1-100000") else "round 3, world ") + _short(run),
                       "x": {"label": "iteration", "min": 0},
                       "y": {"label": "assembly A", "min": 0},
                       "series": [F.line(ts, [p["world"]["A"] for p in life], "the world", ORANGE, width=2),
                                  F.line(ts, [p["twin"]["A"] for p in life], "its random twin", BLUE, width=2),
                                  F.line(ts, [p["world"]["A_floor"] for p in life], "the world, every index at l(m)",
                                         ORANGE, dash=[4, 3], width=1)],
                       "height": 200, "legend": k == 0})
    ch.grid("life", charts, columns=1, title="Through a world's life, its assembly follows its twin's",
            caption=(f"The assembly A of the ego networks after the game of every {LIVES[0][1]}th iteration of a B1 "
                     f"world (top) and every {LIVES[1][1]}th of round 3's healthiest world (bottom), from the "
                     "founders' ring at iteration 0 to the end. Orange: the world; blue: its random twin, made "
                     "again at each moment from the world's degrees then; dashed: the world's A with every shape's "
                     "index set to the floor l(m) of its number of links."),
            recipe=recipe(f"`{LIVES[0][0]}` and `{LIVES[1][0]}`.",
                          "each run's frames, read with `book_data.frame_at(run, t, 2)`; kept in "
                          "`GraphOfLifeRuns/.book/<run>.assembly-life.json`",
                          EGO_STEPS + ["A at each moment read, for the world and for its twin made then."]))

    # ---- who shares a shape ----
    ch.number("shares", {run: who_shares(run) for run in (first_b1, ROUND3[0])})

    # ---- brains: a genotype's depth in the family tree ----
    depths = {run: genotype_depth(run, every) for run, every in DEPTHS}
    names = {DEPTHS[0][0]: f"B1, world {_short(DEPTHS[0][0])}", DEPTHS[1][0]: f"B2, world {_short(DEPTHS[1][0])}",
             DEPTHS[2][0]: f"round 3, world {_short(DEPTHS[2][0])}"}
    series = []
    for (run, _), colour in zip(DEPTHS, (ORANGE, BLUE, GREEN)):
        pts = depths[run]["points"]
        series.append(F.line([p["t"] for p in pts], [p["median_depth"] for p in pts], names[run], colour, width=2))
        series.append(F.line([p["t"] for p in pts], [p["max_depth"] for p in pts], None, colour, width=1,
                             dash=[2, 3]))
    top_t = max(p["t"] for d in depths.values() for p in d["points"])
    series.append(F.line([0, top_t], [0, CHANGE * top_t], f"{CHANGE:g} changes an iteration", GREY, width=1,
                         dash=[5, 4]))
    ch.figure(
        "brains", title="A genotype's path in the family tree is a clock",
        x={"label": "iteration", "min": 0}, y={"label": "changes since the founder", "min": 0},
        series=series, height=260,
        caption=("The depth of a genotype: how many changes its line has had since its founder, the length of the "
                 "only pathway the family tree records. Solid: the median over the living agents' genotypes; "
                 "dotted: the deepest. B1 offers every brain a change after every game and every newborn's brain "
                 f"one at birth, each taken with probability {CHANGE:g}; B2 offers one only when a brain is "
                 "copied, to a newborn or to a node won; round 3's world s3042 mixes a newborn's brain from two "
                 f"parents, of which the tree keeps one, and {depths[DEPTHS[2][0]]['parents_unseen']:,} of its "
                 "genotypes have a parent that never appears in a frame, so its tree is cut into short pieces "
                 "and its depths say little. The grey dashed line is "
                 f"{CHANGE:g} changes an iteration."),
        recipe=recipe(f"`{DEPTHS[0][0]}`, `{DEPTHS[1][0]}` and `{DEPTHS[2][0]}`.",
                      "every frame of each run, in order (`brain_ids`, `parent_brain_ids`); kept in "
                      "`GraphOfLifeRuns/.book/<run>.genotype-depth.json`",
                      ["Give each genotype, the first time it is seen, the depth of its parent plus one (a founder's: "
                       "0; a parent never seen counts as a founder).",
                       f"After the games of every {DEPTHS[0][1]}th iteration ({DEPTHS[2][1]}th in round 3) and the "
                       "last, the median and the largest depth of the genotypes the living carry."]))
    ch.number("depths", depths)

    # ---- numbers and a table ----
    def group_numbers(g: str) -> Dict[str, Any]:
        s = list(sums[g].values())
        out = {}
        for side in ("world", "twin"):
            for key in ("A", "A_hi", "A_floor", "mean_index", "mean_links", "on_floor", "shared", "shapes",
                        "exact", "top_shared", "top_ten", "objects", "inexact_shared", "largest_shared",
                        "largest_inexact", "same"):
                vals = [x[side][key] for x in s]
                out[f"{side}_{key}"] = _median(vals)
                out[f"{side}_{key}_range"] = [float(min(vals)), float(max(vals))]
        out["above_twin"] = sum(1 for x in s if x["world"]["A"] > x["twin"]["A"])
        out["worlds"] = len(s)
        out["ratio"] = _median([x["world"]["A"] / x["twin"]["A"] for x in s])
        out["ratio_range"] = [min(x["world"]["A"] / x["twin"]["A"] for x in s),
                              max(x["world"]["A"] / x["twin"]["A"] for x in s)]
        return out

    ch.number("groups", {g: group_numbers(g) for g in sums})
    ch.number("worlds", {g: {r: v for r, v in s.items()} for g, s in sums.items()})
    ch.number("controls", ctl)
    ch.number("life", {run: lives[run] for run, _ in LIVES})
    rows = []
    for g, label in (("B1", "B1, 30 worlds"), ("B2", "B2, 10 worlds"), ("round 3", "round 3, 5 healthiest"),
                     ("dense", "round 3, dense (s3094)"), ("star", "round 3, star (s3020)")):
        n = group_numbers(g)
        rows.append([label, f"{n['world_mean_links']:.2f} / {n['twin_mean_links']:.2f}",
                     f"{n['world_mean_index']:.2f} / {n['twin_mean_index']:.2f}",
                     f"{100 * n['world_on_floor']:.1f}% / {100 * n['twin_on_floor']:.1f}%",
                     f"{100 * n['world_shared']:.0f}% / {100 * n['twin_shared']:.0f}%",
                     f"{n['world_A']:.1f} / {n['twin_A']:.1f}"])
    for name in ("lattice", "cubic", "tree", "space"):
        c = ctl[name]
        rows.append([CONTROL_WORDS[name], f"{c['world']['mean_links']:.2f} / {c['twin']['mean_links']:.2f}",
                     f"{c['world']['mean_index']:.2f} / {c['twin']['mean_index']:.2f}",
                     f"{100 * c['world']['on_floor']:.1f}% / {100 * c['twin']['on_floor']:.1f}%",
                     f"{100 * c['world']['shared']:.0f}% / {100 * c['twin']['shared']:.0f}%",
                     f"{c['world']['A']:.1f} / {c['twin']['A']:.1f}"])
    ch.number("table:worlds", {"head": ["networks (the median world)", "links of an ego network",
                                        "assembly index", "index at the floor l(m)",
                                        "agents whose shape another shares", "assembly A, at least"],
                               "rows": rows})
