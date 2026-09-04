#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphOfLife (Simplified) — Open-Ended Evolution on a Mutable Graph
==============================================================================

WHAT THIS IS
------------
A population of agents lives on the nodes of an undirected graph. Every agent
owns two things:

  * a pool of TOKENS  — simultaneously its wealth, its life, and its voting
                        power. Tokens are globally conserved.
  * a BRAIN           — a small feed-forward network that is never trained by
                        gradients. It evolves only by copy + mutation.

Nothing is optimized toward a goal. Whatever survives, survives.

This module is the engine. It holds no file paths and no UI: it advances a
world and hands back a frame describing what happened. Persistence lives in
gol_store.py, the HTTP API in gol_server.py, and the visualization in web/.


HOW ONE STEP WORKS
------------------
Each call to `step(t)` runs two phases in order and returns one frame each.

PHASE 1 — REPRODUCTION
  1. Precompute the sensory manifold: log-scaled tokens and degrees for every
     node, plus six-quantile summaries of each neighborhood.
  2. Every agent `u` with at least one token observes its candidate set
     `[u] + neighbors(u)` — one input column per candidate — and runs a single
     forward pass over all columns at once.
  3. The REPRO_FRACTION head decides what fraction of `u`'s tokens to endow a
     child with. Below one whole token, no child is born.
  4. If a child is born it is a mutated copy of the parent's brain, and the
     parent pays for it out of its own tokens. Reproduction is literally
     splitting your own life force.
  5. The LINK head decides, per candidate, whether the newborn is wired to that
     candidate. LINK_MODE lets the agent choose whether LINK is read as a
     probability or as a hard argmax.
  6. Apply the new edges, drop self-loops, then run cleanup (below).

  Phase 1 does NOT move or create any other edges. No walker, no edge
  shifting, no reconnection.

PHASE 2 — BLOTTO (competition)
  1. A message pass: every agent broadcasts its MESSAGE head to itself and its
     neighbors, so phase-2 decisions see fresh signals.
  2. Every agent allocates ALL of its tokens in one shot across
     `[self] + neighbors`, using the BLOTTO score head. BLOTTO_MODE lets the
     agent choose between spreading proportionally to the scores or going
     all-in on its single best target. Integer conservation is preserved by
     largest-remainder apportionment.
  3. For each target, the REV_FRACTION head decides what portion of the tokens
     sent there are flagged as REVOLUTION tokens.
  4. Winner of each node `v` is resolved as follows:
       - The HEGEMON is the largest single allocator to `v`.
       - The MOB is every revolutionary at `v` except the hegemon, sorted
         weakest-first.
       - Walk up the mob accumulating a "lower class" sum. At each rung ask:
             lower_class > (remaining upper class + hegemon) ?
         The first rung where that tips, the revolution wins and the winner is
         drawn from the group that tipped it.
       - If it never tips, the hegemon keeps the node.
     So a coalition of small allocators can unseat the richest one.
  5. ALLOCATE_AND_CONQUER: the winner's brain is copied into `v`. `v`'s new
     token count is everything that was allocated to it. This is the selection
     mechanism — phase 1 spreads genes, phase 2 decides which genes win.
  6. Every brain in the world then mutates.
  7. Edges that carried zero tokens this phase are pruned, then cleanup.

CLEANUP (after both phases)
  1. Nodes at zero tokens starve and are removed.
  2. Only the largest connected component survives; splinter groups are culled.
  3. All tokens from the dead are pooled and redistributed uniformly at random
     across the survivors, keeping the global count conserved.
  4. If the world empties completely, a single fresh agent is resurrected
     holding every token.


AGENT-CONTROLLED RANDOMNESS
---------------------------
There is no global "be probabilistic" switch. Each discrete decision is paired
with a MODE head, and the agent decides for itself whether that decision is
read as a probability or as a hard maximum. The choice is part of the genome
and therefore evolves.


LINEAGE
-------
Two independent ancestries are tracked, and they are not the same thing:

  * NODE lineage   — `parent_of[child] = parent`, who spawned whom in phase 1.
  * BRAIN lineage  — `brain_id` / `parent_brain_id`, which survives conquest.
                     When an agent conquers a node, the node keeps its own id
                     but receives the conqueror's genome.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from gol_config import SimConfig

# Node ids use -1 to mean "no parent" (founder or resurrected agent).
NO_PARENT = -1

def build_heads(cfg: SimConfig):
    """
    Which output rows mean what, for this configuration.

    The optional rows exist only when their option is on. Always reserving them
    would change the brain's shape for every run, and a checkpoint saved under
    one shape cannot be resumed under another — so a run keeps exactly the
    architecture it started with.
    """
    heads = {
        "REPRO_FRACTION": slice(0, 2),   # fraction of my tokens to invest in a child
        "LINK": slice(2, 4),             # link the newborn to this candidate?
        "LINK_MODE": slice(4, 6),        # read LINK as probability, or as maximum?
        "BLOTTO": 6,                     # desirability of allocating tokens here
        "BLOTTO_MODE": slice(7, 9),      # spread proportionally, or go all-in?
    }
    nxt = 9
    if cfg.allow_revolutions:
        heads["REV_FRACTION"] = slice(nxt, nxt + 2)   # portion of this allocation that revolts
        nxt += 2
    if cfg.allow_handover:
        heads["HANDOVER"] = slice(nxt, nxt + 2)       # give this edge to the child?
        heads["HANDOVER_MODE"] = slice(nxt + 2, nxt + 4)
        nxt += 4
    heads["MESSAGE_START"] = nxt
    return heads


# ----------------------------------------------------------------------------
# Mathematical utilities
# ----------------------------------------------------------------------------

def _six_quantiles(sorted_vals: List[float]) -> List[float]:
    """Compress a sorted distribution into 6 representative quantiles."""
    if not sorted_vals:
        return [0.0] * 6
    out: List[float] = []
    for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = (len(sorted_vals) - 1) * q
        i0, i1 = int(np.floor(x)), int(np.ceil(x))
        if i0 == i1:
            out.append(float(sorted_vals[i0]))
        else:
            w = x - i0
            out.append(float((1 - w) * sorted_vals[i0] + w * sorted_vals[i1]))
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    pos = x >= 0
    neg = ~pos
    z = np.empty_like(x, dtype=float)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    z[neg] = ex / (1.0 + ex)
    return z


def _share_of_first(a: float, b: float) -> float:
    """
    Turn a pair of raw logits into the share belonging to `a`.

    Negative logits are clipped to zero; a fully non-positive pair falls back to
    an even split. Used both for fractions ("how much?") and for probabilities
    ("how likely?").
    """
    va, vb = max(0.0, float(a)), max(0.0, float(b))
    total = va + vb
    return (va / total) if total > 0.0 else 0.5


def _choose_binary(yes: float, no: float, mode_yes: float, mode_no: float) -> bool:
    """
    A yes/no decision whose *interpretation* the agent picks for itself.

    `mode_yes > mode_no` means "read (yes, no) as a probability and sample".
    Otherwise the larger of (yes, no) simply wins. Because the mode head is part
    of the genome, whether a lineage decides sharply or stochastically is itself
    subject to selection.
    """
    if mode_yes > mode_no:
        return bool(np.random.random() < _share_of_first(yes, no))
    # An exact tie is not a "no", it is the absence of a preference, and
    # answering no would be a decision the agent never made. Floats effectively
    # never land here; integer outputs do it constantly, and a brain whose
    # every tie fell the same way would carry a bias it could not evolve out
    # of. Undetermined at the maximum falls back to the probabilistic reading,
    # which for a tie is a coin.
    if yes == no:
        return bool(np.random.random() < 0.5)
    return bool(yes > no)


def _pick_index(scores: np.ndarray, sample: bool) -> int:
    """
    Choose one entry from a row of scores.

    Sampled in proportion to the scores, or simply the largest, depending on
    what the agent's own mode head asked for.
    """
    if not sample:
        return int(np.argmax(scores))

    vals = np.maximum(0.0, scores)
    total = float(vals.sum())
    probs = (vals / total) if total > 0.0 else np.full(len(vals), 1.0 / len(vals))
    return int(np.random.choice(len(scores), p=probs))


def _apportion(weights: np.ndarray, total: int) -> np.ndarray:
    """
    Split `total` indivisible tokens across `weights` without losing any.

    Uses largest-remainder apportionment: hand out the floors, then give the
    leftovers to whoever was rounded down hardest.
    """
    vals = np.maximum(0.0, weights)
    s = float(vals.sum())
    probs = (vals / s) if s > 0.0 else np.full(len(vals), 1.0 / len(vals))

    raw = probs * float(total)
    alloc = np.floor(raw).astype(int)
    leftover = total - int(alloc.sum())
    if leftover > 0:
        for idx in np.argsort(-(raw - alloc))[:leftover]:
            alloc[idx] += 1
    return alloc


# ----------------------------------------------------------------------------
# The Brain (neural substrate)
# ----------------------------------------------------------------------------

class Brain:
    """
    A feed-forward network with sigmoid hidden layers and a linear output.

    Evaluated on a whole candidate set at once: pass a matrix of shape
    (n_inputs, n_candidates) and get back (n_outputs, n_candidates), so one
    forward pass scores an entire neighborhood.

    Brain ids are assigned from a counter owned by the World, not a class
    global, so a resumed run continues its lineage numbering exactly.
    """

    __slots__ = ("weights", "biases", "brain_id", "parent_brain_id", "cfg")

    #: What weights are stored as. Subclasses narrow it; the arithmetic below
    #: still happens in double precision, so this is about what is kept rather
    #: than what is computed with.
    dtype = np.float64

    def __init__(self, cfg: SimConfig, brain_id: int, allocate: bool = True) -> None:
        self.cfg = cfg
        self.brain_id = brain_id
        self.parent_brain_id: int = -1
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        if allocate:
            self._allocate()

    def layer_sizes(self) -> List[int]:
        cfg = self.cfg
        return [cfg.n_inputs()] + list(cfg.hidden_layers) + [cfg.n_outputs()]

    def _allocate(self) -> None:
        sizes = self.layer_sizes()
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.weights.append(
                np.random.normal(0.0, 1.0 / np.sqrt(fan_in),
                                 size=(fan_out, fan_in)).astype(self.dtype))
            self.biases.append(np.zeros((fan_out, 1), dtype=self.dtype))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Always returns a 2-D array of shape (n_outputs, n_candidates)."""
        a = np.asarray(x, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # Computed in double precision whatever the weights are stored as:
            # the saving being tested is in what a population has to hold, not
            # in how each sum is added up.
            z = W.astype(np.float64) @ a + b.astype(np.float64)
            a = z if i == len(self.weights) - 1 else _sigmoid(z)
        return a

    def copy_into(self, brain_id: int) -> "Brain":
        clone = type(self)(self.cfg, brain_id, allocate=False)
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = [b.copy() for b in self.biases]
        clone.parent_brain_id = self.brain_id
        return clone

    def mutate(self, brain_id: int) -> bool:
        """
        Sparse Gaussian perturbation, plus an occasional structural reset.

        Returns True if the genome actually changed, in which case it has taken
        on `brain_id` as a new identity.
        """
        cfg = self.cfg
        if np.random.random() > cfg.mutation_probability:
            return False

        reset_fraction = float(np.clip(cfg.mutation_sparsity, 0.0, 1.0))
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            base_scale = 1.0 / np.sqrt(W.shape[1])
            std = cfg.mutation_noise_std * base_scale
            self.weights[i] = self._perturb(
                W, std, base_scale, reset_fraction, cfg).astype(self.dtype)
            self.biases[i] = self._perturb(
                b, std, std if std > 0 else 0.01, reset_fraction, cfg).astype(self.dtype)

        self.parent_brain_id = self.brain_id
        self.brain_id = brain_id
        return True

    @staticmethod
    def _perturb(M: np.ndarray, noise_std: float, reset_std: float,
                 reset_fraction: float, cfg: SimConfig) -> np.ndarray:
        """Jitter a sparse subset of entries, then rarely re-draw some outright."""
        if cfg.mutation_sparsity <= 0.0:
            return M

        if noise_std > 0.0:
            jitter = np.random.random(M.shape) < cfg.mutation_sparsity
            M = M + np.random.normal(0.0, noise_std, size=M.shape) * jitter

        if reset_fraction > 0.0 and np.random.random() < cfg.mutation_sparsity:
            reset_mask = np.random.random(M.shape) < reset_fraction
            if np.any(reset_mask):
                M = np.where(reset_mask, np.random.normal(0.0, reset_std, size=M.shape), M)

        return M.astype(float, copy=False)


# ----------------------------------------------------------------------------
# The Arena
# ----------------------------------------------------------------------------

class Float16Brain(Brain):
    """
    The same network on half-precision weights.

    A quarter of the memory of the float brain, and about three decimal digits
    of precision instead of sixteen. The arithmetic is unchanged — the question
    it asks is whether evolution ever needed the digits that were thrown away,
    and a population that behaves the same on three of them is a population
    whose brains are mostly empty.
    """

    __slots__ = ()
    dtype = np.float16


class BinaryBrain(Brain):
    """
    Weights of -1, 0 or +1, and hidden units that are simply on or off.

    Every input arrives as a ladder of bits: bit i is set when the value clears
    the i-th threshold, so two nearby numbers differ in one bit and a distant
    one differs in many. Plain binary would not do — 127 and 128 share no bits
    at all, and the network would have to learn every magnitude as an unrelated
    pattern rather than as a place on a scale.

    A hidden unit fires when its weighted count of set inputs clears its
    threshold. The output layer does not fire: it hands back the count itself,
    because the decisions downstream do not only ask which output is larger,
    they divide tokens in proportion to them, and an on-or-off answer cannot
    say "twice as much".

    Nothing here is a float. The whole forward pass is integer addition, so a
    run comes out identical on any machine — the one thing the float brains
    cannot promise, since matrix multiplication rounds differently on different
    hardware and this simulation turns a last-bit difference into a different
    history.
    """

    __slots__ = ()
    dtype = np.int8

    # The ladder the magnitudes are spread across. It used to start below zero
    # because the noise and message inputs ran a little under it — they are not
    # on the ladder any more, and everything that is left is log1p of a count
    # or a quantile of one, so nothing can be negative. Three of sixteen rungs
    # sat under zero where nothing could ever reach them.
    #
    # Twelve at the top is about 160,000 tokens held by one agent.
    INPUT_LOW = 0.0
    INPUT_HIGH = 12.0

    def band_width(self) -> float:
        bands, _ = self.cfg.ladder_split()
        return (self.INPUT_HIGH - self.INPUT_LOW) / bands

    def thresholds(self) -> np.ndarray:
        """
        The band ladder: the lower edge of each stretch of the range.

        Edges, not points spread across the range — a band covers everything
        from its own edge up to the next one, and the last one takes the top.
        Reporting linspace here instead described a ladder the encoder was not
        using, which is a fine way to debug the wrong thing.
        """
        bands, _ = self.cfg.ladder_split()
        return self.INPUT_LOW + np.arange(bands, dtype=np.float64) * self.band_width()

    def fine_thresholds(self) -> np.ndarray:
        """The ladder inside one band, in units of that band's width."""
        _, within = self.cfg.ladder_split()
        return np.linspace(0.0, 1.0, within, dtype=np.float64)

    def layer_sizes(self) -> List[int]:
        # Every magnitude became `brain_bits` of them. The rest were already
        # bits and stay one row each.
        cfg = self.cfg
        return [cfg.binary_rows()] + list(cfg.hidden_layers) + [cfg.n_outputs()]

    def _allocate(self) -> None:
        sizes = self.layer_sizes()
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # Two in three weights start at zero. A layer of mostly -1 and +1
            # saturates: with hundreds of inputs the sum is far from its
            # threshold whatever any single input does, and nothing the agent
            # sees can move it.
            draw = np.random.random(size=(fan_out, fan_in))
            weights = np.zeros((fan_out, fan_in), dtype=np.int8)
            weights[draw < 1.0 / 6.0] = -1
            weights[draw > 5.0 / 6.0] = 1
            self.weights.append(weights)
            self.biases.append(np.zeros((fan_out, 1), dtype=np.int8))

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Spread each magnitude across its ladder, and pass the rest straight in.

        Only the tokens, degrees and quantiles are magnitudes. The is-self
        flag, the messages and the noise are already 0 or 1 in a binary world,
        and laddering them spent sixteen rows apiece to say one thing — the
        ladder covers roughly -2 to 12, so a value that never leaves 0 to 1
        could only ever reach its bottom rung. Measured before this: three
        hundred of the eight hundred and sixty-four rows in the first layer
        were permanently zero, carrying weights that were mutated for the whole
        of a run and could never affect anything.
        """
        cfg = self.cfg
        a = np.asarray(x, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(-1, 1)

        flags = a[:cfg.FLAG_INPUTS]
        span = cfg.FLAG_INPUTS + cfg.MAGNITUDE_INPUTS
        scale = a[cfg.FLAG_INPUTS:span]
        rest = a[span:]

        bands, _within = cfg.ladder_split()
        step = self.band_width()

        # Which band, and where inside it. Clipped at the top so a value above
        # the range lands in the last band, fully, rather than wrapping.
        placed = (scale - self.INPUT_LOW) / step
        band = np.clip(np.floor(placed), 0, bands - 1)
        within = np.clip(placed - band, 0.0, 1.0)

        coarse = (band[:, None, :] >= np.arange(bands)[None, :, None])
        fine = (within[:, None, :] >= self.fine_thresholds()[None, :, None])

        # Both fields of one magnitude sit together, so a unit reading a
        # contiguous run of rows is reading one quantity.
        pair = np.concatenate([coarse, fine], axis=1)
        ladder = pair.reshape(scale.shape[0] * cfg.brain_bits, a.shape[1])

        return np.vstack([flags, ladder, rest]).astype(np.int32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = self.encode(x)
        last = len(self.weights) - 1
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W.astype(np.int32) @ a + b.astype(np.int32)
            if i == last:
                # The count itself, so proportions downstream mean something.
                return z.astype(np.float64)
            # On when the count clears zero. Exactly zero counts as off, so the
            # rule is decided rather than left to whichever way a tie falls.
            a = (z > 0).astype(np.int32)
        return a.astype(np.float64)

    def mutate(self, brain_id: int) -> bool:
        """
        A weight steps to a neighbouring value rather than being redrawn.

        Redrawing from -1, 0, +1 outright looked like the obvious thing and was
        far too violent: measured against the float brain's jitter, which moves
        a weight about a sixth of the spread of its layer, a redraw moved it
        twice the spread — thirteen times as far. At the same mutation_sparsity
        every child was substantially brain-damaged, and populations died out
        in five runs of nine.

        Stepping by one keeps the same sparsity meaning roughly the same thing
        in both representations: a small move in genotype space rather than a
        teleport. A weight already at an end sometimes stays there, which makes
        the extremes slightly stickier than the middle — the discrete analogue
        of a jitter that cannot push a value past its range.
        """
        cfg = self.cfg
        if np.random.random() > cfg.mutation_probability:
            return False

        fraction = float(np.clip(cfg.mutation_sparsity, 0.0, 1.0))
        if fraction <= 0.0:
            return False

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            self.weights[i] = self._redraw(W, fraction)
            # Thresholds move by one step at a time, and are held in a range
            # where they can still be reached by a plausible count.
            picked = np.random.random(b.shape) < fraction
            step = np.random.choice(np.array([-1, 1], dtype=np.int8), size=b.shape)
            limit = np.int8(min(127, max(1, W.shape[1] // 4)))
            self.biases[i] = np.clip(b + picked * step, -limit, limit).astype(np.int8)

        self.parent_brain_id = self.brain_id
        self.brain_id = brain_id
        return True

    @staticmethod
    def _redraw(M: np.ndarray, fraction: float) -> np.ndarray:
        """Nudge a sparse subset one step along -1, 0, +1, and no further."""
        picked = np.random.random(M.shape) < fraction
        step = np.random.choice(np.array([-1, 1], dtype=np.int8), size=M.shape)
        stepped = np.clip(M.astype(np.int16) + picked * step, -1, 1)
        return stepped.astype(np.int8)


#: Every kind a brain can be, by the name a configuration uses.
BRAIN_KINDS = {
    "float": Brain,
    "float16": Float16Brain,
    "binary": BinaryBrain,
}


def make_brain(cfg: SimConfig, brain_id: int, allocate: bool = True) -> Brain:
    """The brain this configuration asks for."""
    return BRAIN_KINDS[cfg.brain_kind](cfg, brain_id, allocate=allocate)


class GraphOfLife:
    def __init__(self, G_init: nx.Graph | None, cfg: SimConfig, _empty: bool = False) -> None:
        self.cfg = cfg
        self.heads = build_heads(cfg)
        self.G = nx.Graph()
        self.next_agent_id = 0
        self.next_brain_id = 1
        self.iteration = 0

        self.tokens: Dict[int, int] = {}
        self.brains: Dict[int, Brain] = {}
        self.messages: Dict[int, Dict[int, List[float]]] = {}
        # Set to a callable to be offered a snapshot at each step of a phase.
        # Nothing in the engine sets it; see tools/record_explain_run.py.
        self.on_step = None
        self.parent_of: Dict[int, int] = {}

        if _empty:
            return

        # Relabel the seed graph to dense integer agent ids.
        old2new: Dict[Any, int] = {}
        for n in G_init.nodes():
            old2new[n] = self.next_agent_id
            self.G.add_node(self.next_agent_id)
            self.next_agent_id += 1
        for u, v in G_init.edges():
            self.G.add_edge(old2new[u], old2new[v])

        share = cfg.total_tokens // self.G.number_of_nodes()
        for aid in self.G.nodes():
            self.tokens[aid] = share
            self.brains[aid] = self._new_brain()
            self.messages[aid] = {}
            self.parent_of[aid] = NO_PARENT

        # Hand any rounding remainder to the founders so the count is exact.
        self._settle_remainder()

    # ------------------------------------------------------------------------
    # Identity helpers
    # ------------------------------------------------------------------------

    def _new_brain(self) -> Brain:
        brain = make_brain(self.cfg, self.next_brain_id)
        self.next_brain_id += 1
        return brain

    def _copy_brain(self, source: Brain) -> Brain:
        clone = source.copy_into(self.next_brain_id)
        self.next_brain_id += 1
        return clone

    def _mutate_brain(self, brain: Brain) -> None:
        if brain.mutate(self.next_brain_id):
            self.next_brain_id += 1

    def _settle_remainder(self) -> None:
        """Give any integer-division leftover to random survivors."""
        deficit = self.cfg.total_tokens - sum(self.tokens.values())
        survivors = list(self.G.nodes())
        if deficit > 0 and survivors:
            draws = np.random.multinomial(deficit, [1 / len(survivors)] * len(survivors))
            for u, extra in zip(survivors, draws):
                self.tokens[u] += int(extra)

    # ------------------------------------------------------------------------
    # Sensory manifold
    # ------------------------------------------------------------------------

    def _precompute_features(self) -> Tuple[
        Dict[int, float], Dict[int, List[int]], Dict[int, List[float]],
        Dict[int, List[float]], Dict[int, float]
    ]:
        """
        Build every node-level feature once per phase, in O(N).

        Everything is log-scaled so the brain perceives orders of magnitude
        rather than raw counts — the difference between 10 and 20 tokens should
        not look like the difference between 10,000 and 10,010.
        """
        log_tokens = {u: np.log1p(max(0, val)) for u, val in self.tokens.items()}
        log_degrees = {u: np.log1p(float(self.G.degree[u])) for u in self.G.nodes()}

        # Sorted, and that matters more than it looks. These lists become the
        # columns of the matrix an agent reads its neighbours from, so their
        # order decides which output refers to whom. Taken straight from
        # networkx they came back in insertion order — the order edges happened
        # to be added over the run's history — which meant the same graph could
        # present its neighbours differently depending on how it was arrived at.
        # A checkpoint rebuilt from an edge list did exactly that, so resuming a
        # run silently continued a different one. By node id it is the same
        # ordering every time, however the graph was built.
        neighs = {u: sorted(self.G.neighbors(u)) for u in self.G.nodes()}

        q_tok: Dict[int, List[float]] = {}
        q_deg: Dict[int, List[float]] = {}
        for u, N in neighs.items():
            if not N:
                q_tok[u] = [0.0] * 6
                q_deg[u] = [0.0] * 6
                continue
            q_tok[u] = _six_quantiles(sorted(log_tokens[n] for n in N))
            q_deg[u] = _six_quantiles(sorted(log_degrees[n] for n in N))

        return log_degrees, neighs, q_tok, q_deg, log_tokens

    def _input_vec(self, u: int, v: int, log_deg, q_tok, q_deg, log_tok) -> np.ndarray:
        """Assemble the sensory vector for observer `u` looking at candidate `v`."""
        cfg = self.cfg
        base = (
            [log_tok.get(u, 0.0), log_tok.get(v, 0.0), log_deg[u], log_deg[v]]
            + q_tok[u] + q_tok[v] + q_deg[u] + q_deg[v]
        )

        m = cfg.message_amount

        def msg(src: int, dst: int) -> List[float]:
            vec = self.messages.get(src, {}).get(dst)
            if vec is None:
                return [0.0] * m
            out = list(vec[:m])
            return out + [0.0] * (m - len(out))

        msg_feats = msg(u, u) + msg(u, v) + msg(v, u) + msg(v, v)
        # Coins in a binary world, a spread of magnitudes anywhere else. A
        # binary brain cannot read a magnitude that is not on its ladder, so
        # noise drawn as a float would arrive as a single bit anyway — this
        # just makes it an honest one.
        noise = (np.random.randint(0, 2, size=cfg.random_input_amount).astype(float)
                 if cfg.brain_kind == "binary"
                 else np.random.uniform(-2.0, 2.0, size=cfg.random_input_amount)).tolist()

        return np.array([int(u == v)] + base + msg_feats + noise, dtype=float)

    def _observe(self, u: int, candidates: List[int], log_deg, q_tok, q_deg, log_tok) -> np.ndarray:
        """One forward pass scoring every candidate."""
        X = np.column_stack([
            self._input_vec(u, v, log_deg, q_tok, q_deg, log_tok) for v in candidates
        ])
        return self.brains[u].forward(X)

    def _note(self, step: str, **marks) -> None:
        """
        Offer a snapshot of the world partway through a phase.

        Off unless something sets `on_step`, and then it costs one call per
        step. A frame is only written at the end of a phase, which is the right
        grain for a viewer replaying a run and far too coarse for anything
        trying to explain the algorithm: reproduction, the cull, the staking
        and the conquest all land in the same frame and cannot be told apart.

        The shape handed over is deliberately the same one a frame uses — ids,
        tokens, edges — plus a bag of marks naming who did what. Anything that
        can produce that shape can drive the same picture, which is what keeps
        this useful to more than the one page it was written for.
        """
        if self.on_step is None:
            return
        self.on_step({**(marks.pop("_over", None) or self._snapshot()),
                      "step": step,
                      "iteration": int(self.iteration),
                      "marks": marks})

    def _snapshot(self) -> Dict[str, Any]:
        """The graph as it stands, in the shape a frame uses."""
        nodes = sorted(self.G.nodes())
        return {
            "ids": [int(u) for u in nodes],
            "tokens": [int(self.tokens.get(u, 0)) for u in nodes],
            "edges": [[int(a), int(b)] for a, b in self.G.edges()],
        }

    def _emit_messages(self, u: int, targets: List[int], Y: np.ndarray,
                       outbox: Dict[int, Dict[int, List[float]]]) -> None:
        """
        Write `u`'s message head to each observed target, into an outbox.

        Into an outbox and not straight into self.messages, because the loop
        that calls this is also reading self.messages. Writing live meant an
        agent observed partly with messages from last phase and partly with
        ones written moments earlier in this one, and which it got depended on
        where its id fell in the loop: 17% of all message reads in a phase were
        of values written earlier in that same phase. Low ids read stale
        signals, high ids read fresh ones, for no reason anybody chose. The
        outbox is delivered once the phase is over, so every agent reads the
        same generation of messages.
        """
        cfg = self.cfg
        if not cfg.exchange_messages or cfg.message_amount <= 0:
            return
        start = self.heads["MESSAGE_START"]
        block = Y[start:start + cfg.message_amount, :]
        # A binary brain says bits. Its output layer hands back a count, and
        # squashing that through tanh produced a value that was neither a bit
        # nor a useful magnitude — eleven distinct values across a whole phase,
        # then read back through a ladder that could only see the bottom of it.
        rows = (block > 0).astype(float) if cfg.brain_kind == "binary" else np.tanh(block)
        for j, v in enumerate(targets):
            outbox.setdefault(u, {})[int(v)] = rows[:, j].astype(float).tolist()

    def _message_prepass(self, step: str, features) -> None:
        """
        A look that only talks, before the look that acts.

        Without it a phase is one pass — observe, speak, act — so the messages
        an agent acts on were written by its neighbours a phase ago, before the
        births, deaths and conquests since. It acts on a description of a graph
        that no longer exists.

        This runs the whole population through a forward pass whose only
        product is messages, delivers them, and leaves the acting pass to read
        a generation written from the graph as it stands. The acting pass still
        writes messages of its own afterwards; this adds a fresher generation
        rather than replacing the exchange.

        Everyone speaks here, including agents holding nothing. They are still
        present, their neighbours can still see them, and they still have
        something to say — the same rule the game phase has always used, and
        the reproduction phase's own acting pass does not, because that one
        skips anyone who cannot afford a child.
        """
        cfg = self.cfg
        if not (cfg.message_prepass and cfg.exchange_messages and cfg.message_amount > 0):
            return

        log_deg, neighs, q_tok, q_deg, log_tok = features
        outbox: Dict[int, Dict[int, List[float]]] = {}
        for u in sorted(self.G.nodes()):
            targets = [u] + list(neighs[u])
            Y = self._observe(u, targets, log_deg, q_tok, q_deg, log_tok)
            self._emit_messages(u, targets, Y, outbox)
        self._deliver_messages(outbox)
        self._note(step)

    def _deliver_messages(self, outbox: Dict[int, Dict[int, List[float]]]) -> None:
        """What was written this phase becomes what is read next."""
        if not outbox:
            return
        for u, notes in outbox.items():
            self.messages[u] = notes

    # ------------------------------------------------------------------------
    # Phase 1: Reproduction
    # ------------------------------------------------------------------------

    def reproduction_phase(self, record_decisions: bool) -> Dict[str, Any]:
        """
        Agents spend their own tokens to spawn children and choose the newborn's
        connections. No other topology change happens here.
        """
        decisions: List[Dict[str, Any]] = []
        handovers: List[Tuple[int, int, int]] = []
        outbox: Dict[int, Dict[int, List[float]]] = {}
        # Captured before anything changes, so the viewer can express births and
        # deaths as a share of the population that actually faced this phase,
        # and show how much each agent gained or lost across it.
        nodes_before = self.G.number_of_nodes()
        tokens_before = dict(self.tokens)
        features = self._precompute_features()
        # Nothing here depends on what anyone said, so the features are
        # measured once and both passes read the same ones.
        self._message_prepass("repro.messages", features)
        log_deg, neighs, q_tok, q_deg, log_tok = features
        self._note("repro.observe")

        for u in sorted(self.G.nodes()):
            tokens_u = int(self.tokens.get(u, 0))
            if tokens_u <= 0:
                continue

            candidates = [u] + list(neighs[u])
            Y = self._observe(u, candidates, log_deg, q_tok, q_deg, log_tok)
            self._emit_messages(u, candidates, Y, outbox)

            # How much of myself do I give away? Averaged over the whole view.
            frac = np.mean(Y[self.heads["REPRO_FRACTION"], :], axis=1)
            child_tokens = int(np.floor(_share_of_first(frac[0], frac[1]) * tokens_u))
            child_tokens = max(0, min(tokens_u, child_tokens))

            if child_tokens < 1:
                continue

            child_id, links = self._spawn_child(u, tokens_u, child_tokens, candidates, Y)

            # Which of the parent's own connections move to the newborn. Applied
            # after every birth, so nobody's neighbour list changes underfoot.
            given = self._choose_handovers(u, child_id, candidates, Y)
            for v in given:
                handovers.append((u, int(v), child_id))

            if record_decisions:
                decisions.append({
                    "agent": int(u),
                    "tokens_before": tokens_u,
                    "invested": child_tokens,
                    "child": int(child_id),
                    "links": [int(v) for v in links],
                    "handed_over": [int(v) for v in given],
                })

        # Hand the chosen edges over: the child gains the connection, the parent
        # loses it. If the newborn was already wired to that neighbour by the
        # link decision, adding it again is a no-op — the graph holds an edge
        # once — so a handover can never leave a duplicate behind, only move
        # where the single edge is anchored.
        for parent, v, child in handovers:
            if not (self.G.has_node(child) and self.G.has_edge(parent, v)):
                continue
            if child != v:
                self.G.add_edge(child, v)
            self.G.remove_edge(parent, v)

        self.G.remove_edges_from(list(nx.selfloop_edges(self.G)))
        self._note("repro.born",
                   born=[int(d["child"]) for d in decisions],
                   parents=[[int(d["agent"]), int(d["child"])] for d in decisions],
                   handed=[[int(p), int(v), int(c)] for p, v, c in handovers])
        self._deliver_messages(outbox)

        before = self._snapshot() if self.on_step else None
        alive_before = set(self.G.nodes())
        cleanup = self._cleanup_and_redistribute()
        self._note("repro.cleanup", _over=before,
                   removed=[int(u) for u in sorted(alive_before - set(self.G.nodes()))])

        payload = None
        if record_decisions:
            payload = {"births": decisions}

        return self._frame(phase=1, cleanup=cleanup, nodes_before=nodes_before,
                           tokens_before=tokens_before, decisions=payload)

    def _spawn_child(self, parent: int, parent_tokens: int, child_tokens: int,
                     candidates: List[int], Y: np.ndarray) -> Tuple[int, List[int]]:
        """Create the newborn, wire it up, and debit the parent."""
        child_id = self.next_agent_id
        self.next_agent_id += 1
        self.G.add_node(child_id)

        # The child inherits a mutated copy; the parent pays the full price.
        child_brain = self._copy_brain(self.brains[parent])
        self._mutate_brain(child_brain)
        self.brains[child_id] = child_brain

        self.tokens[child_id] = child_tokens
        self.tokens[parent] = parent_tokens - child_tokens
        self.messages[child_id] = {}
        self.parent_of[child_id] = int(parent)

        link_logits = Y[self.heads["LINK"], :]
        link_mode = Y[self.heads["LINK_MODE"], :]

        linked: List[int] = []
        for col, v in enumerate(candidates):
            if _choose_binary(link_logits[0, col], link_logits[1, col],
                              link_mode[0, col], link_mode[1, col]):
                if v != child_id and self.G.has_node(v):
                    self.G.add_edge(child_id, v)
                    linked.append(v)

        return child_id, linked

    def _choose_handovers(self, parent: int, child: int,
                          candidates: List[int], Y: np.ndarray) -> List[int]:
        """
        Which of the parent's connections are handed to the newborn.

        Read exactly like the link decision: a pair of logits says yes or no,
        and a second pair decides whether that pair is read as a probability or
        as a plain maximum. Only the neighbour columns are considered — column
        zero is the parent looking at itself, and there is no edge there to give
        away.
        """
        if not self.cfg.allow_handover:
            return []

        logits = Y[self.heads["HANDOVER"], :]
        mode = Y[self.heads["HANDOVER_MODE"], :]

        given: List[int] = []
        for col in range(1, len(candidates)):
            v = candidates[col]
            if v == child:
                continue
            if _choose_binary(logits[0, col], logits[1, col], mode[0, col], mode[1, col]):
                given.append(v)
        return given

    # ------------------------------------------------------------------------
    # Phase 2: Blotto
    # ------------------------------------------------------------------------

    def blotto_phase(self, record_decisions: bool) -> Dict[str, Any]:
        """
        Every agent spends its entire token pool bidding on itself and its
        neighbors. The winner of each node implants its brain there.
        """
        nodes_before = self.G.number_of_nodes()
        tokens_before = dict(self.tokens)
        features = self._precompute_features()
        self._message_prepass("game.messages", features)
        log_deg, neighs, q_tok, q_deg, log_tok = features

        # --- 1. One look, which decides both what to say and where to stake ---
        #
        # This used to be two passes: everyone observed and wrote messages, then
        # everyone observed again to place their stakes, so that the stakes read
        # messages written in this phase rather than the last. It cost every
        # agent a second forward pass through its brain for the same
        # neighbourhood, and it made the game phase behave unlike the
        # reproduction phase, which has only ever looked once. One look now, and
        # its output feeds both heads.
        outbox: Dict[int, Dict[int, List[float]]] = {}
        self._note("game.observe")

        # --- 2. One-shot allocation ------------------------------------------
        allocations_to: Dict[int, Dict[int, int]] = {v: {} for v in self.G.nodes()}
        revolution_to: Dict[int, Dict[int, int]] = {v: {} for v in self.G.nodes()}
        incoming_totals: Dict[int, int] = {v: 0 for v in self.G.nodes()}
        edge_flow: Dict[Tuple[int, int], int] = {tuple(sorted(e)): 0 for e in self.G.edges()}
        alloc_records: List[Dict[str, Any]] = []

        for u in sorted(self.G.nodes()):
            targets = [u] + list(neighs[u])
            Y = self._observe(u, targets, log_deg, q_tok, q_deg, log_tok)
            # Written even by an agent with nothing to stake: it is still there,
            # its neighbours can still see it, and it still has something to say.
            self._emit_messages(u, targets, Y, outbox)

            tokens_u = int(self.tokens.get(u, 0))
            if tokens_u <= 0:
                continue

            scores = np.asarray(Y[self.heads["BLOTTO"], :], dtype=float)
            mode = np.mean(Y[self.heads["BLOTTO_MODE"], :], axis=1)

            # The agent picks its own doctrine: spread by score, or all-in.
            spread = bool(mode[0] > mode[1])
            if spread:
                alloc = _apportion(scores, tokens_u)
            else:
                alloc = np.zeros(len(targets), dtype=int)
                alloc[int(np.argmax(scores))] = tokens_u

            revolts_allowed = self.cfg.allow_revolutions
            rev_logits = Y[self.heads["REV_FRACTION"], :] if revolts_allowed else None
            rev_amounts: List[int] = []

            for idx, v in enumerate(targets):
                amount = int(alloc[idx])
                if amount <= 0:
                    rev_amounts.append(0)
                    continue

                incoming_totals[v] += amount
                allocations_to[v][u] = allocations_to[v].get(u, 0) + amount

                # Only part of what I send here needs to be revolutionary.
                if revolts_allowed:
                    rev_share = _share_of_first(rev_logits[0, idx], rev_logits[1, idx])
                    rev_amount = int(np.floor(rev_share * amount))
                    rev_amounts.append(rev_amount)
                    if rev_amount > 0:
                        revolution_to[v][u] = revolution_to[v].get(u, 0) + rev_amount
                else:
                    rev_amounts.append(0)

                if u != v:
                    edge = tuple(sorted((u, v)))
                    if edge in edge_flow:
                        edge_flow[edge] += amount

            if record_decisions:
                record = {
                    "agent": int(u),
                    "tokens": tokens_u,
                    "spread": spread,
                    "targets": [int(v) for v in targets],
                    "alloc": [int(a) for a in alloc],
                }
                # Absent rather than zero when revolutions are off, so the
                # viewer can tell "not part of these rules" from "allowed but
                # nobody used it".
                if revolts_allowed:
                    record["revolt"] = rev_amounts
                alloc_records.append(record)

        # --- 3. Resolve every contested node ---------------------------------
        new_tokens = dict(self.tokens)
        new_brains = dict(self.brains)
        winners: List[Dict[str, int]] = []
        # Who won what is needed by the frame's decision record and by the
        # stage notes, which are switched on separately. Building it for
        # whichever of the two is listening keeps the cost off a plain run and
        # the dependency in one visible place.
        wanted = record_decisions or self.on_step is not None

        for v in list(self.G.nodes()):
            offers = allocations_to[v]
            if not offers:
                # Nobody wanted this node, not even itself. It starves, but its
                # lineage gets one last copy before cleanup decides its fate.
                new_tokens[v] = 0
                new_brains[v] = self._copy_brain(self.brains[v])
                continue

            winner, max_amount, by_revolt = self._resolve_winner(offers, revolution_to[v])
            new_brains[v] = self._copy_brain(self.brains[winner])
            new_tokens[v] = int(incoming_totals[v])

            # Kept whenever anything downstream reads it. This used to be built
            # only under `record_decisions`, which also fed the stage notes
            # below — so a recording of the stages taken without decisions got
            # an empty list of winners and drew a game in which nobody won
            # anything, with nothing to say why.
            if wanted:
                entry = {
                    "node": int(v),
                    "winner": int(winner),
                    "amount": int(max_amount),
                }
                if self.cfg.allow_revolutions:
                    entry["revolt"] = int(by_revolt)
                winners.append(entry)

        self._note("game.stake",
                   flow=[[int(a), int(b), int(f)] for (a, b), f in edge_flow.items()],
                   staked=[[int(v), int(who), int(amount)]
                           for v, offers in allocations_to.items()
                           for who, amount in offers.items()])
        self._note("game.winner",
                   won=[[int(x["node"]), int(x["winner"]), int(x["amount"])] for x in winners],
                   revolts=[[int(x["node"]), int(x.get("revolt", 0))] for x in winners])

        self.tokens = new_tokens
        self.brains = new_brains
        self._note("game.conquer",
                   taken=[[int(x["node"]), int(x["winner"])] for x in winners
                          if int(x["winner"]) != int(x["node"])])

        # --- 4. Aftermath -----------------------------------------------------
        dead_edges = [e for e, flow in edge_flow.items() if flow == 0]
        if dead_edges:
            self.G.remove_edges_from(dead_edges)
        self._note("game.prune", cut=[[int(a), int(b)] for a, b in dead_edges])

        self._deliver_messages(outbox)
        before = self._snapshot() if self.on_step else None
        alive_before = set(self.G.nodes())
        cleanup = self._cleanup_and_redistribute()
        self._note("game.cleanup", _over=before,
                   removed=[int(u) for u in sorted(alive_before - set(self.G.nodes()))])

        # Every brain mutates, and it happens after the clearing-up rather than
        # before it. Nothing between the two reads a brain — the pruning goes on
        # token flow and the cleanup on tokens and connectivity — so the only
        # difference is that the brains of agents about to be removed are no
        # longer jittered on their way out the door.
        for brain in self.brains.values():
            self._mutate_brain(brain)
        self._note("game.mutate")

        self._prune_stale_messages()

        decisions = None
        if record_decisions:
            decisions = {
                "allocations": alloc_records,
                "winners": winners,
                "pruned_edges": [[int(a), int(b)] for a, b in dead_edges],
            }
        return self._frame(phase=2, cleanup=cleanup, nodes_before=nodes_before,
                           tokens_before=tokens_before, decisions=decisions)

    @staticmethod
    def _resolve_winner(offers: Dict[int, int],
                        revolutionaries: Dict[int, int]) -> Tuple[int, int, bool]:
        """
        Decide who takes a node, given every offer made on it.

        The establishment is the HEGEMON — the single largest allocator. Against
        it stands the MOB: every revolutionary except the hegemon itself (it is
        the one being fought, so its own revolution tokens never count).

        The mob is sorted weakest-first and walked upward, accumulating a
        "lower class". At each rung we ask whether the lower class outweighs
        everyone still above it plus the hegemon. The first rung where that
        tips, the revolution succeeds and the winner is drawn from the group
        that tipped it — so a crowd of small allocators can take a node from
        someone who outspent all of them individually.

        With revolutions disabled no tokens are ever flagged, so the mob is
        always empty and the node simply goes to whoever allocated the most,
        ties broken at random. That is the whole of the alternative rule; it
        needs no separate branch.

        Returns (winner_id, hegemon's allocation, whether a revolution won).
        """
        max_amount = max(offers.values())
        hegemon = int(np.random.choice([a for a, amt in offers.items() if amt == max_amount]))

        mob = [(agent, tokens) for agent, tokens in revolutionaries.items() if agent != hegemon]
        if not mob:
            return hegemon, max_amount, False

        mob.sort(key=lambda pair: pair[1])
        total_mob_tokens = sum(tokens for _, tokens in mob)

        lower_class = 0
        i = 0
        while i < len(mob):
            # Everyone allocating the same amount rises together.
            rung_amount = mob[i][1]
            rung: List[int] = []
            while i < len(mob) and mob[i][1] == rung_amount:
                rung.append(mob[i][0])
                lower_class += mob[i][1]
                i += 1

            upper_class = total_mob_tokens - lower_class
            if lower_class > upper_class + max_amount:
                return int(np.random.choice(rung)), max_amount, True

        # The mutiny never reached critical mass.
        return hegemon, max_amount, False

    # ------------------------------------------------------------------------
    # Cleanup: the physics of death
    # ------------------------------------------------------------------------

    def _cleanup_and_redistribute(self) -> Dict[str, Any]:
        """
        Starve the broke, cull the disconnected, and share out the estate.

        1. Nodes holding <= 0 tokens are removed.
        2. Of what remains, only the largest connected component survives.
        3. Every token held by the dead is pooled and scattered uniformly at
           random over the survivors, so the global count stays conserved.
        4. An empty world resurrects a single fresh agent holding everything.
        """
        report: Dict[str, Any] = {
            "resurrected": False,
            "starved": 0,
            "orphaned": 0,
            "redistributed": 0,
        }

        starved = [u for u in self.G.nodes() if self.tokens.get(u, 0) <= 0]

        G_active = self.G.copy()
        G_active.remove_nodes_from(starved)

        orphaned: set[int] = set()
        if G_active.number_of_nodes() > 0:
            components = sorted(nx.connected_components(G_active), key=len, reverse=True)
            for c in components[1:]:
                orphaned.update(c)

        doomed = set(starved) | orphaned
        report["starved"] = len(starved)
        report["orphaned"] = len(orphaned)

        global_pool = self.cfg.tokens_created_per_phase
        for u in doomed:
            global_pool += max(0, self.tokens.get(u, 0))

        if doomed:
            self.G.remove_nodes_from(list(doomed))
            for u in doomed:
                self.tokens.pop(u, None)
                self.brains.pop(u, None)
                self.messages.pop(u, None)
                self.parent_of.pop(u, None)

        survivors = list(self.G.nodes())
        if global_pool > 0 and survivors:
            # Multinomial keeps the token count exactly conserved.
            draws = np.random.multinomial(global_pool, [1 / len(survivors)] * len(survivors))
            for u, extra in zip(survivors, draws):
                self.tokens[u] = self.tokens.get(u, 0) + int(extra)
        report["redistributed"] = int(global_pool)

        if self.G.number_of_nodes() == 0:
            aid = self.next_agent_id
            self.next_agent_id += 1
            self.G.add_node(aid)
            self.tokens = {aid: self.cfg.total_tokens}
            self.brains = {aid: self._new_brain()}
            self.messages = {aid: {}}
            self.parent_of = {aid: NO_PARENT}
            report["resurrected"] = True

        return report

    def _prune_stale_messages(self) -> None:
        """Forget messages to or from anyone who is no longer a neighbor."""
        for u in list(self.messages.keys()):
            if not self.G.has_node(u):
                self.messages.pop(u, None)
                continue
            allowed = {u} | {int(w) for w in self.G.neighbors(u)}
            self.messages[u] = {v: vec for v, vec in self.messages[u].items() if v in allowed}

    # ------------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------------

    def step(self, record_decisions: bool = True) -> List[Dict[str, Any]]:
        """Advance one full iteration. Returns one frame per phase."""
        frames = [
            self.reproduction_phase(record_decisions),
            self.blotto_phase(record_decisions),
        ]
        self.iteration += 1
        return frames

    def is_extinct(self) -> bool:
        return self.G.number_of_nodes() <= self.cfg.extinction_threshold

    # ------------------------------------------------------------------------
    # Frames (what the viewer consumes)
    # ------------------------------------------------------------------------

    def _frame(self, phase: int, cleanup: Dict[str, Any], nodes_before: int,
               tokens_before: Dict[int, int],
               decisions: Dict[str, Any] | None,
               summary_extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Snapshot the world for the viewer.

        Node arrays are parallel and index-aligned; edges reference node ids
        directly. Both lineages travel with every frame so the UI can build
        genealogies without replaying the run.
        """
        nodes = sorted(self.G.nodes())
        brains = self.brains

        frame: Dict[str, Any] = {
            "iteration": self.iteration,
            "phase": phase,
            "nodes_before": int(nodes_before),
            "ids": [int(u) for u in nodes],
            "tokens": [int(self.tokens.get(u, 0)) for u in nodes],
            "brain_ids": [int(brains[u].brain_id) for u in nodes],
            "parent_brain_ids": [int(brains[u].parent_brain_id) for u in nodes],
            "parent_ids": [int(self.parent_of.get(u, NO_PARENT)) for u in nodes],
            # What this phase did to each agent's pile, end to end: a parent
            # paying for a child, a newborn receiving its endowment, a node
            # conquered or defended, plus whatever cleanup redistributed. A
            # node that did not exist beforehand counts its whole balance as
            # gained.
            "delta": [int(self.tokens.get(u, 0)) - int(tokens_before.get(u, 0)) for u in nodes],
            "edges": [[int(u), int(v)] for u, v in self.G.edges()],
            "cleanup": cleanup,
            "summary": {
                "nodes": self.G.number_of_nodes(),
                "edges": self.G.number_of_edges(),
                "tokens": int(sum(self.tokens.values())),
            },
        }
        if summary_extra:
            frame["summary"].update(summary_extra)
        if decisions is not None:
            frame["decisions"] = decisions
        return frame

    # ------------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------------

    def to_checkpoint(self) -> Dict[str, np.ndarray]:
        """
        Flatten the whole world into arrays for np.savez_compressed.

        Every brain shares one architecture, so layer `i` of all agents stacks
        into a single (n_agents, fan_out, fan_in) array. That is what makes
        checkpoints compress well.
        """
        nodes = sorted(self.G.nodes())
        index = {u: i for i, u in enumerate(nodes)}

        blob: Dict[str, np.ndarray] = {
            "ids": np.array(nodes, dtype=np.int64),
            "tokens": np.array([self.tokens[u] for u in nodes], dtype=np.int64),
            "brain_ids": np.array([self.brains[u].brain_id for u in nodes], dtype=np.int64),
            "parent_brain_ids": np.array([self.brains[u].parent_brain_id for u in nodes], dtype=np.int64),
            "parent_ids": np.array([self.parent_of.get(u, NO_PARENT) for u in nodes], dtype=np.int64),
            "edges": np.array([[index[u], index[v]] for u, v in self.G.edges()],
                              dtype=np.int64).reshape(-1, 2),
            "counters": np.array([self.next_agent_id, self.next_brain_id, self.iteration],
                                 dtype=np.int64),
        }

        # How a magnitude was encoded. The shape check on the way back in
        # cannot see this: splitting the ladder into a band and a place inside
        # it keeps the row count identical and changes what every one of those
        # rows means, so weights written under one split are nonsense under
        # another and nothing about their shape says so.
        if self.cfg.brain_kind == "binary":
            blob["ladder"] = np.array(self.cfg.ladder_split(), dtype=np.int64)

        n_layers = len(self.brains[nodes[0]].weights) if nodes else 0
        blob["n_layers"] = np.array([n_layers], dtype=np.int64)
        for layer in range(n_layers):
            blob[f"W{layer}"] = np.stack([self.brains[u].weights[layer] for u in nodes])
            blob[f"b{layer}"] = np.stack([self.brains[u].biases[layer] for u in nodes])

        # Messages in flight are world state like any other. Leaving them out
        # meant a resumed run began deaf for one phase while the run it claimed
        # to continue did not, which is enough to send the two apart.
        #
        # Stored as three flat arrays rather than a nested structure, since
        # savez wants arrays and the shape is entirely regular: who it is for,
        # who sent it, and the values themselves.
        recipients: List[int] = []
        senders: List[int] = []
        payloads: List[List[float]] = []
        for recipient in nodes:
            for sender, values in self.messages.get(recipient, {}).items():
                recipients.append(int(recipient))
                senders.append(int(sender))
                payloads.append([float(v) for v in values])

        width = len(payloads[0]) if payloads else 0
        blob["msg_to"] = np.array(recipients, dtype=np.int64)
        blob["msg_from"] = np.array(senders, dtype=np.int64)
        blob["msg_values"] = (np.array(payloads, dtype=np.float64)
                              if payloads else np.zeros((0, width), dtype=np.float64))

        # Preserve the RNG stream so a resumed run is not merely similar.
        state = np.random.get_state()
        blob["rng_keys"] = state[1].astype(np.uint32)
        blob["rng_scalars"] = np.array([state[2], state[3], state[4]], dtype=np.float64)

        return blob

    @classmethod
    def from_checkpoint(cls, blob: Any, cfg: SimConfig) -> "GraphOfLife":
        """Rebuild a world saved by `to_checkpoint`."""
        world = cls(None, cfg, _empty=True)
        world.heads = build_heads(cfg)

        ids = blob["ids"].tolist()
        tokens = blob["tokens"].tolist()
        brain_ids = blob["brain_ids"].tolist()
        parent_brain_ids = blob["parent_brain_ids"].tolist()
        parent_ids = blob["parent_ids"].tolist()

        world.G.add_nodes_from(ids)
        for a, b in blob["edges"].tolist():
            world.G.add_edge(ids[a], ids[b])

        n_layers = int(blob["n_layers"][0])
        weights = [blob[f"W{i}"] for i in range(n_layers)]
        biases = [blob[f"b{i}"] for i in range(n_layers)]

        # A checkpoint carries weights, not the shape they were for. If the
        # architecture has moved since — a hidden layer resized, or a change in
        # how inputs reach the first one — the arrays still load and the run
        # then dies inside a matrix multiply several steps later, saying
        # nothing about why. Checked here, where the answer is still obvious.
        if cfg.brain_kind == "binary":
            # `in` rather than .files: a checkpoint arrives here as an npz or
            # as a plain dict of arrays, and only one of those has a .files.
            saved = tuple(int(v) for v in blob["ladder"]) if "ladder" in blob else None
            if saved != cfg.ladder_split():
                raise ValueError(
                    f"this checkpoint's magnitudes were encoded as "
                    f"{saved or 'a single ladder'} and these settings encode them "
                    f"as {cfg.ladder_split()} (band rows, rows within a band). "
                    f"The rows are the same in number and not the same in "
                    f"meaning, so the weights cannot be carried across.")

        want = make_brain(cfg, 0, allocate=False).layer_sizes()
        got = [int(weights[0].shape[2])] + [int(w.shape[1]) for w in weights]
        if len(got) != len(want) or got != want:
            raise ValueError(
                f"this checkpoint was written for a brain of shape {got}, and "
                f"these settings describe one of shape {want}. A run can only "
                f"be resumed into the architecture it was saved from.")

        for i, u in enumerate(ids):
            brain = make_brain(cfg, int(brain_ids[i]), allocate=False)
            brain.parent_brain_id = int(parent_brain_ids[i])
            brain.weights = [w[i].copy() for w in weights]
            brain.biases = [b[i].copy() for b in biases]

            world.brains[u] = brain
            world.tokens[u] = int(tokens[i])
            world.messages[u] = {}
            world.parent_of[u] = int(parent_ids[i])

        # Messages, if this checkpoint is new enough to carry them. An older one
        # simply has none, which is the state it was restored with before.
        if "msg_to" in blob:
            msg_to = blob["msg_to"].tolist()
            msg_from = blob["msg_from"].tolist()
            values = blob["msg_values"]
            for i, recipient in enumerate(msg_to):
                if recipient in world.messages:
                    world.messages[recipient][int(msg_from[i])] = values[i].tolist()

        counters = blob["counters"].tolist()
        world.next_agent_id, world.next_brain_id, world.iteration = (
            int(counters[0]), int(counters[1]), int(counters[2])
        )

        scalars = blob["rng_scalars"].tolist()
        np.random.set_state((
            "MT19937", blob["rng_keys"].astype(np.uint32),
            int(scalars[0]), int(scalars[1]), float(scalars[2]),
        ))

        return world


# ----------------------------------------------------------------------------
# Construction helper
# ----------------------------------------------------------------------------

def new_world(cfg: SimConfig) -> GraphOfLife:
    """Create a fresh world from a Watts-Strogatz seed graph."""
    # networkx draws from the `random` module, which np.random.seed does not
    # touch, so without passing the seed through the starting graph came out
    # different every time and a seeded run was not reproducible at all.
    seed = None if cfg.seed is None else int(cfg.seed) % (2 ** 32)
    if seed is not None:
        np.random.seed(seed)
    G0 = nx.watts_strogatz_graph(n=cfg.resolved_n(), k=cfg.resolved_k(),
                                 p=cfg.rewire_p, seed=seed)
    return GraphOfLife(G0, cfg)


def _main() -> None:
    """Headless run, for driving the engine without the web UI."""
    cfg = SimConfig()
    world = new_world(cfg)
    print(f"🌍 n={cfg.resolved_n()} k={cfg.resolved_k()} tokens={cfg.total_tokens}")

    t = 0
    try:
        while True:
            world.step(record_decisions=False)
            print(f"iteration {t}: nodes={world.G.number_of_nodes()} "
                  f"edges={world.G.number_of_edges()} tokens={sum(world.tokens.values())}")
            if world.is_extinct():
                print("⚠️ Extinction.")
                break
            t += 1
    except KeyboardInterrupt:
        print(f"\nstopped at iteration {t}")


if __name__ == "__main__":
    _main()
