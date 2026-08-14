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

  Phase 1 does NOT move, rewire, or create any other edges. No walker, no edge
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
    if cfg.allow_rewire:
        heads["REWIRE"] = slice(nxt, nxt + 2)         # move one of my edges at all?
        heads["REWIRE_MODE"] = slice(nxt + 2, nxt + 4)
        heads["REWIRE_DROP"] = nxt + 4                # which of my edges to give away
        heads["REWIRE_TO"] = nxt + 5                  # which neighbour receives it
        heads["REWIRE_PICK_MODE"] = slice(nxt + 6, nxt + 8)
        nxt += 8
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

    def __init__(self, cfg: SimConfig, brain_id: int, allocate: bool = True) -> None:
        self.cfg = cfg
        self.brain_id = brain_id
        self.parent_brain_id: int = -1
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        if allocate:
            sizes = [cfg.n_inputs()] + list(cfg.hidden_layers) + [cfg.n_outputs()]
            for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
                self.weights.append(np.random.normal(0.0, 1.0 / np.sqrt(fan_in), size=(fan_out, fan_in)))
                self.biases.append(np.zeros((fan_out, 1)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Always returns a 2-D array of shape (n_outputs, n_candidates)."""
        a = np.asarray(x, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ a + b
            a = z if i == len(self.weights) - 1 else _sigmoid(z)
        return a

    def copy_into(self, brain_id: int) -> "Brain":
        clone = Brain(self.cfg, brain_id, allocate=False)
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
            self.weights[i] = self._perturb(W, std, base_scale, reset_fraction, cfg)
            self.biases[i] = self._perturb(b, std, std if std > 0 else 0.01, reset_fraction, cfg)

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
        brain = Brain(self.cfg, self.next_brain_id)
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
        neighs = {u: list(self.G.neighbors(u)) for u in self.G.nodes()}

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
        noise = np.random.uniform(-2.0, 2.0, size=cfg.random_input_amount).tolist()

        return np.array([int(u == v)] + base + msg_feats + noise, dtype=float)

    def _observe(self, u: int, candidates: List[int], log_deg, q_tok, q_deg, log_tok) -> np.ndarray:
        """One forward pass scoring every candidate."""
        X = np.column_stack([
            self._input_vec(u, v, log_deg, q_tok, q_deg, log_tok) for v in candidates
        ])
        return self.brains[u].forward(X)

    def _emit_messages(self, u: int, targets: List[int], Y: np.ndarray) -> None:
        """Broadcast `u`'s message head to each observed target."""
        cfg = self.cfg
        if not cfg.exchange_messages or cfg.message_amount <= 0:
            return
        start = self.heads["MESSAGE_START"]
        rows = np.tanh(Y[start:start + cfg.message_amount, :])
        for j, v in enumerate(targets):
            self.messages.setdefault(u, {})[int(v)] = rows[:, j].astype(float).tolist()

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
        rewires: List[Tuple[int, int, int]] = []
        rewire_records: List[Dict[str, int]] = []
        applied_rewires = 0
        # Captured before anything changes, so the viewer can express births and
        # deaths as a share of the population that actually faced this phase,
        # and show how much each agent gained or lost across it.
        nodes_before = self.G.number_of_nodes()
        tokens_before = dict(self.tokens)
        log_deg, neighs, q_tok, q_deg, log_tok = self._precompute_features()

        for u in list(self.G.nodes()):
            tokens_u = int(self.tokens.get(u, 0))
            if tokens_u <= 0:
                continue

            candidates = [u] + list(neighs[u])
            Y = self._observe(u, candidates, log_deg, q_tok, q_deg, log_tok)
            self._emit_messages(u, candidates, Y)

            # How much of myself do I give away? Averaged over the whole view.
            frac = np.mean(Y[self.heads["REPRO_FRACTION"], :], axis=1)
            child_tokens = int(np.floor(_share_of_first(frac[0], frac[1]) * tokens_u))
            child_tokens = max(0, min(tokens_u, child_tokens))

            # Rewiring is the agent's own business and does not depend on
            # whether it reproduced.
            move = self._choose_rewire(u, candidates, Y)
            if move is not None:
                rewires.append((u, move[0], move[1]))

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

        # Hand the rewired edges across, all against the graph as it stood
        # before any of them moved.
        #
        # Applying them one at a time made the outcome depend on the order
        # agents happened to be visited in, because both ends of an edge can
        # choose to give that same edge away: whoever was reached first won,
        # and the other found its edge gone and silently did nothing. That is
        # not a rare collision. Over twenty-five phases every single one ended
        # with a different graph depending on the order, from 570 edges claimed
        # from both ends.
        #
        # So every claim is judged against the unchanged graph, and a contested
        # edge is settled by drawing one claim rather than by whichever agent
        # sorts first — an id-based rule would hand the edge to the older agent
        # every time. Sorting before the draw keeps it reproducible from the
        # seed without letting node order decide anything.
        claims: Dict[frozenset, List[Tuple[int, int, int]]] = {}
        for agent, old_v, recipient in rewires:
            if recipient == old_v:
                continue
            if not self.G.has_edge(agent, old_v):
                continue
            if not (self.G.has_node(recipient) and self.G.has_node(old_v)):
                continue
            claims.setdefault(frozenset((agent, old_v)), []).append(
                (int(agent), int(old_v), int(recipient)))

        removals: List[Tuple[int, int]] = []
        additions: List[Tuple[int, int]] = []
        for key in sorted(claims, key=lambda k: tuple(sorted(k))):
            proposals = sorted(claims[key])
            chosen = (proposals[0] if len(proposals) == 1
                      else proposals[int(np.random.randint(len(proposals)))])
            agent, old_v, recipient = chosen
            removals.append((agent, old_v))
            additions.append((recipient, old_v))
            applied_rewires += 1
            if record_decisions:
                rewire_records.append(
                    {"agent": agent, "edge": old_v, "to": recipient})

        # Every removal first, then every addition. An edge handed to a node
        # that is itself giving one away therefore does not depend on which of
        # the two was looked at first. An addition landing on a pair the graph
        # already holds collapses into it, since a simple graph holds each pair
        # once — which is why a rewire can lower the edge count but never raise
        # it: one edge leaves for every one that arrives, and arrivals can
        # merge.
        self.G.remove_edges_from(removals)
        self.G.add_edges_from(additions)

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
        cleanup = self._cleanup_and_redistribute()

        payload = None
        if record_decisions:
            payload = {"births": decisions}
            # Absent rather than empty when the rule is off, so the viewer can
            # tell that apart from nobody choosing to move an edge.
            if self.cfg.allow_rewire:
                payload["rewires"] = rewire_records

        extra = {"rewires": applied_rewires} if self.cfg.allow_rewire else None
        return self._frame(phase=1, cleanup=cleanup, nodes_before=nodes_before,
                           tokens_before=tokens_before, decisions=payload,
                           summary_extra=extra)

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

    def _choose_rewire(self, u: int, candidates: List[int], Y: np.ndarray):
        """
        Hand one of this agent's own edges to one of its other neighbours.

        The same shape as handover, but sideways instead of downward: where a
        parent gives a connection to its newborn, here an agent gives one to a
        peer. The edge (u, old) becomes (recipient, old) — this agent drops out
        of the middle and the two are left joined to each other.

        Three choices, read the way every other decision is. Whether to do it at
        all, from a yes/no pair with its own mode pair. Which edge to give away,
        scored per neighbour. Which neighbour receives it, scored per neighbour
        again. A second mode pair says whether those two picks are sampled from
        the scores or taken at their maximum.

        The original script scored exactly these two things and then applied
        them the other way round, joining the agent to the target rather than
        the target to the dropped edge. Since both were drawn from the agent's
        own neighbours, the edge it "reconnected" to already existed and adding
        it did nothing — the whole rule reduced to deleting an edge.

        Returns (edge_to_give_away, recipient) or None.
        """
        if not self.cfg.allow_rewire:
            return None

        neighbours = candidates[1:]
        # Nothing to do without two distinct neighbours: one to give, one to
        # give to.
        if len(neighbours) < 2:
            return None

        heads = self.heads
        # Averaged over the neighbour columns: whether to rewire is one decision
        # about the agent, not one per edge.
        yn = np.mean(Y[heads["REWIRE"], 1:], axis=1)
        mode = np.mean(Y[heads["REWIRE_MODE"], 1:], axis=1)
        if not _choose_binary(yn[0], yn[1], mode[0], mode[1]):
            return None

        pick_mode = np.mean(Y[heads["REWIRE_PICK_MODE"], 1:], axis=1)
        sample = bool(pick_mode[0] > pick_mode[1])

        old_v = neighbours[_pick_index(np.asarray(Y[heads["REWIRE_DROP"], 1:], dtype=float), sample)]
        recipient = neighbours[_pick_index(np.asarray(Y[heads["REWIRE_TO"], 1:], dtype=float), sample)]

        # Giving an edge to the very node on the other end of it would leave a
        # loop from that node to itself.
        if recipient == old_v:
            return None
        return int(old_v), int(recipient)

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
        log_deg, neighs, q_tok, q_deg, log_tok = self._precompute_features()

        # --- 1. Message pass, so allocation decisions see fresh signals -------
        for u in list(self.G.nodes()):
            targets = [u] + list(neighs[u])
            Y = self._observe(u, targets, log_deg, q_tok, q_deg, log_tok)
            self._emit_messages(u, targets, Y)

        # --- 2. One-shot allocation ------------------------------------------
        allocations_to: Dict[int, Dict[int, int]] = {v: {} for v in self.G.nodes()}
        revolution_to: Dict[int, Dict[int, int]] = {v: {} for v in self.G.nodes()}
        incoming_totals: Dict[int, int] = {v: 0 for v in self.G.nodes()}
        edge_flow: Dict[Tuple[int, int], int] = {tuple(sorted(e)): 0 for e in self.G.edges()}
        alloc_records: List[Dict[str, Any]] = []

        for u in list(self.G.nodes()):
            tokens_u = int(self.tokens.get(u, 0))
            if tokens_u <= 0:
                continue

            targets = [u] + list(neighs[u])
            Y = self._observe(u, targets, log_deg, q_tok, q_deg, log_tok)

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

            if record_decisions:
                entry = {
                    "node": int(v),
                    "winner": int(winner),
                    "amount": int(max_amount),
                }
                if self.cfg.allow_revolutions:
                    entry["revolt"] = int(by_revolt)
                winners.append(entry)

        self.tokens = new_tokens
        self.brains = new_brains

        # --- 4. Aftermath -----------------------------------------------------
        for brain in self.brains.values():
            self._mutate_brain(brain)

        dead_edges = [e for e, flow in edge_flow.items() if flow == 0]
        if dead_edges:
            self.G.remove_edges_from(dead_edges)

        cleanup = self._cleanup_and_redistribute()
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

        n_layers = len(self.brains[nodes[0]].weights) if nodes else 0
        blob["n_layers"] = np.array([n_layers], dtype=np.int64)
        for layer in range(n_layers):
            blob[f"W{layer}"] = np.stack([self.brains[u].weights[layer] for u in nodes])
            blob[f"b{layer}"] = np.stack([self.brains[u].biases[layer] for u in nodes])

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

        for i, u in enumerate(ids):
            brain = Brain(cfg, int(brain_ids[i]), allocate=False)
            brain.parent_brain_id = int(parent_brain_ids[i])
            brain.weights = [w[i].copy() for w in weights]
            brain.biases = [b[i].copy() for b in biases]

            world.brains[u] = brain
            world.tokens[u] = int(tokens[i])
            world.messages[u] = {}
            world.parent_of[u] = int(parent_ids[i])

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

    for t in range(cfg.max_steps):
        world.step(record_decisions=False)
        print(f"iteration {t}: nodes={world.G.number_of_nodes()} "
              f"edges={world.G.number_of_edges()} tokens={sum(world.tokens.values())}")
        if world.is_extinct():
            print("⚠️ Extinction.")
            break


if __name__ == "__main__":
    _main()
