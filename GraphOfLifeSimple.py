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


HOW ONE STEP WORKS
------------------
Each call to `step(t)` runs two phases in order.

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
     across the survivors (GLOBAL_UNIFORM), keeping the global count conserved.
  4. If the world empties completely, a single fresh agent is resurrected
     holding every token.

THE OUTER LOOP
  Runs until the graph falls to <= 50 nodes, calls that an extinction, and
  restarts from a fresh random graph. Forever.

Every half-step is written to disk as replayable JSON.


AGENT-CONTROLLED RANDOMNESS
---------------------------
There is no global "be probabilistic" switch. Each discrete decision is paired
with a MODE head, and the agent decides for itself whether that decision is
read as a probability or as a hard maximum. The choice is part of the genome
and therefore evolves.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from datetime import datetime
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Total amount of tokens (conserved across the whole simulation).
TOTAL_TOKENS: int = 150_000

# Tokens injected into the world each cleanup. 0 keeps the economy closed.
CREATE_X_NEW_TOKENS_EACH_PHASE: int = 0

# ---- Mutation ----
MUTATION_PROBABILITY: float = 0.5
MUTATION_NOISE_STD: float = 0.2
MUTATION_SPARSITY: float = 0.1

# ---- Sensory input ----
# Uniform(-2, 2) values appended to every observation, for symmetry breaking.
RANDOM_INPUT_AMOUNT: int = 5

# ---- Messaging ----
EXCHANGE_MESSAGES: bool = True
MESSAGE_NUMBER_AMOUNT: int = 5

# ---- Brain architecture ----
BRAIN_HIDDEN_LAYERS: List[int] = [50, 45, 40, 35, 30]

# ---- Visualization ----
DRAW: bool = True
DRAW_EVERY_X_ITERATIONS: int = 50

# ---- Output ----
BASE_DIR = os.path.join(os.path.dirname(__file__), "GraphOfLifeOutputs")
os.makedirs(BASE_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Brain input / output layout
# ----------------------------------------------------------------------------
#
# INPUT (54 values), built per (observer u, candidate v) pair:
#     1   is this candidate myself?
#     4   log tokens of u and v, log degree of u and v
#    24   six-quantile summaries of u's and v's neighborhoods (tokens, degrees)
#    20   four message vectors: u->u, u->v, v->u, v->v
#     5   uniform noise
#
N_BASE_INPUTS = 29  # 1 + 4 + 24
N_INPUTS = N_BASE_INPUTS + 4 * MESSAGE_NUMBER_AMOUNT + RANDOM_INPUT_AMOUNT

# OUTPUT (16 values), one column per observed candidate.
HEAD = {
    # Fraction of my tokens to invest into a newborn child.
    "REPRO_FRACTION": slice(0, 2),
    # Should the newborn be linked to this candidate? (yes, no)
    "LINK": slice(2, 4),
    # Read LINK as a probability, or as a hard maximum? (yes, no)
    "LINK_MODE": slice(4, 6),
    # Desirability of allocating tokens to this candidate.
    "BLOTTO": 6,
    # Spread allocation proportionally, or go all-in on the best target?
    "BLOTTO_MODE": slice(7, 9),
    # Portion of the tokens sent here that count as revolution tokens.
    "REV_FRACTION": slice(9, 11),
    # Signal broadcast to this candidate.
    "MESSAGE": slice(11, 11 + MESSAGE_NUMBER_AMOUNT),
}
N_OUTPUTS = 11 + MESSAGE_NUMBER_AMOUNT


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

    It is evaluated on a whole candidate set at once: pass a matrix of shape
    (N_INPUTS, n_candidates) and get back (N_OUTPUTS, n_candidates), so one
    forward pass scores an entire neighborhood.
    """

    _next_brain_id = 1
    rec = None  # optional sink for genotype lineage events

    def __init__(self) -> None:
        self.layer_sizes = [N_INPUTS] + list(BRAIN_HIDDEN_LAYERS) + [N_OUTPUTS]

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.weights.append(np.random.normal(0.0, 1.0 / np.sqrt(fan_in), size=(fan_out, fan_in)))
            self.biases.append(np.zeros((fan_out, 1)))

        self.brain_id: int = Brain._next_brain_id
        self.parent_brain_id: int | None = None
        Brain._next_brain_id += 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Always returns a 2-D array of shape (N_OUTPUTS, n_candidates)."""
        a = np.asarray(x, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ a + b
            is_last = i == len(self.weights) - 1
            a = z if is_last else _sigmoid(z)
        return a

    def copy(self) -> "Brain":
        clone = Brain()
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = [b.copy() for b in self.biases]
        clone.parent_brain_id = self.brain_id
        if Brain.rec:
            Brain.rec({"t": "copy", "from": int(self.brain_id), "to": int(clone.brain_id)})
        return clone

    def mutate(self) -> None:
        """Sparse Gaussian perturbation, plus an occasional structural reset."""
        if np.random.random() > MUTATION_PROBABILITY:
            return

        old_id = self.brain_id
        reset_fraction = float(np.clip(MUTATION_SPARSITY, 0.0, 1.0))

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            base_scale = 1.0 / np.sqrt(W.shape[1])
            std = MUTATION_NOISE_STD * base_scale

            self.weights[i] = self._perturb(W, std, base_scale, reset_fraction)
            self.biases[i] = self._perturb(b, std, std if std > 0 else 0.01, reset_fraction)

        self.parent_brain_id = old_id
        self.brain_id = Brain._next_brain_id
        Brain._next_brain_id += 1
        if Brain.rec:
            Brain.rec({"t": "mut", "from": int(old_id), "to": int(self.brain_id)})

    @staticmethod
    def _perturb(M: np.ndarray, noise_std: float, reset_std: float, reset_fraction: float) -> np.ndarray:
        """Jitter a sparse subset of entries, then rarely re-draw some outright."""
        if MUTATION_SPARSITY <= 0.0:
            return M

        if noise_std > 0.0:
            jitter_mask = np.random.random(M.shape) < MUTATION_SPARSITY
            M = M + np.random.normal(0.0, noise_std, size=M.shape) * jitter_mask

        if reset_fraction > 0.0 and np.random.random() < MUTATION_SPARSITY:
            reset_mask = np.random.random(M.shape) < reset_fraction
            if np.any(reset_mask):
                M = np.where(reset_mask, np.random.normal(0.0, reset_std, size=M.shape), M)

        return M.astype(float, copy=False)


# ----------------------------------------------------------------------------
# The Arena
# ----------------------------------------------------------------------------

class GraphOfLife:
    def __init__(self, G_init: nx.Graph, total_tokens: int) -> None:
        # Relabel the seed graph to dense integer agent ids.
        self.G = nx.Graph()
        self.next_agent_id = 0
        old2new: Dict[Any, int] = {}
        for n in G_init.nodes():
            old2new[n] = self.next_agent_id
            self.G.add_node(self.next_agent_id)
            self.next_agent_id += 1
        for u, v in G_init.edges():
            self.G.add_edge(old2new[u], old2new[v])

        # Agent state.
        self.total_tokens = int(total_tokens)
        self.brains: Dict[int, Brain] = {aid: Brain() for aid in self.G.nodes()}
        self.messages: Dict[int, Dict[int, List[float]]] = {aid: {} for aid in self.G.nodes()}

        share = self.total_tokens // self.G.number_of_nodes()
        self.tokens: Dict[int, int] = {aid: share for aid in self.G.nodes()}

        self.run_dir = self._make_run_dir()
        self._snapshot_source()
        self._save_configuration()

        self.genotype_events: List[Dict[str, int]] = []
        Brain.rec = self.genotype_events.append

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

    def _input_vec(
            self,
            u: int,
            v: int,
            log_deg: Dict[int, float],
            q_tok: Dict[int, List[float]],
            q_deg: Dict[int, List[float]],
            log_tok: Dict[int, float],
    ) -> np.ndarray:
        """Assemble the 54-value sensory vector for observer `u` looking at `v`."""
        base = (
            [log_tok.get(u, 0.0), log_tok.get(v, 0.0), log_deg[u], log_deg[v]]
            + q_tok[u] + q_tok[v] + q_deg[u] + q_deg[v]
        )

        M = MESSAGE_NUMBER_AMOUNT

        def msg(src: int, dst: int) -> List[float]:
            vec = self.messages.get(src, {}).get(dst)
            if vec is None:
                return [0.0] * M
            out = list(vec[:M])
            return out + [0.0] * (M - len(out))

        msg_feats = msg(u, u) + msg(u, v) + msg(v, u) + msg(v, v)
        noise = np.random.uniform(-2.0, 2.0, size=RANDOM_INPUT_AMOUNT).tolist()

        return np.array([int(u == v)] + base + msg_feats + noise, dtype=float)

    def _observe(
            self,
            u: int,
            candidates: List[int],
            log_deg: Dict[int, float],
            q_tok: Dict[int, List[float]],
            q_deg: Dict[int, List[float]],
            log_tok: Dict[int, float],
    ) -> np.ndarray:
        """One forward pass scoring every candidate. Returns (N_OUTPUTS, n_candidates)."""
        X = np.column_stack([
            self._input_vec(u, v, log_deg, q_tok, q_deg, log_tok) for v in candidates
        ])
        return self.brains[u].forward(X)

    def _emit_messages(self, u: int, targets: List[int], Y: np.ndarray) -> None:
        """Broadcast `u`'s message head to each observed target."""
        if not EXCHANGE_MESSAGES:
            return
        msg_rows = np.tanh(Y[HEAD["MESSAGE"], :])
        for j, v in enumerate(targets):
            self.messages.setdefault(u, {})[int(v)] = msg_rows[:, j].astype(float).tolist()

    # ------------------------------------------------------------------------
    # Phase 1: Reproduction
    # ------------------------------------------------------------------------

    def reproduction_phase(self, t: int) -> str:
        """
        Agents spend their own tokens to spawn children and choose the newborn's
        connections. No other topology change happens here.
        """
        log: Dict[str, Any] = {
            "phase": "reproduction",
            "pre_state": self._snapshot_graph(),
            "decisions": [],
        }

        log_deg, neighs, q_tok, q_deg, log_tok = self._precompute_features()

        for u in list(self.G.nodes()):
            tokens_u = int(self.tokens.get(u, 0))
            if tokens_u <= 0:
                continue

            candidates = [u] + list(neighs[u])
            Y = self._observe(u, candidates, log_deg, q_tok, q_deg, log_tok)
            self._emit_messages(u, candidates, Y)

            # How much of myself do I give away? Averaged over the whole view.
            frac_logits = np.mean(Y[HEAD["REPRO_FRACTION"], :], axis=1)
            child_tokens = int(np.floor(_share_of_first(frac_logits[0], frac_logits[1]) * tokens_u))
            child_tokens = max(0, min(tokens_u, child_tokens))

            record: Dict[str, Any] = {
                "agent_id": int(u),
                "tokens_before": tokens_u,
                "repro_tokens": child_tokens,
                "child_created": False,
                "link_choices": [],
            }

            if child_tokens >= 1:
                child_id = self._spawn_child(u, tokens_u, child_tokens, candidates, Y, record)
                record.update({"child_created": True, "child_id": int(child_id)})

            log["decisions"].append(record)

        self.G.remove_edges_from(list(nx.selfloop_edges(self.G)))

        log["cleanup"] = self._cleanup_and_redistribute()
        log["post_state"] = self._snapshot_graph()
        log["genotype_events"] = list(self.genotype_events)
        self.genotype_events.clear()

        if DRAW and t % DRAW_EVERY_X_ITERATIONS == 0:
            self._draw(f"Round {t} — After Phase 1", f"step_{2 * t:05d}_phase1.png")

        return self._save_step_file(2 * t, log)

    def _spawn_child(
            self,
            parent: int,
            parent_tokens: int,
            child_tokens: int,
            candidates: List[int],
            Y: np.ndarray,
            record: Dict[str, Any],
    ) -> int:
        """Create the newborn, wire it up, and debit the parent."""
        child_id = self.next_agent_id
        self.next_agent_id += 1
        self.G.add_node(child_id)

        # The child inherits a mutated copy; the parent pays the full price.
        self.brains[child_id] = self.brains[parent].copy()
        self.brains[child_id].mutate()
        self.tokens[child_id] = child_tokens
        self.tokens[parent] = parent_tokens - child_tokens
        self.messages[child_id] = {}

        link_logits = Y[HEAD["LINK"], :]
        link_mode = Y[HEAD["LINK_MODE"], :]

        for col, v in enumerate(candidates):
            linked = _choose_binary(
                link_logits[0, col], link_logits[1, col],
                link_mode[0, col], link_mode[1, col],
            )
            record["link_choices"].append({"candidate": int(v), "chosen": linked})
            if linked and v != child_id and self.G.has_node(v):
                self.G.add_edge(child_id, v)

        return child_id

    # ------------------------------------------------------------------------
    # Phase 2: Blotto
    # ------------------------------------------------------------------------

    def blotto_phase(self, t: int) -> str:
        """
        Every agent spends its entire token pool bidding on itself and its
        neighbors. The winner of each node implants its brain there.
        """
        log: Dict[str, Any] = {
            "phase": "blotto",
            "pre_state": self._snapshot_graph(),
            "allocations": [],
            "winners": {},
            "pruned_edges": [],
        }
        tokens_before = dict(self.tokens)

        log_deg, neighs, q_tok, q_deg, log_tok = self._precompute_features()

        # --- 1. Message pass, so allocation decisions see fresh signals -------
        for u in list(self.G.nodes()):
            targets = [u] + list(neighs[u])
            Y = self._observe(u, targets, log_deg, q_tok, q_deg, log_tok)
            self._emit_messages(u, targets, Y)

        # --- 2. One-shot allocation ------------------------------------------
        # How much each agent sent to each target, and how much of that was
        # flagged as revolutionary.
        allocations_to: Dict[int, Dict[int, int]] = {v: {} for v in self.G.nodes()}
        revolution_to: Dict[int, Dict[int, int]] = {v: {} for v in self.G.nodes()}
        incoming_totals: Dict[int, int] = {v: 0 for v in self.G.nodes()}
        edge_flow: Dict[Tuple[int, int], int] = {tuple(sorted(e)): 0 for e in self.G.edges()}

        for u in list(self.G.nodes()):
            tokens_u = int(self.tokens.get(u, 0))
            if tokens_u <= 0:
                continue

            targets = [u] + list(neighs[u])
            Y = self._observe(u, targets, log_deg, q_tok, q_deg, log_tok)

            scores = np.asarray(Y[HEAD["BLOTTO"], :], dtype=float)
            mode_logits = np.mean(Y[HEAD["BLOTTO_MODE"], :], axis=1)

            # The agent picks its own doctrine: spread by score, or all-in.
            # (Drop this branch and always call _apportion for pure spreading.)
            if mode_logits[0] > mode_logits[1]:
                alloc = _apportion(scores, tokens_u)
            else:
                alloc = np.zeros(len(targets), dtype=int)
                alloc[int(np.argmax(scores))] = tokens_u

            rev_logits = Y[HEAD["REV_FRACTION"], :]

            for idx, v in enumerate(targets):
                amount = int(alloc[idx])
                if amount <= 0:
                    continue

                incoming_totals[v] += amount
                allocations_to[v][u] = allocations_to[v].get(u, 0) + amount

                # Only part of what I send here needs to be revolutionary.
                rev_share = _share_of_first(rev_logits[0, idx], rev_logits[1, idx])
                rev_amount = int(np.floor(rev_share * amount))
                if rev_amount > 0:
                    revolution_to[v][u] = revolution_to[v].get(u, 0) + rev_amount

                if u != v:
                    edge = tuple(sorted((u, v)))
                    if edge in edge_flow:
                        edge_flow[edge] += amount

            log["allocations"].append({
                "agent_id": int(u),
                "tokens_before": int(tokens_before.get(u, 0)),
                "targets": [int(v) for v in targets],
                "alloc": [int(a) for a in alloc],
            })

        # --- 3. Resolve every contested node ---------------------------------
        new_tokens = dict(self.tokens)
        new_brains = dict(self.brains)

        for v in list(self.G.nodes()):
            offers = allocations_to[v]
            if not offers:
                # Nobody wanted this node, not even itself. It starves, but its
                # lineage gets one last copy before cleanup decides its fate.
                new_tokens[v] = 0
                new_brains[v] = self.brains[v].copy()
                continue

            winner, max_amount = self._resolve_winner(offers, revolution_to[v])
            new_brains[v] = self.brains[winner].copy()
            new_tokens[v] = int(incoming_totals[v])
            log["winners"][str(v)] = {"winner": int(winner), "max_amount": int(max_amount)}

        self.tokens = new_tokens
        self.brains = new_brains

        # --- 4. Aftermath -----------------------------------------------------
        for brain in self.brains.values():
            brain.mutate()

        dead_edges = [e for e, flow in edge_flow.items() if flow == 0]
        if dead_edges:
            self.G.remove_edges_from(dead_edges)
        log["pruned_edges"] = [(int(u), int(v)) for (u, v) in dead_edges]

        log["cleanup"] = self._cleanup_and_redistribute()
        log["post_state"] = self._snapshot_graph()

        self._prune_stale_messages()

        log["genotype_events"] = list(self.genotype_events)
        self.genotype_events.clear()

        if DRAW and t % DRAW_EVERY_X_ITERATIONS == 0:
            self._draw(f"Round {t} — After Phase 2", f"step_{2 * t + 1:05d}_phase2.png")

        return self._save_step_file(2 * t + 1, log)

    @staticmethod
    def _resolve_winner(offers: Dict[int, int], revolutionaries: Dict[int, int]) -> Tuple[int, int]:
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

        Returns (winner_id, hegemon's allocation).
        """
        max_amount = max(offers.values())
        hegemon = int(np.random.choice([a for a, amt in offers.items() if amt == max_amount]))

        mob = [(agent, tokens) for agent, tokens in revolutionaries.items() if agent != hegemon]
        if not mob:
            return hegemon, max_amount

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
                return int(np.random.choice(rung)), max_amount

        # The mutiny never reached critical mass.
        return hegemon, max_amount

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
            "resurrect_agent": None,
            "removed_zero_nodes": [],
            "removed_components": [],
            "redistributed_tokens": 0,
            "survivors_count": 0,
        }

        # --- Who dies? --------------------------------------------------------
        starved = [u for u in self.G.nodes() if self.tokens.get(u, 0) <= 0]
        report["removed_zero_nodes"] = [int(u) for u in starved]

        G_active = self.G.copy()
        G_active.remove_nodes_from(starved)

        orphaned: set[int] = set()
        if G_active.number_of_nodes() > 0:
            components = sorted(nx.connected_components(G_active), key=len, reverse=True)
            for c in components[1:]:
                orphaned.update(c)
            report["removed_components"] = [list(map(int, c)) for c in components[1:]]

        doomed = set(starved) | orphaned

        # --- Collect the estate ----------------------------------------------
        global_pool = CREATE_X_NEW_TOKENS_EACH_PHASE
        for u in doomed:
            global_pool += max(0, self.tokens.get(u, 0))

        if doomed:
            self.G.remove_nodes_from(list(doomed))
            for u in doomed:
                self.tokens.pop(u, None)
                self.brains.pop(u, None)
                self.messages.pop(u, None)

        # --- Manna from heaven ------------------------------------------------
        survivors = list(self.G.nodes())
        if global_pool > 0 and survivors:
            # Multinomial keeps the token count exactly conserved.
            draws = np.random.multinomial(global_pool, [1 / len(survivors)] * len(survivors))
            for u, extra in zip(survivors, draws):
                self.tokens[u] = self.tokens.get(u, 0) + int(extra)

        report["redistributed_tokens"] = int(global_pool)
        report["survivors_count"] = self.G.number_of_nodes()

        # --- Resurrection -----------------------------------------------------
        if self.G.number_of_nodes() == 0:
            aid = self.next_agent_id
            self.next_agent_id += 1
            self.G.add_node(aid)
            self.tokens = {aid: self.total_tokens}
            self.brains = {aid: Brain()}
            self.messages = {aid: {}}
            report.update({"resurrected": True, "resurrect_agent": int(aid), "survivors_count": 1})

        return report

    def _prune_stale_messages(self) -> None:
        """Forget messages to or from anyone who is no longer a neighbor."""
        for u in list(self.messages.keys()):
            if not self.G.has_node(u):
                self.messages.pop(u, None)
                continue
            allowed = {u} | {int(w) for w in self.G.neighbors(u)}
            self.messages[u] = {
                v: vec for v, vec in self.messages[u].items() if v in allowed
            }

    def step(self, t: int) -> Tuple[str, str]:
        return self.reproduction_phase(t), self.blotto_phase(t)

    # ------------------------------------------------------------------------
    # Persistence & visualization
    # ------------------------------------------------------------------------

    def _make_run_dir(self) -> str:
        prefix = f"GOLS_{datetime.now().strftime('%Y_%m_%d')}__"
        existing = [
            int(name[len(prefix):])
            for name in os.listdir(BASE_DIR)
            if name.startswith(prefix)
            and os.path.isdir(os.path.join(BASE_DIR, name))
            and name[len(prefix):].isdigit()
            and len(name[len(prefix):]) == 3
        ]
        run_dir = os.path.join(BASE_DIR, f"{prefix}{(max(existing) + 1) if existing else 1:03d}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _snapshot_graph(self) -> Dict[str, Any]:
        """Capture the full world state for replay."""
        return {
            "nodes": [
                {
                    "agent_id": int(u),
                    "tokens": int(self.tokens.get(u, 0)),
                    "brain_id": int(self.brains[u].brain_id),
                    "neighbors": [int(v) for v in self.G.neighbors(u)],
                }
                for u in self.G.nodes()
            ],
            "edges": [(int(u), int(v)) for u, v in self.G.edges()],
        }

    def _save_step_file(self, idx: int, blob: Dict[str, Any]) -> str:
        path = os.path.join(self.run_dir, f"step_{idx:05d}.json")
        with open(path, "w") as f:
            json.dump(blob, f, indent=2)
        return path

    def _snapshot_source(self) -> None:
        """Keep a copy of the exact code that produced this run."""
        try:
            src = os.path.abspath(__file__)
            shutil.copy2(src, os.path.join(self.run_dir, os.path.basename(src)))
        except OSError:
            pass

    def _save_configuration(self) -> None:
        with open(os.path.join(self.run_dir, "config.txt"), "w") as f:
            for k, v in globals().items():
                if k.isupper() and not k.startswith("_"):
                    f.write(f"{k}: {v}\n")

    def _draw(self, title: str, fname: str, k_max: int = 3) -> None:
        """Render the graph, fading out nodes in the shallower k-cores."""
        if self.G.number_of_nodes() == 0:
            return

        pos3d = nx.spring_layout(self.G, dim=3, seed=42)
        pos2d = {n: (c[0], c[1]) for n, c in pos3d.items()}

        coreness = nx.core_number(self.G) if self.G.number_of_edges() > 0 else {u: 0 for u in self.G.nodes()}
        depth = {u: k_max - min(coreness.get(u, 0), k_max) for u in self.G.nodes()}

        nodes_by_depth: Dict[int, List[int]] = {}
        for u, d in depth.items():
            nodes_by_depth.setdefault(d, []).append(u)

        edges_by_depth: Dict[int, List[Tuple[int, int]]] = {}
        for u, v in self.G.edges():
            edges_by_depth.setdefault(max(depth[u], depth[v]), []).append((u, v))

        plt.figure(figsize=(8, 6))
        vmax = max([0] + [self.tokens.get(u, 0) for u in self.G.nodes()])

        for d in sorted(edges_by_depth, reverse=True):
            nx.draw_networkx_edges(
                self.G, pos2d, edgelist=edges_by_depth[d],
                alpha=0.5 / (2 ** d), width=0.5 if d == 0 else 0.3,
            )
        for d in sorted(nodes_by_depth, reverse=True):
            nlist = nodes_by_depth[d]
            nx.draw_networkx_nodes(
                self.G, pos2d, nodelist=nlist,
                node_size=[(self.tokens.get(u, 0) + 1) / 12 for u in nlist],
                node_color=[self.tokens.get(u, 0) for u in nlist],
                cmap=matplotlib.colormaps.get_cmap("viridis"),
                vmin=0, vmax=vmax, alpha=1.0 / (2 ** d),
            )

        plt.title(title)
        plt.axis("off")
        plt.savefig(os.path.join(self.run_dir, fname), dpi=130, bbox_inches="tight")
        plt.close("all")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def _main() -> None:
    max_steps = 500_000
    n = int(TOTAL_TOKENS / 100)
    k = max([int(n / 100), 5])

    def make_simulation() -> GraphOfLife:
        G0 = nx.watts_strogatz_graph(n=n, k=k, p=0.2)
        return GraphOfLife(G0, total_tokens=TOTAL_TOKENS)

    run_counter = 0
    while True:
        simulation = make_simulation()
        run_counter += 1
        print(f"🌍 Starting run {run_counter}, folder: {simulation.run_dir}")

        for t in range(max_steps):
            simulation.step(t)
            print(
                f"Run {run_counter}, iteration {t} finished "
                f"(nodes: {simulation.G.number_of_nodes()}, "
                f"edges: {simulation.G.number_of_edges()}, "
                f"tokens: {sum(simulation.tokens.values())})"
            )

            if simulation.G.number_of_nodes() <= 50:
                print(f"⚠️ Run {run_counter} extinction event. Restarting.")
                break


if __name__ == "__main__":
    _main()
