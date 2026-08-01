#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphOfLife — Open-Ended Evolution on a Mutable Graph
==============================================================================
TODO
More efficient blotto allocation. Simplfiy algorithm a bit.
Still allocate one at a time, but in each direction one can be allocated in one iteration. and if none is allocated at beginning in next iteration in this direction can still be allocated none
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from datetime import datetime
import hashlib
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ----------------------------------------------------------------------------
# Configuration & Constants
# ----------------------------------------------------------------------------

# Total amount of tokens (conserved)
TOTAL_TOKENS = 150_000

# ---- Token Creation ----
CREATE_X_NEW_TOKENS_EACH_PHASE: int = 0

# Determinism
PROBABILISTIC_DECISIONS: bool = False

# ---- Blotto rule variants ----
#   "ALLOCATE_AND_CONQUER"      : highest allocator at node v implants its brain into v
#   "ALLOCATE_AND_ROB"          : winner-for-v *collects* all tokens allocated to v; no brain copying
#   "ALLOCATE_AND_DO_NOTHING"   : No Brain gets overwritten. But they dont want the links to disappear

BLOTTO_MODE: str = "ALLOCATE_AND_CONQUER"

ALLOW_REVOLUTIONS = True
FORCE_REVOLUTIONS =  False

# ---- Blotto allocation scheduling ----
# "FULL_ALLOCATION"              : one observation; each agent allocates all its tokens at once by relative scores
# "STEP_ALLOCATION_WEAKEST_FIRST": (current behavior) every agent with >=1 token allocates 1 token each round
# "STEP_ALLOCATION_STRONGEST_FIRST": only agents with the current per-round max remaining tokens allocate 1 token
# "STEP_ALLOCATION_AS_WISHED": they can choose when to allocate

BLOTTO_ALLOCATION_MODE: str = "STEP_ALLOCATION_STRONGEST_FIRST"

# ---- Reproduction decision variant ----
REPRO_CORE_FRACTIONS_ONLY: bool = False

# ---- Mutation policy toggles ----
MUTATE_ON_REPRO_COPY: bool = False
MUTATE_ON_BLOTTO_COPY: bool = False
MUTATE_ALL_AFTER_BLOTTO: bool = True

# ---- Mutation Hyperparameters ----
MUTATION_PROBABILITY: float = 0.5
MUTATION_NOISE_STD: float = 0.2
MUTATION_SPARSITY: float = 0.1

# ---- Input Noise (New) ----
# Amount of random inputs (Uniform -2 to 2) added to every observation
RANDOM_INPUT_AMOUNT: int = 5

# ---- Walkermode ----
# PSEUDO_RANDOM_ONE_PER_ITERATION -> one per iteration
# PSEUDO_RANDOM_ONE_PER_TOKEN -> one per token
# PSEUDO_RANDOM_ONE_PER_TOKEN_LOG -> one per log2(token)+1
# PSEUDO_RANDOM_ONE_PER_TOKEN_RANDOM_WALK -> one per token random walk
# PSEUDO_RANDOM_ONE_PER_TOKEN_RANDOM_WALK_LOG -> one per log2(token)+1 token random walk
# WALK_ON_OWN_PER_TOKEN -> Active navigation driven by Brain, steps = tokens
# WALK_ON_OWN_PER_LOG_TOKEN -> Active navigation driven by Brain, steps = log2(tokens)+1
# NO_WALKER -> no walker

WALKER_MODE: str = "PSEUDO_RANDOM_ONE_PER_ITERATION"

# ---- Walker/reach reset policy ----
RESET_REACH_ON_CONQUER: bool = True

# ---- Exchange Messages ----
EXCHANGE_MESSAGES: bool = True
MESSAGE_NUMBER_AMOUNT = 5

# ---- Token Redistribution Physics----
# "GLOBAL_UNIFORM": Tokens from removed nodes are distributed to all survivors (Manna from Heaven).
# "LOCAL_SCAVENGING": Tokens from removed nodes go to surviving neighbors. Isolated tokens go global. #TODO wrongly implemented
TOKEN_REDISTRIBUTION_MODE: str = "GLOBAL_UNIFORM"

# ---- Brain Architecture ----
BRAIN_HIDDEN_LAYERS: List[int] = [50, 45, 40, 35, 30]

# Visualization
DRAW: bool = True
DRAW_EVERY_X_ITERATIONS = 50

# Output directory
BASE_DIR = os.path.join(os.path.dirname(__file__), "GraphOfLifeOutputs")
os.makedirs(BASE_DIR, exist_ok=True)

# Output Heads
HEAD = {
    "REPRO": slice(0, 4),
    "LINK": slice(4, 6),
    "SHIFT": slice(6, 8),
    "RECONNECT": slice(8, 12),
    "BLOTTO": 12,
    "WALKER": slice(13, 15),
    "MESSAGE": slice(15, 15 + MESSAGE_NUMBER_AMOUNT),
    "WALKER_DIR": 15 + MESSAGE_NUMBER_AMOUNT,
    "ALLOCATE_YN": slice(15 + MESSAGE_NUMBER_AMOUNT + 1, 15 + MESSAGE_NUMBER_AMOUNT + 3),
    "REVOLUTION_YN": slice(15 + MESSAGE_NUMBER_AMOUNT + 3, 15 + MESSAGE_NUMBER_AMOUNT + 5),
}


# ----------------------------------------------------------------------------
# Mathematical Utilities
# ----------------------------------------------------------------------------

def _six_quantiles(sorted_vals: List[float]) -> List[float]:
    """
    Compresses a distribution into 6 representative quantiles.
    Used to give the brain a summary of neighborhood statistics.
    """
    if not sorted_vals:
        return [0.0] * 6
    qs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    idx = [(len(sorted_vals) - 1) * q for q in qs]
    out: List[float] = []
    for x in idx:
        i0 = int(np.floor(x))
        i1 = int(np.ceil(x))
        if i0 == i1:
            out.append(float(sorted_vals[i0]))
        else:
            w = x - i0
            out.append(float((1 - w) * sorted_vals[i0] + w * sorted_vals[i1]))
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable Sigmoid activation."""
    pos = x >= 0
    neg = ~pos
    z = np.empty_like(x, dtype=float)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    z[neg] = ex / (1.0 + ex)
    return z


# ----------------------------------------------------------------------------
# The Brain (Neural Substrate)
# ----------------------------------------------------------------------------

class Brain:
    _next_brain_id = 1
    rec = None

    def __init__(self) -> None:
        """
        Initializes a Feed-Forward Neural Network.

        Input Layer Anatomy:
        - Base (29): Local topology and wealth (Log-Scaled).
        - Blotto Context (25): Competition metrics + Revolution Metrics.
        - Walker Context (4): Spatial-temporal navigation data.
        - Messages (20): Signals from neighbors.
        - Noise (4): Entropy injection.
        """
        # Base breakdown: 1 (obs_is_self) + 4 (scalars) + 24 (quantiles) = 29
        # Contexts: 25 (Blotto) + 4 (Walker) = 29
        # Total Static Inputs = 58
        n_inputs = 58 + 4 * MESSAGE_NUMBER_AMOUNT + RANDOM_INPUT_AMOUNT

        hidden_sizes = BRAIN_HIDDEN_LAYERS
        # Outputs: 15 (Standard) + Msg + 3 (WalkerDir/AllocYN) + 2 (RevYN)
        n_outputs = 15 + MESSAGE_NUMBER_AMOUNT + 3 + 2

        assert n_inputs > 0 and n_outputs > 0
        self.layer_sizes = [int(n_inputs)] + [int(h) for h in hidden_sizes] + [int(n_outputs)]

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            W = np.random.normal(0.0, 1.0 / np.sqrt(fan_in), size=(fan_out, fan_in))
            b = np.zeros((fan_out, 1))
            self.weights.append(W)
            self.biases.append(b)

        self.brain_id: int = Brain._next_brain_id
        self.parent_brain_id = None
        Brain._next_brain_id += 1

    def forward(self, x: np.ndarray | List[float]) -> np.ndarray:
        a = np.asarray(x, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        for li, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ a + b
            is_last = li == len(self.weights) - 1
            a = z if is_last else _sigmoid(z)
        return a.squeeze() if a.shape[1] == 1 else a

    def copy(self) -> "Brain":
        new_brain = Brain()
        new_brain.weights = [w.copy() for w in self.weights]
        new_brain.biases = [b.copy() for b in self.biases]
        new_brain.parent_brain_id = self.brain_id
        if Brain.rec: Brain.rec({"t": "copy", "from": int(self.brain_id), "to": int(new_brain.brain_id)})
        return new_brain

    def mutate(self) -> None:
        """Applies Gaussian noise and sparsity to weights."""
        if np.random.random() > MUTATION_PROBABILITY:
            return
        old_id = self.brain_id
        reset_fraction = float(np.clip(MUTATION_SPARSITY, 0.0, 1.0))

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            fan_in = W.shape[1]
            base_scale = 1.0 / np.sqrt(fan_in)
            w_std = float(MUTATION_NOISE_STD) * base_scale
            b_std = float(MUTATION_NOISE_STD) * base_scale

            # Weight Perturbation
            if MUTATION_SPARSITY > 0.0 and w_std > 0.0:
                mask = np.random.random(W.shape) < MUTATION_SPARSITY
                W = W + np.random.normal(0.0, w_std, size=W.shape) * mask

            # Structural Reset (Rarely replace weights entirely)
            if (MUTATION_SPARSITY > 0.0) and (reset_fraction > 0.0):
                if np.random.random() < MUTATION_SPARSITY:
                    reset_mask = np.random.random(W.shape) < reset_fraction
                    if np.any(reset_mask):
                        W_new = np.random.normal(0.0, base_scale, size=W.shape)
                        W = np.where(reset_mask, W_new, W)
            self.weights[i] = W.astype(float, copy=False)

            # Bias Perturbation
            if MUTATION_SPARSITY > 0.0 and b_std > 0.0:
                maskb = np.random.random(b.shape) < MUTATION_SPARSITY
                b = b + np.random.normal(0.0, b_std, size=b.shape) * maskb

            if (MUTATION_SPARSITY > 0.0) and (reset_fraction > 0.0):
                if np.random.random() < MUTATION_SPARSITY:
                    reset_maskb = np.random.random(b.shape) < reset_fraction
                    if np.any(reset_maskb):
                        b_new = np.random.normal(0.0, b_std if b_std > 0 else 0.01, size=b.shape)
                        b = np.where(reset_maskb, b_new, b)
            self.biases[i] = b.astype(float, copy=False)

        self.parent_brain_id = old_id
        self.brain_id = Brain._next_brain_id
        Brain._next_brain_id += 1
        if Brain.rec:
            Brain.rec({"t": "mut", "from": int(self.parent_brain_id), "to": int(self.brain_id)})


# ----------------------------------------------------------------------------
# The Arena (GraphOfLife)
# ----------------------------------------------------------------------------

class GraphOfLife:
    def __init__(self, G_init: nx.Graph, total_tokens: int) -> None:
        self.G = nx.Graph()
        self.next_agent_id = 0
        old2new: Dict[Any, int] = {}

        # Initialize Topology
        for n in G_init.nodes():
            aid = self.next_agent_id
            self.next_agent_id += 1
            old2new[n] = aid
            self.G.add_node(aid)
        for u, v in G_init.edges():
            self.G.add_edge(old2new[u], old2new[v])

        # Initialize State
        self.total_tokens = int(total_tokens)
        self.tokens: Dict[int, int] = {aid: 0 for aid in self.G.nodes()}
        self.brains: Dict[int, Brain] = {aid: Brain() for aid in self.G.nodes()}
        self.reach_counts: Dict[int, Dict[int, int]] = {aid: {aid: 1} for aid in self.G.nodes()}
        self.messages: Dict[int, Dict[int, List[float]]] = {aid: {} for aid in self.G.nodes()}

        # Distribute Initial Wealth
        N = self.G.number_of_nodes()
        for aid in self.G.nodes():
            self.tokens[aid] = int(self.total_tokens / N)

        # Setup Logging
        date_str = datetime.now().strftime("%Y_%m_%d")
        prefix = f"GOL_{date_str}__"
        existing_runs = []
        for name in os.listdir(BASE_DIR):
            full_path = os.path.join(BASE_DIR, name)
            if not os.path.isdir(full_path): continue
            if not name.startswith(prefix): continue
            suffix = name[len(prefix):]
            if len(suffix) == 3 and suffix.isdigit():
                existing_runs.append(int(suffix))
        next_idx = (max(existing_runs) + 1) if existing_runs else 1
        folder_name = f"{prefix}{next_idx:03d}"
        self.run_dir = os.path.join(BASE_DIR, folder_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self._snapshot_source()
        self._save_configuration()
        self.genotype_events: List[Dict[str, int]] = []
        Brain.rec = self.genotype_events.append

    def _neighbors(self, u: int) -> List[int]:
        return list(self.G.neighbors(u))

    def _precompute_features(self) -> Tuple[
        Dict[int, float],
        Dict[int, List[int]],
        Dict[int, List[float]],
        Dict[int, List[float]],
        Dict[int, float]
    ]:
        """
        Calculates the Global Sensory Manifold (O(N)).
        Crucially, applies Log-Norm scaling to all wealth and degree values.
        This allows the brain to perceive orders of magnitude differences.
        """
        # Log-Scale the Globals
        log_tokens = {u: np.log1p(max(0, val)) for u, val in self.tokens.items()}
        log_degrees = {u: np.log1p(float(self.G.degree[u])) for u in self.G.nodes()}

        neighs = {u: list(self.G.neighbors(u)) for u in self.G.nodes()}

        q_tok: Dict[int, List[float]] = {}
        q_deg: Dict[int, List[float]] = {}

        # Calculate Local Statistics
        for u, N in neighs.items():
            if not N:
                q_tok[u] = [0.0] * 6
                q_deg[u] = [0.0] * 6
                continue
            # Sort the already-logged values
            n_log_tok = sorted(log_tokens[n] for n in N)
            n_log_deg = sorted(log_degrees[n] for n in N)
            q_tok[u] = _six_quantiles(n_log_tok)
            q_deg[u] = _six_quantiles(n_log_deg)

        return log_degrees, neighs, q_tok, q_deg, log_tokens

    def _input_vec_fast(
            self,
            u: int,
            v: int,
            log_deg: Dict[int, float],
            q_tok: Dict[int, List[float]],
            q_deg: Dict[int, List[float]],
            log_tok_map: Dict[int, float],
            blotto_feats: List[float] | None = None,  # 25 dimensions now
            walker_feats: List[float] | None = None,  # 4 dimensions
            scale: float = 0.1,
    ) -> np.ndarray:
        """
        Constructs the high-dimensional sensory vector for the Brain.
        Enforces strict separation between Blotto and Walker contexts.
        """

        # 1. Base Topology (29 dims)
        own_obs = int(u == v)
        own_t_log = log_tok_map.get(u, 0.0)
        tgt_t_log = log_tok_map.get(v, 0.0)
        own_d_log = log_deg[u]
        tgt_d_log = log_deg[v]

        base = [own_t_log, tgt_t_log, own_d_log, tgt_d_log] + q_tok[u] + q_tok[v] + q_deg[u] + q_deg[v]

        # 2. Blotto Context (25 dims) - Competition Metrics + Revolution
        if blotto_feats is None:
            blotto_vec = [0.0] * 25
        else:
            blotto_vec = [np.log1p(max(0.0, float(x))) for x in blotto_feats]

        # 3. Walker Context (4 dims) - Spatial-Temporal Awareness
        if walker_feats is None:
            walker_vec = [0.0] * 4
        else:
            walker_vec = [np.log1p(max(0.0, float(x))) for x in walker_feats]

        # 4. Social Messaging
        M = MESSAGE_NUMBER_AMOUNT

        def _msg(src: int, dst: int) -> List[float]:
            vec = self.messages.get(src, {}).get(dst, None)
            if vec is None: return [0.0] * M
            out = list(vec[:M])
            if len(out) < M: out += [0.0] * (M - len(out))
            return out

        msg_feats = _msg(u, u) + _msg(u, v) + _msg(v, u) + _msg(v, v)

        # 5. Thermodynamic Noise (Symmetry Breaking)
        if RANDOM_INPUT_AMOUNT > 0:
            noise = np.random.uniform(-2.0, 2.0, size=RANDOM_INPUT_AMOUNT).tolist()
        else:
            noise = []

        # Concatenate all sensory streams
        return np.array([own_obs] + base + blotto_vec + walker_vec + msg_feats + noise, dtype=float)

    def _emit_messages(self, u: int, targets: List[int], Y: np.ndarray) -> None:
        """Propagates communication signals from the Brain to neighbors."""
        if EXCHANGE_MESSAGES:
            M = MESSAGE_NUMBER_AMOUNT
            msg_rows = Y[HEAD["MESSAGE"], :]
            if M == 1 and msg_rows.ndim == 1:
                msg_rows = np.array([msg_rows])
            msg_rows = np.tanh(msg_rows) #TODO why TAN? maybe sigmoid
            for j, v in enumerate(targets):
                #TODO dont understand what those do?
                col = msg_rows[:, j].astype(float).tolist()
                if len(col) < M:
                    col = col + [0.0] * (M - len(col))
                elif len(col) > M:
                    col = col[:M]
                self.messages.setdefault(u, {})[int(v)] = col

    # ------------------------------------------------------------------------
    # Phase 1: The Walker & Reproduction
    # ------------------------------------------------------------------------
    def _update_reach_counts_passive(self) -> None:
        """
        Restored legacy logic for passive diffusion and random walks.
        This handles all WALKER_MODE variants except the new 'WALK_ON_OWN'.
        """
        G = self.G
        new_maps: Dict[int, Dict[int, int]] = {}
        nodes_list = list(G.nodes())

        # --------------------------------------------------------------------
        # Mode 1: Persistent Diffusion (One step per iteration)
        # --------------------------------------------------------------------
        if WALKER_MODE == "PSEUDO_RANDOM_ONE_PER_ITERATION":
            for u in nodes_list:
                # 1. Prune dead nodes from history
                prev = self.reach_counts.get(u, {u: 1})
                prev = {w: c for w, c in prev.items() if G.has_node(w)}

                # 2. If empty after prune, re-init basic
                if not prev:
                    new_maps[u] = {u: 1}
                    continue

                # 3. Single diffusion step
                acc: Dict[int, int] = {}
                for r, c in prev.items():
                    # Flow to neighbors
                    for w in list(G.neighbors(r)):
                        acc[w] = acc.get(w, 0) + int(c)
                    # Stay (self-loop)
                    acc[r] = acc.get(r, 0) + int(c)
                new_maps[u] = acc or {u: 1}

            self.reach_counts = new_maps

        # --------------------------------------------------------------------
        # Mode 2: Multi-step Diffusion (Renormalized 'Soft Attention')
        # --------------------------------------------------------------------
        elif WALKER_MODE == "PSEUDO_RANDOM_ONE_PER_TOKEN" or WALKER_MODE == "PSEUDO_RANDOM_ONE_PER_TOKEN_LOG":
            # Pre-fetch neighbors to avoid repeated G.neighbors calls
            adj_cache = {u: list(G.neighbors(u)) for u in nodes_list}

            for u in nodes_list:
                # 1. Initialize with FLOATs to support continuous diffusion
                #    All core neighbors start with max intensity (1.0).
                current_counts: Dict[int, float] = {u: 1.0}
                for v in adj_cache[u]:
                    current_counts[v] = 1.0

                # 2. Determine horizon based on tokens
                tok_count = int(self.tokens.get(u, 0))
                if WALKER_MODE == "PSEUDO_RANDOM_ONE_PER_TOKEN_LOG":
                    steps = int(np.log2(tok_count)) + 1 if tok_count > 0 else 0
                else:
                    steps = tok_count

                # 3. Iterate diffusion 'steps' times
                for _ in range(steps):
                    next_counts: Dict[int, float] = {}

                    # Standard Diffusion: Mass at r spreads to r and neighbors
                    for r, count in current_counts.items():
                        # Optimization: Skip negligible contributions
                        if count < 1e-9:
                            continue

                        # Retrieve neighbors (safe fallback if node is new/distant)
                        r_neighbors = adj_cache.get(r)
                        if r_neighbors is None:
                            if G.has_node(r):
                                r_neighbors = list(G.neighbors(r))
                            else:
                                r_neighbors = []

                        # Add mass (accumulate)
                        # Self-loop
                        next_counts[r] = next_counts.get(r, 0.0) + count
                        # Neighbors
                        for w in r_neighbors:
                            next_counts[w] = next_counts.get(w, 0.0) + count

                    # --- RENORMALIZATION STEP ---
                    # Normalize so the 'hottest' node has value 1.0 to prevent overflow
                    if next_counts:
                        max_val = max(next_counts.values())
                        if max_val > 0:
                            scale = 1.0 / max_val
                            for k in next_counts:
                                next_counts[k] *= scale

                    current_counts = next_counts

                new_maps[u] = current_counts

            self.reach_counts = new_maps

        # --------------------------------------------------------------------
        # Mode 3: Monte Carlo Random Walk (Dirac Delta)
        # --------------------------------------------------------------------
        elif WALKER_MODE == "PSEUDO_RANDOM_ONE_PER_TOKEN_RANDOM_WALK" or WALKER_MODE == "PSEUDO_RANDOM_ONE_PER_TOKEN_RANDOM_WALK_LOG":
            # 1. Pre-fetch adjacency
            adj_cache = {n: list(G.neighbors(n)) for n in nodes_list}

            for u in nodes_list:
                # 2. Determine Walk Length
                tokens = int(self.tokens.get(u, 0))

                if WALKER_MODE == "PSEUDO_RANDOM_ONE_PER_TOKEN_RANDOM_WALK_LOG":
                    steps = int(np.log2(tokens)) + 2 if tokens > 0 else 1
                else:
                    steps = tokens + 1

                curr = u

                # 3. Perform the Monte Carlo Random Walk
                for _ in range(steps):
                    neighbors = adj_cache.get(curr, [])
                    if not neighbors:
                        break  # Trapped

                    # Uniform random selection of neighbor
                    idx = np.random.randint(len(neighbors))
                    curr = neighbors[idx]

                # 4. Record the final destination
                new_maps[u] = {curr: 1}

            self.reach_counts = new_maps

        # --------------------------------------------------------------------
        # Mode 4: Disabled
        # --------------------------------------------------------------------
        elif WALKER_MODE == "NO_WALKER":
            for u in nodes_list:
                new_maps[u] = {u: 1}
            self.reach_counts = new_maps

        else:
            # Fallback for unknown modes (or if ACTIVE mode is set but this was called by mistake)
            # Default to self-reach only to prevent crashes
            for u in nodes_list:
                new_maps[u] = {u: 1}
            self.reach_counts = new_maps

    def _perform_active_walk(
            self,
            u: int,
            log_deg: Dict[int, float],
            q_tok: Dict[int, List[float]],
            q_deg: Dict[int, List[float]],
            log_tok_map: Dict[int, float]
    ) -> List[int]:
        """
        The Homunculus: An active, brain-driven walker.
        The agent projects a viewpoint to traverse the graph, accumulating
        spatial-temporal context to decide where to form new links.
        """
        curr = u
        tokens = int(self.tokens.get(u, 0))

        # Calculate exploration budget (Log Wealth)
        if "LOG" in WALKER_MODE:
            total_steps = int(np.log2(tokens)) + 1 if tokens > 0 else 1
        else:
            total_steps = tokens

        # Distance helper (robust to disconnected components)
        def get_dist(source, target):
            if source == target: return 0
            try:
                return nx.shortest_path_length(self.G, source=source, target=target)
            except nx.NetworkXNoPath:
                return 9999.0

        for step_idx in range(total_steps):
            if not self.G.has_node(curr): break

            neighbors = list(self.G.neighbors(curr))
            candidates = [curr] + neighbors

            # --- Spatial-Temporal Context ---
            steps_taken = step_idx
            steps_left = total_steps - step_idx
            dist_from_home_current = get_dist(u, curr)

            X_cols = []
            for cand in candidates:
                if cand == curr:
                    dist_from_home_cand = dist_from_home_current
                else:
                    dist_from_home_cand = get_dist(u, cand)

                # Pack Context for the Brain
                # 1. Past Effort, 2. Future Budget, 3. Current Range, 4. Projected Range
                walker_context = [
                    float(steps_taken),
                    float(steps_left),
                    float(dist_from_home_current),
                    float(dist_from_home_cand)
                ]

                # Activate Walker Cortex (blotto_feats=None)
                vec = self._input_vec_fast(
                    curr, cand, log_deg, q_tok, q_deg, log_tok_map,
                    blotto_feats=None,
                    walker_feats=walker_context
                )
                X_cols.append(vec)

            X = np.column_stack(X_cols)
            Y = self.brains[u].forward(X)

            # Decision: Where to step next?
            dir_scores = Y[HEAD["WALKER_DIR"], :]

            if PROBABILISTIC_DECISIONS:
                vals = np.maximum(0.0, dir_scores)
                s = float(vals.sum())
                probs = (vals / s) if s > 0 else np.full_like(vals, 1.0 / len(vals))
                idx = int(np.random.choice(len(candidates), p=probs))
            else:
                idx = int(np.argmax(dir_scores))

            curr = candidates[idx]

        # Return the final destination found by the walker
        return [curr]

    def reproduction_phase(self, t: int) -> str:
        """
        Agents use their wealth to reproduce, modify topology, and create links
        to distant nodes discovered by the Active Walker.
        """
        log: Dict[str, Any] = {"phase": "reproduction", "pre_state": self._snapshot_graph(), "decisions": []}

        # O(N) Precomputation of Sensory Manifold
        log_deg, neighs, q_tok, q_deg, log_tok_map = self._precompute_features()

        # 1. The Walker (Discovery)
        walker_candidates_map: Dict[int, List[int]] = {}
        if "WALK_ON_OWN" in WALKER_MODE:
            for u in list(self.G.nodes()):
                if self.tokens.get(u, 0) > 0:
                    cands = self._perform_active_walk(u, log_deg, q_tok, q_deg, log_tok_map)
                    walker_candidates_map[u] = cands
        else:
            self._update_reach_counts_passive()
            for u in list(self.G.nodes()):
                rc_map = self.reach_counts.get(u, {u: 1})
                neighbors_u = set(neighs[u])
                rc_items_far = [
                    (v, c) for v, c in rc_map.items()
                    if v != u and v not in neighbors_u and self.G.has_node(v) and c > 0
                ]
                if rc_items_far:
                    nodes_far, weights_far = zip(*rc_items_far)
                    weights_arr = np.asarray(weights_far, dtype=float)
                    wsum = float(weights_arr.sum())
                    if wsum > 0.0:
                        probs_far = weights_arr / wsum
                        idx_far = int(np.random.choice(len(nodes_far), p=probs_far))
                        walker_candidates_map[u] = [int(nodes_far[idx_far])]

        # 2. Decision Making
        shifts_to_apply: List[Tuple[int, int, int]] = []
        reconns_to_apply: List[Tuple[int, int, int]] = []
        new_links_to_apply: List[Tuple[int, int]] = []

        for u in list(self.G.nodes()):
            t_u = int(self.tokens.get(u, 0))
            if t_u <= 0: continue

            # Candidates: Self + Neighbors + Walker Discovery
            neighbors_u = list(neighs[u])
            core_candidates = [u] + neighbors_u
            w_cands = walker_candidates_map.get(u, [])
            w_cands = [wc for wc in w_cands if self.G.has_node(wc)]
            print(f"should always be length 1: {len(w_cands)}")
            all_candidates = core_candidates + w_cands

            # Observe Environment (Standard Cortex, No Special Context)
            X_cols = [self._input_vec_fast(
                u, v, log_deg, q_tok, q_deg, log_tok_map,
                blotto_feats=None,
                walker_feats=None
            ) for v in all_candidates]

            X = np.column_stack(X_cols)
            Y = self.brains[u].forward(X)
            self._emit_messages(u, all_candidates, Y)

            # Parse Neural Outputs
            repro_logits_all = Y[HEAD["REPRO"], :]
            link_logits_all = Y[HEAD["LINK"], :]
            shift_logits_all = Y[HEAD["SHIFT"], :]
            reconn_logits_all = Y[HEAD["RECONNECT"], :]
            walker_logits_all = Y[HEAD["WALKER"], :]

            # A. Reproduction (Gate)
            repro_core = repro_logits_all[:, :len(core_candidates)]
            if REPRO_CORE_FRACTIONS_ONLY:
                # Direct investment calculation (Logic from Old Code)
                frac_vec = np.mean(repro_core[2:4, :], axis=1)
                vals = np.maximum(0.0, frac_vec)
                s = float(np.sum(vals))
                probs = (vals / s) if s > 0.0 else np.full_like(vals, 1.0 / len(vals))
                child_tokens = int(np.floor(probs[0] * t_u))
            else:
                # Binary Gate Decision (Logic from Old Code)
                yes_logit = float(np.mean(repro_core[0, :]))
                no_logit = float(np.mean(repro_core[1, :]))

                if PROBABILISTIC_DECISIONS:
                    y = max(0.0, yes_logit)
                    n = max(0.0, no_logit)
                    s = y + n
                    p_yes = (y / s) if s > 0.0 else 0.0
                    will_reproduce = (np.random.rand() < p_yes)
                else:
                    will_reproduce = (yes_logit > no_logit)

                if will_reproduce:
                    # Calculate Investment Fraction
                    frac_vec = np.mean(repro_core[2:4, :], axis=1)
                    vals = np.maximum(0.0, frac_vec)
                    s = float(np.sum(vals))
                    probs = (vals / s) if s > 0.0 else np.full_like(vals, 1.0 / len(vals))
                    child_tokens = int(np.floor(probs[0] * t_u))
                else:
                    child_tokens = 0

            child_tokens = max(0, min(int(t_u), int(child_tokens)))

            rec: Dict[str, Any] = {
                "agent_id": int(u), "tokens_before": int(t_u), "repro_tokens": int(child_tokens),
                "child_created": False, "link_choices": [], "shift_choices": [],
                "reconnect_choices": [], "walker_decisions": []
            }

            if child_tokens >= 1:
                # B. Create Child
                child_tokens = max(1, min(child_tokens, t_u))
                keep_tokens = t_u - child_tokens
                child_brain = self.brains[u].copy()
                if MUTATE_ON_REPRO_COPY: child_brain.mutate()

                cid = self.next_agent_id
                self.next_agent_id += 1
                self.G.add_node(cid)

                # Link Child to Neighbors
                chosen_links = []
                for col_idx, v in enumerate(core_candidates):
                    yes_l = float(link_logits_all[0, col_idx])
                    no_l = float(link_logits_all[1, col_idx])
                    choose = bool(yes_l > no_l)
                    if choose: chosen_links.append(v)
                    rec["link_choices"].append({"candidate": int(v), "yes": yes_l, "no": no_l, "chosen": choose})

                for v in chosen_links:
                    if v != cid and self.G.has_node(v): self.G.add_edge(cid, v)

                self.tokens[cid] = child_tokens
                self.brains[cid] = child_brain
                #self.reach_counts[cid] = {int(cid): 1}
                neighbors_cid = [int(w) for w in self.G.neighbors(cid)]
                self.reach_counts[cid] = {int(cid): 1, **{nv: 1 for nv in neighbors_cid}}
                self.messages[cid] = {}
                self.tokens[u] = keep_tokens
                rec.update({"child_created": True, "child_id": int(cid)})

                # C. Shift Edges (Handover relations to child)
                for idx, v in enumerate(neighbors_u):
                    col_idx = 1 + idx
                    sy, sn = float(shift_logits_all[0, col_idx]), float(shift_logits_all[1, col_idx])
                    shifted = bool(sy > sn)
                    rec["shift_choices"].append({"edge": (u, v), "shifted": shifted})
                    if shifted: shifts_to_apply.append((u, v, cid))

            # D. Reconnect (Rewire existing edges)
            reconnect_votes: List[Dict[str, float]] = []
            for idx, v in enumerate(neighbors_u):
                col_idx = 1 + idx
                reconnect_votes.append({
                    "edge": (int(u), int(v)),
                    "no": float(reconn_logits_all[0, col_idx]),
                    "yes": float(reconn_logits_all[1, col_idx]),
                    "link": float(reconn_logits_all[2, col_idx]),
                    "target": float(reconn_logits_all[3, col_idx])
                })
            if reconnect_votes:
                sum_no = sum(rv["no"] for rv in reconnect_votes)
                sum_yes = sum(rv["yes"] for rv in reconnect_votes)
                if PROBABILISTIC_DECISIONS:
                    y, n = max(0.0, sum_yes), max(0.0, sum_no)
                    p = (y / (y + n)) if (y + n) > 0 else 0.0
                    do_rec = bool(np.random.rand() < p)
                else:
                    do_rec = bool(sum_yes > sum_no)
                if do_rec:
                    scores = [rv["link"] for rv in reconnect_votes]
                    if PROBABILISTIC_DECISIONS:
                        vals = np.maximum(0.0, scores)
                        s = vals.sum()
                        probs = vals / s if s > 0 else np.ones(len(scores)) / len(scores)
                        idx_e = int(np.random.choice(len(reconnect_votes), p=probs))
                    else:
                        idx_e = int(np.argmax(scores))
                    old_v = int(reconnect_votes[idx_e]["edge"][1])
                    t_scores = [rv["target"] for rv in reconnect_votes]
                    if PROBABILISTIC_DECISIONS:
                        vals = np.maximum(0.0, t_scores)
                        s = vals.sum()
                        probs = vals / s if s > 0 else np.ones(len(t_scores)) / len(t_scores)
                        idx_t = int(np.random.choice(len(reconnect_votes), p=probs))
                    else:
                        idx_t = int(np.argmax(t_scores))
                    new_v = int(reconnect_votes[idx_t]["edge"][1])
                    if new_v != u and new_v != old_v and self.G.has_node(new_v):
                        reconns_to_apply.append((u, old_v, new_v))
                        rec["reconnect_choices"].append({"old": (int(u), int(old_v)), "new": int(new_v)})

            # E. Walker Link Creation (Long-distance connections)
            offset = len(core_candidates)
            for i, far_node in enumerate(w_cands):
                col_idx = offset + i
                wy = float(walker_logits_all[0, col_idx])
                wn = float(walker_logits_all[1, col_idx])
                want_link = bool(wy > wn)

                already = self.G.has_edge(u, far_node) or (u == far_node)
                if want_link and not already and self.G.has_node(far_node):
                    new_links_to_apply.append((u, far_node))

                rec["walker_decisions"].append({
                    "candidate": int(far_node), "created_link": want_link and not already
                })

            log["decisions"].append(rec)

        # 3. Apply Topology Edits
        for (u, v, cid) in shifts_to_apply:
            if self.G.has_node(cid) and self.G.has_edge(u, v):
                if not self.G.has_edge(cid, v) and cid != v: self.G.add_edge(cid, v)
                self.G.remove_edge(u, v)
        for (u, old_v, new_v) in reconns_to_apply:
            if self.G.has_edge(u, old_v) and u != new_v:
                self.G.remove_edge(u, old_v)
                if not self.G.has_edge(u, new_v): self.G.add_edge(u, new_v)
        for (u, v) in new_links_to_apply:
            if self.G.has_node(u) and self.G.has_node(v) and u != v:
                self.G.add_edge(u, v)
        self.G.remove_edges_from(list(nx.selfloop_edges(self.G)))

        log["cleanup"] = self._cleanup_and_redistribute()
        log["post_state"] = self._snapshot_graph()
        log["genotype_events"] = list(self.genotype_events)
        self.genotype_events.clear()

        if t % DRAW_EVERY_X_ITERATIONS == 0 and DRAW:
            self._draw(f"Round {t} — After Phase 1", f"step_{2 * t :05d}_phase1.png")

        return self._save_step_file(2 * t, log)

    # ------------------------------------------------------------------------
    # Phase 2: Blotto (Competition)
    # ------------------------------------------------------------------------

    def blotto_phase(self, t: int) -> str:
        """
        Agents compete for node dominance.
        They observe neighbors and 'bid' tokens to conquer or rob them.
        """
        from collections import defaultdict

        log: Dict[str, Any] = {
            "phase": "blotto",
            "pre_state": self._snapshot_graph(),
            "allocations": [],
            "incoming_offers": {},
            "winners": {},
            "pruned_edges": [],
        }
        tokens_before_phase = dict(self.tokens)

        # --------------------------------------------------------------------
        # 1. Precompute Sensory Manifold (New Code Logic)
        # --------------------------------------------------------------------
        log_deg, neighs, q_tok, q_deg, log_tok_map = self._precompute_features()

        # --------------------------------------------------------------------
        # 2. Observation Pass (Generate Messages)
        # --------------------------------------------------------------------
        for u in list(self.G.nodes()):
            targets = [u] + neighs[u]

            # Use New Code input vector (walker_feats=None)
            X_cols = [self._input_vec_fast(
                u, v, log_deg, q_tok, q_deg, log_tok_map,
                blotto_feats=None,
                walker_feats=None
            ) for v in targets]

            X = np.column_stack(X_cols)
            Y = self.brains[u].forward(X)
            self._emit_messages(u, targets, Y)

        # --------------------------------------------------------------------
        # 3. Setup Allocation State
        # --------------------------------------------------------------------
        remaining = {u: int(self.tokens.get(u, 0)) for u in self.G.nodes()}
        incoming_totals = {v: 0 for v in self.G.nodes()}
        per_target_allocators: Dict[int, Dict[int, int]] = {v: {} for v in self.G.nodes()}
        revolution_allocators: Dict[int, Dict[int, int]] = {v: {} for v in self.G.nodes()}  # NEW: Revolution pool
        u_sent_to_v = defaultdict(int)
        edge_flow = {tuple(sorted(e)): 0 for e in self.G.edges()}
        allocation_sequence = {u: [] for u in self.G.nodes()}

        # Helper: Determine "King" of a node
        def leader_info(v: int) -> Tuple[int, set[int]]:
            allocs = per_target_allocators[v]
            if not allocs: return 0, set()
            m = max(allocs.values())
            return m, {s for s, a in allocs.items() if a == m}

        # Helper: Calculate hypothetical Revolution Winner (for Context)
        def check_revolution_status(v: int) -> Tuple[bool, int | None]:
            """Returns (Did Revolution Win?, Winner ID) based on CURRENT state."""
            if not ALLOW_REVOLUTIONS: return False, None

            # 1. Identify the "Hegemon" / Standard King (Global Max Allocator)
            allocs = per_target_allocators.get(v, {})
            if not allocs: return False, None

            max_val = max(allocs.values())
            # Deterministic tie-breaking for input stability: pick largest Agent ID
            hegemon_candidates = [a for a, amt in allocs.items() if amt == max_val]
            hegemon = max(hegemon_candidates)
            hegemon_tokens = max_val

            # 2. Get Revolutionaries
            revs = revolution_allocators.get(v, {})
            if not revs: return False, None

            # 3. Build the "Mob"
            # Mob = All revolutionaries EXCEPT the Hegemon
            # (Even if Hegemon is in revs, they are the target, so they are removed from the mob)
            mob = []
            for ag, tok in revs.items():
                if ag != hegemon:
                    mob.append((ag, tok))

            if not mob: return False, None

            # 4. Sort Mob by tokens (Weakest -> Strongest)
            mob.sort(key=lambda x: x[1])

            # 5. The Mutiny Logic
            current_lower_sum = 0
            total_mob_tokens = sum(t for a, t in mob)

            processed_index = 0
            while processed_index < len(mob):
                # Handle groups of identical token amounts
                current_amount = mob[processed_index][1]
                current_group = []
                while processed_index < len(mob) and mob[processed_index][1] == current_amount:
                    current_group.append(mob[processed_index])
                    processed_index += 1

                # Update Lower Class Sum
                group_sum = sum(t for a, t in current_group)
                current_lower_sum += group_sum

                # Calculate Resistance (Upper Class + Hegemon)
                remaining_upper_tokens = total_mob_tokens - current_lower_sum
                resistance_total = remaining_upper_tokens + hegemon_tokens

                # 6. The Critical Check
                if current_lower_sum > resistance_total:
                    # Revolution Wins.
                    # Winner is one of the group that tipped the scale.
                    # Deterministic pick for features: max ID
                    winner_id = max([a for a, t in current_group])
                    return True, winner_id

            return False, None

        # Helper to snapshot the board state for the next round of decisions
        def compute_snapshot_views():
            snapshot_incoming_totals = incoming_totals.copy()
            snapshot_leader_max = {}
            snapshot_leader_set = {}
            for v in self.G.nodes():
                m, s = leader_info(v)
                snapshot_leader_max[v] = m
                snapshot_leader_set[v] = s
            return snapshot_incoming_totals, snapshot_leader_max, snapshot_leader_set

        # Helper to calculate Neural Scores for a set of targets
        def forward_scores_for(u, snapshot_incoming_totals, snapshot_leader_max, snapshot_leader_set):
            targets = [u] + neighs[u]
            X_cols: List[np.ndarray] = []

            for v in targets:
                # --- Calculate Context Features (Old Code Math) ---
                if u == v:
                    has_flow = 1.0
                else:
                    e = tuple(sorted((u, v)))
                    has_flow = 1.0 if edge_flow.get(e, 0) > 0 else 0.0

                u_to_v = float(u_sent_to_v[(u, v)])
                v_to_v = float(u_sent_to_v[(v, v)])
                max_on_v = float(snapshot_leader_max[v])
                leaders_v = snapshot_leader_set[v]
                leader_cnt = len(leaders_v)
                u_is_leader_now = 1.0 if (max_on_v > 0 and u in leaders_v) else 0.0
                u_wins_v_now = (1.0 / leader_cnt) if (max_on_v > 0 and leader_cnt > 0 and u in leaders_v) else 0.0

                pot_allocators_v = [v] + neighs[v]
                competitors = [w for w in pot_allocators_v if w != u]

                rem_u = int(remaining.get(u, 0))
                rem_v = int(remaining.get(v, 0))
                comp_rem = [int(remaining.get(w, 0)) for w in competitors]
                comp_rem_sum = float(sum(comp_rem))
                comp_rem_max = float(max(comp_rem) if comp_rem else 0.0)

                pot_inflow_total = float(sum(int(remaining.get(w, 0)) for w in pot_allocators_v))
                u_share_upper = (float(rem_u) / pot_inflow_total) if pot_inflow_total > 0.0 else 0.0

                allocs_map = per_target_allocators[v]
                comp_alloc_sum_ex_u = float(sum(a for s, a in allocs_map.items() if s != u))
                comp_alloc_max_ex_u = float(max([a for s, a in allocs_map.items() if s != u], default=0))

                vals_now = list(allocs_map.values())
                if not vals_now:
                    leader_gap, leader_gap_norm, tie_count_now = 0.0, 0.0, 0.0
                else:
                    top = max(vals_now)
                    tie_count_now = float(sum(1 for a in vals_now if a == top))
                    second = max([a for a in vals_now if a < top], default=0)
                    leader_gap = float(top - second)
                    leader_gap_norm = leader_gap / (1.0 + float(top))

                u_now = float(allocs_map.get(u, 0))
                best_rival = float(max([a for s, a in allocs_map.items() if s != u], default=0))
                u_deficit_to_lead = float(max(0, int(best_rival - u_now + 1)))

                # --- Revolution Features (New) ---
                rev_total = float(sum(revolution_allocators[v].values()))
                rev_own = float(revolution_allocators[v].get(u, 0))

                # Check Revolution state on the fly
                is_rev_success, rev_winner_id = check_revolution_status(v)

                # "tell if the one that allocated the most is also the biggest in the total competition"
                # (Interpreted as: Is the standard King also the Revolution Leader?)
                rev_pool = revolution_allocators[v]
                if rev_pool:
                    max_rev_contrib = max(rev_pool.values())
                    rev_leaders = [ag for ag, am in rev_pool.items() if am == max_rev_contrib]
                    # Check if any standard leader is also a revolution leader
                    king_is_rev_king = 1.0 if (not leaders_v.isdisjoint(rev_leaders)) else 0.0
                else:
                    king_is_rev_king = 0.0

                # "if he would currently be the winner" (Considering Revolution logic)
                if is_rev_success:
                    am_i_winner_total = 1.0 if rev_winner_id == u else 0.0
                else:
                    am_i_winner_total = u_wins_v_now

                # Pack the 25 raw metrics (20 old + 5 new)
                extra_feats = [
                    float(snapshot_incoming_totals[v]), max_on_v, has_flow, u_to_v, v_to_v, u_wins_v_now,
                    float(rem_u), float(rem_v), float(pot_inflow_total), float(u_share_upper),
                    float(len(competitors)), float(comp_rem_sum), float(comp_rem_max),
                    float(comp_alloc_sum_ex_u), float(comp_alloc_max_ex_u),
                    float(leader_gap), float(leader_gap_norm), float(tie_count_now),
                    float(u_is_leader_now), float(u_deficit_to_lead),
                    # New Revolution Features:
                    rev_total, rev_own, king_is_rev_king, am_i_winner_total, max_on_v
                ]

                # --- Activate Blotto Cortex (New Code Signature) ---
                # Note: _input_vec_fast will handle log-scaling of extra_feats
                X_cols.append(self._input_vec_fast(
                    u, v, log_deg, q_tok, q_deg, log_tok_map,
                    blotto_feats=extra_feats,
                    walker_feats=None
                ))

            X = np.column_stack(X_cols)
            Y = self.brains[u].forward(X)
            scores = np.asarray(Y[HEAD["BLOTTO"], :], dtype=float)
            alloc_yn_logits = np.asarray(Y[HEAD["ALLOCATE_YN"], :], dtype=float)
            rev_yn_logits = np.asarray(Y[HEAD["REVOLUTION_YN"], :], dtype=float)  # Shape (2, Num_Targets)

            return targets, scores, alloc_yn_logits, rev_yn_logits

        # --------------------------------------------------------------------
        # 4. Allocation Scheduling
        # --------------------------------------------------------------------

        if BLOTTO_ALLOCATION_MODE == "FULL_ALLOCATION":
            # Mode A: One-shot full allocation (Old Code Logic)
            snapshot_incoming_totals, snapshot_leader_max, snapshot_leader_set = compute_snapshot_views()
            for u in list(self.G.nodes()):
                t_u = int(remaining.get(u, 0))
                if t_u <= 0: continue

                targets, scores, _, rev_logits = forward_scores_for(u, snapshot_incoming_totals,
                                                                    snapshot_leader_max,
                                                                    snapshot_leader_set)

                vals = np.maximum(0.0, scores)
                s = float(vals.sum())
                if s <= 0.0:
                    probs = np.full_like(vals, 1.0 / len(vals), dtype=float)
                else:
                    probs = (vals / s).astype(float)

                # Largest remainder apportionment
                raw = probs * float(t_u)
                alloc_int = np.floor(raw).astype(int)
                rem = t_u - int(alloc_int.sum())
                if rem > 0:
                    fractional = raw - alloc_int
                    order = np.argsort(-fractional)[:rem]
                    for k in order: alloc_int[k] += 1

                # Apply
                for idx, v in enumerate(targets):
                    a = int(alloc_int[idx])
                    if a <= 0: continue
                    incoming_totals[v] += a
                    per_target_allocators[v][u] = per_target_allocators[v].get(u, 0) + a
                    u_sent_to_v[(u, v)] += a

                    # Revolution Check (Batch)
                    # For Full Allocation, we check the logits for this target once
                    # If Yes > No, ALL tokens for this target count as Revolution tokens
                    rev_yes = float(rev_logits[0, idx])
                    rev_no = float(rev_logits[1, idx])

                    if FORCE_REVOLUTIONS or (rev_yes > rev_no):
                        revolution_allocators[v][u] = revolution_allocators[v].get(u, 0) + a

                    if u != v:
                        e = tuple(sorted((u, v)))
                        if e in edge_flow: edge_flow[e] += a
                    allocation_sequence[u].extend([int(v)] * a)
                remaining[u] = 0

        elif BLOTTO_ALLOCATION_MODE == "STEP_ALLOCATION_WEAKEST_FIRST":
            # Mode B: Round-robin, 1 token per round, everyone plays (Old Code Logic)
            while True:
                eligible = [u for u in self.G.nodes() if remaining.get(u, 0) >= 1]
                if not eligible: break

                snap_tot, snap_max, snap_set = compute_snapshot_views()
                decisions = {}
                rev_decisions = {}  # Map u -> Bool (is_revolution)

                for u in eligible:
                    targets, scores, _, rev_logits = forward_scores_for(u, snap_tot, snap_max, snap_set)
                    if PROBABILISTIC_DECISIONS:
                        vals = np.maximum(0.0, scores)
                        s = float(vals.sum())
                        probs = (vals / s) if s > 0 else np.full_like(vals, 1.0 / len(vals))
                        idx = int(np.random.choice(len(targets), p=probs))
                    else:
                        idx = int(np.argmax(scores))
                    decisions[u] = int(targets[idx])

                    # Revolution Decision for this specific token
                    rev_yes = float(rev_logits[0, idx])
                    rev_no = float(rev_logits[1, idx])
                    rev_decisions[u] = (rev_yes > rev_no)

                for u, v in decisions.items():
                    remaining[u] -= 1
                    incoming_totals[v] += 1
                    per_target_allocators[v][u] = per_target_allocators[v].get(u, 0) + 1

                    if FORCE_REVOLUTIONS or rev_decisions[u]:
                        revolution_allocators[v][u] = revolution_allocators[v].get(u, 0) + 1

                    u_sent_to_v[(u, v)] += 1
                    if u != v:
                        e = tuple(sorted((u, v)))
                        if e in edge_flow: edge_flow[e] += 1
                    allocation_sequence[u].append(int(v))

        elif BLOTTO_ALLOCATION_MODE == "STEP_ALLOCATION_STRONGEST_FIRST":
            # Mode C: Round-robin, only max-token agents play (New Code / Current Logic)
            while True:
                elig = [u for u in self.G.nodes() if remaining.get(u, 0) >= 1]
                if not elig: break
                max_rem = max(remaining[u] for u in elig)
                eligible = [u for u in elig if remaining[u] == max_rem]

                snap_tot, snap_max, snap_set = compute_snapshot_views()
                decisions = {}
                rev_decisions = {}

                for u in eligible:
                    targets, scores, _, rev_logits = forward_scores_for(u, snap_tot, snap_max, snap_set)
                    if PROBABILISTIC_DECISIONS:
                        vals = np.maximum(0.0, scores)
                        s = float(vals.sum())
                        probs = (vals / s) if s > 0 else np.full_like(vals, 1.0 / len(vals))
                        idx = int(np.random.choice(len(targets), p=probs))
                    else:
                        idx = int(np.argmax(scores))
                    decisions[u] = int(targets[idx])

                    # Revolution Decision for this specific token
                    rev_yes = float(rev_logits[0, idx])
                    rev_no = float(rev_logits[1, idx])
                    rev_decisions[u] = (rev_yes > rev_no)

                for u, v in decisions.items():
                    remaining[u] -= 1
                    incoming_totals[v] += 1
                    per_target_allocators[v][u] = per_target_allocators[v].get(u, 0) + 1

                    if FORCE_REVOLUTIONS or rev_decisions[u]:
                        revolution_allocators[v][u] = revolution_allocators[v].get(u, 0) + 1

                    u_sent_to_v[(u, v)] += 1
                    if u != v:
                        e = tuple(sorted((u, v)))
                        if e in edge_flow: edge_flow[e] += 1
                    allocation_sequence[u].append(int(v))

        elif BLOTTO_ALLOCATION_MODE == "STEP_ALLOCATION_AS_WISHED":
            # Mode D: Voluntary Contribution with Forced Ceiling
            while True:
                elig = [u for u in self.G.nodes() if remaining.get(u, 0) >= 1]
                if not elig: break

                # The "Ceiling": The game cannot last longer than the richest player allows.
                max_rem = max(remaining[u] for u in elig)

                snap_tot, snap_max, snap_set = compute_snapshot_views()
                decisions = {}
                rev_decisions = {}

                for u in elig:
                    # Constraint: If I am the richest, I MUST play to prevent stalling.
                    is_forced = (remaining[u] == max_rem)

                    targets, scores, yn_logits, rev_logits = forward_scores_for(u, snap_tot, snap_max, snap_set)

                    # --- DECISION: Do I want to allocate? ---
                    avg_yes = float(np.mean(yn_logits[0, :]))
                    avg_no = float(np.mean(yn_logits[1, :]))

                    wants_to_play = False

                    if PROBABILISTIC_DECISIONS:
                        vy = np.exp(avg_yes)
                        vn = np.exp(avg_no)
                        p_yes = vy / (vy + vn) if (vy + vn) > 0 else 0.5
                        if np.random.rand() < p_yes:
                            wants_to_play = True
                    else:
                        if avg_yes > avg_no:
                            wants_to_play = True

                    # --- EXECUTION ---
                    if is_forced or wants_to_play:
                        if PROBABILISTIC_DECISIONS:
                            vals = np.maximum(0.0, scores)
                            s = float(vals.sum())
                            probs = (vals / s) if s > 0 else np.full_like(vals, 1.0 / len(vals))
                            idx = int(np.random.choice(len(targets), p=probs))
                        else:
                            idx = int(np.argmax(scores))

                        decisions[u] = int(targets[idx])

                        # Revolution Decision
                        rev_yes = float(rev_logits[0, idx])
                        rev_no = float(rev_logits[1, idx])
                        rev_decisions[u] = (rev_yes > rev_no)

                # Apply Decisions
                for u, v in decisions.items():
                    remaining[u] -= 1
                    incoming_totals[v] += 1
                    per_target_allocators[v][u] = per_target_allocators[v].get(u, 0) + 1

                    if FORCE_REVOLUTIONS or rev_decisions[u]:
                        revolution_allocators[v][u] = revolution_allocators[v].get(u, 0) + 1

                    u_sent_to_v[(u, v)] += 1
                    if u != v:
                        e = tuple(sorted((u, v)))
                        if e in edge_flow: edge_flow[e] += 1
                    allocation_sequence[u].append(int(v))

        else:
            raise ValueError(f"Unknown BLOTTO_ALLOCATION_MODE: {BLOTTO_ALLOCATION_MODE}")

        # --------------------------------------------------------------------
        # 5. Resolution (Conquest vs Robbery vs Transfer)
        # --------------------------------------------------------------------
        new_tokens = dict(self.tokens)
        new_brains = dict(self.brains)
        walker_resets: List[int] = []

        # Helper: Determine Winner with Revolution Logic
        def resolve_winner(v: int) -> Tuple[int, int]:
            """
            Returns (winner_id, winning_amount).
            Implements 'Tipping Point' Mutiny logic: Lower Class vs (Hegemon + Upper Class)
            """
            offers = per_target_allocators[v]
            if not offers: return v, 0

            # 1. Identify the "Hegemon" / Standard King
            # This is the max allocator from the GLOBAL offers, not just the revolution.
            max_amt = max(offers.values())
            standard_winners = [s for s, a in offers.items() if a == max_amt]

            # If standard win (no revolution allowed), pick random max allocator
            if not ALLOW_REVOLUTIONS:
                return int(np.random.choice(standard_winners)), max_amt

            # We pick one "Hegemon" to represent the establishment.
            # (If there are multiple Kings with equal tokens, we pick one randomly to be the target)
            hegemon = int(np.random.choice(standard_winners))
            hegemon_tokens = max_amt

            revs = revolution_allocators.get(v, {})
            if not revs:
                # No revolution exists -> Standard King wins
                return hegemon, max_amt

            # 2. Build the "Mob"
            # The Mob is everyone in the revolution EXCEPT the Hegemon.
            # Even if the Hegemon allocated revolution tokens, they are removed from the "Mutiny" pool
            # because they are the one being fought against.
            mob = []
            for ag, tok in revs.items():
                if ag != hegemon:
                    mob.append((ag, tok))

            # If the Mob is empty (e.g. only the King is in the revolution), King wins.
            if not mob:
                return hegemon, max_amt

            # 3. Sort Mob by tokens (Weakest -> Strongest)
            mob.sort(key=lambda x: x[1])

            # 4. The Mutiny Logic (Lower Class vs Upper Class + Hegemon)
            current_lower_sum = 0
            total_mob_tokens = sum(t for a, t in mob)

            processed_index = 0
            while processed_index < len(mob):
                # Handle groups of identical token amounts (e.g. multiple agents with 1 token)
                current_amount = mob[processed_index][1]
                current_group = []
                while processed_index < len(mob) and mob[processed_index][1] == current_amount:
                    current_group.append(mob[processed_index])
                    processed_index += 1

                # Update Lower Class Sum (The attackers)
                group_sum = sum(t for a, t in current_group)
                current_lower_sum += group_sum

                # Calculate Resistance (The defenders)
                # Resistance = (Remaining Mob / Upper Class) + (The Hegemon's Tokens)
                remaining_upper_tokens = total_mob_tokens - current_lower_sum
                resistance_total = remaining_upper_tokens + hegemon_tokens

                # 5. The Critical Check
                # Does the Lower Class overpower the Establishment + Upper Class?
                if current_lower_sum > resistance_total:
                    # Revolution Wins!
                    # The winner is one of the "tipping point" agents.
                    rev_winner = int(np.random.choice([a for a, t in current_group]))

                    # Return the winner ID, and max_amt (to keep logs consistent)
                    return rev_winner, max_amt

            # 6. If loop finishes, the Mutiny Failed (Hegemon was too strong)
            return hegemon, max_amt

        if BLOTTO_MODE == "ALLOCATE_AND_CONQUER":
            for v in list(self.G.nodes()):
                offers = per_target_allocators[v]
                if not offers:
                    new_tokens[v] = 0
                    # Survival: Clone self
                    new_brains[v] = self.brains[v].copy()
                    if MUTATE_ON_BLOTTO_COPY: new_brains[v].mutate()

                    if RESET_REACH_ON_CONQUER:
                        neighbors_vid = [int(w) for w in self.G.neighbors(v)]
                        self.reach_counts[v] = {int(v): 1, **{nv: 1 for nv in neighbors_vid}}
                        walker_resets.append(int(v))
                    continue

                # RESOLVE
                winner, max_amt = resolve_winner(v)

                # Genetic Takeover (Brain Overwrite)
                new_brains[v] = self.brains[winner].copy()
                if MUTATE_ON_BLOTTO_COPY: new_brains[v].mutate()
                new_tokens[v] = int(incoming_totals[v])
                log["winners"][str(v)] = {"winner": int(winner), "max_amount": int(max_amt)}

                if winner != v and RESET_REACH_ON_CONQUER:
                    # Reset memory of the conquered node
                    neighbors_vid = [int(w) for w in self.G.neighbors(v)]
                    self.reach_counts[v] = {int(v): 1, **{nv: 1 for nv in neighbors_vid}}
                    walker_resets.append(int(v))

        elif BLOTTO_MODE == "ALLOCATE_AND_ROB":
            winnings: Dict[int, int] = {u: 0 for u in self.G.nodes()}
            for v in list(self.G.nodes()):
                offers = per_target_allocators[v]
                if not offers:
                    new_tokens[v] = 0
                    new_brains[v] = self.brains[v]  # Keep own brain
                    continue

                # RESOLVE
                winner, max_amt = resolve_winner(v)

                # Winner gets the tokens, Loser (v) keeps the brain but loses tokens
                winnings[winner] = winnings.get(winner, 0) + int(incoming_totals[v])
                new_tokens[v] = 0
                new_brains[v] = self.brains[v]
                log["winners"][str(v)] = {"winner": int(winner), "max_amount": int(max_amt)}

            # Apply winnings to the robbers
            for w, gain in winnings.items():
                new_tokens[w] = new_tokens.get(w, 0) + int(gain)

        # === NEW MODE ADDED HERE ===
        elif BLOTTO_MODE == "ALLOCATE_AND_DO_NOTHING":
            # Simple Wealth Transfer:
            # Tokens move based on allocation, but Brains are never overwritten.
            # useful for cooperative maintenance of the network structure.
            for v in list(self.G.nodes()):
                # The node simply possesses whatever was allocated to it
                # (either by itself or by neighbors).
                new_tokens[v] = int(incoming_totals[v])

                # Brains persist (no conquest)
                new_brains[v] = self.brains[v]

                # We do not reset walkers because no conquest occurred.

        else:
            raise ValueError(f"Unknown BLOTTO_MODE: {BLOTTO_MODE}")

        log["walker_resets"] = walker_resets

        # Commit outcome state
        self.tokens = new_tokens
        self.brains = new_brains

        # --------------------------------------------------------------------
        # 6. Cleanup & Logging
        # --------------------------------------------------------------------

        # Mutation Pulse (Global)
        if MUTATE_ALL_AFTER_BLOTTO:
            for b in self.brains.values(): b.mutate()

        # Prune dead edges
        to_remove = [e for e, f in edge_flow.items() if f == 0]
        if to_remove: self.G.remove_edges_from(to_remove)
        log["pruned_edges"] = [(int(u), int(v)) for (u, v) in to_remove]

        # Call the robust cleanup function
        log["cleanup"] = self._cleanup_and_redistribute()
        log["post_state"] = self._snapshot_graph()

        # Messaging Dictionary Hygiene
        for u in list(self.messages.keys()):
            if not self.G.has_node(u):
                self.messages.pop(u, None)
                continue
            allowed = set([u] + [int(w) for w in self.G.neighbors(u)])
            keep: Dict[int, List[float]] = {}
            for v, vec in self.messages.get(u, {}).items():
                if v in allowed and self.G.has_node(v):
                    keep[v] = vec
            self.messages[u] = keep

        # Aggregated Logs
        for u in list(self.G.nodes()) + [x for x in allocation_sequence.keys() if x not in self.G.nodes()]:
            tgs = [u] + (neighs.get(u, []))
            # Safe access for logging even if u died
            alloc_counts = [int(u_sent_to_v[(u, v)]) for v in tgs]
            log["allocations"].append({
                "agent_id": int(u),
                "tokens_before": int(tokens_before_phase.get(u, 0)),
                "targets": [int(v) for v in tgs],
                "alloc": alloc_counts
            })
        log["allocation_sequence"] = {str(u): [int(x) for x in seq] for u, seq in allocation_sequence.items()}

        log["genotype_events"] = list(self.genotype_events)
        self.genotype_events.clear()

        if t % DRAW_EVERY_X_ITERATIONS == 0 and DRAW:
            self._draw(f"Round {t} — After Phase 2", f"step_{2 * t + 1:05d}_phase2.png")

        return self._save_step_file(2 * t + 1, log)

    # ------------------------------------------------------------------------
    # Cleanup: The Physics of Death
    # ------------------------------------------------------------------------
    def _cleanup_and_redistribute(self) -> Dict[str, Any]:
        """
        Applies the 'Physics of Death' and topology cleanup.

        Rules:
        1. Nodes with <= 0 tokens are removed (Starvation).
        2. Only the Largest Connected Component survives (Isolation).
        3. Redistribution (Conservation of Resources):
           - GLOBAL_UNIFORM: Tokens from dead/pruned nodes are pooled and shared active-globally.
           - LOCAL_SCAVENGING: Tokens flow to surviving neighbors. Isolated tokens go active-globally.
        """
        report: Dict[str, Any] = {
            "resurrected": False,
            "resurrect_agent": None,
            "removed_zero_nodes": [],
            "removed_components": [],
            "redistributed_tokens": 0,
            "survivors_count": 0,
        }

        # --------------------------------------------------------------------
        # 1. Identification Phase (Who dies? Who survives?)
        # --------------------------------------------------------------------

        # A. Identify Starved Nodes (Zero Tokens)
        zero_nodes = [u for u in self.G.nodes() if self.tokens.get(u, 0) <= 0]
        report["removed_zero_nodes"] = [int(u) for u in zero_nodes]

        # B. Simulate Graph after starvation to find components
        # We need a temporary view of the graph where zero_nodes are gone
        G_active = self.G.copy()
        G_active.remove_nodes_from(zero_nodes)

        # C. Identify Isolated Components (Pruning)
        survivors_set = set()
        pruned_nodes = set()

        if G_active.number_of_nodes() > 0:
            comps = list(nx.connected_components(G_active))
            comps.sort(key=len, reverse=True)

            # The largest component survives
            survivors_set = set(comps[0])

            # All smaller components are pruned
            if len(comps) > 1:
                for c in comps[1:]:
                    pruned_nodes.update(c)
                report["removed_components"] = [list(map(int, c)) for c in comps[1:]]

        # The total list of nodes being removed from the simulation
        # (We use a set to avoid duplicates, though sets should be disjoint here)
        all_doomed_nodes = set(zero_nodes) | pruned_nodes

        # --------------------------------------------------------------------
        # 2. Redistribution Phase (The Physics)
        # --------------------------------------------------------------------
        global_pool = CREATE_X_NEW_TOKENS_EACH_PHASE

        # Iterate through every dying node to distribute its earthly possessions
        for u in all_doomed_nodes:
            amt = self.tokens.get(u, 0)
            if amt <= 0:
                continue

            if TOKEN_REDISTRIBUTION_MODE == "LOCAL_SCAVENGING":
                # Scavenging Rule:
                # Tokens flow to "neighbors that are still alive".
                # We check the ORIGINAL graph (self.G) to find who 'u' was connected to.
                neighbors = list(self.G.neighbors(u))

                # Filter for neighbors who are in the 'survivors_set' (The Giant Component)
                beneficiaries = [v for v in neighbors if v in survivors_set]

                if beneficiaries:
                    # Distribute locally to survivors
                    count = len(beneficiaries)
                    share = amt // count
                    remainder = amt % count
                    for idx, v in enumerate(beneficiaries):
                        self.tokens[v] += (share + (1 if idx < remainder else 0))
                else:
                    # No connection to the survivors? (e.g., totally isolated island).
                    # The tokens evaporate to the global atmosphere.
                    global_pool += amt
            else:
                # GLOBAL_UNIFORM Rule:
                # All tokens from dying nodes go to the global pot.
                global_pool += amt

        # --------------------------------------------------------------------
        # 3. Execution Phase (Surgery)
        # --------------------------------------------------------------------

        if all_doomed_nodes:
            self.G.remove_nodes_from(list(all_doomed_nodes))
            for u in all_doomed_nodes:
                self.tokens.pop(u, None)
                self.brains.pop(u, None)
                self.reach_counts.pop(u, None)
                self.messages.pop(u, None)

        # --------------------------------------------------------------------
        # 4. Global Settling & Resurrection
        # --------------------------------------------------------------------

        # Distribute the global pool (Manna)
        survivors = list(self.G.nodes())
        if global_pool > 0 and survivors:
            # Multinomial for integer conservation
            draws = np.random.multinomial(global_pool, [1 / len(survivors)] * len(survivors))
            for u, add in zip(survivors, draws):
                self.tokens[u] = self.tokens.get(u, 0) + int(add)

        report["redistributed_tokens"] = int(global_pool)
        report["survivors_count"] = self.G.number_of_nodes()

        # Resurrect if the world is empty
        if self.G.number_of_nodes() == 0:
            aid = self.next_agent_id
            self.next_agent_id += 1
            self.G.add_node(aid)
            self.tokens = {aid: self.total_tokens}
            self.brains = {aid: Brain()}
            self.reach_counts = {aid: {aid: 1}}
            self.messages = {aid: {}}
            report.update({"resurrected": True, "resurrect_agent": int(aid), "survivors_count": 1})

        return report

    def step(self, t: int) -> Tuple[str, str]:
        p1 = self.reproduction_phase(t)
        p2 = self.blotto_phase(t)
        return p1, p2

    # ------------------------------------------------------------------------
    # Visualization & IO (Standard)
    # ------------------------------------------------------------------------

    def _snapshot_graph(self) -> Dict[str, Any]:
        """Captures graph state for replay."""
        nodes = []
        for u in self.G.nodes():
            nodes.append({
                "agent_id": int(u), "tokens": int(self.tokens.get(u, 0)),
                "brain_id": int(self.brains[u].brain_id),
                "neighbors": [int(v) for v in self._neighbors(u)]
            })
        return {"nodes": nodes, "edges": [(int(u), int(v)) for u, v in self.G.edges()]}

    def _save_step_file(self, idx: int, blob: Dict[str, Any]) -> str:
        path = os.path.join(self.run_dir, f"step_{idx:05d}.json")
        with open(path, "w") as f: json.dump(blob, f, indent=2)
        return path

    def _snapshot_source(self) -> None:
        try:
            src = os.path.abspath(__file__)
            dst = os.path.join(self.run_dir, os.path.basename(src))
            shutil.copy2(src, dst)
        except:
            pass

    def _save_configuration(self) -> None:
        with open(os.path.join(self.run_dir, "config.txt"), "w") as f:
            for k, v in globals().items():
                if k.isupper() and not k.startswith("_"): f.write(f"{k}: {v}\n")

    def _draw(self, title: str, fname: str, k_max: int = 3) -> None:
        if self.G.number_of_nodes() == 0: return
        pos3d = nx.spring_layout(self.G, dim=3, seed=42)
        pos2d = {n: (c[0], c[1]) for n, c in pos3d.items()}
        coreness = nx.core_number(self.G) if self.G.number_of_edges() > 0 else {u: 0 for u in self.G.nodes()}
        layer_k = {u: min(coreness.get(u, 0), k_max) for u in self.G.nodes()}
        L_node = {u: (k_max - layer_k[u]) for u in self.G.nodes()}
        layers_nodes: Dict[int, List[int]] = {}
        for u, L in L_node.items(): layers_nodes.setdefault(L, []).append(u)
        L_edge = {(u, v): max(L_node[u], L_node[v]) for (u, v) in self.G.edges()}
        layers_edges: Dict[int, List[Tuple[int, int]]] = {}
        for e, L in L_edge.items(): layers_edges.setdefault(L, []).append(e)

        plt.figure(figsize=(8, 6))
        vmin = 0
        vmax = max([0] + [self.tokens.get(u, 0) for u in self.G.nodes()])
        cmap = matplotlib.colormaps.get_cmap("viridis")
        for L in sorted(layers_edges.keys(), reverse=True):
            edgelist = layers_edges[L]
            if not edgelist: continue
            edge_alpha = 0.5 / (2 ** L)
            nx.draw_networkx_edges(self.G, pos2d, edgelist=edgelist, alpha=edge_alpha, width=0.5 if L == 0 else 0.3)
        for L in sorted(layers_nodes.keys(), reverse=True):
            nlist = layers_nodes[L]
            if not nlist: continue
            node_alpha = 1.0 / (2 ** L)
            nx.draw_networkx_nodes(
                self.G, pos2d, nodelist=nlist,
                node_size=[(self.tokens.get(u, 0) + 1) / 12 for u in nlist],
                node_color=[self.tokens.get(u, 0) for u in nlist],
                cmap=cmap, vmin=vmin, vmax=vmax, alpha=node_alpha,
            )
        plt.title(title)
        plt.axis("off")
        plt.savefig(os.path.join(self.run_dir, fname), dpi=130, bbox_inches="tight")
        plt.close("all")


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
                f"Run {run_counter}, iteration {t} finished (nodes: {simulation.G.number_of_nodes()}, edges: {simulation.G.number_of_edges()}, tokens: {sum(simulation.tokens.values())})")

            if simulation.G.number_of_nodes() <= 50:
                print(f"⚠️ Run {run_counter} extinction event. Restarting.")
                break


if __name__ == "__main__":
    _main()