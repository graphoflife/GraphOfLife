#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphOfLife — Open‑ended Evolution on a Mutable Graph (GitHub v1)
=================================================================

This engine simulates **open‑ended evolution** on a dynamic, undirected graph.
Each node is an *agent* with a lightweight neural "brain" that observes local
state and makes discrete decisions in two phases per iteration:

1) **Reproduction phase** — agents may spawn a child, rewire edges, and create
   new links. Topology edits are applied simultaneously.

2) **Blotto phase** — agents compete for positions using a round‑based,
   *one‑token‑at‑a‑time* allocation game. Decisions are lockstep per round so all
   agents observe the same snapshot when allocating each token. Winners implant
   their brain into the occupied node. Edges without flow are pruned.

**Goals**
- Minimal, reproducible code that still supports *ever‑evolving strategies*.
- Rich, per‑phase JSON logs for analysis and reverse‑engineering.
- Token conservation across phases (sum of tokens is invariant).

**Outputs per iteration**
- `step_XXXXX.json` files capturing pre/post state, decisions, and cleanup.
- Optional PNG snapshots (`DRAW=True`).
- A copy of this source file + SHA256 in each run folder for reproducibility.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from datetime import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess

import matplotlib  # set backend before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ----------------------------------------------------------------------------
# Configuration & constants
# ----------------------------------------------------------------------------

# Determinism of decisions (per-token in Blotto; per-comparison elsewhere)
PROBABILISTIC_DECISIONS: bool = False

# ---- Blotto rule variants ----
#   "ALLOCATE_AND_CONQUER": highest allocator at node v implants its brain into v
#   "ALLOCATE_AND_ROB"    : winner-for-v *collects* all tokens allocated to v; no brain copying
BLOTTO_MODE: str = "ALLOCATE_AND_CONQUER"   # default keeps current behavior

# ---- Mutation policy toggles ----
# Mutate when a brain is copied in reproduction (child creation)
MUTATE_ON_REPRO_COPY: bool = False          # default preserves current behavior (no mutation there)

# Mutate when a brain is copied in Blotto (only relevant for ALLOCATE_AND_CONQUER)
MUTATE_ON_BLOTTO_COPY: bool = False          # default preserves current behavior (there was mutation already)

# Mutate all brains at the very end of Blotto (after outcomes are applied)
MUTATE_ALL_AFTER_BLOTTO: bool = True       # default off

# ---- Reproduction decision variant ----
# True  -> ignore yes/no logits; use averaged fraction logits over core to set child_tokens
# False -> use standard yes/no logits to decide whether reproduction occurs at all
REPRO_CORE_FRACTIONS_ONLY: bool = False  # default preserves current behavior

# ---- Walker/reach reset policy (only for ALLOCATE_AND_CONQUER) ----
# True  -> (default) reset reach_counts[v] and record walker_resets when a brain is (re)implanted at v
# False -> keep previous reach_counts[v]; do not push v into walker_resets
RESET_REACH_ON_CONQUER: bool = True

# ---- Exchange Messages ----
# True  -> Agents can exchange messages
# False -> CANNOT
EXCHANGE_MESSAGES: bool = True
MESSAGE_NUMBER_AMOUNT = 4

# ---- Blotto allocation scheduling ----
# "FULL_ALLOCATION"              : one observation; each agent allocates all its tokens at once by relative scores
# "STEP_ALLOCATION_WEAKEST_FIRST": (current behavior) every agent with >=1 token allocates 1 token each round
# "STEP_ALLOCATION_STRONGEST_FIRST": only agents with the current per-round max remaining tokens allocate 1 token
BLOTTO_ALLOCATION_MODE: str = "STEP_ALLOCATION_STRONGEST_FIRST"


# Draw k-core visualizations every 10 steps.
DRAW: bool = False

# Output directory (created beside this file by default)
BASE_DIR = os.path.join(os.path.dirname(__file__), "GraphOfLifeOutputs")
os.makedirs(BASE_DIR, exist_ok=True)

# Indices of output heads (rows of the Brain's output)
HEAD = {
    "REPRO": slice(0, 4),        # yes/no to reproduce (aggregated)
    "LINK": slice(4, 6),         # yes/no to link new child to candidate
    "SHIFT": slice(6, 8),        # move existing (u,v) edge to (child,v)
    "RECONNECT": slice(8, 12),   # choose edge to drop & new neighbor
    "BLOTTO": 12,                # single scalar score for blotto choice
    "WALKER": slice(13, 15),     # yes/no to create walker link
    "MESSAGE": slice(15, 15 + MESSAGE_NUMBER_AMOUNT),   # Numbers to exchange messages
}

# ----------------------------------------------------------------------------
# Utility: math helpers
# ----------------------------------------------------------------------------

def _softmax_logits(vals: List[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    m = np.max(arr)
    exp = np.exp(arr - m)  # numeric stability
    z = exp.sum()
    return exp / z if z > 0 else np.full_like(arr, 1.0 / len(arr))


def _six_quantiles(sorted_vals: List[float]) -> List[float]:
    """Return [q0, q20, q40, q60, q80, q100] with linear interpolation.
    Expects *sorted* values. Returns zeros for empty input.
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
    # Stable sigmoid
    pos = x >= 0
    neg = ~pos
    z = np.empty_like(x, dtype=float)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    z[neg] = ex / (1.0 + ex)
    return z

# ----------------------------------------------------------------------------
# Brain: simple FFNN
# ----------------------------------------------------------------------------


class Brain:
    """Feed‑forward NN with configurable hidden layers.

    Hidden layers use sigmoid; output layer is linear. We treat the output rows
    as *heads* (see HEAD), accessed by slices / indices.
    """

    _next_brain_id = 1
    rec = None  # set to a callable(event_dict) by the engine (e.g., list.append)

    def __init__(self) -> None:
        # Base features were 29 + 20 blotto extras = 49.
        # Messaging adds 4 * MESSAGE_NUMBER_AMOUNT inputs:
        # (u->u, u->v, v->u, v->v), each of length M.
        n_inputs = 49 + 4 * MESSAGE_NUMBER_AMOUNT
        hidden_sizes = [50, 45, 40, 35, 30]
        n_outputs = 15 + MESSAGE_NUMBER_AMOUNT

        assert n_inputs > 0 and n_outputs > 0
        self.layer_sizes = [int(n_inputs)] + [int(h) for h in hidden_sizes] + [int(n_outputs)]

        # Xavier/Glorot-ish initialization
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
        """Forward pass.
        - Input: shape (n_inputs,) or (n_inputs, batch)
        - Returns: shape (n_outputs,) or (n_outputs, batch)
        """
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
        if Brain.rec: Brain.rec({"t":"copy","from":int(self.brain_id),"to":int(new_brain.brain_id)})

        return new_brain

    def mutate(
            self,
            mutate_prob: float = 0.35,
            weight_noise_std: float = 0.2,
            bias_noise_std: float = 0.2,
            p_weight_noise: float = 0.1,
            p_bias_noise: float = 0.1,
            p_weight_reset: float = 0.1,
            p_bias_reset: float = 0.1,
            reset_fraction: float = 0.10,
            layerwise_scale: bool = True,
    ) -> None:
        """Mutate in place via Gaussian noise + optional random resets.

        - With probability `mutate_prob`, this brain is mutated.
        - Noise is applied sparsely (p_*_noise masks).
        - A subset of parameters can be fully re-sampled (resets).
        - If `layerwise_scale=True`, noise and resets are scaled per-layer
          using 1/sqrt(fan_in) similar to Xavier init.
        """
        if np.random.random() > mutate_prob:
            return

        old_id = self.brain_id
        reset_fraction = float(np.clip(reset_fraction, 0.0, 1.0))

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            fan_in = W.shape[1]

            # Scale factors per layer (smaller steps for wider layers)
            if layerwise_scale and fan_in > 0:
                base_scale = 1.0 / np.sqrt(fan_in)
            else:
                base_scale = 1.0

            # Effective std per layer
            w_noise_std_layer = float(weight_noise_std) * base_scale
            b_noise_std_layer = float(bias_noise_std) * base_scale

            # --- Weights: sparse additive noise ---
            if p_weight_noise > 0.0 and w_noise_std_layer > 0.0:
                mask = np.random.random(W.shape) < p_weight_noise
                W = W + np.random.normal(
                    loc=0.0,
                    scale=w_noise_std_layer,
                    size=W.shape,
                ) * mask

            # --- Weights: random resets (Xavier-like, consistent with init) ---
            if (p_weight_reset > 0.0) and (reset_fraction > 0.0):
                if np.random.random() < p_weight_reset:
                    reset_mask = np.random.random(W.shape) < reset_fraction
                    if np.any(reset_mask):
                        W_new = np.random.normal(
                            loc=0.0,
                            scale=1.0 / np.sqrt(fan_in) if fan_in > 0 else 1.0,
                            size=W.shape,
                        )
                        W = np.where(reset_mask, W_new, W)

            self.weights[i] = W.astype(float, copy=False)

            # --- Biases: sparse additive noise ---
            if p_bias_noise > 0.0 and b_noise_std_layer > 0.0:
                maskb = np.random.random(b.shape) < p_bias_noise
                b = b + np.random.normal(
                    loc=0.0,
                    scale=b_noise_std_layer,
                    size=b.shape,
                ) * maskb

            # --- Biases: random resets (small, zero-centered) ---
            if (p_bias_reset > 0.0) and (reset_fraction > 0.0):
                if np.random.random() < p_bias_reset:
                    reset_maskb = np.random.random(b.shape) < reset_fraction
                    if np.any(reset_maskb):
                        # Biases were initialized to 0; we keep them small
                        b_new = np.random.normal(
                            loc=0.0,
                            scale=b_noise_std_layer if b_noise_std_layer > 0 else 0.01,
                            size=b.shape,
                        )
                        b = np.where(reset_maskb, b_new, b)

            self.biases[i] = b.astype(float, copy=False)

        # Track lineage
        self.parent_brain_id = old_id
        self.brain_id = Brain._next_brain_id
        Brain._next_brain_id += 1
        if Brain.rec:
            Brain.rec({"t": "mut", "from": int(self.parent_brain_id), "to": int(self.brain_id)})


# ----------------------------------------------------------------------------
# GraphOfLife
# ----------------------------------------------------------------------------


class GraphOfLife:
    """Two-phase evolutionary dynamics on a mutable graph with rich logs."""

    def __init__(self, G_init: nx.Graph, total_tokens: int) -> None:
        # Persistent agent IDs: 0..N-1 initially
        self.G = nx.Graph()
        self.next_agent_id = 0
        old2new: Dict[Any, int] = {}
        for n in G_init.nodes():
            aid = self.next_agent_id
            self.next_agent_id += 1
            old2new[n] = aid
            self.G.add_node(aid)
        for u, v in G_init.edges():
            self.G.add_edge(old2new[u], old2new[v])

        self.total_tokens = int(total_tokens)

        # Per-agent state
        self.tokens: Dict[int, int] = {aid: 0 for aid in self.G.nodes()}
        self.brains: Dict[int, Brain] = {aid: Brain() for aid in self.G.nodes()}
        self.reach_counts: Dict[int, Dict[int, int]] = {aid: {aid: 1} for aid in self.G.nodes()}
        self.messages: Dict[int, Dict[int, List[float]]] = {aid: {} for aid in self.G.nodes()}

        # Initialize tokens uniformly
        N = self.G.number_of_nodes()
        for aid in self.G.nodes():
            self.tokens[aid] = int(self.total_tokens / N)

        # Run folder
        date_str = datetime.now().strftime("%Y_%m_%d")
        prefix = f"GOL_{date_str}__"

        # Find existing run folders for today
        existing_runs = []
        for name in os.listdir(BASE_DIR):
            full_path = os.path.join(BASE_DIR, name)
            if not os.path.isdir(full_path):
                continue
            if not name.startswith(prefix):
                continue

            # Extract the numeric suffix, e.g. "001" from "run_20251214_001"
            suffix = name[len(prefix):]
            if len(suffix) == 3 and suffix.isdigit():
                existing_runs.append(int(suffix))

        # Determine next index (start at 1 → "001")
        next_idx = (max(existing_runs) + 1) if existing_runs else 1

        folder_name = f"{prefix}{next_idx:03d}"
        self.run_dir = os.path.join(BASE_DIR, folder_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self._snapshot_source()
        self.genotype_events: List[Dict[str, int]] = []
        Brain.rec = self.genotype_events.append

    # ----------------- helpers -----------------
    def _neighbors(self, u: int) -> List[int]:
        return list(self.G.neighbors(u))

    def _precompute_features(self) -> Tuple[Dict[int, float], Dict[int, List[int]], Dict[int, List[float]], Dict[int, List[float]]]:
        deg = {u: float(self.G.degree[u]) for u in self.G.nodes()}
        neighs = {u: list(self.G.neighbors(u)) for u in self.G.nodes()}
        q_tok: Dict[int, List[float]] = {}
        q_deg: Dict[int, List[float]] = {}
        for u, N in neighs.items():
            neigh_tokens = sorted(int(self.tokens.get(n, 0)) for n in N)
            neigh_degs = sorted(float(self.G.degree[n]) for n in N)
            q_tok[u] = _six_quantiles(neigh_tokens) if neigh_tokens else [0.0] * 6
            q_deg[u] = _six_quantiles(neigh_degs) if neigh_degs else [0.0] * 6
        return deg, neighs, q_tok, q_deg

    def _input_vec_fast(
        self,
        u: int,
        v: int,
        deg: Dict[int, float],
        q_tok: Dict[int, List[float]],
        q_deg: Dict[int, List[float]],
        extra_feats: List[float] | None = None,
        scale: float = 0.1,
    ) -> np.ndarray:
        """Build the per-(observer,target) input vector.

        During reproduction `extra_feats` is None (zeros appended).
        During blotto, `extra_feats` contains information important for blotto phase
          0) total_on_v            : sum of tokens allocated to node v so far
          1) current_max_on_v      : max tokens any single bidder has on v so far
          2) edge_has_flow         : 1 if some token flowed on (u,v); for u==v we set 1
          3) u_to_v                : tokens already sent by u to v
          4) v_to_v                : tokens already sent by v to itself
          5) u_wins_v_now          : tie‑aware win prob for u on v (1/|leaders| or 0)
          6) remaining_alloc_u     : tokens u still can allocate in this phase
          7) remaining_alloc_v     : tokens v still can allocate in this phase
        """
        own_obs = int(u == v)
        own_t, tgt_t = int(self.tokens.get(u, 0)), int(self.tokens.get(v, 0))
        own_deg, tgt_deg = deg[u], deg[v]

        base = [own_t, tgt_t, own_deg, tgt_deg] + q_tok[u] + q_tok[v] + q_deg[u] + q_deg[v]
        base = [f * scale for f in base]


        if extra_feats is None:
            extras_scaled = [0.0] * 20
        else:
            extras_scaled = [float(x) * scale for x in extra_feats]

        # ---- Messaging features (always included) ----
        # Pull 4*M values: (u->u), (u->v), (v->u), (v->v)
        M = MESSAGE_NUMBER_AMOUNT
        def _msg(src: int, dst: int) -> List[float]:
            vec = self.messages.get(src, {}).get(dst, None)
            if vec is None:
                return [0.0] * M
            out = list(vec[:M])
            if len(out) < M:
                out += [0.0] * (M - len(out))
            return out
        msg_feats = _msg(u, u) + _msg(u, v) + _msg(v, u) + _msg(v, v)
        msg_feats = [m * scale for m in msg_feats]

        return np.array([own_obs] + base + extras_scaled + msg_feats, dtype=float)




    # ---------------------------------------------------------------------
    # Messaging emission helper: write Y[HEAD["MESSAGE"]] into self.messages
    # for each (u -> v) in the order of 'targets' (columns of X / Y).
    # Call this immediately after ANY observation forward pass.
    # ---------------------------------------------------------------------
    def _emit_messages(self, u: int, targets: List[int], Y: np.ndarray) -> None:
        if EXCHANGE_MESSAGES:
            M = MESSAGE_NUMBER_AMOUNT
            msg_rows = Y[HEAD["MESSAGE"], :]  # shape: (M, len(targets)) or (len(targets),) if M==1
            # Ensure 2D shape even if M==1
            if M == 1 and msg_rows.ndim == 1:
                msg_rows = np.array([msg_rows])
            # Hard-coded squashing to [-1, 1]
            msg_rows = np.tanh(msg_rows)

            for j, v in enumerate(targets):
                col = msg_rows[:, j].astype(float).tolist()
                # Defensive pad/truncate to length M
                if len(col) < M:
                    col = col + [0.0] * (M - len(col))
                elif len(col) > M:
                    col = col[:M]
                self.messages.setdefault(u, {})[int(v)] = col

    # ----- logging utilities -----
    def _snapshot_graph(self) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        for u in self.G.nodes():
            nodes.append(
                {
                    "agent_id": int(u),
                    "tokens": int(self.tokens.get(u, 0)),
                    "brain_id": int(self.brains[u].brain_id) if u in self.brains else None,
                    "degree": int(self.G.degree[u]),
                    "neighbors": [int(v) for v in self._neighbors(u)],
                }
            )
        edges = [(int(u), int(v)) for u, v in self.G.edges()]
        return {"nodes": nodes, "edges": edges}

    def _save_step_file(self, idx: int, blob: Dict[str, Any]) -> str:
        path = os.path.join(self.run_dir, f"step_{idx:05d}.json")
        with open(path, "w") as f:
            json.dump(blob, f, indent=2)
        return path

    def _draw(self, title: str, fname: str, k_max: int = 3) -> None:
        """Layered k‑core visualization with decaying transparency."""
        if self.G.number_of_nodes() == 0:
            return
        pos3d = nx.spring_layout(self.G, dim=3, seed=42)
        pos2d = {n: (c[0], c[1]) for n, c in pos3d.items()}

        coreness = nx.core_number(self.G) if self.G.number_of_edges() > 0 else {u: 0 for u in self.G.nodes()}
        layer_k = {u: min(coreness.get(u, 0), k_max) for u in self.G.nodes()}
        L_node = {u: (k_max - layer_k[u]) for u in self.G.nodes()}

        layers_nodes: Dict[int, List[int]] = {}
        for u, L in L_node.items():
            layers_nodes.setdefault(L, []).append(u)

        L_edge: Dict[Tuple[int, int], int] = {}
        for (u, v) in self.G.edges():
            L_edge[(u, v)] = max(L_node[u], L_node[v])
        layers_edges: Dict[int, List[Tuple[int, int]]] = {}
        for e, L in L_edge.items():
            layers_edges.setdefault(L, []).append(e)

        def size_of(u: int) -> float:
            return (self.tokens.get(u, 0) + 1) * 8

        def color_of(u: int) -> int:
            return self.tokens.get(u, 0)

        vmin = 0
        vmax = max([0] + [self.tokens.get(u, 0) for u in self.G.nodes()])
        cmap = matplotlib.colormaps.get_cmap("viridis")

        layer_keys_sorted = sorted(layers_nodes.keys(), reverse=True)
        plt.figure(figsize=(8, 6))

        for L in sorted(layers_edges.keys(), reverse=True):
            edgelist = layers_edges[L]
            if not edgelist:
                continue
            edge_alpha = 0.5 / (2**L)
            nx.draw_networkx_edges(self.G, pos2d, edgelist=edgelist, alpha=edge_alpha, width=2.5 if L == 0 else 2.0)

        mappable_for_cb = None
        for L in layer_keys_sorted:
            nlist = layers_nodes[L]
            if not nlist:
                continue
            node_alpha = 1.0 / (2**L)
            sizes = [size_of(u) for u in nlist]
            colors = [color_of(u) for u in nlist]
            coll = nx.draw_networkx_nodes(
                self.G,
                pos2d,
                nodelist=nlist,
                node_size=sizes,
                node_color=colors,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=node_alpha,
            )
            mappable_for_cb = coll

        if mappable_for_cb is not None:
            plt.colorbar(mappable_for_cb, label="Tokens")
        plt.title(title)
        plt.axis("off")
        plt.savefig(os.path.join(self.run_dir, fname), dpi=130, bbox_inches="tight")
        plt.close("all")

    # ----- reproducibility -----
    def _snapshot_source(self) -> None:
        """Copy this .py into the run folder and write a SHA‑256 + git meta."""
        try:
            src_path = os.path.abspath(__file__)
            dst_path = os.path.join(self.run_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            h = hashlib.sha256()
            with open(src_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            with open(dst_path + ".sha256", "w") as f:
                f.write(h.hexdigest() + "\n")

            git_info: Dict[str, Any] = {}
            try:
                probe = os.path.dirname(src_path)
                for _ in range(5):
                    if os.path.isdir(os.path.join(probe, ".git")):
                        break
                    parent = os.path.dirname(probe)
                    if parent == probe:
                        probe = os.path.dirname(src_path)
                        break
                    probe = parent
                commit = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=probe, stderr=subprocess.DEVNULL)
                    .decode()
                    .strip()
                )
                dirty = subprocess.call(["git", "diff", "--quiet"], cwd=probe, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                git_info = {"repo_root": probe, "commit": commit, "dirty": bool(dirty)}
            except Exception:
                pass

            meta = {
                "source_filename": os.path.basename(src_path),
                "sha256": h.hexdigest(),
                "git": git_info,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            with open(os.path.join(self.run_dir, "source_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            try:
                with open(os.path.join(self.run_dir, "source_snapshot_error.txt"), "w") as f:
                    f.write(repr(e))
            except Exception:
                pass

    # ----------------------------------------------------------------------------
    # Phase 1: Reproduction
    # ----------------------------------------------------------------------------
    def _advance_reach_counts(self) -> None:
        """Advance per-agent walker reach counts by one step (including stay)."""
        G = self.G
        new_maps: Dict[int, Dict[int, int]] = {}
        for u in list(G.nodes()):
            prev = self.reach_counts.get(u, {u: 1})
            prev = {w: c for w, c in prev.items() if G.has_node(w)}  # prune dead
            if not prev:
                new_maps[u] = {u: 1}
                continue
            acc: Dict[int, int] = {}
            for r, c in prev.items():
                for w in list(G.neighbors(r)):
                    acc[w] = acc.get(w, 0) + int(c)
                acc[r] = acc.get(r, 0) + int(c)  # stay
            new_maps[u] = acc or {u: 1}
        self.reach_counts = new_maps

    def _cleanup_and_redistribute(self) -> Dict[str, Any]:
        """Cleanup and report exactly what happened (see return dict)."""
        report: Dict[str, Any] = {
            "resurrected": False,
            "resurrect_agent": None,
            "removed_zero_nodes": [],
            "removed_components": [],
            "redistributed_tokens": 0,
            "survivors_count": self.G.number_of_nodes(),
        }

        # Remove zero-token nodes (optional in P1; always in P2)
        zero_nodes = [u for u in list(self.G.nodes()) if self.tokens.get(u, 0) <= 0]
        if zero_nodes:
            self.G.remove_nodes_from(zero_nodes)
            for u in zero_nodes:
                self.tokens.pop(u, None)
                self.brains.pop(u, None)
                self.reach_counts.pop(u, None)
            report["removed_zero_nodes"] = [int(u) for u in zero_nodes]

        # If empty, resurrect a single node holding all tokens
        if self.G.number_of_nodes() == 0:
            aid = self.next_agent_id
            self.next_agent_id += 1
            self.G.add_node(aid)
            self.tokens = {aid: self.total_tokens}
            self.brains = {aid: Brain()}
            self.reach_counts = {aid: {aid: 1}}
            report.update({"resurrected": True, "resurrect_agent": int(aid), "survivors_count": 1})
            return report

        # Keep only largest connected component; redistribute removed tokens uniformly
        comps = list(nx.connected_components(self.G))
        comps.sort(key=len, reverse=True)
        if len(comps) > 1:
            keep = comps[0]
            remove = set().union(*comps[1:])
            tokens_to_redistribute = int(sum(self.tokens.get(u, 0) for u in remove))
            self.G.remove_nodes_from(list(remove))
            for u in remove:
                self.tokens.pop(u, None)
                self.brains.pop(u, None)
                self.reach_counts.pop(u, None)

            survivors = list(self.G.nodes())
            if tokens_to_redistribute > 0 and survivors:
                draws = np.random.multinomial(tokens_to_redistribute, [1 / len(survivors)] * len(survivors))
                for u, add in zip(survivors, draws):
                    self.tokens[u] = self.tokens.get(u, 0) + int(add)

            report["removed_components"] = [list(map(int, c)) for c in comps[1:]]
            report["redistributed_tokens"] = int(tokens_to_redistribute)

        report["survivors_count"] = self.G.number_of_nodes()
        assert sum(self.tokens.values()) == self.total_tokens, "Token conservation violated!"
        return report

    def reproduction_phase(self, t: int) -> str:
        log: Dict[str, Any] = {"phase": "reproduction", "pre_state": self._snapshot_graph(), "decisions": []}

        self._advance_reach_counts()
        deg, neighs, q_tok, q_deg = self._precompute_features()

        shifts_to_apply: List[Tuple[int, int, int]] = []  # (u, v, child_id)
        reconns_to_apply: List[Tuple[int, int, int]] = []  # (u, old_v, new_v)
        new_links_to_apply: List[Tuple[int, int]] = []     # (u, far_node)

        for u in list(self.G.nodes()):
            t_u = int(self.tokens.get(u, 0))
            if t_u <= 0:
                continue

            neighbors_u = list(neighs[u])
            core_candidates = [u] + neighbors_u

            # Walker choice: pick one far node from reach memory (weighted)
            rc_map = self.reach_counts.get(u, {u: 1})
            neigh_set = set(neighbors_u)
            rc_items_far = [
                (v, c)
                for v, c in rc_map.items()
                if v != u and v not in neigh_set and self.G.has_node(v) and c > 0
            ]
            chosen_far = None
            far_weight = None
            if rc_items_far:
                nodes_far, weights_far = zip(*rc_items_far)
                weights_arr = np.asarray(weights_far, dtype=float)
                wsum = float(weights_arr.sum())
                if wsum > 0.0:
                    probs_far = weights_arr / wsum
                    idx_far = int(np.random.choice(len(nodes_far), p=probs_far))
                    chosen_far = int(nodes_far[idx_far])
                    far_weight = int(weights_far[idx_far])

            all_candidates = core_candidates + [chosen_far] if chosen_far is not None else core_candidates
            X_cols = [self._input_vec_fast(u, v, deg, q_tok, q_deg) for v in all_candidates]
            X = np.column_stack(X_cols)
            Y = self.brains[u].forward(X)
            # Emit messages for reproduction-time observation
            self._emit_messages(u, all_candidates, Y)

            repro_logits_all = Y[HEAD["REPRO"], :]
            link_logits_all = Y[HEAD["LINK"], :]
            shift_logits_all = Y[HEAD["SHIFT"], :]
            reconn_logits_all = Y[HEAD["RECONNECT"], :]
            walker_logits_all = Y[HEAD["WALKER"], :]

            # Reproduction decision (aggregated over core candidates)
            repro_core = repro_logits_all[:, :len(core_candidates)]  # shape (4, K_core)
            # rows: [yes_logit_row, no_logit_row, child_frac_row, keep_frac_row]
            # t_u: current tokens at u (int)

            if REPRO_CORE_FRACTIONS_ONLY:
                # (A) Ignore explicit yes/no; deterministically split using averaged fractions
                frac_vec = np.mean(repro_core[2:4, :], axis=1)  # shape (2,) -> [child_part, keep_part]
                vals = np.maximum(0.0, frac_vec)
                s = float(np.sum(vals))
                probs = (vals / s) if s > 0.0 else np.full_like(vals, 1.0 / len(vals))

                child_tokens = int(np.floor(probs[0] * t_u))

            else:
                # (B) Original yes/no gate to decide reproduction at all
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
                    frac_vec = np.mean(repro_core[2:4, :], axis=1)  # [child_part, keep_part]
                    vals = np.maximum(0.0, frac_vec)
                    s = float(np.sum(vals))
                    probs = (vals / s) if s > 0.0 else np.full_like(vals, 1.0 / len(vals))

                    child_tokens = int(np.floor(probs[0] * t_u))
                else:
                    child_tokens = 0

            # Safety: ensure conservation and non-negativity
            child_tokens = max(0, min(int(t_u), int(child_tokens)))

            rec: Dict[str, Any] = {
                "agent_id": int(u),
                "tokens_before": int(t_u),
                "repro_tokens": int(child_tokens),
                "child_created": False,
                "child_id": None,
                "link_choices": [],
                "shift_choices": [],
                "reconnect_choices": [],
                "reached_nodes_hist": {int(k): int(v) for k, v in (rc_map.items() if rc_map else [])},
                "walker_choice": {
                    "far_node": int(chosen_far) if chosen_far is not None else None,
                    "far_weight": int(far_weight) if far_weight is not None else None,
                    "already_connected": None,
                    "created_new_link": False,
                },
            }

            if child_tokens >= 1:
                child_tokens = max(1, min(child_tokens, t_u))
                keep_tokens = t_u - child_tokens

                # Create child & copy brain
                child_brain = self.brains[u].copy()
                if MUTATE_ON_REPRO_COPY:
                    child_brain.mutate()  # optional child-on-copy mutation
                cid = self.next_agent_id
                self.next_agent_id += 1
                self.G.add_node(cid)

                # Child link creation (core candidates only)
                chosen_links: List[int] = []
                link_logs: List[Dict[str, Any]] = []
                for col_idx, v in enumerate(core_candidates):
                    yes_logit = float(link_logits_all[0, col_idx])
                    no_logit = float(link_logits_all[1, col_idx])
                    if PROBABILISTIC_DECISIONS:

                        y = max(0.0, yes_logit)
                        n = max(0.0, no_logit)
                        s = y + n
                        p_yes = (y / s) if s > 0.0 else 0.0
                        choose = bool(np.random.rand() < p_yes)
                    else:
                        choose = bool(yes_logit > no_logit)
                    if choose:
                        chosen_links.append(v)
                    link_logs.append({"candidate": int(v), "yes_logit": yes_logit, "no_logit": no_logit, "chosen": choose})

                for v in chosen_links:
                    if v != cid and self.G.has_node(v):
                        self.G.add_edge(cid, v)

                # Finalize tokens/brains
                self.tokens[cid] = child_tokens
                self.brains[cid] = child_brain
                neighbors_cid = [int(v) for v in self.G.neighbors(cid)]
                self.reach_counts[cid] = {int(cid): 1, **{nv: 1 for nv in neighbors_cid}}
                self.messages[cid] = {}
                self.tokens[u] = keep_tokens

                rec.update({"child_created": True, "child_id": int(cid), "link_choices": link_logs})

                # Shifting (per original neighbor only)
                for idx, v in enumerate(neighbors_u):
                    col_idx = 1 + idx
                    shift_yes = float(shift_logits_all[0, col_idx])
                    shift_no = float(shift_logits_all[1, col_idx])
                    if PROBABILISTIC_DECISIONS:
                        y = max(0.0, shift_yes)
                        n = max(0.0, shift_no)
                        s = y + n
                        p_yes = (y / s) if s > 0.0 else 0.0
                        shifted = bool(np.random.rand() < p_yes)
                    else:
                        shifted = bool(shift_yes > shift_no)
                    rec["shift_choices"].append({"edge": (int(u), int(v)), "yes_logit": shift_yes, "no_logit": shift_no, "shifted": shifted})
                    if shifted:
                        shifts_to_apply.append((u, v, cid))

                # Reconnection: pick one edge and one target if "yes" wins
                reconnect_votes: List[Dict[str, float]] = []
                for idx, v in enumerate(neighbors_u):
                    col_idx = 1 + idx
                    no_val = float(reconn_logits_all[0, col_idx])
                    yes_val = float(reconn_logits_all[1, col_idx])
                    which_link = float(reconn_logits_all[2, col_idx])
                    which_target = float(reconn_logits_all[3, col_idx])
                    reconnect_votes.append({"edge": (int(u), int(v)), "no_val": no_val, "yes_val": yes_val, "link_val": which_link, "target_val": which_target})

                if reconnect_votes:
                    sum_no = sum(rv["no_val"] for rv in reconnect_votes)
                    sum_yes = sum(rv["yes_val"] for rv in reconnect_votes)
                    if PROBABILISTIC_DECISIONS:
                        y = max(0.0, sum_yes)
                        n = max(0.0, sum_no)
                        s = y + n
                        p_yes = (y / s) if s > 0.0 else 0.0
                        do_reconnect = bool(np.random.rand() < p_yes)
                    else:
                        do_reconnect = bool(sum_yes > sum_no)
                    if do_reconnect:
                        # --- pick which edge to drop (by link_val) ---
                        if PROBABILISTIC_DECISIONS:
                            link_scores = [max(0.0, rv["link_val"]) for rv in reconnect_votes]
                            s = float(sum(link_scores))
                            link_probs = ([w / s for w in link_scores]
                                          if s > 0.0 else [1.0 / len(reconnect_votes)] * len(reconnect_votes))
                            idx_edge = int(np.random.choice(len(reconnect_votes), p=link_probs))
                        else:
                            idx_edge = int(np.argmax([rv["link_val"] for rv in reconnect_votes]))
                        edge_choice = reconnect_votes[idx_edge]
                        old_v = int(edge_choice["edge"][1])

                        # --- pick the new neighbor target (by target_val) ---
                        if PROBABILISTIC_DECISIONS:
                            targ_scores = [max(0.0, rv["target_val"]) for rv in reconnect_votes]
                            s = float(sum(targ_scores))
                            targ_probs = ([w / s for w in targ_scores]
                                          if s > 0.0 else [1.0 / len(reconnect_votes)] * len(reconnect_votes))
                            idx_target = int(np.random.choice(len(reconnect_votes), p=targ_probs))
                        else:
                            idx_target = int(np.argmax([rv["target_val"] for rv in reconnect_votes]))
                        target_choice = reconnect_votes[idx_target]
                        new_v = int(target_choice["edge"][1])

                        if new_v != u and new_v != old_v and self.G.has_node(new_v):
                            reconns_to_apply.append((u, old_v, new_v))
                            rec["reconnect_choices"].append({
                                "old_edge": (int(u), int(old_v)),
                                "new_neighbor": int(new_v)
                            })

            # Walker-link (applies even if no child was created)
            if chosen_far is not None:
                last_idx = len(all_candidates) - 1
                wl_yes = float(walker_logits_all[0, last_idx])
                wl_no = float(walker_logits_all[1, last_idx])
                if PROBABILISTIC_DECISIONS:
                    y = max(0.0, wl_yes)
                    n = max(0.0, wl_no)
                    s = y + n
                    p_yes = (y / s) if s > 0.0 else 0.0
                    want_link = bool(np.random.rand() < p_yes)
                else:
                    want_link = bool(wl_yes > wl_no)
                already_connected = self.G.has_edge(u, chosen_far) or (u == chosen_far)
                rec["walker_choice"]["already_connected"] = bool(already_connected)
                #rec["walker_choice"]["decision_logits"] = [wl_yes, wl_no]
                if want_link and not already_connected and self.G.has_node(chosen_far):
                    new_links_to_apply.append((u, chosen_far))
                    rec["walker_choice"]["created_new_link"] = True

            log["decisions"].append(rec)

        # Apply deferred topology edits
        for (u, v, cid) in shifts_to_apply:
            if self.G.has_node(cid) and self.G.has_edge(u, v):
                if not self.G.has_edge(cid, v) and cid != v:
                    self.G.add_edge(cid, v)
                self.G.remove_edge(u, v)
        for (u, old_v, new_v) in reconns_to_apply:
            if self.G.has_edge(u, old_v) and u != new_v:
                self.G.remove_edge(u, old_v)
                if not self.G.has_edge(u, new_v):
                    self.G.add_edge(u, new_v)
        for (u, v) in new_links_to_apply:
            if self.G.has_node(u) and self.G.has_node(v) and u != v and not self.G.has_edge(u, v):
                self.G.add_edge(u, v)
        self.G.remove_edges_from(list(nx.selfloop_edges(self.G)))

        log["cleanup"] = self._cleanup_and_redistribute()
        log["post_state"] = self._snapshot_graph()
        log["genotype_events"] = list(self.genotype_events)
        self.genotype_events.clear()
        path = self._save_step_file(2 * t, log)
        if t % 10 == 0 and DRAW:
            self._draw(f"Round {t} — After Phase 1", f"step_{2 * t:05d}_phase1.png")
        return path


    # ----------------------------------------------------------------------------
    # Phase 2: Blotto
    # ----------------------------------------------------------------------------
    def blotto_phase(self, t: int) -> str:
        from collections import defaultdict
        from typing import Dict, Any, List, Tuple

        log: Dict[str, Any] = {
            "phase": "blotto",
            "pre_state": self._snapshot_graph(),
            "allocations": [],
            "incoming_offers": {},
            "winners": {},
            "pruned_edges": [],
        }

        # Snapshot pre-blotto state for logging
        tokens_before_phase = dict(self.tokens)
        brains_before_phase = {u: b.brain_id for u, b in self.brains.items()}

        deg, neighs, q_tok, q_deg = self._precompute_features()


        # ---------------------------------------------------------------
        # Pre-blotto messaging-only observation:
        # Every agent observes self+neighbors, emits MESSAGE head stored as
        # self.messages[u][v] (last message u->v).
        # ---------------------------------------------------------------
        for u in list(self.G.nodes()):
            targets = [u] + neighs[u]
            X_cols = [self._input_vec_fast(u, v, deg, q_tok, q_deg, extra_feats=None) for v in targets]
            X = np.column_stack(X_cols)
            Y = self.brains[u].forward(X)
            # NEW: emit messages for this observation
            self._emit_messages(u, targets, Y)

        # Per-round state for allocation
        remaining = {u: int(self.tokens.get(u, 0)) for u in self.G.nodes()}
        incoming_totals = {v: 0 for v in self.G.nodes()}
        per_target_allocators: Dict[int, Dict[int, int]] = {v: {} for v in self.G.nodes()}
        u_sent_to_v = defaultdict(int)
        edge_flow = {tuple(sorted(e)): 0 for e in self.G.edges()}
        allocation_sequence = {u: [] for u in self.G.nodes()}

        def leader_info(v: int) -> Tuple[int, set[int]]:
            """Max allocation on v so far and the set of current leaders."""
            allocs = per_target_allocators[v]
            if not allocs:
                return 0, set()
            m = max(allocs.values())
            return m, {s for s, a in allocs.items() if a == m}

        # --------------------- Allocation scheduling ---------------------
        def compute_snapshot_views():
            # Snapshot shared by this round (or for FULL_ALLOCATION: single snapshot)
            snapshot_incoming_totals = incoming_totals.copy()
            snapshot_leader_max: Dict[int, int] = {}
            snapshot_leader_set: Dict[int, set[int]] = {}
            for v in self.G.nodes():
                m, s = leader_info(v)
                snapshot_leader_max[v] = m
                snapshot_leader_set[v] = s
            return snapshot_incoming_totals, snapshot_leader_max, snapshot_leader_set

        def forward_scores_for(u, snapshot_incoming_totals, snapshot_leader_max, snapshot_leader_set):
            targets = [u] + neighs[u]
            X_cols: List[np.ndarray] = []
            for v in targets:
                # --- existing lightweight context ---
                if u == v:
                    has_flow = 1.0
                else:
                    e = tuple(sorted((u, v)))
                    has_flow = 1.0 if edge_flow.get(e, 0) > 0 else 0.0
                u_to_v = float(u_sent_to_v[(u, v)])  # how much u already sent to v in this phase
                v_to_v = float(u_sent_to_v[(v, v)])  # how much v self-sent so far
                max_on_v = float(snapshot_leader_max[v])
                leaders_v = snapshot_leader_set[v]
                leader_cnt = len(leaders_v)
                u_is_leader_now = 1.0 if (max_on_v > 0 and u in leaders_v) else 0.0
                u_wins_v_now = (1.0 / leader_cnt) if (max_on_v > 0 and leader_cnt > 0 and u in leaders_v) else 0.0

                # --- NEW: opponent capacity / competitiveness at v ---
                # Potential allocators for v are v plus its neighbors
                pot_allocators_v = [v] + neighs[v]
                # Competitors for u at v exclude u
                competitors = [w for w in pot_allocators_v if w != u]

                # Remaining tokens among parties
                rem_u = int(remaining.get(u, 0))
                rem_v = int(remaining.get(v, 0))
                comp_rem = [int(remaining.get(w, 0)) for w in competitors]
                comp_rem_sum = float(sum(comp_rem))
                comp_rem_max = float(max(comp_rem) if comp_rem else 0.0)

                # Upper bound of what could still flow into v this phase
                pot_inflow_total = float(sum(int(remaining.get(w, 0)) for w in pot_allocators_v))
                u_share_upper = (float(rem_u) / pot_inflow_total) if pot_inflow_total > 0.0 else 0.0

                # Allocations to v so far (this phase)
                allocs_map = per_target_allocators[v]
                comp_alloc_sum_ex_u = float(sum(a for s, a in allocs_map.items() if s != u))
                comp_alloc_max_ex_u = float(max([a for s, a in allocs_map.items() if s != u], default=0))

                # Leader gap (top minus second)
                vals_now = list(allocs_map.values())
                if not vals_now:
                    leader_gap = 0.0
                    leader_gap_norm = 0.0
                    tie_count_now = 0.0
                else:
                    top = max(vals_now)
                    tie_count_now = float(sum(1 for a in vals_now if a == top))
                    second = max([a for a in vals_now if a < top], default=0)
                    leader_gap = float(top - second)
                    leader_gap_norm = leader_gap / (1.0 + float(top))

                # How many *more* tokens u would need to strictly lead right now
                u_now = float(allocs_map.get(u, 0))
                # Best rival's count (if u is not unique leader)
                best_rival = float(max([a for s, a in allocs_map.items() if s != u], default=0))
                # If u already strictly leads, deficit is 0; if tied or behind, need (best_rival - u_now + 1)
                u_deficit_to_lead = float(max(0, int(best_rival - u_now + 1)))

                extra_feats = [
                    # --- original extras (keep order) ---
                    float(snapshot_incoming_totals[v]),  # cumulative inflow to v so far
                    max_on_v,  # current max offers on v
                    has_flow,  # has any edge flow on (u,v)
                    u_to_v,  # u->v so far
                    v_to_v,  # v->v so far
                    u_wins_v_now,  # if leading, split of the tie
                    float(rem_u),  # remaining[u]
                    float(rem_v),  # remaining[v]
                    # --- new opponent-awareness & competitiveness ---
                    float(pot_inflow_total),  # upper bound of tokens that can still go to v
                    float(u_share_upper),  # u's share of that bound
                    float(len(competitors)),  # number of competitors at v (excl. u)
                    float(comp_rem_sum),  # competitors' remaining sum
                    float(comp_rem_max),  # competitors' remaining max
                    float(comp_alloc_sum_ex_u),  # competitor allocations to v so far
                    float(comp_alloc_max_ex_u),  # competitor max allocation to v so far
                    float(leader_gap),  # top-second gap at v now
                    float(leader_gap_norm),  # normalized gap
                    float(tie_count_now),  # number of current co-leaders at v
                    float(u_is_leader_now),  # binary: is u among leaders now
                    float(u_deficit_to_lead),  # tokens u needs to strictly lead at v now
                ]

                X_cols.append(self._input_vec_fast(u, v, deg, q_tok, q_deg, extra_feats=extra_feats))

            X = np.column_stack(X_cols)
            Y = self.brains[u].forward(X)
            # Emit messages for this blotto observation
            self._emit_messages(u, targets, Y)
            scores = np.asarray(Y[HEAD["BLOTTO"], :], dtype=float)
            return targets, scores

        if BLOTTO_ALLOCATION_MODE == "FULL_ALLOCATION":
            # One observation; each agent allocates all tokens at once by relative scores.
            snapshot_incoming_totals, snapshot_leader_max, snapshot_leader_set = compute_snapshot_views()
            for u in list(self.G.nodes()):
                t_u = int(remaining.get(u, 0))
                if t_u <= 0:
                    continue
                targets, scores = forward_scores_for(u, snapshot_incoming_totals, snapshot_leader_max,
                                                     snapshot_leader_set)

                vals = np.maximum(0.0, scores)
                s = float(vals.sum())
                if s <= 0.0:
                    probs = np.full_like(vals, 1.0 / len(vals), dtype=float)
                else:
                    probs = (vals / s).astype(float)

                # Largest remainder apportionment to conserve tokens exactly
                raw = probs * float(t_u)
                alloc_int = np.floor(raw).astype(int)
                rem = t_u - int(alloc_int.sum())
                if rem > 0:
                    fractional = raw - alloc_int
                    order = np.argsort(-fractional)[:rem]
                    for k in order:
                        alloc_int[k] += 1

                # Apply all allocations for u
                for idx, v in enumerate(targets):
                    a = int(alloc_int[idx])
                    if a <= 0:
                        continue
                    incoming_totals[v] += a
                    per_target_allocators[v][u] = per_target_allocators[v].get(u, 0) + a
                    u_sent_to_v[(u, v)] += a
                    if u != v:
                        e = tuple(sorted((u, v)))
                        if e in edge_flow:
                            edge_flow[e] += a
                    allocation_sequence[u].extend([int(v)] * a)
                remaining[u] = 0  # all tokens allocated in one shot

        elif BLOTTO_ALLOCATION_MODE == "STEP_ALLOCATION_WEAKEST_FIRST":
            # Current behavior: each round every node with >=1 token allocates 1 token.
            while True:
                eligible = [u for u in self.G.nodes() if remaining.get(u, 0) >= 1]
                if not eligible:
                    break
                snapshot_incoming_totals, snapshot_leader_max, snapshot_leader_set = compute_snapshot_views()
                decisions: Dict[int, int] = {}
                for u in eligible:
                    targets, scores = forward_scores_for(u, snapshot_incoming_totals, snapshot_leader_max,
                                                         snapshot_leader_set)
                    if PROBABILISTIC_DECISIONS:
                        vals = np.maximum(0.0, scores)
                        s = float(vals.sum())
                        probs = (vals / s) if s > 0 else np.full_like(vals, 1.0 / len(vals))
                        idx = int(np.random.choice(len(targets), p=probs))
                    else:
                        idx = int(np.argmax(scores))
                    decisions[u] = int(targets[idx])

                # Apply all one-token decisions simultaneously
                for u, v in decisions.items():
                    remaining[u] -= 1
                    incoming_totals[v] += 1
                    per_target_allocators[v][u] = per_target_allocators[v].get(u, 0) + 1
                    u_sent_to_v[(u, v)] += 1
                    if u != v:
                        e = tuple(sorted((u, v)))
                        if e in edge_flow:
                            edge_flow[e] += 1
                    allocation_sequence[u].append(int(v))

        elif BLOTTO_ALLOCATION_MODE == "STEP_ALLOCATION_STRONGEST_FIRST":
            # Only nodes with the current max remaining tokens allocate 1 per round.
            while True:
                elig = [u for u in self.G.nodes() if remaining.get(u, 0) >= 1]
                if not elig:
                    break
                max_rem = max(remaining[u] for u in elig)
                eligible = [u for u in elig if remaining[u] == max_rem]

                snapshot_incoming_totals, snapshot_leader_max, snapshot_leader_set = compute_snapshot_views()
                decisions: Dict[int, int] = {}
                for u in eligible:
                    targets, scores = forward_scores_for(u, snapshot_incoming_totals, snapshot_leader_max,
                                                         snapshot_leader_set)
                    if PROBABILISTIC_DECISIONS:
                        vals = np.maximum(0.0, scores)
                        s = float(vals.sum())
                        probs = (vals / s) if s > 0 else np.full_like(vals, 1.0 / len(vals))
                        idx = int(np.random.choice(len(targets), p=probs))
                    else:
                        idx = int(np.argmax(scores))
                    decisions[u] = int(targets[idx])

                # Apply all one-token decisions for the strongest set
                for u, v in decisions.items():
                    remaining[u] -= 1
                    incoming_totals[v] += 1
                    per_target_allocators[v][u] = per_target_allocators[v].get(u, 0) + 1
                    u_sent_to_v[(u, v)] += 1
                    if u != v:
                        e = tuple(sorted((u, v)))
                        if e in edge_flow:
                            edge_flow[e] += 1
                    allocation_sequence[u].append(int(v))
        else:
            raise ValueError(f"Unknown BLOTTO_ALLOCATION_MODE: {BLOTTO_ALLOCATION_MODE}")
        # ------------------- end allocation scheduling -------------------

        # --------------------------- Outcomes ----------------------------
        new_tokens = dict(self.tokens)
        new_brains = dict(self.brains)
        walker_resets: List[int] = []

        if BLOTTO_MODE == "ALLOCATE_AND_CONQUER":
            for v in list(self.G.nodes()):
                offers_map = per_target_allocators[v]
                log["incoming_offers"][str(v)] = [(int(s), int(a)) for (s, a) in offers_map.items()]
                if not offers_map:
                    new_tokens[v] = 0
                    new_brains[v] = self.brains[v].copy()
                    if MUTATE_ON_BLOTTO_COPY:
                        new_brains[v].mutate()
                    if RESET_REACH_ON_CONQUER:
                        neighbors_vid = [int(w) for w in self.G.neighbors(v)]
                        self.reach_counts[v] = {int(v): 1, **{nv: 1 for nv in neighbors_vid}}
                        walker_resets.append(int(v))
                    continue

                max_amt = max(offers_map.values())
                contenders = [s for s, a in offers_map.items() if a == max_amt]
                winner = int(np.random.choice(contenders))

                # Implant winner's brain into v
                new_brains[v] = self.brains[winner].copy()
                if MUTATE_ON_BLOTTO_COPY:
                    new_brains[v].mutate()

                new_tokens[v] = int(incoming_totals[v])
                log["winners"][str(v)] = {"winner": int(winner), "max_amount": int(max_amt)}

                if winner != v and RESET_REACH_ON_CONQUER:
                    neighbors_vid = [int(w) for w in self.G.neighbors(v)]
                    self.reach_counts[v] = {int(v): 1, **{nv: 1 for nv in neighbors_vid}}
                    walker_resets.append(int(v))

        elif BLOTTO_MODE == "ALLOCATE_AND_ROB":
            # Winners collect tokens from targets; brains are never copied here.
            winnings: Dict[int, int] = {u: 0 for u in self.G.nodes()}
            for v in list(self.G.nodes()):
                offers_map = per_target_allocators[v]
                log["incoming_offers"][str(v)] = [(int(s), int(a)) for (s, a) in offers_map.items()]

                if not offers_map:
                    new_tokens[v] = 0
                    new_brains[v] = self.brains[v]  # unchanged
                    continue

                max_amt = max(offers_map.values())
                contenders = [s for s, a in offers_map.items() if a == max_amt]
                winner = int(np.random.choice(contenders))

                winnings[winner] = winnings.get(winner, 0) + int(incoming_totals[v])
                new_tokens[v] = 0
                new_brains[v] = self.brains[v]  # no copying
                log["winners"][str(v)] = {"winner": int(winner), "max_amount": int(max_amt)}

            # Apply winnings to winner nodes
            for w, gain in winnings.items():
                new_tokens[w] = new_tokens.get(w, 0) + int(gain)

        else:
            raise ValueError(f"Unknown BLOTTO_MODE: {BLOTTO_MODE}")

        log["walker_resets"] = walker_resets

        # Commit outcome state
        self.tokens = new_tokens
        self.brains = new_brains

        # Optional global mutation at end of blotto (applies to both modes)
        if MUTATE_ALL_AFTER_BLOTTO:
            for b in self.brains.values():
                b.mutate()

        # -------------------- Prune, cleanup, persist --------------------
        # Prune edges with no flow
        to_remove = [e for e, f in edge_flow.items() if f == 0]
        if to_remove:
            self.G.remove_edges_from(to_remove)
        log["pruned_edges"] = [(int(u), int(v)) for (u, v) in to_remove]

        # Cleanup, persist
        log["cleanup"] = self._cleanup_and_redistribute()
        log["post_state"] = self._snapshot_graph()


        # Messaging dictionary hygiene: at end of Blotto, keep only self+neighbors
        # and only entries pointing to nodes still present.
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

        # Aggregated per-agent allocation counts (compat) + exact sequence
        for u in list(self.G.nodes()) + [x for x in allocation_sequence.keys() if x not in self.G.nodes()]:
            tgs = [u] + (neighs.get(u, []))
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
        path = self._save_step_file(2 * t + 1, log)
        if t % 10 == 0 and DRAW:
            self._draw(f"Round {t} — After Phase 2", f"step_{2 * t + 1:05d}_phase2.png")
        return path

    # ----------------- One full round -----------------
    def step(self, t: int) -> Tuple[str, str]:
        p1 = self.reproduction_phase(t)
        p2 = self.blotto_phase(t)
        return p1, p2


# ----------------------------------------------------------------------------
# Demo / Quickstart
# ----------------------------------------------------------------------------

def _main() -> None:

    total_tokens = 20_000
    max_steps = 500_000

    n = int(total_tokens/100)
    k = int(n/100)

    def make_simulation() -> GraphOfLife:
        G0 = nx.watts_strogatz_graph(n=n, k=k, p=0.2)
        return GraphOfLife(G0, total_tokens=total_tokens)

    run_counter = 0
    while True:
        simulation = make_simulation()
        run_counter += 1
        print(f"🌍 Starting run {run_counter}, folder: {simulation.run_dir}")
        for t in range(max_steps):
            simulation.step(t)
            print(f"Run {run_counter}, iteration {t} finished (nodes: {simulation.G.number_of_nodes()}, edges: {simulation.G.number_of_edges()})")
            if simulation.G.number_of_nodes() <= 10:
                print(f"⚠️ Run {run_counter} crashed at step {t} (nodes ≤ 10), restarting…")
                break
        else:
            print(f"✅ Run {run_counter} finished {max_steps} iterations, restarting fresh…")


if __name__ == "__main__":
    _main()
