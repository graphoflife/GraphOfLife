# Graph of Life

**Exploring Open‑ended-Evolution with Spatial-Evolutionary-Game-Theory, Graph Theory and Neural Networks**

*An Attempt at Engineering Artificial Life*


Graph of Life is a compact research engine for exploring **open‑ended evolution** in a spatial setting. Agents live on the nodes of an undirected graph and carry tiny neural “brains.” Each iteration has two phases.

1. **Reproduction phase** — agents may reproduce and rewire the topology of the graph.
2. **Blotto phase** — agents compete for positions via a one‑token‑at‑a‑time allocation game (inspired by the Colonel-Blotto-Game).

Over time, natural selection eliminates fragile behaviors (in context of the given game rules). The aim is to find a game, where no optimal strategy exists such that they can evolve forever. For that the game was designed as simple as possible but as complex as necessary. 

The current version of **GraphOfLife.py** is the result of a four year endeavor of exploring different game rules and setups. This specific version is now the baseline simulation for further analysis because of its simplicity while still being able to create complex emergent behavior. Different game rules, setups and implementations lead to different emerging behaviors. All the past versions and variations can be found in the Folder *Old*.

---

## Table of contents
- [Motivation](#motivation)
- [Quick start](#quick-start)
- [Outputs & logging](#outputs--logging)
- [World model](#world-model)
- [Algorithm overview](#algorithm-overview)
  - [Phase 1 — Reproduction](#phase-1--reproduction)
  - [Phase 2 — Blotto (1‑token rounds)](#phase-2--blotto-1token-rounds)
  - [Brains & feature vectors](#brains--feature-vectors)
  - [Cleanup, LCC, and resurrection](#cleanup-lcc-and-resurrection)
- [Configuration flags](#configuration-flags)
- [Performance tips](#performance-tips)
- [Roadmap](#roadmap)
- [License](#license)

---

## Motivation

This project aims to **engineer artificial life** in a graph‑based system called **Graph of Life**, drawing on **spatial evolutionary game theory**. Agents are connected in a network and we delegate as many decisions as possible to them. Each agent has its **own neural network** (a small feed‑forward brain) that observes local state and decides what to do. By letting agents compete and evolve under resource constraints, the system becomes **self‑organizing**.

Agents control the **topology** of the network: they can create new links, drop or move existing ones, and effectively choose who they interact with. Interactions are **local**—agents only interact with their neighbors—and they compete for **scarce fungible tokens** in order to survive and reproduce. If the graph splits in multiple sub graphs, only the largest survives, giving an incentive to cooperate, since the survival of a critical node can be important for many nodes.

A manually defined **total token budget** is a crucial constant: an agent needs at least one token to survive. To give an incentive to attack each other, all unused edges in each iteration get deleted (unused meaning: no token was sent along this edge during the blotto phase to attack another). So the token budget implicitly bounds the network’s size. This makes it easy to run small simulations on a laptop while also enabling, in principle, much larger runs on clusters or supercomputers. Over many iterations, **natural selection** eliminates fragile behaviors and promotes strategies that learn to survive and reproduce.

The competitive interaction is inspired by the **Colonel Blotto** game from game theory (our implementation uses one‑token‑at‑a‑time rounds). Other competitive games could be plugged in as well. Unlike many evolutionary game‑theory algorithms, **reproduction is not global**; it is **local** and emerges from the decisions agents make, so it becomes part of the strategies themselves. Newly reproduced agents are embedded into the existing network according to those decisions.

> **Open‑ended evolution?** Further analysis is needed to determine whether the system exhibits genuinely open‑ended evolution in the long run.

---

## Quick start

```bash
# Clone the repo
git clone https://github.com/graphoflife/GraphOfLife
cd GraphOfLife

# (Optional) create a venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Run the demo
python GraphOfLife.py
```

The demo writes a timestamped folder in `GraphOfLifeOutputs/` with JSON logs (`step_00000.json`, …) and a copy of `GraphOfLife.py` plus a `.sha256` checksum for reproducibility.

> **Note:** The default demo uses a Watts–Strogatz graph (small‑world) and may run for a while. Tweak `n`, `k`, `total_tokens`, and `max_steps` inside `_demo()` to suit your machine.

---

## GraphOfLife Class 

- **Graph**: undirected, dynamic. Agents occupy nodes; edges represent interaction/neighborhood.
- **Tokens**: dictionary that stores integer amount of tokens per node. Total tokens are conserved; tokens can "flow" across the graph by reproduction, redistribution and token allocation during blotto attack.
- **Brains**: dictionary that stores brains of nodes. Small feed‑forward networks (sigmoid hidden layers, linear output). Brains are **copied** during reproduction and during Blotto when a winner implants into the occupied node; after each Blotto phase, all surviving brains mutate.
- **Reach memory**: per‑agent multiset of nodes reachable by random walks (with “stay”). Used to propose one new link candidate per iteration per agent.

---

## Algorithm overview
Each iteration consists of two phases.

### Phase 1 — Reproduction
For each agent `u` with tokens:
1. **Observe**: build inputs for `[u + neighbors + (optional) far node]`. Inputs include own/target tokens & degrees and neighborhood **quantiles**.
2. **Reproduction decision**: aggregate reproduction logits over core candidates, convert to a probability of spawning a child, and split tokens (at least one if reproducing).
3. **Child links**: for each core candidate, a yes/no head decides whether the child connects to that node.
4. **Shift**: optionally move some of `u`’s existing edges to the child.
5. **Reconnection**: optionally sever one edge and attach `u` to a different neighbor.
6. **Walker link**: optionally create a new edge toward the far candidate (sampled from reach memory by weight).
7. **Apply edits simultaneously** (to avoid order effects), remove self‑loops.
8. **Cleanup**: remove zero‑token nodes and keep only the **largest connected component**; redistribute tokens from removed components uniformly; . If the graph is empty, **resurrect** a single agent holding all tokens.

### Phase 2 — Blotto
We play a round‑based Colonel Blotto variant:
1. **Initialize**, track the remaining tokens to allocate `remaining[u] = tokens[u]`.
2. **While any agent can still allocate a token (remaining[u]>=1)**:
   - **Current Blotto Snapshot**: Freeze a **snapshot** of the current per‑target allocation totals and current leaders (max allocations and which sources tie at the max).
   - **Observe** Every eligible agent independently evaluates the same snapshot and **allocates exactly 1 token** to either itself or one neighbor. Decision uses the Brain’s Blotto head.
   - **Allocation** Apply all choices **simultaneously**; update per‑target totals, per‑edge flow, and per‑agent `allocation_sequence`.
3. **Determine Winners**: for each node `v`, the source(s) with the **maximum** offers are the contenders; ties break uniformly. The winner’s brain is **copied** into `v`. The node’s token count becomes the **sum** of all offers it received. (Nodes can either reproduce by reproduction or by winning the blotto game at multiple nodes)
4. **Pruning**: remove edges with **zero** token flow this blotto phase.
5. **Cleanup**: (Same as in reproduction phase)
6. **Mutation** then **mutate** all surviving brains.

### Brains & feature vectors
A Brain outputs 12 values, organized as heads:
```
REPRO:      rows 0..1  (yes/no and how much tokens to give to spawn)
LINK:       rows 2..3  (yes/no)
SHIFT:      rows 4..5  (yes/no)
RECONNECT:  rows 6..8  (no, which_link, which_target)
BLOTTO:     row  9     (scalar score per candidate)
WALKER:     rows 10..11 (yes/no)
```

**Inputs (scaled by 0.1, except a leading 0/1 self‑indicator):**
- Base (29 dims):
  - own tokens, target tokens, own degree, target degree
  - neighbor‑token quantiles **(q0, q20, q40, q60, q80, q100)** of **own** neighbors and of the **target**’s neighbors (12 dims)
  - neighbor‑degree quantiles of **own** neighbors and of the **target**’s neighbors (12 dims)
- Blotto extras (8 dims; **zeroed during Reproduction**):
  1. `total_on_v` — total tokens currently allocated to target `v`
  2. `current_max_on_v` — maximum allocated by any single bidder on `v`
  3. `edge_has_flow` — 1 if any token flowed on `(u,v)` so far this phase (1 for self‑target)
  4. `u_to_v` — tokens already sent by `u` to `v`
  5. `v_to_v` — tokens `v` sent to itself
  6. `u_wins_v_now` — **tie‑aware** win probability for `u` on `v` (`1/|leaders|` if `u` is among leaders and `current_max_on_v>0`, else 0)
  7. `remaining_alloc_u` — tokens `u` still can allocate
  8. `remaining_alloc_v` — tokens `v` still can allocate

> Determinism: if `PROBABILISTIC_DECISIONS=False`, decisions use greedy argmax; otherwise they are sampled from ReLU‑normalized scores.

### Cleanup, LCC, and resurrection
- After both phases, a cleanup step ensures **token conservation** and prevents fragmentation by keeping only the **largest connected component**.
- Tokens from removed components are **redistributed uniformly** to survivors.
- If the graph empties out, a single node is **resurrected** with all tokens (fresh brain).

---

## Configuration flags
Set these at the top of `GraphOfLife.py`:

- `PROBABILISTIC_DECISIONS` (bool): sample vs. greedy choices
- `DRAW` (bool): save k‑core visualizations every 10 steps
- `BASE_DIR` (path): where run folders are written (default `GraphOfLifeOutputs/`)

---

## Outputs & logging
Each iteration produces a `step_<index>.json` file. Both phases record their own logs:

**Common fields for Reproduction phase and Blotto phase**
- `pre_state` — nodes (id, tokens, brain_id, degree, neighbors) and edges
- `post_state` — same shape after the phase completes
- `cleanup` — explicit report: resurrected, removed nodes/components, redistributed tokens, survivors_count

**Reproduction phase extras**
- `decisions` — per‑agent record containing:
  - reproduction logits (aggregated), tokens split, if a child was created
  - child’s link choices, any edge **shifts** from parent to child
  - **reconnections** (drop one neighbor, attach to another)
  - walker link decision & candidate far node

**Blotto phase extras**
- `allocation_sequence` — for each agent, an **ordered list** of targets (index 0 is the first token it allocated this phase)
- `allocations` — per‑agent **aggregated** counts over its `[self + neighbors]`
- `incoming_offers` — for each target node: list of `(source, amount)` at the end of the phase
- `winners` — winning source per node (ties broken uniformly at random)
- `pruned_edges` — edges with zero flow during Blotto were removed
- `walker_resets` — nodes whose reach memory was reset (lost their own place)

**Reproducibility sidecars**
- A verbatim copy of the Python file that produced the run
- A `*.sha256` checksum of that file
- `source_meta.json` with an optional Git commit hash (best‑effort)

---

## Performance tips
- The computational expense and needed storage is dependent on the `total_tokens` amount: Start with small `n`, `k`, and `total_tokens` to gauge speed.
- Consider turning `DRAW=False` unless you need visualizations.

---

## Roadmap
- More observation features (e.g., neighborhood stats of allocations per round)
- Rich analysis scripts for the JSON logs (envelopes, survivorship, core dynamics, open-ended analysis, identify correlations, power laws and other dependencies)
- Scale Simulations
- More efficient implementation
- Optional GPU‑backed inference for larger graphs

---

## License

SPDX-License-Identifier: MIT

---

*If you use this engine in research, consider citing the repository so others can reproduce your setup. PRs and ideas are welcome!*

