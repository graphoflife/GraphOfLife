# Graph of Life

**Exploring open-ended evolution with spatial evolutionary game theory, graph theory and neural networks.**

Agents live on the nodes of a graph, each carrying a small neural network. They
spend a conserved supply of tokens to reproduce, rewire the graph they live on,
and fight each other for position. Nothing is optimised and nothing is
selected for by hand — whatever survives, survives.

**▶ Try it in your browser: <https://graphoflife.github.io/GraphOfLife/>**

No install, no account, no server. The page runs the same Python engine this
repository contains, compiled to WebAssembly, entirely on your own machine.

---

## Contents

- [What the simulation does](#what-the-simulation-does)
- [The two phases](#the-two-phases)
- [Agents decide their own randomness](#agents-decide-their-own-randomness)
- [Optional mechanics](#optional-mechanics)
- [Watching a run](#watching-a-run)
- [Running it yourself](#running-it-yourself)
- [How the website works](#how-the-website-works)
- [What the numbers mean](#what-the-numbers-mean)
- [Is this open-ended evolution?](#is-this-open-ended-evolution)
- [Repository layout](#repository-layout)

---

## What the simulation does

The world is an undirected graph. Every node is an agent; every agent holds
some number of tokens and a feed-forward neural network — its *brain*. Tokens
are the only currency and, by default, the total never changes. There is no
fitness function. An agent that runs out of tokens is removed.

A brain is never trained. It is copied from a parent with mutation, and that is
the only way behaviour ever changes. Everything an agent does — whether to
reproduce, how much to invest, who to attack, which edge to give away — is read
off the outputs of that network, applied to what it can see: its own tokens and
degree, its neighbours' tokens and degrees, messages passed to it, and some
noise.

## The two phases

Each iteration is two phases, and the viewer records a frame after each.

**1. Reproduction.** An agent decides what fraction of its tokens to spend on a
child. The child is placed on a new node, inherits a mutated copy of the brain,
and is wired to whichever of the parent's neighbours the parent chooses. The
parent pays the full cost.

**2. The game.** Every agent distributes its tokens across itself and its
neighbours — a Colonel Blotto game played on the graph. Whoever commits most to
a node takes it. Tokens spent are not lost; they move. Losing every node you
hold means starvation.

Afterwards the world is cleaned up: starved agents are removed, anything
detached from the largest connected component is culled, and the tokens freed
are redistributed.

## Agents decide their own randomness

Every discrete choice comes in a pair. One output says *what* to do; a second,
the **mode**, says whether to take the maximum or to sample in proportion to
the scores. Determinism is therefore not a global setting — it is part of the
genome, and evolvable. An agent can become reliably predictable in one decision
and stay a gambler in another.

## Optional mechanics

Three rules can be switched on or off per run. They change the brain's output
layout, so they are fixed when the run is created.

| Mechanic | What it does |
|---|---|
| **Handover** | A parent gives one of its own edges to its newborn instead of copying it. The parent ends up with one fewer. |
| **Revolutions** | The winner of a node is not automatically whoever paid most. A coalition can form against the leader, and a fraction of each agent's stake is a revolt token. |
| **Rewire** | An agent hands one of its edges to another of its neighbours. The edge `(agent, other)` becomes `(recipient, other)` — the agent drops out of the middle and the two it stood between are joined directly. |

Rewiring is applied against the graph as it stood before any rewire moved, so
the outcome does not depend on the order agents are visited in. Both ends of an
edge can ask to give the same edge away; the contest is settled by drawing one
claim rather than by whichever agent sorts first.

## Watching a run

The viewer is the point of the project. A run is recorded frame by frame and
can be replayed, stepped, and measured.

- **The graph**, in 2D or 3D, with a force layout that carries positions
  between frames so you can watch structure persist rather than re-form. A
  newborn is placed beside its nearest surviving ancestor, so lineages stay
  together even when playback skips generations.
- **Colour and size** by any of two dozen quantities — tokens, degree, token
  change, curvature, loops through a node, triangles, lineage — for nodes and
  for edges, each with its own log toggle.
- **Focus.** Click a node to keep only what lies within a few steps of it. When
  that node dies the view follows its densest surviving neighbour, and only
  gives up when the whole neighbourhood has gone.
- **Three charts**: the distribution of any quantity, a two-dimensional heatmap
  pairing any two, and a trajectory tracing how two statistics move together
  over the whole run, coloured by time.
- **Statistics** grouped into general, reproduction, game and structure — 
  including cycle rank, bridges, triangles, transitivity, entropy, an estimated
  dimension, and sampled radius, diameter and mean path length. Click any of
  them for an explanation and its history.

**Token curvature** deserves a note. For each agent it is the sum of its
neighbours' tokens minus its own times its degree — the graph Laplacian applied
to wealth. Positive means an agent sits in a valley poorer than its
neighbourhood; negative means it is a peak its neighbours drain toward. It sums
to exactly zero over the graph. Measured on the state *before* a phase and
plotted against the change that phase produced, it behaves like a diffusion
law: on one 1,949-node frame the correlation was 0.76.

## Running it yourself

The website is capped by what a browser tab can hold. For real work, run it
locally — same engine, no ceiling, and runs are written to disk.

```bash
git clone https://github.com/graphoflife/GraphOfLife.git
cd GraphOfLife
pip install -r requirements.txt
python3 gol_server.py
```

That serves the same interface at <http://127.0.0.1:8000/> and stores runs in
`GraphOfLifeRuns/`. The engine also runs headless:

```bash
python3 GraphOfLifeSimple.py
```

A run has no iteration ceiling. It goes until you stop it or the population
dies out.

## How the website works

There is no backend. The page decides once, on load, where the simulation
should run:

- If `/api/defaults` answers, there is a local server and it uses that.
- If nothing answers — which is the case on GitHub Pages — it starts a Web
  Worker, loads [Pyodide](https://pyodide.org), and imports
  `GraphOfLifeSimple.py` **unchanged**.

The engine is not reimplemented in JavaScript. A translation would have been a
second implementation to keep honest, and this simulation is far too sensitive
to arithmetic for that to hold. Measured against native CPython on the same
seed, the browser costs about **1.3×** at the default brain size.

Two things are worth knowing about the browser version:

**Runs live in the page.** Frames carry the whole topology, so a few thousand
agents over a few thousand iterations is already hundreds of megabytes. Keeping
them past a reload would mean deciding what to throw away, and being plainly
temporary beats being quietly lossy.

**A seed does not reproduce across machines.** The random number stream is
bit-identical everywhere — verified. Matrix multiplication is not: native
OpenBLAS and the WebAssembly build round differently in the last bit. The
simulation is chaotic and makes decisions by comparing floats, so a one-ulp
difference eventually flips a comparison and the trajectories part. With a
small brain a run matched for 20 iterations exactly; with the default brain it
diverged after three. This is not a quirk of the browser — two machines with
different BLAS libraries do the same thing.

## What the numbers mean

Some statistics are estimates, and the viewer says so rather than implying a
precision it does not have.

- **Diameter and radius** come from sampled breadth-first sweeps plus a second
  sweep from the furthest node found. The diameter can only under-report and
  the radius can only over-report. Exact on paths, cycles, stars, grids and
  trees; about one step short on small-world graphs.
- **Dimension** measures how fast a ball grows with radius, in the spirit of
  the Wolfram Physics Project. Calibrated against lattices it returns 1.00 for
  a chain, 1.92 for a square grid and 2.56 for a cubic one — exact in one
  dimension, increasingly conservative above it. Read it as an index, not a
  measurement.
- **Time series are sampled** to at most 1,000 iterations, since a chart a few
  hundred pixels wide cannot show more and the structural statistics are far
  too slow to compute for every frame.

## Is this open-ended evolution?

Measured over 27,366 iterations of one run: no, not yet. Every statistic was
stationary, the strategy heads had saturated, and the median agent held three
tokens across five targets. Conserved tokens against a growing population
collapse the decision space; mutation is unconditional and cannot itself
evolve; the genome is a fixed size, so novelty is exploratory rather than
expansive.

That is a result, not a failure — it says what would have to change. Evolvable
mutation rates, a token supply that grows, and variable-size brains are the
obvious candidates. The interface exists so that these questions can be asked
of a run rather than guessed at.

## Repository layout

```
GraphOfLifeSimple.py   the engine: agents, brains, both phases, cleanup
gol_config.py          every setting, with validation
gol_store.py           runs on disk — gzipped frames and a rolling checkpoint
gol_series.py          per-frame statistics for the charts
gol_server.py          the local server (standard library only)
build_site.sh          assembles the static site into _site/
web/                   the interface: viewer, renderer, layout, charts
  js/sim-worker.js     the browser backend — Pyodide, running the engine above
  py/gol_browser.py    run management without a disk
```

The original four-year research codebase this grew out of is preserved on the
[`original-engine`](https://github.com/graphoflife/GraphOfLife/tree/original-engine)
branch, including the `Old/` folder of earlier variants.
