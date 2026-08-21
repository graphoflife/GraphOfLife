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
- [Checking that it still works](#checking-that-it-still-works)
- [Reproducibility](#reproducibility)
- [How the website works](#how-the-website-works)
- [What the numbers mean](#what-the-numbers-mean)
- [Is this open-ended evolution?](#is-this-open-ended-evolution)
- [Repository layout](#repository-layout)
- [Licence](#licence)

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
- **Statistics** grouped into general, reproduction, game, structure and power
  laws — 
  including cycle rank, bridges, triangles, transitivity, entropy, an estimated
  dimension, and sampled radius, diameter and mean path length. Click any of
  them for an explanation and its history.

**Power laws** get their own group, each relationship reported as an exponent
and the R² of its log-log fit. The degree and token distributions are fitted
from their complementary CDFs rather than from binned histograms, which are
worthless out in the tail. Alongside them: how wealth scales with degree, how
triangles scale with degree, how the size of a token change scales with
wealth, and the degree assortativity.

The one to watch is **clustering against degree**. A slope near −1 is the
signature of a hierarchical, self-similar network — small dense neighbourhoods
grouped into larger sparser ones, the same arrangement repeating at every
scale. A flat slope means a hub's neighbourhood looks like a leaf's, and there
is no hierarchy at all.

R² is reported because it is worth knowing before trusting an exponent, not
because it settles anything. A straight line on log-log is famously weak
evidence for a power law — log-normal and stretched-exponential distributions
look just as straight over two decades. Read it as "is this exponent
meaningful", not as "is this network scale free".

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

## Checking that it still works

```bash
python3 tests/test_engine.py        # invariants: tokens, topology, resuming
python3 tests/test_stats_parity.py  # the two statistics implementations agree
```

Neither needs pytest, though `python3 -m pytest tests/` works too. A research
repository whose tests need a toolchain installed first is a repository whose
tests do not get run.

The invariants are properties rather than expected values — a simulation whose
point is that nobody knows what it will do cannot be tested by writing down
what it should do, but it can be held to what must be true regardless. Tokens
are conserved. A rewire never invents an edge. Rewiring does not depend on the
order agents are visited in. A seed reproduces a run.

The parity test exists because forty-odd statistics are computed twice — in
Python for the charts, in JavaScript for the panel under the graph — and two
implementations of the same thing drift. They had: an optimisation to the
JavaScript edge-flow map quietly began skipping agents culled during cleanup,
so the panel and the chart disagreed about the same frame and neither looked
wrong on its own. Both now run over identical frames and every shared value is
compared, including the structural ones, which are genuinely independent
implementations on the two sides.

Nothing is published until both pass; the deploy is gated on them.

## Reproducibility

A seed reproduces a run, and a checkpoint resumes one exactly. Neither was true
until recently, and both failed for the same kind of reason.

`numpy.random.seed` does not reach networkx, which draws the starting graph
from the `random` module, so two runs with the same seed began from different
graphs. The seed is passed through now.

Resuming was worse, because nothing about the saved state looked wrong: the
graph, the tokens, the brains and the random stream all came back correctly.
What did not was the *order* of each node's neighbours. networkx keeps
adjacency in insertion order, so a graph rebuilt from an edge list presents its
neighbours differently from the one it was copied from — and the engine reads
neighbours as the columns of a matrix, so a column meant a different agent and
every decision naming a neighbour by column landed elsewhere. Neighbours and
the agent loop are both sorted by id now, which makes the run independent of
how the graph was arrived at.

Two things still differ between machines. Matrix multiplication is not
bit-identical across BLAS implementations, and this simulation is chaotic
enough to turn a last-bit difference into a different history — so the same
seed gives the same run on the same machine, not across machines or between
native Python and the browser. And messages in flight are part of the world
state; they are checkpointed now, but a checkpoint written before that is
resumed without them.

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

**Runs are saved in your browser.** Frames go into IndexedDB one at a time as
they are produced, so a run that is still going is already saved and closing
the tab loses only the iteration in flight. Reopening the page finds your runs
where you left them, ready to inspect. If checkpointing is on — it is by
default — a run can also be carried on from where it stopped, because the
checkpoint restores the agents' brains and the random stream along with them.

Nothing leaves your machine. Clearing your browser's site data removes it, and
a browser short of disk space may evict it, so the page asks for persistent
storage and tells you how much has been used.

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
LICENSE                MIT — use it for anything, including commercially
GraphOfLifeSimple.py   the engine: agents, brains, both phases, cleanup
gol_config.py          every setting, with validation
gol_store.py           runs on disk — gzipped frames and a rolling checkpoint
gol_series.py          per-frame statistics for the charts
gol_server.py          the local server (standard library only)
build_site.sh          assembles the static site into _site/
web/                   the interface: renderer, layout, charts
  js/viewer.js         frames, camera, playback, the animation loop
  js/viewer-controls.js  the settings panel, and keeping it in step
  js/viewer-focus.js   cropping to a neighbourhood, and fullscreen
  js/viewer-panels.js  statistics, charts and the hover card
  js/sim-worker.js     the browser backend — Pyodide, running the engine above
  js/runstore.js       runs, frames and checkpoints in IndexedDB
  py/gol_browser.py    live worlds, advanced a slice at a time
tests/                 invariants, and parity between the two statistics
docs/IDEAS.md          what might come next
```

## Licence

MIT. Use it, change it, build on it, sell it — the only condition is that the
notice travels with it. See [LICENSE](LICENSE).

The original four-year research codebase this grew out of is preserved whole on
the [`original-engine`](https://github.com/graphoflife/GraphOfLife/tree/original-engine)
branch, including the `Old/` folder of earlier variants. It is not duplicated
here; the branch is the copy.
