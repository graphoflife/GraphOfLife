# Graph of Life

**A new kind of Artificial Life Algorithm**

**Exploring open-ended evolution with spatial evolutionary game theory,
graph theory and neural networks.**

![A run in progress: several thousand agents on the graph they built, coloured
by how many tokens each holds](docs/images/graph.png)

Agents live on the nodes of a graph, each carrying a small neural network. They
spend a conserved supply of tokens to reproduce, rewire the graph they live on,
and fight each other in a game of blotto for position, tokens and reproduction.
Nothing is optimised and nothing is selected for by hand. Instead, Natural
Selection decides what survives and mutations are how they can evolve.

**▶ Try it in your browser: <https://graphoflife.github.io/GraphOfLife/>**

No install, no account, no server. The page runs the same Python engine this
repository contains, compiled to WebAssembly, entirely on your own machine.

---

## Contents

- [What the simulation does](#what-the-simulation-does)
- [The two phases](#the-two-phases)
- [Different mechanics](#different-mechanics)
- [Watching a run](#watching-a-run)
- [Running it yourself](#running-it-yourself)
- [Checking that it still works](#checking-that-it-still-works)
- [How the website works](#how-the-website-works)
- [Goal](#goal)
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
a node takes it. Total Token Amount is conserved.

Afterwards the world is cleaned up: starved agents and edges with no token flow
are removed, anything detached from the largest connected component is culled,
and the tokens freed are redistributed.

## Different mechanics

Messages, handover, revolutions and rewiring can each be switched off, and the
brain comes in three kinds. All of them change what the brain has to decide,
so they are fixed when a run is created rather than partway through. Deciding
its own randomness is not optional — it is how every choice is read.

**Brains.** Three kinds, chosen with `brain_kind`. `float` uses 64-bit weights,
`float16` the same arithmetic on a quarter of the memory, and `binary` weights
of only −1, 0 and +1 with hidden units that are simply on or off. The binary
brain uses no floating point anywhere, so its runs come out identical on any
machine. It needs a gentler mutation rate than the others — around 0.02 rather
than 0.1 — because a whole-number weight cannot make a small move.

**Agents decide their own randomness.** Every choice comes as a pair of
outputs: one says what to do, the other says how to read it. Take the best
option, or sample among them in proportion to their scores. That second output
is part of the genome, so whether a lineage decides sharply or gambles is
itself something evolution settles — and it can differ from one decision to
the next.

**Exchange messages.** An agent writes a short list of numbers to each of its
neighbours, and a separate one to itself. Next phase it reads what its
neighbours wrote to it, and what it wrote to itself, which gives it a memory.
Nothing forces a message to mean anything. Whatever they come to signal is
whatever survives.

**Handover.** A parent can give one of its own edges to its newborn instead of
copying it. The parent ends up with one connection fewer and the child with one
more, so a lineage can pass on position and not only tokens.

**Revolutions.** The winner of a node is not automatically whoever paid most.
Part of each agent's stake is a revolt token, and a coalition can form against
the leader — so the largest bid can be beaten by agreement among smaller ones.

**Rewire.** An agent can hand one of its edges to another of its neighbours.
The connection (agent, other) becomes (recipient, other): the agent drops out
of the middle and the two it stood between are left joined directly. A rewire
never creates an edge. The count stays the same, or falls by one where the two
collapse into the single edge a simple graph can hold.

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
  laws — including cycle rank, bridges, triangles, transitivity, entropy, an
  estimated dimension, and sampled radius, diameter and mean path length. Click
  any of them for an explanation and its history.
- **Analyses.** The same run can be read in several ways. The power-law group
  fits an exponent and an R² to each relationship: how wealth scales with
  degree, how triangles scale with degree, and the shape of the degree and
  token distributions. Clustering against degree is the one that measures
  self-similarity — a slope near −1 means dense small neighbourhoods nested
  inside sparser larger ones, the same arrangement at every scale. Token
  curvature, the neighbours' tokens less an agent's own times its degree, read
  against the change that followed, behaves like a law of diffusion.

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

## Goal

The goal is open-ended evolution: a system that keeps producing genuinely new
behaviour instead of settling on one way of playing and staying there.

That is why the game is built the way it is. There should be no best strategy —
every way of playing ought to be beatable by some other, so there is always
somewhere left to go. The rules are kept as simple as they can be while still
leaving that room, since a complicated game can hide novelty that came from the
rules rather than from the agents.

Whether this particular set of rules gets there is an open question, and the
viewer and the statistics exist to answer it.

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
