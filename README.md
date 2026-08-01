# GraphOfLife

Open-ended evolution on a mutable graph, with a local web UI for running and
inspecting simulations.

Agents live on the nodes of a graph. Each owns a pool of **tokens** — wealth,
life, and voting power at once, globally conserved — and a **brain**, a small
feed-forward network that is never trained by gradients and evolves only by
copy and mutation. Nothing is optimized toward a goal. Whatever survives,
survives.

## Running it

Requires Python 3.10+, `numpy`, and `networkx`.

```bash
python3 gol_server.py
```

That opens the UI at <http://127.0.0.1:8000>. It binds to localhost and serves
only from `web/` — everything stays on the machine you run it on. Clone the
repo anywhere, generate data there, inspect it there.

To run headless without the UI:

```bash
python3 GraphOfLifeSimple.py
```

## How one step works

Each iteration runs two phases.

**Phase 1 — reproduction.** Every agent observes itself and its neighbors in a
single batched forward pass, decides what fraction of its tokens to endow a
child with, and chooses which of those candidates the newborn is wired to. The
child gets a mutated copy of the parent's brain, paid for out of the parent's
own tokens. Nothing else about the topology changes.

**Phase 2 — Blotto.** Every agent spends its entire pool bidding on itself and
its neighbors. Part of each allocation can be flagged as *revolution* tokens.
The largest single allocator to a node is the **hegemon**; every other
revolutionary forms a **mob**, sorted weakest-first. Walking up the mob, the
first point where the accumulated lower class outweighs everyone above it plus
the hegemon is where the revolution succeeds — so a coalition of small
allocators can unseat someone who outspent all of them individually. The winner
implants its brain into the node. Then every brain mutates and edges that
carried no tokens are pruned.

**Cleanup.** Broke nodes starve, only the largest connected component survives,
and the estate of the dead is scattered uniformly over the survivors.

### Agent-controlled randomness

There is no global "be probabilistic" switch. Each discrete decision is paired
with a mode head, and the agent decides for itself whether its outputs are read
as a probability or a hard maximum. That choice is part of the genome, so it
evolves too.

## Files

| File | Role |
| --- | --- |
| `GraphOfLifeSimple.py` | The engine. Holds no file paths and no UI. |
| `gol_config.py` | `SimConfig` — every knob the UI can set. |
| `gol_store.py` | Run directories, gzipped frames, checkpoints. |
| `gol_server.py` | Local HTTP server and JSON API. Standard library only. |
| `web/` | The viewer: canvas rendering, force layout, controls. |
| `GraphOfLife.py` | The original single-file version, kept for reference. |

## Data on disk

Runs live in `GraphOfLifeRuns/<run_id>/`:

- `meta.json` — name, status, config, progress
- `checkpoint.npz` — the single rolling resume point
- `frames/frame_NNNNN.json.gz` — one file per recorded phase

Frames store topology, tokens, brain ids, and **parent ids** for both node and
brain lineage, plus the decisions each agent made (the outcomes, not the raw
logits). Frames are numbered sequentially; the iteration and phase live inside
each frame.

### Checkpoints and resuming

Only **one** checkpoint is kept per run, overwritten every `checkpoint_every`
iterations. This matters because checkpoints are large: at the default brain
size (9,896 parameters) a run with a few thousand agents writes roughly 100 MB
each time.

Resuming **truncates** every frame recorded after the checkpoint. If a run
reached iteration 27 but its checkpoint is at 20, resuming discards frames for
21–27 and replays them. The recorded history always matches the saved state
rather than describing a future the resumed world never lived through.

The numpy RNG stream is stored in the checkpoint, so a resumed run continues the
same random sequence rather than merely a statistically similar one.

## Viewer

Step through frames with the arrow keys (`←`/`→` one frame, `↑`/`↓` a whole
iteration, `Shift` for ten, `Home`/`End`, `Space` to play). Drag to pan, scroll
to zoom, hover a node for its details.

Node colour and size, edge colour and width, opacities, colour maps, and the
background (solid, radial, or linear gradient) are all independently
configurable, along with what each channel represents. Edge **token flow** is
derived from the recorded phase-2 allocations, so it is only available on
phase-2 frames of runs that recorded decisions.

Layout is a force simulation running in the browser. Positions carry across
frames and newborns are seeded beside their parent, so stepping forward does not
scramble the picture. Repulsion is evaluated against a spatial grid rather than
all pairs, which is what lets it handle a few thousand nodes.

## Performance

Runtime scales with `total_tokens`, not just node count. The defaults
(150,000 tokens, ~1,500 seed nodes) run at roughly 4–5 seconds per iteration and
grow as the graph does. For quick experiments, drop `total_tokens` and shrink
the hidden layers.
