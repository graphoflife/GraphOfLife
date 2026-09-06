# research/

Working material towards a paper on this system. Nothing here is part of the
site or the engine; nothing in `web/` or the test suite depends on it.

- `Research.md` — the current state of knowledge. One claim under test —
  *this algorithm shows open-ended evolution* — and everything measured,
  retracted, or still open, arranged around it.
- `phylogeny.py` — reconstructs the lineage forest of a run and measures what
  it does over time. `--selftest` checks it against a forest small enough to
  work out by hand; `--simulate` runs a world and traces every link as the
  engine makes it. Recorded frames alone are **not** sufficient — see
  `Research.md` §5.2.
- `pilot_dynamics.py` — a small pilot behind two of the numbers in
  `Research.md` §8. Runs in about a minute.
- `pilot_cycles.py` — looks for agents taking each other's nodes in a cycle,
  against a null where each conquest goes to a random neighbour. The null is
  the point: cycles are abundant and mostly at or below chance.
- `pilot_sweep_scale.py` — how long a world takes before one founding lineage
  swallows it, against the size of the world.
- `pilot_brain_inputs.py` — the two numbers in the brains that nobody chose
  (`BACKLOG.md` §4b): how loud each kind of input is, and how finely a binary
  brain can say what it wants. Measures only — neither preset moves without a
  survival comparison, which is the `sweep` part and takes half an hour.
  Two methodological notes worth keeping: measurements have to be taken on a
  **warmed** world, since at iteration 0 every agent holds an equal share and
  every magnitude is nearly constant, which made noise look like half of what a
  brain hears; and architectures have to be compared on the **same**
  observations, since giving each its own world compares populations rather
  than brains.

Everything is meant to be run locally. Disk is cheap; RAM is the binding
constraint, because memory is population multiplied by policy size.

The tools are on the site too, under **Research**: the lineage forest, and the
flow modules with their compression and turnover. `tests/test_flowmodules.js`
checks the map equation against graphs whose answer is known by hand — run it
with `node tests/test_flowmodules.js`.

The same tab has a **Literature** page — the work this project is built on and
measured against, one entry per idea, each with a note on what it means in this
system. It covers the same ground as `Research.md` §9 but written for a reader
rather than for a paper, so the two need keeping in step: a retraction here is
a retraction there.
