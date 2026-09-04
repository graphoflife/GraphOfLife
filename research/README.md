# research/

Working material towards a paper on this system. Nothing here is part of the
site or the engine; nothing in `web/` or the test suite depends on it.

- `Research.md` — the design document: definitions, hypotheses, methods,
  experiment protocols, and what is measurable today.
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

Everything is meant to be run locally. Disk is cheap; RAM is the binding
constraint, because memory is population multiplied by policy size.
