# B1. How big is one mutation, in behaviour?

**Answer: too coarse.** One mutation of the engine makes a brain decide differently from its parent in 34% to 67% of 300 real situations. In s4003 at 15,000 that is 0.98 of the distance between two unrelated brains of the world, and 47% of mutants are at least that far. One number changed: under 1.1%, and 43% to 72% of such mutants decide exactly as their parent.

The table and the reading are in `../RESULTS.md`; the numbers are in `b1.json`; `python3 research/efficient/B1/b1.py` makes them again (about 9 minutes).

**Caveats.** A decision here is the child's tokens and the first token's target with an empty board, not a whole game. Worlds: Experiment 48's three, at 10,000 tokens. Parents are drawn among agents, so clones count as often as they occur.
