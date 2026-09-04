"""
Do agents eat each other in a circle?

A castle need not be a family and need not even be a fixed set of agents. The
sharpest case: three nodes take each other's nodes in a cycle, so every member
is replaced every iteration while the arrangement stands. That is a glider, not
an organism — the pattern persists and the matter turns over — and no
lineage-based or membership-based definition would ever see it.

Conquest gives each node exactly one winner, so the conquest map is a
functional graph and its cycles are cheap to find. This looks for them, and
asks whether the same set of nodes cycles again on the next iteration.
"""
import random, sys
from collections import Counter
import numpy as np
sys.path.insert(0, ".")
from gol_config import SimConfig
import GraphOfLifeSimple as G


def cycles_of(winner_of):
    """Every cycle in a map from node to whoever took it."""
    colour, found = {}, []
    for start in winner_of:
        if start in colour:
            continue
        path, at = [], start
        while at in winner_of and at not in colour:
            colour[at] = "walking"
            path.append(at)
            at = winner_of[at]
        if at in colour and colour[at] == "walking":
            found.append(tuple(path[path.index(at):]))
        for node in path:
            colour[node] = "done"
    return found


def run(seed, iterations=40, **over):
    random.seed(seed); np.random.seed(seed)
    cfg = SimConfig(total_tokens=2500, n_nodes=50, k_neighbors=6,
                    hidden_layers=[12, 10], seed=seed, **over)
    world = G.new_world(cfg)

    lengths, null = Counter(), Counter()
    seen_before, repeats, total, iters = set(), 0, 0, 0
    for _ in range(iterations):
        # Read who took what from the recorded decisions rather than from the
        # order the resolver happens to be called in — the engine walks
        # `self.G.nodes()`, which is insertion order, not sorted, and only
        # calls resolve where somebody staked. A first version of this zipped
        # the two together and attributed every winner to the wrong node.
        frames = world.step(record_decisions=True)
        iters += 1
        winners = {}
        for frame in frames:
            for entry in ((frame.get("decisions") or {}).get("winners") or []):
                node, winner = int(entry["node"]), int(entry["winner"])
                if node != winner:
                    winners[node] = winner

        here = set()
        for cycle in cycles_of(winners):
            if len(cycle) >= 2:
                lengths[len(cycle)] += 1
                total += 1
                here.add(frozenset(cycle))
        repeats += len(here & seen_before)
        seen_before = here

        # The control. A count of cycles means nothing on its own: a map where
        # every node points at one of its neighbours has cycles whatever the
        # agents were thinking. So the same graph is rewired at random —
        # each conquered node taken by a uniformly chosen neighbour instead of
        # by whoever actually won it — and counted the same way.
        chance = {}
        for node in winners:
            near = [v for v in world.G.neighbors(node)] if world.G.has_node(node) else []
            if near:
                chance[node] = int(random.choice(near))
        for cycle in cycles_of(chance):
            if len(cycle) >= 2:
                null[len(cycle)] += 1

        if world.is_extinct():
            break
    return lengths, null, total, repeats, world.G.number_of_nodes(), iters


print(f"{'seed':>5} {'len':>4} {'observed':>9} {'if winners were random':>23} {'ratio':>7}")
for seed in (3, 7, 11):
    lengths, null, total, repeats, n, iters = run(seed)
    for length in (2, 3, 4):
        got, expected = lengths.get(length, 0), null.get(length, 0)
        ratio = f"{got / expected:.2f}x" if expected else ("--" if not got else "inf")
        print(f"{seed:>5} {length:>4} {got:>9,} {expected:>23,} {ratio:>7}")
    print(f"      {n:,} agents after {iters} iterations, "
          f"{repeats:,} cycles seen again next iteration with the same members\n")
