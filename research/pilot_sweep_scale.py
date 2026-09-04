"""
Does a bigger world take longer to be taken over by one lineage?

If the time to lose all but one founding family grows with the size of the
world, then "so big that it takes ages" is a real answer to whether the
dynamics settle, and not a dodge. If it does not grow, size buys nothing.
"""
import random, sys, time
import numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, "research")
from gol_config import SimConfig
import GraphOfLifeSimple as G
from phylogeny import Forest

NO_PARENT = -1


def sweep_time(tokens, seed, cap=60):
    """Iterations until every survivor shares one founder, or None."""
    random.seed(seed); np.random.seed(seed)
    cfg = SimConfig(total_tokens=tokens, hidden_layers=[10, 8], seed=seed,
                    export_decisions=False)
    world = G.new_world(cfg)
    forest = Forest()
    links = []

    made, copied, mutated = world._new_brain, world._copy_brain, world._mutate_brain
    world._copy_brain = lambda s: copied(s)          # a copy keeps its id now
    def mutate(brain):
        before = brain.brain_id
        mutated(brain)
        if brain.brain_id != before:
            links.append((brain.brain_id, before))
    world._mutate_brain = mutate
    for b in world.brains.values():
        links.append((b.brain_id, b.parent_brain_id))

    founders = cfg.resolved_n()
    for it in range(cap):
        world.step(record_decisions=False)
        for child, parent in links:
            forest.link(it, child, parent)
        links.clear()
        if world.is_extinct():
            return None, world.G.number_of_nodes(), founders, it
        alive = [b.brain_id for b in world.brains.values()]
        memo = {}
        roots = {forest.ancestor_at(b, 0, memo) for b in alive}
        if len(roots) == 1:
            return it, world.G.number_of_nodes(), founders, it
    return None, world.G.number_of_nodes(), founders, cap


print(f"{'tokens':>8} {'founders':>9} {'seed':>5} {'agents':>8} {'one family after':>18} {'s':>6}")
for tokens in (600, 1500, 4000, 10000, 25000):
    for seed in (3, 11, 23):
        t0 = time.perf_counter()
        swept, agents, founders, ran = sweep_time(tokens, seed)
        dt = time.perf_counter() - t0
        note = f"{swept} iterations" if swept is not None else (
            "died out" if agents <= 20 else f"not by {ran}")
        print(f"{tokens:>8,} {founders:>9} {seed:>5} {agents:>8,} {note:>18} {dt:>6.1f}")
