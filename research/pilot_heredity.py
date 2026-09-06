"""
Pilot: is there heredity at all?

Cumulative evolution needs three things — variation, differential survival, and
**heredity**: offspring must resemble their parents more than they resemble the
population at large, and keep resembling them long enough for selection to act.
The first two are obviously present here. The third has never been measured,
and there is a structural reason to doubt it.

`blotto_phase` ends with

    for brain in self.brains.values():
        self._mutate_brain(brain)

which mutates *every living agent* every iteration, at `mutation_probability`
0.5 by default. Mutation is applied to the population, not at replication. A
genome that is winning is jittered exactly as hard as one that is losing, and
it is jittered again the next iteration, and the next.

So this asks three questions with numbers:

  1. How long does a genotype last before it is mutated away?
  2. How fast does a lineage forget itself — in weights, and in behaviour?
  3. Is that faster than selection can act?

    python3 research/pilot_heredity.py
"""
import random
import sys

import numpy as np

sys.path.insert(0, ".")
from gol_config import SimConfig
import GraphOfLifeSimple as G

# Five is enough to see the shape of parts 1 and 2, where the effect is
# enormous. Part 3 compares two populations and is held to Research.md's
# standing rule of about thirty; pass a count to raise it.
SEEDS = tuple(range(1, int(sys.argv[2]) + 1)) if len(sys.argv) > 2 else (1, 2, 3, 4, 5)
ITERS = 40


def world(seed, **over):
    random.seed(seed)
    np.random.seed(seed)
    settings = dict(total_tokens=2500, n_nodes=50, k_neighbors=6,
                    hidden_layers=[12, 10], seed=seed)
    settings.update(over)
    cfg = SimConfig(**settings)
    return cfg, G.new_world(cfg)


def flat(brain):
    """A genome as one vector, so two can be compared."""
    return np.concatenate([w.ravel().astype(float) for w in brain.weights]
                          + [b.ravel().astype(float) for b in brain.biases])


def behaviour(brain, X):
    """What a brain does, rather than what it is: outputs over fixed inputs."""
    return np.asarray(brain.forward(X), dtype=float).ravel()


def probe_inputs(w, cfg, columns=24):
    """One fixed batch of observations, so every brain is asked the same thing."""
    log_deg, adj, q_tok, q_deg, log_tok = w._precompute_features()
    seen = []
    for u in list(w.G.nodes()):
        for v in [u] + [int(x) for x in w.G.neighbors(u)]:
            seen.append(w._input_vec(u, v, log_deg, q_tok, q_deg, log_tok))
            if len(seen) >= columns:
                return np.array(seen).T
    return np.array(seen).T if seen else None


# ---------------------------------------------------------------------------

def genotype_lifetime():
    """
    How many iterations a brain_id survives before mutation renames it.

    A brain_id names a genotype: a copy keeps its source's id and only a
    mutation makes a new one. So the lifetime of an id is the lifetime of an
    exact genome.
    """
    print("1. How long a genotype lasts before it is mutated away\n")
    print(f"   {'mutation_probability':>21}{'median life':>13}{'90th pct':>10}"
          f"{'ever >5 iters':>15}")
    for p in (0.5, 0.2, 0.05, 0.0):
        lives = []
        for seed in SEEDS:
            cfg, w = world(seed, mutation_probability=p)
            first_seen, last_seen = {}, {}
            for t in range(ITERS):
                w.step(record_decisions=False)
                if w.is_extinct():
                    break
                for brain in w.brains.values():
                    bid = int(brain.brain_id)
                    first_seen.setdefault(bid, t)
                    last_seen[bid] = t
            lives.extend(last_seen[b] - first_seen[b] + 1 for b in last_seen)
        lives = np.array(lives) if lives else np.array([0])
        print(f"   {p:>21.2f}{np.median(lives):>13.1f}"
              f"{np.percentile(lives, 90):>10.1f}{(lives > 5).mean():>14.1%}")
    print("\n   At the default a genotype is gone before anything could have")
    print("   selected for it. Selection needs something to hold still.")


# ---------------------------------------------------------------------------

def lineage_memory():
    """
    How fast a lineage forgets itself, against how different two strangers are.

    Take one agent, remember its genome and its behaviour, and follow its
    descendants. Compare each to what the ancestor was. Background is the same
    comparison between unrelated agents at the same moment: once a descendant
    is as far from its ancestor as a stranger is, the lineage has retained
    nothing and heredity is over.
    """
    print("\n2. How fast a lineage forgets itself\n")
    print("   distance from an agent's own ancestor, as a share of the distance")
    print("   between two unrelated agents. 1.0 means nothing is left.\n")
    print(f"   {'iterations later':>17}{'weights':>10}{'behaviour':>12}")

    horizons = [1, 2, 3, 5, 10]
    rows = {h: {"w": [], "b": []} for h in horizons}

    for seed in SEEDS:
        cfg, w = world(seed)
        for _ in range(10):                       # settle before measuring
            w.step(record_decisions=False)
        if w.is_extinct():
            continue
        X = probe_inputs(w, cfg)
        if X is None:
            continue

        # The ancestors, and the node lineage that carries them forward.
        ancestors = {u: (flat(w.brains[u]), behaviour(w.brains[u], X))
                     for u in list(w.G.nodes())}
        descend = {u: u for u in ancestors}       # ancestor -> living descendant

        for t in range(1, max(horizons) + 1):
            w.step(record_decisions=False)
            if w.is_extinct():
                break
            alive = set(w.G.nodes())
            # Follow each ancestor to a living descendant, or lose it.
            for a in list(descend):
                cur = descend[a]
                if cur not in alive:
                    kids = [c for c, p in w.parent_of.items() if p == cur and c in alive]
                    descend[a] = kids[0] if kids else None
                    if descend[a] is None:
                        del descend[a]
            if t not in rows or not descend:
                continue

            living = [u for u in descend.values() if u in alive]
            if len(living) < 3:
                continue
            strangers_w, strangers_b = [], []
            for _ in range(60):
                x, y = random.sample(living, 2)
                strangers_w.append(np.linalg.norm(flat(w.brains[x]) - flat(w.brains[y])))
                strangers_b.append(np.linalg.norm(behaviour(w.brains[x], X)
                                                  - behaviour(w.brains[y], X)))
            bg_w = np.mean(strangers_w) or 1.0
            bg_b = np.mean(strangers_b) or 1.0

            for a, cur in descend.items():
                if cur not in alive:
                    continue
                aw, ab = ancestors[a]
                rows[t]["w"].append(np.linalg.norm(flat(w.brains[cur]) - aw) / bg_w)
                rows[t]["b"].append(np.linalg.norm(behaviour(w.brains[cur], X) - ab) / bg_b)

    for h in horizons:
        wv, bv = rows[h]["w"], rows[h]["b"]
        if not wv:
            print(f"   {h:>17}{'—':>10}{'—':>12}")
            continue
        print(f"   {h:>17}{np.mean(wv):>10.2f}{np.mean(bv):>12.2f}")
    print("\n   Read down the columns: if these reach 1.0 within a few")
    print("   iterations, a lineage retains nothing selection could act on.")


# ---------------------------------------------------------------------------

def selection_ablation():
    """
    Does selection change the outcome at all?

    The comparison the Literature page's own method section calls for and that
    nothing here has ever run: take the mechanism away and see whether what is
    being claimed goes away with it. If a world where nodes are won at random
    looks like a world where they are won by the biggest stake, then the stake
    is not doing the work.
    """
    print("\n3. Does the conquest rule matter?\n")
    print("   Same worlds, with the winner of each node decided by stake as")
    print("   usual, and then decided by a coin among everyone who staked.\n")
    print(f"   {'winner chosen by':>20}{'extinct':>9}{'median n':>10}"
          f"{'distinct brains':>17}{'mean degree':>13}")

    # The descriptor, not the function it wraps. Reading the attribute off the
    # class unwraps the staticmethod, and putting a bare function back makes it
    # an instance method — so the engine then calls it with `self` as the first
    # argument and it dies on the arity.
    real_resolve = G.GraphOfLife.__dict__["_resolve_winner"]

    def by_chance(offers, revolutionaries):
        """Same signature and same return shape; only the choice is different."""
        who = random.choice(list(offers))
        return int(who), int(offers[who]), False

    for label, fn in (("stake (as shipped)", real_resolve),
                      ("chance", staticmethod(by_chance))):
        G.GraphOfLife._resolve_winner = fn
        died, sizes, brains, degs = 0, [], [], []
        for seed in SEEDS:
            cfg, w = world(seed)
            for _ in range(ITERS):
                w.step(record_decisions=False)
                if w.is_extinct():
                    died += 1
                    break
            else:
                n, e = w.G.number_of_nodes(), w.G.number_of_edges()
                sizes.append(n)
                brains.append(len({b.brain_id for b in w.brains.values()}))
                degs.append(2 * e / max(1, n))
        print(f"   {label:>20}{died:>4}/{len(SEEDS):<4}"
              f"{int(np.median(sizes)) if sizes else 0:>10,}"
              f"{int(np.median(brains)) if brains else 0:>17,}"
              f"{np.mean(degs) if degs else 0:>13.2f}")
    G.GraphOfLife._resolve_winner = real_resolve
    print("\n   Columns that do not separate mean the rule is not what shapes")
    # The count comes from SEEDS rather than the sentence, because the sentence
    # went on saying "five" after the run was widened to thirty.
    enough = "" if len(SEEDS) >= 30 else " — indicative, not an effect"
    print(f"   the run. {len(SEEDS)} seeds{enough}.")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "lifetime"):
        genotype_lifetime()
    if what in ("all", "memory"):
        lineage_memory()
    if what in ("all", "ablation"):
        selection_ablation()
