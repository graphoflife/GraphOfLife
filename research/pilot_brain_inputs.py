"""
Pilot: two numbers in the brains that nobody chose.

BACKLOG.md section 4b records both, and says of each that it would change how a
population behaves and so wants an opinion before a patch. This measures; it
changes nothing. Research.md's standing rule is that nothing under about thirty
seeds is an effect, so the survival comparisons here run thirty.

  1. How loud is each kind of input?  The inputs are not normalised, so how
     much an input is heard depends on the range it happens to live in. Noise
     is drawn uniform(-2, 2) against magnitudes that are logs and quantiles,
     which was never a decision that gambling should be a third of what an
     agent attends to.

  2. Does the noise share matter?  The lever available without touching the
     engine is how many noise inputs there are. If a third of the first layer's
     variance being noise is bad for a population, fewer of them should show up
     in survival.

  3. How coarse is a binary brain's output?  Its BLOTTO scores are counts, and
     the range is set by the last hidden layer: 64 units of -1/0/+1 with two
     thirds zero sums to roughly +/-8. Ties are the visible consequence, and
     ties are answered by a coin.

    python3 research/pilot_brain_inputs.py            # everything, ~15 min
    python3 research/pilot_brain_inputs.py loudness   # part 1 only, seconds
"""
import random
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, ".")
from gol_config import SimConfig
import GraphOfLifeSimple as G

SEEDS = range(1, 31)          # the standing rule: thirty, not six
ITERS = 60


def world(seed, **over):
    random.seed(seed)
    np.random.seed(seed)
    settings = dict(total_tokens=2500, n_nodes=50, k_neighbors=6,
                    hidden_layers=[12, 10], seed=seed)
    settings.update(over)                 # so a variant can replace a default
    cfg = SimConfig(**settings)
    return cfg, G.new_world(cfg)


def groups(cfg):
    """Where each kind of input sits in the vector _input_vec builds."""
    at = 0
    out = {}
    out["flag"] = (at, at + cfg.FLAG_INPUTS); at += cfg.FLAG_INPUTS
    out["magnitude"] = (at, at + cfg.MAGNITUDE_INPUTS); at += cfg.MAGNITUDE_INPUTS
    out["message"] = (at, at + 4 * cfg.message_amount); at += 4 * cfg.message_amount
    out["noise"] = (at, at + cfg.random_input_amount); at += cfg.random_input_amount
    assert at == cfg.n_inputs(), f"{at} != {cfg.n_inputs()}"
    return out


def observations(w, cfg, phases=2, warm=15):
    """
    Every input vector the population actually built, for a few phases.

    Warmed up first, and that is not a detail. At iteration 0 every founder
    holds an equal share of the tokens and the seed graph is near-regular, so
    every magnitude is very nearly constant and noise is the only thing varying
    at all. Measured there, noise looks like half of what a brain hears; the
    question is what it is worth once a world has spread out, so the world is
    given time to spread out.
    """
    for _ in range(warm):
        w.step(record_decisions=False)
        if w.is_extinct():
            return np.array([])

    seen = []
    for _ in range(phases):
        log_deg, adj, q_tok, q_deg, log_tok = w._precompute_features()
        for u in list(w.G.nodes()):
            targets = [u] + [int(v) for v in w.G.neighbors(u)]
            for v in targets:
                seen.append(w._input_vec(u, v, log_deg, q_tok, q_deg, log_tok))
        w.step(record_decisions=False)
        if w.is_extinct():
            break
    return np.array(seen)


# ---------------------------------------------------------------------------

def loudness():
    """
    How much of the first hidden layer's variance each kind of input supplies.

    Contribution rather than input spread on its own, because what matters is
    what reaches the next layer: an input's variance multiplied by the weights
    reading it. Weights are drawn from one distribution for every input, so the
    split is a statement about the inputs.
    """
    print("1. What a float brain is listening to")
    print("   variance contributed to the first hidden layer, by kind of input\n")
    cfg, w = world(7)
    X = observations(w, cfg)
    if not len(X):
        print("   the world died before anything could be measured")
        return

    W = w.brains[next(iter(w.brains))].weights[0]      # (hidden, inputs)
    var = X.var(axis=0)
    # Every unit reads every input, so the mean squared weight per column is
    # the fair way to turn a per-input variance into a contribution.
    contribution = var * (W ** 2).mean(axis=0)
    total = contribution.sum()

    print(f"   {'kind':<12}{'inputs':>8}{'share':>9}{'per input':>12}{'loudest':>10}")
    for name, (lo, hi) in groups(cfg).items():
        n = hi - lo
        if not n:
            continue
        share = contribution[lo:hi].sum() / total
        print(f"   {name:<12}{n:>8}{share:>8.1%}{share / n:>11.2%}"
              f"{contribution[lo:hi].max() / total:>10.2%}")

    g = groups(cfg)
    per = {k: (contribution[lo:hi].sum() / total) / max(1, hi - lo)
           for k, (lo, hi) in g.items() if hi > lo}
    if "noise" in per and "magnitude" in per:
        print(f"\n   one noise input is {per['noise'] / per['magnitude']:.1f}x as loud as "
              f"one magnitude, and {per['noise'] / per['message']:.1f}x a message.")
    print("\n   Saturation, as a health check on the network itself:")
    H = np.tanh(W @ X.T)
    print(f"   {(np.abs(H) > 0.99).mean():.1%} of first-layer units are pinned.")


# ---------------------------------------------------------------------------

def survives(seed, **over):
    """Run one world and report how it ended."""
    try:
        cfg, w = world(seed, **over)
    except ValueError:
        return None
    for _ in range(ITERS):
        w.step(record_decisions=False)
        if w.is_extinct():
            return {"extinct": True, "at": w.iteration, "n": w.G.number_of_nodes()}
    return {"extinct": False, "at": w.iteration, "n": w.G.number_of_nodes()}


def sweep(label, variants):
    print(f"\n{label}")
    print(f"   {'setting':<22}{'extinct':>9}{'median n':>10}{'mean n':>9}")
    for name, over in variants:
        rows = [survives(s, **over) for s in SEEDS]
        rows = [r for r in rows if r]
        if not rows:
            print(f"   {name:<22}{'invalid':>9}")
            continue
        died = sum(r["extinct"] for r in rows)
        alive = [r["n"] for r in rows if not r["extinct"]] or [0]
        print(f"   {name:<22}{died:>4}/{len(rows):<4}"
              f"{int(np.median(alive)):>10,}{int(np.mean(alive)):>9,}")
    print("   Nothing here is an effect unless the columns separate by more than "
          "they\n   wander between neighbouring settings.")


# ---------------------------------------------------------------------------

def blotto_spread():
    """
    How many distinct staking scores each kind of brain can express, and how
    often two heads land exactly equal — which _choose_binary answers with a
    coin and _share_of_first with an even split.

    One world, warmed, and every architecture reads the *same* observations.
    Giving each variant its own world instead compared architectures across
    different populations — and two of them had died by the time the
    measurement was taken, which is a fact about survival and not about how
    finely a brain can speak.
    """
    print("\n3. How finely a brain can say what it wants")
    print("   one warmed world, the same observations through every architecture\n")

    cfg, w = world(7)
    X = observations(w, cfg, phases=1)
    if not len(X):
        print("   the world died before anything could be measured")
        return
    X = X.T                                  # brains read (inputs, candidates)

    print(f"   {'brain':<22}{'distinct':>10}{'ties':>8}{'both<=0':>10}{'range':>16}")
    variants = [
        ("float", dict(brain_kind="float")),
        ("float16", dict(brain_kind="float16")),
        ("binary, last 10", dict(brain_kind="binary", hidden_layers=[12, 10])),
        ("binary, last 32", dict(brain_kind="binary", hidden_layers=[12, 32])),
        ("binary, last 64", dict(brain_kind="binary", hidden_layers=[12, 64])),
        ("binary, last 128", dict(brain_kind="binary", hidden_layers=[12, 128])),
    ]
    for name, over in variants:
        settings = dict(total_tokens=2500, n_nodes=50, k_neighbors=6,
                        hidden_layers=[12, 10], seed=7)
        settings.update(over)
        vcfg = SimConfig(**settings)
        np.random.seed(7)                    # the same weights every time
        brain = G.make_brain(vcfg, 1)
        heads = G.build_heads(vcfg)

        Y = brain.forward(X)
        scores = np.asarray(Y[heads["BLOTTO"], :], dtype=float).ravel()
        mode = np.asarray(Y[heads["BLOTTO_MODE"], :], dtype=float)
        ties = float((mode[0] == mode[1]).mean())
        both = float(((mode[0] <= 0) & (mode[1] <= 0)).mean())
        print(f"   {name:<22}{len(set(scores.tolist())):>10,}{ties:>7.1%}{both:>10.1%}"
              f"{f'{scores.min():.1f} to {scores.max():.1f}':>16}")

    print("\n   A tie is decided by a coin and both-non-positive by an even split,")
    print("   so those two columns are how often the brain is not the one deciding.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"

    if what in ("all", "loudness"):
        loudness()
    if what in ("all", "spread"):
        blotto_spread()
    if what in ("all", "sweep"):
        sweep("2. Does the noise share change how a population fares?",
              [(f"{n} noise inputs", dict(random_input_amount=n))
               for n in (0, 2, 5, 10)])
        sweep("4. Does a wider last hidden layer help a binary brain?",
              [(f"binary, last {n}", dict(brain_kind="binary", hidden_layers=[12, n]))
               for n in (10, 32, 64, 128)])
