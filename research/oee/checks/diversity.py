"""
How different are the agents? Every brain of a world put into the same situations.

For one world, after one game so that its messages are fresh:

1. Situations: what SITUATIONS living agents read — their tokens and degree,
   their neighbours', the messages they hold — with the random input rows drawn
   DRAWS times from a stream of our own: the same draws for every brain, so two
   brains are compared on exactly the same inputs.
2. Brains: those of AGENTS agents drawn at random (so a brain carried by many
   agents is drawn as often as it is carried), and as many fresh founder brains,
   which have never met selection, as the reference of what brains do by chance.
3. Each brain decides in each situation and draw, by the engine's own rules:
     stake   the shares of its tokens it stakes on itself and on each neighbour
     home    the share it keeps on its own node (keep_home included)
     child   the share it would give a child (0 when no child would be made)
     spread  whether it spreads its stake (1) or goes all in (0)
     link    the mean probability of linking a newborn to a neighbour
     revolt  the share of its stakes on others marked revolutionary
4. What is reported:
   - for each decision, how its variance splits into where the agent is (the
     situation), who it is (the brain), who it is in that situation (the
     interaction) and chance (the draws): a two-way analysis of variance;
   - for the stake, how far apart two random agents' stakes are in the same
     situation and draw (total variation), against one brain's two draws (its
     own noise) and two founders;
   - in the decision's own units, how far apart two agents in the same
     situation and draw are (the standard deviation over agents), against how
     far one brain is from itself over the draws;
   - kinds of behaviour: brains are grouped while they differ by no more than
     95% of brains differ from themselves between two halves of the draws
     (their own noise sets the line), and the kinds are counted, by the agents
     carrying them, as an effective number (1 / sum of squared shares), beside
     the effective number of genotypes; and the share of pairs of agents that
     act measurably differently. Once over every decision, once over the stake
     alone;
   - niches: how much of the spread of the carriers' tokens and degrees the
     kind accounts for (eta squared), beyond what kinds drawn at random would.

    python3 research/oee/checks/diversity.py RUN[:ITERATION] ... --label B2       a lab run (no iteration: its final checkpoint)
    python3 research/oee/checks/diversity.py pilot:DIR:AT ... --label P            a pilot's checkpoint
    python3 research/oee/checks/diversity.py --persist A B --label B2-long         kinds of world A still alive in world B

measure(spec, game=False) reads a lab run's archive as kept, with no game played first.

Writes GraphOfLifeRuns/.pilots/diversity/<label>.json.
"""
from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

ROOT = "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "oee", "pilots"))
sys.path.insert(0, os.path.join(ROOT, "research", "oee", "checks"))

import numpy as np

OUT = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "diversity")
SITUATIONS = 200
AGENTS = 400
DRAWS = 8
#: Two brains count as acting differently when they differ by more than this
#: share of brains differ from themselves between two halves of the draws: the
#: usual five per cent of a test, so that a pair of brains that act alike is
#: split once in twenty.
SAME = 0.95
DECISIONS = ("home", "child", "spread", "link", "revolt")


# ---------------------------------------------------------------------------
# A world, and the situations its agents are in
# ---------------------------------------------------------------------------

def load(spec: str, game: bool = True):
    """
    A world from 'RUN', 'RUN:ITERATION' (a lab run's archive) or 'pilot:DIR:AT', one game on. Without
    the game (`game=False`) it is the world as kept: an archive holds the messages its agents would
    read next, so nothing is played, as the screen's judge reads its worlds (research/screen/judge.py).
    """
    if spec.startswith("pilot:"):
        import pilot_frames
        _, d, at = spec.split(":")
        world = pilot_frames.load(d, int(at))
    else:
        import gol_store as store
        from gol_config import SimConfig
        from GraphOfLifeSimple import GraphOfLife
        run, _, it = spec.partition(":")
        cfg = SimConfig.from_dict(store.load_meta(run)["config"])
        world = (GraphOfLife.from_checkpoint(store.load_archive(run, int(it)), cfg) if it
                 else store.load_checkpoint(run, cfg))
    if game:
        world.step(record_decisions=False)
    return world


def brain_key(brain) -> str:
    """A brain's identity by its weights: in B1 every brain changes after every game, whatever its id says."""
    h = hashlib.sha1()
    for a in list(brain.weights) + list(brain.biases):
        h.update(np.ascontiguousarray(a).tobytes())
    if hasattr(brain, "active"):
        h.update(str(getattr(brain, "active")).encode())
    return h.hexdigest()


def situations(w, rng: np.random.RandomState, n: int) -> List[Dict[str, Any]]:
    """What n living agents read, each with DRAWS draws of the random rows that every brain will be given."""
    features = w._precompute_features()
    log_deg, neighs, q_tok, q_deg, log_tok, at_risk = features
    living = sorted(u for u in w.G.nodes() if int(w.tokens.get(u, 0)) > 0)
    chosen = sorted(rng.choice(living, size=min(n, len(living)), replace=False).tolist())
    noise = w.cfg.input_layout()["noise"]
    rows = slice(noise[0], noise[1])
    out = []
    for u in chosen:
        cands = [u] + list(neighs[u])
        X = w._inputs(u, cands, log_deg, q_tok, q_deg, log_tok, at_risk)
        draws = []
        for _ in range(DRAWS):
            Z = X.copy()
            Z[rows] = w.kind.draw_noise((len(cands), w.cfg.random_input_amount), rng).T
            draws.append(Z)
        out.append({"u": u, "tokens": int(w.tokens[u]), "cands": cands, "draws": draws,
                    "degree": len(cands) - 1})
    return out


# ---------------------------------------------------------------------------
# What a brain does in them
# ---------------------------------------------------------------------------

def decide(w, Y: np.ndarray, sit: Dict[str, Any]) -> Tuple[Dict[str, float], np.ndarray]:
    """One brain's decisions from its outputs in one situation and draw, by the engine's rules."""
    from GraphOfLifeSimple import _apportion, _share_of_first
    from probe import p_yes
    H, cfg, t, cands = w.heads, w.cfg, sit["tokens"], sit["cands"]
    scores = np.asarray(Y[H["BLOTTO"], :], float)
    mode = np.mean(Y[H["BLOTTO_MODE"], :], axis=1)
    spread = bool(mode[0] > mode[1])
    kept = max(1, int(np.ceil(cfg.keep_home * t))) if cfg.keep_home > 0 else 0
    stake = t - kept
    if spread:
        alloc = _apportion(scores, stake)
    else:
        alloc = np.zeros(len(cands), dtype=int)
        alloc[int(np.argmax(scores))] = stake
    alloc[0] += kept
    frac = np.mean(Y[H["REPRO_FRACTION"], :], axis=1)
    wanted = int(np.floor(_share_of_first(frac[0], frac[1]) * t))
    child = w._endowment(sit["u"], t, max(0, min(t, wanted)))
    L, LM = Y[H["LINK"], :], Y[H["LINK_MODE"], :]
    link = (float(np.mean([p_yes(L[0, c], L[1, c], LM[0, c], LM[1, c]) for c in range(1, len(cands))]))
            if len(cands) > 1 else 0.0)
    revolt = 0.0
    if cfg.allow_revolutions:
        R = Y[H["REV_FRACTION"], :]
        given = sum(int(alloc[c]) for c in range(1, len(cands)))
        if given:
            revolt = sum(int(np.floor(_share_of_first(R[0, c], R[1, c]) * alloc[c]))
                         for c in range(1, len(cands))) / given
    return ({"home": alloc[0] / t, "child": child / t if child >= 1 else 0.0, "spread": float(spread),
             "link": link, "revolt": revolt}, alloc / t)


def play(w, brains: List[Any], sits: List[Dict[str, Any]]):
    """
    Every brain in every situation and draw: the scalar decisions as arrays of
    brains x situations x draws, and the stakes as one array of brains x
    candidates for each situation and draw.
    """
    columns, segments = [], []
    at = 0
    for s, sit in enumerate(sits):
        for k, X in enumerate(sit["draws"]):
            columns.append(X)
            segments.append((s, k, at, at + X.shape[1]))
            at += X.shape[1]
    X_all = np.concatenate(columns, axis=1)
    B, S = len(brains), len(sits)
    D = {d: np.zeros((B, S, DRAWS)) for d in DECISIONS}
    stakes = {(s, k): np.zeros((B, len(sits[s]["cands"]))) for s in range(S) for k in range(DRAWS)}
    for b, brain in enumerate(brains):
        Y_all = brain.forward(X_all)
        for s, k, a, z in segments:
            got, shares = decide(w, Y_all[:, a:z], sits[s])
            for d in DECISIONS:
                D[d][b, s, k] = got[d]
            stakes[(s, k)][b] = shares
    return D, stakes


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def split(D: np.ndarray) -> Dict[str, float]:
    """
    A two-way analysis of variance with the draws as replicates: the shares of
    the variance due to the situation, the brain, the brain in that situation,
    and chance.
    """
    grand = D.mean()
    mb, ms, mbs = D.mean(axis=(1, 2)), D.mean(axis=(0, 2)), D.mean(axis=2)
    B, S, K = D.shape
    parts = {
        "situation": B * K * float(((ms - grand) ** 2).sum()),
        "brain": S * K * float(((mb - grand) ** 2).sum()),
        "interaction": K * float(((mbs - mb[:, None] - ms[None, :] + grand) ** 2).sum()),
        "chance": float(((D - mbs[:, :, None]) ** 2).sum()),
    }
    total = sum(parts.values())
    out = {k: (v / total if total > 0 else 0.0) for k, v in parts.items()}
    out["mean"] = float(grand)
    out["sd"] = float(D.std())
    # In the decision's own units: two agents in the same situation and draw,
    # and one brain over its draws.
    out["sd_between_agents"] = float(np.sqrt(D.var(axis=0).mean()))
    out["sd_one_brain"] = float(np.sqrt(D.var(axis=2).mean()))
    return out


def pair_tv(A: np.ndarray) -> float:
    """The mean total-variation distance between two different rows of A (shares over the same candidates)."""
    B = A.shape[0]
    if B < 2:
        return 0.0
    x = np.sort(A, axis=0)
    weights = (2 * np.arange(B) - B + 1)[:, None]
    return float(0.5 * (weights * x).sum() / (B * (B - 1) / 2))


def stake_spread(stakes, B: int, S: int) -> Dict[str, float]:
    """Two agents' stakes in the same situation and draw, against one brain's two draws."""
    between = float(np.mean([pair_tv(stakes[(s, k)]) for s in range(S) for k in range(DRAWS)]))
    within = []
    for s in range(S):
        for k1 in range(DRAWS):
            for k2 in range(k1 + 1, DRAWS):
                within.append(float(np.mean(0.5 * np.abs(stakes[(s, k1)] - stakes[(s, k2)]).sum(axis=1))))
    return {"between": between, "within": float(np.mean(within))}


def profiles(D, stakes, S: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Each brain's mean decisions per situation over each half of the draws."""
    half = DRAWS // 2
    out = []
    for draws in (range(0, half), range(half, DRAWS)):
        idx = list(draws)
        out.append({"scalars": np.stack([D[d][:, :, idx].mean(axis=2) for d in DECISIONS], axis=2),
                    "stakes": [np.mean([stakes[(s, k)] for k in idx], axis=0) for s in range(S)]})
    return out[0], out[1]


def distance(p: Dict[str, Any], q: Dict[str, Any], S: int, stake_only: bool = False) -> np.ndarray:
    """
    How differently every brain of p acts from every brain of q, from 0 to 1:
    over the situations, the mean of the stake's total variation and of the
    five decisions' absolute differences — or the stake's alone.
    """
    B = p["scalars"].shape[0]
    total = np.zeros((B, q["scalars"].shape[0]))
    for s in range(S):
        a, b = p["stakes"][s], q["stakes"][s]
        total += 0.5 * np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)
        if not stake_only:
            total += np.abs(p["scalars"][:, None, s, :] - q["scalars"][None, :, s, :]).sum(axis=2)
    return total / (S * (1 if stake_only else 1 + len(DECISIONS)))


def cluster(dist: np.ndarray, floor: float) -> np.ndarray:
    """
    Average-linkage grouping, merged while the closest two groups are no further
    apart than the noise floor: a kind is brains no more different from one
    another than a brain is from itself.
    """
    n = dist.shape[0]
    label = np.arange(n)
    groups = {i: [i] for i in range(n)}
    d = dist.copy().astype(float)
    np.fill_diagonal(d, np.inf)
    alive = np.ones(n, dtype=bool)
    while alive.sum() > 1:
        sub = np.where(alive[:, None] & alive[None, :], d, np.inf)
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        if sub[i, j] > floor:
            break
        ni, nj = len(groups[i]), len(groups[j])
        d[i, :] = (d[i, :] * ni + d[j, :] * nj) / (ni + nj)
        d[:, i] = d[i, :]
        d[i, i] = np.inf
        alive[j] = False
        d[j, :] = np.inf
        d[:, j] = np.inf
        groups[i] += groups.pop(j)
    for g, members in enumerate(groups.values()):
        label[members] = g
    return label


def effective(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return float(1.0 / (p ** 2).sum())


def eta2(values: np.ndarray, label: np.ndarray) -> float:
    """The share of the variance of `values` that the groups of `label` account for."""
    total = float(((values - values.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    between = sum(float(((values[label == g].mean() - values.mean()) ** 2) * (label == g).sum())
                  for g in np.unique(label))
    return between / total


def kinds_of(D, stakes, S: int, carriers: np.ndarray, stake_only: bool = False):
    """Group brains into kinds of behaviour, at their own noise, and say how many there are."""
    p, q = profiles(D, stakes, S)
    cross = distance(p, q, S, stake_only)
    self_distance = np.diag(cross).copy()
    sym = (cross + cross.T) / 2.0
    floor = float(np.quantile(self_distance, SAME))
    label = cluster(sym, floor)
    counts = np.array([carriers[label == g].sum() for g in np.unique(label)], dtype=float)
    # Pairs of agents (not of brains): two carriers of one brain act alike.
    w = carriers / carriers.sum()
    apart = float((np.outer(w, w) * (sym > floor)).sum())
    upper = np.triu_indices(len(sym), 1)
    return label, {
        "floor": floor,
        "self_median": float(np.median(self_distance)),
        "between_median": float(np.median(sym[upper])) if len(sym) > 1 else 0.0,
        "pairs_apart": apart,
        "kinds": int(len(counts)),
        "effective": effective(counts),
        "largest": float(counts.max() / counts.sum()),
        "shares": sorted((counts / counts.sum()).round(4).tolist(), reverse=True)[:12],
    }


def niche(values: np.ndarray, label: np.ndarray, rng: np.random.RandomState, shuffles: int = 200) -> Dict[str, float]:
    """eta squared of the carriers' values by kind, and its mean over kinds shuffled among the carriers."""
    observed = eta2(values, label)
    null = [eta2(values, rng.permutation(label)) for _ in range(shuffles)]
    return {"eta2": observed, "shuffled": float(np.mean(null)),
            "above_shuffled": float(np.mean([observed > x for x in null]))}


def measure(spec: str, seed: int = 7, game: bool = True) -> Dict[str, Any]:
    from GraphOfLifeSimple import make_brain
    w = load(spec, game)
    rng = np.random.RandomState(seed)
    sits = situations(w, rng, SITUATIONS)
    S = len(sits)
    agents = sorted(int(u) for u in w.G.nodes())
    keys = {u: brain_key(w.brains[u]) for u in agents}
    carried = {}
    for u in agents:
        carried[keys[u]] = carried.get(keys[u], 0) + 1
    drawn = rng.choice(agents, size=min(AGENTS, len(agents)), replace=False)
    distinct, carriers, first = [], [], {}
    for u in sorted(int(x) for x in drawn):
        k = keys[u]
        if k not in first:
            first[k] = len(distinct)
            distinct.append(w.brains[u])
            carriers.append(0)
        carriers[first[k]] += 1
    carriers = np.array(carriers, dtype=float)
    # The decisions of the drawn agents' brains, each brain once and counted by its carriers.
    D, stakes = play(w, distinct, sits)
    weight = carriers.astype(int)
    D_agents = {d: np.repeat(D[d], weight, axis=0) for d in DECISIONS}
    st_agents = {key: np.repeat(v, weight, axis=0) for key, v in stakes.items()}
    founders = [make_brain(w.cfg, -1 - i, np.random.RandomState(1000 + i)) for i in range(len(distinct))]
    F, f_stakes = play(w, founders, sits)

    label, kinds = kinds_of(D, stakes, S, carriers)
    _, stake_kinds = kinds_of(D, stakes, S, carriers, stake_only=True)
    _, founder_kinds = kinds_of(F, f_stakes, S, np.ones(len(founders)))
    _, founder_stake_kinds = kinds_of(F, f_stakes, S, np.ones(len(founders)), stake_only=True)
    # Niches: do the kinds sit in different places? Over every agent carrying a drawn brain.
    carriers_of = {}
    for u in agents:
        if keys[u] in first:
            carriers_of.setdefault(first[keys[u]], []).append(u)
    lab, tok, deg = [], [], []
    for b, us in carriers_of.items():
        for u in us:
            lab.append(label[b])
            tok.append(np.log1p(w.tokens[u]))
            deg.append(np.log1p(w.G.degree(u)))
    lab = np.array(lab)
    return {
        "world": spec, "agents": len(agents), "situations": S, "drawn": int(len(drawn)),
        "distinct_drawn": len(distinct),
        "genotypes": len(carried), "genotypes_effective": effective(np.array(list(carried.values()), float)),
        "decisions": {d: split(D_agents[d]) for d in DECISIONS},
        "founders": {d: split(F[d]) for d in DECISIONS},
        "stake": stake_spread(st_agents, int(weight.sum()), S),
        "stake_founders": stake_spread(f_stakes, len(founders), S),
        "kinds": kinds,
        "stake_kinds": stake_kinds,
        "founder_kinds": founder_kinds,
        "founder_stake_kinds": founder_stake_kinds,
        "niche_tokens": niche(np.array(tok), lab, rng) if len(lab) > 2 else None,
        "niche_degree": niche(np.array(deg), lab, rng) if len(lab) > 2 else None,
    }


def persist(spec_a: str, spec_b: str, seed: int = 7) -> Dict[str, Any]:
    """
    Kinds over time: the brains of world A and of a later world B in A's
    situations, grouped together; how many of A's agents belong to a kind that
    still has carriers in B, and how many of B's to one that was there in A.
    """
    wa, wb = load(spec_a), load(spec_b)
    rng = np.random.RandomState(seed)
    sits = situations(wa, rng, SITUATIONS)
    S = len(sits)
    brains, carriers, era = [], [], []
    for w, tag in ((wa, 0), (wb, 1)):
        agents = sorted(int(u) for u in w.G.nodes())
        drawn = rng.choice(agents, size=min(AGENTS // 2, len(agents)), replace=False)
        first = {}
        for u in sorted(int(x) for x in drawn):
            k = brain_key(w.brains[u])
            if k not in first:
                first[k] = len(brains)
                brains.append(w.brains[u])
                carriers.append(0)
                era.append(tag)
            carriers[first[k]] += 1
    carriers, era = np.array(carriers, float), np.array(era)
    D, stakes = play(wa, brains, sits)
    label, kinds = kinds_of(D, stakes, S, carriers)
    in_a, in_b = set(label[era == 0]), set(label[era == 1])
    share = lambda mask, keep: float(carriers[mask & np.isin(label, list(keep))].sum() / carriers[mask].sum())
    return {"from": spec_a, "to": spec_b, "kinds": kinds,
            "a_kinds": len(in_a), "b_kinds": len(in_b), "shared_kinds": len(in_a & in_b),
            "a_agents_whose_kind_lives_on": share(era == 0, in_b),
            "b_agents_whose_kind_was_there": share(era == 1, in_a)}


def measured(spec: str):
    """measure(spec), or None with the reason printed: one world that cannot be read stops no other."""
    try:
        return measure(spec)
    except Exception as e:  # noqa: BLE001 - a long batch reports and goes on
        print(f"{spec}: not measured ({type(e).__name__}: {e})", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("worlds", nargs="*")
    ap.add_argument("--label", required=True)
    ap.add_argument("--persist", nargs=2, action="append", default=[])
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{a.label}.json")
    done = json.load(open(path)) if os.path.exists(path) else {"worlds": [], "persist": []}
    measured_before = {w["world"] for w in done["worlds"]}
    todo = [w for w in a.worlds if w not in measured_before]
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for result in pool.map(measured, todo):
            if result is None:
                continue
            done["worlds"].append(result)
            json.dump(done, open(path, "w"), indent=1)
            k = result["kinds"]
            print(f"{result['world']}: brain share of home {result['decisions']['home']['brain']:.2f}, "
                  f"stake TV {result['stake']['between']:.3f} (noise {result['stake']['within']:.3f}, "
                  f"founders {result['stake_founders']['between']:.3f}); kinds {k['kinds']} "
                  f"(effective {k['effective']:.2f}) of {result['genotypes_effective']:.0f} effective genotypes",
                  flush=True)
        for result in pool.map(persist, *zip(*a.persist)) if a.persist else []:
            done["persist"].append(result)
            json.dump(done, open(path, "w"), indent=1)
            print(f"{result['from']} -> {result['to']}: {result['a_kinds']} kinds then, {result['b_kinds']} later, "
                  f"{result['shared_kinds']} shared; {result['a_agents_whose_kind_lives_on']:.2f} of the agents' "
                  f"kinds live on", flush=True)


if __name__ == "__main__":
    main()
