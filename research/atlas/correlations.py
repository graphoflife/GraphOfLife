"""
What goes with what: a search for correlations among the statistics of agents.

For every agent at the start of a game, from a lab run's recorded frames, a row
of quantities: its state, its neighbourhood and its place in the network, what
it staked in that game, and what then happened to it. Every pair of them is
correlated in each world, on linear and on logarithmic axes, and by rank.

    python3 research/atlas/correlations.py extract RUN [...] --from 1000 --to 2000 --step 20
    python3 research/atlas/correlations.py analyse --runs B2-10000-s001 ... --label B2

`extract` writes one table per run to GraphOfLifeRuns/.pilots/atlas/<run>.npz;
`analyse` reads them and writes the correlations to
GraphOfLifeRuns/.pilots/atlas/correlations_<label>.json.

The agent of a row is the agent at that node when the game began. A node keeps
its id when it is taken, so what is measured after the game (its change of
tokens, ten games later) is of the node, held by whoever holds it then; the
column `kept` says whether that was still the same agent.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")

import networkx as nx
import numpy as np

import book_data as D

OUT = "/home/stefan/Documents/CodingProjects/GraphOfLife Claude/GraphOfLifeRuns/.pilots/atlas"

#: name: (what it is, kind). Kind decides its logarithmic axis: "pos" is positive,
#: and log is the logarithm; "count" can be zero, and log is log(1 + x); "signed"
#: takes either sign, and log is sign(x) log(1 + |x|); "share" is a fraction and
#: "bool" a yes or no, which keep their linear axis only.
VARIABLES = {
    # the agent
    "tokens": ("tokens it held at the start of the game", "pos"),
    "age": ("iterations since it was born", "count"),
    "clones": ("agents carrying its brain, itself included", "pos"),
    # its neighbourhood
    "degree": ("its connections", "pos"),
    "curvature": ("token curvature: the sum over its neighbours of their tokens minus its own", "signed"),
    "nbr_tokens": ("its neighbours' mean tokens", "pos"),
    "nbr_max": ("its richest neighbour's tokens", "pos"),
    "poorer_nbrs": ("the share of its neighbours holding fewer tokens than it", "share"),
    "nbr_degree": ("its neighbours' mean connections", "pos"),
    "kin_nbrs": ("the share of its neighbours carrying its brain", "share"),
    "tokens_per_link": ("its tokens divided by its connections", "pos"),
    # its place in the network
    "clustering": ("the share of its neighbours' pairs that are linked", "share"),
    "triangles": ("triangles it belongs to", "count"),
    "core": ("its k-core number", "pos"),
    "forman": ("Forman-Ricci curvature of the network at it, the mean over its links", "signed"),
    "pagerank": ("PageRank", "pos"),
    "betweenness": ("betweenness, from 64 sampled sources", "count"),
    # what it did in the game
    "home_share": ("the share of its tokens staked on its own node", "share"),
    "targets": ("neighbours it staked on", "count"),
    "max_share": ("its largest stake on a neighbour, as a share of its tokens", "share"),
    "rev_share": ("the share of its tokens marked revolutionary", "share"),
    # what happened
    "change": ("its node's change of tokens over the game", "signed"),
    "rel_change": ("that change divided by the tokens it held", "signed"),
    "kept": ("it kept its own node", "bool"),
    "took": ("nodes it took from others in the game", "count"),
    "alive_10": ("its node still exists ten iterations later", "bool"),
    "change_10": ("its node's change of tokens over ten iterations", "signed"),
    "degree_change_10": ("its node's change of connections over ten iterations", "signed"),
    "child_next": ("it had a child in the next reproduction phase", "bool"),
}
COLUMNS = list(VARIABLES)
#: Pairs whose correlation says how the quantities are made, not how the world behaves:
#: one is computed from the other (curvature contains minus degree times tokens; a change
#: over the game contains minus the tokens held), or both from the same numbers.
DEFINITIONAL = {frozenset(p) for p in [
    *[("curvature", x) for x in ("tokens", "degree", "nbr_tokens", "nbr_max", "poorer_nbrs", "tokens_per_link")],
    *[("poorer_nbrs", x) for x in ("tokens", "nbr_tokens", "nbr_max", "tokens_per_link")],
    ("tokens_per_link", "tokens"), ("tokens_per_link", "degree"), ("nbr_tokens", "nbr_max"),
    *[("forman", x) for x in ("degree", "nbr_degree", "triangles", "clustering")],
    ("clustering", "triangles"), ("triangles", "degree"), ("clustering", "degree"), ("core", "degree"),
    ("pagerank", "degree"),
    ("home_share", "targets"), ("home_share", "max_share"), ("targets", "max_share"),
    ("change", "rel_change"), ("rel_change", "tokens"), ("change", "tokens"), ("change_10", "tokens"),
    ("change_10", "alive_10"), ("degree_change_10", "alive_10"), ("degree_change_10", "degree"),
]}


def definitional(a, b):
    return frozenset((a, b)) in DEFINITIONAL


def forman(G):
    """Forman-Ricci curvature of every link, 4 - deg(u) - deg(v) + 3 triangles(u, v), averaged at each node."""
    deg = dict(G.degree())
    total, count = Counter(), Counter()
    adj = {u: set(G[u]) for u in G}
    for u, v in G.edges():
        f = 4 - deg[u] - deg[v] + 3 * len(adj[u] & adj[v])
        total[u] += f
        total[v] += f
        count[u] += 1
        count[v] += 1
    return {u: total[u] / count[u] for u in count}


def pagerank(G, damping=0.85, tol=1e-10, iters=500):
    """PageRank by power iteration (networkx's needs scipy): the share of a random surfer's time at each node."""
    nodes = list(G)
    index = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    if not n:
        return {}
    src = np.array([index[u] for u, v in G.edges()] + [index[v] for u, v in G.edges()], dtype=int)
    dst = np.array([index[v] for u, v in G.edges()] + [index[u] for u, v in G.edges()], dtype=int)
    deg = np.bincount(src, minlength=n).astype(float)
    p = np.full(n, 1.0 / n)
    for _ in range(iters):
        share = np.where(deg > 0, p / np.maximum(deg, 1), 0.0)
        new = np.bincount(dst, weights=share[src], minlength=n)
        new = damping * (new + p[deg == 0].sum() / n) + (1 - damping) / n
        if np.abs(new - p).sum() < tol:
            p = new
            break
        p = new
    return {u: float(p[index[u]]) for u in nodes}


def rows_at(run, t, horizon=10):
    repro, game = D.frame_at(run, t, 1), D.frame_at(run, t, 2)
    if not repro or not game or not repro["ids"] or "decisions" not in game:
        return []
    try:
        nxt = D.frame_at(run, t + 1, 1)
        later = D.frame_at(run, t + horizon, 2)
    except Exception:
        return []
    ids = repro["ids"]
    tokens = dict(zip(ids, repro["tokens"]))
    brain = dict(zip(ids, repro["brain_ids"]))
    age = dict(zip(ids, repro["ages"]))
    G = nx.Graph()
    G.add_nodes_from(ids)
    G.add_edges_from((a, b) for a, b in repro["edges"] if a in tokens and b in tokens)
    deg = dict(G.degree())
    clones = Counter(brain.values())
    clust = nx.clustering(G)
    tri = nx.triangles(G)
    core = nx.core_number(G)
    form = forman(G)
    pr = pagerank(G)
    btw = nx.betweenness_centrality(G, k=min(64, len(ids)), seed=1, normalized=False)
    delta = dict(zip(game["ids"], game["delta"]))
    dec = game["decisions"]
    won = {w["node"]: w["winner"] for w in dec["winners"]}
    took = Counter(w for v, w in won.items() if w != v)
    alloc = {r["agent"]: r for r in dec["allocations"]}
    parents = {b["agent"] for b in ((nxt.get("decisions") or {}).get("births") or [])} if nxt else set()
    later_tokens = dict(zip(later["ids"], later["tokens"])) if later else {}
    later_deg = Counter()
    for a, b in (later["edges"] if later else []):
        later_deg[a] += 1
        later_deg[b] += 1
    out = []
    for a in ids:
        x = tokens[a]
        nb = list(G[a])
        nt = [tokens[b] for b in nb]
        r = alloc.get(a)
        spent = r["tokens"] if r else 0
        stakes = r["alloc"][1:] if r else []
        revolt = r.get("revolt") if r else None
        row = {
            "tokens": x, "age": age.get(a, np.nan), "clones": clones[brain[a]],
            "degree": deg[a],
            "curvature": sum(nt) - deg[a] * x if nb else np.nan,
            "nbr_tokens": float(np.mean(nt)) if nb else np.nan,
            "nbr_max": max(nt) if nb else np.nan,
            "poorer_nbrs": float(np.mean([y < x for y in nt])) if nb else np.nan,
            "nbr_degree": float(np.mean([deg[b] for b in nb])) if nb else np.nan,
            "kin_nbrs": float(np.mean([brain[b] == brain[a] for b in nb])) if nb else np.nan,
            "tokens_per_link": x / deg[a] if deg[a] else np.nan,
            "clustering": clust[a], "triangles": tri[a], "core": core[a],
            "forman": form.get(a, np.nan), "pagerank": pr[a], "betweenness": btw[a],
            "home_share": (r["alloc"][0] / spent) if r and spent else np.nan,
            "targets": sum(1 for s in stakes if s > 0) if r else np.nan,
            "max_share": (max(stakes) / spent) if r and spent and stakes else (0.0 if r and spent else np.nan),
            "rev_share": (sum(revolt) / spent) if revolt is not None and spent else np.nan,
            "change": delta.get(a, np.nan),
            "rel_change": (delta[a] / x) if a in delta and x else np.nan,
            "kept": float(won.get(a) == a) if a in won else np.nan,
            "took": took.get(a, 0),
            "alive_10": float(a in later_tokens) if later else np.nan,
            "change_10": (later_tokens[a] - x) if a in later_tokens else np.nan,
            "degree_change_10": (later_deg[a] - deg[a]) if a in later_tokens else np.nan,
            "child_next": float(a in parents) if nxt else np.nan,
        }
        out.append([t] + [row[c] for c in COLUMNS])
    return out


def extract(run, t0, t1, step):
    rows = []
    for t in range(t0, t1, step):
        rows += rows_at(run, t)
    os.makedirs(OUT, exist_ok=True)
    table = np.array(rows, dtype=float)
    np.savez_compressed(os.path.join(OUT, f"{run}.npz"), table=table, columns=np.array(["t"] + COLUMNS))
    return table


def axis(values, kind, log):
    if not log or kind in ("share", "bool"):
        return values, np.isfinite(values)
    if kind == "pos":
        ok = np.isfinite(values) & (values > 0)
        return np.log(np.where(ok, values, 1.0)), ok
    if kind == "count":
        ok = np.isfinite(values) & (values >= 0)
        return np.log1p(np.where(ok, values, 0.0)), ok
    ok = np.isfinite(values)
    return np.sign(values) * np.log1p(np.abs(np.where(ok, values, 0.0))), ok


def rank(v):
    """Ranks from 0, ties sharing the mean of the ranks they span."""
    uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    stop = np.cumsum(counts)
    return ((stop - counts + stop - 1) / 2.0)[inv]


def pearson(x, y):
    if len(x) < 30 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pair(table, cols, a, b):
    """Pearson's r on the four combinations of linear and logarithmic axes, and Spearman's rho."""
    ka, kb = VARIABLES[a][1], VARIABLES[b][1]
    va, vb = table[:, cols[a]], table[:, cols[b]]
    out = {}
    for la in (False, True):
        for lb in (False, True):
            if (la and ka in ("share", "bool")) or (lb and kb in ("share", "bool")):
                continue
            xa, oka = axis(va, ka, la)
            xb, okb = axis(vb, kb, lb)
            ok = oka & okb
            out[f"{'log' if la else 'lin'}-{'log' if lb else 'lin'}"] = pearson(xa[ok], xb[ok])
    ok = np.isfinite(va) & np.isfinite(vb)
    out["rank"] = pearson(rank(va[ok]), rank(vb[ok])) if ok.sum() >= 30 else None
    out["n"] = int(ok.sum())
    return out


def analyse(runs, label):
    tables = {}
    for run in runs:
        z = np.load(os.path.join(OUT, f"{run}.npz"))
        tables[run] = z["table"]
        names = list(z["columns"])
    cols = {name: i for i, name in enumerate(names)}
    results = []
    for i, a in enumerate(COLUMNS):
        for b in COLUMNS[i + 1:]:
            per_world = {run: pair(t, cols, a, b) for run, t in tables.items()}
            entry = {"a": a, "b": b, "definitional": definitional(a, b)}
            for key in ("lin-lin", "log-lin", "lin-log", "log-log", "rank"):
                vals = [w.get(key) for w in per_world.values() if w.get(key) is not None]
                if len(vals) >= max(2, len(runs) // 2):
                    entry[key] = {"median": float(np.median(vals)), "min": float(min(vals)),
                                  "max": float(max(vals)),
                                  "same_sign": int(max(sum(v > 0 for v in vals), sum(v < 0 for v in vals))),
                                  "worlds": len(vals)}
            results.append(entry)
    json.dump({"label": label, "runs": runs, "variables": VARIABLES, "pairs": results},
              open(os.path.join(OUT, f"correlations_{label}.json"), "w"), indent=0)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("extract", "analyse"))
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--runs", dest="named", nargs="*", default=[])
    ap.add_argument("--label", default="")
    ap.add_argument("--from", dest="t0", type=int, default=1000)
    ap.add_argument("--to", dest="t1", type=int, default=2000)
    ap.add_argument("--step", type=int, default=20)
    a = ap.parse_args()
    if a.what == "extract":
        for run in a.runs:
            table = extract(run, a.t0, a.t1, a.step)
            print(run, table.shape, flush=True)
    else:
        analyse(a.runs or a.named, a.label)


if __name__ == "__main__":
    main()
