"""Space alone: minimal wound repair — after a death, rejoin only those neighbours that lost touch within r hops."""
import sys
sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")
import numpy as np, networkx as nx
from gol_series import _structure
from cuts import node_cut_risk

def ball(G, a, r):
    seen = {a}; frontier = [a]
    for _ in range(r):
        nxt = []
        for x in frontier:
            for y in G[x]:
                if y not in seen:
                    seen.add(y); nxt.append(y)
        frontier = nxt
    return seen

def repair(G, nb, r, rng):
    nb = [x for x in nb if x in G]
    if len(nb) < 2:
        return 0
    added = 0
    groups = []
    for x in nb:
        for g in groups:
            if x in g["ball"]:
                g["members"].append(x); break
        else:
            groups.append({"members": [x], "ball": ball(G, x, r)})
    # join the groups in a chain, lowest-degree members first
    for g1, g2 in zip(groups, groups[1:]):
        a = min(g1["members"], key=lambda z: G.degree(z))
        b = min(g2["members"], key=lambda z: G.degree(z))
        G.add_edge(a, b); added += 1
    return added

def run(k_extra=1, target=1500, steps=300, death=0.02, seed=0, r=3):
    rng = np.random.RandomState(seed)
    G = nx.watts_strogatz_graph(50, 6, 0.2, seed=seed)
    nxt = 50; repairs = 0
    for t in range(steps):
        n = G.number_of_nodes()
        birth = death * target / max(n, 1) + (0.05 if n < target else 0)
        for u in list(G.nodes()):
            if rng.random_sample() < birth:
                nb = list(G[u])
                c = nxt; nxt += 1
                G.add_edge(c, u)
                if nb:
                    for v in rng.choice(nb, size=min(k_extra, len(nb)), replace=False):
                        G.add_edge(c, int(v))
        for u in list(G.nodes()):
            if u in G and rng.random_sample() < death:
                nb = list(G[u])
                G.remove_node(u)
                repairs += repair(G, nb, r, rng)
        big = max(nx.connected_components(G), key=len)
        G = G.subgraph(big).copy()
    ids = list(G.nodes())
    s = _structure(ids, [list(e) for e in G.edges()])
    core = nx.core_number(G)
    return dict(n=len(ids), deg=2*G.number_of_edges()/len(ids), core2=s["coreShare"], core3=np.mean([c >= 3 for c in core.values()]),
                ncut=node_cut_risk(G), cut=s["cutRisk"], dim=s["dimension"], trans=s["transitivity"], diam=s["diameter"],
                maxd=max(dict(G.degree()).values())/len(ids), repairs_per_step=repairs/steps)

which = int(sys.argv[1])
cases = [dict(k_extra=1, r=1), dict(k_extra=2, r=1), dict(k_extra=1, r=1, seed=1), dict(k_extra=0, r=1), dict(k_extra=1, r=0), dict(k_extra=2, r=1, seed=1)]
for c in cases[which::3]:
    r = run(**c)
    print(c, " ".join(f"{a}={b:.3g}" for a, b in r.items()), flush=True)
