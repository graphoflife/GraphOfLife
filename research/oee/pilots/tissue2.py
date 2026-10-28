"""Space alone, with wound healing (neighbours of the dead join up) or token-limited walks closing loops."""
import sys
sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife")
import numpy as np, networkx as nx
from gol_series import _structure
from cuts import node_cut_risk

def run(k_extra=1, target=1500, steps=300, death=0.02, seed=0, heal="none", walk_p=0.0, walk_len=4):
    rng = np.random.RandomState(seed)
    G = nx.watts_strogatz_graph(50, 6, 0.2, seed=seed)
    nxt = 50
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
        if walk_p:
            for u in list(G.nodes()):
                if rng.random_sample() < walk_p:
                    here = u
                    L = rng.randint(1, walk_len + 1) if walk_len > 0 else 1
                    for _ in range(L):
                        nb = list(G[here])
                        if not nb: break
                        here = nb[rng.randint(len(nb))]
                    if here != u:
                        G.add_edge(u, here)
        for u in list(G.nodes()):
            if rng.random_sample() < death:
                nb = list(G[u])
                G.remove_node(u)
                if heal == "ring" and len(nb) > 1:
                    rng.shuffle(nb)
                    for a, b in zip(nb, nb[1:] + nb[:1]):
                        if a != b: G.add_edge(a, b)
                elif heal == "path" and len(nb) > 1:
                    rng.shuffle(nb)
                    for a, b in zip(nb, nb[1:]):
                        G.add_edge(a, b)
        big = max(nx.connected_components(G), key=len)
        G = G.subgraph(big).copy()
    ids = list(G.nodes())
    s = _structure(ids, [list(e) for e in G.edges()])
    core = nx.core_number(G)
    return dict(n=len(ids), deg=2*G.number_of_edges()/len(ids), core2=s["coreShare"], core3=np.mean([c >= 3 for c in core.values()]),
                ncut=node_cut_risk(G), cut=s["cutRisk"], dim=s["dimension"], trans=s["transitivity"], diam=s["diameter"],
                maxd=max(dict(G.degree()).values())/len(ids))

cases = [dict(k_extra=1, heal="path"), dict(k_extra=1, heal="ring"), dict(k_extra=0, heal="path"),
         dict(k_extra=1, walk_p=0.02, walk_len=4), dict(k_extra=1, walk_p=0.05, walk_len=8), dict(k_extra=1, walk_p=0.02, walk_len=16)]
which = int(sys.argv[1])
for c in cases[which::3]:
    r = run(**c)
    print(c, " ".join(f"{a}={b:.3g}" for a, b in r.items()), flush=True)
