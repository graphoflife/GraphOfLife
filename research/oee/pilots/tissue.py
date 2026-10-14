"""Space alone: births that insert a node into triangles, random deaths, the cull. What network results?"""
import sys
sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")
import numpy as np, networkx as nx
from gol_series import _structure
from cuts import node_cut_risk

def run(k_extra, target=1500, steps=300, death=0.02, seed=0, prune_leaves=False):
    rng = np.random.RandomState(seed)
    G = nx.watts_strogatz_graph(50, 6, 0.2, seed=seed)
    nxt = 50
    for t in range(steps):
        n = G.number_of_nodes()
        birth = death * target / max(n, 1) * 1.0 + (0.05 if n < target else 0)
        nodes = list(G.nodes())
        for u in nodes:
            if rng.random_sample() < birth:
                nb = list(G[u])
                c = nxt; nxt += 1
                G.add_edge(c, u)
                if nb:
                    for v in rng.choice(nb, size=min(k_extra, len(nb)), replace=False):
                        G.add_edge(c, int(v))
        for u in list(G.nodes()):
            if rng.random_sample() < death:
                G.remove_node(u)
        if G.number_of_nodes() == 0:
            break
        big = max(nx.connected_components(G), key=len)
        G = G.subgraph(big).copy()
    ids = list(G.nodes())
    s = _structure(ids, [list(e) for e in G.edges()])
    core = nx.core_number(G)
    return dict(n=len(ids), deg=2*G.number_of_edges()/len(ids), core2=s["coreShare"], core3=np.mean([c >= 3 for c in core.values()]),
                ncut=node_cut_risk(G), cut=s["cutRisk"], dim=s["dimension"], trans=s["transitivity"], diam=s["diameter"],
                maxd=max(dict(G.degree()).values())/len(ids))

for k in (0, 1, 2, 3):
    for d in (0.01, 0.03):
        r = run(k, death=d)
        print(f"extra links {k} death {d}: " + " ".join(f"{a}={b:.3g}" for a, b in r.items()), flush=True)
