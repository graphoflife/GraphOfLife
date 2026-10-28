"""Space alone: path healing that happens with probability q per death."""
import sys
sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife")
import numpy as np, networkx as nx
from gol_series import _structure
from cuts import node_cut_risk

def run(k_extra=1, target=1500, steps=300, death=0.02, seed=0, q=0.5):
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
        for u in list(G.nodes()):
            if u in G and rng.random_sample() < death:
                nb = list(G[u])
                G.remove_node(u)
                if len(nb) > 1 and rng.random_sample() < q:
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

which = int(sys.argv[1])
cases = [dict(k_extra=1, q=0.2), dict(k_extra=1, q=0.4), dict(k_extra=1, q=0.6), dict(k_extra=2, q=0.2), dict(k_extra=2, q=0.4), dict(k_extra=1, q=0.4, seed=1)]
for c in cases[which::3]:
    r = run(**c)
    print(c, " ".join(f"{a}={b:.3g}" for a, b in r.items()), flush=True)
