"""Space alone with a link cap: births need room; repair as a chain or a ring of the dead agent's neighbour groups."""
import sys
sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife")
import numpy as np, networkx as nx
from gol_series import _structure
from cuts import node_cut_risk

def groups_of(G, nb):
    groups, where = [], {}
    for x in sorted(nb):
        joined = [where[y] for y in G[x] if y in where]
        if not joined:
            where[x] = len(groups); groups.append([x]); continue
        t = min(joined); groups[t].append(x); where[x] = t
        for g in set(joined) - {t}:
            for y in groups[g]: where[y] = t
            groups[t].extend(groups[g]); groups[g] = []
    return [g for g in groups if g]

def run(K=8, k_extra=2, ring=False, need_room=True, target=1500, steps=300, death=0.02, seed=0):
    rng = np.random.RandomState(seed)
    G = nx.watts_strogatz_graph(50, 6, 0.2, seed=seed)
    nxt = 50
    for t in range(steps):
        n = G.number_of_nodes()
        birth = death * target / max(n, 1) + (0.05 if n < target else 0)
        for u in list(G.nodes()):
            if G.degree(u) >= K or rng.random_sample() >= birth:
                continue
            room = [v for v in G[u] if G.degree(v) < K]
            if need_room and k_extra and not room:
                continue
            c = nxt; nxt += 1
            G.add_edge(c, u)
            for v in rng.permutation(room)[:k_extra]:
                G.add_edge(c, int(v))
        for u in list(G.nodes()):
            if u in G and rng.random_sample() < death:
                nb = list(G[u]); G.remove_node(u)
                gs = groups_of(G, nb)
                pairs = list(zip(gs, gs[1:])) + ([(gs[-1], gs[0])] if ring and len(gs) >= 3 else [])
                for g1, g2 in pairs:
                    a = min(g1, key=lambda z: (G.degree(z), z)); b = min(g2, key=lambda z: (G.degree(z), z))
                    if a != b: G.add_edge(a, b)
        big = max(nx.connected_components(G), key=len)
        G = G.subgraph(big).copy()
    ids = list(G.nodes())
    s = _structure(ids, [list(e) for e in G.edges()])
    core = nx.core_number(G)
    deg = np.array([d for _, d in G.degree()])
    return dict(n=len(ids), deg=deg.mean(), leaves=np.mean(deg == 1), core2=s["coreShare"], core3=np.mean([c >= 3 for c in core.values()]),
                ncut=node_cut_risk(G), cut=s["cutRisk"], dim=s["dimension"], diam=s["diameter"], maxd=deg.max())

which = int(sys.argv[1])
cases = [dict(K=8, ring=False), dict(K=8, ring=True), dict(K=12, ring=False), dict(K=12, ring=True), dict(K=8, ring=True, need_room=False), dict(K=6, ring=True)]
for c in cases[which::3]:
    r = run(**c)
    print(c, " ".join(f"{a}={b:.3g}" for a, b in r.items()), flush=True)
