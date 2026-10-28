"""The B2 recipe run by the engine itself (not the prototype): health and economy over time."""
import sys, json, time
sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife")
import numpy as np, networkx as nx
from gol_config import SimConfig
from GraphOfLifeSimple import new_world, worst_node_cut_share
from gol_series import _structure

B2 = dict(brain_kind="float16", hidden_layers=[16], message_amount=8, message_prepass=False,
          allow_handover=False, mutation_probability=0.2, mutate_on_replication=True, incumbency=1.5,
          birth_links=2, mend_on_death=True, prune_needs=2, valence=8, keep_home=0.3,
          max_age=100, min_child=10)

def main(seed, T=5000, iters=400, out=None):
    cfg = SimConfig(total_tokens=T, seed=seed, **B2)
    cfg.validate()
    w = new_world(cfg)
    rows = []
    t0 = time.time()
    for it in range(iters):
        f = w.step(record_decisions=False)
        if (it + 1) % 25 == 0:
            G = w.G; ids = list(G)
            s = _structure(ids, [list(e) for e in G.edges()])
            deg = np.array([d for _, d in G.degree()])
            tok = np.array([w.tokens[u] for u in ids])
            rows.append(dict(it=w.iteration, n=len(ids), deg=float(deg.mean()), core2=s["coreShare"],
                             ncut=worst_node_cut_share(ids, G.adj), cut=s["cutRisk"], hub=float(deg.max()/len(ids)),
                             dim=s["dimension"], one=float(np.mean(tok == 1)), secs=round(time.time()-t0)))
            print(json.dumps(rows[-1]), flush=True)
    return rows

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 5000)
