"""
Pilot: three questions that shape the research design.

  1. Does the graph erode?  Edges are created only at birth and destroyed
     every iteration by pruning, which is an asymmetry worth checking.
  2. Does one lineage fixate, or is there continual turnover?
  3. Do the agents' own choices leave a structural signature, or does the
     graph look like what chance would produce?

Small and short on purpose: this is for calibrating the real experiments.
"""
import random, sys
import numpy as np
sys.path.insert(0, ".")
from gol_config import SimConfig
import GraphOfLifeSimple as G

def run(seed, iters=45, **over):
    random.seed(seed); np.random.seed(seed)
    cfg = SimConfig(total_tokens=2500, n_nodes=50, k_neighbors=6,
                    hidden_layers=[12, 10], seed=seed, **over)
    w = G.new_world(cfg)
    rows = []
    for _ in range(iters):
        w.step(record_decisions=False)
        if w.is_extinct():
            break
        n, e = w.G.number_of_nodes(), w.G.number_of_edges()
        roots = [b.brain_id for b in w.brains.values()]
        counts = np.bincount(np.array(roots) % 100000)
        top = counts.max() / max(1, len(roots))
        rows.append(dict(it=w.iteration, n=n, e=e, deg=2 * e / max(1, n),
                         lineages=len(set(roots)), top=top,
                         top_id=int(counts.argmax())))
    return cfg, w, rows

print("edge dynamics and lineage turnover, 5 seeds\n")
print(f"{'seed':>5} {'agents':>7} {'<k> start':>10} {'<k> end':>9} "
      f"{'distinct brains':>16} {'largest share':>14} {'dominant changes':>17}")
for seed in (2, 5, 8, 13, 21):
    cfg, w, rows = run(seed)
    if len(rows) < 5:
        print(f"{seed:>5}   died out at iteration {w.iteration}")
        continue
    changes = sum(1 for a, b in zip(rows, rows[1:]) if a["top_id"] != b["top_id"])
    print(f"{seed:>5} {rows[-1]['n']:>7,} {rows[0]['deg']:>10.2f} {rows[-1]['deg']:>9.2f} "
          f"{rows[-1]['lineages']:>16,} {rows[-1]['top']:>13.1%} "
          f"{changes:>10} / {len(rows) - 1}")
