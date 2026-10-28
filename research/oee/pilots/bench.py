import sys, time
sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife")
from gol_config import SimConfig
from GraphOfLifeSimple import new_world
import numpy as np

def bench(label, iters=40, **kw):
    cfg = SimConfig.for_new_run(seed=1, **kw)
    cfg.validate()
    w = new_world(cfg)
    t0 = time.time(); times = []
    for t in range(iters):
        a = time.time()
        w.step(record_decisions=False)
        times.append(time.time() - a)
    n = w.G.number_of_nodes(); e = w.G.number_of_edges()
    print(f"{label:40s} n={n:5d} e={e:6d} last10 {np.mean(times[-10:]):.3f}s/it  per-agent {1e3*np.mean(times[-10:])/max(n,1):.2f} ms", flush=True)

which = sys.argv[1]
if which == "a":
    bench("B1 5k", total_tokens=5000)
    bench("B1 5k [16]", total_tokens=5000, hidden_layers=[16])
elif which == "b":
    bench("B1 5k [16] m5", total_tokens=5000, hidden_layers=[16], message_amount=5)
    bench("B1 5k [16] m5 noprepass", total_tokens=5000, hidden_layers=[16], message_amount=5, message_prepass=False)
