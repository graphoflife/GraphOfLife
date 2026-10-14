import sys
sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")
from gol_config import SimConfig
from GraphOfLifeSimple import new_world
from proto import make_world
cfg = SimConfig.for_new_run(seed=7, total_tokens=3000, hidden_layers=[16])
a = new_world(cfg); b = make_world(SimConfig.for_new_run(seed=7, total_tokens=3000, hidden_layers=[16]), {"mor": "off"})
for t in range(25):
    fa = a.step(record_decisions=False); fb = b.step(record_decisions=False)
    for x, y in zip(fa, fb):
        assert x["ids"] == y["ids"] and x["tokens"] == y["tokens"] and x["brain_ids"] == y["brain_ids"] and x["edges"] == y["edges"], t
print("identical for 25 iterations; nodes", len(fa[1]["ids"]), "counters", b.c)
