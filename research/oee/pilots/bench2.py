import sys, time
sys.path.insert(0, ".")
import numpy as np
from proto import make_world, make_config, set_proto
proto = {"mor": "all", "incumbency": 1.5, "birth_links": 2, "repair": True, "safe_prune": 2, "newborn_home": True, "max_age": 100, "valence": 8, "keep_one": True, "min_child": 10}
for label, kw in (("m30 prepass", {}), ("m8 no prepass", {"message_amount": 8, "message_prepass": False})):
    set_proto(proto)
    cfg = make_config(seed=6, total_tokens=4000, hidden_layers=[16], **kw)
    w = make_world(cfg, proto)
    for t in range(40):
        w.step(record_decisions=False)
    t0 = time.time()
    for t in range(10):
        w.step(record_decisions=False)
    dt = (time.time() - t0) / 10
    print(f"{label:16s} nodes {w.G.number_of_nodes()}  {dt:.2f} s/it  {1000*dt/w.G.number_of_nodes():.2f} ms/agent", flush=True)
