"""Brains of two different runs, same rules, in one arena: does the bigger world evolve better brains?

    python3 cross.py ARENA_RUN:IT A_RUN:IT B_RUN:IT [--T 20] [--coins 4] [--jobs 4]
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from proto import ProtoWorld, ProtoConfig, set_proto

LABEL_B = 1_000_000

def blob(spec):
    run, it = spec.split(":")
    with np.load(os.path.join(run, f"ckpt_{int(it):06d}.npz")) as z:
        return {k: z[k] for k in z.files}

def one(args):
    arena, a, b, T, seed = args
    run = arena.split(":")[0]
    meta = json.load(open(os.path.join(run, "config.json")))
    set_proto(meta.get("proto", {}))
    cfg = ProtoConfig.from_dict(meta["cfg"], stored=False)
    cfg.mutation_probability = 0.0
    cfg.extinction_threshold = 0
    S, A, B = blob(arena), blob(a), blob(b)
    rng = np.random.RandomState(seed)
    n = len(S["ids"]); nl = int(S["n_layers"][0])
    from_b = rng.random_sample(n) < 0.5
    pa = rng.randint(0, len(A["ids"]), size=n); pb = rng.randint(0, len(B["ids"]), size=n)
    out = dict(S)
    for i in range(nl):
        for k in (f"W{i}", f"b{i}"):
            out[k] = np.where(from_b[:, None, None], B[k][pb], A[k][pa])
    out["brain_ids"] = np.where(from_b, LABEL_B + pb, pa).astype(np.int64)
    out["parent_brain_ids"] = np.full(n, -1, dtype=np.int64)
    out["msg_to"] = np.zeros(0, dtype=np.int64); out["msg_from"] = np.zeros(0, dtype=np.int64)
    out["msg_values"] = np.zeros((0, cfg.message_amount))
    w = ProtoWorld.from_checkpoint(out, cfg)
    w.rng = np.random.RandomState(seed + 7)
    for _ in range(T):
        w.step(record_decisions=False)
    ids = list(w.G.nodes())
    return float(np.mean([w.brains[u].brain_id >= LABEL_B for u in ids]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arenas"); ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--T", type=int, default=20); ap.add_argument("--coins", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    x = ap.parse_args()
    tasks = [(arena, x.a, x.b, x.T, 100 + k) for arena in x.arenas.split(",") for k in range(x.coins)]
    with ProcessPoolExecutor(x.jobs) as ex:
        res = list(ex.map(one, tasks))
    print(f"B ({x.b}) holds {np.mean(res):.3f} of agents against A ({x.a}); contests: {', '.join(f'{r:.2f}' for r in res)}")
