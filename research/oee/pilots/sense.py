"""Which inputs change an evolved agent's stakes? Perturb one input group at a time and count changed decisions."""
import sys, json, os
sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from GraphOfLifeSimple import _apportion
from proto import ProtoWorld, ProtoConfig, set_proto

def alloc(w, u, Y, t):
    H = w.heads
    scores = np.asarray(Y[H["BLOTTO"], :], dtype=float)
    mode = np.mean(Y[H["BLOTTO_MODE"], :], axis=1)
    stake = t - 1
    if mode[0] > mode[1]:
        a = _apportion(scores, max(0, stake))
    else:
        a = np.zeros(Y.shape[1], dtype=int); a[int(np.argmax(scores))] = max(0, stake)
    a[0] += 1
    return a

def run(path, it, sample=300, rng=np.random.RandomState(0)):
    meta = json.load(open(path + "/config.json"))
    set_proto(meta["proto"])
    if meta["cfg"].get("brain_kind") == "grow":
        import grow; grow.install()
    cfg = ProtoConfig.from_dict(meta["cfg"], stored=False)
    z = np.load(f"{path}/ckpt_{it:06d}.npz"); blob = {k: z[k] for k in z.files}
    w = ProtoWorld.from_checkpoint(blob, cfg)
    import grow; grow.restore_active(w, blob)
    # One iteration first, so the memory of last game's stakes is filled in:
    # a checkpoint does not carry it.
    w.step(record_decisions=False)
    feats = w._precompute_features()
    lay = cfg.input_layout()
    groups = {"own state": ["own_tokens", "own_degree", "own_token_quantiles", "own_degree_quantiles"],
              "its state": ["its_tokens", "its_degree", "its_token_quantiles", "its_degree_quantiles"],
              "messages": ["message_to_self", "message_to_it", "message_from_it", "message_its_own"],
              "noise": ["noise"], "memory": ["stake_from_it", "stake_to_it"]}
    nodes = sorted(w.G.nodes())
    nodes = [nodes[i] for i in rng.choice(len(nodes), min(sample, len(nodes)), replace=False)]
    counts = {g: 0 for g in groups}; used = 0
    for u in nodes:
        t = int(w.tokens[u])
        if t < 3:
            continue
        cands = [u] + sorted(w.G.neighbors(u))
        log_deg, neighs, q_tok, q_deg, log_tok, at_risk = feats
        X = w._inputs(u, cands, log_deg, q_tok, q_deg, log_tok, at_risk)
        base = alloc(w, u, w.brains[u].forward(X), t)
        used += 1
        for g, names in groups.items():
            X2 = X.copy()
            for name in names:
                if name not in lay:
                    continue
                a, b, _ = lay[name]
                if name == "noise":
                    X2[a:b] = rng.uniform(-2, 2, size=X2[a:b].shape)
                elif name.startswith("message") or name.startswith("stake"):
                    X2[a:b] = 0
                else:
                    X2[a:b] = X2[a:b] + np.log(2)
            if not np.array_equal(alloc(w, u, w.brains[u].forward(X2), t), base):
                counts[g] += 1
    return used, {g: round(c / max(1, used), 2) for g, c in counts.items()}

for path, it in [(p.split(":")[0], int(p.split(":")[1])) for p in sys.argv[1:]]:
    print(path, it, *run(path, it))
