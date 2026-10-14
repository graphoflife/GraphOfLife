import sys
sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")
import numpy as np
from gol_config import SimConfig
from GraphOfLifeSimple import make_brain

rng = np.random.RandomState(0)
def inputs(cfg, cols=8):
    lay = cfg.input_layout()
    X = np.zeros((cfg.n_inputs(), cols))
    for name, (a, b, kind) in lay.items():
        if kind == "magnitude":
            X[a:b] = rng.uniform(0, 4, size=(b - a, cols))
        elif name == "noise":
            X[a:b] = rng.uniform(-2, 2, size=(b - a, cols))
        elif kind == "bit":
            X[a:b] = np.tanh(rng.normal(0, 1, size=(b - a, cols)))
        else:
            X[a:b] = rng.randint(0, 2, size=(b - a, cols))
    return X

for hl in ([50, 45, 40, 35, 30], [16], [32], [16, 16], []):
    for m in (30, 5):
        cfg = SimConfig.for_new_run(hidden_layers=hl, message_amount=m, seed=1)
        gains, colspread = [], []
        for i in range(30):
            br = make_brain(cfg, i, rng)
            X = inputs(cfg)
            eps = rng.normal(0, 0.1, size=X.shape)
            Y0 = br.forward(X); Y1 = br.forward(X + eps)
            gains.append(np.sqrt(np.mean((Y1 - Y0) ** 2)) / np.sqrt(np.mean(eps ** 2)))
            # how much outputs differ between candidates relative to their mean size
            colspread.append(np.mean(np.std(Y0, axis=1)) / (np.mean(np.abs(Y0)) + 1e-12))
        print(f"hidden={str(hl):22s} m={m:2d} gain median {np.median(gains):.4f}  column spread {np.median(colspread):.3f}")
