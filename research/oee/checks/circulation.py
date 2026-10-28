"""
Tokens that go round over time: the circulation in a world's token flow (Helmholtz–Hodge).

Add up, over a window of games, the tokens each connection carried each way;
the net flow F on a connection is what went one way minus what came back. Any
flow on a network splits into two parts that cannot be mistaken for each other:

  gradient     F = phi(u) - phi(v): tokens running downhill along a "pressure"
               phi, from some agents to others, one way only; no loop can be
               made of it.
  circulation  the rest: flow that goes round loops of the network. It needs
               cycles in the network to exist at all, so it is a property of the
               topology as much as of the play.

The gradient part is found by least squares (the pressure phi that explains as
much of F as possible, by conjugate gradients on the network's Laplacian), the
circulation is what is left. For a flow with no
structure at all — random amounts on every connection — the circulation would
hold the share (E - N + C) / E of the flow's energy (E connections, N agents, C
pieces): the share of the network's dimensions that are loops. Less than that
means the flow runs downhill; more means it goes round.

The random level assumes every connection carries a flow of the same size. A
fairer chance level keeps each connection's amount and only throws its
direction at random (sign flips, `draws` times): what is left above that is
direction coordinated along loops and routes, which no single connection's
amount can make.

The flow is taken whole and split in two: gifts (stakes that did not take their
target: tokens handed to whoever holds the node) and takeovers (stakes that took
it: wealth and brain moving on together). Gifts going round a ring would be
generalised reciprocity, a Kula ring; takeovers going round would be raiding in
a circle.

Per window: the circulation's share of the flow's energy, the random level, the
sign-flip level, and how much of the flow, of its gradient part and of its
circulation repeats in the next window (the correlation of each connection's
value between consecutive windows, over the connections both windows used):
rings that persist.

    python3 research/oee/checks/circulation.py RUN [...] --from 1000 --to 2000 --window 50
"""
import argparse
import json
import sys

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife")

import numpy as np

import book_data as D


def pressure(iu, iv, F, N, tol=1e-8, iters=20000):
    """The phi that best explains F as phi(u) - phi(v): conjugate gradients on the Laplacian,
    preconditioned by the degrees."""
    deg = (np.bincount(iu, minlength=N) + np.bincount(iv, minlength=N)).astype(float)

    def L(x):
        return deg * x - np.bincount(iu, weights=x[iv], minlength=N) - np.bincount(iv, weights=x[iu], minlength=N)

    d = np.bincount(iu, weights=F, minlength=N) - np.bincount(iv, weights=F, minlength=N)
    inv = 1.0 / np.maximum(deg, 1.0)
    x = np.zeros(N)
    r = d.copy()
    z = inv * r
    p = z.copy()
    rz = r @ z
    dd = max(d @ d, 1e-300)
    for _ in range(iters):
        if r @ r <= tol * tol * dd:
            break
        Ap = L(p)
        alpha = rz / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        z = inv * r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def pieces(iu, iv, N):
    parent = list(range(N))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(iu.tolist(), iv.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(N)})


KINDS = ("all", "gift", "take")


def window_flows(run, t0, t1):
    """Net tokens carried over each connection from the lower id to the higher, games t0..t1-1, by kind."""
    nets = {k: {} for k in KINDS}
    for t in range(t0, t1):
        f = D.frame_at(run, t, 2)
        if not f or "decisions" not in f:
            continue
        winners = {w["node"]: w["winner"] for w in f["decisions"]["winners"]}
        for a in f["decisions"]["allocations"]:
            u = a["agent"]
            for v, x in zip(a["targets"][1:], a["alloc"][1:]):
                if x > 0 and v != u:
                    key = (u, v) if u < v else (v, u)
                    x = x if u < v else -x
                    for k in ("all", "take" if winners.get(v) == u else "gift"):
                        nets[k][key] = nets[k].get(key, 0) + x
    return nets


def split(iu, iv, F, N):
    phi = pressure(iu, iv, F, N)
    G = phi[iu] - phi[iv]
    return G, F - G


def hodge(net, draws=0, rng=None):
    """Split a net flow into gradient and circulation; per-edge parts and the shares."""
    edges = list(net)
    nodes = sorted({u for e in edges for u in e})
    index = {u: i for i, u in enumerate(nodes)}
    E, N = len(edges), len(nodes)
    iu = np.array([index[u] for u, v in edges])
    iv = np.array([index[v] for u, v in edges])
    F = np.array([net[e] for e in edges], dtype=float)
    G, C = split(iu, iv, F, N)
    energy = float((F ** 2).sum())
    row = {"edges": E, "agents": N, "circulation_share": float((C ** 2).sum() / energy) if energy else float("nan"),
           "random_share": (E - N + pieces(iu, iv, N)) / E if E else float("nan")}
    if draws and energy:
        flips = []
        for _ in range(draws):
            Fs = F * rng.choice((-1.0, 1.0), size=E)
            Cs = split(iu, iv, Fs, N)[1]
            flips.append(float((Cs ** 2).sum() / energy))
        row["flip_share"] = float(np.mean(flips))
    return {e: (f, g, c) for e, f, g, c in zip(edges, F, G, C)}, row


def correlation(now, prev, part):
    common = [e for e in now if e in prev]
    if len(common) <= 10:
        return None
    a = np.array([now[e][part] for e in common])
    b = np.array([prev[e][part] for e in common])
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyse(run, t0, t1, window, draws=4, seed=1):
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in KINDS}
    prev = {k: None for k in KINDS}
    for start in range(t0, t1, window):
        nets = window_flows(run, start, min(t1, start + window))
        for k in KINDS:
            if len(nets[k]) < 10:
                prev[k] = None
                continue
            parts, row = hodge(nets[k], draws, rng)
            if prev[k] is not None:
                for name, i in (("flow", 0), ("gradient", 1), ("circulation", 2)):
                    row["persistence_" + name] = correlation(parts, prev[k], i)
            row["from"] = start
            rows[k].append(row)
            prev[k] = parts
    out = {"run": run, "from": t0, "to": t1, "window": window}
    for k in KINDS:
        summary = {"windows": len(rows[k])}
        for key in ("circulation_share", "random_share", "flip_share", "persistence_flow",
                    "persistence_gradient", "persistence_circulation"):
            v = [r[key] for r in rows[k] if r.get(key) is not None]
            summary[key] = float(np.median(v)) if v else None
        out[k] = summary
        out[k + "_rows"] = rows[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--from", dest="t0", type=int, default=1000)
    ap.add_argument("--to", dest="t1", type=int, default=2000)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = []
    for run in a.runs:
        r = analyse(run, a.t0, a.t1, a.window, a.draws)
        res.append(r)
        print(json.dumps({k: r[k] for k in ("run", "window") + KINDS}), flush=True)
        if a.out:
            json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
