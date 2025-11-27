
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, math, random, statistics
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

# --------------------------- Utilities ---------------------------

def list_step_files(run_dir: str) -> List[str]:
    fs = [f for f in os.listdir(run_dir) if f.startswith("step_") and f.endswith(".json")]
    fs.sort()
    return [os.path.join(run_dir, f) for f in fs]

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def build_graph(nodes, edges) -> Tuple[Dict[int, List[int]], List[int]]:
    """Return (adj, node_list) where adj is dict: node -> list of neighbors."""
    node_ids = [int(n["agent_id"]) for n in nodes]
    idx = {u:i for i,u in enumerate(node_ids)}
    adj = {u: [] for u in node_ids}
    for (u,v) in edges:
        u = int(u); v = int(v)
        if u==v: 
            continue
        if u in adj and v in adj:
            adj[u].append(v)
            adj[v].append(u)
    return adj, node_ids

def largest_cc_size(adj: Dict[int, List[int]]) -> Tuple[int, int]:
    """Return sizes (S1,S2) of the largest and 2nd largest connected components."""
    seen = set()
    sizes = []
    for u in adj.keys():
        if u in seen: 
            continue
        stack = [u]; seen.add(u); s = 1
        while stack:
            x = stack.pop()
            for w in adj[x]:
                if w not in seen:
                    seen.add(w); stack.append(w); s += 1
        sizes.append(s)
    sizes.sort(reverse=True)
    s1 = sizes[0] if sizes else 0
    s2 = sizes[1] if len(sizes) > 1 else 0
    return s1, s2

def spectral_radius_power_iteration(adj: Dict[int, List[int]], max_iter: int = 500, tol: float = 1e-6) -> float:
    """Largest eigenvalue of A via power iteration without SciPy; A is unweighted adjacency."""
    if not adj:
        return 0.0
    nodes = list(adj.keys())
    n = len(nodes)
    node2i = {u:i for i,u in enumerate(nodes)}
    v = np.random.rand(n)
    v = v / max(np.linalg.norm(v), 1e-12)
    lam_old = 0.0
    for _ in range(max_iter):
        Av = np.zeros(n)
        for u in nodes:
            i = node2i[u]
            s = 0.0
            for w in adj[u]:
                s += v[node2i[w]]
            Av[i] = s
        norm = max(np.linalg.norm(Av), 1e-12)
        v = Av / norm
        lam = float(np.dot(v, Av) / (np.dot(v, v) + 1e-12))
        if abs(lam - lam_old) < tol * max(1.0, abs(lam_old)):
            break
        lam_old = lam
    return lam_old

def ccdf(vals: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return CCDF (x, P(X >= x)) for integer vals."""
    if not vals:
        return np.array([]), np.array([])
    counts = Counter(vals)
    xs = sorted(counts.keys())
    tail = []
    total = len(vals)
    s = 0
    for x in reversed(xs):
        s += counts[x]
        tail.append((x, s / total))
    tail.reverse()
    xs2 = np.array([x for (x, p) in tail], dtype=float)
    ps2 = np.array([p for (x, p) in tail], dtype=float)
    return xs2, ps2

# ---------------------- Power-law (CSN-style) ----------------------
# Minimal implementation for discrete data with continuous approx for MLE.

def mle_alpha_continuous(x, xmin):
    """Continuous power-law MLE (alpha > 1)."""
    x = np.asarray([v for v in x if v >= xmin], dtype=float)
    n = len(x)
    if n == 0: 
        return None
    return 1.0 + n / np.sum(np.log(x / float(xmin)))

def ks_distance_powerlaw(x, xmin, alpha):
    """KS distance between empirical CDF and fitted continuous power law for x >= xmin."""
    x = sorted([v for v in x if v >= xmin])
    n = len(x)
    if n == 0: 
        return float("inf")
    uniq = sorted(set(x))
    def F(v):
        return 1.0 - (v/float(xmin))**(1.0 - alpha)
    maxd = 0.0
    # Precompute cumulative counts for speed
    counts = Counter(x)
    cum = 0
    for v in uniq:
        cum += counts[v]
        emp = cum / n
        model = F(v)
        if emp > 1 or emp < 0 or model > 1 or model < 0:
            continue
        maxd = max(maxd, abs(emp - model))
    return maxd

def fit_powerlaw_csn(x, xmin_candidates=None):
    """Select xmin by minimizing KS distance (continuous approx), return dict with alpha, xmin, ks."""
    x = [int(v) for v in x if v >= 1]
    if len(x) < 20:
        return {"alpha": None, "xmin": None, "ks": None, "n_tail": 0}
    xs = sorted(set(x))
    if xmin_candidates is None:
        xmin_candidates = [v for v in xs if v <= xs[-1] and v >= min(xs)]
        xmin_candidates = [v for v in xmin_candidates if sum(1 for z in x if z >= v) >= 20]
        if not xmin_candidates:
            xmin_candidates = [xs[max(0, len(xs)//2)]]
    best = {"alpha": None, "xmin": None, "ks": float("inf"), "n_tail": 0}
    for xmin in xmin_candidates:
        alpha = mle_alpha_continuous(x, xmin)
        if alpha is None or not (1.1 < alpha < 5.0):
            continue
        ks = ks_distance_powerlaw(x, xmin, alpha)
        n_tail = sum(1 for z in x if z >= xmin)
        if ks < best["ks"]:
            best = {"alpha": alpha, "xmin": xmin, "ks": ks, "n_tail": n_tail}
    if best["xmin"] is None:
        return {"alpha": None, "xmin": None, "ks": None, "n_tail": 0}
    return best

def sample_powerlaw_continuous(n, xmin, alpha, rng=random):
    """Inverse-CDF sampling for continuous power law x >= xmin."""
    u = np.array([rng.random() for _ in range(n)])
    return xmin * (1 - u) ** (1.0 / (1.0 - alpha))

def gof_bootstrap_powerlaw(x, xmin, alpha, B=200, rng=random):
    """Semi-parametric bootstrap p-value as in Clauset-Shalizi-Newman (approx.)."""
    x = [v for v in x if v >= xmin]
    n = len(x)
    if n < 20:
        return None
    D_obs = ks_distance_powerlaw(x, xmin, alpha)
    greater = 0
    for _ in range(B):
        xb = sample_powerlaw_continuous(n, xmin, alpha, rng=rng)
        xb = [int(max(xmin, math.floor(z))) for z in xb]
        fitb = fit_powerlaw_csn(xb, xmin_candidates=[xmin])
        if fitb["alpha"] is None:
            continue
        Db = ks_distance_powerlaw(xb, xmin, fitb["alpha"])
        if Db >= D_obs - 1e-12:
            greater += 1
    return greater / B

# ------------------- Parsing GraphOfLife logs -------------------

def parse_step(path: str) -> Dict[str, Any]:
    blob = load_json(path)
    phase = blob.get("phase", "unknown")
    pre = blob.get("pre_state", {})
    post = blob.get("post_state", {})
    return {
        "phase": phase,
        "pre_nodes": pre.get("nodes", []),
        "pre_edges": pre.get("edges", []),
        "post_nodes": post.get("nodes", []),
        "post_edges": post.get("edges", []),
        "allocations": blob.get("allocations", []),
        "genotype_events": blob.get("genotype_events", []),
        "cleanup": blob.get("cleanup", {}),
        "pruned_edges": blob.get("pruned_edges", []),
        "decisions": blob.get("decisions", []),
        "winners": blob.get("winners", {}),
        "incoming_offers": blob.get("incoming_offers", {}),
    }

def distinct_brains(nodes: List[Dict[str, Any]]) -> int:
    return len(set(int(n.get("brain_id")) for n in nodes if n.get("brain_id") is not None))

def count_births_from_repro(decisions: List[Dict[str, Any]]) -> int:
    return sum(1 for d in decisions if d.get("child_created", False))

def deaths_from_cleanup(cleanup: Dict[str, Any]) -> int:
    zero = cleanup.get("removed_zero_nodes", [])
    comps = cleanup.get("removed_components", [])
    return len(zero) + sum(len(c) for c in comps)

# ------------------- Time-Series Extraction -------------------

def extract_timeseries(run_dir: str) -> Dict[str, Any]:
    paths = list_step_files(run_dir)
    rows = []
    avalanches_edges = []
    avalanches_nodes = []
    for i, p in enumerate(paths):
        s = parse_step(p)
        adj_pre, nodes_pre = build_graph(s["pre_nodes"], s["pre_edges"])
        adj_post, nodes_post = build_graph(s["post_nodes"], s["post_edges"])
        lam1 = spectral_radius_power_iteration(adj_post)
        S1, S2 = largest_cc_size(adj_post)
        B_pre = distinct_brains(s["pre_nodes"])
        events = s["genotype_events"]
        parent_ids = set(int(e["from"]) for e in events if "from" in e and e.get("t") in ("copy","mut"))
        offspring_count = sum(1 for e in events if e.get("t") in ("copy","mut"))
        R_all = offspring_count / max(B_pre, 1)
        R_active = (offspring_count / max(len(parent_ids), 1)) if parent_ids else 0.0
        births = count_births_from_repro(s["decisions"]) if s["phase"] == "reproduction" else 0
        deaths = deaths_from_cleanup(s["cleanup"])
        N_pre = len(nodes_pre); N_post = len(nodes_post)
        E_pre = len(s["pre_edges"]); E_post = len(s["post_edges"])
        if s["phase"] == "blotto":
            avalanches_edges.append(len(s.get("pruned_edges", [])))
        avalanches_nodes.append(deaths)
        rows.append({
            "idx": i,
            "phase": s["phase"],
            "B_pre": B_pre,
            "events": offspring_count,
            "R_all": R_all,
            "R_active": R_active,
            "births": births,
            "deaths": deaths,
            "N_pre": N_pre,
            "N_post": N_post,
            "E_pre": E_pre,
            "E_post": E_post,
            "lambda1_post": lam1,
            "S1_post": S1,
            "S2_post": S2,
        })
    return {"rows": rows, "avalanches_edges": avalanches_edges, "avalanches_nodes": avalanches_nodes}

# ------------------- Plotting -------------------

def plot_branching_and_spectral(rows: List[Dict[str, Any]], outdir: str) -> None:
    xs = [r["idx"] for r in rows]
    R = [r["R_all"] for r in rows]
    Lam = [r["lambda1_post"] for r in rows]
    Ns = [r["N_post"] for r in rows]
    fig1 = plt.figure(figsize=(9,5))
    ax1 = fig1.add_subplot(111)
    ax1.plot(xs, R, label="R_all (genotype offspring / distinct brains)")
    ax1.axhline(1.0, linestyle="--")
    ax1.set_xlabel("step index")
    ax1.set_ylabel("branching ratio")
    ax1.set_title("Genotype branching ratio over time")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(outdir, "fig_branching_ratio.png"), dpi=160)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(9,5))
    ax2 = fig2.add_subplot(111)
    ax2.plot(xs, Lam, label="λ1 (post-state adjacency)")
    ax2.set_xlabel("step index")
    ax2.set_ylabel("spectral radius λ1")
    ax2.set_title("Spectral radius over time")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "fig_spectral_radius.png"), dpi=160)
    plt.close(fig2)

    fig3 = plt.figure(figsize=(9,5))
    ax3 = fig3.add_subplot(111)
    ax3.plot(xs, Ns, label="N_post (nodes)")
    ax3.set_xlabel("step index")
    ax3.set_ylabel("nodes")
    ax3.set_title("Node count over time")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(outdir, "fig_nodes_over_time.png"), dpi=160)
    plt.close(fig3)

def plot_avalanche_ccdf(vals: List[int], name: str, outdir: str) -> None:
    xs, ps = ccdf(vals)
    if len(xs) == 0:
        return
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    ax.loglog(xs, ps, marker="o", linestyle="none")
    ax.set_xlabel(f"{name} size")
    ax.set_ylabel("CCDF")
    ax.set_title(f"Avalanche CCDF: {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig_avalanche_ccdf_{name}.png"), dpi=160)
    plt.close(fig)

# ------------------- Main -------------------

def main():

    run_dir = "/home/stefan/Documents/PythonProjects/GraphOfLifeGithub/GraphOfLife/GraphOfLifeOutputs/run_20251124_185724_662"
    outdir = os.path.join(run_dir, "criticality_analysis")
    os.makedirs(outdir, exist_ok=True)

    data = extract_timeseries(run_dir)
    rows = data["rows"]
    aval_edges = [v for v in data["avalanches_edges"] if v > 0]
    aval_nodes = [v for v in data["avalanches_nodes"] if v > 0]

    # Save time series CSV
    import csv
    csv_path = os.path.join(outdir, "timeseries.csv")
    fields = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plots
    plot_branching_and_spectral(rows, outdir)
    plot_avalanche_ccdf(aval_edges, "edge_pruning", outdir)
    plot_avalanche_ccdf(aval_nodes, "node_death", outdir)

    # Power-law fits (CSN-style) for avalanches
    results = {}
    for name, vals in [("edge_pruning", aval_edges), ("node_death", aval_nodes)]:
        fit = fit_powerlaw_csn(vals)
        if fit["xmin"] is not None:
            pval = gof_bootstrap_powerlaw(vals, fit["xmin"], fit["alpha"], B=200)
        else:
            pval = None
        results[name] = {"fit": fit, "gof_p": pval, "n": len(vals)}
    with open(os.path.join(outdir, "powerlaw_fits.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Quick textual summary
    def summarize(rows):
        R = [r["R_all"] for r in rows if r["R_all"] is not None]
        Lam = [r["lambda1_post"] for r in rows if r["lambda1_post"] is not None]
        return {
            "R_all_mean": float(np.mean(R)) if R else None,
            "R_all_median": float(np.median(R)) if R else None,
            "lambda1_mean": float(np.mean(Lam)) if Lam else None,
            "lambda1_median": float(np.median(Lam)) if Lam else None,
            "steps": len(rows)
        }
    summary = {
        "branching_and_spectral": summarize(rows),
        "powerlaw": results
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:")
    print(" -", csv_path)
    print(" -", os.path.join(outdir, "fig_branching_ratio.png"))
    print(" -", os.path.join(outdir, "fig_spectral_radius.png"))
    print(" -", os.path.join(outdir, "fig_nodes_over_time.png"))
    print(" -", os.path.join(outdir, "fig_avalanche_ccdf_edge_pruning.png"))
    print(" -", os.path.join(outdir, "fig_avalanche_ccdf_node_death.png"))
    print(" -", os.path.join(outdir, "powerlaw_fits.json"))
    print(" -", os.path.join(outdir, "summary.json"))

if __name__ == "__main__":
    main()
