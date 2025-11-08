
from __future__ import annotations
import os, json, math, glob, re, gzip
from typing import Dict, List, Tuple, Any, Callable, Optional, Iterable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
from types import SimpleNamespace

from matplotlib import colormaps

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

CENTRALITY_FAST = True
BETWEENNESS_SAMPLE_MAX_K = 24          # cap #sources for sampled betweenness
BETWEENNESS_SAMPLE_FRAC = 0.005        # 0.5% of nodes -> k = min(MAX_K, n * FRAC)
CLOSENESS_SAMPLE_N = 24                # BFS seeds for approximate max closeness
SKIP_CLOSENESS_ABOVE_N = 5000          # skip closeness entirely if n > this

# ---- Distance / dimension speed tunables ----
USE_LCC_FOR_DISTANCE = True        # work on largest component view (no copy)
DIAM_SWEEPS = 4                    # # of double-sweep rounds (each = 2 BFS)
ASP_SAMPLE = 12                    # BFS seeds for avg shortest-path estimate
ASP_SAMPLE_FRAC = 0.002            # or 0.2% of nodes (uses min(ASP_SAMPLE, n*frac))

WOLFRAM_SAMPLE = 16                # # nodes to sample
WOLFRAM_R_MAX = 7                 # max radius (compute BFS once with cutoff=R, reuse)

PHASE_MAP = {"reproduction": 1, "blotto": 2}

class LogCache:
    def __init__(self):
        self._ln = {0: -np.inf, 1: 0.0}
    def ln(self, n: int) -> float:
        if n in self._ln:
            return self._ln[n]
        v = math.log(n)
        self._ln[n] = v
        return v

log_cache = LogCache()

def shannon_entropy_from_counts(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    s = 0.0
    for c in counts.values():
        if c <= 0: 
            continue
        s += c * log_cache.ln(c)
    return math.log(total) - (s / total)

@dataclass
class Record:
    t: int
    phase_name: str
    phase: int
    path: str
    data: Dict[str, Any]

class RunReader:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        self._paths = sorted(self.run_dir.glob("step_*.json"))
        if not self._paths:
            raise FileNotFoundError(f"No step_*.json files in {run_dir}")
    def __len__(self):
        return len(self._paths)
    def iter_paths(self):
        for path in self._paths:
            name = path.name
            m = re.search(r"step_(\d+)\.json$", name)
            idx = int(m.group(1)) if m else 0
            t = idx // 2
            phase = 1 if (idx % 2 == 0) else 2
            phase_name = "reproduction" if phase == 1 else "blotto"
            yield str(path), t, phase_name, phase
    def iter_records(self) -> Iterable[Record]:
        for p in self._paths:
            with open(p, "r") as f:
                data = json.load(f)
            phase_name = data.get("phase")
            if phase_name not in PHASE_MAP:
                m = re.search(r"step_(\d+)\.json$", p.name)
                idx = int(m.group(1)) if m else 0
                phase = 1 if (idx % 2 == 0) else 2
                phase_name = "reproduction" if phase == 1 else "blotto"
            phase = PHASE_MAP[phase_name]
            m = re.search(r"step_(\d+)\.json$", p.name)
            idx = int(m.group(1)) if m else 0
            t = idx // 2
            yield Record(t=t, phase_name=phase_name, phase=phase, path=str(p), data=data)

MetricFn = Callable[[Record, nx.Graph], Dict[str, Any]]

class MetricRegistry:
    def __init__(self):
        self._metrics: Dict[str, MetricFn] = {}
    def metric(self, name: str):
        def deco(fn: MetricFn):
            self._metrics[name] = fn
            return fn
        return deco
    def items(self):
        return list(self._metrics.items())

def graph_from_post_state(data: Dict[str, Any]) -> nx.Graph:
    ps = data.get("post_state", {})
    G = nx.Graph()
    for n in ps.get("nodes", []):
        aid = int(n["agent_id"])
        G.add_node(aid, tokens=int(n.get("tokens", 0)), degree=int(n.get("degree", 0)))
    for u, v in ps.get("edges", []):
        G.add_edge(int(u), int(v))
    return G

def tokens_from_post_state(data: Dict[str, Any]) -> Dict[int,int]:
    ps = data.get("post_state", {})
    out = {}
    for n in ps.get("nodes", []):
        out[int(n["agent_id"])] = int(n.get("tokens", 0))
    return out

def degrees_dict(G: nx.Graph) -> Dict[int,int]:
    return {int(n): int(d) for n, d in G.degree()}

from time import perf_counter as _perf_counter


class _MetricTimer:
    def __init__(self, name: str, rec):
        self.name = name
        self.rec = rec
        self.t0 = None
    def __enter__(self):
        self.t0 = _perf_counter()
    def __exit__(self, exc_type, exc, tb):
        dt = (_perf_counter() - self.t0) if self.t0 is not None else 0.0
        t = getattr(self.rec, "t", "?")
        phase = getattr(self.rec, "phase", "?")
        print(f"[metric] {self.name} t={t} phase={phase} took {dt*1000:.2f} ms")

def metric_timer(name: str, rec):
    return _MetricTimer(name, rec)

def _basic_counts(rec: Record, G: nx.Graph) -> Dict[str, Any]:
    with metric_timer("basic_counts", rec):
        n = G.number_of_nodes()
        m = G.number_of_edges()
        return {"nodes": n, "links": m, "links_per_node": (m / n) if n > 0 else 0.0}

def _entropy_metrics(rec: Record, G: nx.Graph) -> Dict[str, Any]:
    with metric_timer("entropy", rec):
        degs = degrees_dict(G)
        tokens = tokens_from_post_state(rec.data)
        deg_counts = Counter(degs.values())
        H_deg = shannon_entropy_from_counts(deg_counts)
        aug = {u: degs.get(u,0) + max(0, tokens.get(u,0)-1) for u in set(degs) | set(tokens)}
        aug_counts = Counter(aug.values())
        H_aug = shannon_entropy_from_counts(aug_counts)
        return {"entropy_degree": H_deg, "entropy_degree_aug_tokens": H_aug}

def _diameter_and_path_metrics(rec: Record, G: nx.Graph) -> Dict[str, Any]:
    with metric_timer("diameter_path", rec):
        out = {"diameter_est": 0, "avg_shortest_path_est": 0.0}
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return out

        # Work on LCC view if requested
        H = G
        if USE_LCC_FOR_DISTANCE and not nx.is_connected(G):
            # view, not copy (faster/lower RAM)
            largest = max(nx.connected_components(G), key=len)
            H = G.subgraph(largest)

        n = H.number_of_nodes()
        if n == 0:
            return out

        # --- Diameter via a few double-sweeps (2 BFS per sweep) ---
        rng = np.random.default_rng(101 + rec.t + rec.phase)
        nodes = list(H.nodes())
        diam_est = 0
        sweeps = max(1, DIAM_SWEEPS)
        for _ in range(sweeps):
            s = int(rng.choice(nodes))
            # 1st BFS
            d1 = nx.single_source_shortest_path_length(H, s)
            v = max(d1, key=d1.get)
            # 2nd BFS from farthest
            d2 = nx.single_source_shortest_path_length(H, v)
            w = max(d2, key=d2.get)
            ecc_v = d2[w]
            if ecc_v > diam_est:
                diam_est = ecc_v

        # --- Avg shortest path via sampled BFS ---
        k = min(ASP_SAMPLE, max(1, int(n * ASP_SAMPLE_FRAC)))
        if k <= 0:
            k = 1
        seeds = rng.choice(nodes, size=min(k, len(nodes)), replace=False) if len(nodes) > k else nodes
        asp_vals = []
        for s in seeds:
            dist = nx.single_source_shortest_path_length(H, int(s))
            # mean finite distance among reachable nodes (exclude 0-distance self)
            if len(dist) > 1:
                denom = len(dist) - 1
                asp_vals.append((sum(dist.values()) / denom))
        asp_est = float(np.mean(asp_vals)) if asp_vals else 0.0

        out["diameter_est"] = int(diam_est)
        out["avg_shortest_path_est"] = asp_est
        return out

def _centrality_metrics(rec: Record, G: nx.Graph) -> Dict[str, Any]:
    with metric_timer("centrality", rec):
        n = G.number_of_nodes()
        out = {"degree_centrality_max": 0.0, "betweenness_max_est": 0.0, "closeness_max": 0.0}
        if n == 0:
            return out

        # --- Degree centrality (fast): no full map, just max degree ---
        try:
            max_deg = max(d for _, d in G.degree())
        except ValueError:
            max_deg = 0
        out["degree_centrality_max"] = (max_deg / (n - 1)) if n > 1 else 0.0

        # --- Betweenness (approx): small k sample scales sublinearly ---
        try:
            if CENTRALITY_FAST:
                k = max(1, int(n * BETWEENNESS_SAMPLE_FRAC))
                k = min(BETWEENNESS_SAMPLE_MAX_K, k)
            else:
                # original heuristic (slower)
                k = min(64, max(1, n // 20))
            btw = nx.betweenness_centrality(G, k=k, seed=rec.t * 3 + rec.phase)
            out["betweenness_max_est"] = float(max(btw.values()) if btw else 0.0)
        except Exception:
            pass

        # --- Closeness (approx): BFS from a few seeds; or skip on very large graphs ---
        try:
            if CENTRALITY_FAST and n > SKIP_CLOSENESS_ABOVE_N:
                out["closeness_max"] = 0.0   # explicitly skip
            else:
                # use LCC to avoid tiny components biasing the max
                if not nx.is_connected(G) and n > 0:
                    H = max((G.subgraph(c).copy() for c in nx.connected_components(G)), key=lambda g: g.number_of_nodes())
                else:
                    H = G
                nodes = list(H.nodes())
                rng = np.random.default_rng(2025 + rec.t + rec.phase)
                s = min(CLOSENESS_SAMPLE_N if CENTRALITY_FAST else max(16, len(nodes)), len(nodes))
                seeds = rng.choice(nodes, size=s, replace=False) if len(nodes) > s else nodes

                best = 0.0
                for u in seeds:
                    lengths = nx.single_source_shortest_path_length(H, u)  # one BFS
                    reach = len(lengths)
                    denom = sum(lengths.values())
                    # standard closeness for connected graphs; harmonic variant optional
                    c = ((reach - 1) / denom) if (denom > 0 and reach > 1) else 0.0
                    if c > best:
                        best = c
                out["closeness_max"] = float(best)
        except Exception:
            pass

        return out

def _wolfram_dimensionality(rec: Record, G: nx.Graph) -> Dict[str, Any]:
    with metric_timer("wolfram_dim", rec):
        n = G.number_of_nodes()
        if n == 0:
            return {"dim_slope": 0.0}

        # LCC view to avoid tiny comps skewing volumes
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            H = G.subgraph(largest)
        else:
            H = G

        nodes = list(H.nodes())
        if not nodes:
            return {"dim_slope": 0.0}

        rng = np.random.default_rng(1234 + rec.t + rec.phase)
        S = min(WOLFRAM_SAMPLE, len(nodes))
        seeds = rng.choice(nodes, size=S, replace=False) if len(nodes) > S else nodes

        R = int(max(1, WOLFRAM_R_MAX))
        r_vals = list(range(1, R + 1))
        # accumulate ball sizes across seeds; compute via single BFS per seed
        ball_sums = np.zeros(R, dtype=float)

        for u in seeds:
            lengths = nx.single_source_shortest_path_length(H, int(u), cutoff=R)
            # histogram distances 0..R
            dist_count = [0] * (R + 1)
            for d in lengths.values():
                if 0 <= d <= R:
                    dist_count[d] += 1
            # cumulative counts for radii 1..R (include center)
            cum = 0
            for r in range(0, R + 1):
                cum += dist_count[r]
                if r >= 1:
                    ball_sums[r - 1] += cum

        vols = (ball_sums / S) if S > 0 else np.zeros(R, dtype=float)

        # Fit slope log(vol) vs log(r) ignoring zeros
        xs, ys = [], []
        for r, v in zip(r_vals, vols):
            if r > 0 and v > 0:
                xs.append(math.log(r))
                ys.append(math.log(v))
        if len(xs) >= 2:
            A = np.vstack([xs, np.ones(len(xs))]).T
            slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
            dim = float(slope)
        else:
            dim = 0.0

        out = {"dim_slope": dim}
        for r, v in zip(r_vals, vols):
            out[f"ball_avg_r{r}"] = float(v)
        return out
def _blotto_self_alloc(rec: Record, G: nx.Graph) -> Dict[str, Any]:
    with metric_timer("blotto_self_alloc", rec):
        if rec.phase_name != "blotto":
            return {}
        allocs = rec.data.get("allocations", []) or []
        fracs = []
        for a in allocs:
            t_before = int(a.get("tokens_before", 0))
            if t_before <= 0:
                continue
            targets = a.get("targets", []) or []
            alloc = a.get("alloc", []) or []
            if not targets or not alloc or len(targets) != len(alloc):
                continue
            agent = int(a.get("agent_id"))
            try:
                idx = targets.index(agent)
            except ValueError:
                continue
            self_alloc = int(alloc[idx])
            fracs.append(self_alloc / t_before if t_before>0 else 0.0)
        avg_frac = float(np.mean(fracs)) if fracs else 0.0
        return {"blotto_self_alloc_avg": avg_frac}

def _distributions_and_extremes(rec: Record, G: nx.Graph) -> Dict[str, Any]:
    with metric_timer("distributions_extremes", rec):
        toks = tokens_from_post_state(rec.data)
        degs = degrees_dict(G)
        tok_counts = Counter(toks.values())
        deg_counts = Counter(degs.values())
        n_toks = len(toks)
        n_degree = len(degs)
        max_token = max(toks.values()) if toks else 0
        avg_token = sum(toks.values())/n_toks if toks else 0
        max_degree = max(degs.values()) if degs else 0
        avg_degree = sum(degs.values())/n_degree if degs else 0

        pre = rec.data.get("pre_state", {})
        pre_nodes = {int(n["agent_id"]) for n in pre.get("nodes", [])}
        pre_edges = {tuple(sorted((int(u), int(v)))) for u, v in pre.get("edges", [])}
        post_nodes = set(G.nodes())
        post_edges = {tuple(sorted(e)) for e in G.edges()}
        removed_nodes = len(pre_nodes - post_nodes)
        added_nodes = len(post_nodes - pre_nodes)
        removed_edges = len(pre_edges - post_edges)
        added_edges = len(post_edges - pre_edges)
        pruned_edges = rec.data.get("pruned_edges", [])
        pruned_edges_n = len(pruned_edges) if pruned_edges else 0
        return {
            "max_tokens": int(max_token),
            "avg_tokens": int(avg_token),
            "max_degree": int(max_degree),
            "avg_degree": int(avg_degree),
            "removed_nodes": int(removed_nodes),
            "added_nodes": int(added_nodes),
            "removed_edges": int(removed_edges),
            "added_edges": int(added_edges),
            "pruned_edges": int(pruned_edges_n),
        }

def _edge_flow_histogram(rec: Record, G: nx.Graph) -> Dict[str, Any]:
    with metric_timer("edge_flow_hist", rec):
        if rec.phase_name != "blotto":
            return {}
        edge_flow = Counter()
        for a in rec.data.get("allocations", []) or []:
            u = int(a.get("agent_id"))
            targets = a.get("targets", []) or []
            alloc = a.get("alloc", []) or []
            if not targets or not alloc:
                continue
            for v, w in zip(targets, alloc):
                v = int(v); w = int(w)
                if v == u or w <= 0:
                    continue
                e = tuple(sorted((u, v)))
                edge_flow[e] += w
        flow_counts = Counter(edge_flow.values())
        return {"edge_flow_hist": dict(flow_counts)}

# ---- NEW METRICS (cleanup, reproduction decisions, blotto competition) ----

def _cleanup_pruning_metrics(rec: Record, G: nx.Graph) -> Dict[str, Any]:
    pre = rec.data.get("pre_state", {}) or {}
    pre_nodes_n = len(pre.get("nodes", []))
    pre_edges_n = len(pre.get("edges", []))

    cl = rec.data.get("cleanup", {}) or {}
    redist = int(cl.get("redistributed_tokens", 0))
    removed_zero = cl.get("removed_zero_nodes", []) or []
    removed_comps = cl.get("removed_components", []) or []
    removed_comp_nodes = sum(len(c) for c in removed_comps)

    pruned_edges = rec.data.get("pruned_edges", []) or []
    pruned_edges_n = len(pruned_edges)

    out = {
        "redistributed_tokens": redist,
        "removed_zero_nodes_count": int(len(removed_zero)),
        "removed_components_nodes_count": int(removed_comp_nodes),
        "removed_components_count": int(len(removed_comps)),
        "pruned_edges_count": int(pruned_edges_n),
        "pct_removed_zero_nodes": (len(removed_zero) / pre_nodes_n) if pre_nodes_n else 0.0,
        "pct_removed_components_nodes": (removed_comp_nodes / pre_nodes_n) if pre_nodes_n else 0.0,
        "pct_pruned_edges": (pruned_edges_n / pre_edges_n) if pre_edges_n else 0.0,
    }
    return out

def _reproduction_decision_metrics(rec: Record, G: nx.Graph) -> Dict[str, Any]:
    if rec.phase_name != "reproduction":
        return {}
    pre = rec.data.get("pre_state", {}) or {}
    pre_nodes_n = len(pre.get("nodes", []))
    total_tokens_pre = sum(int(n.get("tokens", 0)) for n in pre.get("nodes", []))

    decisions = rec.data.get("decisions", []) or []
    reproduced = 0
    walker_new = 0
    repro_tokens_sum = 0
    for d in decisions:
        if d.get("child_created"):
            reproduced += 1
        wc = d.get("walker_choice", {}) or {}
        if wc.get("created_new_link"):
            walker_new += 1
        repro_tokens_sum += int(d.get("repro_tokens", 0))

    return {
        "repro_frac_nodes": (reproduced / pre_nodes_n) if pre_nodes_n else 0.0,
        "walker_newlink_frac_nodes": (walker_new / pre_nodes_n) if pre_nodes_n else 0.0,
        "repro_tokens_frac": (repro_tokens_sum / total_tokens_pre) if total_tokens_pre else 0.0,
        "repro_tokens_sum": int(repro_tokens_sum),
        "reproduced_count": int(reproduced),
        "walker_newlink_count": int(walker_new),
    }

def _blotto_competition_metrics(rec: Record, G: nx.Graph) -> Dict[str, Any]:
    if rec.phase_name != "blotto":
        return {}
    pre = rec.data.get("pre_state", {}) or {}
    pre_nodes_n = len(pre.get("nodes", []))
    pre_edges = {tuple(sorted((int(u), int(v)))) for u, v in pre.get("edges", [])}
    pre_edges_n = len(pre_edges)

    incoming = rec.data.get("incoming_offers", {}) or {}
    winners = rec.data.get("winners", {}) or {}

    needed_fracs = []
    self_win_count = 0

    pos_edges = set()
    for a in rec.data.get("allocations", []) or []:
        u = int(a.get("agent_id"))
        targets = a.get("targets", []) or []
        alloc = a.get("alloc", []) or []
        for v, w in zip(targets, alloc):
            v = int(v); w = int(w)
            if v != u and w > 0:
                pos_edges.add(tuple(sorted((u, v))))

    for v_key, offers in incoming.items():
        try:
            v_int = int(v_key)
        except Exception:
            v_int = v_key
        total = sum(int(a) for _, a in offers)
        if total > 0:
            winfo = winners.get(str(v_key)) or winners.get(str(v_int)) or {}
            max_amt = int(winfo.get("max_amount", 0))
            needed_fracs.append(max_amt / total if total > 0 else 0.0)
            winner_id = winfo.get("winner", None)
            if winner_id is not None and int(winner_id) == v_int:
                self_win_count += 1

    inactive_edges_n = max(0, pre_edges_n - len(pre_edges & pos_edges))
    return {
        "blotto_needed_to_win_avg": (float(np.mean(needed_fracs)) if needed_fracs else 0.0),
        "inactive_edges_frac": (inactive_edges_n / pre_edges_n) if pre_edges_n else 0.0,
        "self_win_frac": (self_win_count / pre_nodes_n) if pre_nodes_n else 0.0,
        "self_win_count": int(self_win_count),
        "inactive_edges_count": int(inactive_edges_n),
    }

class Analyzer:
    def __init__(self, reader: RunReader, registry: MetricRegistry, out_dir: str = "./analysis_out"):
        self.reader = reader
        self.registry = registry
        self.out_dir = out_dir
        self.records_cache: List[Record] = []
    def _ensure_out(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
    def compute_all(self, stream: bool = True, max_iter: int | None = None, progress: bool = True) -> pd.DataFrame:
        # Use tqdm if available
        try:
            from tqdm import tqdm as _tqdm
        except Exception:
            _tqdm = None
        self._ensure_out()
        rows = []
        if stream:
            def _infer_t_from_name(name: str) -> int:
                m = re.search(r"step_(\d+)\.json$", name)
                idx = int(m.group(1)) if m else 0
                return idx // 2
            paths = []
            for path in self.reader._paths:
                t = _infer_t_from_name(path.name)
                if max_iter is not None and t >= max_iter:
                    continue
                paths.append(path)
            iterator = (
                (str(path),
                 _infer_t_from_name(path.name),
                 ("reproduction" if (int(re.search(r"step_(\d+)\.json$", path.name).group(1)) % 2 == 0) else "blotto"),
                 (1 if (int(re.search(r"step_(\d+)\.json$", path.name).group(1)) % 2 == 0) else 2))
                for path in paths
            )
            if progress and _tqdm is not None:
                iterator = _tqdm(iterator, total=len(paths), desc="Analyzing steps", unit="step")
            for path, t, phase_name_guess, phase_guess in iterator:
                if max_iter is not None and t >= max_iter:
                    break
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                except Exception as e:
                    rows.append({"t": t, "phase": phase_guess, "phase_name": phase_name_guess, "path": path, "load_error": str(e)})
                    continue
                phase_name = data.get("phase", phase_name_guess)
                phase = 1 if phase_name == "reproduction" else 2
                rec = Record(t=t, phase_name=phase_name, phase=phase, path=path, data=data)
                G = graph_from_post_state(rec.data)
                row = {"t": rec.t, "phase": rec.phase, "phase_name": rec.phase_name, "path": rec.path}
                for name, fn in self.registry.items():
                    try:
                        vals = fn(rec, G)
                        if isinstance(vals, dict):
                            for k, v in vals.items():
                                if isinstance(v, dict):
                                    row[f"{name}.{k}"] = json.dumps(v, separators=(",", ":"))
                                else:
                                    row[f"{name}.{k}"] = v
                    except Exception as e:
                        row[f"{name}.__error"] = str(e)
                rows.append(row)
                del data, rec, G
        else:
            if not self.records_cache:
                self.records_cache = list(self.reader.iter_records())
            it = self.records_cache
            if progress and _tqdm is not None:
                it = _tqdm(it, total=len(self.records_cache), desc="Analyzing steps", unit="step")
            for rec in it:
                if max_iter is not None and rec.t >= max_iter:
                    continue
                G = graph_from_post_state(rec.data)
                row = {"t": rec.t, "phase": rec.phase, "phase_name": rec.phase_name, "path": rec.path}
                for name, fn in self.registry.items():
                    try:
                        vals = fn(rec, G)
                        if isinstance(vals, dict):
                            for k, v in vals.items():
                                if isinstance(v, dict):
                                    row[f"{name}.{k}"] = json.dumps(v, separators=(",", ":"))
                                else:
                                    row[f"{name}.{k}"] = v
                    except Exception as e:
                        row[f"{name}.__error"] = str(e)
                rows.append(row)
        df = pd.DataFrame(rows).sort_values(["t", "phase"]).reset_index(drop=True)
        csv_path = Path(self.out_dir) / "metrics_per_step.csv"
        df.to_csv(csv_path, index=False)
        return df
    def _savefig(self, name: str):
        self._ensure_out()
        out = Path(self.out_dir) / f"{name}.png"
        plt.savefig(out, dpi=140, bbox_inches="tight")
        plt.close()
        return out
    def plot_timeseries(self, df: pd.DataFrame, y: str, title: str):
        plt.figure(figsize=(9,4.5))
        for phase in [1,2]:
            sub = df[df["phase"]==phase]
            if sub.empty or y not in sub.columns:
                continue
            plt.plot(sub["t"].values[20:], sub[y].values[20:], label=f"phase {phase}")
        plt.xlabel("iteration t"); plt.ylabel(y); plt.title(title); plt.legend()
        return self._savefig(f"ts_{y}")


    def plot_histogram_over_time_heatmap(self, df: pd.DataFrame, col: str, title: str, vmax: Optional[float]=None):
        for phase in [1,2]:
            sub = df[df["phase"]==phase][["t", col]].dropna()
            if sub.empty:
                continue
            all_bins = set()
            parsed = []
            for _, row in sub.iterrows():
                d = json.loads(row[col]) if isinstance(row[col], str) else {}
                parsed.append((int(row["t"]), d))
                all_bins |= set(int(k) for k in d.keys())
            if not all_bins:
                continue
            bins = sorted(all_bins)
            ts = sorted(set(int(t) for t,_ in parsed))
            mat = np.zeros((len(bins), len(ts)), dtype=float)
            t_index = {t:i for i,t in enumerate(ts)}
            b_index = {b:i for i,b in enumerate(bins)}
            for t, d in parsed:
                for k, v in d.items():
                    mat[b_index[int(k)], t_index[t]] = float(v)
            plt.figure(figsize=(9,5))
            plt.imshow(mat, aspect="auto", origin="lower", vmax=vmax)
            plt.colorbar(label="count")
            plt.yticks(range(len(bins)), bins)
            plt.xticks(range(len(ts)), ts, rotation=90)
            plt.xlabel("iteration t"); plt.ylabel("bin"); plt.title(f"{title} — phase {phase}")
            self._savefig(f"heatmap_{col}_phase{phase}")


    def plot_degree_powerlaw_fit(self, df: pd.DataFrame, min_k: int = 1):
        for phase in [1,2]:
            sub = df[df["phase"]==phase][["t","distributions_extremes.hist_degree"]].dropna()
            if sub.empty:
                continue
            agg = Counter()
            for _, row in sub.iterrows():
                d = json.loads(row["distributions_extremes.hist_degree"])
                for k, v in d.items():
                    k = int(k); v = int(v)
                    if k >= min_k:
                        agg[k] += v
            if not agg:
                continue
            ks = sorted(agg.keys())
            counts = np.array([agg[k] for k in ks], dtype=float)
            p = counts / counts.sum() if counts.sum()>0 else counts
            xs = [math.log(k) for k,pi in zip(ks,p) if k>=min_k and pi>0]
            ys = [math.log(pi) for pi in p if pi>0]
            if len(xs) >= 2 and len(xs)==len(ys):
                A = np.vstack([xs, np.ones(len(xs))]).T
                slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
                yhat = A @ np.array([slope, intercept])
                ss_res = np.sum((ys - yhat)**2)
                ss_tot = np.sum((ys - np.mean(ys))**2)
                r2 = 1 - ss_res/ss_tot if ss_tot>0 else 0.0
            else:
                slope, intercept, r2 = 0.0, 0.0, 0.0
            plt.figure(figsize=(6,5))
            plt.scatter(ks, p)
            if len(xs) >= 2:
                x_line = np.linspace(min(ks), max(ks), 200)
                y_line = np.exp(intercept) * np.power(x_line, slope)
                plt.plot(x_line, y_line)
            plt.xscale("log"); plt.yscale("log")
            plt.xlabel("degree k (log)"); plt.ylabel("P(k) (log)")
            plt.title(f"Degree power-law fit — phase {phase}\nalpha≈{-slope:.3f}, R²={r2:.3f}")
            self._savefig(f"powerlaw_degree_phase{phase}")

def default_metrics(reg: MetricRegistry):
    reg.metric("basic_counts")(_basic_counts)
    reg.metric("entropy")(_entropy_metrics)
    reg.metric("diameter_path")(_diameter_and_path_metrics)
    #reg.metric("centrality")(_centrality_metrics)
    reg.metric("wolfram_dim")(_wolfram_dimensionality)
    reg.metric("blotto_self_alloc")(_blotto_self_alloc)
    reg.metric("distributions_extremes")(_distributions_and_extremes)
    reg.metric("edge_flow_hist")(_edge_flow_histogram)
    reg.metric("cleanup_pruning")(_cleanup_pruning_metrics)
    reg.metric("reproduction_decision")(_reproduction_decision_metrics)
    reg.metric("blotto_competition")(_blotto_competition_metrics)

def default_plots(an: Analyzer, df: pd.DataFrame):
    an.plot_timeseries(df, "basic_counts.nodes", "Nodes over time (phase 1 & 2)")
    an.plot_timeseries(df, "basic_counts.links", "Links over time (phase 1 & 2)")
    an.plot_timeseries(df, "basic_counts.links_per_node", "Links per node over time")
    an.plot_timeseries(df, "entropy.entropy_degree", "Shannon entropy of degree distribution")
    an.plot_timeseries(df, "entropy.entropy_degree_aug_tokens", "Entropy of augmented degree (+tokens-1)")
    an.plot_timeseries(df, "diameter_path.diameter_est", "Estimated diameter over time")
    an.plot_timeseries(df, "diameter_path.avg_shortest_path_est", "Avg shortest path length (est.)")
    an.plot_timeseries(df, "centrality.degree_centrality_max", "Max degree centrality over time")
    an.plot_timeseries(df, "centrality.betweenness_max_est", "Max betweenness (est.) over time")
    an.plot_timeseries(df, "centrality.closeness_max", "Max closeness over time")
    an.plot_timeseries(df, "wolfram_dim.dim_slope", "Approx. graph dimension (slope)")
    an.plot_timeseries(df, "distributions_extremes.max_tokens", "Max tokens per node")
    an.plot_timeseries(df, "distributions_extremes.max_degree", "Max degree per node")
    an.plot_timeseries(df, "distributions_extremes.removed_nodes", "Removed nodes per step")
    an.plot_timeseries(df, "distributions_extremes.removed_edges", "Removed edges per step")
    an.plot_timeseries(df, "distributions_extremes.added_nodes", "Added nodes per step")
    an.plot_timeseries(df, "distributions_extremes.added_edges", "Added edges per step")
    an.plot_timeseries(df, "distributions_extremes.pruned_edges", "Pruned edges (phase 2)")
    an.plot_timeseries(df, "blotto_self_alloc.blotto_self_alloc_avg", "Avg self-allocation (blotto)")
    an.plot_timeseries(df, "cleanup_pruning.redistributed_tokens", "Redistributed tokens (from smaller comps)")
    an.plot_timeseries(df, "cleanup_pruning.pct_removed_zero_nodes", "Pct zero-token nodes removed")
    an.plot_timeseries(df, "cleanup_pruning.pct_removed_components_nodes", "Pct nodes removed (small comps)")
    an.plot_timeseries(df, "cleanup_pruning.pct_pruned_edges", "Pct pruned edges (phase 2)")
    an.plot_timeseries(df, "reproduction_decision.repro_frac_nodes", "Frac nodes reproducing (phase 1)")
    an.plot_timeseries(df, "reproduction_decision.walker_newlink_frac_nodes", "Frac nodes creating walker link (phase 1)")
    an.plot_timeseries(df, "reproduction_decision.repro_tokens_frac", "Fraction of tokens spent on reproduction")
    an.plot_timeseries(df, "blotto_competition.blotto_needed_to_win_avg", "Avg fraction needed to win (blotto)")
    an.plot_timeseries(df, "blotto_competition.inactive_edges_frac", "Pct inactive edges (blotto)")
    an.plot_timeseries(df, "blotto_competition.self_win_frac", "Pct self-wins (blotto)")

# ---- Extra plotting helpers (scatter & conditional hist) ----

def _load_record_for(an: Analyzer, t: int, phase: int) -> Record | None:
    target_idx = 2 * t + (phase - 1)
    step_name = f"step_{target_idx:05d}.json"
    for path in an.reader._paths:
        if path.name == step_name:
            with open(path, "r") as f:
                data = json.load(f)
            phase_name = data.get("phase", "reproduction" if phase==1 else "blotto")
            return Record(t=t, phase_name=phase_name, phase=phase, path=str(path), data=data)
    return None

def plot_scatter_tokens_vs_degree_for_step(an: Analyzer, t: int, phase: int = 1, sample: int | None = None, title: str | None = None):
    rec = _load_record_for(an, t, phase)
    if rec is None:
        raise FileNotFoundError(f"No step file for t={t}, phase={phase}")
    G = graph_from_post_state(rec.data)
    toks = tokens_from_post_state(rec.data)
    xs, ys = [], []
    for u in G.nodes():
        xs.append(int(toks.get(u, 0)))
        ys.append(int(G.degree[u]))
    if sample is not None and len(xs) > sample:
        idx = np.random.default_rng(123).choice(len(xs), size=sample, replace=False)
        xs = [xs[i] for i in idx]; ys = [ys[i] for i in idx]
    plt.figure(figsize=(6.5,5))
    plt.scatter(xs, ys, s=12, alpha=0.7)
    plt.xlabel("tokens"); plt.ylabel("degree")
    plt.title(title or f"Tokens vs Degree — t={t}, phase={phase}")
    return Analyzer._savefig(an, f"scatter_tokens_degree_t{t}_p{phase}")

def plot_scatter_tokens_vs_degree_aggregate(an: Analyzer, t_min: int | None = None, t_max: int | None = None, phases=(1,2), sample: int | None = 20000, title: str | None = None):
    xs, ys = [], []
    for path, t, phase_name, phase in an.reader.iter_paths():
        if t_min is not None and t < t_min: 
            continue
        if t_max is not None and t > t_max: 
            continue
        if phase not in phases:
            continue
        with open(path, "r") as f:
            data = json.load(f)
        G = graph_from_post_state(data)
        toks = tokens_from_post_state(data)
        for u in G.nodes():
            xs.append(int(toks.get(u, 0)))
            ys.append(int(G.degree[u]))
    if sample is not None and len(xs) > sample:
        idx = np.random.default_rng(123).choice(len(xs), size=sample, replace=False)
        xs = [xs[i] for i in idx]; ys = [ys[i] for i in idx]
    plt.figure(figsize=(6.5,5))
    plt.scatter(xs, ys, s=8, alpha=0.5)
    plt.xlabel("tokens"); plt.ylabel("degree")
    ttl = title or "Tokens vs Degree — aggregate"
    if t_min is not None or t_max is not None:
        ttl += f" (t∈[{t_min if t_min is not None else '-'}, {t_max if t_max is not None else '-'}], phases={phases})"
    plt.title(ttl)
    return Analyzer._savefig(an, "scatter_tokens_degree_aggregate")
# ---- Helpers for color scaling ----
def _avg_from_counts(count_mat, val_sum_mat):
    import numpy as np
    with np.errstate(invalid='ignore', divide='ignore'):
        avg = np.where(count_mat > 0, val_sum_mat / count_mat, np.nan)
    return avg

def _auto_color_limits(avg, default=(0.0, 1.0)):
    import numpy as np
    if avg is None:
        return default
    finite = avg[np.isfinite(avg)]
    if finite.size == 0:
        return default
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if not (vmin < vmax):
        return default
    return (vmin, vmax)

def plot_token_degree_aggregate_ranges(
    an: Analyzer,
    token_cap: int = 100,
    degree_cap: int = 30,
    range_len: int = 30,
    skip_first: int = 20,
    num_ranges: int = 5,
    cmap: str = "jet",

):
    """
    Builds token-vs-degree aggregate scatter plots in 5 ranges:
      - area encodes how many agents fall in a (token, degree) bin (if scale_sizes=True)
      - color = weighted average of the event/value per bin
      - squares at x=-1 and y=-1 show axis-weighted averages (same color scale)
      - optional monochrome variant with black markers

    The 5 ranges are picked as: first 30 after `skip_first`, last 30, and 3 evenly spaced
    30-iteration windows between them (requires both phases present for an iteration).
    """
    import json, math
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    from collections import Counter

    # ---------- helpers: iteration windows ----------
    def _valid_iterations():
        have_p1, have_p2 = set(), set()
        for path, t, phase_name, phase in an.reader.iter_paths():
            (have_p1 if phase == 1 else have_p2).add(t)
        return sorted(have_p1 & have_p2)

    def _pick_ranges(valid_ts):
        if not valid_ts:
            return []
        start_min = max(min(valid_ts), skip_first)
        last_t = max(valid_ts)
        last_start = max(start_min, last_t - (range_len - 1))
        if last_start < start_min:
            return []
        if num_ranges <= 1:
            return [(start_min, start_min + range_len - 1)]
        starts = np.linspace(start_min, last_start, num=num_ranges)
        starts = sorted(set(int(round(s)) for s in starts))
        ranges = []
        for s in starts:
            e = s + range_len - 1
            have = [t for t in valid_ts if s <= t <= e]
            if len(have) >= max(1, range_len // 2):
                ranges.append((s, e))
        first = (start_min, start_min + range_len - 1)
        last = (last_start, last_start + range_len - 1)
        if first not in ranges: ranges.insert(0, first)
        if last not in ranges:  ranges.append(last)
        if len(ranges) > num_ranges:
            idx = np.linspace(0, len(ranges)-1, num=num_ranges).astype(int)
            ranges = [ranges[i] for i in idx]
        seen, uniq = set(), []
        for r in ranges:
            if r not in seen: uniq.append(r); seen.add(r)
        return uniq[:num_ranges]

    # ---------- binning helpers ----------
    def _clip_tok_deg(tok, deg):
        x = 0 if tok < 0 else token_cap if tok > token_cap else tok
        y = 0 if deg < 0 else degree_cap if deg > degree_cap else deg
        return x, y

    def _prestate_tokens_degrees(pre_state):
        toks, degs = {}, {}
        G = nx.Graph()
        for n in pre_state.get("nodes", []):
            u = int(n["agent_id"]); G.add_node(u)
            toks[u] = int(n.get("tokens", 0))
        for u, v in pre_state.get("edges", []):
            G.add_edge(int(u), int(v))
        for u, d in G.degree():
            degs[int(u)] = int(d)
        return toks, degs, G

    # ---------- extractors: return list[(u, x, y, value)] ----------
    def _extract_repro_metrics(step_data):
        """
        Reproduction-phase metrics (phase 1).
          • bool_reproduced
          • repro_token_frac
          • bool_new_link
          • NEW: shift_frac (shifted / degree), link_new_frac (chosen / (degree+1)), did_reconnect (bool)
        Bins (x=tokens_before, y=degree) are taken from phase-1 pre-state.
        """
        pre = step_data.get("pre_state", {}) or {}
        toks, degs, _ = _prestate_tokens_degrees(pre)

        out_repro, out_frac, out_newlink = [], [], []
        out_shift_frac, out_link_new_frac, out_did_reconnect = [], [], []

        for d in step_data.get("decisions", []) or []:
            u = int(d.get("agent_id"))
            tb = int(d.get("tokens_before", toks.get(u, 0)))
            dg = int(degs.get(u, 0))
            x, y = _clip_tok_deg(tb, dg)

            # base
            child = bool(d.get("child_created"))
            out_repro.append((u, x, y, 1.0 if child else 0.0))

            rtok = int(d.get("repro_tokens", 0))
            out_frac.append((u, x, y, (rtok / tb) if tb > 0 else 0.0))

            wc = d.get("walker_choice", {}) or {}
            out_newlink.append((u, x, y, 1.0 if wc.get("created_new_link") else 0.0))

            # NEW (1): shift fraction over neighbors (0 if no reproduction)
            shifts = d.get("shift_choices", []) or []
            if child and dg > 0:
                shifted = sum(1 for sc in shifts if sc.get("shifted"))
                out_shift_frac.append((u, x, y, float(shifted) / float(dg)))
            else:
                out_shift_frac.append((u, x, y, 0.0))

            # NEW (2): link_new_frac over core candidates (degree+1) (0 if no reproduction)
            links = d.get("link_choices", []) or []
            denom = (dg + 1)
            if child and denom > 0 and links:
                chosen = sum(1 for lc in links if lc.get("chosen"))
                out_link_new_frac.append((u, x, y, float(chosen) / float(denom)))
            else:
                out_link_new_frac.append((u, x, y, 0.0))

            # NEW (3): did reconnect? (bool)
            reconn = d.get("reconnect_choices", []) or []
            out_did_reconnect.append((u, x, y, 1.0 if (reconn and child) else 0.0))

        return {
            "reproduction.bool_reproduced": out_repro,
            "reproduction.repro_token_frac": out_frac,
            "reproduction.bool_new_link": out_newlink,

            # NEW:
            "reproduction.shift_frac": out_shift_frac,
            "reproduction.link_new_frac": out_link_new_frac,
            "reproduction.bool_did_reconnect": out_did_reconnect,
        }

    def _extract_blotto_metrics(step_data):
        """
        Returns per-node lists of (u, x_tokens, y_degree, value) for several blotto metrics.
        Notes:
          • x,y bins are always taken from *pre-state* (before phase 2).
          • token_gain_factor uses post-state tokens (0 if node absent after p2).
          • self/others allocation fractions use the agent's own allocation row.
          • largest_incoming_offer_factor, win_share are per target node.
        """
        pre = step_data.get("pre_state", {}) or {}
        toks, degs, Gpre = _prestate_tokens_degrees(pre)

        # --- post-state tokens (after p2) for token_gain_factor
        post = step_data.get("post_state", {}) or {}
        post_tokens = {}
        for n in post.get("nodes", []) or []:
            post_tokens[int(n["agent_id"])] = int(n.get("tokens", 0))

        # winners + incoming offers
        winners = step_data.get("winners", {}) or {}
        incoming = step_data.get("incoming_offers", {}) or {}
        win = {}
        for k, v in winners.items():
            kk = int(k) if str(k).isdigit() else k
            ww = v.get("winner", None)
            win[kk] = int(ww) if ww is not None else None

        # who won anywhere (per agent)
        won_anywhere = Counter()
        for target, w in win.items():
            if w is not None:
                won_anywhere[int(w)] += 1

        # ---- outputs (old + new) ----
        out_self = []
        out_any = []
        out_win_ratio_ego = []

        out_token_gain_factor = []  # NEW (1)
        out_self_alloc_frac = []  # NEW (2)
        out_incoming_max_over_tokens = []  # NEW (4)
        out_win_share = []  # NEW (5)

        # ------------- per AGENT (source) metrics -------------
        # (self/others allocation fractions)
        for a in step_data.get("allocations", []) or []:
            u = int(a.get("agent_id"))
            tb = int(a.get("tokens_before", 0))
            if tb <= 0:  # nothing allocated
                continue
            targets = a.get("targets", []) or []
            alloc = a.get("alloc", []) or []
            if not targets or not alloc or len(targets) != len(alloc):
                continue
            try:
                idx_self = targets.index(u)
            except ValueError:
                idx_self = None

            self_alloc = int(alloc[idx_self]) if idx_self is not None else 0
            frac_self = (self_alloc / tb) if tb > 0 else 0.0
            frac_others = 1.0 - frac_self if tb > 0 else 0.0

            # bin from pre-state of the AGENT
            x, y = _clip_tok_deg(int(toks.get(u, 0)), int(degs.get(u, 0)))
            out_self_alloc_frac.append((u, x, y, float(frac_self)))

        # ------------- per TARGET (receiver) metrics -------------
        for v in toks.keys():
            tb_v = int(toks.get(v, 0))
            dg_v = int(degs.get(v, 0))
            x, y = _clip_tok_deg(tb_v, dg_v)

            # (1) token_gain_factor = tokens_after / tokens_before_for_that_node
            ta_v = int(post_tokens.get(v, 0))
            tgf = (ta_v / tb_v) if tb_v > 0 else 0.0
            out_token_gain_factor.append((v, x, y, float(tgf)))

            # (4) largest_incoming_offer_factor
            offers = incoming.get(str(v), []) or incoming.get(int(v), []) or []
            if offers:
                max_offer = max(int(a) for (_src, a) in offers)
                lif = (max_offer / tb_v) if tb_v > 0 else 0.0
                # (5) winner share among incoming
                tot_in = sum(int(a) for (_src, a) in offers)
                win_amt = max_offer  # consistent with our 'max' selection & log
                wshare = (win_amt / tot_in) if tot_in > 0 else 0.0
            else:
                lif = 0.0
                wshare = 0.0
            out_incoming_max_over_tokens.append((v, x, y, float(lif)))
            out_win_share.append((v, x, y, float(wshare)))

            # (existing) self win / win anywhere / wins_ratio_ego (keep as before)
            w = win.get(v, None)
            val_self = 1.0 if (w is not None and int(w) == int(v)) else 0.0
            out_self.append((v, x, y, val_self))
            val_any = 1.0 if won_anywhere.get(v, 0) > 0 else 0.0
            out_any.append((v, x, y, val_any))

        # ego wins ratio (same as your previous return)
        out_win_ratio_ego = []
        for u in toks.keys():
            dg = int(degs.get(u, 0))
            ego_nodes = [u] + [int(v) for v in Gpre.neighbors(u)]
            cnt_ego = sum(1 for tnode in ego_nodes if win.get(tnode, None) == u)
            denom = 1 + dg
            ratio = (cnt_ego / denom) if denom > 0 else 0.0
            x, y = _clip_tok_deg(int(toks.get(u, 0)), dg)
            out_win_ratio_ego.append((u, x, y, float(ratio)))

        return {
            "blotto.bool_self_win": out_self,
            "blotto.bool_win_anywhere": out_any,
            "blotto.wins_ratio_ego": out_win_ratio_ego,

            # NEW:
            "blotto.token_gain_factor": out_token_gain_factor,
            "blotto.self_alloc_frac": out_self_alloc_frac,
            "blotto.incoming_max_over_tokens": out_incoming_max_over_tokens,
            "blotto.win_share": out_win_share,
        }

    def _extract_death_metrics(step_data):
        pre = step_data.get("pre_state", {}) or {}
        toks, degs, _ = _prestate_tokens_degrees(pre)
        cl = step_data.get("cleanup", {}) or {}
        removed_zero = set(int(u) for u in (cl.get("removed_zero_nodes", []) or []))
        removed_comps = set()
        for comp in (cl.get("removed_components", []) or []):
            removed_comps.update(int(u) for u in comp)
        out_zero, out_comp, out_any = [], [], []
        for u in toks.keys():
            tb = int(toks.get(u, 0)); dg = int(degs.get(u, 0))
            x, y = _clip_tok_deg(tb, dg)
            z = 1.0 if u in removed_zero else 0.0
            c = 1.0 if u in removed_comps else 0.0
            out_zero.append((u, x, y, z))
            out_comp.append((u, x, y, c))
            out_any.append((u, x, y, 1.0 if (z > 0.0 or c > 0.0) else 0.0))
        return {"died.zero": out_zero, "died.component": out_comp, "died.any": out_any}

    def _extract_survival_metrics(t, data_p1, data_p2, path_map):
        """
        Survival booleans binned by the *starting* snapshot's tokens/degree:
          • survival.p1pre_to_p2post   : node in P1 pre_state(t) and also in P2 post_state(t)
          • survival.p1post_to_next_p1post : node in P1 post_state(t) and also in P1 post_state(t+1)
        """
        out_pre_to_p2post = []
        out_post_to_nextpost = []

        # --- helpers to get node sets and bins from a given state's nodes/edges ---
        def _toks_degs_from_state(state):
            toks, degs, G = _prestate_tokens_degrees(state)
            return toks, degs

        # current iteration
        p1_pre = data_p1.get("pre_state", {}) or {}
        p1_post = data_p1.get("post_state", {}) or {}
        p2_post = data_p2.get("post_state", {}) or {}

        toks_pre, degs_pre = _toks_degs_from_state(p1_pre)
        toks_post, degs_post = _toks_degs_from_state(p1_post)

        nodes_pre = set(int(n["agent_id"]) for n in p1_pre.get("nodes", []) or [])
        nodes_p2post = set(int(n["agent_id"]) for n in p2_post.get("nodes", []) or [])
        nodes_post = set(int(n["agent_id"]) for n in p1_post.get("nodes", []) or [])

        # (A) P1-pre → P2-post
        for u in nodes_pre:
            x, y = _clip_tok_deg(int(toks_pre.get(u, 0)), int(degs_pre.get(u, 0)))
            survived = 1.0 if u in nodes_p2post else 0.0
            out_pre_to_p2post.append((u, x, y, survived))

        # (B) P1-post(t) → P1-post(t+1)
        next_p1_path = path_map.get((t + 1, 1))
        if next_p1_path:
            try:
                with open(next_p1_path, "r") as f:
                    data_next_p1 = json.load(f)
            except Exception:
                data_next_p1 = None
            if data_next_p1 is not None:
                next_post = data_next_p1.get("post_state", {}) or {}
                nodes_next_post = set(int(n["agent_id"]) for n in next_post.get("nodes", []) or [])
                for u in nodes_post:
                    x, y = _clip_tok_deg(int(toks_post.get(u, 0)), int(degs_post.get(u, 0)))
                    survived = 1.0 if u in nodes_next_post else 0.0
                    out_post_to_nextpost.append((u, x, y, survived))

        return {
            "survival.bool_p1pre_to_p2post": out_pre_to_p2post,
            "survival.bool_p1post_to_next_p1post": out_post_to_nextpost,
        }

    def _extract_neighbor_factor_metrics(step_data, phase: int):
        """
        For each node in PRE-STATE of the given step, compute 12 neighbor-based factors:

        Denominator = own tokens:
          - max neigh tokens / own tokens
          - avg neigh tokens / own tokens
          - min neigh tokens / own tokens
          - max neigh degree / own tokens
          - avg neigh degree / own tokens
          - min neigh degree / own tokens

        Denominator = own degree:
          - max neigh tokens / own degree
          - avg neigh tokens / own degree
          - min neigh tokens / own degree
          - max neigh degree / own degree
          - avg neigh degree / own degree
          - min neigh degree / own degree

        Returns lists of (u, x_bin, y_bin, value) keyed by metric name.
        Names are prefixed with neigh_p1.* or neigh_p2.* depending on `phase`.
        """
        pre = step_data.get("pre_state", {}) or {}
        toks, degs, Gpre = _prestate_tokens_degrees(pre)

        prefix = "neigh_p1" if phase == 1 else "neigh_p2"

        out = {
            f"{prefix}.max_tok": [],
            f"{prefix}.avg_tok": [],
            f"{prefix}.min_tok": [],
            f"{prefix}.max_deg": [],
            f"{prefix}.avg_deg": [],
            f"{prefix}.min_deg": [],

        }

        for u in Gpre.nodes():
            tb = int(toks.get(u, 0))
            dg = int(degs.get(u, 0))
            neighs = list(Gpre.neighbors(u))
            if not neighs:
                continue

            # neighbor tokens & degrees
            n_tok = [int(toks.get(v, 0)) for v in neighs]
            n_deg = [int(degs.get(v, 0)) for v in neighs]

            nmax_tok = max(n_tok);
            navg_tok = (sum(n_tok) / len(n_tok));
            nmin_tok = min(n_tok)
            nmax_deg = max(n_deg);
            navg_deg = (sum(n_deg) / len(n_deg));
            nmin_deg = min(n_deg)

            # bin by own (tokens, degree) BEFORE the phase
            x, y = _clip_tok_deg(tb, dg)

            # --- denominators: own tokens ---
            out[f"{prefix}.max_tok"].append((u, x, y, float(nmax_tok)))
            out[f"{prefix}.avg_tok"].append((u, x, y, float(navg_tok)))
            out[f"{prefix}.min_tok"].append((u, x, y, float(nmin_tok)))

            out[f"{prefix}.max_deg"].append((u, x, y, float(nmax_deg)))
            out[f"{prefix}.avg_deg"].append((u, x, y, float(navg_deg)))
            out[f"{prefix}.min_deg"].append((u, x, y, float(nmin_deg)))



        return out

    def _accumulate(vals_list, count_mat, val_sum_mat, val_sq_sum_mat):
        for _, x, y, v in vals_list:
            if 0 <= x <= token_cap and 0 <= y <= degree_cap:
                vv = float(v)
                count_mat[y, x] += 1.0
                val_sum_mat[y, x] += vv
                val_sq_sum_mat[y, x] += vv * vv

    # ---------- fast, bulk vmin/vmax estimation (single pass over sampled steps) ----------
    def _estimate_global_color_limits_bulk(
            METRICS: list[tuple[str, int, float, float | None]],
            ranges: list[tuple[int, int]],
            path_map: dict[tuple[int, int], str],
            sample_cap: int = 100,
            seed: int = 987654321,
            fallback_vmax: float = 1.0,
    ) -> dict[tuple[str, int], tuple[float, float]]:
        """
        For all metrics with default_vmax=None:
          • sample up to `sample_cap` iterations per phase from the union of `ranges`
          • load each JSON once per (t, phase) and run extractors ONCE
          • pool raw per-node values per metric
          • return global limits { (name, phase): (0.0, vmax) } using:
                vmax = q80 if mean < 0.25 * max else max
        For metrics with fixed vmax, just pass through their (vmin, vmax).
        """
        import json, numpy as np
        from collections import defaultdict

        # Which metrics actually need estimation?
        need = {(name, phase) for (name, phase, _vmin, vmax) in METRICS if vmax is None}
        if not need:
            return {(name, phase): (vmin, vmax) for (name, phase, vmin, vmax) in METRICS}

        # Prepare aggregators for values only
        agg_vals: dict[tuple[str, int], list[float]] = defaultdict(list)

        # Helper to add values from lists of (u,x,y,v)
        def _add(name: str, phase: int, rows: list[tuple[int, int, int, float]]):
            if (name, phase) not in need:
                return
            for _, _, _, v in rows:
                try:
                    vv = float(v)
                    if np.isfinite(vv):
                        agg_vals[(name, phase)].append(vv)
                except Exception:
                    pass

        # Candidate t per phase from the union of selected ranges
        def _candidate_ts_for_phase(phase: int) -> list[int]:
            cand = {t for (a, b) in ranges for t in range(a, b + 1) if (t, phase) in path_map}
            return sorted(cand)

        rng = np.random.default_rng(seed)

        # -------- Phase 1-only metrics (repro, died_p1, neigh_p1) --------
        cand_p1 = _candidate_ts_for_phase(1)
        if len(cand_p1) > sample_cap:
            cand_p1 = list(rng.choice(cand_p1, size=sample_cap, replace=False))
        for t in cand_p1:
            p1 = path_map.get((t, 1))
            if not p1:
                continue
            try:
                with open(p1, "r") as f:
                    d1 = json.load(f)
            except Exception:
                continue

            # reproduction
            repro = _extract_repro_metrics(d1)
            for k, rows in repro.items():
                _add(k, 1, rows)

            # deaths (phase 1) -> mapped metric names
            m1 = _extract_death_metrics(d1)
            map_p1 = {"died.zero": "died_p1.zero", "died.component": "died_p1.component", "died.any": "died_p1.any"}
            for k, rows in m1.items():
                name = map_p1.get(k)
                if name:
                    _add(name, 1, rows)

            # neighbor factors (phase 1)
            neigh1 = _extract_neighbor_factor_metrics(d1, phase=1)
            for k, rows in neigh1.items():
                _add(k, 1, rows)

        # -------- Phase 2-only metrics (blotto, died_p2, neigh_p2) --------
        cand_p2 = _candidate_ts_for_phase(2)
        if len(cand_p2) > sample_cap:
            cand_p2 = list(rng.choice(cand_p2, size=sample_cap, replace=False))
        for t in cand_p2:
            p2 = path_map.get((t, 2))
            if not p2:
                continue
            try:
                with open(p2, "r") as f:
                    d2 = json.load(f)
            except Exception:
                continue

            # blotto
            bl = _extract_blotto_metrics(d2)
            for k, rows in bl.items():
                _add(k, 2, rows)

            # deaths (phase 2)
            m2 = _extract_death_metrics(d2)
            map_p2 = {"died.zero": "died_p2.zero", "died.component": "died_p2.component", "died.any": "died_p2.any"}
            for k, rows in m2.items():
                name = map_p2.get(k)
                if name:
                    _add(name, 2, rows)

            # neighbor factors (phase 2)
            neigh2 = _extract_neighbor_factor_metrics(d2, phase=2)
            for k, rows in neigh2.items():
                _add(k, 2, rows)

        # -------- Survival metrics (need both phases; we only sample t where both exist) --------
        need_surv_p1 = ("survival.bool_p1pre_to_p2post", 1) in need
        need_surv_p2 = ("survival.bool_p1post_to_next_p1post", 2) in need
        if need_surv_p1 or need_surv_p2:
            both = {t for (a, b) in ranges for t in range(a, b + 1) if (t, 1) in path_map and (t, 2) in path_map}
            both = sorted(both)
            if len(both) > sample_cap:
                both = list(rng.choice(both, size=sample_cap, replace=False))
            for t in both:
                p1 = path_map.get((t, 1));
                p2 = path_map.get((t, 2))
                try:
                    with open(p1, "r") as f:
                        d1 = json.load(f)
                    with open(p2, "r") as f:
                        d2 = json.load(f)
                except Exception:
                    continue
                surv = _extract_survival_metrics(t, d1, d2, path_map)
                if need_surv_p1:
                    _add("survival.bool_p1pre_to_p2post", 1, surv.get("survival.bool_p1pre_to_p2post", []))
                if need_surv_p2:
                    _add("survival.bool_p1post_to_next_p1post", 2, surv.get("survival.bool_p1post_to_next_p1post", []))

        # -------- Build final limits dict (0 .. vmax by rule) --------
        limits: dict[tuple[str, int], tuple[float, float]] = {}
        for (name, phase, vmin_def, vmax_def) in METRICS:
            if vmax_def is not None:
                limits[(name, phase)] = (vmin_def, vmax_def)
                continue

            vals = agg_vals.get((name, phase), [])
            if not vals:
                limits[(name, phase)] = (0.0, float(fallback_vmax))
                continue

            arr = np.asarray(vals, dtype=float)
            vmax_all = float(np.nanpercentile(arr, 95))
            vmin_all = float(np.nanmin(arr))
            if not np.isfinite(vmax_all) or vmax_all <= 0:
                limits[(name, phase)] = (0.0, float(fallback_vmax))
                continue


            limits[(name, phase)] = (vmin_all, vmax_all)

        return limits

    def _plot_for_range(metric_name, count_mat, val_sum_mat, val_sq_sum_mat, default_vmin, default_vmax, title):
        import numpy as np
        import matplotlib.pyplot as plt
        import math
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        def _mean_from_cs(c, s):
            with np.errstate(invalid='ignore', divide='ignore'):
                return np.where(c > 0, s / c, np.nan)

        def _std_from_css(c, s, ss):
            # population std: sqrt(E[X^2] - (E[X])^2)
            with np.errstate(invalid='ignore', divide='ignore'):
                mean = np.where(c > 0, s / c, np.nan)
                ex2 = np.where(c > 0, ss / c, np.nan)
                var = ex2 - np.square(mean)
            # numerical guard
            var = np.where(var < 0, 0.0, var)
            return np.sqrt(var), mean

        def _avg_from_counts_local(c, s):
            with np.errstate(invalid='ignore', divide='ignore'):
                return np.where(c > 0, s / c, np.nan)

        def _auto_limits(mat, fallback=(0.0, 1.0)):
            finite = mat[np.isfinite(mat)]
            if finite.size == 0:
                return fallback
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            return (vmin, vmax) if vmin < vmax else fallback

        def _weighted_quantiles(vals, weights, probs):
            order = np.argsort(vals)
            v = np.asarray(vals)[order]
            w = np.asarray(weights)[order]
            cw = np.cumsum(w)
            if cw.size == 0 or cw[-1] <= 0:
                return np.array([np.nan] * len(probs))
            qs = []
            for p in probs:
                thresh = p * cw[-1]
                idx = np.searchsorted(cw, thresh, side="left")
                idx = min(idx, len(v) - 1)
                qs.append(v[idx])
            return np.array(qs, dtype=float)



        # shapes
        H, W = count_mat.shape
        token_cap_local = W - 1
        degree_cap_local = H - 1

        # per-bin average
        avg = _avg_from_counts_local(count_mat, val_sum_mat)

        # node-weighted marginals
        row_cnt = np.nansum(count_mat, axis=1)  # per degree
        col_cnt = np.nansum(count_mat, axis=0)  # per token
        with np.errstate(invalid='ignore', divide='ignore'):
            row_avg = np.where(row_cnt > 0, np.nansum(val_sum_mat, axis=1) / row_cnt, np.nan)
            col_avg = np.where(col_cnt > 0, np.nansum(val_sum_mat, axis=0) / col_cnt, np.nan)

        # global weighted average of the metric (over all nodes/bins)
        tot_nodes = float(np.nansum(count_mat))
        glob_avg = float(np.nansum(val_sum_mat) / tot_nodes) if tot_nodes > 0 else np.nan


        # per-bin mean/std
        std_bin, avg = _std_from_css(count_mat, val_sum_mat, val_sq_sum_mat)

        # node-weighted marginals (sum over axis, then std from totals)
        row_cnt = np.nansum(count_mat, axis=1)  # length H
        row_sum = np.nansum(val_sum_mat, axis=1)
        row_sumsq = np.nansum(val_sq_sum_mat, axis=1)
        row_std, row_avg = _std_from_css(row_cnt, row_sum, row_sumsq)

        col_cnt = np.nansum(count_mat, axis=0)  # length W
        col_sum = np.nansum(val_sum_mat, axis=0)
        col_sumsq = np.nansum(val_sq_sum_mat, axis=0)
        col_std, col_avg = _std_from_css(col_cnt, col_sum, col_sumsq)

        # full matrices with marginals (shape H+1, W+1)
        full_mean = np.full((H + 1, W + 1), np.nan, dtype=float)
        full_std = np.full((H + 1, W + 1), np.nan, dtype=float)
        full_mean[1:, 1:] = avg
        full_std[1:, 1:] = std_bin
        full_mean[1:, 0] = row_avg
        full_mean[0, 1:] = col_avg
        full_std[1:, 0] = row_std
        full_std[0, 1:] = col_std
        # (0,0) stays NaN

        # color limits
        vmin, vmax = (default_vmin, default_vmax)

        vmin_std, vmax_std = _auto_limits(full_std, fallback=(0.0, 0.5))

        # occupied bins
        ys, xs = np.where(count_mat > 0)
        if len(xs) == 0:
            return None, None
        ns = count_mat[ys, xs]

        # figure layout
        fig = plt.figure(constrained_layout=True, figsize=(15.0, 10.0))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1])

        gsTL = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], width_ratios=[1, 1], wspace=0.05)
        axTL_counts = fig.add_subplot(gsTL[0, 0])  # nodes per degree (horizontal bars)
        axTL_frac = fig.add_subplot(gsTL[0, 1])  # stacked fraction per degree

        axTopMid = fig.add_subplot(gs[0, 1])  # mean heatmap + colorbar
        axTopRight = fig.add_subplot(gs[0, 2])  # std heatmap

        gsBM = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], height_ratios=[1, 1], hspace=0.05)
        axBM_frac = fig.add_subplot(gsBM[0, 0])  # stacked fraction per token
        axBM_counts = fig.add_subplot(gsBM[1, 0])  # nodes per token (vertical bars)

        axBotLeft = fig.add_subplot(gs[1, 0])  # viridis counts + quantiles + centroid
        axBotRight = fig.add_subplot(gs[1, 2])  # counts heatmap (log1p)

        # common extent
        extent = (-1.5, token_cap_local + 0.5, -1.5, degree_cap_local + 0.5)

        # --- mean heatmap (top-middle)
        im_mean = axTopMid.imshow(full_mean, origin="lower", aspect="equal", cmap=cmap,
                                  vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest")
        axTopMid.set_title(title)
        axTopMid.set_xlabel("tokens (clipped)")
        axTopMid.set_ylabel("degree (clipped)")
        axTopMid.grid(False)
        div = make_axes_locatable(axTopMid)
        cax = div.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im_mean, cax=cax, format="%.2f")
        cbar.set_label("weighted average")

        # --- std heatmap (top-right)
        im_std = axTopRight.imshow(full_std, origin="lower", aspect="equal", cmap=cmap,
                                   vmin=vmin_std, vmax=vmax_std, extent=extent, interpolation="nearest")
        axTopRight.set_title("std deviation (per bin, Bernoulli approx)")
        axTopRight.set_xlabel("tokens (clipped)")
        axTopRight.set_ylabel("degree (clipped)")
        axTopRight.grid(False)

        # --- bottom-left: counts circles + quantile markers + black centroid
        axBotLeft.set_xlim(-1.5, 25 + 0.5)
        axBotLeft.set_ylim(-1.5, 25 + 0.5)
        axBotLeft.set_aspect("equal", adjustable="box")
        axBotLeft.set_xlabel("tokens (clipped)")
        axBotLeft.set_ylabel("degree (clipped)")
        axBotLeft.grid(False)

        # pack circles edge-to-edge
        p0 = axBotLeft.transData.transform((0, 0))
        p1 = axBotLeft.transData.transform((1, 0))
        p2 = axBotLeft.transData.transform((0, 1))
        cell_px = min(abs(p1[0] - p0[0]), abs(p2[1] - p0[1]))
        r_pts = 0.5 * cell_px
        s_max_circle = math.pi * (r_pts ** 2)

        nmax = float(ns.max()) if ns.size else 1.0
        areas_count = s_max_circle * (ns.astype(float) / nmax if nmax > 0 else 0.0)
        axBotLeft.scatter(xs, ys, s=areas_count, c=ns, cmap="viridis", alpha=0.95,
                          edgecolors="none", marker="o")
        axBotLeft.set_title("node count (size + viridis)")

        # centroid (black ×)
        total = float(ns.sum()) if ns.size else 0.0
        if total > 0:
            x_bar = float((xs * ns).sum() / total)
            y_bar = float((ys * ns).sum() / total)
            axBotLeft.scatter([x_bar], [y_bar], marker="x", c="black",
                              s=s_max_circle * 0.7, linewidths=1.8, zorder=3)

        # --- ONLY 5 diagonal quantile crosses (10, 30, 50, 70, 90) ---
        from matplotlib import colormaps  # or: from matplotlib import cm; jet = cm.get_cmap("jet")
        probs5 = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=float)

        # weighted quantiles along tokens (x) and degrees (y)
        qx = _weighted_quantiles(xs, ns, probs5)  # tokens
        qy = _weighted_quantiles(ys, ns, probs5)  # degrees

        # keep only finite pairs
        mask = np.isfinite(qx) & np.isfinite(qy)
        qx_d = qx[mask]
        qy_d = qy[mask]

        # color each of the 5 crosses along jet
        jet = colormaps["jet"]
        color_vals = np.linspace(0.0, 1.0, len(qx_d)) if len(qx_d) else []

        axBotLeft.scatter(
            qx_d, qy_d,
            s=s_max_circle * 0.7,  # same size scheme as before
            c=color_vals, cmap=jet, vmin=0.0, vmax=1.0,
            marker="x", linewidths=2.0, zorder=4
        )

        # --- top-left-left: nodes per degree (horizontal bars), hide node-axis numbers
        y_idx = np.arange(H)
        axTL_counts.barh(y_idx, row_cnt, height=1.0, color="0.35", edgecolor="none")
        axTL_counts.set_ylim(-0.5, H - 0.5)
        axTL_counts.set_xlabel("nodes")
        axTL_counts.set_ylabel("degree")
        axTL_counts.set_title("nodes per degree")
        axTL_counts.grid(False)
        # hide numbers on the NODES axis (x for this plot)
        axTL_counts.set_xticklabels([])
        axTL_counts.set_xticks([])

        # weighted mean & quantiles (degree) as horizontal lines
        deg_vals = np.arange(H)
        deg_mean = float(np.sum(deg_vals * row_cnt) / np.sum(row_cnt)) if np.sum(row_cnt) > 0 else np.nan
        deg_qs = _weighted_quantiles(deg_vals, row_cnt, probs5)
        jet = colormaps["jet"]
        for pval, qv in zip(probs5, deg_qs):
            if np.isfinite(qv):
                axTL_counts.axhline(qv, color=jet(pval), linewidth=1.6, alpha=0.9, zorder=3)
        if np.isfinite(deg_mean):
            axTL_counts.axhline(deg_mean, color="black", linewidth=2.0, alpha=0.9, zorder=4)


        from matplotlib import ticker as mtick

        # --- top-left-right: degree marginals (stacked along x-axis) + global avg line
        valid_y = np.isfinite(row_avg)
        y_pos = np.arange(H)[valid_y]
        vals_y = row_avg[valid_y]

        # clip into [vmin, vmax] and split each bar into two widths
        vals_y_clip = np.clip(vals_y, vmin, vmax)
        widths_blue = vals_y_clip - vmin  # bottom/left portion
        widths_orng = vmax - vals_y_clip  # top/right portion

        axTL_frac.barh(y_pos, widths_blue, left=vmin, height=1.0, color="C0", edgecolor="none")
        axTL_frac.barh(y_pos, widths_orng, left=vals_y_clip, height=1.0, color="C1", edgecolor="none")

        axTL_frac.set_xlim(vmin, vmax)
        axTL_frac.set_ylim(-0.5, H - 0.5)
        axTL_frac.set_yticks([])  # avoid duplicate degree labels
        axTL_frac.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        axTL_frac.set_xlabel("value")
        axTL_frac.set_title("degree marginals (avg / remainder)")
        axTL_frac.grid(False)

        # global average (clipped to the axis range)
        if np.isfinite(glob_avg):
            axTL_frac.axvline(np.clip(glob_avg, vmin, vmax), color="black", linewidth=2.0, alpha=0.9, zorder=4)

        # --- bottom-middle top: token marginals (stacked along y-axis) + global avg line
        valid_x = np.isfinite(col_avg)
        x_pos = np.arange(W)[valid_x]
        vals_x = col_avg[valid_x]

        # clip into [vmin, vmax] and split each bar into two heights
        vals_x_clip = np.clip(vals_x, vmin, vmax)
        heights_blue = vals_x_clip - vmin  # lower (blue) portion
        heights_orng = vmax - vals_x_clip  # upper (orange) portion

        axBM_frac.bar(x_pos, heights_blue, bottom=vmin, width=1.0, color="C0", edgecolor="none")
        axBM_frac.bar(x_pos, heights_orng, bottom=vals_x_clip, width=1.0, color="C1", edgecolor="none")

        axBM_frac.set_xlim(-0.5, W - 0.5)
        axBM_frac.set_ylim(vmin, vmax)
        axBM_frac.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        axBM_frac.set_xticks([])  # avoid duplication
        axBM_frac.set_ylabel("value")
        axBM_frac.set_title("token marginals (avg / remainder)")
        axBM_frac.grid(False)

        # global average (clipped to the axis range)
        if np.isfinite(glob_avg):
            axBM_frac.axhline(np.clip(glob_avg, vmin, vmax), color="black", linewidth=2.0, alpha=0.9, zorder=4)

        # --- bottom-middle bottom: nodes per token (vertical bars), hide node-axis numbers
        x_idx = np.arange(W)
        axBM_counts.bar(x_idx, col_cnt, width=1.0, color="0.35", edgecolor="none")
        axBM_counts.set_xlim(-0.5, W - 0.5)

        axBM_counts.set_xlabel("tokens")
        axBM_counts.set_ylabel("nodes")
        axBM_counts.set_title("nodes per token")
        axBM_counts.grid(False)
        # hide numbers on the NODES axis (y for this plot)
        axBM_counts.set_yticklabels([])
        axBM_counts.set_yticks([])

        # weighted mean & quantiles (token) as vertical lines
        tok_vals = np.arange(W)
        tok_mean = float(np.sum(tok_vals * col_cnt) / np.sum(col_cnt)) if np.sum(col_cnt) > 0 else np.nan
        tok_qs = _weighted_quantiles(tok_vals, col_cnt, probs5)
        for pval, qv in zip(probs5, tok_qs):
            if np.isfinite(qv):
                axBM_counts.axvline(qv, color=jet(pval), linewidth=1.6, alpha=0.9, zorder=3)
        if np.isfinite(tok_mean):
            axBM_counts.axvline(tok_mean, color="black", linewidth=2.0, alpha=0.9, zorder=4)

        # --- bottom-right: counts heatmap (log1p), hide NaN/zero ---
        count_full = np.full((H + 1, W + 1), np.nan, dtype=float)
        count_full[1:, 1:] = count_mat
        count_full[1:, 0] = row_cnt
        count_full[0, 1:] = col_cnt

        # Mask NaNs and zeros
        mask = ~np.isfinite(count_full) | (count_full <= 0)
        count_masked = np.ma.masked_where(mask, count_full)

        # Log-transform while preserving mask
        count_vis = np.ma.log10(count_masked)

        # Transparent for masked/underflowed cells
        from matplotlib import cm
        cmap_counts = colormaps["viridis"]
        cmap_counts.set_bad(alpha=0.0)  # masked -> transparent
        cmap_counts.set_under(alpha=0.0)  # values < vmin -> transparent

        # Color limits from available (unmasked) data; safe fallback
        if count_vis.count() > 0:
            vmin_c = float(count_vis.min())
            vmax_c = float(count_vis.max())
        else:
            vmin_c, vmax_c = 0.0, 1.0

        axBotRight.imshow(
            count_vis,
            origin="lower",
            aspect="equal",
            cmap=cmap_counts,
            extent=extent,
            interpolation="nearest",
            vmin=vmin_c,
            vmax=vmax_c,
        )
        axBotRight.set_title("log(count+1)")
        axBotRight.set_xlabel("tokens (clipped)")
        axBotRight.set_ylabel("degree (clipped)")
        axBotRight.grid(False)

        # save
        path_color = an._savefig(f"sixpack_{metric_name}")
        return path_color


    # ---------- main aggregation ----------
    valid_ts = _valid_iterations()
    ranges = _pick_ranges(valid_ts)
    if not ranges:
        print("No valid iteration ranges found for plotting.")
        return []

    saved = []
    # TODO metric conditional, nodes at p2 pre state were freshly born ()
    METRICS = [
        ("reproduction.bool_reproduced",  1,  0.0, None),
        ("reproduction.repro_token_frac", 1,  0.0, None),
        ("reproduction.bool_new_link",    1,  0.0, None),
        ("reproduction.shift_frac",       1,  0.0, None),   # NEW
        ("reproduction.link_new_frac",    1,  0.0, None),   # NEW
        ("reproduction.bool_did_reconnect",1, 0.0, None),   # NEW
        ("blotto.bool_self_win",          2,  0.0, None),
        ("blotto.bool_win_anywhere",      2,  0.0, None),
        ("blotto.wins_ratio_ego",         2,  0.0, None),
        ("blotto.token_gain_factor",      2,  0.0, None),   # NEW (auto scaling can override)
        ("blotto.self_alloc_frac",        2,  0.0, None),   # NEW
        ("blotto.incoming_max_over_tokens",2, 0.0, None),   # NEW
        ("blotto.win_share",              2,  0.0, None),   # NEW
        ("died_p1.zero",                  1,  0.0, None),
        ("died_p1.component",             1,  0.0, None),
        ("died_p1.any",                   1,  0.0, None),
        ("died_p2.zero",                  2,  0.0, None),
        ("died_p2.component",             2,  0.0, None),
        ("died_p2.any",                   2,  0.0, None),

        # Survival (derived across phases/iterations)
        ("survival.bool_p1pre_to_p2post", 1, 0.0, None),  # computed while iterating t
        ("survival.bool_p1post_to_next_p1post", 2, 0.0, None),

    ]

    _neigh_items = [
        ("max_tok", None),
        ("avg_tok", None),
        ("min_tok", None),
        ("max_deg", None),
        ("avg_deg", None),
        ("min_deg", None),


    ]

    _existing = {m[0] for m in METRICS}  # avoid duplicates if rerun
    for prefix, phase in (("neigh_p1", 1), ("neigh_p2", 2)):
        for key, vmax in _neigh_items:
            name = f"{prefix}.{key}"
            if name not in _existing:
                METRICS.append((name, phase, 0.0, vmax))


    # Build (t, phase) -> path map (keep first occurrence)
    path_map = {}
    for path, t, phase_name, phase in an.reader.iter_paths():
        if (t, phase) not in path_map:
            path_map[(t, phase)] = path

    # Pick global limits per metric (same vmin/vmax reused for every range)
    global_limits: dict[tuple[str, int], tuple[float, float]] = _estimate_global_color_limits_bulk(
        METRICS=METRICS,
        ranges=ranges,  # union of selected windows
        path_map=path_map,
        sample_cap=500,
        seed=987654321,
        fallback_vmax=1.0,
    )


    for r_i, (t_start, t_end) in enumerate(ranges, start=1):
        # allocate matrices for exactly the metrics we plan to plot
        mats = {
            name: (
                np.zeros((degree_cap + 1, token_cap + 1), dtype=float),  # count
                np.zeros((degree_cap + 1, token_cap + 1), dtype=float),  # sum
                np.zeros((degree_cap + 1, token_cap + 1), dtype=float),  # sum of squares
            )
            for (name, *_rest) in METRICS
        }

        for t in range(t_start, t_end + 1):
            d1 = d2 = None

            # ---- Phase 1 ----
            p1 = path_map.get((t, 1))
            if p1:
                try:
                    with open(p1, "r") as f:
                        d1 = json.load(f)
                except Exception:
                    d1 = None

            if d1 is not None:
                # reproduction metrics
                for key, vals in _extract_repro_metrics(d1).items():
                    if key in mats:
                        _cnt, _sum, _sumsq = mats[key]
                        _accumulate(vals, _cnt, _sum, _sumsq)

                # ---- Phase 1 deaths ----
                m = _extract_death_metrics(d1)
                map_p1 = {
                    "died.zero": "died_p1.zero",
                    "died.component": "died_p1.component",
                    "died.any": "died_p1.any",
                }
                for key, vals in m.items():
                    dst = map_p1.get(key)
                    if not dst:
                        continue
                    if dst in mats:
                        _cnt, _sum, _sumsq = mats[dst]  # <-- use dst here
                        _accumulate(vals, _cnt, _sum, _sumsq)
            # ---- Phase 2 ----
            p2 = path_map.get((t, 2))


            if p2:
                try:
                    with open(p2, "r") as f:
                        d2 = json.load(f)
                except Exception:
                    d2 = None

            if d2 is not None:
                # blotto metrics (incl. the new ones)
                for key, vals in _extract_blotto_metrics(d2).items():
                    if key in mats:
                        _cnt, _sum, _sumsq = mats[key]
                        _accumulate(vals, _cnt, _sum, _sumsq)
                # ---- Phase 2 deaths ----
                m = _extract_death_metrics(d2)
                map_p2 = {
                    "died.zero": "died_p2.zero",
                    "died.component": "died_p2.component",
                    "died.any": "died_p2.any",
                }
                for key, vals in m.items():
                    dst = map_p2.get(key)
                    if not dst:
                        continue
                    if dst in mats:
                        _cnt, _sum, _sumsq = mats[dst]  # <-- use dst here
                        _accumulate(vals, _cnt, _sum, _sumsq)
                # After handling phase 2 JSON (d2) for iteration t:
                if d2 is not None:
                    for key, vals in _extract_neighbor_factor_metrics(d2, phase=2).items():
                        if key in mats:
                            _cnt, _sum, _sumsq = mats[key]
                            _accumulate(vals, _cnt, _sum, _sumsq)
            # ---- Survival metrics (need both d1 and d2; also uses next t's p1 via path_map) ----
            if (d1 is not None) and (d2 is not None):
                surv = _extract_survival_metrics(t, d1, d2, path_map)
                for key, vals in surv.items():
                    if key in mats:
                        _cnt, _sum, _sumsq = mats[key]
                        _accumulate(vals, _cnt, _sum, _sumsq)


        for name, phase, default_vmin, default_vmax in METRICS:
            vmin_auto, vmax_auto = global_limits[(name, phase)]
            count_mat, sum_mat, sumsq_mat = mats[name]
            path_color = _plot_for_range(
                metric_name=f"p{phase}-{name}_r{r_i}_t{t_start}-{t_end}",
                count_mat=count_mat,
                val_sum_mat=sum_mat,
                val_sq_sum_mat=sumsq_mat,  # <-- NEW
                default_vmin=vmin_auto,
                default_vmax=vmax_auto,
                title=f"p{phase}-{name} — \n t∈[{t_start},{t_end}] (range {r_i}/{len(ranges)})"

            )
            if path_color: saved.append(path_color)
    return saved


def load_or_compute_df(
    an,
    out_dir,
    load_path=None,
    max_iter=50,
    progress=True,
    chunk_iter=10,   # NEW: save after every N iterations (t values)
):
    """
    Load metrics from CSV if available; otherwise compute.
    If the CSV is missing any (t, phase) pairs for t < max_iter, compute only the missing ones
    and append them in streaming fashion. Progress is flushed to disk every `chunk_iter` iterations.

    Returns a DataFrame filtered to t < max_iter.
    """

    # --- locate existing CSV ---
    def _find_csv(load_path, out_dir):
        if load_path:
            p = Path(load_path)
            if p.is_file():
                return p
            if p.is_dir():
                for c in [p / "metrics_per_step.csv", p / "analysis_out" / "metrics_per_step.csv"]:
                    if c.exists():
                        return c
        c = Path(out_dir) / "metrics_per_step.csv"
        return c if c.exists() else None

    csv_path = _find_csv(load_path, out_dir)
    if csv_path:
        print(f"Loading metrics from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    # --- what we need vs have ---
    target = int(max_iter)
    need_pairs = {(t, p) for t in range(target) for p in (1, 2)}
    have_pairs = set()
    if not df.empty and {"t", "phase"}.issubset(df.columns):
        have_pairs = {(int(t), int(p)) for t, p in zip(df["t"], df["phase"]) if int(t) < target}

    missing = sorted(need_pairs - have_pairs)
    if missing:
        # Map (t,phase) -> (path, phase_name_guess)
        path_map = {}
        for path, t, phase_name_guess, phase in an.reader.iter_paths():
            if t < target:
                path_map[(t, phase)] = (path, phase_name_guess)

        # Group missing by iteration t
        missing_ts = sorted({t for (t, _p) in missing})

        # Progress bar over iterations (t)
        iterator = missing_ts
        if progress:
            try:
                from tqdm import tqdm as _tqdm
                iterator = _tqdm(iterator, total=len(missing_ts), desc="Computing missing iterations", unit="iter")
            except Exception:
                pass

        rows_buffer = []
        processed_t = 0
        final_csv = Path(out_dir) / "metrics_per_step.csv"
        final_csv.parent.mkdir(parents=True, exist_ok=True)

        def _flush():
            nonlocal df, rows_buffer
            if not rows_buffer:
                return
            df_chunk = pd.DataFrame(rows_buffer)
            # Union columns with existing df (fill missing with NaN)
            if df.empty:
                df = df_chunk
            else:
                df = pd.concat([df, df_chunk], ignore_index=True, sort=False)
            # Sort & save
            df.sort_values(["t", "phase"], inplace=True, ignore_index=True)
            df.to_csv(final_csv, index=False)
            print(f"[autosave] Saved {len(rows_buffer)} new rows → {final_csv}")
            rows_buffer = []

        for t in iterator:
            # For each iteration, attempt both phases (if missing & file exists)
            for phase in (1, 2):
                if (t, phase) not in missing:
                    continue
                item = path_map.get((t, phase))
                if not item:
                    # No step file on disk (run didn't reach this far yet)
                    continue
                path, phase_name_guess = item

                # Load JSON (streaming)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"⚠️  Failed to read {path}: {e}")
                    continue

                phase_name = data.get("phase", phase_name_guess)
                # Build minimal Record-like object for metric fns
                rec = SimpleNamespace(t=t, phase=phase, phase_name=phase_name, path=path, data=data)
                G = graph_from_post_state(data)

                # Compute all metrics for this step
                row = {"t": t, "phase": phase, "phase_name": phase_name, "path": path}
                for name, fn in an.registry.items():
                    try:
                        vals = fn(rec, G)
                        if isinstance(vals, dict):
                            for k, v in vals.items():
                                if isinstance(v, dict):
                                    row[f"{name}.{k}"] = json.dumps(v, separators=(",", ":"))
                                else:
                                    row[f"{name}.{k}"] = v
                    except Exception as e:
                        row[f"{name}.__error"] = str(e)

                rows_buffer.append(row)
                # Drop big objects
                del data, rec, G

            processed_t += 1
            if processed_t % chunk_iter == 0:
                _flush()

        # Final flush
        _flush()
    else:
        # Ensure we have a CSV on disk for consistent downstream use
        final_csv = Path(out_dir) / "metrics_per_step.csv"
        if not final_csv.exists() and not df.empty:
            final_csv.parent.mkdir(parents=True, exist_ok=True)
            df.sort_values(["t", "phase"], inplace=True, ignore_index=True)
            df.to_csv(final_csv, index=False)
            print(f"Saved metrics to: {final_csv}")

    # Always return only the requested range
    return df[df["t"] < target].reset_index(drop=True)

if __name__ == "__main__":

    run_dir = "/home/stefan/Documents/PythonProjects/GraphOfLife/GraphOfLifeOutputs/run_20251106_163130_536"
    out_dir = "/home/stefan/Documents/PythonProjects/GraphOfLife/GraphOfLifeOutputs/run_20251106_163130_536_out"

    reader = RunReader(run_dir)
    reg = MetricRegistry()
    default_metrics(reg)

    an = Analyzer(reader, reg, out_dir=out_dir)
    df = load_or_compute_df(an, out_dir, load_path=run_dir,  max_iter=999999, progress=True)
    default_plots(an, df)

    plot_token_degree_aggregate_ranges(an, token_cap = 80, degree_cap = 80, range_len = 500, skip_first = 1000, num_ranges = 3, cmap = "jet")
