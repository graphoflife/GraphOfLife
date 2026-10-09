# -*- coding: utf-8 -*-
"""
Part II, the inside of a world: Chapter 18 — what the brains are like — and
Chapter 19 — what agents say to each other.

Both read the final checkpoint of every world that lived to the end, which holds
every living agent's brain and every message in flight at iteration 3,000. The
probes rebuild the world from it and run each brain on the inputs it would read
there — once as it is, and again with one input changed — so what a brain
responds to is measured on the states it actually meets.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, describe, dots, line, lived, recipe, runs_of, survivors
from book_chapters.common import BLUE, CYAN, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW, baseline

#: What each group of a brain's input rows reads, by the blocks of
#: SimConfig.input_layout() it is made of.
GROUPS = [("me?", ("self",)), ("my tokens", ("own_tokens",)), ("its tokens", ("its_tokens",)),
          ("my links", ("own_degree",)), ("its links", ("its_degree",)),
          ("around", ("own_token_quantiles", "its_token_quantiles",
                      "own_degree_quantiles", "its_degree_quantiles")),
          ("I→me", ("message_to_self",)), ("I→it", ("message_to_it",)),
          ("it→me", ("message_from_it",)), ("it→it", ("message_its_own",)),
          ("random", ("noise",))]
MESSAGE_BLOCKS = ("message_to_self", "message_to_it", "message_from_it", "message_its_own")


def rows_of(cfg, blocks) -> slice:
    """
    The input rows a run of neighbouring blocks occupies, in this configuration's
    layout — as a slice, so that a measure over them sums exactly as it did when
    the rows were written out by hand.
    """
    layout = cfg.input_layout()
    spans = [layout[name][:2] for name in blocks]
    for (_, stop), (start, _) in zip(spans, spans[1:]):
        assert stop == start, f"{blocks} are not next to each other in the input"
    return slice(spans[0][0], spans[-1][1])
FINAL = ("the final checkpoint of each run, `GraphOfLifeRuns/<run>/checkpoint.npz`, read with "
         "`gol_store.load_checkpoint(run, config)`: the world after its last iteration, every living "
         "agent's brain and every message in flight")
CHANGELESS = ("the 30 worlds *brains never change* of Experiment 7 (`B1-10000-a8525d-s001` … `-s030`, "
              "Chapter 37), whose brains are copies of their founders'")


def world_at_end(run_id: str):
    import gol_store as store
    return store.load_checkpoint(run_id, store.load_config(run_id))


def _cached(run_id: str, kind: str, make) -> Dict[str, Any]:
    """A per-run result kept beside the runs, keyed by the run's last iteration."""
    path = os.path.join(F.store.BASE_DIR, ".book", f"{run_id}.{kind}.json")
    stamp = F.store.load_meta(run_id).get("iteration")
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        if cached.get("stamp") == stamp:
            return cached
    out = {"stamp": stamp, **make(run_id)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def _decide(brain, heads, X: np.ndarray, tokens: int) -> Tuple[int, bool, np.ndarray, float, np.ndarray]:
    """
    What a brain would stake and give a child, given one input matrix, by the engine's
    own rules — and how many of its stake scores are positive, since only those count:
    with none, the tokens are split evenly over every candidate — and the mean, over the
    candidates, of each of the fifteen outputs that make decisions.
    """
    from GraphOfLifeSimple import _apportion, _share_of_first
    Y = brain.forward(X)
    scores = np.asarray(Y[heads["BLOTTO"], :], float)
    mode = np.mean(Y[heads["BLOTTO_MODE"], :], axis=1)
    spread = bool(mode[0] > mode[1])
    if spread:
        alloc = _apportion(scores, tokens)
    else:
        alloc = np.zeros(X.shape[1], dtype=int)
        alloc[int(np.argmax(scores))] = tokens
    frac = np.mean(Y[heads["REPRO_FRACTION"], :], axis=1)
    return (int((scores > 0).sum()), spread, alloc / tokens, float(_share_of_first(frac[0], frac[1])),
            Y[:heads["MESSAGE_START"]].mean(axis=1))


def probe(run_id: str) -> Dict[str, Any]:
    """
    Every living agent's decisions at the end of a world, as its brain makes them,
    with one input changed at a time, and as a fresh founder's brain would make them
    on the same inputs.
    """
    def make(run_id_: str) -> Dict[str, Any]:
        from GraphOfLifeSimple import make_brain
        w = world_at_end(run_id_)
        log_deg, neighs, q_tok, q_deg, log_tok, at_risk = w._precompute_features()
        rng = np.random.default_rng(18)
        founders = np.random.RandomState(1800)
        rows: Dict[str, List[float]] = {}

        def note(key, value):
            rows.setdefault(key, []).append(float(value))
        for u in sorted(w.G.nodes()):
            tokens = int(w.tokens.get(u, 0))
            if tokens <= 0:
                continue
            targets = [u] + list(neighs[u])
            X = w._inputs(u, targets, log_deg, q_tok, q_deg, log_tok, at_risk)
            founder = make_brain(w.cfg, 0, founders)
            note("tokens", tokens)
            note("degree", len(targets) - 1)
            note("even", 1 / len(targets))
            for who, brain in (("", w.brains[u]), ("founder_", founder)):
                positive, spread, alloc, share, means = _decide(brain, w.heads, X, tokens)
                for r, m in enumerate(means):
                    note(f"{who}out{r}", m)
                note(who + "positive", positive)
                note(who + "spread", spread)
                note(who + "home", alloc[0])
                note(who + "share", share)
                for name in ("messages", "noise", "richer"):
                    Z = X.copy()
                    if name == "messages":
                        Z[rows_of(w.cfg, MESSAGE_BLOCKS)] = 0.0
                    elif name == "noise":
                        noise = rows_of(w.cfg, ("noise",))
                        Z[noise] = rng.uniform(-2.0, 2.0, Z[noise].shape)
                    else:
                        its = rows_of(w.cfg, ("its_tokens",)).start
                        Z[its, 1:] = np.log1p(2.0 * np.expm1(Z[its, 1:]))
                    _, _, a2, s2, _ = _decide(brain, w.heads, Z, tokens)
                    note(f"{who}tv_{name}", 0.5 * float(np.abs(alloc - a2).sum()))
                    note(f"{who}share_{name}", s2)
        return rows
    return _cached(run_id, "probe3", make)


def weights(run_id: str, pairs: int = 3000) -> Dict[str, Any]:
    """Input-weight norms by group, and distances between brains, at the end of a world."""
    def make(run_id_: str) -> Dict[str, Any]:
        from gol_config import SimConfig
        cfg = SimConfig.from_dict(F.store.load_meta(run_id_)["config"])
        z = np.load(F.store.checkpoint_path(run_id_), allow_pickle=False)
        W0 = z["W0"].astype(np.float64)                                   # agents × 50 × 154
        norms = np.sqrt((W0 ** 2).sum(axis=1))                            # agents × 154
        group = {name: float(norms[:, rows_of(cfg, blocks)].mean()) for name, blocks in GROUPS}
        flat = np.concatenate([z[k].astype(np.float32).reshape(len(z["ids"]), -1)
                               for k in z.files if k[0] in "Wb" and k[1:].isdigit()], axis=1)
        geno, parent = z["brain_ids"], z["parent_brain_ids"]
        first = {}
        for i, g in enumerate(geno):
            first.setdefault(int(g), i)
        rng = np.random.default_rng(19)

        def rms(i, j):
            return float(np.sqrt(np.mean((flat[i] - flat[j]) ** 2)))
        n = len(geno)
        random_pairs = [rms(*rng.choice(n, 2, replace=False)) for _ in range(pairs)]
        # A genotype and its parent, where both are alive: one change apart.
        lineage = [rms(i, first[int(parent[i])]) for i in range(n) if int(parent[i]) in first
                   and parent[i] != geno[i]]
        return {"groups": group, "random": random_pairs, "parent": lineage[:pairs],
                "agents": n, "genotypes": len(first)}
    return _cached(run_id, "weights", make)


def founder_weights(cfg, count: int = 400, seed: int = 1801) -> Tuple[Dict[str, float], List[float]]:
    """The same measures for fresh founders' brains: input norms by group, and distances between two."""
    from GraphOfLifeSimple import make_brain
    rs = np.random.RandomState(seed)
    brains = [make_brain(cfg, 0, rs) for _ in range(count)]
    W0 = np.stack([b.weights[0].astype(np.float64) for b in brains])
    norms = np.sqrt((W0 ** 2).sum(axis=1))
    groups = {name: float(norms[:, rows_of(cfg, blocks)].mean()) for name, blocks in GROUPS}
    flat = [np.concatenate([w.astype(np.float32).ravel() for w in b.weights] +
                           [x.astype(np.float32).ravel() for x in b.biases]) for b in brains]
    dist = [float(np.sqrt(np.mean((flat[i] - flat[i + 1]) ** 2))) for i in range(0, count - 1, 2)]
    return groups, dist


def message_table(run_id: str) -> Dict[str, Any]:
    """Every message in flight at the end of a world, with who wrote it, to whom, and their states."""
    def make(run_id_: str) -> Dict[str, Any]:
        z = np.load(F.store.checkpoint_path(run_id_), allow_pickle=False)
        ids = list(z["ids"])
        at = {int(a): k for k, a in enumerate(ids)}
        degree = np.zeros(len(ids))
        for a, b in z["edges"]:
            degree[at[int(a)]] += 1
            degree[at[int(b)]] += 1
        src = np.array([at[int(a)] for a in z["msg_from"]])
        dst = np.array([at[int(a)] for a in z["msg_to"]])
        return {"values": z["msg_values"].round(5).tolist(), "src": src.tolist(), "dst": dst.tolist(),
                "tokens": z["tokens"].tolist(), "degree": degree.tolist(), "genotype": z["brain_ids"].tolist()}
    return _cached(run_id, "messages", make)


def _message_summary(M: np.ndarray, sender: np.ndarray, genotype: np.ndarray, features: np.ndarray,
                     rng: np.random.Generator) -> Dict[str, float]:
    """
    What a set of messages is like: how many independent numbers they use, how much of
    their variety is between writers and how much between the agents one writer writes
    to, how much is shared by writers of one genotype, and how much simple facts about
    writer and reader explain.
    """
    centred = M - M.mean(axis=0)
    total = float((centred ** 2).sum())
    eig = np.clip(np.linalg.eigvalsh(np.cov(M.T)), 0, None)
    effective = float(eig.sum() ** 2 / (eig ** 2).sum()) if eig.sum() > 0 else 0.0

    def between(groups: np.ndarray) -> float:
        out = 0.0
        for g in np.unique(groups):
            rows = groups == g
            out += rows.sum() * float(((M[rows].mean(axis=0) - M.mean(axis=0)) ** 2).sum())
        return out / total if total else 0.0
    by_writer = between(sender)
    # Writers' own means, grouped by their genotype, against the same with genotypes shuffled.
    writers = np.unique(sender)
    means = np.array([M[sender == s].mean(axis=0) for s in writers])
    their = np.array([genotype[s] for s in writers])

    def share(labels):
        c = means - means.mean(axis=0)
        tot = float((c ** 2).sum())
        b = sum((labels == g).sum() * float(((means[labels == g].mean(axis=0) - means.mean(axis=0)) ** 2).sum())
                for g in np.unique(labels))
        return b / tot if tot else 0.0
    kin = share(their)
    shuffled = [share(rng.permutation(their)) for _ in range(50)]
    A = np.column_stack([np.ones(len(M)), features])
    coef, *_ = np.linalg.lstsq(A, M, rcond=None)
    fitted = A @ coef
    explained = 1 - float(((M - fitted) ** 2).sum()) / total if total else 0.0
    # How different one writer's messages to two of its readers are, against two writers'.
    to_two, by_two, previous = [], [], None
    for s in writers:
        rows = M[sender == s]
        if len(rows) > 1:
            to_two.append(float(np.sqrt(((rows[0] - rows[1]) ** 2).mean())))
        if previous is not None:
            by_two.append(float(np.sqrt(((rows[0] - previous) ** 2).mean())))
        previous = rows[0]
    return {"effective": effective, "by_writer": by_writer, "kin": kin, "kin_null": float(np.mean(shuffled)),
            "kin_null_hi": float(np.quantile(shuffled, 0.95)), "explained": explained,
            "saturated": float(np.mean(np.abs(M) > 0.9)), "count": int(len(M)),
            "to_two_readers": float(np.median(to_two)), "by_two_writers": float(np.median(by_two)),
            "values": np.histogram(M, bins=40, range=(-1, 1))[0].tolist()}


def message_stats(run_id: str) -> Dict[str, Any]:
    """The messages every agent would write to each of its candidates at the end of a world,
    by its own brain and by a fresh founder's brain on the same inputs, summarised."""
    def make(run_id_: str) -> Dict[str, Any]:
        from GraphOfLifeSimple import make_brain
        w = world_at_end(run_id_)
        log_deg, neighs, q_tok, q_deg, log_tok, at_risk = w._precompute_features()
        founders = np.random.RandomState(1900)
        start = w.heads["MESSAGE_START"]
        out = {"evolved": [], "founder": []}
        sender, features, geno = [], [], {}
        for u in sorted(w.G.nodes()):
            targets = [u] + list(neighs[u])
            X = w._inputs(u, targets, log_deg, q_tok, q_deg, log_tok, at_risk)
            founder = make_brain(w.cfg, 0, founders)
            for who, brain in (("evolved", w.brains[u]), ("founder", founder)):
                out[who].append(np.tanh(brain.forward(X)[start:start + w.cfg.message_amount]).T)
            for v in targets:
                sender.append(u)
                features.append([log_tok[u], log_deg[u], log_tok.get(v, 0.0), log_deg[v], float(u == v)])
            geno[u] = w.brains[u].brain_id
        sender = np.array(sender)
        features = np.array(features)
        genotype = {u: g for u, g in geno.items()}
        labels = np.array([genotype[s] for s in sender])
        rng = np.random.default_rng(1901)
        index = {u: k for k, u in enumerate(sorted(geno))}
        sender_idx = np.array([index[s] for s in sender])
        geno_by_idx = np.array([genotype[u] for u in sorted(geno)])
        return {who: _message_summary(np.concatenate(out[who]), sender_idx, geno_by_idx, features, rng)
                for who in out} | {"genotypes": int(len(set(labels.tolist())))}
    return _cached(run_id, "msgstats2", make)


def gain(run_id: str, agents: int = 300) -> Dict[str, Any]:
    """
    How strongly a brain's outputs follow its inputs: every input moved by a small random
    amount, and the outputs' change divided by the inputs' — for the evolved brains of a
    world at its end and for fresh founders' brains, on the same inputs. Also each output's
    spread across one agent's candidates against its spread across agents.
    """
    def make(run_id_: str) -> Dict[str, Any]:
        from GraphOfLifeSimple import make_brain
        w = world_at_end(run_id_)
        log_deg, neighs, q_tok, q_deg, log_tok, at_risk = w._precompute_features()
        rng = np.random.default_rng(2000)
        founders = np.random.RandomState(2001)
        nodes = sorted(w.G.nodes())
        picked = [nodes[i] for i in rng.choice(len(nodes), min(agents, len(nodes)), replace=False)]
        out = {"evolved": [], "founder": [], "within": [], "between": [], "layers": []}
        firsts = []
        for u in picked:
            targets = [u] + list(neighs[u])
            X = w._inputs(u, targets, log_deg, q_tok, q_deg, log_tok, at_risk)
            eps = rng.normal(0.0, 0.1, X.shape)
            for who, brain in (("evolved", w.brains[u]), ("founder", make_brain(w.cfg, 0, founders))):
                d = brain.forward(X + eps) - brain.forward(X)
                out[who].append(float(np.sqrt((d ** 2).mean()) / np.sqrt((eps ** 2).mean())))
            Y = w.brains[u].forward(X)
            if Y.shape[1] > 1:
                out["within"].append(float(Y.std(axis=1).mean()))
            firsts.append(Y[:, 0])
            # How the spread of a small change shrinks from layer to layer, for this brain.
            a, b = X.astype(float), (X + eps).astype(float)
            ratios = []
            for i, (Wm, bm) in enumerate(zip(w.brains[u].weights, w.brains[u].biases)):
                za = Wm.astype(float) @ a + bm.astype(float)
                zb = Wm.astype(float) @ b + bm.astype(float)
                last = i == len(w.brains[u].weights) - 1
                a2 = za if last else 1 / (1 + np.exp(-za))
                b2 = zb if last else 1 / (1 + np.exp(-zb))
                ratios.append(float(np.sqrt(((b2 - a2) ** 2).mean()) / max(1e-300, np.sqrt(((b - a) ** 2).mean()))))
                a, b = a2, b2
            out["layers"].append(ratios)
        out["between"] = float(np.stack(firsts).std(axis=0).mean())
        out["layers"] = np.median(np.array(out["layers"]), axis=0).tolist()
        return out
    return _cached(run_id, "gain", make)


# ---------------------------------------------------------------------------
# Chapter 18 · What the brains are like
# ---------------------------------------------------------------------------

ENDS = ("the 26 baseline worlds that lived to the end (`B1-10000-s001` … `-s030`, Chapter 9), each at "
        "the end of its 3,000 iterations")
PROBE = ("For every agent of the world with at least one token, its inputs exactly as the engine builds "
         "them for a look at its candidates (`World._precompute_features` and `World._inputs`), and its "
         "brain's outputs (`Brain.forward`); the decisions follow by the engine's own rules "
         "(`_apportion` for the stakes, `_share_of_first` for a child's share).")


def _founder_note() -> str:
    return ("a fresh founder's brain — weights drawn as for the founders of a world, normal with standard "
            "deviation 1/√(fan-in), biases 0 — given exactly the same inputs")


@chapter
def minds(ch: F.Chapter) -> None:
    alive = survivors(baseline())
    fixed = survivors(runs_of("E07", "brains never change"))
    probes = {s.run_id: {k: np.array(v) for k, v in probe(s.run_id).items() if k != "stamp"} for s in alive}
    gains = {s.run_id: gain(s.run_id) for s in alive}

    # ---- a change in, a change out ----
    ge = [float(np.median(g["evolved"])) for g in gains.values()]
    gf = [float(np.median(g["founder"])) for g in gains.values()]
    layers = np.median(np.array([g["layers"] for g in gains.values()]), axis=0)
    cats = ["evolved brains", "founders' brains"]
    lcats = ["layer 1", "layer 2", "layer 3", "layer 4", "layer 5", "outputs"]
    ch.grid(
        "gain", [
            dict(title="A change in, a change out", x={"label": "", "categories": cats},
                 y={"label": "output change ÷ input change (logarithmic)", "log": True, "min": 1e-4, "max": 1e-2},
                 series=dots(cats, [ge, gf], [BLUE, GREY]), legend=False),
            dict(title="Layer by layer", x={"label": "", "categories": lcats},
                 y={"label": "change after the layer ÷ change before", "min": 0, "max": 1.4},
                 series=[{"label": None, "kind": "bars", "x0": [k - 0.32 for k in range(6)],
                          "x1": [k + 0.32 for k in range(6)], "y": layers.tolist(), "colour": BLUE}],
                 guides=[{"axis": "y", "at": 0.25, "label": "¼: a sigmoid's steepest slope"},
                         {"axis": "y", "at": 1.0, "label": "unchanged"}], legend=False)],
        columns=2, title="How much of what a brain sees reaches what it does",
        caption="Left: for 300 agents of each world, every one of the 154 inputs of each of its columns moved by "
                "a small random amount (normal, standard deviation 0.1), and the root-mean-square change of the "
                "45 outputs divided by that of the inputs — the brain's gain; one dot per world, the median of "
                "its agents, for the agents' own brains (blue) and for " + _founder_note() + " (grey). Right: "
                "the same ratio layer by layer for the evolved brains, the median over worlds: each of the five "
                "hidden layers ends in a sigmoid, the last is linear.",
        recipe=recipe("The 26 baseline worlds that lived to the end.", FINAL,
                      [PROBE, "Draw ε, normal with standard deviation 0.1, the shape of the input matrix X; the "
                       "gain is RMS(forward(X + ε) − forward(X)) / RMS(ε). The same after each layer for the "
                       "layer-by-layer ratios (`book_chapters.inner.gain`)."]))
    ch.number("gain", {"evolved": describe(ge), "founder": describe(gf), "layers": layers.tolist(),
                       "within": describe([float(np.median(g["within"])) for g in gains.values()]),
                       "between": describe([g["between"] for g in gains.values()])})

    # ---- what evolution changed ----
    def per_world(fn):
        return [float(fn(d)) for d in probes.values()]
    spread_e = per_world(lambda d: d["spread"].mean())
    spread_f = per_world(lambda d: d["founder_spread"].mean())
    child_e = per_world(lambda d: (np.floor(d["share"] * d["tokens"]) >= 1).mean())
    child_f = per_world(lambda d: (np.floor(d["founder_share"] * d["tokens"]) >= 1).mean())
    even_e = per_world(lambda d: (d["positive"] == 0).mean())
    even_f = per_world(lambda d: (d["founder_positive"] == 0).mean())
    ch.grid(
        "decisions", [
            dict(title="Spread the stakes", x={"label": "", "categories": cats},
                 y={"label": "share of agents", "min": 0, "max": 1}, series=dots(cats, [spread_e, spread_f], [BLUE, GREY]),
                 legend=False),
            dict(title="Have a child", x={"label": "", "categories": cats},
                 y={"label": "share of agents", "min": 0, "max": 1}, series=dots(cats, [child_e, child_f], [BLUE, GREY]),
                 legend=False),
            dict(title="Split evenly, by default", x={"label": "", "categories": cats},
                 y={"label": "share of agents", "min": 0, "max": 1}, series=dots(cats, [even_e, even_f], [BLUE, GREY]),
                 legend=False)],
        columns=3, title="What evolution changed",
        caption="One dot per world: of its agents with a token or more at the end, the share whose brain would "
                "spread its stakes rather than go all in; the share that would have a child (a share of its "
                "tokens of at least one token); and the share whose every stake score is zero or below, so that "
                "the rules split its tokens evenly over all its candidates. Blue: the agents' own brains; grey: "
                + _founder_note() + ".",
        recipe=recipe("The 26 baseline worlds that lived to the end.", FINAL,
                      [PROBE, "Spread: the mean of the two BLOTTO_MODE outputs over the columns, first larger than "
                       "second. Child: ⌊f(ā, b̄) · tokens⌋ ≥ 1 with ā, b̄ the REPRO_FRACTION outputs averaged over "
                       "the columns. Even by default: every BLOTTO score ≤ 0 (`book_chapters.inner.probe`)."]))
    ch.number("decisions", {"spread": [describe(spread_e), describe(spread_f)],
                            "child": [describe(child_e), describe(child_f)],
                            "even": [describe(even_e), describe(even_f)]})

    # ---- where selection pushed ----
    heads = [("Reproduction: a child's share", [("a", 0), ("b", 1)]),
             ("Staking", [("score", 6), ("spread", 7), ("all in", 8), ("revolutionary a", 9),
                          ("revolutionary b", 10)]),
             ("Joining a child, handing connections over", [("join", 2), ("do not", 3), ("hand over", 11),
                                                           ("keep", 12)])]
    panels, means = [], {}
    for title, rows_ in heads:
        cat = [name for name, _ in rows_]
        groups = []
        for name, r in rows_:
            v = per_world(lambda d, r=r: d[f"out{r}"].mean())
            means[name] = {"evolved": describe(v), "founder": describe(per_world(lambda d, r=r: d[f"founder_out{r}"].mean()))}
            groups.append(v)
        panels.append(dict(title=title, x={"label": "", "categories": cat}, y={"label": "mean output", "min": -1.4, "max": 1.4},
                           series=dots(cat, groups, [BLUE] * len(cat)), legend=False,
                           guides=[{"axis": "y", "at": 0.0, "label": "founders: 0"}]))
    ch.grid(
        "outputs", panels, columns=1, title="Where selection pushed",
        caption="One dot per world: the mean, over its agents and their columns, of each of the brain's outputs "
                "that make decisions (Chapter 4). A founder's brain averages 0 on every one of them (the dashed "
                "line). A child's share is f(a, b), the share of the positive part of a in the positive parts "
                "of a and b ([The share function](../notes/share-function.md)): a below 0 and b above it means no child. "
                "Likewise for the revolutionary part of a stake; spread or all in follows whichever of its two "
                "outputs is larger.",
        recipe=recipe("The 26 baseline worlds that lived to the end.", FINAL,
                      [PROBE, "For each agent, the mean over its columns of output rows 0–14 (`World.heads`: "
                       "REPRO_FRACTION 0–1, LINK 2–3, LINK_MODE 4–5, BLOTTO 6, BLOTTO_MODE 7–8, REV_FRACTION 9–10, "
                       "HANDOVER 11–12, HANDOVER_MODE 13–14); then the mean over agents."]))
    ch.number("outputs", means)

    # ---- what changes a decision ----
    sens = []
    labels = [("messages", "without messages"), ("noise", "fresh random numbers"),
              ("richer", "neighbours twice as rich")]
    for key, title in labels:
        e = per_world(lambda d, k=key: (d[f"tv_{k}"] > 0).mean())
        f = per_world(lambda d, k=key: (d[f"founder_tv_{k}"] > 0).mean())
        sens.append(dict(title=title.capitalize(), x={"label": "", "categories": cats},
                         y={"label": "share of agents whose stakes change", "min": 0, "max": 0.6},
                         series=dots(cats, [e, f], [BLUE, GREY]), legend=False))
        ch.number(f"changed_{key}", [describe(e), describe(f)])
    ch.grid(
        "sensitivity", sens, columns=3, title="What changes a decision",
        caption="One dot per world: the share of its agents whose stakes would differ at all if one thing about "
                "their inputs were different — every message they read set to 0; the five random numbers of "
                "every column drawn again; or every neighbour's tokens, as the agent sees them, doubled. Blue: "
                "the agents' own brains; grey: " + _founder_note() + ".",
        recipe=recipe("The 26 baseline worlds that lived to the end.", FINAL,
                      [PROBE, "Change the inputs — rows 29–148 set to 0; rows 149–153 drawn again, uniform on "
                       "(−2, 2); row 2 of every neighbour's column replaced by ln(1 + 2·(e^x − 1)) — and decide again; "
                       "count the agents whose shares of stake differ in any candidate."]))

    # ---- what the brains listen to ----
    import gol_store as store
    cfg = store.load_config(alive[0].run_id)
    fgroups, fdist = founder_weights(cfg)
    neutral = float(np.mean(list(fgroups.values())) * np.sqrt(1.396))
    W = {s.run_id: weights(s.run_id) for s in alive}
    Wfix = {s.run_id: weights(s.run_id) for s in fixed}
    gnames = [name for name, _ in GROUPS]
    ch.figure(
        "listen", title="What a brain listens to",
        x={"label": "", "categories": gnames}, y={"label": "size of the weights reading it", "min": 0.4, "max": 0.85},
        series=dots(gnames, [[w["groups"][g] for w in W.values()] for g in gnames], [BLUE] * len(gnames)),
        guides=[{"axis": "y", "at": float(np.mean(list(fgroups.values()))), "label": "founders"},
                {"axis": "y", "at": neutral, "label": "mutation alone, at its balance"}], legend=False,
        caption="For each group of the brain's inputs ([What a brain sees](../notes/brain-inputs.md)) — whether "
                "the candidate is the agent itself; the agent's and the candidate's tokens and connections; the "
                "summaries of both neighbourhoods (around); the four messages, from writer to reader (I→me is what "
                "the agent wrote to itself, it→me what the candidate wrote to it, and so on); and the random "
                "numbers — the size of the first layer's weights that read it: for each input, the length of its column of 50 weights, averaged over the inputs of "
                "the group and the agents of the world; one dot per world. Lower dashed line: the same for "
                "founders' brains. Upper: what the changes of [How a brain changes](../notes/mutation.md) alone "
                "would bring every weight to, whatever it reads — its jitters adding variance, its resets "
                "drawing weights back — at the balance of the two.",
        recipe=recipe("The 26 baseline worlds that lived to the end.", FINAL,
                      ["Read the first weight matrix W0 (agents × 50 × 154) from the checkpoint; for each input "
                       "the norm of its column; average within each group of inputs and over agents.",
                       "Mutation alone: each change jitters a weight with probability 0.1 by a normal amount of "
                       "variance 0.2²/k and redraws it with probability 0.1 · 0.1 from variance 1/k; the variance v "
                       "at balance satisfies v = 0.99 (v + 0.004/k) + 0.01/k, so v = 1.396/k: √1.396 = 1.18 times "
                       "a founder's."]))
    ch.number("listen", {"evolved": {g: describe([w["groups"][g] for w in W.values()]) for g in gnames},
                         "changeless": {g: describe([w["groups"][g] for w in Wfix.values()]) for g in gnames},
                         "founders": fgroups, "neutral": neutral})

    # ---- how far apart brains are ----
    one = [float(np.median(w["parent"])) for w in W.values() if w["parent"]]
    two = [float(np.median(w["random"])) for w in W.values()]
    dcats = ["a genotype and its parent", "two agents of one world", "two founders"]
    ch.figure(
        "apart", title="How far apart brains are",
        x={"label": "", "categories": dcats}, y={"label": "root-mean-square difference of all weights (logarithmic)",
                                                 "log": True, "min": 0.003, "max": 0.5},
        series=dots(dcats, [one, two, fdist], [CYAN, BLUE, GREY]), legend=False,
        caption="The difference between two brains: the root mean square of the differences of all their "
                "15,795 numbers (weights and biases). Left: a genotype and its parent genotype, where both are "
                "alive at the end of a world — one change apart; one dot per world, the median of its pairs. "
                "Middle: two agents drawn at random from one world; one dot per world, the median of 3,000 pairs. "
                "Right: two fresh founders' brains, one dot per pair (200 pairs).",
        recipe=recipe("The 26 baseline worlds that lived to the end.", FINAL,
                      ["Concatenate every weight matrix and bias vector of a brain into one vector of 15,795 "
                       "numbers; the difference of two brains is the root mean square of the difference of their "
                       "vectors (`book_chapters.inner.weights`)."]))
    ch.number("apart", {"parent": describe(one), "world": describe(two), "founders": describe(fdist)})


# ---------------------------------------------------------------------------
# Chapter 19 · What agents say to each other
# ---------------------------------------------------------------------------

@chapter
def talk(ch: F.Chapter) -> None:
    alive = survivors(baseline())
    stats = {s.run_id: message_stats(s.run_id) for s in alive}
    e = [m["evolved"] for m in stats.values()]
    f = [m["founder"] for m in stats.values()]
    cats = ["to two of its readers", "by two writers"]
    ch.figure(
        "names", title="A name, not news",
        x={"label": "", "categories": cats}, y={"label": "root-mean-square difference of two messages (logarithmic)",
                                                "log": True, "min": 1e-5, "max": 1},
        series=dots(cats, [[m["to_two_readers"] for m in e], [m["by_two_writers"] for m in e]], [CYAN, BLUE]),
        legend=False,
        caption="For every agent of a world at its end, the message its brain writes to each of its candidates "
                "(the 30 message outputs, squashed by tanh into −1 to 1). Left: how different the messages one "
                "agent writes to two of its readers are; right: how different the messages of two agents are. "
                "One dot per world, each the median over its agents.",
        recipe=recipe("The 26 baseline worlds that lived to the end.", FINAL,
                      ["For every agent, its inputs as the engine builds them and tanh of output rows 15–44 of "
                       "its brain: one message per candidate.",
                       "Left: the root-mean-square difference between its messages to its first two candidates; "
                       "right: between its first message and the previous agent's (`book_chapters.inner.message_stats`)."]))
    ch.number("names", {"readers": describe([m["to_two_readers"] for m in e]),
                        "writers": describe([m["by_two_writers"] for m in e]),
                        "founder_readers": describe([m["to_two_readers"] for m in f]),
                        "founder_writers": describe([m["by_two_writers"] for m in f])})

    kcats = ["by genotype", "by shuffled genotype"]
    ecats = ["evolved brains", "founders' brains"]
    ch.grid(
        "what", [
            dict(title="Kin speak alike", x={"label": "", "categories": kcats},
                 y={"label": "share of writers' variety", "min": 0, "max": 1},
                 series=dots(kcats, [[m["kin"] for m in e], [m["kin_null"] for m in e]], [GREEN, GREY]), legend=False),
            dict(title="Numbers used", x={"label": "", "categories": ecats},
                 y={"label": "effective number of the 30", "min": 0, "max": 30},
                 series=dots(ecats, [[m["effective"] for m in e], [m["effective"] for m in f]], [BLUE, GREY]),
                 legend=False),
            dict(title="Explained by the state", x={"label": "", "categories": ecats},
                 y={"label": "share of the variety", "min": 0, "max": 0.5},
                 series=dots(ecats, [[m["explained"] for m in e], [m["explained"] for m in f]], [BLUE, GREY]),
                 legend=False)],
        columns=3, title="What the messages carry",
        caption="One dot per world. Left: of how much the agents' messages differ from one agent to the next, "
                "the share that lies between genotypes — 1 if agents of one genotype write exactly alike — "
                "against the same share with the genotypes shuffled among the agents (the median of 50 "
                "shuffles). Middle: the effective number of independent numbers the messages use, of their 30 "
                "(the participation ratio of the eigenvalues of their covariance). Right: the share of the "
                "messages' variety that a straight-line fit on the writer's and reader's tokens and connections "
                "(their logarithms) and whether the message is to itself accounts for. Blue: the agents' own "
                "brains; grey: " + _founder_note() + ".",
        recipe=recipe("The 26 baseline worlds that lived to the end.", FINAL,
                      ["As in the figure above, every message an agent's brain writes to each candidate.",
                       "Kin: each writer's mean message; between-genotype sum of squares of those means over their "
                       "total; the null shuffles the genotypes among writers.",
                       "Numbers used: (Σλ)² / Σλ² of the covariance eigenvalues λ of the 30 components.",
                       "Explained: least squares of all 30 components on [1, ln(1+τ_writer), ln(1+k_writer), "
                       "ln(1+τ_reader), ln(1+k_reader), self]; 1 − residual sum of squares / total."]))
    for key in ("kin", "kin_null", "effective", "explained", "saturated", "by_writer"):
        ch.number(key, {"evolved": describe([m[key] for m in e]), "founder": describe([m[key] for m in f])})

    hist_e = np.sum([m["values"] for m in e], axis=0).astype(float)
    hist_f = np.sum([m["values"] for m in f], axis=0).astype(float)
    centres = np.linspace(-1, 1, 41)
    centres = (centres[:-1] + centres[1:]) / 2
    ch.figure(
        "values", title="The numbers in a message",
        x={"label": "value of one of a message's 30 numbers", "min": -1, "max": 1},
        y={"label": "share of all numbers, per 0.05"},
        series=[line(centres, hist_e / hist_e.sum(), "evolved brains", BLUE, width=2),
                line(centres, hist_f / hist_f.sum(), "founders' brains", GREY, width=2, dash=[5, 4])],
        caption="Every one of the 30 numbers of every message written at the end of the 26 worlds, in bins 0.05 "
                "wide: by the agents' own brains (blue) and by " + _founder_note() + " (grey, dashed).",
        recipe=recipe("The 26 baseline worlds that lived to the end.", FINAL,
                      ["As above; count the numbers in 40 bins from −1 to 1 and divide by all numbers."]))
