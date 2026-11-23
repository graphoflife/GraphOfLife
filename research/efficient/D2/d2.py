"""D2: exact small games. A team of n agents (one joint plan) against one outsider, one phase.
Rules follow GraphOfLife.py: blotto_phase (every agent stakes ALL tokens on itself/neighbours; node keeps
every staked token: new_tokens[v] = incoming_totals[v]; a node with no offers gets 0 and starves),
_resolve_winner (largest stake, ties random among the tied agents), _lottery (win chance = stake/total),
_pooled + _claims (stakes pooled by side, owner's side keeps the node; equal pools: random).
Solver: double oracle (full best responses over ALL plans, so the gap is certified) with regret matching
on the restricted game. numpy only.
usage: python d2.py [outfile.json] [filter-substring]
"""
import os, sys, json, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

RULES = ["largest", "lottery", "pool"]

def compositions(t, k):
    """all ways to split t whole tokens over k targets"""
    if k == 1:
        yield (t,); return
    for a in range(t + 1):
        for rest in compositions(t - a, k - 1):
            yield (a,) + rest

def motif(kind, n, attach):
    V = n + 1  # node n is the outsider
    nb = {v: set() for v in range(V)}
    def link(a, b): nb[a].add(b); nb[b].add(a)
    for i in range(n):
        for j in range(i + 1, n):
            if kind == "clique" or j == i + 1:
                link(i, j)
    if attach == "one":
        link(0, n)
    else:
        for i in range(n): link(i, n)
    return V, nb

def plans_of(v, nb, tokens, V):
    targets = [v] + sorted(nb[v])
    comps = list(compositions(tokens, len(targets)))
    A = np.zeros((len(comps), V), dtype=np.int16)
    for r, c in enumerate(comps):
        for t, a in zip(targets, c): A[r, t] += a
    return A

class Game:
    def __init__(self, kind, n, attach, w, outs):
        self.V, nb = motif(kind, n, attach)
        self.n, self.w = n, w
        self.t_out = n * w if outs == "A" else w
        agent = [plans_of(i, nb, w, self.V) for i in range(n)]
        shape = [len(a) for a in agent]
        self.P = int(np.prod(shape))
        self.Q_plans = plans_of(n, nb, self.t_out, self.V)
        self.Q = len(self.Q_plans)
        idx = np.indices(shape, dtype=np.int32).reshape(n, -1) if self.P < 3_000_000 else None
        S = np.stack([agent[i][idx[i]] for i in range(n)], 0)       # n,P,V
        self.T = S.sum(0).astype(np.int16)
        self.M = S.max(0).astype(np.int16)
        self.cM = ((S == self.M[None]) & (self.M[None] > 0)).sum(0).astype(np.int16)
        self.total_team = n * w
        self.shape, self.agent = shape, agent
        self.S = S if self.P <= 50000 else None
        self.idx = idx

    def p_team(self, rule, T, M, cM, O):
        """chance the team wins each node; broadcast arrays"""
        if rule == "largest":
            return np.where(M > O, 1.0, np.where(M < O, 0.0, cM / (cM + 1.0)))
        if rule == "lottery":
            tot = T + O
            return np.where(tot > 0, T / np.maximum(tot, 1), 0.0)
        if rule == "pool":
            return (T > O) + 0.5 * (T == O)
        raise ValueError(rule)

    def pay(self, rule, pi, qi, outsider=False):
        """team's expected tokens, matrix len(pi) x len(qi); with outsider=True the outsider's instead"""
        Tm, Mm, cm = self.T[pi], self.M[pi], self.cM[pi]
        O = self.Q_plans[qi]
        res = np.empty((len(pi), len(qi)))
        step = max(1, 4_000_000 // max(1, len(qi) * self.V))
        for a in range(0, len(pi), step):
            T = Tm[a:a+step, None, :]; M = Mm[a:a+step, None, :]; c = cm[a:a+step, None, :]
            p = self.p_team(rule, T, M, c, O[None])
            tot = T + O[None]
            res[a:a+step] = (tot * (1 - p if outsider else p)).sum(2)
        return res

def regret_matching(A, rounds, plus=False, seed=0):
    """zero-sum: row player maximises x'Ay. Returns average strategies."""
    m, k = A.shape
    Rx = np.zeros(m); Ry = np.zeros(k)
    sx = np.zeros(m); sy = np.zeros(k)
    for t in range(1, rounds + 1):
        px = np.maximum(Rx, 0); s = px.sum(); x = px / s if s > 0 else np.full(m, 1.0 / m)
        py = np.maximum(Ry, 0); s = py.sum(); y = py / s if s > 0 else np.full(k, 1.0 / k)
        u = A @ y; v = x @ A
        Rx += u - x @ u
        Ry += (x @ u) - v          # column player minimises
        if plus:
            np.maximum(Rx, 0, out=Rx); np.maximum(Ry, 0, out=Ry)
        wgt = t if plus else 1.0
        sx += wgt * x; sy += wgt * y
    return sx / sx.sum(), sy / sy.sum()

def solve(g, rule, tol=0.008, max_iter=300, deadline=None):
    rng = np.random.RandomState(1)
    ST = [int(rng.randint(g.P))]; SO = [int(rng.randint(g.Q))]
    allP = np.arange(g.P); allQ = np.arange(g.Q)
    best = None
    rounds = 5000
    for it in range(max_iter):
        A = g.pay(rule, np.array(ST), np.array(SO))
        if A.size == 1:
            x = np.ones(1); y = np.ones(1)
        else:
            x, y = regret_matching(A, rounds, plus=True)
        # certified bounds with full best responses
        u = g.pay(rule, allP, np.array(SO)) @ y      # team payoff of every team plan vs outsider mix
        upper = u.max(); pbr = int(u.argmax())
        v = x @ g.pay(rule, np.array(ST), allQ)      # team payoff of x vs every outsider plan
        lower = v.min(); qbr = int(v.argmin())
        gap = upper - lower
        if best is None or gap < best["gap"]:
            best = dict(gap=gap, lower=lower, upper=upper, ST=list(ST), SO=list(SO), x=x.copy(), y=y.copy())
        if gap < tol or (deadline and time.time() > deadline):
            break
        grew = False
        if pbr not in ST: ST.append(pbr); grew = True
        if qbr not in SO: SO.append(qbr); grew = True
        if not grew: rounds = min(rounds * 2, 200000)
    # final: plain regret matching, 200,000 rounds, on the final support (spec); keep it if better
    if best["gap"] > 0.0005 and len(best["ST"]) * len(best["SO"]) > 1:
        A = g.pay(rule, np.array(best["ST"]), np.array(best["SO"]))
        x, y = regret_matching(A, 200000, plus=False)
        u = g.pay(rule, allP, np.array(best["SO"])) @ y; upper = u.max()
        v = x @ g.pay(rule, np.array(best["ST"]), allQ); lower = v.min()
        if upper - lower < best["gap"]:
            best.update(gap=upper - lower, lower=lower, upper=upper, x=x, y=y)
    return best

def describe(g, best, top=3):
    """words: the team's and the outsider's heaviest plans"""
    out = {}
    order = np.argsort(-best["x"])[:top]
    team = []
    for o in order:
        p = best["ST"][o]
        rows = [[int(g.agent[i][g.idx[i, p]][v]) for v in range(g.V)] for i in range(g.n)]
        team.append((round(float(best["x"][o]), 3), rows))   # rows[i][v] = tokens agent i puts on node v
    order = np.argsort(-best["y"])[:top]
    outs = [(round(float(best["y"][o]), 3), [int(a) for a in g.Q_plans[best["SO"][o]]]) for o in order]
    return team, outs

def check_conservation(g, rule, trials=300):
    rng = np.random.RandomState(5)
    pi = rng.randint(g.P, size=trials); qi = rng.randint(g.Q, size=trials)
    err = 0.0
    for a, b in zip(pi, qi):
        t = g.pay(rule, np.array([a]), np.array([b]))[0, 0]
        o = g.pay(rule, np.array([a]), np.array([b]), outsider=True)[0, 0]
        err = max(err, abs(t + o - (g.total_team + g.t_out)))
    return err

def cases():
    for kind, ns in (("path", (1, 2, 3, 4)), ("clique", (2, 3, 4))):
        for attach in ("one", "all"):
            for w in (2, 3):
                for outs in ("A", "B"):
                    for n in ns:
                        yield kind, n, attach, w, outs

if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else "d2_result.json"
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    res = json.load(open(outfile)) if os.path.exists(outfile) else {}
    for kind, n, attach, w, outs in cases():
        name = f"{kind}-n{n}-{attach}-w{w}-{outs}"
        if filt not in name or name in res: continue
        g = Game(kind, n, attach, w, outs)
        cells = g.P * g.Q
        entry = {"team_plans": g.P, "outsider_plans": g.Q, "cells": cells, "tokens_team": n * w, "tokens_out": g.t_out}
        for rule in RULES:
            t0 = time.time()
            b = solve(g, rule, deadline=time.time() + 150)
            val = (b["lower"] + b["upper"]) / 2
            team, outs_ = describe(g, b)
            entry[rule] = dict(value=val, per_token=val / (n * w), lower=b["lower"], upper=b["upper"],
                               gap=b["gap"], secs=round(time.time() - t0, 1), support=[len(b["ST"]), len(b["SO"])],
                               conservation_err=check_conservation(g, rule), team_plan=team, outsider_plan=outs_)
            print(name, rule, f"val/tok={val/(n*w):.4f} gap={b['gap']:.4f} P={g.P} Q={g.Q} {entry[rule]['secs']}s", flush=True)
        res[name] = entry
        json.dump(res, open(outfile, "w"))
