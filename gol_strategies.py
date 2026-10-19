#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategies written by hand, as brains a world can carry (Chapters 51 and 94).

A strategy reads exactly what a brain reads — the input rows, one column per
candidate, the agent's own first — and writes exactly what a brain writes: the
output rows the engine turns into a child, its links, the stakes and the
messages. So it plugs into a world where a brain does, spreads as a brain does,
into a child and into a node its owner takes, and can do nothing an evolved
brain could not.

There are two kinds:

  rules     a policy in Python: `turtle`, `raider`, `lone`, `factory`, what
            Chapter 51 planted
  weights   a brain of the world's own shape whose weights were set by hand:
            `raider-weights`, Chapter 94's raider, 12 of B2's 16 units

and `resident`, a frozen copy of one evolved brain, the control.

Teammates know each other by what they write: the first four numbers of every
message a strategy sends are a fixed signature, and the fifth carries a number
it wants its teammates to know. Every copy of a strategy carries the
strategy's own brain id (STRATEGY_IDS), so a patch can be counted wherever its
copies go. None of them ever mutates, and a strategy breeds true: a child of
one is the same strategy, whatever its mate. A resident's child by a mate
that is a strategy mixes with it unit by unit, as with any mate, which only a
strategy with weights can be; so only those are planted in worlds whose
brains are mixed at birth. A world carrying a rule cannot be checkpointed:
a rule has no weights to keep.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional

import numpy as np

#: Brain ids at or above this belong to a strategy; each strategy has its own.
STRATEGY_BASE = 1_900_000_000
SIGNATURE = np.array([0.9, -0.9, 0.9, -0.9])
SIGN = 4                            # message numbers taken by the signature
CAP = 19                            # distances are sent as d / 20, d at most this
EPS = 0.02                          # how near the signature a teammate's message must be


def atanh(v):
    return np.arctanh(np.clip(v, -0.999, 0.999))


class View:
    """What one look shows a strategy, read off the input rows."""

    def __init__(self, cfg, X):
        lay = cfg.input_layout()
        row = lambda name: X[lay[name][0]:lay[name][1], :]
        self.n = X.shape[1]
        self.me_tokens = float(np.expm1(row("own_tokens")[0, 0]))
        self.tokens = np.expm1(row("its_tokens")[0, :])
        self.is_self = row("self")[0, :] > 0.5
        heard = row("message_from_it")            # what each candidate wrote to me
        self.mate = np.all(np.abs(heard[:SIGN, :] - SIGNATURE[:, None]) < EPS, axis=0) & ~self.is_self
        self.word = np.round(heard[SIGN, :] * 20) if heard.shape[0] > SIGN else np.zeros(self.n)
        self.enemy = ~self.mate & ~self.is_self


class Writer:
    """Builds the output rows from plain decisions."""

    def __init__(self, cfg, n):
        from GraphOfLifeSimple import build_heads
        self.h = build_heads(cfg)
        self.cfg = cfg
        self.Y = np.zeros((cfg.n_outputs(), n))
        self.child(0.0)
        self.no_links()
        self.revolt(0.0)
        self.say(0)

    def child(self, share: float) -> None:
        """Give this share of my tokens to a child (0: none)."""
        s = self.h["REPRO_FRACTION"]
        self.Y[s.start, :], self.Y[s.start + 1, :] = share, 1.0 - share

    def no_links(self) -> None:
        L, M = self.h["LINK"], self.h["LINK_MODE"]
        self.Y[L.start, :], self.Y[L.start + 1, :] = 0.0, 1.0
        self.Y[M.start, :], self.Y[M.start + 1, :] = 0.0, 1.0     # decide, do not draw

    def link(self, cols) -> None:
        """Join a newborn to these candidates."""
        L = self.h["LINK"]
        self.Y[L.start, cols], self.Y[L.start + 1, cols] = 1.0, 0.0

    def stake(self, amounts) -> None:
        """Spread the staked tokens in proportion to `amounts` (all zero: everything at home)."""
        a = np.maximum(0.0, np.asarray(amounts, dtype=float))
        if a.sum() <= 0:
            a = np.zeros(len(a))
            a[0] = 1.0
        self.Y[self.h["BLOTTO"], :] = a
        m = self.h["BLOTTO_MODE"]
        self.Y[m.start, :], self.Y[m.start + 1, :] = 1.0, 0.0        # spread, as given

    def revolt(self, share: float) -> None:
        if "REV_FRACTION" in self.h:
            r = self.h["REV_FRACTION"]
            self.Y[r.start, :], self.Y[r.start + 1, :] = share, 1.0 - share

    def say(self, word: float) -> None:
        start = self.h["MESSAGE_START"]
        m = self.cfg.message_amount
        vec = np.zeros(m)
        vec[:SIGN] = SIGNATURE
        if m > SIGN:
            vec[SIGN] = min(word, CAP) / 20.0
        self.Y[start:start + m, :] = atanh(vec)[:, None]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def turtle(cfg, X):
    """Everything at home; a child of a third when rich; the child joined to its parent alone."""
    v, w = View(cfg, X), Writer(cfg, X.shape[1])
    w.stake(np.eye(v.n)[0])
    if v.me_tokens >= 30:
        w.child(1 / 3)
    return w.Y


def raider(cfg, X):
    """All on the weakest stranger next door if that can take it, else everything at home."""
    v, w = View(cfg, X), Writer(cfg, X.shape[1])
    staked = 0.7 * v.me_tokens                     # B2 keeps at least 30% at home anyway
    target = None
    if v.enemy.any():
        j = int(np.argmin(np.where(v.enemy, v.tokens, np.inf)))
        if staked > 1.5 * v.tokens[j]:
            target = j
    w.stake(np.eye(v.n)[target if target is not None else 0])
    if v.me_tokens >= 30:
        w.child(1 / 3)
        w.link(np.flatnonzero(v.mate | v.is_self))
    return w.Y


def lone(cfg, X):
    """The raider without a team: no password, every neighbour a stranger, children joined to nobody chosen."""
    v, w = View(cfg, X), Writer(cfg, X.shape[1])
    w.Y[w.h["MESSAGE_START"]:w.h["MESSAGE_START"] + cfg.message_amount, :] = 0.0
    others = ~v.is_self
    target = None
    if others.any():
        j = int(np.argmin(np.where(others, v.tokens, np.inf)))
        if 0.7 * v.me_tokens > 1.5 * v.tokens[j]:
            target = j
    w.stake(np.eye(v.n)[target if target is not None else 0])
    if v.me_tokens >= 30:
        w.child(1 / 3)
    return w.Y


def factory(cfg, X):
    """
    A team with a pipeline. Every member knows how far it is from the team's
    frontier — 0 if a stranger is next door, else one more than the nearest
    teammate says — and writes it to its teammates. The interior pumps its
    tokens downhill, towards the frontier; the frontier spends everything it can
    stake on the weakest stranger next door when that can take it, and otherwise
    holds. The interior breeds when rich, and joins its children to teammates.
    """
    v, w = View(cfg, X), Writer(cfg, X.shape[1])
    frontier = bool(v.enemy.any())
    if frontier:
        d = 0
    else:
        heard = v.word[v.mate]
        d = int(min(CAP, (heard.min() if heard.size else CAP - 1) + 1))
    w.say(d)
    staked = 0.7 * v.me_tokens
    amounts = np.zeros(v.n)
    if frontier:
        j = int(np.argmin(np.where(v.enemy, v.tokens, np.inf)))
        if staked > 1.5 * v.tokens[j]:
            amounts[j] = 1.0
    elif v.mate.any():
        nearest = np.where(v.mate, v.word, np.inf)
        amounts[int(np.argmin(nearest))] = 1.0
    w.stake(amounts)
    if not frontier and v.me_tokens >= 30:
        w.child(1 / 3)
        w.link(np.flatnonzero(v.mate | v.is_self))
    return w.Y


RULES: Dict[str, Callable] = {"turtle": turtle, "raider": raider, "lone": lone, "factory": factory}


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

#: The raider's score weights: takeable, weak, password, self (see `raider_weights`).
A, D, P, Q = 40.0, 1.0, 10.0, 5.0
WALL = 10000.0                  # steepness of the password walls: 200 steps inside the band, and no
                                # remainder outside it that could tip a choice between equal targets
SHUT = 30000.0                  # what the self row takes off every password wall
TAKE = 6000.0                   # steepness of the takeable unit: 3 against 1 lies 0.003 past the margin
STEEP = 1500.0                  # steepness of the rich unit: 29 and 30 tokens lie 0.033 apart in log1p
MARGIN = 0.69                   # "can be taken": ln(1 + own) - ln(1 + its) beyond this


def raider_weights(cfg, margin: float = MARGIN):
    """
    The raider as a brain of the world's shape, one hidden layer of at least 12 units
    (Chapter 94). Unit 0 is the self row; 1-8 the password, two walls per number, which
    the self row shuts, since what an agent heard from itself is a phase old and a node
    just won or an agent just born has not spoken yet; 9 whether the candidate can be
    taken, a line just below ln 2 in log1p space, where the rule's own line is straight in
    tokens; 10 how weak the candidate is; 11 whether the agent has 30 tokens or more.

    The stake score, all-in on the highest, keeps three orders: a stranger that can be
    taken above home even when its messages match three of the password's four numbers
    (A - 3P > Q + D), home above a stranger that cannot be taken (Q > D), home above any
    teammate (Q + 4P > A + D). The child's share is the float16 just above a third, since
    the one below would give a parent of 30 tokens a child of 9, below B2's minimum.
    """
    from GraphOfLifeSimple import BRAIN_KINDS, build_heads
    if len(cfg.hidden_layers) != 1 or cfg.hidden_layers[0] < 12:
        raise ValueError("the raider's weights need one hidden layer of at least 12 units")
    if cfg.message_amount < SIGN:
        raise ValueError(f"the raider's password needs messages of at least {SIGN} numbers")
    lay = cfg.input_layout()
    heads = build_heads(cfg)
    brain = BRAIN_KINDS[cfg.brain_kind](cfg, STRATEGY_IDS["raider-weights"])
    n_in, n_hid, n_out = cfg.n_inputs(), cfg.hidden_layers[0], cfg.n_outputs()
    W1, b1 = np.zeros((n_hid, n_in)), np.zeros((n_hid, 1))
    W2, b2 = np.zeros((n_out, n_hid)), np.zeros((n_out, 1))

    self_row, own, its = lay["self"][0], lay["own_tokens"][0], lay["its_tokens"][0]
    heard = lay["message_from_it"][0]
    W1[0, self_row], b1[0] = 40.0, -20.0
    for i, s in enumerate(SIGNATURE):
        lo, hi = 1 + 2 * i, 2 + 2 * i
        W1[lo, heard + i], b1[lo] = WALL, -WALL * (s - EPS)
        W1[hi, heard + i], b1[hi] = WALL, -WALL * (s + EPS)
        W1[lo, self_row] = W1[hi, self_row] = -SHUT
    W1[9, own], W1[9, its], b1[9] = TAKE, -TAKE, -TAKE * margin
    W1[10, its], b1[10] = -1.0, 3.0
    W1[11, own], b1[11] = STEEP, -STEEP * 0.5 * (np.log1p(29) + np.log1p(30))

    blotto = heads["BLOTTO"]
    W2[blotto, 9], W2[blotto, 10], W2[blotto, 0] = A, D, Q
    for i in range(SIGN):
        W2[blotto, 1 + 2 * i], W2[blotto, 2 + 2 * i] = -P, P
    b2[heads["BLOTTO_MODE"].start + 1] = 1.0            # all-in on the best
    rf = heads["REPRO_FRACTION"]
    third = float(np.float16(1 / 3) + np.float16(2 ** -12))
    W2[rf.start, 11], W2[rf.start + 1, 11], b2[rf.start + 1] = third, -third, 1.0
    link, lmode = heads["LINK"], heads["LINK_MODE"]
    W2[link.start, 0] = 1.0
    for i in range(SIGN):
        W2[link.start, 1 + 2 * i], W2[link.start, 2 + 2 * i] = 0.25, -0.25
    b2[link.start + 1] = 0.5
    b2[lmode.start + 1] = 1.0                           # decide, do not draw
    if "REV_FRACTION" in heads:
        b2[heads["REV_FRACTION"].start + 1] = 1.0
    start = heads["MESSAGE_START"]
    b2[start:start + SIGN, 0] = atanh(SIGNATURE)

    brain.weights = [W1.astype(brain.dtype), W2.astype(brain.dtype)]
    brain.biases = [b1.astype(brain.dtype), b2.astype(brain.dtype)]
    for w in brain.weights + brain.biases:
        if not np.all(np.isfinite(w)):
            raise ValueError("a weight the brain's number format cannot hold")
    return brain


WEIGHTS: Dict[str, Callable] = {"raider-weights": raider_weights}

#: Every strategy's own brain id; `resident` is the control, a frozen evolved brain.
STRATEGY_IDS: Dict[str, int] = {name: STRATEGY_BASE + k for k, name in
                                enumerate(["resident", *RULES, *WEIGHTS])}


def is_strategy(brain_id: int) -> bool:
    return brain_id >= STRATEGY_BASE


# ---------------------------------------------------------------------------
# A strategy as a brain
# ---------------------------------------------------------------------------

class StrategyBrain:
    """A brain that is a strategy: a rule, or a frozen brain. It never mutates and breeds true."""

    __slots__ = ("cfg", "brain_id", "parent_brain_id", "policy", "frozen", "weights", "biases")

    def __init__(self, cfg, brain_id: int, policy: Optional[Callable] = None, frozen: Any = None):
        self.cfg, self.brain_id, self.parent_brain_id = cfg, brain_id, -1
        self.policy, self.frozen = policy, frozen
        self.weights = frozen.weights if frozen is not None else []
        self.biases = frozen.biases if frozen is not None else []

    def forward(self, x):
        return self.frozen.forward(x) if self.frozen is not None else self.policy(self.cfg, np.asarray(x, float))

    def copy(self) -> "StrategyBrain":
        b = StrategyBrain(self.cfg, self.brain_id, self.policy, self.frozen)
        b.parent_brain_id = self.parent_brain_id
        return b

    def mixed_with(self, other, rng) -> "StrategyBrain":
        return self.copy()

    def same_weights(self, other) -> bool:
        return isinstance(other, StrategyBrain) and other.brain_id == self.brain_id

    def mutate(self, brain_id, rng) -> bool:
        return False


def brain_for(cfg, strategy: str, resident: Any = None) -> StrategyBrain:
    """A brain playing `strategy`; for `resident`, a frozen copy of the brain given."""
    sid = STRATEGY_IDS[strategy]
    if strategy == "resident":
        return StrategyBrain(cfg, sid, frozen=resident.copy())
    if strategy in RULES:
        if cfg.message_amount < SIGN:
            raise ValueError(f"{strategy} signs its messages with {SIGN} numbers, and this world's "
                             f"messages hold {cfg.message_amount}")
        if cfg.mix_at_birth:
            raise ValueError(f"{strategy} is a rule, which has no weights to mix with a mate's: "
                             f"in a world whose brains are mixed at birth only a strategy with "
                             f"weights can be planted")
        return StrategyBrain(cfg, sid, policy=RULES[strategy])
    return StrategyBrain(cfg, sid, frozen=WEIGHTS[strategy](cfg))


def patch_of(world, size: float, rng: np.random.RandomState) -> List[int]:
    """
    The agents nearest a random agent, step by step: `size` agents, or a share of the
    world when `size` is below 1.
    """
    nodes = sorted(world.G.nodes())
    want = int(size) if size >= 1 else max(1, int(round(size * len(nodes))))
    start = nodes[rng.randint(len(nodes))]
    seen, order, queue = {start}, [], deque([start])
    while queue and len(order) < want:
        u = queue.popleft()
        order.append(u)
        for x in sorted(world.G[u]):
            if x not in seen:
                seen.add(x)
                queue.append(x)
    return order


def plant(world, strategy: str, size: float, place: int) -> List[int]:
    """
    Give a patch of `world` the strategy, as Chapter 51 did: on seed `place`, an agent
    at random and its nearest neighbours. The control, `resident`, gives the patch
    frozen copies of one patch member's brain, drawn at random; the draw is made for
    every strategy, so a strategy and its control land on the same patch.
    """
    rng = np.random.RandomState(1000 + place)
    patch = patch_of(world, size, rng)
    resident = world.brains[patch[rng.randint(len(patch))]]
    brain = brain_for(world.cfg, strategy, resident)
    for u in patch:
        world.brains[u] = brain.copy()
    return patch


def fill(world, strategy: str) -> None:
    """Every agent of `world` given the strategy."""
    brain = brain_for(world.cfg, strategy, next(iter(world.brains.values())))
    for u in list(world.brains):
        world.brains[u] = brain.copy()


# ---------------------------------------------------------------------------
# What a trial says (gol_trial.py runs them; gol_plan reads them from a plan)
# ---------------------------------------------------------------------------

#: What each kind of trial must say, and may.
TRIAL_NEEDS = {"contest": ("run", "T", "home", "a", "b", "seed"),
               "invasion": ("run", "T", "era", "strategy", "size", "place")}
TRIAL_MAY = {"contest": (), "invasion": ("background", "settle")}


def check_trial(trial: Dict[str, Any]) -> None:
    """Refuse a trial that says too little, or what no trial reads, or names no strategy there is."""
    kind = trial.get("kind")
    if kind not in TRIAL_NEEDS:
        raise ValueError(f"a trial is a {' or an '.join(TRIAL_NEEDS)}, not {kind!r}")
    missing = [k for k in TRIAL_NEEDS[kind] if k not in trial]
    unknown = sorted(set(trial) - {"kind", *TRIAL_NEEDS[kind], *TRIAL_MAY[kind]})
    if missing or unknown:
        raise ValueError(f"a {kind} says {', '.join(TRIAL_NEEDS[kind])}"
                         + (f"; {', '.join(missing)} missing" if missing else "")
                         + (f"; {', '.join(unknown)} is not something it reads" if unknown else ""))
    for name in ("strategy", "background"):
        if trial.get(name) is not None and trial[name] not in STRATEGY_IDS:
            raise ValueError(f"no strategy called {trial[name]!r}: there are {', '.join(STRATEGY_IDS)}")
    if kind == "contest" and trial["home"] not in (trial["a"], trial["b"]):
        raise ValueError(f"a contest is played on one of its two eras' worlds, not on {trial['home']}")

