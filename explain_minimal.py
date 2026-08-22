#!/usr/bin/env python3
"""
GraphOfLife, cut down to the smallest thing that still is it.

This is the script the Explanation tab walks through. It has no settings, no
storage, no statistics and no options — every mechanic is simply on, because a
branch you can switch off is a branch you have to explain. Copy it, run it,
change it:

    python3 explain_minimal.py

The full version lives in GraphOfLifeSimple.py. It is the same algorithm in the
same order; what it adds is the parts a research tool needs and a reader does
not — configuration, checkpoints, recording, three kinds of brain, and the
switches that make each mechanic optional. The observation vector here is
shorter than the real one, so this will not reproduce a particular run of it
token for token. The structure is the point, and the structure is identical.

Nothing here is trained and nothing is optimised. Agents hold tokens, spend
them on children, and fight over position. What survives, survives.
"""

from __future__ import annotations

import random

import numpy as np

TOKENS = 500          # the whole economy, and it never changes
AGENTS = 40           # how many the world starts with
NEIGHBOURS = 4        # how many each is joined to at the start
HIDDEN = [24, 16]     # the brain's hidden layers
MESSAGE = 3           # numbers written to each neighbour
NOISE = 2             # random inputs, so identical agents can still differ
ITERATIONS = 200


# ---------------------------------------------------------------------------
# The brain
# ---------------------------------------------------------------------------

class Brain:
    """
    A feed-forward network, never trained.

    It is made once at random, copied when an agent reproduces or conquers, and
    jittered when it is copied. That is the whole of how behaviour changes:
    there is no gradient anywhere in this file.

    One column of inputs goes in per thing being looked at — the agent itself
    and each of its neighbours — and one column of outputs comes back, so an
    agent reads its whole neighbourhood in a single pass.
    """

    def __init__(self, n_in: int, n_out: int):
        sizes = [n_in] + HIDDEN + [n_out]
        self.weights = [np.random.randn(b, a) * 0.5
                        for a, b in zip(sizes, sizes[1:])]
        self.biases = [np.random.randn(b) * 0.5 for b in sizes[1:]]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x is (inputs, columns); the answer is (outputs, columns)."""
        for w, b in zip(self.weights, self.biases):
            x = 1.0 / (1.0 + np.exp(-(w @ x + b[:, None])))
        return x

    def copy(self) -> "Brain":
        clone = object.__new__(Brain)
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = [b.copy() for b in self.biases]
        return clone

    def mutate(self) -> None:
        """A sparse jitter. Most weights are untouched; a few move a little."""
        for w in self.weights:
            mask = np.random.random(w.shape) < 0.1
            w += mask * np.random.randn(*w.shape) * 0.2
        for b in self.biases:
            mask = np.random.random(b.shape) < 0.1
            b += mask * np.random.randn(*b.shape) * 0.2


# ---------------------------------------------------------------------------
# Reading the outputs
# ---------------------------------------------------------------------------
#
# Every decision is a pair of numbers: one says what to do, the other says how
# to read the first. Take the better option, or draw between them in proportion
# to their scores. That second number is part of the brain, so how boldly a
# lineage decides is itself something evolution settles.

def decide(yes: float, no: float, mode_a: float, mode_b: float) -> bool:
    """A yes/no choice, read sharply or gambled on."""
    if mode_a > mode_b:
        if yes == no:
            return random.random() < 0.5
        return yes > no
    total = yes + no
    return random.random() < (yes / total if total > 0 else 0.5)


def pick(scores: np.ndarray, sample: bool) -> int:
    """Which of several — the best, or drawn in proportion."""
    if not sample:
        return int(np.argmax(scores))
    total = float(scores.sum())
    if total <= 0:
        return int(np.random.randint(len(scores)))
    return int(np.random.choice(len(scores), p=scores / total))


def share_of(a: float, b: float) -> float:
    """Two outputs read as a fraction between 0 and 1."""
    total = a + b
    return (a / total) if total > 0 else 0.0


def apportion(scores: np.ndarray, total: int) -> np.ndarray:
    """
    Split a whole number of tokens by score, losing none.

    Largest remainder: floor everything, then hand the leftovers to whoever was
    cut hardest. Tokens are conserved, so they cannot be rounded away.
    """
    if total <= 0 or scores.sum() <= 0:
        out = np.zeros(len(scores), dtype=int)
        if total > 0:
            out[int(np.argmax(scores))] = total
        return out
    exact = scores / scores.sum() * total
    whole = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - whole))[:total - int(whole.sum())]:
        whole[i] += 1
    return whole


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

class World:
    """
    A graph where every node is an agent: some tokens, and a brain.

    The graph is kept as a set of neighbours per agent. Nothing here needs more
    than that, and it makes every rule below readable as what it does to who
    knows whom.
    """

    def __init__(self):
        # ---- the starting graph ----
        # A ring where everyone is joined to their nearest few, then a fifth of
        # the links redrawn to somewhere random. Short paths everywhere, but
        # still mostly local — the shape most real networks have.
        self.adj = {i: set() for i in range(AGENTS)}
        for i in range(AGENTS):
            for j in range(1, NEIGHBOURS // 2 + 1):
                self.link(i, (i + j) % AGENTS)
        for i in list(self.adj):
            for v in list(self.adj[i]):
                if random.random() < 0.2:
                    self.unlink(i, v)
                    self.link(i, random.randrange(AGENTS))

        # ---- the tokens ----
        # A fixed pile, split evenly, and the total never changes again.
        self.tokens = {i: TOKENS // AGENTS for i in range(AGENTS)}
        for i in range(TOKENS - sum(self.tokens.values())):
            self.tokens[i % AGENTS] += 1

        # ---- the brains ----
        self.brains = {i: Brain(self.n_inputs(), self.n_outputs())
                       for i in range(AGENTS)}
        self.inbox = {i: {} for i in range(AGENTS)}
        self.next_id = AGENTS

    # -- the graph, as three small operations --------------------------------

    def link(self, a: int, b: int) -> None:
        if a != b and a in self.adj and b in self.adj:
            self.adj[a].add(b)
            self.adj[b].add(a)

    def unlink(self, a: int, b: int) -> None:
        self.adj.get(a, set()).discard(b)
        self.adj.get(b, set()).discard(a)

    def remove(self, a: int) -> None:
        for v in list(self.adj.get(a, ())):
            self.unlink(a, v)
        self.adj.pop(a, None)
        self.tokens.pop(a, None)
        self.brains.pop(a, None)
        self.inbox.pop(a, None)

    # -- what an agent sees, and what it can say ------------------------------

    def n_inputs(self) -> int:
        # is-it-me, my tokens, its tokens, my degree, its degree,
        # what it wrote to me last phase, and some noise
        return 5 + MESSAGE + NOISE

    def n_outputs(self) -> int:
        # 2 how much of me goes into a child
        # 2 + 2 link to this neighbour? and how to read that
        # 2 + 2 hand this edge to the child? and how to read that
        # 1 + 2 how much to stake here, and whether to spread or go all in
        # 2 how much of that stake is a revolt
        # MESSAGE what I write to it
        return 15 + MESSAGE

    def observe(self, u: int, targets: list[int]) -> np.ndarray:
        """
        One column per thing being looked at, all of it in one pass.

        Tokens and degrees go in logged, because what matters is the order of
        magnitude — the difference between 1 token and 10 is everything, and
        between 500 and 510 is nothing.
        """
        x = np.zeros((self.n_inputs(), len(targets)))
        for col, v in enumerate(targets):
            note = self.inbox.get(u, {}).get(v, [0.0] * MESSAGE)
            x[:, col] = [
                1.0 if v == u else 0.0,
                np.log1p(self.tokens.get(u, 0)),
                np.log1p(self.tokens.get(v, 0)),
                np.log1p(len(self.adj.get(u, ()))),
                np.log1p(len(self.adj.get(v, ()))),
                *note,
                *np.random.random(NOISE),
            ]
        return self.brains[u].forward(x)

    def write_messages(self, u: int, targets: list[int], y: np.ndarray,
                       outbox: dict) -> None:
        """
        A short note to every neighbour, and one to itself.

        The note to itself is the only memory an agent has: nothing else
        survives from one phase to the next. Nothing forces a message to mean
        anything — whatever they come to signal is whatever survives.
        """
        rows = np.tanh(y[15:15 + MESSAGE, :])
        for col, v in enumerate(targets):
            outbox.setdefault(v, {})[u] = rows[:, col].tolist()

    # -- phase one: reproduction ---------------------------------------------

    def reproduction(self) -> None:
        """
        Agents spend their own tokens on children. Nothing else moves an edge.

        One pass: each agent observes, writes its messages, and acts, all in
        the same breath. So the messages it reads were written last phase, and
        the ones it writes now will be read next.
        """
        outbox: dict = {}
        for u in sorted(self.adj):
            if self.tokens.get(u, 0) <= 0:
                continue
            targets = [u] + sorted(self.adj[u])
            y = self.observe(u, targets)
            self.write_messages(u, targets, y, outbox)

            # ---- how much of me goes into a child ----
            # The parent pays the full price out of its own pocket; the child
            # starts with exactly what was spent. No tokens are created.
            fraction = share_of(float(y[0].mean()), float(y[1].mean()))
            spend = int(self.tokens[u] * fraction)
            if spend < 1:
                continue

            child = self.next_id
            self.next_id += 1
            self.adj[child] = set()
            self.tokens[u] -= spend
            self.tokens[child] = spend
            self.inbox[child] = {}

            # ---- the child inherits a mutated copy ----
            # This is the only way a new brain ever enters the world.
            self.brains[child] = self.brains[u].copy()
            self.brains[child].mutate()
            self.link(child, u)

            # ---- which of my neighbours does the child meet ----
            for col, v in enumerate(targets[1:], start=1):
                if decide(y[2, col], y[3, col], y[4, col], y[5, col]):
                    self.link(child, v)
                    # ---- handover: give the edge away instead of copying ----
                    # The parent drops that connection and the child takes its
                    # place, so a lineage can pass on position, not just tokens.
                    if decide(y[6, col], y[7, col], y[8, col], y[9, col]):
                        self.unlink(u, v)

        self.deliver(outbox)
        self.cleanup()

    # -- phase two: the game --------------------------------------------------

    def game(self) -> None:
        """
        Everyone stakes their whole pile on themselves and their neighbours,
        and whoever commits most to a node takes it.

        One look each, exactly like reproduction. What comes back decides both
        what the agent says to its neighbours and where it puts its tokens.
        """
        outbox: dict = {}

        # ---- everyone stakes at once ----
        staked = {v: {} for v in self.adj}      # node -> who staked what
        revolt = {v: {} for v in self.adj}      # node -> how much of it revolts
        flow = {}                               # edge -> tokens that crossed it
        for u in sorted(self.adj):
            targets = [u] + sorted(self.adj[u])
            y = self.observe(u, targets)
            # Written even by an agent with nothing left to stake: it is still
            # here, and it still has something to say.
            self.write_messages(u, targets, y, outbox)

            pot = self.tokens.get(u, 0)
            if pot <= 0:
                continue

            # Spread the pile by score, or put all of it on one node. Which of
            # those it does is the brain's own choice, not a rule.
            scores = np.asarray(y[10, :], dtype=float)
            mode = y[11:13, :].mean(axis=1)
            if mode[0] > mode[1]:
                amounts = apportion(scores, pot)
            else:
                amounts = np.zeros(len(targets), dtype=int)
                amounts[pick(scores, sample=False)] = pot

            self.tokens[u] = 0
            for col, v in enumerate(targets):
                if amounts[col] <= 0:
                    continue
                staked[v][u] = staked[v].get(u, 0) + int(amounts[col])
                # ---- part of a stake is flagged as a revolt ----
                share = share_of(float(y[13, col]), float(y[14, col]))
                revolt[v][u] = int(amounts[col] * share)
                if v != u:
                    flow[frozenset((u, v))] = flow.get(frozenset((u, v)), 0) + int(amounts[col])

        # ---- links nobody used are cut ----
        # Here, next to the staking that decided it, rather than further down:
        # the flow it reads is complete the moment the loop above ends, and
        # nothing between the two touches an edge.
        for a, b in [tuple(e) for e in self.adj_edges() if flow.get(frozenset(e), 0) == 0]:
            self.unlink(a, b)

        # ---- who takes each node ----
        winners = {}
        for v in sorted(self.adj):
            if staked[v]:
                winners[v] = resolve(staked[v], revolt[v])
            self.tokens[v] = sum(staked[v].values())

        # ---- the winner moves in ----
        # The node stays; whatever was thinking in it does not. This is the
        # only place a brain is ever selected.
        for v, winner in winners.items():
            if winner != v:
                self.brains[v] = self.brains[winner].copy()

        self.deliver(outbox)
        self.cleanup()

        # ---- everyone mutates ----
        # Not only the newborns and not only the winners. Every brain still
        # standing, every iteration. This is the whole engine of variation, and
        # it comes after the clearing-up so that brains about to be removed are
        # not jittered on their way out.
        for brain in self.brains.values():
            brain.mutate()

    def adj_edges(self) -> list:
        seen = set()
        for u in self.adj:
            for v in self.adj[u]:
                if (v, u) not in seen:
                    seen.add((u, v))
        return list(seen)

    def deliver(self, outbox: dict) -> None:
        """
        Messages written this phase become what is read next.

        Every inbox is replaced, not just the ones written to, so an agent
        nobody addressed reads nothing rather than reading something old.
        """
        for v in self.inbox:
            self.inbox[v] = outbox.get(v, {})

    # -- after every phase ----------------------------------------------------

    def cleanup(self) -> None:
        """
        Two removals, in this order, and it runs after both phases.

        Anyone holding nothing is gone. Then, of what is left, only the largest
        connected piece survives — and the second rule bites because of the
        first: a group hanging off the rest through one agent comes adrift when
        that agent starves, however healthy the group itself is.

        Everything the dead held is scattered over the survivors, so the total
        still comes to TOKENS.
        """
        estate = 0
        for u in [u for u in self.adj if self.tokens.get(u, 0) <= 0]:
            estate += self.tokens.get(u, 0)
            self.remove(u)

        for piece in components(self.adj)[1:]:          # everything but the largest
            for u in piece:
                estate += self.tokens.get(u, 0)
                self.remove(u)

        survivors = sorted(self.adj)
        for _ in range(estate):
            self.tokens[random.choice(survivors)] += 1

    def step(self) -> None:
        self.reproduction()
        self.game()


# ---------------------------------------------------------------------------
# Who wins a node
# ---------------------------------------------------------------------------

def resolve(staked: dict, revolt: dict) -> int:
    """
    The biggest stake usually wins. Sometimes the small ones combine and take
    it instead.

    The largest single staker is the HEGEMON. Against it stands the MOB: every
    other agent that flagged part of its stake as a revolt. The mob is sorted
    weakest first and walked upward, gathering a lower class as it goes. At
    each rung the question is whether that lower class now outweighs everyone
    still above it plus the hegemon.

    At the first rung where it does, the revolution carries, and the node goes
    to the STRONGEST staker in that rung — not to a random member of the
    crowd. Ties at that exact amount are split by drawing; nothing else is.

    So a crowd of small stakers can take a node from someone who outspent every
    one of them individually, and the best-placed of them collects it.
    """
    hegemon = max(staked, key=lambda a: (staked[a], a))
    mob = sorted(((a, t) for a, t in revolt.items() if a != hegemon and t > 0),
                 key=lambda pair: pair[1])
    if not mob:
        return hegemon

    total_mob = sum(t for _, t in mob)
    lower = 0
    i = 0
    while i < len(mob):
        rung_amount = mob[i][1]
        rung = []
        while i < len(mob) and mob[i][1] == rung_amount:   # equals rise together
            rung.append(mob[i][0])
            lower += mob[i][1]
            i += 1
        if lower > total_mob - lower + staked[hegemon]:
            return random.choice(rung)
    return hegemon


def components(adj: dict) -> list:
    """Connected pieces, largest first."""
    seen, pieces = set(), []
    for start in adj:
        if start in seen:
            continue
        piece, stack = [], [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            piece.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        pieces.append(piece)
    return sorted(pieces, key=len, reverse=True)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    world = World()
    for i in range(ITERATIONS):
        world.step()
        if not world.adj:
            print(f"iteration {i + 1}: everyone is gone")
            break
        print(f"iteration {i + 1:4d}   {len(world.adj):4d} agents   "
              f"{len(world.adj_edges()):4d} links   "
              f"{sum(world.tokens.values()):5d} tokens")
