#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assembly theory's numbers for small graphs: how many joining steps it takes to
build a graph from single links, reusing anything built before.

Assembly theory (Marshall et al. 2021; Sharma et al. 2023) builds an object from
basic blocks. For a graph the block is one link. A **join** takes two graphs
already made — a link, or anything built earlier — and lays them together,
making some of their nodes one node (Seet et al. 2025, Definition 3.2: "the
union of x and y plus an identification of a nontrivial set of vertices"). Their
links stay apart, so the result has as many links as the two together. Whatever
is made goes into the pool, and can be used again for free. The **assembly
index** a(G) is the fewest joins after which G is in the pool.

Three facts make it computable for small graphs:

- Every graph a shortest pathway makes is a connected piece of G (a join never
  removes a link), so the pool can be searched among G's connected pieces.
- A pathway is a set S of pieces that contains G, in which every piece of two
  links or more splits into two parts, each a single link or a member of S.
  Its length is |S|: each member is made by one join, and a member used twice is
  made once. So a(G) = min |S|.
- The links of the members form an addition chain (each a sum of two smaller
  ones), so a(G) ≥ l(m), the shortest addition chain of G's number of links m.
  And a(G) ≤ m − 1, one link at a time.

`assembly_index` finds the least S by a depth-first search from G down,
deepening the allowed length from l(m) until one is found. Two graphs are the
same piece when they are isomorphic; `canonical` names each by a form that is
equal exactly for isomorphic graphs (individualisation and refinement, as
nauty does, with swaps of twin nodes skipped).

Stars, paths and cycles of n links have the index l(n): their connected pieces
are stars, paths and paths, joined end to end, so a pathway is an addition
chain. `self_check()` tests these and a few more cases against a brute-force
search over pools.
"""
from __future__ import annotations

import itertools
from array import array
import math
from collections import Counter, defaultdict
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

#: A graph's canonical form: (number of nodes, its links as sorted pairs under the canonical labels).
Form = Tuple[int, Tuple[Tuple[int, int], ...]]


# ---------------------------------------------------------------------------
# Addition chains
# ---------------------------------------------------------------------------

_CHAIN: Dict[int, int] = {1: 0, 2: 1}


def chain_length(n: int) -> int:
    """
    l(n): the fewest additions that reach n from 1, each adding two numbers
    already reached (the same one twice allowed). l(7) = 4: 1, 2, 3, 4, 7.
    Exact, by iterative deepening; fast to a few thousand.
    """
    if n < 1:
        raise ValueError("an addition chain starts at 1")
    if n in _CHAIN:
        return _CHAIN[n]
    length = max(1, (n - 1).bit_length())          # ceil(log2 n)
    chain = [1]

    def grow(steps: int) -> bool:
        last = chain[-1]
        if last == n:
            return True
        left = steps - (len(chain) - 1)
        if left <= 0 or last << left < n:
            return False
        sums = sorted({a + b for i, a in enumerate(chain) for b in chain[i:] if last < a + b <= n}, reverse=True)
        for s in sums:
            if s << (left - 1) < n:
                break
            chain.append(s)
            if grow(steps):
                return True
            chain.pop()
        return False

    while not grow(length):
        length += 1
    _CHAIN[n] = length
    return length


def chain_bounds(n: int) -> Tuple[int, int]:
    """Bounds on l(n) without the search: ceil(log2 n), and the binary method's length."""
    if n <= 1:
        return 0, 0
    return (n - 1).bit_length(), n.bit_length() - 1 + bin(n).count("1") - 1


# ---------------------------------------------------------------------------
# Canonical forms
# ---------------------------------------------------------------------------

def _refine(colours: List[int], nbrs: List[List[int]]) -> List[int]:
    """
    Colour refinement: split every colour class by the colours of the
    neighbours, until nothing splits. The new colours are ranks of the
    signatures, so they depend only on the graph, not on the labels.
    """
    count = len(set(colours))
    while True:
        sigs = [(colours[v], tuple(sorted(colours[u] for u in nbrs[v]))) for v in range(len(colours))]
        rank = {s: i for i, s in enumerate(sorted(set(sigs)))}
        colours = [rank[s] for s in sigs]
        if len(rank) == count:
            return colours
        count = len(rank)


class TooSymmetric(RuntimeError):
    """The search for a canonical form met more labellings than it was allowed to try."""


def canonical(n: int, edges: Sequence[Tuple[int, int]], limit: Optional[int] = None) -> Form:
    """
    A form of the graph (nodes 0..n−1, these links) that is the same for two
    graphs exactly when they are isomorphic: the least list of relabelled links
    over every labelling that refinement and individualisation can reach. Nodes
    that are twins (the same neighbours, with or without each other) can be
    swapped without changing the graph, so only one of them is tried. With a
    `limit`, it gives up (TooSymmetric) after that many labellings.
    """
    nbrs: List[List[int]] = [[] for _ in range(n)]
    mask = [0] * n
    for u, v in edges:
        nbrs[u].append(v)
        nbrs[v].append(u)
        mask[u] |= 1 << v
        mask[v] |= 1 << u
    open_twins = Counter(mask)
    closed_twins = Counter(mask[v] | 1 << v for v in range(n))
    twin = []
    for v in range(n):
        if open_twins[mask[v]] > 1:
            twin.append(("o", mask[v]))
        elif closed_twins[mask[v] | 1 << v] > 1:
            twin.append(("c", mask[v] | 1 << v))
        else:
            twin.append(("v", v))
    best: List[Optional[Tuple[Tuple[int, int], ...]]] = [None]
    leaves = [0]

    def search(colours: List[int]) -> None:
        cells: Dict[int, List[int]] = defaultdict(list)
        for v, c in enumerate(colours):
            cells[c].append(v)
        if len(cells) == n:
            leaves[0] += 1
            if limit is not None and leaves[0] > limit:
                raise TooSymmetric(f"more than {limit} labellings of a graph of {n} nodes")
            form = tuple(sorted((min(colours[u], colours[v]), max(colours[u], colours[v])) for u, v in edges))
            if best[0] is None or form < best[0]:
                best[0] = form
            return
        target = min((len(vs), c) for c, vs in cells.items() if len(vs) > 1)[1]
        tried = set()
        for v in cells[target]:
            if twin[v] in tried:
                continue
            tried.add(twin[v])
            split = [2 * c + 1 for c in colours]
            split[v] = 2 * colours[v]
            search(_refine(split, nbrs))

    search(_refine([len(nbrs[v]) for v in range(n)], nbrs))
    return n, best[0] or ()


def form_of(edges: Iterable[Tuple[int, int]], limit: Optional[int] = None) -> Form:
    """The canonical form of a graph given by its links, whatever its node names."""
    edges = list(edges)
    names: Dict[int, int] = {}
    for u, v in edges:
        names.setdefault(u, len(names))
        names.setdefault(v, len(names))
    return canonical(len(names), [(names[u], names[v]) for u, v in edges], limit)


def links(form: Form) -> int:
    return len(form[1])


EDGE: Form = form_of([(0, 1)])


# ---------------------------------------------------------------------------
# The ways to split a graph into two connected parts
# ---------------------------------------------------------------------------

class _Pieces:
    """One graph's links, as bits, with what each link touches."""

    def __init__(self, form: Form) -> None:
        self.form = form
        self.edges = form[1]
        m = len(self.edges)
        self.full = (1 << m) - 1
        at: Dict[int, int] = defaultdict(int)
        for i, (u, v) in enumerate(self.edges):
            at[u] |= 1 << i
            at[v] |= 1 << i
        self.touch = [at[u] | at[v] for u, v in self.edges]

    def connected(self, mask: int) -> bool:
        if not mask:
            return False
        seen = frontier = mask & -mask
        while frontier:
            grown = 0
            while frontier:
                bit = frontier & -frontier
                frontier ^= bit
                grown |= self.touch[bit.bit_length() - 1]
            frontier = grown & mask & ~seen
            seen |= frontier
        return seen == mask

    def part(self, mask: int) -> Form:
        return form_of(e for i, e in enumerate(self.edges) if mask >> i & 1)

    def invariant(self, mask: int) -> Tuple[int, Tuple[int, ...]]:
        """A part's nodes and degrees: equal for isomorphic parts, and cheap."""
        degree: Counter = Counter()
        while mask:
            bit = mask & -mask
            mask ^= bit
            u, v = self.edges[bit.bit_length() - 1]
            degree[u] += 1
            degree[v] += 1
        return len(degree), tuple(sorted(degree.values()))


_INVARIANTS: Dict[Form, Tuple[int, Tuple[int, ...]]] = {}


def invariant(form: Form) -> Tuple[int, Tuple[int, ...]]:
    """A graph's nodes and degrees, as _Pieces.invariant gives them for a part."""
    if form in _INVARIANTS:
        return _INVARIANTS[form]
    degree: Counter = Counter()
    for u, v in form[1]:
        degree[u] += 1
        degree[v] += 1
    found = (len(degree), tuple(sorted(degree.values())))
    if len(form[1]) <= 16:                 # the small pieces recur; the large are let go
        _INVARIANTS[form] = found
    return found


def _twin_grids(form: Form) -> Tuple[List[List[List[int]]], List[int]]:
    """
    Groups of nodes with the same neighbours and no link between them (open
    twins), whose links can be shuffled among them without changing anything:
    for each group, its rows (one per node: the indices of its links, in the
    order of the shared neighbours); and the links in no group. Groups whose
    links meet are not both used, so each shuffle moves its own links only.
    """
    n, edges = form
    nb: Dict[int, set] = defaultdict(set)
    for u, v in edges:
        nb[u].add(v)
        nb[v].add(u)
    index = {e: i for i, e in enumerate(edges)}
    groups: Dict[FrozenSet[int], List[int]] = defaultdict(list)
    for v in range(n):
        groups[frozenset(nb[v])].append(v)
    candidates = []
    for common, nodes in groups.items():
        if len(nodes) < 2:
            continue
        cols = sorted(common)
        rows = [[index[(min(v, w), max(v, w))] for w in cols] for v in sorted(nodes)]
        candidates.append(rows)
    candidates.sort(key=lambda rows: -len(rows) * len(rows[0]))
    used: set = set()
    grids = []
    for rows in candidates:
        mine = {i for row in rows for i in row}
        if mine & used:
            continue
        used |= mine
        grids.append(rows)
    free = [i for i in range(len(edges)) if i not in used]
    return grids, free


def _grid_masks(rows: List[List[int]]) -> List[int]:
    """Every way to put a grid's links in the first part, up to shuffling its rows."""
    width = len(rows[0])
    patterns = range(1 << width)
    out = []
    for combo in itertools.combinations_with_replacement(patterns, len(rows)):
        mask = 0
        for row, pattern in zip(rows, combo):
            for j, i in enumerate(row):
                if pattern >> j & 1:
                    mask |= 1 << i
        out.append(mask)
    return out


class TooHard(RuntimeError):
    """The exact search would take more work than it was allowed."""


class OutOfWork(TooHard):
    """The search used up the work it was given."""


#: The most ways of splitting one graph the search will try; a graph with more is left to its bounds.
MASKS = 1 << 20


def split_count(form: Form) -> int:
    """How many subsets of its links split_masks tries for a graph: fewer than all, where twins are shuffled."""
    grids, free = _twin_grids(form)
    return math.prod([math.comb((1 << len(rows[0])) + len(rows) - 1, len(rows)) for rows in grids] + [1 << len(free)])


def split_masks(form: Form, naive: bool = False, limit: int = MASKS) -> Dict[Tuple[int, int], List[int]]:
    """
    Every split of a graph's links into two connected parts, up to the swaps of
    twin nodes, by the sizes of the parts (smaller first); each split as the
    links of its smaller part. `naive` tries every subset instead, to check.
    More than `limit` subsets to try raises TooHard.
    """
    pieces = _Pieces(form)
    m = len(form[1])
    if naive:
        choices: Iterable[int] = range(1, pieces.full)
    else:
        grids, free = _twin_grids(form)
        if split_count(form) > limit:
            raise TooHard(f"more than {limit} ways to split a graph of {m} links")
        options = [_grid_masks(rows) for rows in grids]
        options.append([sum(1 << free[j] for j in range(len(free)) if s >> j & 1) for s in range(1 << len(free))])
        choices = (sum(parts) for parts in itertools.product(*options))
    out: Dict[Tuple[int, int], set] = defaultdict(set)
    for mask in choices:
        if not mask or mask == pieces.full:
            continue
        p = bin(mask).count("1")
        q = m - p
        if p > q or (p == q and mask > pieces.full ^ mask):
            mask, p, q = pieces.full ^ mask, q, p
        if (p, q) in out and mask in out[(p, q)]:
            continue
        if pieces.connected(mask) and pieces.connected(pieces.full ^ mask):
            out[(p, q)].add(mask)
    return {k: array("Q", sorted(v)) for k, v in out.items()}


def sample_masks(form: Form, tries: int = 64) -> Dict[Tuple[int, int], List[int]]:
    """
    Some splits of a graph too large to split every way: every single link
    against the rest, and for each size, `tries` connected parts grown link by
    link from a random link (seeded by the graph's form, so the same graph
    always gets the same sample). A pathway found with these is a real
    pathway, so its length is an upper bound on the index.
    """
    import random
    import zlib
    pieces = _Pieces(form)
    m = len(form[1])
    rng = random.Random(zlib.crc32(repr(form).encode()))
    out: Dict[Tuple[int, int], set] = defaultdict(set)

    def add(mask: int) -> None:
        p = bin(mask).count("1")
        q = m - p
        if p > q or (p == q and mask > pieces.full ^ mask):
            mask, p, q = pieces.full ^ mask, q, p
        if pieces.connected(mask) and pieces.connected(pieces.full ^ mask):
            out[(p, q)].add(mask)

    for i in range(m):
        add(1 << i)
    for p in range(2, m // 2 + 1):
        for _ in range(tries):
            start = rng.randrange(m)
            mask = 1 << start
            border = pieces.touch[start] & ~mask
            while border and bin(mask).count("1") < p:
                options = [j for j in range(m) if border >> j & 1]
                j = rng.choice(options)
                mask |= 1 << j
                border = (border | pieces.touch[j]) & ~mask
            if bin(mask).count("1") == p:
                add(mask)
    return {k: array("Q", sorted(v)) for k, v in out.items()}


# ---------------------------------------------------------------------------
# The assembly index
# ---------------------------------------------------------------------------

class Assembler:
    """
    Assembly indices of graphs, exact where the search can finish, with what
    is learned on the way kept for the next graph: each piece's splits, and
    which sets of pieces are known not to be buildable within a number of
    joins.
    """

    def __init__(self) -> None:
        self.index: Dict[Form, int] = {EDGE: 0}
        self._masks: Dict[Form, Dict[Tuple[int, int], List[int]]] = {}
        self._sampled: Dict[Form, Dict[Tuple[int, int], List[int]]] = {}
        self._pieces: Dict[Form, _Pieces] = {}
        self._pairs: Dict[Tuple[Form, Tuple[int, int], bool], List[Tuple[Form, Form]]] = {}
        self._fails: Dict[FrozenSet[Form], int] = {}
        self._fails_sampled: Dict[FrozenSet[Form], int] = {}
        #: Every piece met, once: equal forms made again share one copy in memory.
        self._known: Dict[Form, Form] = {}
        #: The two parts of a split already given forms, by piece and split.
        self._parts: Dict[Tuple[Form, int], Tuple[Form, Form]] = {}
        #: For lazy_pairs: how far through a piece's splits of some sizes it got, and what it made.
        self._lazy: Dict[Tuple[Form, Tuple[int, int]], list] = {}
        #: How many times a piece too large to split every way was passed over: a "no" found
        #: while this grew proves nothing.
        self._unsure = 0
        self.searched = 0
        self._stop: Optional[int] = None

    # -- splits, made once per piece --

    def _spend(self, units: int) -> None:
        """Count work done: a search step is one unit, a part given a form two, 64 subsets tried one."""
        self.searched += units
        if self._stop is not None and self.searched > self._stop:
            raise OutOfWork("the search used up the work it was given")

    def _splits(self, x: Form, sampled: bool) -> Tuple[Dict[Tuple[int, int], List[int]], bool]:
        """A piece's splits, and whether they are only a sample (allowed when `sampled`)."""
        if x in self._masks:
            return self._masks[x], False
        if x not in self._sampled:
            try:
                limit = MASKS if not sampled else MASKS >> 4
                count = split_count(x)
                if count <= limit:
                    self._spend(count // 64)
                self._masks[x] = split_masks(x, limit=limit)
                return self._masks[x], False
            except TooHard:
                if not sampled:
                    raise
                self._sampled[x] = sample_masks(x)
        if not sampled:
            raise TooHard("too many ways to split")
        return self._sampled[x], True

    def _once(self, form: Form) -> Form:
        return self._known.setdefault(form, form)

    def lazy_pairs(self, x: Form, sizes: Tuple[int, int]) -> Iterable[Tuple[Form, Form]]:
        """
        The pairs of parts of `pairs`, given forms only as they are asked for;
        what was made is kept, and once every split is made the list is whole.
        """
        key = (x, sizes, False)
        if key in self._pairs:
            yield from self._pairs[key]
            return
        made = self._lazy.setdefault((x, sizes), [0, [], set()])
        pieces = self._pieces.setdefault(x, _Pieces(x))
        masks = self._masks[x][sizes]
        i = 0
        while True:
            if i < len(made[1]):
                yield made[1][i]
                i += 1
                continue
            if made[0] >= len(masks):
                break
            mask = masks[made[0]]
            made[0] += 1
            pair = self._parts.get((x, mask))
            if pair is None:
                self._spend(2)
                a, b = self._once(pieces.part(mask)), self._once(pieces.part(pieces.full ^ mask))
                pair = (a, b) if a <= b else (b, a)
            if pair not in made[2]:
                made[2].add(pair)
                made[1].append(pair)
        self._pairs[key] = sorted(made[1])
        self._lazy.pop((x, sizes), None)

    def matching_pairs(self, x: Form, sizes: Tuple[int, int], invariants: set) -> List[Tuple[Form, Form]]:
        """
        The splits of these sizes with a part whose nodes and degrees are among
        `invariants`, or with two parts of equal nodes and degrees: the only
        ones that can reuse a piece. The other parts are never given forms.
        """
        if (x, sizes, False) in self._pairs:
            return [(a, b) for a, b in self._pairs[(x, sizes, False)]
                    if a == b or invariant(a) in invariants or invariant(b) in invariants]
        pieces = self._pieces.setdefault(x, _Pieces(x))
        seen = set()
        for mask in self._masks[x][sizes]:
            ia, ib = pieces.invariant(mask), pieces.invariant(pieces.full ^ mask)
            if ia == ib or ia in invariants or ib in invariants:
                pair = self._parts.get((x, mask))
                if pair is None:
                    self._spend(2)
                    a, b = self._once(pieces.part(mask)), self._once(pieces.part(pieces.full ^ mask))
                    pair = self._parts[(x, mask)] = (a, b) if a <= b else (b, a)
                seen.add(pair)
        return sorted(seen)

    def pairs(self, x: Form, sizes: Tuple[int, int], sample: bool = False) -> List[Tuple[Form, Form]]:
        key = (x, sizes, sample)
        if key not in self._pairs:
            pieces = self._pieces.setdefault(x, _Pieces(x))
            seen = set()
            for mask in (self._sampled if sample else self._masks)[x][sizes]:
                self._spend(2)
                a, b = self._once(pieces.part(mask)), self._once(pieces.part(pieces.full ^ mask))
                seen.add((a, b) if a <= b else (b, a))
            self._pairs[key] = sorted(seen)
        return self._pairs[key]

    # -- bounds --

    @staticmethod
    def lower_bound(need: Iterable[Form]) -> int:
        """
        The fewest pieces a pathway that still needs these pieces can have: each
        of them, and for each, the pieces of the sizes its own addition chain
        needs that none of the others has.
        """
        need = list(need)
        sizes = sorted({links(x) for x in need})
        extra = 0
        for k, s in enumerate(sizes):
            extra = max(extra, chain_length(s) - 1 - k)
        return len(need) + extra

    # -- the search --

    def feasible(self, need: FrozenSet[Form], budget: int, sampled: bool = False) -> bool:
        """
        Can these pieces, and everything they are made of, be made in `budget`
        joins? A piece too large to split every way is passed over (and the
        "no" is then unsure); or, `sampled`, it is split in a sample of ways
        only. Either way a "yes" is a real pathway.
        """
        if not need:
            return True
        if self._fails.get(need, -1) >= budget or self.lower_bound(need) > budget:
            return False
        if sampled and self._fails_sampled.get(need, -1) >= budget:
            return False
        self._spend(1)
        unsure = self._unsure
        x = max(need, key=lambda f: (links(f), f))
        rest = need - {x}
        try:
            masks, partial = self._splits(x, sampled)
        except OutOfWork:
            raise
        except TooHard:
            self._unsure += 1
            return False
        rest_sizes = Counter(links(f) for f in rest)

        def hopeful(sizes: Tuple[int, int]) -> int:
            p, q = sizes
            new = len(rest)
            for s in {p, q}:
                if s > 1 and not rest_sizes[s]:
                    new += 1
            sizes_after = sorted({links(f) for f in rest} | {s for s in (p, q) if s > 1})
            extra = 0
            for k, s in enumerate(sizes_after):
                extra = max(extra, chain_length(s) - 1 - k)
            return new + extra

        def all_new(sizes: Tuple[int, int]) -> int:
            """The bound when neither part is a piece already needed, nor the two parts alike."""
            new = sum(1 for s in sizes if s > 1)
            sizes_after = sorted({links(f) for f in rest} | {s for s in sizes if s > 1})
            return len(rest) + new + max([chain_length(s) - 1 - k for k, s in enumerate(sizes_after)], default=0)

        groups = sorted(masks, key=lambda s: (hopeful(s), s[1] - s[0]))
        for sizes in groups:
            if hopeful(sizes) > budget - 1:
                continue
            if x not in self._masks:
                found = self.pairs(x, sizes, partial)
            else:
                # The splits that reuse something come first: a part alike to a needed piece, or
                # two parts alike. Parts with other nodes and degrees are not given forms here.
                found = self.matching_pairs(x, sizes, {invariant(f) for f in rest})
            options = []
            for a, b in found:
                after = rest | {f for f in (a, b) if links(f) > 1}
                options.append((len(after), after))
            options.sort(key=lambda o: o[0])

            def candidates(options=options, sizes=sizes):
                yield from (after for _, after in options)
                if x in self._masks and all_new(sizes) <= budget - 1:
                    # Splits into two new pieces fit as well: the others, given forms one by one,
                    # so that a pathway found early spares the rest.
                    tried = {frozenset(after) for _, after in options}
                    for pair in self.lazy_pairs(x, sizes):
                        after = rest | {f for f in pair if links(f) > 1}
                        if frozenset(after) not in tried:
                            tried.add(frozenset(after))
                            yield after

            for after in candidates():
                if self.lower_bound(after) > budget - 1:
                    continue
                if self.feasible(frozenset(after), budget - 1, sampled):
                    return True
        if sampled or partial:
            self._fails_sampled[need] = max(self._fails_sampled.get(need, -1), budget)
        elif self._unsure == unsure:
            self._fails[need] = max(self._fails.get(need, -1), budget)
        return False

    def bounds(self, form: Form, work: Optional[int] = None) -> Tuple[int, int]:
        """
        The assembly index as an interval: deepening from l(m), the first
        length at which a pathway is found is an upper bound, and every length
        ruled out completely below it raises the lower bound. With `work`, the
        search stops after that many steps. Equal ends: the index, kept.
        """
        if form in self.index:
            return self.index[form], self.index[form]
        m = links(form)
        if m <= 1:
            return 0, 0
        lo, proven = chain_length(m), True
        self._stop = None if work is None else self.searched + work
        try:
            for depth in range(chain_length(m), m):
                unsure = self._unsure
                if self.feasible(frozenset([form]), depth):
                    if proven:
                        self.index[form] = depth
                    return lo, depth
                if proven and self._unsure == unsure:
                    lo = depth + 1
                else:
                    proven = False
        except TooHard:
            return lo, m - 1
        finally:
            self._stop = None
        if proven:
            self.index[form] = m - 1
        return lo, m - 1

    def assembly_index(self, form: Form) -> int:
        """The fewest joins that make this graph from single links; TooHard if that cannot be settled."""
        lo, hi = self.bounds(form)
        if lo != hi:
            raise TooHard(f"the index of a graph of {links(form)} links lies between {lo} and {hi}")
        return lo

    def forget(self, larger_than: int) -> None:
        """Let go of the splits of pieces with more links than this: they are many, and seldom met again."""
        for store in (self._masks, self._sampled):
            for x in [x for x in store if links(x) > larger_than]:
                del store[x]
        for key in [k for k in self._pairs if links(k[0]) > larger_than]:
            del self._pairs[key]
        self._pieces = {x: p for x, p in self._pieces.items() if links(x) <= larger_than}
        self._known = {x: x for x in self._known if links(x) <= larger_than}
        self._parts = {k: v for k, v in self._parts.items() if links(k[0]) <= larger_than}
        self._lazy = {k: v for k, v in self._lazy.items() if links(k[0]) <= larger_than}
        for fails in (self._fails, self._fails_sampled):
            for need in [k for k in fails if any(links(x) > larger_than for x in k)]:
                del fails[need]

    def upper_bound(self, form: Form, low: int, high: int, work: int) -> int:
        """
        A shorter pathway than `high`, a length known to be possible, found by
        the sampled search: halving the gap to `low` while a pathway is found
        within `work` steps.
        """
        while low < high:
            mid = (low + high) // 2
            self._stop = self.searched + work
            try:
                found = self.feasible(frozenset([form]), mid, sampled=True)
            except TooHard:
                found = False
            finally:
                self._stop = None
            if found:
                high = mid
            else:
                low = mid + 1
        return high


def star_index(degree: int) -> int:
    """A star of d links: every piece is a star, joined at the centre, so its index is l(d)."""
    return chain_length(degree) if degree >= 1 else 0


#: The largest number whose shortest addition chain is searched for; beyond it, bounds.
CHAIN_SEARCH = 4096


def chain_range(n: int) -> Tuple[int, int]:
    """l(n) as an interval: exact (both ends equal) up to CHAIN_SEARCH, bounds beyond."""
    if n <= CHAIN_SEARCH:
        exact = chain_length(n)
        return exact, exact
    return chain_bounds(n)


def ego_index(asm: "Assembler", key: Tuple, nodes: int, m: int,
              edges: Optional[List[Tuple[int, int]]] = None, work: Optional[int] = None) -> Tuple[int, int]:
    """
    The assembly index of an ego network as an interval (lo, hi), equal ends
    where it is exact. A shape of at most the cap's links: the exact search. A
    star, a centre and its leaves with no link among them: l(d). A larger
    shape of at most SEARCH_LINKS links, given its links and some `work`: the
    exact search, if it ends within that work, and otherwise its bounds and a
    pathway found by the sampled search. Otherwise, for a centre joined to d
    neighbours with t links among them: at least l(d + t), and at most
    l(d) + t, the star built first and the t links added one join each.
    """
    if key[0] == "form":
        return asm.bounds(key[1])
    degree = nodes - 1
    if m == degree:
        return chain_range(degree)
    lo, hi = chain_range(m)[0], chain_range(degree)[1] + (m - degree)
    if edges is not None and work is not None and m <= SEARCH_LINKS:
        try:
            form = form_of(edges, Shapes.LIMIT)
        except TooSymmetric:
            return lo, hi
        found_lo, found_hi = asm.bounds(form, work)
        lo, hi = max(lo, found_lo), min(hi, found_hi)
        if lo < hi:
            hi = asm.upper_bound(form, lo, hi, work)
        asm.forget(FORGET)
    return lo, hi


#: Pieces with more links than this have their splits let go after each large shape (Assembler.forget).
FORGET = 16
#: The largest shape, in links, the search is tried on at all; a larger one keeps its bounds.
SEARCH_LINKS = 40


# ---------------------------------------------------------------------------
# Ego networks, and how many agents share each shape
# ---------------------------------------------------------------------------

def ego_links(adj: Dict[int, set], agent: int) -> List[Tuple[int, int]]:
    """An agent's ego network: its links to its neighbours, and the links among them."""
    nb = sorted(adj[agent])
    out = [(agent, b) for b in nb]
    for i, b in enumerate(nb):
        near = adj[b]
        out.extend((b, c) for c in nb[i + 1:] if c in near)
    return out


class Shapes:
    """
    Graphs sorted into shapes, one shape for each class of isomorphic graphs,
    with the number of copies of each. A graph of at most `cap` links is named
    by its canonical form. A larger one is first sorted by its numbers of nodes
    and links, its degrees and a Weisfeiler–Lehman hash; graphs that share all
    of these are told apart by their canonical forms, or, where that search
    meets too many labellings, by networkx's VF2++ isomorphism test. So the
    shapes are exact at every size.
    """

    #: Labellings the canonical form of a large graph may try before VF2++ decides instead.
    LIMIT = 5000

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.copies: Counter = Counter()
        #: For each shape: its nodes, its links, and its links as one example has them.
        self.example: Dict[Tuple, Tuple[int, int, List[Tuple[int, int]]]] = {}
        self._buckets: Dict[Tuple, List[Tuple[Tuple, List[Tuple[int, int]], Optional[Form]]]] = defaultdict(list)

    def add(self, edges: List[Tuple[int, int]]) -> Tuple:
        nodes = len({v for e in edges for v in e})
        if len(edges) <= self.cap:
            key: Tuple = ("form", form_of(edges))
        else:
            key = self._large(edges, nodes)
        self.copies[key] += 1
        if key not in self.example:
            self.example[key] = (nodes, len(edges), edges)
        return key

    def _large(self, edges: List[Tuple[int, int]], nodes: int) -> Tuple:
        import warnings
        import networkx as nx
        g = nx.Graph(edges)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wl = nx.weisfeiler_lehman_graph_hash(g, iterations=3)
        bucket_key = (nodes, len(edges), tuple(sorted(d for _, d in g.degree())), wl)
        bucket = self._buckets[bucket_key]
        if not bucket:
            key = ("large", bucket_key, 0)
            bucket.append((key, edges, None))
            return key
        mine = self._form(edges)
        for i, (key, other, form) in enumerate(bucket):
            if form is None and mine is not None:
                form = self._form(other)
                bucket[i] = (key, other, form)
            if mine is not None and form is not None:
                if mine == form:
                    return key
            elif nx.vf2pp_is_isomorphic(nx.Graph(other), g):
                return key
        key = ("large", bucket_key, len(bucket))
        bucket.append((key, edges, mine))
        return key

    def _form(self, edges: List[Tuple[int, int]]) -> Optional[Form]:
        try:
            return form_of(edges, self.LIMIT)
        except TooSymmetric:
            return None


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

def assembly(indices: Sequence[int], copies: Sequence[int], total: Optional[int] = None) -> float:
    """
    The assembly of an ensemble (Sharma et al. 2023, equation 1):
    A = Σ_i e^{a_i} (n_i − 1) / N_T, over the distinct objects i, with a_i the
    index and n_i the copy number of each, and N_T all the objects counted.
    """
    total = total if total is not None else sum(copies)
    return sum(math.exp(a) * (n - 1) for a, n in zip(indices, copies)) / total if total else 0.0


# ---------------------------------------------------------------------------
# A check against what is known, and against a search that tries everything
# ---------------------------------------------------------------------------

def brute_force_index(form: Form) -> int:
    """
    The index by breadth-first search over pools: every set of connected pieces
    of the graph that can be made one join at a time, smallest first. Slow;
    for checking graphs of a few links only.
    """
    pieces = _Pieces(form)
    m = len(form[1])
    if m <= 1:
        return 0
    subs = {}
    for mask in range(1, pieces.full + 1):
        if pieces.connected(mask):
            subs.setdefault(pieces.part(mask), mask)
    made_of = {}
    for f in subs:
        if links(f) > 1:
            own = _Pieces(f)
            made_of[f] = {(own.part(k), own.part(own.full ^ k))
                          for masks in split_masks(f, naive=True).values() for k in masks}
    level = {frozenset()}
    for joins in range(1, m):
        nxt = set()
        for pool in level:
            have = pool | {EDGE}
            for f, ways in made_of.items():
                if f in pool:
                    continue
                if any(a in have and b in have for a, b in ways):
                    if f == form:
                        return joins
                    nxt.add(pool | {f})
        level = nxt
    return m - 1


def _path(n: int) -> Form:
    return form_of([(i, i + 1) for i in range(n)])


def _star(n: int) -> Form:
    return form_of([(0, i) for i in range(1, n + 1)])


def _cycle(n: int) -> Form:
    return form_of([(i, (i + 1) % n) for i in range(n)])


def _complete(n: int) -> Form:
    return form_of([(i, j) for i in range(n) for j in range(i + 1, n)])


def _wheel(n: int) -> Form:
    return form_of([(0, i) for i in range(1, n + 1)] + [(i, i % n + 1) for i in range(1, n + 1)])


#: Known indices: stars, paths and cycles of n links take l(n) joins; K4, K5 and the wheel of six
#: spokes are built from two equal halves (two paths of three, two cycles of five, two spiders with
#: three legs of two links), so they too meet the bound l(m).
KNOWN = ([(f"star of {n}", _star(n), chain_length(n)) for n in range(1, 17)]
         + [(f"path of {n}", _path(n), chain_length(n)) for n in range(1, 17)]
         + [(f"cycle of {n}", _cycle(n), chain_length(n)) for n in range(3, 17)]
         + [("K4", _complete(4), 3), ("K5", _complete(5), 4), ("wheel of 6 spokes", _wheel(6), 4)])


def self_check(brute_up_to: int = 6, seed: int = 1, samples: int = 60) -> Dict[str, int]:
    """
    The checks the page relies on, run by tests/test_book.py:
    the canonical form tells random graphs apart exactly as an isomorphism test
    does; the twin-aware splits find every split a plain enumeration finds; the
    known indices come out; and random small graphs get the index a search over
    every pool gives.
    """
    import random
    import networkx as nx

    rng = random.Random(seed)

    def random_graph(nodes: int, m: int) -> List[Tuple[int, int]]:
        while True:
            g = nx.gnm_random_graph(nodes, m, seed=rng.randrange(10 ** 9))
            if nx.is_connected(g):
                return list(g.edges())

    graphs = []
    for _ in range(samples):
        nodes = rng.randint(3, 8)
        m = rng.randint(nodes - 1, min(nodes * (nodes - 1) // 2, 12))
        graphs.append(random_graph(nodes, m))
    # Isomorphic copies under shuffled names must get the same form, and only they.
    forms = []
    for edges in graphs:
        names = list(range(max(max(e) for e in edges) + 1))
        rng.shuffle(names)
        a, b = form_of(edges), form_of([(names[u], names[v]) for u, v in edges])
        assert a == b, f"two labellings of one graph got two forms: {edges}"
        forms.append((a, nx.Graph(edges)))
    for (fa, ga), (fb, gb) in itertools.combinations(forms, 2):
        if (fa == fb) != nx.is_isomorphic(ga, gb):
            raise AssertionError(f"the form and the isomorphism test disagree: {fa} {fb}")
    # Twin-aware splits: the same pairs of parts as every subset gives.
    for form, _ in forms[:30]:
        def pairs(masks):
            p = _Pieces(form)
            return {tuple(sorted((p.part(k), p.part(p.full ^ k)))) for ks in masks.values() for k in ks}
        assert pairs(split_masks(form)) == pairs(split_masks(form, naive=True)), f"splits differ: {form}"
    for extra in (_star(7), _wheel(5), _complete(4)):
        def pairs(masks, form=extra):
            p = _Pieces(form)
            return {tuple(sorted((p.part(k), p.part(p.full ^ k)))) for ks in masks.values() for k in ks}
        assert pairs(split_masks(extra)) == pairs(split_masks(extra, naive=True))
    asm = Assembler()
    for name, form, want in KNOWN:
        got = asm.assembly_index(form)
        assert got == want, f"{name}: index {got}, expected {want}"
    brute = 0
    for form, _ in forms:
        if links(form) <= brute_up_to:
            assert asm.assembly_index(form) == brute_force_index(form), f"search and brute force differ: {form}"
            brute += 1
    return {"graphs": len(forms), "known": len(KNOWN), "brute": brute}


if __name__ == "__main__":
    print(self_check())
