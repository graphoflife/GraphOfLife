#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The spectral gap: one number for how hard a graph is to cut in half.

λ₂ is the second-smallest eigenvalue of the normalised Laplacian, and it is the
closest thing this project has to a single measure of *decentralisation*. Near
zero and the graph has a cheap cut somewhere — two halves joined by very
little. Bounded away from zero and every way of splitting it is expensive,
which is what an expander is.

It matters here for a reason beyond description. The Flow modules view finds
communities by compressing a random walk, and a partition of an expander is an
artefact: the method always returns *something*, and without λ₂ there is no way
to tell a real division from a line drawn through an indivisible graph. Cheeger's
inequality is what connects the two —

    h(G)² / 2  ≤  λ₂  ≤  2 h(G)

— so λ₂ bounds the sparsest cut from both sides, and a small λ₂ is a promise
that a good cut exists while a large one is a proof that none does.

Computed on the **largest connected component**. The definition gives exactly
zero for a disconnected graph, which is true and useless: one stray pair of
agents would answer for the whole population. Cleanup keeps only the largest
component anyway, so in practice this is the whole graph.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

#: Lanczos vectors. The cost is one pass over the edges each, plus keeping them
#: orthogonal to one another, which is what actually dominates.
#:
#: Power iteration was tried first and is not adequate here. It separates the
#: top two eigenvalues at a rate set by the space between them, and on a graph
#: that is nearly in two pieces there is no space. Measured on a 35,000-node
#: frame, its estimate *halved* every time the iteration count doubled — 0.038,
#: 0.019, 0.010, 0.0045 at 25, 50, 100, 200 passes — so any of those would have
#: been a reading of the iteration count. Lanczos on the same graphs is exact to
#: machine precision wherever a closed form exists to check against.
STEPS = 48

#: Below this, two Lanczos vectors are the same vector and the walk has found
#: everything it is going to.
EXHAUSTED = 1e-12

#: Where the eigenvalues of a normalised Laplacian live.
SPECTRUM = (0.0, 2.0)

#: Bisections used to pin the smallest eigenvalue of the little tridiagonal
#: matrix. Fifty halvings of [0, 2] is far finer than the estimate deserves and
#: costs nothing, and bisection is used rather than a library eigensolver so
#: that the browser can run the identical arithmetic.
BISECTIONS = 50


def smallest_eigenvalue(alpha, beta):
    """
    The smallest eigenvalue of a symmetric tridiagonal, by bisection.

    Sturm's theorem gives, for any x, how many eigenvalues lie below it — from
    a single sweep of the diagonal. Halving the interval on that count finds
    the bottom of the spectrum without ever forming the matrix, and does it the
    same way in both languages, which a library eigensolver would not.
    """
    n = len(alpha)
    if n == 0:
        return None

    def below(x: float) -> int:
        """How many eigenvalues are less than x."""
        count = 0
        d = alpha[0] - x
        if d < 0.0:
            count += 1
        for i in range(1, n):
            # A zero pivot is stepped over rather than divided by; the standard
            # dodge, and the count either side of it is unaffected.
            if d == 0.0:
                d = 1e-300
            d = (alpha[i] - x) - (beta[i - 1] * beta[i - 1]) / d
            if d < 0.0:
                count += 1
        return count

    lo, hi = SPECTRUM
    for _ in range(BISECTIONS):
        mid = 0.5 * (lo + hi)
        if below(mid) >= 1:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _largest_component(ids: List[int], adj) -> List[int]:
    """The biggest island, in a stable order so the arithmetic is repeatable."""
    seen = set()
    best: List[int] = []
    for start in ids:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        island = []
        while stack:
            node = stack.pop()
            island.append(node)
            for other in adj[node]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        if len(island) > len(best):
            best = island
    best.sort()
    return best


def spectral_gap(ids, adj) -> Optional[float]:
    """
    λ₂ of the normalised Laplacian of the largest component, or None.

    Found by Lanczos on `L = I - D^-1/2 A D^-1/2` with the trivial eigenvector
    projected out at every step. `L`'s smallest eigenvalue is always zero, with
    the square roots of the degrees as its eigenvector; removing that direction
    leaves λ₂ at the bottom, and Lanczos finds the bottom of a spectrum quickly
    even when it is crowded — which is exactly the case power iteration cannot
    handle and this substrate keeps producing.

    Read as an **upper bound**. A Ritz value approaches an eigenvalue from
    above, so on a well-connected graph this is λ₂ to machine precision, and on
    one that is nearly in two pieces it is a small number whose exact size is
    less informative than its smallness. Either way small means *cuts cheaply*,
    which is the reading that matters.
    """
    nodes = _largest_component(ids, adj)
    n = len(nodes)
    if n < 3:
        # A single node has no cut; a pair has only the trivial one. Neither is
        # a reading of how divisible a population is.
        return None

    at = {node: i for i, node in enumerate(nodes)}
    inside = set(nodes)
    neighbours: List[List[int]] = [[] for _ in range(n)]
    for node in nodes:
        i = at[node]
        for other in adj[node]:
            if other in inside and other != node:
                neighbours[i].append(at[other])

    # Sorted, because the two implementations add these up in the order they
    # are stored and floating-point addition is not associative. A Python set
    # and a JavaScript Set do not iterate alike, and without this the same
    # graph would give two answers that differ in the last few digits.
    for row in neighbours:
        row.sort()

    degree = [len(row) for row in neighbours]
    if any(d == 0 for d in degree):
        return None                      # not actually one component

    counts = np.array(degree, dtype=np.int64)
    weights = np.array(degree, dtype=np.float64)
    inverse_root = 1.0 / np.sqrt(weights)
    source = np.repeat(np.arange(n, dtype=np.int64), counts)
    target = np.array([j for row in neighbours for j in row], dtype=np.int64)

    # The eigenvector of the eigenvalue that is always zero.
    trivial = np.sqrt(weights)
    trivial /= math.sqrt(float(weights.sum()))

    def laplacian(vector: np.ndarray) -> np.ndarray:
        """L x, with the neighbours summed before the node's own weight is applied."""
        gathered = np.bincount(source, weights=vector[target] * inverse_root[target],
                               minlength=n)
        return vector - inverse_root * gathered

    # A fixed starting vector, built from integers and a power of two so every
    # entry is exact in binary floating point. A trigonometric one read better
    # and was a trap: the last bit of `sin` is not guaranteed to agree between
    # Python and a JavaScript engine.
    index = np.arange(n, dtype=np.int64)
    vector = ((index * 2654435761) % 4096) / 4096.0 - 0.5
    vector -= float(vector @ trivial) * trivial
    length = math.sqrt(float(vector @ vector))
    if length <= 0:
        return None
    vector /= length

    basis = [vector]
    alpha: List[float] = []
    beta: List[float] = []

    for step in range(min(STEPS, n - 1)):
        w = laplacian(basis[-1])
        a = float(basis[-1] @ w)
        alpha.append(a)
        w = w - a * basis[-1]
        if step:
            w = w - beta[-1] * basis[-2]

        # Reorthogonalised in full, and against the trivial direction first.
        # Rounding pulls the walk back toward the eigenvector it is meant to
        # have left, and letting it return would put zero back at the bottom of
        # the spectrum and report a gap of nothing on every graph.
        w -= float(w @ trivial) * trivial
        for earlier in basis:
            w -= float(w @ earlier) * earlier

        b = math.sqrt(float(w @ w))
        if b < EXHAUSTED:
            break
        beta.append(b)
        basis.append(w / b)

    gap = smallest_eigenvalue(alpha, beta)
    if gap is None:
        return None
    return min(2.0, max(0.0, gap))
