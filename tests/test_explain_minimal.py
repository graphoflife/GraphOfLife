#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explain_minimal.py, held to the engine where it says it is the same algorithm.

    python3 tests/test_explain_minimal.py

The teaching script is a copy a reader can take away, so it is not the engine
and does not import it. It says its structure is identical, and the
Explanation tab walks through it line by line: so the decisions it spells out
— how tokens are split, who takes a node, how a yes or no is read, how a pair
of outputs becomes a share, what a brain's last layer does — are fed the same
inputs as the engine's and must give the same answers. Inputs are drawn so no
two candidates tie, because ties are broken by a random draw on both sides,
and two generators never draw alike.
"""
from __future__ import annotations

import atexit
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Nothing here touches the runs folder, but nothing should be able to.
os.environ["GOL_RUNS_DIR"] = tempfile.mkdtemp(prefix="gol-tests-")
atexit.register(shutil.rmtree, os.environ["GOL_RUNS_DIR"], True)

import numpy as np

import explain_minimal as M
import GraphOfLifeSimple as E


def test_tokens_are_split_as_the_engine_splits_them():
    rng = np.random.default_rng(1)
    for _ in range(500):
        n = int(rng.integers(1, 8))
        # Scores of either sign, as a linear output layer gives them, and now
        # and then none above zero, where both split evenly.
        scores = rng.normal(0.0, 1.0, n) - (2.0 if rng.random() < 0.2 else 0.0)
        total = int(rng.integers(0, 60))
        assert np.array_equal(M.apportion(scores, total), E._apportion(scores, total)), (scores, total)


def test_a_pair_of_outputs_is_read_as_the_same_share():
    rng = np.random.default_rng(2)
    for a, b in rng.normal(0.0, 1.0, (500, 2)):
        assert M.share_of(a, b) == E._share_of_first(a, b), (a, b)


def test_a_yes_or_no_is_read_as_the_engine_reads_it():
    rng = np.random.default_rng(3)
    state = np.random.RandomState(3)
    for _ in range(500):
        yes, no, mode_a, mode_b = rng.normal(0.0, 1.0, 4)
        if mode_a > mode_b:
            # Read as a chance: made certain, so one draw on each side agrees.
            yes, no = (abs(yes), -abs(no)) if rng.random() < 0.5 else (-abs(yes), abs(no))
        assert M.decide(yes, no, mode_a, mode_b) == E._choose_binary(yes, no, mode_a, mode_b, state), \
            (yes, no, mode_a, mode_b)


def test_a_node_goes_to_whom_the_engine_gives_it():
    rng = np.random.default_rng(4)
    state = np.random.RandomState(4)
    for _ in range(500):
        agents = [int(a) for a in rng.choice(1000, int(rng.integers(1, 9)), replace=False)]
        amounts = rng.choice(np.arange(1, 400), len(agents), replace=False)
        staked = {a: int(t) for a, t in zip(agents, amounts)}
        flagged = rng.choice(np.arange(1, 400), len(agents), replace=False)
        revolt = {a: min(staked[a], int(f)) for a, f in zip(agents, flagged) if rng.random() < 0.7}
        if len(set(revolt.values())) < len(revolt):
            continue                       # a rung of two: drawn on both sides
        want = E.GraphOfLife._resolve_winner(staked, revolt, state)[0]
        assert M.resolve(staked, revolt) == want, (staked, revolt)


def test_a_brain_ends_in_a_linear_layer():
    random.seed(5)
    np.random.seed(5)
    brain = M.Brain(4, 3)
    x = np.random.random((4, 2))
    a = x
    for i, (w, b) in enumerate(zip(brain.weights, brain.biases)):
        z = w @ a + b[:, None]
        a = z if i == len(brain.weights) - 1 else 1.0 / (1.0 + np.exp(-z))
    assert np.allclose(brain.forward(x), a), "the last layer is squashed, as the engine's is not"


def _main() -> int:
    import time
    import traceback

    tests = sorted((name, fn) for name, fn in globals().items()
                   if name.startswith("test_") and callable(fn))
    failures = []
    started = time.perf_counter()
    for name, fn in tests:
        try:
            fn()
            print(".", end="", flush=True)
        except Exception:
            failures.append((name, traceback.format_exc()))
            print("F", end="", flush=True)
    print(f"\n\n{len(tests) - len(failures)} passed, {len(failures)} failed "
          f"in {time.perf_counter() - started:.1f}s")
    for name, trace in failures:
        print(f"\n--- {name} ---\n{trace}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
