#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What the book reads from a run beyond the statistics it records.

Most figures are drawn from the rows every run records (stats.jsonl). The rest
measure a run's frames or its final checkpoint, and that is slow — minutes for
one world — so each such measure is made once and kept beside the runs, in
GraphOfLifeRuns/.book/, until the run moves on. This module is the one place
that knows how: a Measure says which file it is kept in, what a kept copy must
say about the run to be used again, and how to make it.

    GOL_BOOK_CACHE=strict python3 book_figures.py

makes nothing: a measure that is not kept raises instead. That is how a change
to the book's code is shown to change no figure without remaking any.

It is also the one place that says where an iteration's frames are (frame_at).
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
from typing import Any, Callable, Dict, Optional

import numpy as np

import gol_store as store


class NotKept(RuntimeError):
    """A measure that is not kept was asked for, and the cache is strict."""


def strict() -> bool:
    return os.environ.get("GOL_BOOK_CACHE") == "strict"


def _iteration(run_id: str) -> Any:
    """What most measures depend on: how far the run has gone."""
    return store.load_meta(run_id).get("iteration")


def _same(kept: Any, wanted: Any) -> bool:
    return kept == wanted


@dataclasses.dataclass(frozen=True)
class Measure:
    """
    One thing measured from a run and kept beside it. A kept copy holds, under
    `field`, what it was made from — `want(run_id, **params)` — and is used
    again when `fits(kept, wanted)`. Anything that changes what `make` returns
    has to be in what `want` returns, or a kept copy made from something else
    would be handed out.
    """
    name: str
    make: Callable[..., Dict[str, Any]]
    file: str
    field: str = "stamp"
    want: Callable[..., Any] = _iteration
    fits: Callable[[Any, Any], bool] = _same

    def path(self, run_id: str) -> str:
        return os.path.join(store.BASE_DIR, ".book", self.file.format(run=run_id))


MEASURES: Dict[str, Measure] = {}


def measure(name: str, file: Optional[str] = None, **how: Any) -> Callable:
    """
    Keep what the decorated function makes of a run, per run, as the measure
    `name`: in `<runs>/.book/<run>.<name>.json` unless `file` says otherwise
    (a `.npz` file keeps arrays). Calling the decorated function reads the kept
    copy, and makes and keeps it only when there is none that fits.
    """
    def register(make: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
        # The same function seen twice is book_figures run as a script: it is
        # imported once as __main__ and again by the chapters, as book_figures.
        if name in MEASURES and _where(MEASURES[name].make) != _where(make):
            raise ValueError(f"two measures are called {name!r}")
        m = MEASURES[name] = Measure(name, make, file or f"{{run}}.{name}.json", **how)

        @functools.wraps(make)
        def kept(run_id: str, **params: Any) -> Dict[str, Any]:
            return get(m, run_id, **params)
        kept.measure = m
        return kept
    return register


def _where(fn: Callable) -> tuple:
    return fn.__code__.co_filename, fn.__code__.co_firstlineno, fn.__qualname__


def get(m: Measure, run_id: str, **params: Any) -> Dict[str, Any]:
    """A run's measure: the kept copy if one fits, otherwise made and kept."""
    wanted = m.want(run_id, **params)
    path = m.path(run_id)
    kept = _read(path)
    if kept is not None and m.field in kept and m.fits(_plain(kept[m.field]), wanted):
        return kept
    if strict():
        raise NotKept(f"the {m.name} of {run_id} is not kept, and GOL_BOOK_CACHE=strict "
                      f"makes nothing")
    made = {m.field: np.array(wanted) if path.endswith(".npz") else wanted,
            **m.make(run_id, **params)}
    _write(path, made)
    return made


def needs_decisions(run_id: str, what: str) -> None:
    """
    Refuse a run that did not record its agents' decisions: a measure of what
    they decided would read nothing there and quietly come out as nothing.
    """
    if not store.load_config(run_id).export_decisions:
        raise ValueError(f"{run_id} did not record its agents' decisions, so {what} cannot "
                         f"be read from it")


def _plain(value: Any) -> Any:
    """A number kept in an .npz comes back as an array of no dimensions."""
    return value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value


def _read(path: str) -> Optional[Dict[str, Any]]:
    try:
        if path.endswith(".npz"):
            with np.load(path) as kept:
                return {k: kept[k] for k in kept.files}
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _write(path: str, out: Dict[str, Any]) -> None:
    """Whole or not at all: a measure killed half-way leaves the old copy, or none."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = f"{path}.{os.getpid()}.part"
    if path.endswith(".npz"):
        part += ".npz"                   # numpy would add it, and the rename would miss
        np.savez_compressed(part, **out)
    else:
        with open(part, "w") as f:
            json.dump(out, f)
    os.replace(part, path)
