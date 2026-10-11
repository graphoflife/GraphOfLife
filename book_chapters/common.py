# -*- coding: utf-8 -*-
"""
What every chapter's figures share: the palette's colours by what they stand for,
how a band is said and made, and the worlds Parts II and III read.
"""
from __future__ import annotations

import functools
import os
from typing import Any, List, Tuple

from book_figures import lived_text, runs_of, runs_text, surviving_text, survivors

# Colours of the shared palette (web/js/ink.js), named for what they stand for.
BLUE, YELLOW, RED, GREEN, VIOLET, ORANGE, CYAN, GREY = range(8)

BAND_WORDS = ("The line is the median of the worlds at each iteration, the darker band holds "
              "the middle half of them (from the 25th to the 75th percentile) and the paler "
              "band nine in ten (5th to 95th)")


class Worlds:
    """
    The worlds Parts II and III read — one condition of one experiment's plan —
    and the stretches of their lives every chapter measures them over.

    Every chapter of the two parts reads the same worlds, so they are named here
    and nowhere else: the 30 worlds of 10,000 tokens of Experiment 2 at first,
    the 30 of 100,000 tokens of Experiment 10 since. A chapter asks for the
    worlds, their survivors, their last iteration or their settled stretch, and
    says them in its captions through the same names.
    """

    #: From this iteration on a world counts as settled: its boom is long over
    #: (Chapter 10). Equal to book_figures.SETTLED, which the cached passes use.
    settled = 500

    def __init__(self, experiment: str, lineage: str, condition: str = "baseline") -> None:
        self.experiment = experiment
        #: The experiment whose analysis followed these worlds' family trees
        #: (`gol_lab.py analyse`, its `lineage` block).
        self.lineage = lineage
        self.condition = condition

    @functools.cached_property
    def specs(self) -> List[Any]:
        return runs_of(self.experiment, self.condition)

    @functools.cached_property
    def alive(self) -> List[Any]:
        """The worlds that lived to their last iteration."""
        return survivors(self.specs)

    @property
    def tokens(self) -> int:
        return int(self.specs[0].config["total_tokens"])

    @property
    def until(self) -> int:
        """How many iterations each world was run for: iterations 0 to until − 1."""
        return int(self.specs[0].until)

    @property
    def last(self) -> int:
        return self.until - 1

    @property
    def window(self) -> Tuple[int, int]:
        """The settled stretch, both ends included."""
        return (self.settled, self.last)

    def tail(self, length: int) -> Tuple[int, int]:
        """The last `length` iterations, both ends included."""
        return (self.until - length, self.last)

    def run(self, seed: int) -> str:
        return next(s.run_id for s in self.specs if s.lab["seed"] == seed)

    def frame(self, iteration: int, phase: int = 2) -> int:
        """The index of an iteration's frame: 2·t after reproduction, 2·t + 1 after the game."""
        return 2 * iteration + (phase - 1)

    # ---- how the captions and recipes say them ----

    def runs_text(self) -> str:
        return runs_text(self.experiment, self.condition)

    def surviving_text(self) -> str:
        return surviving_text(self.experiment, self.condition) if self.dead else self.runs_text()

    def lived_text(self, short: bool = False) -> str:
        return lived_text(self.experiment, self.condition, short)

    @property
    def dead(self) -> List[Any]:
        return [s for s in self.specs if s not in self.alive]

    @property
    def lived(self) -> str:
        """'the 26 worlds that lived to the end', or 'the 30 worlds' when none died."""
        return f"the {len(self.alive)} worlds that lived to the end" if self.dead else f"the {len(self.specs)} worlds"

    @property
    def per_lived(self) -> str:
        """'world that lived to the end', or 'world' when none died: as in 'one dot per …'."""
        return "world that lived to the end" if self.dead else "world"

    @property
    def stretch(self) -> str:
        """'500 to 2,999'."""
        return f"{self.settled} to {self.last:,}"

    @property
    def span(self) -> str:
        """'500–2,999'."""
        return f"{self.settled}–{self.last:,}"


#: The worlds of Parts II and III. GOL_BOOK_WORLDS=E02 draws them from the
#: worlds of 10,000 tokens instead, which is how the chapters were first made.
WORLD_SETS = {"E02": Worlds("E02", lineage="E06"), "E10": Worlds("E10", lineage="E10")}
WORLDS = WORLD_SETS[os.environ.get("GOL_BOOK_WORLDS", "E10")]


def band_steps(worlds: Worlds = WORLDS) -> Tuple[str, str]:
    """How every band over the worlds is made, said for their length of life."""
    from gol_analysis import FIGURE_POINTS
    return (f"Cut the {worlds.until:,} iterations into {FIGURE_POINTS} stretches of "
            f"{worlds.until / FIGURE_POINTS:g} and take each world's mean in each stretch.",
            "In each stretch, take the median, the 25th and 75th and the 5th and 95th "
            "percentiles of those means over the worlds that still have a value there "
            "(a world that died has none after its death).")


BAND_STEPS = band_steps()


def baseline() -> List[Any]:
    return WORLDS.specs


_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def count_word(n: int) -> str:
    """'four' for 4, '26' for 26: small counts in words, as the text writes them."""
    return _WORDS[n] if 0 <= n < len(_WORDS) else f"{n:,}"


def listed(values: List[Any]) -> str:
    """'4, 313, 1,547 and 2,184'."""
    words = [f"{v:,}" if isinstance(v, int) else str(v) for v in values]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def ceiling(x: float) -> float:
    """A round number a little above x, for the top of an axis: 3,700 → 4,000, 21,222 → 25,000."""
    import math
    if x <= 0:
        return 1.0
    half = 10 ** math.floor(math.log10(x)) / 2
    return math.ceil(x * 1.0001 / half) * half
