# -*- coding: utf-8 -*-
"""
What every chapter's figures share: the palette's colours by what they stand for,
how a band is said and made, and the thirty baseline runs.
"""
from __future__ import annotations

from typing import Any, List

from book_figures import runs_of

# Colours of the shared palette (web/js/ink.js), named for what they stand for.
BLUE, YELLOW, RED, GREEN, VIOLET, ORANGE, CYAN, GREY = range(8)

BAND_WORDS = ("The line is the median of the worlds at each iteration, the darker band holds "
              "the middle half of them (from the 25th to the 75th percentile) and the paler "
              "band nine in ten (5th to 95th)")
BAND_STEPS = ("Cut the iterations into stretches of 5 (600 stretches over 3,000 iterations) "
              "and take each world's mean in each stretch.",
              "In each stretch, take the median, the 25th and 75th and the 5th and 95th "
              "percentiles of those means over the worlds that still have a value there "
              "(a world that died has none after its death).")


def baseline() -> List[Any]:
    return runs_of("E02")
