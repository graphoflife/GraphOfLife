#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The statistics of one frame, for the strip under the Viewer's canvas.

They are the row gol_series.frame_stats makes of every frame a run records,
asked for one frame at a time: the frame on screen, whole or cropped to a
region. The strip used to work them out itself, in JavaScript, from a second
copy of every formula that tests had to hold to this one; now there is one.

How deep to go is the caller's. The graph statistics (`structure`) are five
sixths of what a frame costs, and the circulating flow (`flow`) is its own
cost, so each is only worked out when the part of the strip that shows it is
open. The server and the browser's worker both answer with strip(); the
server also answers from a run's record when it holds what is asked
(gol_server), which for a large world is the difference between a
millisecond and seconds.

Not part of the engine, and free of the store, so the browser can run it.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import gol_series
from gol_lightning import lightning
from gol_stats import finite

#: The keys only a flow reading fills, which a light row leaves empty.
FLOW_KEYS = ("lightningScore", "cyclingShare", "lightningLongest", "flowImbalance",
             "netLightningScore", "netCyclingShare", "netLightningLongest", "netFlowShare")


def strip(frame: Dict[str, Any], previous: Optional[Dict[str, Any]] = None,
          structure: bool = False, flow: bool = False) -> Dict[str, Any]:
    """
    A frame's row, as deep as asked: the graph statistics with `structure`,
    which take the flow along, and the flow on its own with `flow`. What is
    not asked for comes back empty (None), as a light row's does.
    """
    row = gol_series.frame_stats(frame, previous, heavy=structure)
    if flow and not structure:
        row.update(lightning(frame))
    return finite(row)


def holds(row: Dict[str, Any], structure: bool, flow: bool) -> bool:
    """Whether a recorded row already answers a request this deep."""
    if structure and not row.get("_heavy"):
        return False
    if flow and not row.get("_heavy") and any(row.get(k) is None for k in FLOW_KEYS):
        # A light row has no flow, unless the phase moved no tokens at all,
        # in which case every heavy row says None too. Not worth telling
        # apart: the frame is read and the flow worked out.
        return False
    return True
