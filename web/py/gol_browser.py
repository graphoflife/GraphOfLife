"""
gol_browser.py -- the worlds, for when there is no server.

The engine is the same file the desktop version runs; nothing about the
simulation changes here. What changes is what surrounds it. There is no disk
and no worker thread, so this module holds live worlds and advances them a
slice at a time, and the caller — a Web Worker — decides when to stop and where
the results go.

It deliberately does not keep frames. Frames are handed back the moment they
are produced and stored in IndexedDB by the caller, which is what lets a run
outlive the page and what stops the interpreter's memory growing without bound
over a long run. The same goes for run metadata: this module knows about worlds
that are currently in memory, and nothing about runs that merely exist.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List

import numpy as np

import gol_series
from gol_config import SimConfig
from GraphOfLifeSimple import GraphOfLife, new_world


class Worlds:
    """Every world currently loaded, by run id."""

    def __init__(self) -> None:
        self._worlds: Dict[str, Dict[str, Any]] = {}

    # ---- settings --------------------------------------------------------

    def defaults(self) -> Dict[str, Any]:
        return {"config": SimConfig().to_dict(),
                "brain_presets": SimConfig.BRAIN_PRESETS}

    def normalise(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a configuration and fill in what it left out."""
        return SimConfig.from_dict(config or {}, stored=False).to_dict()

    # ---- getting a world ready -------------------------------------------

    def create(self, run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = SimConfig.from_dict(config or {}, stored=False).resolve_seed()
        self._worlds[run_id] = {"cfg": cfg, "world": new_world(cfg)}
        return {"config": cfg.to_dict(), "iteration": 0}

    def restore(self, run_id: str, config: Dict[str, Any], path: str) -> Dict[str, Any]:
        """
        Rebuild a world from a checkpoint written earlier.

        The checkpoint carries the random number generator's state as well as
        the agents, so a resumed run continues the same stream rather than
        starting a new one that merely looks similar.
        """
        cfg = SimConfig.from_dict(config or {})
        with np.load(path) as blob:
            world = GraphOfLife.from_checkpoint(blob, cfg)
        self._worlds[run_id] = {"cfg": cfg, "world": world}
        return {"config": cfg.to_dict(), "iteration": world.iteration}

    def has(self, run_id: str) -> bool:
        return run_id in self._worlds

    def drop(self, run_id: str) -> None:
        self._worlds.pop(run_id, None)

    # ---- advancing -------------------------------------------------------

    def step(self, run_id: str, iterations: int = 1) -> Dict[str, Any]:
        """
        Advance a few iterations and hand back whatever they recorded.

        Returns as soon as the slice is done so the caller can look at its
        message queue; that is what makes a run interruptible. Frames are
        returned rather than kept, because keeping them is the caller's job.
        """
        entry = self._require(run_id)
        cfg, world = entry["cfg"], entry["world"]
        produced: List[Dict[str, Any]] = []

        for _ in range(max(1, iterations)):
            record = (world.iteration % cfg.export_every == 0)
            frames = world.step(record_decisions=cfg.export_decisions and record)
            if record:
                produced.extend(frames)
            if world.is_extinct():
                return {"iteration": world.iteration, "extinct": True, "frames": produced}

        return {"iteration": world.iteration, "extinct": False, "frames": produced}

    def checkpoint(self, run_id: str, path: str) -> int:
        """Write a resume point, and say how large it turned out."""
        world = self._require(run_id)["world"]
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **world.to_checkpoint())
        data = buffer.getvalue()
        with open(path, "wb") as handle:
            handle.write(data)
        return len(data)

    # ---- statistics ------------------------------------------------------

    def stats(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reduce frames to the scalars the charts plot.

        The very same gol_series the server uses. Frames arrive from storage in
        order, so each one can still be handed its predecessor for the runs
        that need a delta reconstructed.
        """
        rows: List[Dict[str, Any]] = []
        previous = None
        for frame in frames:
            rows.append(gol_series.frame_stats(frame, previous))
            previous = frame
        return rows

    def sample_stride(self, total_iterations: int) -> int:
        return gol_series._sample_stride(total_iterations)

    def node_count_keys(self) -> List[str]:
        return list(gol_series.NODE_COUNT_KEYS)

    # ---- helpers ---------------------------------------------------------

    def _require(self, run_id: str) -> Dict[str, Any]:
        entry = self._worlds.get(run_id)
        if entry is None:
            raise KeyError(f"no world loaded for {run_id}")
        return entry


WORLDS = Worlds()
