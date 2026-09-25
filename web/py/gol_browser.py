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
from typing import Any, Dict, List, Optional

import numpy as np

import gol_lineage
import gol_series
from gol_config import SimConfig
from GraphOfLifeSimple import GraphOfLife, new_world


class Worlds:
    """Every world currently loaded, by run id."""

    def __init__(self) -> None:
        self._worlds: Dict[str, Dict[str, Any]] = {}
        # Each run's history, for as long as the page is open. The server keeps
        # its in series.json; here there is no disk, and keeping nothing meant
        # summarising every sample again at every step of a climb.
        self._histories: Dict[str, gol_series.History] = {}

    # ---- settings --------------------------------------------------------

    def defaults(self) -> Dict[str, Any]:
        return {"config": SimConfig.for_new_run().to_dict(),
                "brain_presets": SimConfig.BRAIN_PRESETS}

    def normalise(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a configuration and fill in what it left out."""
        return SimConfig.from_dict(config or {}, stored=False).to_dict()

    # ---- getting a world ready -------------------------------------------

    def create(self, run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = SimConfig.from_dict(config or {}, stored=False).resolve_seed()
        self._worlds[run_id] = {"cfg": cfg, "world": new_world(cfg)}
        # The strain goes back with the config so the browser can store it on
        # the run, the same way gol_store does on a machine with a server.
        # Two backends, one label.
        return {"config": cfg.to_dict(), "strain": cfg.strain_id(), "iteration": 0}

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
        return {"config": cfg.to_dict(), "iteration": world.iteration,
                "frames": cfg.frames_before(world.iteration)}

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
            record = cfg.records(world.iteration)
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

    def series_plan(self, run_id: str, total_frames: int, points: Optional[int],
                    keys: Optional[List[str]]) -> Dict[str, Any]:
        """
        Which iterations a history request still needs, and how deep.

        The same gol_series.History the server keeps, so the two backends agree
        on what is done and what a reply holds. The frames themselves live in
        IndexedDB, out of reach from here, so the worker reads the ones this
        names and hands them to series_absorb.
        """
        history = self._histories.setdefault(run_id, gol_series.History())
        heavy = gol_series.needs_graph(keys) if keys is not None else True
        wanted = history.plan(int(total_frames), points, heavy)
        return {"iterations": sorted({f // 2 for f in wanted}),
                "stride": history.stride, "heavy": heavy}

    def series_absorb(self, run_id: str, frames: List[Dict[str, Any]], heavy: bool,
                      export_every: int = 1, strain: Optional[str] = None) -> Dict[str, Any]:
        """Summarise the frames series_plan asked for, and hand back everything known."""
        history = self._histories[run_id]
        indexed = sorted(((int(f["index"]), f["frame"]) for f in frames), key=lambda p: p[0])
        history.summarise(indexed, heavy, can_reconstruct=int(export_every or 1) == 1)
        return history.reply(heavy, strain=strain)

    def lineage(self, frames: List[Dict[str, Any]], phase: str = "all") -> Dict[str, Any]:
        """The genotype forest of a window, same code the server runs."""
        return gol_lineage.forest(frames, phase)

    def lineage_fields(self) -> List[str]:
        """What the forest reads of a frame, so the worker can hand it no more."""
        return list(gol_lineage.FIELDS)

    # ---- helpers ---------------------------------------------------------

    def _require(self, run_id: str) -> Dict[str, Any]:
        entry = self._worlds.get(run_id)
        if entry is None:
            raise KeyError(f"no world loaded for {run_id}")
        return entry


WORLDS = Worlds()
