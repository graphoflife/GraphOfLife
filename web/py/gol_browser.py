"""
gol_browser.py -- the run manager, for when there is no server.

The engine is the same file that runs locally; nothing about the simulation
changes here. What changes is everything around it. There is no disk to write
frames to and no worker thread to run them, so runs live in memory and are
advanced a slice at a time by whoever calls `step`, which in the browser is a
Web Worker between message deliveries. That is what keeps a long run
interruptible: the loop belongs to the caller, not to this module.

Runs last as long as the page does. Frames carry the whole topology, and a few
thousand agents over a few thousand iterations is already hundreds of
megabytes, so keeping them past a reload would mean deciding what to throw
away. Better to be plainly temporary than quietly lossy.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import gol_series
from gol_config import SimConfig
from GraphOfLifeSimple import new_world


class Runs:
    """Every run this page knows about."""

    def __init__(self) -> None:
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    # ---- identity --------------------------------------------------------

    def _next_id(self) -> str:
        """Same shape the server uses, so a name reads the same either way."""
        self._counter += 1
        return f"GOL_{time.strftime('%y_%m_%d')}_n{self._counter:03d}"

    # ---- the shape the interface expects ---------------------------------

    def _meta(self, run: Dict[str, Any]) -> Dict[str, Any]:
        frames = run["frames"]
        return {
            "id": run["id"],
            "name": run["name"],
            "created_at": run["created_at"],
            "status": run["status"],
            "iteration": run["world"].iteration,
            "frame_count": len(frames),
            # No checkpoint file exists, but a live run can always carry on
            # from where it is, which is what the button is really asking.
            "checkpoint_iteration": run["world"].iteration if frames else None,
            "has_checkpoint": bool(frames),
            "running": run["status"] == "running",
            "error": run["error"],
            "config": run["config"].to_dict(),
            "size_bytes": run["bytes"],
        }

    # ---- managing runs ---------------------------------------------------

    def defaults(self) -> Dict[str, Any]:
        return {"config": SimConfig().to_dict()}

    def create(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = SimConfig.from_dict(config or {})
        run_id = self._next_id()
        self._runs[run_id] = {
            "id": run_id,
            "name": (name or "").strip() or run_id,
            "created_at": time.time(),
            "status": "idle",
            "error": None,
            "config": cfg,
            "world": new_world(cfg),
            "frames": [],
            "bytes": 0,
            "series": None,
        }
        return self._meta(self._runs[run_id])

    def list(self) -> List[Dict[str, Any]]:
        return [self._meta(r) for r in
                sorted(self._runs.values(), key=lambda r: r["created_at"], reverse=True)]

    def get(self, run_id: str) -> Dict[str, Any]:
        return self._meta(self._require(run_id))

    def delete(self, run_id: str) -> None:
        self._runs.pop(run_id, None)

    def set_status(self, run_id: str, status: str) -> Dict[str, Any]:
        run = self._require(run_id)
        run["status"] = status
        return self._meta(run)

    # ---- advancing -------------------------------------------------------

    def step(self, run_id: str, iterations: int = 1) -> Dict[str, Any]:
        """
        Advance a few iterations and keep whatever they recorded.

        Returns as soon as the slice is done so the caller can look at its
        message queue. Extinction stops the run here rather than leaving the
        caller to notice, since carrying on would only produce empty frames.
        """
        run = self._require(run_id)
        cfg = run["config"]
        world = run["world"]

        for _ in range(max(1, iterations)):
            record = (world.iteration % cfg.export_every == 0)
            frames = world.step(record_decisions=cfg.export_decisions and record)

            if record:
                for frame in frames:
                    run["frames"].append(frame)
                    # Rough, and deliberately so: it is for showing the reader
                    # how much of their memory a run is using, not accounting.
                    run["bytes"] += 120 * len(frame.get("ids", ())) + 40 * len(frame.get("edges", ()))

            if world.is_extinct():
                run["status"] = "extinct"
                break

        run["series"] = None   # the history grew; whatever was summarised is stale
        return self._meta(run)

    # ---- reading ---------------------------------------------------------

    def frame(self, run_id: str, index: int) -> Dict[str, Any]:
        frames = self._require(run_id)["frames"]
        if not 0 <= index < len(frames):
            raise KeyError(f"frame {index} does not exist")
        return frames[index]

    def series(self, run_id: str) -> Dict[str, Any]:
        """
        Per-frame statistics, the same ones the server computes.

        Sampled the same way too: a run of many thousand iterations is reduced
        to at most gol_series.MAX_SAMPLED_ITERATIONS of them, since a chart a
        few hundred pixels wide cannot show more and the structural statistics
        are much too slow to compute for every frame.
        """
        run = self._require(run_id)
        if run["series"] is not None:
            return run["series"]

        frames = run["frames"]
        total_iterations = max(0, len(frames) // 2)
        stride = gol_series._sample_stride(total_iterations)

        rows: List[Dict[str, Any]] = []
        previous = None
        for index, frame in enumerate(frames):
            if (index // 2) % stride:
                continue
            rows.append(gol_series.frame_stats(frame, previous))
            previous = frame

        if not rows:
            payload = {"count": 0, "keys": [], "series": {}, "stride": stride,
                       "sampled": False, "nodeCountKeys": list(gol_series.NODE_COUNT_KEYS)}
        else:
            keys = list(rows[0].keys())
            payload = {
                "count": len(rows),
                "keys": keys,
                "series": {k: [row.get(k) for row in rows] for k in keys},
                "stride": stride,
                "sampled": stride > 1,
                "totalIterations": total_iterations,
                "nodeCountKeys": list(gol_series.NODE_COUNT_KEYS),
            }
        run["series"] = payload
        return payload

    # ---- helpers ---------------------------------------------------------

    def _require(self, run_id: str) -> Dict[str, Any]:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"no run called {run_id}")
        return run


RUNS = Runs()
