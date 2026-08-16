"""
gol_store.py -- a stand-in for the on-disk store, used only in the browser.

gol_series imports this module for the functions that read runs off disk. In
the browser nothing is on disk: frames are held in memory by gol_browser,
which calls gol_series.frame_stats directly and never goes near build_series.

The names below exist so the import succeeds. Calling one means something has
taken the server's path through the code by mistake, and saying so plainly is
more useful than returning an empty result that looks like an empty run.
"""

from __future__ import annotations

from typing import Any, Dict


def _unavailable(name: str):
    raise NotImplementedError(
        f"gol_store.{name} has no meaning in the browser: runs are held in "
        f"memory by gol_browser, not written to disk"
    )


def run_dir(run_id: str) -> str:
    _unavailable("run_dir")


def count_frames(run_id: str) -> int:
    _unavailable("count_frames")


def read_frame(run_id: str, index: int) -> Dict[str, Any]:
    _unavailable("read_frame")


def load_meta(run_id: str) -> Dict[str, Any]:
    _unavailable("load_meta")
