"""
What more than one test file runs: small worlds, runs folders of their own,
runs made in them two ways, and graphs drawn by hand.
"""
from __future__ import annotations

import contextlib
import tempfile

import runner  # noqa: F401  (first: the repository on the path, and a runs folder of the tests' own)

import gol_run
import gol_store
from gol_config import SimConfig
from GraphOfLifeSimple import new_world


def small(**overrides) -> SimConfig:
    """A world small enough to run many times in a test."""
    settings = dict(
        total_tokens=3000, n_nodes=60, k_neighbors=6,
        hidden_layers=[14, 12], message_amount=2, random_input_amount=2,
        seed=17,
    )
    settings.update(overrides)
    return SimConfig(**settings)


@contextlib.contextmanager
def scratch_runs():
    """A runs folder of its own for the length of a test."""
    with tempfile.TemporaryDirectory() as tmp:
        original = gol_store.BASE_DIR
        gol_store.BASE_DIR = tmp
        try:
            yield tmp
        finally:
            gol_store.BASE_DIR = original


def advanced_run(cfg, until, record=None, name="x"):
    """
    A run made in the current runs folder and advanced to `until` as the
    server and the lab advance one, recording its statistics as it goes.
    """
    extra = {} if record is None else {"record": record}
    run_id = gol_store.create_run(name, cfg, **extra)["id"]
    gol_run.advance(run_id, until=until)
    return run_id


def unrecorded_run(iterations: int, seed: int = 4):
    """
    A small run in the current runs folder that wrote its frames and nothing
    else, as every run did before runs recorded their statistics: one whose
    history is worked out from the frames. Its id, its world, and how many
    frames it has.
    """
    cfg = SimConfig(total_tokens=400, n_nodes=30, k_neighbors=4,
                    seed=seed, hidden_layers=[6], export_decisions=False)
    run_id = gol_store.create_run("x", cfg)["id"]
    world = new_world(cfg)
    return run_id, world, write_frames(run_id, world, 0, iterations)


def write_frames(run_id: str, world, written: int, iterations: int) -> int:
    """Run `world` on and write down what it does, returning the new frame count."""
    for _ in range(iterations):
        for frame in world.step(record_decisions=False):
            gol_store.write_frame(run_id, written, frame)
            written += 1
    gol_store.update_meta(run_id, frame_count=written, iteration=world.iteration)
    return written


def adjacency(edges):
    """A graph from a list of pairs, as the bridge walk and the spectral gap read one."""
    out = {}
    for a, b in edges:
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out
