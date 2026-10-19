"""
A brain that can grow: one hidden layer of up to H units, of which the first
`active` take part. A mutation can switch the next unit on (its outgoing weights
start at zero, so a new unit changes nothing until later mutations give it a
voice) or switch the last one off. `active` is inherited with the weights.

Registered as brain_kind "grow"; the number of units H is hidden_layers[0], and
the starting number of active units is GROW_START.
"""
import os
import sys
# Its own folder first, then the repository's root and the checks beside it: a copy of these files
# frozen beside the engine (the lab's snapshot, gol_pilot.py) then imports the copies it sits with.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (os.path.join(_HERE, "..", "checks"), os.path.join(_HERE, "..", "..", "..")):
    if os.path.exists(os.path.join(_path, "rules.py")) or os.path.exists(os.path.join(_path, "GraphOfLifeSimple.py")):
        sys.path.insert(0, os.path.abspath(_path))
sys.path.insert(0, _HERE)
import numpy as np

import GraphOfLifeSimple as E
from gol_config import SimConfig

GROW_START = 4
GROW_P = 0.25          # chance, given that a brain mutates, that its size changes too
MEM_START = 1          # games of memory a founder looks back over
MEM_MAX = 8


class GrowBrain(E.Float16Brain):
    # `shadow` is the neutral control: a size that is inherited and mutated exactly
    # like `active`, but that nothing reads. Whatever `active` does beyond what
    # `shadow` does is selection's doing, not drift's.
    __slots__ = ("active", "shadow", "mem", "memshadow")

    def __init__(self, cfg, brain_id, rng=None):
        self.active = GROW_START
        self.shadow = GROW_START
        self.mem = MEM_START
        self.memshadow = MEM_START
        super().__init__(cfg, brain_id, rng)
        if rng is not None and self.weights:
            H = self.weights[0].shape[0]
            self.active = min(GROW_START, H)
            self.weights[1][:, self.active:] = 0

    def forward(self, x):
        a = np.asarray(x, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        W0, W1 = self.weights
        b0, b1 = self.biases
        k = self.active
        z = W0[:k].astype(np.float64) @ a + b0[:k].astype(np.float64)
        h = E._sigmoid(z)
        return W1[:, :k].astype(np.float64) @ h + b1.astype(np.float64)

    def copy(self):
        clone = super().copy()
        clone.active = self.active
        clone.shadow = self.shadow
        clone.mem = self.mem
        clone.memshadow = self.memshadow
        return clone

    def mutate(self, brain_id, rng):
        changed = super().mutate(brain_id, rng)
        if changed and rng.random() < GROW_P:
            H = self.weights[0].shape[0]
            up = rng.random() < 0.5
            if up and self.active < H:
                self.weights[1][:, self.active] = 0     # a new unit starts silent
                self.active += 1
            elif not up and self.active > 1:
                self.active -= 1
            # The shadow takes a step of its own, drawn the same way and bounded
            # the same way, but nothing it does reaches the brain's decisions.
            up = rng.random() < 0.5
            if up and self.shadow < H:
                self.shadow += 1
            elif not up and self.shadow > 1:
                self.shadow -= 1
        if changed and rng.random() < GROW_P:
            # Memory length, with a shadow of its own, bounded at 0 and MEM_MAX.
            up = rng.random() < 0.5
            self.mem = min(MEM_MAX, self.mem + 1) if up else max(0, self.mem - 1)
            up = rng.random() < 0.5
            self.memshadow = min(MEM_MAX, self.memshadow + 1) if up else max(0, self.memshadow - 1)
        return changed


def install():
    E.BRAIN_KINDS["grow"] = GrowBrain
    SimConfig.BRAIN_PRESETS["grow"] = {"hidden_layers": [32], "mutation_sparsity": 0.1}
    SimConfig.MODES["brain_kind"] = tuple(SimConfig.BRAIN_PRESETS)


def save_active(world, blob):
    """Write each agent's number of active units beside the weights, in checkpoint order."""
    nodes = sorted(world.G.nodes())
    if nodes and hasattr(world.brains[nodes[0]], "active"):
        blob["active"] = np.array([world.brains[u].active for u in nodes], dtype=np.int64)
        blob["shadow"] = np.array([world.brains[u].shadow for u in nodes], dtype=np.int64)
        blob["mem"] = np.array([world.brains[u].mem for u in nodes], dtype=np.int64)
        blob["memshadow"] = np.array([world.brains[u].memshadow for u in nodes], dtype=np.int64)


def restore_active(world, blob):
    if "active" not in blob:
        return
    shadow = blob["shadow"].tolist() if "shadow" in blob else blob["active"].tolist()
    mem = blob["mem"].tolist() if "mem" in blob else [MEM_START] * len(shadow)
    memshadow = blob["memshadow"].tolist() if "memshadow" in blob else mem
    for u, a, s, m, ms in zip(blob["ids"].tolist(), blob["active"].tolist(), shadow, mem, memshadow):
        if u in world.brains and hasattr(world.brains[u], "active"):
            world.brains[u].active = int(a)
            world.brains[u].shadow = int(s)
            world.brains[u].mem = int(m)
            world.brains[u].memshadow = int(ms)
