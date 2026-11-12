"""
Selection on behaviour, measured (test 4 of book/pages/did-evolution-happen.md,
2026-11-11).

Natural selection shows as a covariance: if agents that act one way leave more
descendants than agents that act another, the population's average moves the
first way (Price 1970). For every iteration, research/screen/braintree.py's
pass over the frames keeps, for every agent alive at its start (a unit):

- four behaviours it showed in that iteration:
    child_share  the share of its tokens it gave a child (invested over what it
                 held), seen only where a child was born: an agent whose share
                 would make less than a whole token has no child, and its share
                 is not seen (the trait is censored at the low end);
    links        the links its newborn got at birth (seen where a child was born);
    spread       whether it spread its stake over several nodes (1) or went all
                 in (0); in the worlds that stake a token at a time, whether its
                 tokens went to more than one node;
    home         the share of its tokens it staked on its own node;
- its fitness: its descendants alive at the start of the next iteration on the
  brain-copy tree, itself included if its brain still holds its node; the
  children and the nodes it (or its newborn) won, if they live: survival plus
  copies;
- and its tokens, so that selection on a behaviour can be told from selection
  on wealth (Lande and Arnold 1983: a gradient holds the other traits fixed).

From these, era by era (250 iterations) and behaviour by behaviour:
- the selection differential S, the Price covariance of relative fitness with
  the behaviour, summed over the era's iterations, and its two parts, through
  survival and through copies;
- the selection gradient, the slope of relative fitness on the behaviour with
  the logarithm of tokens held fixed, as a standardized gradient (per standard
  deviation of the behaviour);
- against the permutation null: fitness shuffled among the units of each
  iteration. Its mean is 0 and its variance is known exactly (a sample
  covariance under permutation has variance var(w) var(z) / (n - 1)), so the
  z-score of an era needs no shuffling. Iterations follow the same agents, so
  their tests are not independent: each era is also cut into ten blocks of 25
  iterations, and the share of blocks whose differential has the era's sign is
  kept beside the z-score;
- the change that follows: the behaviour's mean in the next era less its mean
  in this one, set against the era's differential.

    python3 research/screen/selection.py RUN      one world's summary, from its kept rows
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import braintree as BT  # noqa: E402

#: The fewest units an iteration needs for a behaviour to be read in it.
FEWEST = 10
#: The blocks each era is cut into for the sign of its differential.
BLOCKS = 10


def iteration_terms(sums: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Per iteration, from the sums the pass kept (BT.SUMS): the units seen, the behaviour's mean and spread,
    the selection differential S = cov(w, z) / mean(w) and its part through survival, its variance under the
    permutation null, and the gradient on the behaviour with log tokens held fixed, with its null variance.
    """
    col = {name: sums[:, i] for i, name in enumerate(BT.SUMS)}
    n = col["n"]
    ok = n >= FEWEST
    nn = np.where(ok, n, 1.0)
    mz, mw, ms, mx = col["z"] / nn, col["w"] / nn, col["s"] / nn, col["x"] / nn
    vz = col["zz"] / nn - mz ** 2
    vw = col["ww"] / nn - mw ** 2
    vx = col["xx"] / nn - mx ** 2
    cwz = col["wz"] / nn - mw * mz
    csz = col["sz"] / nn - ms * mz
    cwx = col["wx"] / nn - mw * mx
    czx = col["zx"] / nn - mz * mx
    ok &= (vz > 1e-12) & (mw > 0)
    wbar = np.where(ok, mw, 1.0)
    S = np.where(ok, cwz / wbar, np.nan)
    S_survival = np.where(ok, csz / wbar, np.nan)
    var_S = np.where(ok, vw * vz / (np.maximum(nn - 1, 1) * wbar ** 2), np.nan)
    # The gradient: relative fitness regressed on the behaviour and log tokens together.
    det = vz * vx - czx ** 2
    partial = ok & (vx > 1e-12) & (det > 1e-12 * np.maximum(vz * vx, 1e-300))
    beta = np.where(partial, (cwz * vx - cwx * czx) / np.where(partial, det, 1.0) / wbar,
                    np.where(ok, cwz / np.where(ok, vz, 1.0) / wbar, np.nan))
    residual = np.where(partial, det / np.where(vx > 0, vx, 1.0), vz)          # var(z | log tokens)
    var_beta = np.where(ok, vw / (wbar ** 2 * np.maximum(nn - 1, 1) * np.maximum(residual, 1e-300)), np.nan)
    return {"n": np.where(ok, n, 0.0), "mean": np.where(ok, mz, np.nan), "sd": np.where(ok, np.sqrt(np.maximum(vz, 0)), np.nan),
            "S": S, "S_survival": S_survival, "var_S": var_S, "beta": beta, "var_beta": var_beta,
            "wbar": np.where(ok, mw, np.nan)}


def _z(total: float, var: float) -> Optional[float]:
    return float(total / math.sqrt(var)) if var > 0 and math.isfinite(total) else None


def eras_of(t: np.ndarray, terms: Dict[str, np.ndarray], eras: Sequence[int]) -> List[Dict[str, Any]]:
    """One behaviour's readings, era by era."""
    out = []
    bounds = [0] + [int(e) for e in eras]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        m = (t >= lo) & (t < hi) & np.isfinite(terms["S"])
        if m.sum() < BLOCKS:
            out.append({"from": lo, "to": hi, "iterations": int(m.sum())})
            continue
        S = terms["S"][m]
        sd = terms["sd"][m]
        total, var = float(S.sum()), float(terms["var_S"][m].sum())
        beta, var_b = terms["beta"][m], terms["var_beta"][m]
        blocks = np.array_split(S, BLOCKS)
        sign = np.sign(total)
        out.append({
            "from": lo, "to": hi, "iterations": int(m.sum()), "units": float(terms["n"][m].mean()),
            "mean": float(np.average(terms["mean"][m], weights=terms["n"][m])), "sd": float(sd.mean()),
            "S": total, "S_survival": float(terms["S_survival"][m].sum()), "z_S": _z(total, var),
            # Per standard deviation of the behaviour, per iteration: comparable across behaviours and worlds.
            "intensity": float(np.mean(S / sd)),
            "gradient": float(np.mean(beta * sd)), "z_gradient": _z(float(beta.sum()), float(var_b.sum())),
            "blocks_agreeing": float(np.mean([np.sign(x.sum()) == sign for x in blocks])) if sign else None,
        })
    for this, after in zip(out[:-1], out[1:]):
        if "mean" in this and "mean" in after:
            this["change"] = after["mean"] - this["mean"]
    return out


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 4:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def summary(rows: Dict[str, np.ndarray], eras: Sequence[int]) -> Dict[str, Any]:
    """
    A world's test 4: every behaviour's eras, and over them all how visible selection is (the root mean square
    of the eras' z-scores), how strong (the mean size of the standardized gradient, with wealth held fixed),
    and whether the behaviours moved the way the differentials point (the rank correlation, over every era and
    behaviour, of the differential, in standard deviations of the behaviour, with the change that followed).
    """
    t = rows["t"]
    traits = {}
    for k in BT.TRAITS:
        traits[k] = eras_of(t, iteration_terms(rows[f"sums_{k}"]), eras)
    cells = [(k, e) for k in traits for e in traits[k] if e.get("z_S") is not None]
    zs = [e["z_S"] for _, e in cells]
    zg = [e["z_gradient"] for _, e in cells if e.get("z_gradient") is not None]
    moved = [(e["S"] / e["sd"], e["change"] / e["sd"]) for _, e in cells if "change" in e and e["sd"] > 0]
    agree = [np.sign(a) == np.sign(b) for a, b in moved if a and b]
    strength = [float(np.mean([abs(e["gradient"]) for e in traits[k] if "gradient" in e]))
                for k in traits if any("gradient" in e for e in traits[k])]
    return {"traits": traits,
            "visible": float(np.sqrt(np.mean(np.square(zs)))) if zs else None,
            "visible_gradient": float(np.sqrt(np.mean(np.square(zg)))) if zg else None,
            "cells": len(cells), "clear": int(sum(abs(z) > 3.29 for z in zs)),
            "clear_gradient": int(sum(abs(z) > 3.29 for z in zg)),
            "strength": float(np.mean(strength)) if strength else None,
            "follows": _spearman([a for a, _ in moved], [b for _, b in moved]),
            "same_sign": float(np.mean(agree)) if agree else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    a = ap.parse_args()
    path = os.path.join(BT.readings_dir(), f"{a.run}.rows.npz")
    with np.load(path) as z:
        rows = {k: z[k] for k in z.files}
    eras = list(range(250, int(rows["t"].max()) + 2, 250))
    print(json.dumps(BT._plain(summary(rows, eras)), indent=1)[:4000])


if __name__ == "__main__":
    main()
