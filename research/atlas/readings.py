"""
Lab runs read at chosen iterations by the four measures of the author's picture
(Chapters 98 to 101): the seven marks of the health bar and the diversity score
(profile.py), the typical distance over the random twin's (space.py), and the
Gini of the tokens. Each world is read from its archive; no world is played.

    python3 research/atlas/readings.py OUT.json --runs B2-10000-12215f-s001 ... --at 1000 3000 [--jobs 4]

A reading already in OUT.json (same run, same iteration) is kept, not read again.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import gol_store  # noqa: E402


def read(item: Tuple[str, int, str]) -> Dict[str, Any]:
    run_id, at, family = item
    import profile as P
    spec = importlib.util.spec_from_file_location("space_atlas", os.path.join(HERE, "space.py"))
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    path = gol_store.archive_path(run_id, at)
    config = json.load(open(os.path.join(gol_store.run_dir(run_id), "meta.json"))).get("config") or {}
    r = P.measure((run_id, path, family, bool(config.get("mix_at_birth"))))
    if r.get("dead") or r.get("error"):
        return r
    r["at"] = at
    r["bar"] = P.bar(r)
    r["marks"] = sum(1 for v in r["bar"].values() if v)
    r["diversity"] = P.diversity(r, P.reference())[1]
    r["wide"] = float(S.typical_distance(S.one((family, "", run_id, path, at))))
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--at", nargs="+", type=int, required=True)
    ap.add_argument("--family", default="", help="the family the readings are filed under")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    have = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["id"], r.get("at")) for r in have}
    todo = [(run, at, a.family) for run in a.runs for at in a.at if (run, at) not in done]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = have + list(ex.map(read, todo))
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        print(r["id"], r.get("at"), r.get("agents"), r.get("marks"), r.get("diversity"), r.get("wide"),
              r.get("gini_tokens"))


if __name__ == "__main__":
    main()
