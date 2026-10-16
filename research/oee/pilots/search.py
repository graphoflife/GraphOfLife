"""
Batch Z: settings by elimination — worlds with settings drawn at random, and the
values that make worlds worse dropped, round after round.

The author's suggestion (2026-10-16): run many worlds with random settings, find
out which settings tend to make things worse, note them and get rid of them.

The space. Every setting the book has used, at the values it has used, nothing
new: the thirteen on which B1 and B2 differ (B2's value first, then B1's), the
other values the lab tried on B2 (Experiments 9, 14, 15 and 16), and the
switches of the pilot batches. Two pairs that only work together are one
setting each: a brain whose units cost a token is a growing brain, and links
that hold a token have no cap on links.

A round. ROUND worlds of B2 at 10,000 tokens, each setting's value drawn
uniformly and independently from the values still allowed, each world with a
seed of its own, played for 1,000 iterations (GraphOfLifeRuns/.pilots/Z/).
Every world is first stepped twice; a combination the code cannot play is
noted and drawn again.

How good a world is. Its closeness to the author's picture of a world that
might keep evolving (Meta X): the mean of the share of the health bar's seven
marks it meets and its diversity score (research/atlas/profile.py), both read
at its last checkpoint. A world that died scores 0.

What is worse. A value's effect is the mean score of the worlds that have it
minus the mean of those with another value of the same setting, taken within
each round (the rounds draw from different sets of values) and averaged with
weights n_with * n_without / n. Its interval comes from resampling the worlds
within each round, at 95% for all the values judged together (each at
1 - 0.05 / k for k values). A value whose whole interval lies below zero is
worse: it is written down and drawn no more. Every setting keeps one value.

    python3 search.py sample [ROUND]     draw a round, step each world twice, write jobs_Z<ROUND>.txt
    python3 search.py judge              score every world so far, drop what is worse, write research/oee/search.md
    python3 search.py loop --rounds 4    sample, run, profile, judge; again
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from make_jobs_TU import B2, GIFTS  # noqa: E402

PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
OUT = "../../../GraphOfLifeRuns/.pilots/Z"
STATE = os.path.join(HERE, "search_state.json")
LOG = os.path.join(ROOT, "research", "oee", "search.md")
SMOKE = "../../../GraphOfLifeRuns/.pilots/Z-smoke"
ROUND = 32
ITERS = 1000
SLOTS = 4


def _kv(text: str) -> Dict[str, str]:
    return dict(kv.split("=", 1) for kv in text.split())


#: (setting, [(value, settings, prototype keys), ...]), B2's own value first.
SPACE: List[Tuple[str, List[Tuple[str, Dict[str, str], Dict[str, str]]]]] = [
    # The thirteen settings on which B1 and B2 differ, and the lab's other values on B2.
    ("brain", [("16 units (B2)", {}, {}), ("50-45-40-35-30 (B1)", {"hidden_layers": "50,45,40,35,30"}, {}),
               ("grows to 32", {"brain_kind": "grow", "hidden_layers": "32"}, {}),
               ("grows to 32, units cost a token", {"brain_kind": "grow", "hidden_layers": "32"}, {"unit_token": "true"})]),
    ("messages", [("8 (B2)", {}, {}), ("30 (B1)", {"message_amount": "30"}, {})]),
    ("messages before the game", [("no (B2)", {}, {}), ("yes (B1)", {"message_prepass": "true"}, {})]),
    ("handover", [("off (B2)", {}, {}), ("on (B1)", {"allow_handover": "true"}, {})]),
    ("brains change", [("when copied (B2)", {}, {}), ("after every game (B1)", {"mutate_on_replication": "false"}, {})]),
    ("home advantage", [("1.5 (B2)", {}, {}), ("none (B1)", {"incumbency": "1.0"}, {})]),
    ("links at birth", [("2 (B2)", {}, {}), ("0 (B1)", {"birth_links": "0"}, {})]),
    ("mending", [("on (B2)", {}, {}), ("off (B1)", {"mend_on_death": "false"}, {})]),
    ("unused links", [("lapse unless they share 2 (B2)", {}, {}), ("all lapse (B1)", {"prune_needs": "0"}, {})]),
    ("links", [("8 at most (B2)", {}, {}), ("no cap (B1)", {"valence": "0"}, {}),
               ("no cap, each holds a token", {"valence": "0"}, {"link_token": "true"})]),
    ("kept at home", [("0.3 (B2)", {}, {}), ("nothing (B1)", {"keep_home": "0.0"}, {}),
                      ("the last token", {"keep_home": "0.01"}, {}), ("half", {"keep_home": "0.5"}, {})]),
    ("old age", [("100 (B2)", {}, {}), ("none (B1)", {"max_age": "0"}, {})]),
    ("smallest child", [("10 (B2)", {}, {}), ("1 (B1)", {"min_child": "1"}, {})]),
    ("mutation", [("0.2 (B1, B2)", {}, {})] + [(v, {"mutation_probability": v}, {}) for v in ("0.05", "0.1", "0.4", "0.8")]),
    ("random inputs", [("5 (B1, B2)", {}, {}), ("none", {"random_input_amount": "0"}, {})]),
    # The switches of the pilot batches.
    ("gifts", [("off", {}, {}), ("on", _kv(GIFTS), {})]),
    ("revolutions", [("on", {}, {}), ("off", {"allow_revolutions": "false"}, {})]),
    ("estate", [("evenly", {}, {})] + [(v, {"redistribution": v}, {}) for v in ("by_tokens", "poorest", "newborns", "parents")]),
    ("brains mixed at birth", [("no", {}, {}), ("yes", {}, {"mix": "true"})]),
    ("where my brain came from", [("unseen", {}, {}), ("seen", {}, {"came_from": "true"})]),
    ("who helped me", [("unseen", {}, {}), ("seen", {}, {"helped_by": "true"})]),
    ("contest", [("largest stake", {}, {}), ("lottery", {}, {"contest": "lottery"})]),
    ("sides", [("alone", {}, {}), ("by genotype", {}, {"pool": "genotype"}),
               ("by family", {}, {"tag_rate": "0.05", "pool": "family"})]),
    ("eye for kin", [("none", {}, {}), ("kin", {}, {"kin_sense": "true"}),
                     ("family", {}, {"tag_rate": "0.05", "kin_sense": "true"})]),
]
VALUES = {setting: {label: (s, p) for label, s, p in values} for setting, values in SPACE}
FIRST = {setting: values[0][0] for setting, values in SPACE}


def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"allowed": {setting: [label for label, _, _ in values] for setting, values in SPACE},
            "dropped": [], "worlds": [], "unplayable": [], "rounds": 0}


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE + ".tmp", "w") as f:
        json.dump(state, f, indent=1)
    os.replace(STATE + ".tmp", STATE)


def command(out: str, choice: Dict[str, str], seed: int, iters: int = ITERS, ckpt: int = 100, health: int = 50) -> str:
    sets, proto = _kv(B2.format(seed=seed)), {"engine": "true"}
    for setting, label in choice.items():
        s, p = VALUES[setting][label]
        sets.update(s)
        proto.update(p)
    return (f"python3 run.py {shlex.quote(out)} --iters {iters} --ckpt {ckpt} --health {health} "
            f"--set {' '.join(f'{k}={v}' for k, v in sets.items())} --proto {' '.join(f'{k}={v}' for k, v in proto.items())}")


def playable(name: str, choice: Dict[str, str], seed: int) -> Optional[str]:
    """Step the world twice: None if it plays, else the error's last line."""
    out = f"{SMOKE}/{name}"
    r = subprocess.run(command(out, choice, seed, iters=2, ckpt=0, health=0), shell=True, cwd=HERE,
                       capture_output=True, text=True)
    shutil.rmtree(os.path.join(HERE, out), ignore_errors=True)
    return None if r.returncode == 0 else ((r.stderr or r.stdout).strip().splitlines() or ["?"])[-1]


def sample(state: Dict[str, Any], round_no: int, n: int) -> List[str]:
    rng = random.Random(1000 + round_no)
    jobs, pending = [], [(f"r{round_no}-{i:02d}-s{round_no * 1000 + i}", round_no * 1000 + i) for i in range(1, n + 1)]
    while pending:
        tries = [(name, seed, {s: rng.choice(state["allowed"][s]) for s, _ in SPACE}) for name, seed in pending]
        with ThreadPoolExecutor(4) as pool:
            errors = list(pool.map(lambda t: playable(t[0], t[2], t[1]), tries))
        if all(errors):
            # Not one world played: the fault is not in a combination.
            raise SystemExit(f"no world of round {round_no} plays: {errors[0]}")
        pending = []
        for (name, seed, choice), error in zip(tries, errors):
            if error:
                state["unplayable"].append({"round": round_no, "choice": choice, "error": error})
                pending.append((name, seed))
                continue
            state["worlds"].append({"id": f"Z/{name}", "round": round_no, "seed": seed, "choice": choice})
            jobs.append(command(f"{OUT}/{name}", choice, seed))
    with open(os.path.join(HERE, f"jobs_Z{round_no}.txt"), "w") as f:
        f.write(f"# Batch Z, round {round_no}: settings drawn at random from those still allowed (search.py).\n"
                + "\n".join(sorted(jobs, key=lambda j: j.split()[2])) + "\n")
    state["rounds"] = max(state["rounds"], round_no)
    return jobs


def running() -> set:
    """The world folders a run.py is still writing."""
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    return {os.path.realpath(os.path.join(HERE, line.split()[2])) for line in ps
            if line.startswith("python3 run.py ") and len(line.split()) > 2}


def outcome(wid: str, busy: set) -> Optional[str]:
    """'died', 'lived', 'stopped' (it broke off), or None while it is still to run or running."""
    d = os.path.join(PILOTS, wid)
    series = os.path.join(d, "series.jsonl")
    if not os.path.exists(series) or os.path.realpath(d) in busy:
        return None
    with open(series) as f:
        lines = f.read().splitlines()
    if not lines:
        return "stopped"
    last = json.loads(lines[-1])
    if last["nodes"] < 2:
        return "died"
    return "lived" if last["it"] >= ITERS else "stopped"


def scores(state: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Each finished world's score, and what became of it."""
    sys.path.insert(0, ROOT)
    from book_chapters.healthy import _load
    profiled = {w["id"]: w for w in _load()}
    busy = running()
    score, fate = {}, {}
    for w in state["worlds"]:
        what = outcome(w["id"], busy)
        if what is None:
            continue
        fate[w["id"]] = what
        if what == "died":
            score[w["id"]] = 0.0
        elif what == "lived" and w["id"] in profiled:
            p = profiled[w["id"]]
            score[w["id"]] = (p["marks"] / 7.0 + p["diversity"]) / 2.0
    return score, fate


def effects(state: Dict[str, Any], score: Dict[str, float], resamples: int = 20000) -> List[Dict[str, Any]]:
    """For every value drawn so far: its effect on the score and its interval (see the module's docstring)."""
    import numpy as np
    worlds = [w for w in state["worlds"] if w["id"] in score]
    if len(worlds) < 8:
        return []
    y = np.array([score[w["id"]] for w in worlds])
    rounds = np.array([w["round"] for w in worlds])
    rng = np.random.default_rng(7)
    # One resampling of the worlds within each round, shared by every value.
    members = {r: np.flatnonzero(rounds == r) for r in np.unique(rounds)}
    drawn = {r: m[rng.integers(0, len(m), (resamples, len(m)))] for r, m in members.items()}

    def weighted(has, index):
        """Within one round (or each resample of it): the difference of means times its weight, and the weight."""
        h = has[index]
        n1 = h.sum(-1)
        n0 = h.shape[-1] - n1
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = (y[index] * h).sum(-1) / n1 - (y[index] * ~h).sum(-1) / n0
        weight = np.where((n1 > 0) & (n0 > 0), n1 * n0 / h.shape[-1], 0.0)
        return np.nan_to_num(diff) * weight, weight

    found = []
    for setting, values in SPACE:
        labels = np.array([w["choice"][setting] for w in worlds])
        for label, _, _ in values:
            has = labels == label
            if has.sum() < 2 or (~has).sum() < 2:
                continue
            point = [weighted(has, m) for m in members.values()]
            total = sum(w for _, w in point)
            if not total:
                continue
            boot = [weighted(has, m) for m in drawn.values()]
            with np.errstate(invalid="ignore", divide="ignore"):
                dist = sum(d for d, _ in boot) / sum(w for _, w in boot)
            found.append({"setting": setting, "value": label, "worlds": int(has.sum()),
                          "effect": float(sum(d for d, _ in point) / total), "dist": dist[np.isfinite(dist)]})
    for e in found:
        dist = e.pop("dist")
        e["low"], e["high"] = float(np.quantile(dist, 0.025 / len(found))), float(np.quantile(dist, 1 - 0.025 / len(found)))
    return found


def judge(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    score, fate = scores(state)
    found = effects(state, score)
    dropped = []
    for e in sorted(found, key=lambda e: e["high"]):
        allowed = state["allowed"][e["setting"]]
        if e["high"] < 0 and e["value"] in allowed and len(allowed) > 1:
            allowed.remove(e["value"])
            dropped.append(dict(e, round=state["rounds"], when=time.strftime("%Y-%m-%d %H:%M")))
    state["dropped"] += dropped
    write_log(state, score, fate, found)
    return dropped


def write_log(state: Dict[str, Any], score: Dict[str, float], fate: Dict[str, str], found: List[Dict[str, Any]]) -> None:
    counts = {k: list(fate.values()).count(k) for k in ("lived", "died", "stopped")}
    lines = ["# Settings by elimination (batch Z)", "",
             "Made by `research/oee/pilots/search.py`, whose docstring says how. Worlds of B2 at 10,000 tokens with",
             "every setting drawn at random from the values the book has used, 1,000 iterations each. A world's",
             "score is the mean of the share of the health bar's seven marks it meets and its diversity score",
             "(Meta X); a world that died scores 0. A value whose effect on the score lies below zero across its",
             "whole interval (95% for all values together) is worse, and is drawn no more.", "",
             f"Rounds: {state['rounds']}. Worlds finished: {sum(counts.values())} ({counts['lived']} lived, "
             f"{counts['died']} died, {counts['stopped']} broke off); scored: {len(score)}. Combinations the code "
             f"could not play, drawn again: {len(state['unplayable'])}.", "",
             "## Values dropped as worse", ""]
    if state["dropped"]:
        lines += ["| after round | setting | value | worlds with it | effect on the score | interval |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {d['round']} | {d['setting']} | {d['value']} | {d['worlds']} | {d['effect']:+.3f} | "
                  f"{d['low']:+.3f} to {d['high']:+.3f} |" for d in state["dropped"]]
    else:
        lines.append("None yet.")
    lines += ["", "## Every value, as it stands", "",
              "| setting | value | worlds with it | effect on the score | interval | still drawn |",
              "|---|---|---|---|---|---|"]
    for e in sorted(found, key=lambda e: e["effect"]):
        lines.append(f"| {e['setting']} | {e['value']} | {e['worlds']} | {e['effect']:+.3f} | {e['low']:+.3f} to "
                     f"{e['high']:+.3f} | {'yes' if e['value'] in state['allowed'][e['setting']] else 'no'} |")
    choice = {w["id"]: w["choice"] for w in state["worlds"]}
    lines += ["", "## The best worlds so far", "", "| world | score | what differs from B2 |", "|---|---|---|"]
    for wid, s in sorted(score.items(), key=lambda kv: -kv[1])[:12]:
        differ = [f"{k}: {v}" for k, v in choice[wid].items() if v != FIRST[k]]
        lines.append(f"| `{wid}` | {s:.3f} | {'; '.join(differ) or 'nothing'} |")
    if state["unplayable"]:
        lines += ["", "## Combinations the code could not play", ""]
        lines += sorted({f"- {u['error']}" for u in state["unplayable"]})
    with open(LOG, "w") as f:
        f.write("\n".join(lines) + "\n")


def loop(rounds: int, n: int, slots: int) -> None:
    for _ in range(rounds):
        state = load_state()
        round_no = state["rounds"] + 1
        sample(state, round_no, n)
        save_state(state)
        print(f"round {round_no}: {n} worlds queued ({time.strftime('%Y-%m-%d %H:%M')})", flush=True)
        with open(os.path.join(HERE, f"jobq_Z{round_no}.log"), "w") as log:
            subprocess.run(["python3", "jobq.py", f"jobs_Z{round_no}.txt", str(slots)], cwd=HERE, stdout=log,
                           stderr=subprocess.STDOUT)
        subprocess.run(["nice", "-n", "10", "python3", os.path.join(ROOT, "research", "atlas", "profile.py"),
                        "--jobs", "2"], cwd=ROOT)
        state = load_state()
        dropped = judge(state)
        save_state(state)
        print(f"round {round_no} judged ({time.strftime('%Y-%m-%d %H:%M')}): dropped "
              + (", ".join(f"{d['setting']} = {d['value']}" for d in dropped) or "nothing"), flush=True)
    print("search done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("sample", "judge", "loop"))
    ap.add_argument("round", nargs="?", type=int)
    ap.add_argument("--n", type=int, default=ROUND)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--slots", type=int, default=SLOTS)
    a = ap.parse_args()
    if a.what == "loop":
        return loop(a.rounds, a.n, a.slots)
    state = load_state()
    if a.what == "sample":
        sample(state, a.round or state["rounds"] + 1, a.n)
    else:
        print(judge(state))
    save_state(state)


if __name__ == "__main__":
    main()
