"""
Round 3's candidates, compared and ruled out one by one (the author,
2026-10-28). Every world of the round is a candidate for open-ended evolution;
this reads each one's dossier (research/screen/candidates.py), rules out the
ones least likely to evolve open-endedly, each for a reason it states, and
ranks the rest in every category, with no one verdict over them all.

The rules and the categories are lists below, like the judge's lines: what
each asks is in words beside it, and moving one changes nothing else.

Rules (a world is out at the first it meets):
  died          it did not live to its end
  a star        one agent is linked to more than half the world, on average
  one holding   80% or more of the agents hold the same number of tokens
  no heredity   a child acts no more like its parent than a stranger does
                (a heredity index below 0.1 in both kept worlds read)
  standing      its line of descent hardly moves from era to era, and two
  still         agents of one era hardly act differently (both less than
                MIN_NEW above the noise): nothing varies for selection to act on

The four pillars of evolution that the categories read (a world that misses
one cannot evolve open-endedly however long it runs):
  persistence   it lives, at a size and steadiness that let it go on
  heredity      children are like their parents
  selection     reproduction is decided by what agents are, not by chance:
                the larger the effective population, the smaller a
                difference selection can see
  novelty       its line of descent keeps reaching behaviour it has not
                shown before, rather than standing still or coming back

    python3 research/screen/funnel.py E40        prints the table and writes research/screen/E40_candidates.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import gol_plan  # noqa: E402
import gol_screen  # noqa: E402
import gol_store  # noqa: E402

import candidates as K  # noqa: E402
import criteria as C  # noqa: E402

#: An era of the line of descent comes back to earlier behaviour when it lies closer to an earlier era than
#: this many times a brain lies from itself between two tries, or than MIN_NEW where brains draw no chance
#: and lie at no distance from themselves. Distances run from 0 (acting alike) to 1.
NEW, MIN_NEW = 2.0, 0.05
#: The heredity index below which a child acts like a stranger.
NO_HEREDITY = 0.1


# ---------------------------------------------------------------------------
# What a world's dossier says, in numbers
# ---------------------------------------------------------------------------

def _mean(values: List[Optional[float]]) -> Optional[float]:
    have = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(have)) if have else None


def numbers(w: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """The numbers a world is ruled out and ranked by, from its dossier."""
    d, h = w["dossier"], w["dossier"].get("health") or {}
    out: Dict[str, Optional[float]] = {k: h.get(k) for k in (
        "agents", "hub", "links", "core", "agent_cut", "link_cut", "dimension", "gini", "max_tokens",
        "tokens_alike", "ages_alike", "genotypes_alike", "core_links_alike", "children_alike", "children")}
    out["tokens_per_agent"] = (w["tokens"] / h["agents"]) if h.get("agents") else None
    lin = d.get("lineage") or {}
    gens = (lin.get("generations") or {}).get(str(w["until"])) or {}
    out["generations"] = gens.get("mean")
    out["iterations_per_generation"] = (w["until"] / gens["mean"]) if gens.get("mean") else None
    out["ancestry_generations"] = lin.get("common_ancestor_generations")
    out["effective_population"] = effective_population(w, lin)
    out["heredity"] = _mean([(d.get(f"heredity {at}") or {}).get("index") for at in K.HEREDITY_AT])
    line = d.get("line")
    if line:
        D = np.array(line["distance"])
        eras = line["eras"]
        steps = [D[i, i + 1] for i in range(len(eras) - 1)]
        novelty = [min(D[i, j] for j in range(i)) for i in range(1, len(eras))]
        late = [v for e, v in zip(eras[1:], novelty) if e > eras[-1] / 2]
        floor = line["itself"]
        # Every distance is read in noisy draws, so a brain lies `floor` from itself: what is above it is behaviour.
        out["floor"] = floor
        out["step"] = float(np.median(steps)) - floor
        out["late_novelty"] = float(np.mean(late)) - floor if late else None
        out["revisits"] = float(sum(v < max(NEW * floor, MIN_NEW) for v in novelty))
        out["contemporaries"] = line["strangers"] - floor
    return out


def effective_population(w: Dict[str, Any], lin: Dict[str, Any]) -> Optional[float]:
    """
    A rough effective population, from how fast the ancestry of the living narrows: where k of their
    ancestors are found t generations back, chance alone would leave about 2N / t of them in a population of
    N, so N is about k t / 2. Read at the furthest kept world back where more than one ancestor is left, or at
    the one ancestor of all of them. An order of magnitude, assuming chance alone narrows the ancestry.
    """
    gens = lin.get("generations") or {}
    end = (gens.get(str(w["until"])) or {}).get("mean")
    if not end:
        return None
    if lin.get("common_ancestor_generations"):
        return lin["common_ancestor_generations"] / 2
    back = sorted((int(at) for at in gens if int(at) < w["until"]), key=int)
    if not back:
        return None
    at = back[0]
    return lin["ancestors"][str(at)] * (end - gens[str(at)]["mean"]) / 2


def capability(full: Dict[str, Any]) -> Dict[str, Any]:
    """What a world's rules make possible in principle: the channels its agents can act through, and its brain."""
    channels = {
        "messages": full["message_amount"] > 0, "gifts": bool(full["allow_gifting"]),
        "handover": bool(full["allow_handover"]), "mates pay": bool(full["mix_at_birth"] and full["mate_pays"]),
        "partner search": full["link_search"] != "off", "upkeep": full["link_upkeep"] != "off",
        "revolutions": bool(full["allow_revolutions"] and full["contest"] == "largest"),
        "mixing": bool(full["mix_at_birth"]),
    }
    layers = full["hidden_layers"]
    return {"channels": [k for k, v in channels.items() if v], "open": sum(channels.values()),
            "hidden_units": int(sum(layers)), "layers": len(layers),
            "chance": full["contest"] == "lottery", "estate": full["redistribution"],
            "valence": full["valence"], "germline": bool(full["mutate_on_replication"])}


def notes(w: Dict[str, Any], n: Dict[str, Optional[float]], cap: Dict[str, Any]) -> List[str]:
    """Where a measure can say little for this world, and why."""
    out = []
    if cap["layers"] and "mixing" in cap["channels"]:
        out.append("brains mixed at birth: the family tree follows the paying parent; a child's brain has two")
    if not cap["germline"]:
        out.append("brains change every iteration, not only at birth: a genotype lasts briefly, so behaviour "
                   "is measured instead")
    if (n.get("tokens_per_agent") or 99) < 3 or (n.get("tokens_alike") or 0) >= 0.6:
        out.append("most agents hold a token or two: their stakes can hardly differ, and the behaviour "
                   "measures say little")
    if n.get("agents") is not None and n["agents"] < 100:
        out.append("under a hundred agents: no width or same-position share can be read")
    if cap["chance"]:
        out.append("the lottery: who wins a node is drawn by lot, so selection is weaker by construction")
    return out


# ---------------------------------------------------------------------------
# Rules and categories
# ---------------------------------------------------------------------------

Rule = Callable[[Dict[str, Any], Dict[str, Optional[float]]], Optional[str]]


def _died(w, n):
    return f"died at iteration {w['ended_at']:,}" if w["fate"] == "died" else None


def _star(w, n):
    return (f"a star: one agent linked to {n['hub']:.1%} of the others on average"
            if (n.get("hub") or 0) > 0.5 else None)


def _one_holding(w, n):
    return (f"one holding: {n['tokens_alike']:.0%} of the agents hold the same number of tokens"
            if (n.get("tokens_alike") or 0) >= 0.8 else None)


def _no_heredity(w, n):
    return (f"no heredity: children act like strangers (index {n['heredity']:.2f})"
            if n.get("heredity") is not None and n["heredity"] < NO_HEREDITY else None)


def _still(w, n):
    return ("standing still: its line of descent does not move and its agents act alike "
            f"(steps {n['step']:.2f}, contemporaries {n['contemporaries']:.2f} above the noise)"
            if n.get("step") is not None and n["step"] < MIN_NEW and n["contemporaries"] < MIN_NEW else None)


RULES: List[Rule] = [_died, _star, _one_holding, _no_heredity, _still]

#: The categories: a key, the pillar it reads, the number, whether more is better, and what it says.
CATEGORIES = [
    ("size", "persistence", "agents", True, "agents on average from iteration 100"),
    ("generations", "persistence", "generations", True, "generations lived by the end"),
    ("heredity", "heredity", "heredity", True, "how like their parents children act (0 strangers, 1 parent)"),
    ("population", "selection", "effective_population", True,
     "a rough effective population: the smaller, the larger a difference selection needs to see"),
    ("novelty", "novelty", "late_novelty", True,
     "how far the line of descent's later eras lie from every earlier one (0 acting alike, 1 wholly unlike)"),
    ("contemporaries", "variation", "contemporaries", True, "how far apart two agents of one era act"),
    ("health", "health", "health_lines", True, "the author's lines met, of those read"),
    ("riches", "economy", "tokens_per_agent", True, "tokens per agent"),
]


def worlds_of(name: str) -> List[Dict[str, Any]]:
    """Every finished world of a round, with its draw, how it ended, and its dossier."""
    plan = gol_plan.read_plan(name)
    version = int(plan["screen"]["sampler"])
    drawn = {s[0] for s in gol_screen.space(version)}
    out = []
    for spec in gol_plan.experiment_runs(name):
        if not os.path.exists(os.path.join(gol_store.run_dir(spec.run_id), "meta.json")):
            continue
        meta = gol_store.load_meta(spec.run_id)
        at = int(meta.get("iteration") or 0)
        fate = "died" if meta.get("status") == "extinct" else ("lived" if at >= spec.until else None)
        if fate is None:
            continue
        changed = {**spec.lab["set"], **{k: v for k, v in spec.lab["world"].items() if k in drawn}}
        full = {**gol_plan.baseline("B1"), **changed}
        out.append({"id": spec.run_id, "seed": spec.lab["seed"], "fate": fate, "ended_at": at, "until": spec.until,
                    "tokens": spec.lab["world"]["total_tokens"], "set": changed, "full": full,
                    "dossier": K.kept(spec.run_id)})
    return out


def judge(name: str) -> Dict[str, Any]:
    worlds = worlds_of(name)
    for w in worlds:
        n = numbers(w)
        verdict = C.judged(n, w["fate"] == "lived")
        read = [line for line in C.LINES if n.get(line["measure"]) is not None]
        n["health_lines"] = float(sum(verdict["lines"][line["key"]] for line in read)) if read else None
        w["numbers"], w["lines_read"] = n, len(read)
        w["capability"] = capability(w["full"])
        w["notes"] = notes(w, n, w["capability"])
        w["out"] = next((reason for rule in RULES if (reason := rule(w, n))), None)
    left = [w for w in worlds if w["out"] is None]
    ranks = {}
    for key, _, measure, more, _ in CATEGORIES:
        have = sorted((w for w in left if w["numbers"].get(measure) is not None),
                      key=lambda w: w["numbers"][measure], reverse=more)
        ranks[key] = [w["id"] for w in have]
        for i, w in enumerate(have, 1):
            w.setdefault("ranks", {})[key] = i
    return {"round": name, "worlds": worlds, "ranks": ranks,
            "rules": [r.__name__.strip("_").replace("_", " ") for r in RULES],
            "categories": [{"key": k, "pillar": p, "measure": m, "more_is_better": b, "says": s}
                           for k, p, m, b, s in CATEGORIES]}


def _f(v: Optional[float], form: str = "{:.2f}") -> str:
    return "-" if v is None else form.format(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("round")
    a = ap.parse_args()
    out = judge(a.round)
    with open(os.path.join(HERE, f"{a.round}_candidates.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    worlds = out["worlds"]
    gone = [w for w in worlds if w["out"]]
    print(f"{len(worlds)} worlds finished; {len(gone)} ruled out, {len(worlds) - len(gone)} candidates left\n")
    for w in sorted(gone, key=lambda w: w["seed"]):
        print(f"  out  s{w['seed']}  {w['out']}")
    print("\ncandidates (rank in each category; 1 is best):")
    keys = [c[0] for c in CATEGORIES]
    print("  world    " + " ".join(f"{k[:8]:>8s}" for k in keys) + "   channels")
    left = [w for w in worlds if not w["out"]]
    for w in sorted(left, key=lambda w: np.mean([w.get("ranks", {}).get(k, len(left)) for k in keys])):
        print(f"  s{w['seed']}  " + " ".join(f"{w.get('ranks', {}).get(k, '-'):>8}" for k in keys)
              + f"   {w['capability']['open']} ({', '.join(w['capability']['channels'])})")
    print("\nthe numbers:")
    for w in sorted(left, key=lambda w: w["seed"]):
        n = w["numbers"]
        print(f"  s{w['seed']} agents {_f(n['agents'], '{:,.0f}'):>7s}  generations {_f(n['generations'], '{:.0f}'):>5s}"
              f" ({_f(n['iterations_per_generation'], '{:.1f}')} it/gen)  ancestry {_f(n['ancestry_generations'], '{:.1f}'):>5s} gen"
              f"  N_e~{_f(n['effective_population'], '{:.0f}')}"
              f"  heredity {_f(n['heredity'])}  floor {_f(n.get('floor'), '{:.3f}')}  step {_f(n.get('step'), '{:.2f}')}"
              f"  novelty {_f(n.get('late_novelty'), '{:.2f}')}  revisits {_f(n.get('revisits'), '{:.0f}')}"
              f"  contemporaries {_f(n.get('contemporaries'), '{:.2f}')}"
              f"  lines {_f(n['health_lines'], '{:.0f}')}/{w['lines_read']}  tokens/agent {_f(n['tokens_per_agent'], '{:.1f}')}")
        for note in w["notes"]:
            print(f"        note: {note}")


if __name__ == "__main__":
    main()
