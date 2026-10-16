# -*- coding: utf-8 -*-
"""
Appendix C · Every simulation: what was run, with which settings, and what was
measured on it. Made from the records themselves — the lab's plans
(book/experiments/), every pilot world's config.json, the job lists, and the
result files of every measure — so that it stays complete as worlds are added.

The tables are stored in book/results/catalogue.json and written into the
appendix by book_fill's `table` blocks.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Set, Tuple

import book_figures as F
from book_figures import chapter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = os.path.join(ROOT, "book")
RUNS = os.path.join(ROOT, "GraphOfLifeRuns")
PILOTS = os.path.join(RUNS, ".pilots")
JOBS = os.path.join(ROOT, "research", "oee", "pilots")
#: Settings that say how a world is kept, not what happens in it.
BOOKKEEPING = {"seed", "total_tokens", "extinction_threshold", "archive_every", "checkpoint_every", "export_every",
               "export_decisions"}

#: What each pilot batch asked, and where the book answers it.
BATCHES: Dict[str, Tuple[str, List[str]]] = {
    "A3k": ("Alliances, kin pooling and a stake gain, first tried at 3,000 tokens (kept only as hints: the book's "
            "worlds have at least 10,000 tokens since)", ["56"]),
    "A10k": ("Cooperation with a surplus: alliances by mutual consent, sides pooled by kinship, support that counts "
             "twice, against a control", ["56"]),
    "AL": ("Alliances over a long time and at 20,000 tokens", ["56"]),
    "N": ("Rules without numbers (B3, with and without allies) and alliances without the home advantage", ["56", "57"]),
    "S": ("The switches the world was built with (revolutions, gifts, lapsing links, estates by wealth) and other "
          "contests (a lottery, sides pooled by genotype or family, kin and family senses)", ["54", "55"]),
    "L": ("Long runs: Experiment 16's B2 worlds played on to 10,000 iterations, B1 to 10,000, B1 without "
          "revolutions to 3,000", ["50", "69"]),
    "P": ("Two inputs (where my brain came from, who helped me) and links that hold a token", ["57", "58"]),
    "Q": ("Brains mixed at birth, brain units that hold a token, an eye for raids on kin, and links that hold a "
          "token without old age", ["78"]),
    "R": ("Who inherits the estate of the dead: the poorest, the newborns, their parents", ["73"]),
    "T": ("Gifts together with each other switch", []),
    "U": ("Eight switches in combination on B2: a fractional factorial design", []),
    "V": ("Eight switches in combination on B1: a fractional factorial design", []),
    "W": ("No revolutions together with each other switch, on B2", []),
    "X": ("Handover together with each other switch, on B2", []),
}
#: The measures, by what to call them; each finds the worlds it was taken on.
MEASURES = ["the record of every game", "lifetime contests (T = 100)", "games recorded and read again",
            "flow modules", "how different the agents are", "the table of every world"]


def _json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _chapters() -> Dict[str, Dict[str, Any]]:
    """Every written chapter, by id."""
    out = {}
    for part in _json(os.path.join(BOOK, "book.json"))["parts"]:
        for c in part["chapters"]:
            out[c["id"]] = c
    return out


def _cell_chapters(ids: List[str], chapters: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"text": i, "href": chapters[i]["file"]} for i in ids if i in chapters and chapters[i].get("file")]


def _show(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


# ---------------------------------------------------------------------------
# What was measured where
# ---------------------------------------------------------------------------

def _pilot_id(path: str) -> str:
    """'GraphOfLifeRuns/.pilots/S/S2-gifts-s1' (or a spec 'pilot:…:1000') -> 'S/S2-gifts-s1'."""
    tail = path.split(".pilots/")[-1]
    return tail.split(":")[0].rstrip("/")


def measured() -> Dict[str, Set[str]]:
    """For every measure, the worlds it was taken on: lab run ids and pilot ids ('batch/name')."""
    out: Dict[str, Set[str]] = {m: set() for m in MEASURES}
    for d in os.listdir(RUNS):
        if os.path.exists(os.path.join(RUNS, d, "stats.jsonl")):
            out["the record of every game"].add(d)
    for path in glob.glob(os.path.join(PILOTS, "*", "*", "series.jsonl")):
        out["the record of every game"].add(_pilot_id(os.path.dirname(path)))
    for path in glob.glob(os.path.join(PILOTS, "*T100*.json")) + [os.path.join(PILOTS, "long_steps.json")]:
        try:
            rows = _json(path)
        except (OSError, ValueError):
            continue
        for c in rows if isinstance(rows, list) else []:
            if isinstance(c, dict) and int(c.get("T", 0) or 0) == 100:
                w = c.get("world") or c.get("run") or ""
                out["lifetime contests (T = 100)"].add(_pilot_id(w) if ".pilots/" in w else os.path.basename(w))
    for path in glob.glob(os.path.join(PILOTS, "frames_*.json")):
        name = os.path.basename(path)[len("frames_"):-len(".json")]
        name = re.sub(r"_\d+$", "", name)
        batch, _, world = name.partition("_")
        if world:
            out["games recorded and read again"].add(f"{batch}/{world}")
    for label in ("structures_B1.json", "structures_B2.json"):
        path = os.path.join(PILOTS, label)
        if os.path.exists(path):
            for run in re.findall(r'"(B[12]-10000[^"]*-s\d{3})"', open(path).read()):
                out["games recorded and read again"].add(run)
    for path in glob.glob(os.path.join(PILOTS, "atlas", "flowmodules", "*.json")):
        r = _json(path)
        if (r.get("floor") or {}).get("summary"):
            out["flow modules"].add(r["id"])
    for path in glob.glob(os.path.join(PILOTS, "diversity", "*.json")):
        for w in _json(path).get("worlds") or []:
            spec = w["world"]
            out["how different the agents are"].add(_pilot_id(spec) if spec.startswith("pilot:")
                                                     else spec.split(":")[0])
    table = os.path.join(BOOK, "results", "worlds.json")
    if os.path.exists(table):
        out["the table of every world"] = {w["id"] for w in _json(table)["worlds"]}
    return out


# ---------------------------------------------------------------------------
# The lab's experiments
# ---------------------------------------------------------------------------

def lab_rows(chapters, done) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import gol_plan
    by_experiment = {c.get("experiment"): c["id"] for c in chapters.values() if c.get("experiment")}
    base = {b: _json(os.path.join(BOOK, "experiments", f"{b}.json"))["settings"] for b in ("B1", "B2")}
    rows, measures = [], []
    for path in sorted(glob.glob(os.path.join(BOOK, "experiments", "E*.json")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)))):
        plan = _json(path)
        name = plan["id"]
        spec, borrowed = plan.get("runs"), None
        while isinstance(spec, str):
            borrowed = spec.replace("same as ", "")
            spec = _json(os.path.join(BOOK, "experiments", f"{borrowed}.json"))["runs"]
        try:
            ids = [r.run_id for r in gol_plan.experiment_runs(borrowed or name)]
        except Exception:
            ids = []
        present = [i for i in ids if os.path.exists(os.path.join(RUNS, i, "meta.json"))]
        conditions = []
        for c in spec["conditions"]:
            changed = ", ".join(f"{k} = {_show(v)}" for k, v in (c.get("set") or {}).items())
            conditions.append(f"*{c['name']}*" + (f" ({changed})" if changed else ""))
        tokens = spec["world"]["total_tokens"]
        rows.append([name, plan["title"], spec["baseline"],
                     ("the runs of " + borrowed + "; " if borrowed else "") + "; ".join(conditions),
                     str(spec["seeds"]).replace("..", "–"),
                     f"{tokens:,}" if not isinstance(tokens, list) else f"{tokens[0]:,}–{tokens[-1]:,}",
                     f"{spec['iterations']:,}", str(len(present)),
                     _cell_chapters([by_experiment[name]] if name in by_experiment else [], chapters)])
        measures.append([name] + [str(sum(i in done[m] for i in present)) for m in MEASURES])
    head = ["", "question", "on", "conditions (and what each changes)", "seeds", "tokens", "iterations",
            "runs on disk", "chapter"]
    return ({"head": head, "rows": rows},
            {"rows": measures})


# ---------------------------------------------------------------------------
# The pilot batches
# ---------------------------------------------------------------------------

def _last_iteration(d: str) -> int:
    path = os.path.join(d, "series.jsonl")
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        try:
            f.seek(-4000, 2)
        except OSError:
            f.seek(0)
        lines = f.read().decode(errors="ignore").strip().splitlines()
    for line in reversed(lines):
        try:
            return int(json.loads(line).get("it", 0))
        except ValueError:
            continue
    return 0


def _running() -> Set[str]:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return {os.path.realpath(os.path.join(JOBS, line.split()[2]))
            for line in out.splitlines() if line.startswith("python3 run.py ") and len(line.split()) > 2}


def _planned(batch: str) -> Dict[str, Tuple[int, int]]:
    """The worlds a batch's job lists hold, each once: name -> (tokens, iterations)."""
    out: Dict[str, Tuple[int, int]] = {}
    for path in glob.glob(os.path.join(JOBS, "jobs_*.txt")):
        for line in open(path):
            if not line.startswith("python3 run.py") or f"/.pilots/{batch}/" not in line:
                continue
            words = line.split()
            name = os.path.basename(words[2].rstrip("/"))
            tokens = re.search(r"total_tokens=(\d+)", line)
            iters = re.search(r"--iters (\d+)", line)
            out[name] = (int(tokens.group(1)) if tokens else 0, int(iters.group(1)) if iters else 0)
    return out


def pilot_rows(chapters, done) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    base = {b: _json(os.path.join(BOOK, "experiments", f"{b}.json"))["settings"] for b in ("B1", "B2")}
    busy = _running()
    batches: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
    for path in sorted(glob.glob(os.path.join(PILOTS, "*", "*", "config.json"))):
        d = os.path.dirname(path)
        batch, name = d.split(os.sep)[-2:]
        if batch not in BATCHES:
            continue
        batches.setdefault(batch, []).append((name, d, _json(path)))
    rows, details, measures = [], [], []
    for batch in BATCHES:
        worlds = batches.get(batch) or []
        jobs = _planned(batch)
        names = set(jobs) | {name for name, _, _ in worlds}
        if not names:
            continue
        conditions: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
        for name, d, meta in worlds:
            conditions.setdefault(re.sub(r"-s\d+$", "", name), []).append((name, d, meta))
        # A job that only goes on from a checkpoint names no size: it is the size it was.
        tokens = sorted(({int(m["cfg"].get("total_tokens") or 0) for _, _, m in worlds} | {t for t, _ in jobs.values()}) - {0})
        iters = sorted({int(m.get("iters") or 0) for _, _, m in worlds if m.get("iters")} | {i for _, i in jobs.values()})
        done_n = running_n = stopped_n = 0
        for name, d, m in worlds:
            if os.path.realpath(d) in busy:
                running_n += 1
            elif _last_iteration(d) >= int(m.get("iters") or 0) > 0:
                done_n += 1
            else:
                stopped_n += 1
        waiting = len(names) - len(worlds)
        if done_n == len(names):
            status = "done"
        else:
            parts = [f"{done_n} done"] + ([f"{running_n} running"] if running_n else []) \
                + ([f"{waiting} queued"] if waiting else []) + ([f"{stopped_n} stopped early"] if stopped_n else [])
            status = ", ".join(parts)
        planned = len(names)
        for cond, members in sorted(conditions.items()):
            meta = members[0][2]
            cfg, proto = meta["cfg"], {k: v for k, v in (meta.get("proto") or {}).items() if k != "engine"}
            engine = bool((meta.get("proto") or {}).get("engine"))
            on = "B2" if cfg.get("mend_on_death") else "B1"
            changed = [f"{k} = {_show(v)}" for k, v in sorted(cfg.items())
                       if k not in BOOKKEEPING and k in base[on] and v != base[on][k]]
            changed += [f"{k} = {_show(v)}" for k, v in sorted(proto.items())]
            seeds = sorted(int(m["cfg"].get("seed") or 0) for _, _, m in members)
            details.append([batch, cond, on + ("" if engine else " (the prototype's own code)"),
                            ", ".join(changed) or f"nothing: {on} itself",
                            ", ".join(str(s) for s in seeds), f"{int(meta.get('iters') or 0):,}"])
        rows.append([batch, BATCHES[batch][0], str(len({re.sub(r"-s\d+$", "", n) for n in names})), str(planned),
                     ", ".join(f"{t:,}" for t in tokens), ", ".join(f"{i:,}" for i in iters), status,
                     _cell_chapters(BATCHES[batch][1], chapters)])
        ids = [f"{batch}/{name}" for name, _, _ in worlds]
        measures.append([batch] + [str(sum(i in done[m] for i in ids)) for m in MEASURES])
    return ({"head": ["batch", "what it asks", "conditions", "worlds", "tokens", "iterations", "status", "chapter"],
             "rows": rows},
            {"head": ["batch", "condition", "on", "what it changes", "seeds", "iterations"], "rows": details},
            {"rows": measures})


def baselines() -> Dict[str, Any]:
    """The settings in which the two rule sets differ, with what each is (book/settings.json)."""
    b1 = _json(os.path.join(BOOK, "experiments", "B1.json"))["settings"]
    b2 = _json(os.path.join(BOOK, "experiments", "B2.json"))["settings"]
    glossary = _json(os.path.join(BOOK, "settings.json"))
    order = glossary["order"] + [k for k in b1 if k not in glossary["order"]]
    rows = []
    for key in order:
        if key in BOOKKEEPING or b1.get(key) == b2.get(key):
            continue
        rows.append([{"text": f"`{key}`", "href": f"notes/settings.md#{key}"}, _show(b1.get(key)), _show(b2.get(key)),
                     (glossary["settings"].get(key) or {}).get("meaning", "")])
    return {"head": ["setting", "B1", "B2", "what it is"], "rows": rows}


@chapter
def catalogue(ch: F.Chapter) -> None:
    chapters = _chapters()
    done = measured()
    ch.number("table:baselines", baselines())
    lab, lab_measures = lab_rows(chapters, done)
    pilots, pilot_details, pilot_measures = pilot_rows(chapters, done)
    ch.number("table:lab", lab)
    ch.number("table:pilots", pilots)
    ch.number("table:pilot_conditions", pilot_details)
    ch.number("table:measures", {"head": ["", *MEASURES],
                                 "rows": lab_measures["rows"] + pilot_measures["rows"]})
    ch.number("totals", {"lab experiments": len(lab["rows"]),
                         "lab runs on disk": sum(int(r[7]) for r in lab["rows"]),
                         "pilot batches": len(pilots["rows"]),
                         "pilot worlds": sum(int(r[3]) for r in pilots["rows"])})
