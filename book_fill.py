#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The parts of the book's text that are made rather than written.

    python3 book_fill.py          fills them in, in every Markdown file of book/

The book is plain Markdown and SVG, so that it reads the same in the Book
tab, in Obsidian (open book/ as a vault) and on GitHub. A few of its blocks
are made from the runs, the plans and book.json rather than typed, so that
they cannot drift from what they describe. Such a block sits between two
comment lines, which no reader shows:

    <!-- figure tokens/lorenz -->     the figure book_figures.py drew, what it
    …                                 shows, and how to make it again
    <!-- /figure -->

    <!-- thesis E04 -->               an experiment's claim, quoted from its plan
    <!-- runs E02 -->                 what was run: every setting, the seeds, the
                                      command that runs it again
    <!-- contents -->                 the chapters and notes, from book.json
    <!-- turns -->                    links to the chapter before and after
    <!-- costs -->                    what a simulation costs, as last fitted
    <!-- table chapter/name -->       a table a chapter's code made from the records

Everything between the opening and the closing line is replaced each time:
write only outside them. An opening line alone is filled in the first time.
book_figures.py fills the book after drawing its figures.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

BOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "book")

def read_json(path: str) -> Any:
    with open(os.path.join(BOOK, path), encoding="utf-8") as f:
        return json.load(f)


def link(md: str, target: str) -> str:
    """The path from the Markdown file `md` to `target`, both under book/."""
    return os.path.relpath(os.path.join(BOOK, target), os.path.dirname(os.path.join(BOOK, md))) \
        .replace(os.sep, "/")


def quote(text: str, callout: str) -> str:
    """Text as an Obsidian callout: `> [!kind] title` and every line quoted."""
    body = "\n".join(f"> {line}" if line.strip() else ">" for line in text.strip("\n").split("\n"))
    return f"> {callout}\n{body}"


def number(n: int) -> str:
    return f"{n:,}"


# ---------------------------------------------------------------------------
# The blocks
# ---------------------------------------------------------------------------

def linked(md: str, text: str) -> str:
    """Every "Chapter 13" in a text made by code, as a link to that chapter."""
    files = {c["id"]: c["file"] for c in chapters() if c.get("file")}

    def to(match: "re.Match[str]") -> str:
        target = files.get(match.group(1))
        return f"[{match.group(0)}]({link(md, target)})" if target else match.group(0)
    return re.sub(r"(?<!\[)\bChapter (\d+)\b(?!\]\()", to, text)


def figure(md: str, name: str) -> str:
    chapter, fig = name.split("/")
    made = read_json(f"results/{chapter}.json").get("figures", {}).get(fig)
    if made is None:
        raise KeyError(f"no figure {name}: run `python3 book_figures.py {chapter}`")
    return (f"![{made['title']}]({link(md, f'figures/{chapter}/{fig}.svg')})\n\n"
            f"**{made['title']}{'' if made['title'][-1:] in '.?!' else '.'}** {linked(md, made['caption'])}\n\n"
            + quote(linked(md, made["recipe"]), "[!example]- How to make this figure"))


def thesis(md: str, name: str) -> str:
    plan = read_json(f"experiments/{name}.json")
    said = plan.get("thesis") or {}
    parts = [("claim", "The claim."), ("why", "Why it would be so."),
             ("confirm", "It holds if:"), ("refute", "It fails if:")]
    text = "\n\n".join(f"**{label}** {said[key]}" for key, label in parts if said.get(key))
    return quote(text, f"[!quote] The thesis of Experiment {int(name[1:])}, "
                       "written down before any of its runs existed")


def resolve(name: str) -> tuple:
    """An experiment's own runs, or the runs of the one it borrows them from."""
    plan = read_json(f"experiments/{name}.json")
    borrowed = None
    while isinstance(plan.get("runs"), str):
        borrowed = plan["runs"].replace("same as ", "")
        plan = read_json(f"experiments/{borrowed}.json")
    return plan, borrowed


def show(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return ", ".join(show(v) for v in value)
    if isinstance(value, int):
        return number(value)
    return str(value)


def seeds_text(seeds: str) -> tuple:
    if ".." in str(seeds):
        a, b = (int(x) for x in str(seeds).split(".."))
        return f"{a} to {b}", b - a + 1
    listed = [s for s in str(seeds).replace(",", " ").split() if s]
    return ", ".join(listed), len(listed)


def trials(md: str, name: str, plan: Dict[str, Any]) -> str:
    """The trials behind a chapter: how many, of what kind, and what each set varies."""
    import gol_plan
    found = gol_plan.trials_of(name)
    kind, T = plan["trials"]["kind"], plan["trials"]["T"]
    what = ("contests: the world of one era, its agents given brains drawn by coin from two eras, "
            "every mutation off, and the later era's share of the agents counted at the end"
            if kind == "contest" else
            "invasions: a patch of the world of one era given a strategy written by hand, and "
            "its agents counted after every iteration")
    lines = [f"**{number(len(found))} trials**, {what}; each plays {number(T)} iterations, from a "
             "world another run kept in its archive.\n"]
    def span(values: List[Any]) -> str:
        """Numbers as a run when they are evenly spaced and many, else listed."""
        steps = {b - a for a, b in zip(values, values[1:])} if all(isinstance(v, int) for v in values) else set()
        if len(values) > 4 and len(steps) == 1:
            return f"{number(values[0])} to {number(values[-1])}, every {number(steps.pop())}"
        return ", ".join(show(v) for v in values)

    words = [("run", "worlds"), ("eras", "eras"), ("era", "at iteration"), ("strategy", "strategies"),
             ("size", "patches of"), ("place", "places"), ("background", "in a world of"),
             ("settle", "settled for"), ("seed", "coins")]
    for label, fields in gol_plan.trial_sets(name).items():
        count = sum(1 for f in found if f["set"] == label)
        if "a" in fields:
            fields["eras"] = sorted(set(fields.pop("a")) | set(fields.pop("b")))
            fields.pop("home", None)                   # every pair is played on both its eras' worlds
        said = "; ".join(f"{word} " + (", ".join(f"`{v}`" for v in fields[key]) if key == "run"
                                       else span(fields[key]))
                         for key, word in words if key in fields)
        lines.append(f"- **{label}** — {number(count)} trials: {said}.")
    lines.append("")
    lines.append(f"To play these trials again: `python3 gol_lab.py run {name}`, or ▶ in the Book tab "
                 "of a computer running `gol_server.py`. Each writes its settings and engine beside "
                 "its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.")
    return quote("\n".join(lines), "[!info] The trials behind this chapter")


def runs(md: str, name: str) -> str:
    import gol_plan                                    # only here: it reads the runs folder
    plan, borrowed = resolve(name)
    if "trials" in plan:
        return trials(md, borrowed or name, plan)
    spec = plan["runs"]
    base = read_json(f"experiments/{spec['baseline']}.json")
    glossary = read_json("settings.json")
    conditions = spec["conditions"]
    world = spec["world"]["total_tokens"]

    def sizes(c: Dict[str, Any]) -> List[int]:
        return list(c.get("sizes") or (world if isinstance(world, list) else [world]))

    seeds, count = seeds_text(spec["seeds"])
    ids: Dict[str, List[str]] = {}
    for r in gol_plan.experiment_runs(borrowed or name):
        ids.setdefault(r.condition, []).append(r.run_id)

    lines = []
    if borrowed:
        lines.append(f"These are the runs of Experiment {int(borrowed[1:])}; nothing new was run for "
                     "this chapter.\n")
    worlds = sum(len(sizes(c)) * count for c in conditions)
    lines.append(f"**{number(worlds)} runs.** Every condition below is run once for every seed "
                 f"({seeds})" + (" and every number of tokens it lists" if any(len(sizes(c)) > 1 for c in conditions)
                                 else "") + f", for {number(spec['iterations'])} iterations — or "
                 "until its world dies out. Every setting is that of the baseline "
                 f"**{base['id']}** ({base['title'].lower()}) unless the condition changes it.\n")
    for c in conditions:
        changed = ", ".join(f"`{k}` = {show(v)}" for k, v in (c.get("set") or {}).items()) \
            or f"nothing: {base['id']} as it is"
        named = ids.get(c["name"], [])
        where = f" Runs `{named[0]}` … `{named[-1]}`." if named else ""
        lines.append(f"- **{c['name']}** — changes {changed}; {', '.join(number(s) for s in sizes(c))} "
                     f"tokens.{where}")
    lines.append("")
    lines.append(f"To make these runs again: `python3 gol_lab.py run {borrowed or name}`, or ▶ in "
                 "the Book tab of a computer running `gol_server.py`. Each run writes its "
                 "settings, seed and engine version beside its frames, in "
                 "`GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.")
    summary = quote("\n".join(lines), "[!info] The runs behind this chapter")

    head = "| setting | " + " | ".join(c["name"] for c in conditions) + " | what it is |"
    rule = "|---" * (len(conditions) + 2) + "|"
    rows = []
    keys = glossary["order"] + [k for k in base["settings"] if k not in glossary["order"]]
    for key in keys:
        if key not in base["settings"]:
            continue
        cells = []
        for c in conditions:
            if key == "total_tokens":
                cells.append(", ".join(number(s) for s in sizes(c)))
                continue
            value = (c.get("set") or {}).get(key, spec.get("world", {}).get(key, base["settings"][key]))
            text = show(value)
            cells.append(f"**{text}**" if value != base["settings"][key] else text)
        meaning = (glossary["settings"].get(key) or {}).get("meaning", "")
        rows.append(f"| [`{key}`]({link(md, 'notes/settings.md')}#{key}) | " + " | ".join(cells)
                    + f" | {meaning} |")
    rows.append("| seeds | " + " | ".join(seeds for _ in conditions) + " | one run per seed |")
    rows.append("| iterations | " + " | ".join(number(spec["iterations"]) for _ in conditions)
                + " | how far each run goes, unless its world dies first |")
    table = "\n".join([head, rule, *rows])
    every = quote(table + "\n\nA value in **bold** differs from the baseline "
                  f"{base['id']}.", "[!info]- Every setting of these runs")
    return f"{summary}\n\n{every}"


def chapters() -> List[Dict[str, Any]]:
    return [c for part in read_json("book.json")["parts"] for c in part["chapters"]]


def heading(c: Dict[str, Any]) -> str:
    if c["id"].isdigit():
        return f"Chapter {c['id']} · {c['title']}"
    if re.fullmatch(r"[A-Z]", c["id"]):
        return f"Appendix {c['id']} · {c['title']}"
    return c["title"]


def contents(md: str, _: str) -> str:
    out = []
    for part in read_json("book.json")["parts"]:
        if all(c.get("file") == md for c in part["chapters"]):
            continue                        # the part that is only this page
        out.append(f"\n**{part['title']}**\n")
        for c in part["chapters"]:
            if c.get("file") and c["file"] != md:
                out.append(f"- [{heading(c)}]({link(md, c['file'])})")
            elif not c.get("file"):
                out.append(f"- {heading(c)} — *not written yet*")
    return "\n".join(out).strip("\n")


def turns(md: str, _: str) -> str:
    """The chapters before and after this one, in the order of book.json."""
    written = [c for c in chapters() if c.get("file") and re.fullmatch(r"\d+|[A-Z]", c["id"])]
    files = [c["file"] for c in written]
    if md not in files:
        return f"[Contents]({link(md, 'README.md')})"
    at = files.index(md)
    parts = []
    if at > 0:
        parts.append(f"← [{heading(written[at - 1])}]({link(md, files[at - 1])})")
    parts.append(f"[Contents]({link(md, 'README.md')})")
    if at + 1 < len(written):
        parts.append(f"[{heading(written[at + 1])}]({link(md, files[at + 1])}) →")
    return "---\n\n" + " · ".join(parts)


def costs(md: str, _: str) -> str:
    made = read_json("results/costs.json")
    fit = made["fitted"]
    lines = [
        f"- **Time:** {1000 * fit['secondsPerAgentIteration']:.2f} ms per agent per iteration, for "
        "every 10,000 weights in a brain.",
        f"- **Agents:** {fit['agentsPerToken']:.2f} alive per token of the world, once settled.",
        f"- **Disk:** {round(fit['bytesPerAgentIteration'])} bytes per agent per iteration.",
        f"- **Memory:** {round(fit['baseMB'])} MB, plus {fit['peakBytesPerWeightByte']:.1f} times "
        "every byte of every brain.",
        "",
        f"Fitted to {len(made.get('runs') or [])} recorded runs on {made.get('updated') or '—'} "
        "by `python3 gol_lab.py costs`."]
    return "\n".join(lines)


def table(md: str, name: str) -> str:
    """
    A table a chapter's code made from the records (`table chapter/name`): a head
    and rows, each cell text or links to chapters ({text, href} relative to book/).
    """
    chapter, key = name.split("/")
    made = read_json(f"results/{chapter}.json").get("numbers", {}).get(f"table:{key}")
    if made is None:
        raise KeyError(f"no table {name}: run `python3 book_figures.py {chapter}`")

    def cell(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(cell(v) for v in value) or "—"
        if isinstance(value, dict):
            return f"[{value['text']}]({link(md, value['href'])})"
        return str(value).replace("|", "\\|").replace("\n", " ") or "—"

    head = made.get("head") or [""] * len(made["rows"][0])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in made["rows"]]
    return "\n".join(lines)


MAKERS = {"figure": figure, "thesis": thesis, "runs": runs, "contents": contents, "turns": turns,
          "costs": costs, "table": table}
#: The blocks a page can ask for, and how they are found in it.
KINDS = tuple(MAKERS)
BLOCK = re.compile(r"^<!-- (%s)((?: [^\n]*?)?) -->\n(?:(?:(?!<!-- ).)*?^<!-- /\1 -->\n?)?"
                   % "|".join(KINDS), re.S | re.M)


# ---------------------------------------------------------------------------
# Filling
# ---------------------------------------------------------------------------

def fill_text(md: str, text: str) -> str:
    def made(match: "re.Match[str]") -> str:
        kind, arg = match.group(1), match.group(2).strip()
        body = MAKERS[kind](md, arg)
        return f"<!-- {kind}{' ' + arg if arg else ''} -->\n{body}\n<!-- /{kind} -->\n"
    return BLOCK.sub(made, text)


def markdown_files() -> List[str]:
    out = []
    for root, _, files in os.walk(BOOK):
        for name in files:
            if name.endswith(".md"):
                out.append(os.path.relpath(os.path.join(root, name), BOOK).replace(os.sep, "/"))
    return sorted(out)


def fill(only: Optional[List[str]] = None) -> int:
    """
    Fill every Markdown file of the book, or those named; the number changed.
    Filling them all also checks that every figure a chapter drew is shown
    somewhere: one that is not was drawn for nothing, or forgotten.
    """
    changed = 0
    placed = set()
    for md in only or markdown_files():
        path = os.path.join(BOOK, md)
        with open(path, encoding="utf-8") as f:
            before = f.read()
        after = fill_text(md, before)
        placed |= {arg.strip() for kind, arg in re.findall(r"^<!-- (\w+)((?: [^\n]*?)?) -->$", after, re.M)
                   if kind == "figure"}
        if after != before:
            with open(path, "w", encoding="utf-8") as f:
                f.write(after)
            changed += 1
    if only is None:
        unplaced = sorted(set(drawn()) - placed)
        if unplaced:
            raise ValueError(f"drawn but shown nowhere: {', '.join(unplaced)}")
    return changed


def drawn() -> List[str]:
    """Every figure the chapters' results say was drawn, as chapter/name."""
    out = []
    for name in sorted(os.listdir(os.path.join(BOOK, "results"))):
        figures = read_json(f"results/{name}").get("figures")
        if name.endswith(".json") and isinstance(figures, dict):
            out += [f"{name[:-5]}/{figure}" for figure in figures]
    return out


if __name__ == "__main__":
    print(f"{fill(sys.argv[1:] or None)} files filled in")
