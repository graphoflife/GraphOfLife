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

KINDS = ("figure", "thesis", "runs", "contents", "turns", "costs")
BLOCK = re.compile(r"^<!-- (%s)((?: [^\n]*?)?) -->\n(?:(?:(?!<!-- ).)*?^<!-- /\1 -->\n?)?"
                   % "|".join(KINDS), re.S | re.M)


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
    """Every "Chapter 12" in a text made by code, as a link to that chapter."""
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
            f"**{made['title']}.** {linked(md, made['caption'])}\n\n"
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


def runs(md: str, name: str) -> str:
    import gol_lab                                     # only here: it reads the runs folder
    plan, borrowed = resolve(name)
    spec = plan["runs"]
    base = read_json(f"experiments/{spec['baseline']}.json")
    glossary = read_json("settings.json")
    conditions = spec["conditions"]
    world = spec["world"]["total_tokens"]

    def sizes(c: Dict[str, Any]) -> List[int]:
        return list(c.get("sizes") or (world if isinstance(world, list) else [world]))

    seeds, count = seeds_text(spec["seeds"])
    ids: Dict[str, List[str]] = {}
    for r in gol_lab.experiment_runs(borrowed or name):
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


MAKERS = {"figure": figure, "thesis": thesis, "runs": runs, "contents": contents, "turns": turns,
          "costs": costs}


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
    """Fill every Markdown file of the book, or those named; the number changed."""
    changed = 0
    for md in only or markdown_files():
        path = os.path.join(BOOK, md)
        with open(path, encoding="utf-8") as f:
            before = f.read()
        after = fill_text(md, before)
        if after != before:
            with open(path, "w", encoding="utf-8") as f:
                f.write(after)
            changed += 1
    return changed


if __name__ == "__main__":
    print(f"{fill(sys.argv[1:] or None)} files filled in")
