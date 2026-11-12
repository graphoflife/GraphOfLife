#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The parts of the notebook's text that are made rather than written.

    python3 book_fill.py          fills them in, in every Markdown file of book/

The notebook is plain Markdown and SVG, so that it reads the same in the
Research Notebook tab, in Obsidian (open book/ as a vault) and on GitHub. A few of its blocks
are made from the runs, the plans and book.json rather than typed, so that
they cannot drift from what they describe. Such a block sits between two
comment lines, which no reader shows:

    <!-- figure tokens/lorenz -->     the figure book_figures.py drew, what it
    …                                 shows, and how to make it again
    <!-- /figure -->

    <!-- thesis E04 -->               an experiment's claim, quoted from its plan
    <!-- runs E02 -->                 what was run: what each condition changes, the
                                      seeds, the command that runs it again
    <!-- contents -->                 every page, under its topic, from book.json
    <!-- turns -->                    the page's topic, and the pages before and after it there
    <!-- used degree -->              the pages that link to a section, or to one within it
    <!-- costs -->                    what a simulation costs, as last fitted
    <!-- table chapter/name -->       a table a page's code made from the records
    <!-- glossary -->                 every term of terms.json, what it is, and its page

The first use of every term of book/terms.json on a page is linked to the page
that explains it (link_terms), with what it is as the link's title.

Everything between the opening and the closing line is replaced each time:
write only outside them. An opening line alone is filled in the first time.
book_figures.py fills the notebook after drawing its figures.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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


def slug(heading: str) -> str:
    """
    A heading's id as GitHub makes it, and the notebook's tab with it (markdown.js):
    the text a reader sees, in lower case, every character but a letter, a digit,
    `_`, `-` or a space dropped, each space a dash.
    """
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]*>|[*`$]|(?<!\w)_|_(?!\w)", "", text)
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-") or "section"


def anchors(md_text: str) -> List[str]:
    """The ids of a page's headings, in order; a second of the same name gets `-1`, as on GitHub."""
    text = re.sub(r"^```.*?^```", "", md_text, flags=re.S | re.M)
    seen: Dict[str, int] = {}
    out = []
    for heading in re.findall(r"^#{1,6}\s+(.*?)\s*$", text, re.M):
        base = slug(heading)
        out.append(f"{base}-{seen[base]}" if base in seen else base)
        seen[base] = seen.get(base, 0) + 1
    return out


# ---------------------------------------------------------------------------
# The blocks
# ---------------------------------------------------------------------------

#: What the blocks made so far on the page being filled know of one another, so
#: that a page does not say the same thing twice: the recipes of the figures above,
#: the screens whose worlds it has listed, and the sentences its boxes of runs and
#: its theses have said. fill_text starts it anew.
PAGE: Dict[str, Any] = {}


def recipe_of(md: str, title: str, recipe: str) -> str:
    """
    A figure's recipe, with its runs, its data and its steps said once on a page:
    what a figure above said already, word for word, is pointed to instead (the
    first figure that said it), wherever the pointer is the shorter. Its steps are
    pointed to as far as they begin as that figure's did.
    """
    paragraphs = recipe.split("\n\n")
    said = {key: next((p for p in paragraphs if p.startswith(f"**{key}.**")), None) for key in ("Runs", "Data")}
    listed = next((p for p in paragraphs if re.match(r"1\. ", p)), None)
    steps = listed.split("\n") if listed else []
    plain = [re.sub(r"^\d+\. ", "", s) for s in steps]
    above = PAGE.setdefault("figures", [])
    above.append((title.rstrip("."), said, plain))
    first = {key: next((t for t, s, _ in above[:-1] if said[key] and s[key] == said[key]), None) for key in said}
    # Runs and data together point to the first figure that read both, though it may point on for its runs.
    first["both"] = next((t for t, s, _ in above[:-1] if said["Runs"] and s == said), None)
    shared, source = 0, None                           # the longest beginning of the steps a figure above had
    for name, _, theirs in above[:-1]:
        k = next((i for i, (a, b) in enumerate(zip(plain, theirs)) if a != b), min(len(plain), len(theirs)))
        if k > shared:
            shared, source = k, name

    def as_for(name: Optional[str]) -> str:
        return f"As for the figure *{name}* above."
    out: Dict[str, Optional[str]] = {p: p for p in paragraphs}
    if steps and shared == len(steps) and first["both"] == source:
        out[said["Runs"]] = f"**Runs, data and steps.** {as_for(source)}"
        out[said["Data"]] = out[listed] = None
    else:
        both = f"**Runs and data.** {as_for(first['both'])}"
        if first["both"] and len(both) < len(said["Runs"]) + len(said["Data"]):
            out[said["Runs"]], out[said["Data"]] = both, None
        else:
            for key, paragraph in said.items():
                line = f"**{key}.** {as_for(first[key])}"
                if first[key] and len(line) < len(paragraph):
                    out[paragraph] = line
        line = f"{'Step 1' if shared == 1 else f'Steps 1–{shared}'} as for the figure *{source}* above."
        if shared and len(line) < len("\n".join(steps[:shared])):
            # The steps after them keep their numbers: the list goes on from those told above.
            out[listed] = "\n\n".join([line] + (["\n".join(steps[shared:])] if shared < len(steps) else []))
    return "\n\n".join(p for p in (out[p] for p in paragraphs) if p is not None)


def figure(md: str, name: str) -> str:
    """A figure, its caption and how to make it; their links are written as from a page in book/pages/."""
    chapter, fig = name.split("/")
    made = read_json(f"results/{chapter}.json").get("figures", {}).get(fig)
    if made is None:
        raise KeyError(f"no figure {name}: run `python3 book_figures.py {chapter}`")
    return (f"![{made['title']}]({link(md, f'figures/{chapter}/{fig}.svg')})\n\n"
            f"**{made['title']}{'' if made['title'][-1:] in '.?!' else '.'}** {made['caption']}\n\n"
            + quote(recipe_of(md, made["title"], made["recipe"]), "[!example]- How to make this figure"))


def thesis(md: str, name: str) -> str:
    """
    An experiment's thesis, word for word. A thesis written while the notebook had
    chapter numbers keeps them; each is a link to the page it meant that day, or
    to its section where that page was merged into another (book/legacy.json),
    and a line under the thesis says so.
    """
    plan = read_json(f"experiments/{name}.json")
    said = plan.get("thesis") or {}
    cited = read_json("legacy.json")["theses"].get(name, {})

    def where(words: str) -> str:
        target, _, section = cited[words].partition("#")
        if target.startswith("topic:"):
            title = next(t["title"] for t in topics() if t["id"] == target[6:])
            return f"{link(md, 'README.md')}#{slug(title)}"
        return link(md, page_file(target)) + (f"#{section}" if section else "")

    def numbered(text: str) -> str:
        def chapters(m: "re.Match[str]") -> str:
            return m.group(1) + re.sub(r"\d+", lambda n: f"[{n.group(0)}]({where('Chapter ' + n.group(0))})",
                                       m.group(2))
        text = re.sub(r"\b(Chapters? )(\d+(?:(?:, | and | or )\d+)*)\b", chapters, text)
        return re.sub(r"\b(?:Meta|Part) [IVXL]+\b", lambda m: f"[{m.group(0)}]({where(m.group(0))})", text)

    parts = [("claim", "The claim."), ("why", "Why it would be so."),
             ("confirm", "It holds if:"), ("refute", "It fails if:")]
    text = "\n\n".join(f"**{label}** {numbered(said[key]) if cited else said[key]}"
                       for key, label in parts if said.get(key))
    if cited and not PAGE.get("numbered"):            # said under the first such thesis of a page
        PAGE["numbered"] = True
        text += ("\n\n*Written while the notebook still numbered its chapters: each number is a link to the "
                 "page it meant that day.*")
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
    if isinstance(value, dict):                        # the pilots' rules: came_from, contest = lottery
        return ", ".join(k if v is True else f"{k} = {show(v)}" for k, v in sorted(value.items())) or "none"
    if isinstance(value, int):
        return number(value)
    return str(value)


def seeds_text(seeds: str) -> tuple:
    if ".." in str(seeds):
        a, b = (int(x) for x in str(seeds).split(".."))
        return f"{a} to {b}", b - a + 1
    listed = [s for s in str(seeds).replace(",", " ").split() if s]
    return ", ".join(listed), len(listed)


def once(what: str) -> bool:
    """Whether the page being filled has yet to say `what`: true the first time it is asked, and never after."""
    said = PAGE.setdefault("said", set())
    if what in said:
        return False
    said.add(what)
    return True


#: What every run of the lab writes beside its frames: said in a page's first box of runs.
WRITES = ("Each run writes its settings, seed and engine version beside its frames, in "
          "`GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.")


def made_again(what: str, name: str, writes: str = "") -> str:
    """
    How a box's runs are made again: the command, and, in the first box of a page
    that says so, the ▶ of the Research Notebook tab and what each run writes.
    """
    line = f"To {what} again: `python3 gol_lab.py run {name}`"
    if once("▶"):
        line += ", or ▶ in the Research Notebook tab of a computer running `gol_server.py`"
    return line + "." + (f" {writes}" if writes and once(writes) else "")


def baseline(md: str, base: Dict[str, Any]) -> str:
    """Whose settings a box's runs have but where a condition changes them: in full the first time a page says so."""
    if once(f"baseline {base['id']}"):
        return (f"Every setting is that of the baseline **{base['id']}** ({base['title'].lower()}) unless the "
                f"condition changes it ([At a glance]({link(md, page_file('settings'))}#at-a-glance)).")
    return f"Every setting is {base['id']}'s unless the condition changes it."


def trials(md: str, name: str, plan: Dict[str, Any]) -> str:
    """The trials behind a page: how many, of what kind, and what each set varies."""
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
    def sizes(values: List[Any]) -> str:
        """Patches of so many agents, then patches of a share of the world."""
        agents = [number(int(v)) for v in values if v >= 1]
        shares = [f"{v:.0%}" for v in values if v < 1]
        return "; ".join(([", ".join(agents) + " agents"] if agents else [])
                         + ([", ".join(shares) + " of the world"] if shares else []))

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
                                       else sizes(fields[key]) if key == "size"
                                       else span(fields[key]))
                         for key, word in words if key in fields)
        lines.append(f"- **{label}** — {number(count)} trials: {said}.")
    lines.append("")
    lines.append(made_again("play these trials", name, "Each writes its settings and engine beside its result, in "
                       "`GraphOfLifeRuns/<trial>/meta.json` and `result.json`."))
    return quote("\n".join(lines), "[!info]- The trials behind this page")


def same_draws(name: str, other: str) -> bool:
    """
    Whether two screens drew the same worlds, seed for seed and setting for
    setting, but for what each forces on all its worlds (`fixed`).
    """
    import gol_plan
    a, b = read_json(f"experiments/{name}.json")["screen"], read_json(f"experiments/{other}.json")["screen"]
    if any(a.get(k) != b.get(k) for k in ("sampler", "seed", "worlds", "tokens", "iterations", "skip",
                                          "budget_hours", "forced", "references")):
        return False
    fixed = set(a.get("fixed") or {}) | set(b.get("fixed") or {})

    def drawn(plan: str) -> List[Tuple[int, Dict[str, Any]]]:
        return [(d["seed"], {k: v for k, v in d["set"].items() if k not in fixed})
                for d in gol_plan.screen_draws(plan)]
    return drawn(name) == drawn(other)


def screen(md: str, name: str, plan: Dict[str, Any]) -> str:
    """
    The worlds of a screen: how their settings were drawn, and each one's, world
    by world (in the Archive, how to list them); or, where the page has listed the
    same draws for another screen already, what this one changed in all of them.
    """
    import gol_plan
    import gol_screen
    sc = plan["screen"]
    found = gol_plan.screen_draws(name)
    ids = {s.lab["seed"]: s.run_id for s in gol_plan.experiment_runs(name)}
    again = next((other for other in PAGE.setdefault("screens", []) if same_draws(name, other)), None)
    PAGE["screens"].append(name)
    if again:
        fixed = "; ".join(f"`{key}` = {show(value)}" for key, value in sorted((sc.get("fixed") or {}).items()))
        lines = [f"**{number(len(found))} worlds**: the worlds of Experiment {int(again[1:])} above, drawn "
                 f"alike (sampler {sc['sampler']}, round seed {sc['seed']}) and played alike, "
                 + (f"but with {fixed} in every world. " if fixed else "")
                 + "Runs, world by world: " + ", ".join(f"`{ids[d['seed']]}`" for d in found) + ".\n",
                 made_again("make these worlds", name)]
        return quote("\n".join(lines), "[!info]- The worlds behind this page")
    budget = (f", or until {sc['budget_hours']:g} hours of playing have gone by: a world that slow stops "
              "as too slow" if sc.get("budget_hours") else "")
    how, only = gol_screen.WORDS[int(sc["sampler"])]
    lines = [f"**{number(len(found))} worlds**, each with its settings drawn at random (sampler "
             f"{sc['sampler']}, round seed {sc['seed']}): B1's rules, changed as the draw says, at "
             f"{number(sc['tokens'])} tokens, played for {number(sc['iterations'])} iterations — or until "
             f"the world dies out{budget}.\n",
             f"- **How a setting is drawn:** {how} (`gol_screen.py`).",
             f"- **Only where it can matter:** {only}."]
    refs = gol_screen.references(sc)
    if refs:
        lines.append(f"- **Beside them, for reference:** " + ", ".join(
            f"{base} at seed {seed} (`{ids[seed]}`)" for base, seed in refs) + ".")
    forced = sc.get("forced") or {}
    if forced:
        lines.append("- **Forced off in this round:** " + "; ".join(
            f"`{key}` = {', '.join(show(v) for v in values)}" for key, values in sorted(forced.items())) + ".")
    fixed = sc.get("fixed") or {}
    if fixed:
        lines.append("- **The same in every world:** " + "; ".join(
            f"`{key}` = {show(value)}" for key, value in sorted(fixed.items())) + ".")
    skip = sc.get("skip") or []
    if skip:
        lines.append(f"- **Given up unplayed:** {number(len(skip))} of the {number(sc['worlds'])} worlds drawn, "
                     f"world{'s' if len(skip) > 1 else ''} " + ", ".join(str(i) for i in skip)
                     + "; every other world keeps its number, seed and name.")
    if archived(md):                                   # the Archive keeps the summary; the draws are a call away
        lines.append(f"- **Every world's settings:** `gol_plan.screen_draws(\"{name}\")` lists each world's "
                     "number, seed and what was drawn for it.")
    lines.append("")
    lines.append(made_again("make these worlds", name, WRITES))
    summary = quote("\n".join(lines), "[!info]- The worlds behind this page")
    if archived(md):
        return summary
    rows = ["| world | seed | run | what differs from B1 |", "|---|---|---|---|"]
    for d in found:
        what = ", ".join(f"[`{k}`]({link(md, page_file('settings'))}#{k}) = {show(v)}"
                         for k, v in sorted(d["set"].items()))
        rows.append(f"| {d['world']} | {d['seed']} | `{ids[d['seed']]}` | {what} |")
    every = quote("\n".join(rows), "[!info]- Every world's settings")
    return f"{summary}\n\n{every}"


def box_of(experiment: str) -> str:
    """Where an experiment's box of runs is, as a path under book/ and the section it sits in."""
    home = home_of(experiment)["file"]
    with open(os.path.join(BOOK, home), encoding="utf-8") as f:
        text = f.read()
    at = text.find(f"<!-- runs {experiment} -->")
    heads = len(re.findall(r"^#{1,6}\s", re.sub(r"^```.*?^```", "", text[:at], flags=re.S | re.M), re.M))
    return home + (f"#{anchors(text)[heads - 1]}" if at > 0 and heads else "")


#: Every screen's drawn worlds, by plan: drawn once, as every box of runs asks.
DRAWS: Dict[str, List[Dict[str, Any]]] = {}


def drawn_as(condition: Dict[str, Any], base: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    The world of a screen whose drawn settings a condition of B1 plays again, if
    any: the screen, the draw, and what the condition plays otherwise (its
    founders, say, for a world played at another size).
    """
    import gol_plan
    plays = {**(condition.get("set") or {}), **(condition.get("world") or {})}
    rules = {k: v for k, v in plays.items() if k not in gol_plan.WORLD}
    if base["id"] != "B1" or not rules:                # a screen's worlds are B1's rules, changed as drawn
        return None
    if not DRAWS:
        for f in sorted(os.listdir(os.path.join(BOOK, "experiments"))):
            if f.startswith("E") and f.endswith(".json") and "screen" in read_json(f"experiments/{f}"):
                DRAWS[f[:-5]] = gol_plan.screen_draws(f[:-5])
    for plan, found in DRAWS.items():
        for d in found:
            if {k: v for k, v in d["set"].items() if k not in gol_plan.WORLD} == rules:
                world = [k for k in gol_plan.WORLD if k != "total_tokens"]
                other = {k: plays.get(k, base["settings"][k]) for k in world
                         if plays.get(k, base["settings"][k]) != d["set"].get(k, base["settings"][k])}
                return plan, d, other
    return None


def withdrawn(plan: Dict[str, Any], what: str) -> str:
    """A plan given up before any of its runs was made: why, in one line."""
    note = plan["withdrawn"]
    return quote(f"None: {note[0].lower()}{note[1:]}", f"[!info] The {what} behind this page")


def listed_at(md: str, name: str, why: str) -> str:
    """A box of runs that would list again what another box lists: whose runs they are, a link there, the command."""
    return quote(f"These are the runs of [Experiment {int(name[1:])}]({link(md, box_of(name))}){why}. "
                 f"To make them again: `python3 gol_lab.py run {name}`.", "[!info]- The runs behind this page")


def runs(md: str, name: str) -> str:
    import gol_plan                                    # only here: it reads the runs folder
    plan, borrowed = resolve(name)
    if plan.get("withdrawn"):
        return withdrawn(plan, "trials" if "trials" in plan else "worlds" if "screen" in plan else "runs")
    if borrowed:                                       # a plan that reads another's runs: one line, to their box
        return listed_at(md, borrowed, "; nothing new was run")
    if archived(md) and home_of(name)["file"] != md:   # the Archive lists an experiment's runs on its own page
        return listed_at(md, name, ", listed on the page that reports it")
    if "trials" in plan:
        return trials(md, name, plan)
    if "screen" in plan:
        return screen(md, name, plan)
    spec = plan["runs"]
    base = read_json(f"experiments/{spec['baseline']}.json")
    conditions = spec["conditions"]
    world = spec["world"]["total_tokens"]

    def sizes(c: Dict[str, Any]) -> List[int]:
        return list(c.get("sizes") or (world if isinstance(world, list) else [world]))

    seeds, count = seeds_text(spec["seeds"])
    # A condition may have seeds of its own (gol_plan): a world found by one seed, played again.
    own = any("seeds" in c for c in conditions)

    def seeds_of(c: Dict[str, Any]) -> tuple:
        return seeds_text(c.get("seeds", spec["seeds"]))

    ids: Dict[str, List[str]] = {}
    for r in gol_plan.experiment_runs(name):
        ids.setdefault(r.condition, []).append(r.run_id)

    lines = []
    worlds = sum(len(sizes(c)) * seeds_of(c)[1] for c in conditions)
    lines.append(f"**{number(worlds)} runs.** Every condition below is run once for "
                 + ("each of its seeds" if own else f"every seed ({seeds})")
                 + (" and every number of tokens it lists" if any(len(sizes(c)) > 1 for c in conditions)
                                 else "") + f", for {number(spec['iterations'])} iterations — or "
                 "until its world dies out. " + baseline(md, base) + "\n")
    for c in conditions:
        # A condition may start its worlds with founders of its own (gol_plan): a world played again at a new size.
        drawn = drawn_as(c, base)
        if drawn:
            plan_id, d, other = drawn
            changed = (f"the settings of world {d['world']} of [Experiment {int(plan_id[1:])}]"
                       f"({link(md, box_of(plan_id))}) (seed {d['seed']}), as drawn"
                       + (" but with " + ", ".join(f"`{k}` = {show(v)}" for k, v in other.items()) if other else ""))
        else:
            changed = "changes " + (", ".join(f"`{k}` = {show(v)}" for k, v in
                                              {**(c.get("set") or {}), **(c.get("world") or {})}.items())
                                    or f"nothing: {base['id']} as it is")
        named = ids.get(c["name"], [])
        where = f" Runs `{named[0]}` … `{named[-1]}`." if named else ""
        lines.append(f"- **{c['name']}** — {changed}; {', '.join(number(s) for s in sizes(c))} "
                     f"tokens" + (f"; seeds {seeds_of(c)[0]}" if own else "") + f".{where}")
    lines.append("")
    lines.append(made_again("make these runs", name, WRITES))
    if any((c.get("set") or {}).get("proto") for c in conditions) and once("proto"):
        lines.append("")
        lines.append("Runs that ask for the pilots' rules are played by the research prototype, frozen beside "
                     "the engine (`gol_pilot.py`): they keep no frames, but one row of the world's statistics "
                     "every iteration (`series.jsonl`) and a whole world at every checkpoint.")
    return quote("\n".join(lines), "[!info]- The runs behind this page")


def pages() -> List[Dict[str, Any]]:
    """Every page of the notebook's index, written or not, in reading order."""
    return read_json("book.json")["pages"]


def topics() -> List[Dict[str, Any]]:
    """The topics, in the order the contents list them. A page's first topic is its home."""
    return read_json("book.json")["topics"]


def page_file(page: str) -> str:
    """The file of a written page, by its id, as a path under book/."""
    for c in pages():
        if c["id"] == page and c.get("file"):
            return c["file"]
    raise KeyError(f"no written page {page!r} in book.json")


def home_of(experiment: str) -> Dict[str, Any]:
    """The page that reports an experiment: the first, in reading order, that lists it."""
    for c in pages():
        if experiment in c.get("experiments", []) and c.get("file"):
            return c
    raise LookupError(f"no page of book.json reports {experiment}")


def at_home(topic: str) -> List[Dict[str, Any]]:
    """The pages whose home is a topic, in reading order."""
    return [c for c in pages() if c["topics"][0] == topic]


def parts() -> List[Dict[str, Any]]:
    """The parts of the notebook, each a colour and a few topics, in the order the contents show them."""
    return read_json("book.json").get("parts", [])


def contents(md: str, _: str) -> str:
    """
    Every page under its home topic, in reading order, and the topics under their parts (the author,
    2026-11-09: seven parts, each with its colour, which a coloured square stands for where Markdown has
    no colours): the front page's list.
    """
    out = []
    by_id = {t["id"]: t for t in topics()}
    for part in parts() or [{"topics": list(by_id)}]:
        if part.get("title"):
            out.append(f"\n### {part['title']}\n\n{part.get('mark', '')} {part['about']}".replace("\n ", "\n"))
        for tid in part["topics"]:
            topic = by_id[tid]
            listed = at_home(tid)
            if not listed:
                continue
            # Every topic keeps its heading, even alone in its part: pages link to it, and a page's
            # topic, above its title, opens the contents there.
            out.append(f"\n#### {topic['title']}\n")
            for c in listed:
                if c.get("file") and c["file"] != md:
                    out.append(f"- [{c['title']}]({link(md, c['file'])})")
                elif not c.get("file"):
                    out.append(f"- {c['title']} — *not written yet*")
    return "\n".join(out).strip("\n")


#: The pages that link to each page, by file, in reading order: what a page says,
#: not how it is found, so the links of the navigation, the contents, the boxes
#: of runs and the lists of where a section is used are left out. fill() works
#: it out once the other blocks are filled, and the navigation shows it.
BACKLINKS: Dict[str, List[str]] = {}
NAVIGATION = ("turns", "contents", "runs", "used")
#: For every section a page's words link to, as (file, anchor), the pages whose
#: words link to it and the section each link sits in, in reading order; the
#: glossary, which links every term, aside. fill() works it out with BACKLINKS.
USED: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}


def links_from(md: str, text: str) -> Set[str]:
    """The pages a page links to, as files under book/, its navigation aside."""
    text = BLOCK.sub(lambda m: "" if m.group(1) in NAVIGATION else m.group(0), text)
    out = set()
    for href in re.findall(r"\]\(([^)\s#:]+\.md)(?:#[^)\s]*)?\)", text):
        target = os.path.normpath(os.path.join(os.path.dirname(md), href)).replace(os.sep, "/")
        if target != md:
            out.add(target)
    return out


def backlinks(texts: Dict[str, str]) -> Dict[str, List[str]]:
    """For every page, the pages that link to it, in reading order."""
    order = {c["file"]: i for i, c in enumerate(pages()) if c.get("file")}
    into: Dict[str, List[str]] = {}
    for md, text in texts.items():
        for target in links_from(md, text):
            into.setdefault(target, []).append(md)
    return {target: sorted(sources, key=lambda m: (order.get(m, len(order)), m)) for target, sources in into.items()}


def headings(text: str) -> List[Tuple[int, int, str]]:
    """A page's headings: where each starts, its level and its id, as anchors() gives them."""
    blank = re.sub(r"^```.*?^```", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S | re.M)
    found = [(m.start(), len(m.group(1))) for m in re.finditer(r"^(#{1,6})\s+.*?\s*$", blank, re.M)]
    return [(at, level, anchor) for (at, level), anchor in zip(found, anchors(text))]


def used_by(texts: Dict[str, str]) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    """For every section that a page's words link to, the pages that link to it (see USED)."""
    order = {c["file"]: i for i, c in enumerate(pages()) if c.get("file")}
    into: Dict[Tuple[str, str], Dict[str, str]] = {}
    for md, text in sorted(texts.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0])):
        aside = [m.span() for m in BLOCK.finditer(text) if m.group(1) in NAVIGATION + ("glossary",)]
        heads = headings(text)
        for m in re.finditer(r"\]\(([^)\s#:]+\.md)#([^)\s]+)(?: \"[^\"]*\")?\)", text):
            if any(a <= m.start() < b for a, b in aside):
                continue
            target = os.path.normpath(os.path.join(os.path.dirname(md), m.group(1))).replace(os.sep, "/")
            if target != md:
                under = [anchor for at, _, anchor in heads if at < m.start()]
                into.setdefault((target, m.group(2)), {}).setdefault(md, under[-1] if under else "")
    return {key: list(sources.items()) for key, sources in into.items()}


def archived(md: str) -> bool:
    """Whether a page is in the Archive: frozen, its home a topic of the Archive's part."""
    shut = {t for part in parts() if part.get("id") == "archive" for t in part["topics"]}
    return any(c.get("file") == md and c["topics"][0] in shut for c in pages())


def used(md: str, anchor: str) -> str:
    """
    Where a section is used (`used anchor`): the pages whose words link to it, or
    to a section within it, each at the section where the link is; the Archive's
    folded apart.
    """
    with open(os.path.join(BOOK, md), encoding="utf-8") as f:
        heads = headings(f.read())
    at = next((i for i, (_, _, a) in enumerate(heads) if a == anchor), None)
    if at is None:
        raise KeyError(f"{md} has no section {anchor} to say where it is used")
    within = [anchor]
    for _, level, a in heads[at + 1:]:
        if level <= heads[at][1]:
            break
        within.append(a)
    order = {c["file"]: i for i, c in enumerate(pages()) if c.get("file")}
    sources: Dict[str, str] = {}
    for a in within:
        for page, section in USED.get((md, a), []):
            sources.setdefault(page, section)
    titles = {c["file"]: c["title"] for c in pages() if c.get("file")}

    def item(page: str) -> str:
        return f"[{titles.get(page, page)}]({link(md, page)}{'#' + sources[page] if sources[page] else ''})"
    ordered = sorted(sources, key=lambda m: (order.get(m, len(order)), m))
    now = [f"- {item(page)}" for page in ordered if not archived(page)]
    frozen = [item(page) for page in ordered if archived(page)]
    lines = now or [f"No page{' outside the Archive' if frozen else ''} links here yet."]
    if frozen:
        lines += ["", quote(" · ".join(frozen), f"[!info]- And in the Archive ({len(frozen)})")]
    return "\n".join(lines)


def turns(md: str, _: str) -> str:
    """
    The foot of a page: its home topic with the pages before and after it there,
    the other topics it is about, and the pages that link to it.
    """
    here = next((c for c in pages() if c.get("file") == md), None)
    if here is None or md == "README.md":
        return f"[Contents]({link(md, 'README.md')})"
    by_id = {t["id"]: t for t in topics()}
    topic = by_id[here["topics"][0]]
    written = [c for c in at_home(topic["id"]) if c.get("file")]
    at = written.index(here)
    steps = []
    if at > 0:
        steps.append(f"← [{written[at - 1]['title']}]({link(md, written[at - 1]['file'])})")
    steps.append(f"[{topic['title']}]({link(md, 'README.md')}#{slug(topic['title'])})")
    if at + 1 < len(written):
        steps.append(f"[{written[at + 1]['title']}]({link(md, written[at + 1]['file'])}) →")
    lines = ["---", "", " · ".join(steps)]
    also = [by_id[t] for t in here["topics"][1:]]
    if also:
        lines += ["", "**Also about:** " + ", ".join(
            f"[{t['title']}]({link(md, 'README.md')}#{slug(t['title'])})" for t in also) + "."]
    sources = BACKLINKS.get(md, [])
    if sources:
        titles = {c["file"]: c["title"] for c in pages() if c.get("file")}
        lines += ["", quote(" · ".join(f"[{titles.get(s, s)}]({link(md, s)})" for s in sources),
                            f"[!info]- Pages that link here ({len(sources)})")]
    return "\n".join(lines)


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
    A table a page's code made from the records (`table chapter/name`): a head
    and rows, each cell text or links ({text, href}): to pages, relative to
    book/, or to a world, `run:<run id>`, which the notebook tab opens in the
    viewer.
    """
    chapter, key = name.split("/")
    made = read_json(f"results/{chapter}.json").get("numbers", {}).get(f"table:{key}")
    if made is None:
        raise KeyError(f"no table {name}: run `python3 book_figures.py {chapter}`")

    def cell(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(cell(v) for v in value) or "—"
        if isinstance(value, dict):
            href = value["href"]
            return f"[{value['text']}]({href if href.startswith('run:') else link(md, href)})"
        return str(value).replace("|", "\\|").replace("\n", " ") or "—"

    head = made.get("head") or [""] * len(made["rows"][0])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in made["rows"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Terms, linked as an encyclopedia links its words
# ---------------------------------------------------------------------------

def terms() -> List[Dict[str, Any]]:
    """The notebook's terms (book/terms.json): names, the page that explains each, and a line that says it."""
    return read_json("terms.json")["terms"]


#: A link to a term, as link_terms writes it: titled with what the term is. No other link has a title.
TERM_LINK = re.compile(r'\[([^\[\]]+)\]\([^)\s]+ "[^"]*"\)')


def names_of(term: Dict[str, Any]) -> "re.Pattern[str]":
    """
    Every name of a term in running text: on word edges, its words across line
    breaks, with its plural; a name in lower case with either case of its first
    letter, one with capitals only as written.
    """
    alternatives = []
    for name in sorted([term["name"], *term.get("also", [])], key=len, reverse=True):
        words = [re.escape(word) for word in name.split()]
        if name[0].islower():
            words[0] = f"[{name[0].upper()}{name[0]}]{words[0][1:]}"
        alternatives.append(r"\s+".join(words))
    return re.compile(rf"(?<![\w\-/.#])(?:{'|'.join(alternatives)})(?:e?s)?(?![\w\-])")


def link_terms(md: str, text: str) -> str:
    """
    A page with the first use of every term linked to the page that explains it.
    Its own earlier term links are undone first, so linking twice changes
    nothing. Only words outside code, formulas, links, headings, callouts'
    titles, lists of references and the blocks made here are linked; the
    longest term first, so that "spectral dimension" is not taken for
    "dimension"; never on the term's own page, nor where the page links to it
    already, nor in the phrases a term lists as `not` ("game theory" is no
    game), nor for a term the page opts out of (`skip`).
    """
    text = TERM_LINK.sub(r"\1", text)
    index = read_json("terms.json")
    here = next((c["id"] for c in pages() if c.get("file") == md), None)
    linked = links_from(md, text)
    out = text
    kept = protected_spans(out) + [m.span() for m in re.finditer(r"^(?:#{1,6} |> \[!)[^\n]*", out, re.M)]
    # A list of references names papers, in their own words.
    kept += [m.span() for m in re.finditer(r"^## (?:References|Literature)\b.*?(?=^## |\Z)", out, re.M | re.S)]

    def free(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
        return not any(a < end and start < b for a, b in spans)

    ordered = sorted(index["terms"], key=lambda t: -max(len(n) for n in [t["name"], *t.get("also", [])]))
    for term in ordered:
        home = page_file(term["page"])
        if home == md or home in linked or term["name"] in index.get("skip", {}).get(here, []):
            continue
        spans = kept + [m.span() for phrase in term.get("not", [])
                        for m in re.finditer(re.escape(phrase), out, re.I)]
        found = next((m for m in names_of(term).finditer(out) if free(m.start(), m.end(), spans)), None)
        if found is None:
            continue
        target = link(md, home) + (f"#{term['anchor']}" if term.get("anchor") else "")
        made = f'[{found.group(0)}]({target} "{term["says"]}")'
        out = out[:found.start()] + made + out[found.end():]
        shift = len(made) - (found.end() - found.start())
        kept = [(a + shift, b + shift) if a >= found.end() else (a, b) for a, b in kept]
        kept.append((found.start(), found.start() + len(made)))
    return out


def glossary(md: str, _: str) -> str:
    """Every term of book/terms.json, A to Z: its names, what it is, and the page that explains it."""
    rows = []
    for term in sorted(terms(), key=lambda t: t["name"].lower()):
        target = link(md, page_file(term["page"])) + (f"#{term['anchor']}" if term.get("anchor") else "")
        also = f" (also {', '.join(term['also'])})" if term.get("also") else ""
        rows.append(f"- **[{term['name']}]({target})**{also} — {term['says']}")
    return "\n".join(rows)


MAKERS = {"figure": figure, "thesis": thesis, "runs": runs, "contents": contents, "turns": turns, "used": used,
          "costs": costs, "table": table, "glossary": glossary}
#: The blocks a page can ask for, and how they are found in it.
KINDS = tuple(MAKERS)
BLOCK = re.compile(r"^<!-- (%s)((?: [^\n]*?)?) -->\n(?:(?:(?!<!-- ).)*?^<!-- /\1 -->\n?)?"
                   % "|".join(KINDS), re.S | re.M)


# ---------------------------------------------------------------------------
# Filling
# ---------------------------------------------------------------------------

def protected_spans(text: str, links: bool = True) -> List[Tuple[int, int]]:
    """
    The stretches of a page that a rewriting of its words must leave as they are:
    the blocks made here, comments, code, formulas, and every link and picture whole
    (or, with links=False, only where a link points, so that its words can change).
    Sorted, and merged where they touch.
    """
    patterns = [r"<!--.*?-->", r"^```.*?^```", r"\$\$.*?\$\$", r"`[^`\n]*`", r"(?<![$\\])\$[^$\n]+\$",
                r'!?\[[^\]]*\]\([^)\s]*(?: "[^"]*")?\)' if links else r'\]\([^)\s]*(?: "[^"]*")?\)']
    spans = [m.span() for m in BLOCK.finditer(text)]
    for pattern in patterns:
        spans += [m.span() for m in re.finditer(pattern, text, re.S | re.M)]
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def outside(text: str, change: Callable[[str], str], links: bool = True) -> str:
    """The text with `change` applied to every stretch outside protected_spans."""
    out, at = [], 0
    for start, end in protected_spans(text, links):
        out += [change(text[at:start]), text[start:end]]
        at = end
    return "".join(out + [change(text[at:])])


def fill_text(md: str, text: str, kinds: Tuple[str, ...] = KINDS) -> str:
    """A page with its blocks of these kinds made anew; the others as they are."""
    PAGE.clear()

    def made(match: "re.Match[str]") -> str:
        kind, arg = match.group(1), match.group(2).strip()
        if kind not in kinds:
            return match.group(0)
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


def fill(only: Optional[List[str]] = None, write: bool = True) -> int:
    """
    Fill every Markdown file of the notebook, or those named; the number changed
    (or, with write=False, that would change). Two passes: every block but the
    navigation and the lists of where a section is used, then those two, which
    show the pages that link to a page or a section and so need every other
    block filled first. Filling them all also checks
    that every figure a page drew is shown somewhere: one that is not was drawn
    for nothing, or forgotten.
    """
    texts = {}
    for md in markdown_files():
        with open(os.path.join(BOOK, md), encoding="utf-8") as f:
            texts[md] = f.read()
    asked = only or list(texts)
    after = dict(texts)
    for md in asked:
        after[md] = link_terms(md, fill_text(md, texts[md], tuple(k for k in KINDS if k not in ("turns", "used"))))
    BACKLINKS.clear()
    BACKLINKS.update(backlinks(after))
    USED.clear()
    USED.update(used_by(after))
    changed = 0
    placed = set()
    for md in asked:
        after[md] = fill_text(md, after[md], ("turns", "used"))
        placed |= {arg.strip() for kind, arg in re.findall(r"^<!-- (\w+)((?: [^\n]*?)?) -->$", after[md], re.M)
                   if kind == "figure"}
        if after[md] != texts[md]:
            changed += 1
            if write:
                with open(os.path.join(BOOK, md), "w", encoding="utf-8") as f:
                    f.write(after[md])
    if only is None:
        unplaced = sorted(set(drawn()) - placed)
        if unplaced:
            raise ValueError(f"drawn but shown nowhere: {', '.join(unplaced)}")
    return changed


def drawn() -> List[str]:
    """Every figure the pages' results say was drawn, as chapter/name."""
    out = []
    for name in sorted(os.listdir(os.path.join(BOOK, "results"))):
        figures = read_json(f"results/{name}").get("figures")
        if name.endswith(".json") and isinstance(figures, dict):
            out += [f"{name[:-5]}/{figure}" for figure in figures]
    return out


if __name__ == "__main__":
    print(f"{fill(sys.argv[1:] or None)} files filled in")
