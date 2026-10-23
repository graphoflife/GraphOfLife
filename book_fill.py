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
    <!-- runs E02 -->                 what was run: every setting, the seeds, the
                                      command that runs it again
    <!-- contents -->                 every page, under its topic, from book.json
    <!-- turns -->                    the page's topic, and the pages before and after it there
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

def figure(md: str, name: str) -> str:
    """A figure, its caption and how to make it; their links are written as from a page in book/pages/."""
    chapter, fig = name.split("/")
    made = read_json(f"results/{chapter}.json").get("figures", {}).get(fig)
    if made is None:
        raise KeyError(f"no figure {name}: run `python3 book_figures.py {chapter}`")
    return (f"![{made['title']}]({link(md, f'figures/{chapter}/{fig}.svg')})\n\n"
            f"**{made['title']}{'' if made['title'][-1:] in '.?!' else '.'}** {made['caption']}\n\n"
            + quote(made["recipe"], "[!example]- How to make this figure"))


def thesis(md: str, name: str) -> str:
    """
    An experiment's thesis, word for word. A thesis written while the notebook had
    chapter numbers keeps them; each is a link to the page it meant that day
    (book/legacy.json), and a line under the thesis says so.
    """
    plan = read_json(f"experiments/{name}.json")
    said = plan.get("thesis") or {}
    cited = read_json("legacy.json")["theses"].get(name, {})

    def where(words: str) -> str:
        target = cited[words]
        if target.startswith("topic:"):
            title = next(t["title"] for t in topics() if t["id"] == target[6:])
            return f"{link(md, 'README.md')}#{slug(title)}"
        return link(md, page_file(target))

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
    if cited:
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
    lines.append(f"To play these trials again: `python3 gol_lab.py run {name}`, or ▶ in the Research Notebook tab "
                 "of a computer running `gol_server.py`. Each writes its settings and engine beside "
                 "its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.")
    return quote("\n".join(lines), "[!info] The trials behind this page")


def screen(md: str, name: str, plan: Dict[str, Any]) -> str:
    """The worlds of a screen: how their settings were drawn, and each one's, world by world."""
    import gol_plan
    import gol_screen
    sc = plan["screen"]
    found = gol_plan.screen_draws(name)
    ids = {s.lab["seed"]: s.run_id for s in gol_plan.experiment_runs(name)}
    budget = (f", or until {sc['budget_hours']:g} hours of playing have gone by: a world that slow stops "
              "as too slow" if sc.get("budget_hours") else "")
    lines = [f"**{number(len(found))} worlds**, each with its settings drawn at random (sampler "
             f"{sc['sampler']}, round seed {sc['seed']}): B1's rules, changed as the draw says, at "
             f"{number(sc['tokens'])} tokens, played for {number(sc['iterations'])} iterations — or until "
             f"the world dies out{budget}.\n",
             "- **How a setting is drawn:** an on/off switch is on in half the worlds, and a choice among "
             "several options takes each equally often; a number keeps its default in three worlds of "
             "four, and otherwise takes one of a few levels; the size of a message and the number of noise "
             "inputs are 0 in one world of ten (`gol_screen.py`).",
             "- **Only where it can matter:** buying life only with old age, the mate's kind only with "
             "mixing, a search's steering and consent only with a search, messages before the game and "
             "the pooled self-message only with messages."]
    refs = gol_screen.references(sc)
    if refs:
        lines.append(f"- **Beside them, for reference:** " + ", ".join(
            f"{base} at seed {seed} (`{ids[seed]}`)" for base, seed in refs) + ".")
    forced = sc.get("forced") or {}
    if forced:
        lines.append("- **Forced off in this round:** " + "; ".join(
            f"`{key}` = {', '.join(show(v) for v in values)}" for key, values in sorted(forced.items())) + ".")
    lines.append("")
    lines.append(f"To make these worlds again: `python3 gol_lab.py run {name}`, or ▶ in the Research Notebook tab of a "
                 "computer running `gol_server.py`. Each run writes its settings, seed and engine version "
                 "beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.")
    summary = quote("\n".join(lines), "[!info] The worlds behind this page")
    rows = ["| world | seed | run | what differs from B1 |", "|---|---|---|---|"]
    for d in found:
        what = ", ".join(f"[`{k}`]({link(md, page_file('settings'))}#{k}) = {show(v)}"
                         for k, v in sorted(d["set"].items()))
        rows.append(f"| {d['world']} | {d['seed']} | `{ids[d['seed']]}` | {what} |")
    every = quote("\n".join(rows), "[!info]- Every world's settings")
    return f"{summary}\n\n{every}"


def runs(md: str, name: str) -> str:
    import gol_plan                                    # only here: it reads the runs folder
    plan, borrowed = resolve(name)
    if "trials" in plan:
        return trials(md, borrowed or name, plan)
    if "screen" in plan:
        return screen(md, borrowed or name, plan)
    spec = plan["runs"]
    base = read_json(f"experiments/{spec['baseline']}.json")
    glossary = read_json("settings.json")
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
    for r in gol_plan.experiment_runs(borrowed or name):
        ids.setdefault(r.condition, []).append(r.run_id)

    lines = []
    if borrowed:
        lines.append(f"These are the runs of Experiment {int(borrowed[1:])}; nothing new was run for "
                     "this page.\n")
    worlds = sum(len(sizes(c)) * seeds_of(c)[1] for c in conditions)
    lines.append(f"**{number(worlds)} runs.** Every condition below is run once for "
                 + ("each of its seeds" if own else f"every seed ({seeds})")
                 + (" and every number of tokens it lists" if any(len(sizes(c)) > 1 for c in conditions)
                                 else "") + f", for {number(spec['iterations'])} iterations — or "
                 "until its world dies out. Every setting is that of the baseline "
                 f"**{base['id']}** ({base['title'].lower()}) unless the condition changes it.\n")
    for c in conditions:
        changed = ", ".join(f"`{k}` = {show(v)}" for k, v in (c.get("set") or {}).items()) \
            or f"nothing: {base['id']} as it is"
        named = ids.get(c["name"], [])
        where = f" Runs `{named[0]}` … `{named[-1]}`." if named else ""
        lines.append(f"- **{c['name']}** — changes {changed}; {', '.join(number(s) for s in sizes(c))} "
                     f"tokens" + (f"; seeds {seeds_of(c)[0]}" if own else "") + f".{where}")
    lines.append("")
    lines.append(f"To make these runs again: `python3 gol_lab.py run {borrowed or name}`, or ▶ in "
                 "the Research Notebook tab of a computer running `gol_server.py`. Each run writes its "
                 "settings, seed and engine version beside its frames, in "
                 "`GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.")
    if any((c.get("set") or {}).get("proto") for c in conditions):
        lines.append("")
        lines.append("Runs that ask for the pilots' rules are played by the research prototype, frozen beside "
                     "the engine (`gol_pilot.py`): they keep no frames, but one row of the world's statistics "
                     "every iteration (`series.jsonl`) and a whole world at every checkpoint.")
    summary = quote("\n".join(lines), "[!info] The runs behind this page")

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
        rows.append(f"| [`{key}`]({link(md, page_file('settings'))}#{key}) | " + " | ".join(cells)
                    + f" | {meaning} |")
    rows.append("| seeds | " + " | ".join(seeds_of(c)[0] for c in conditions) + " | one run per seed |")
    rows.append("| iterations | " + " | ".join(number(spec["iterations"]) for _ in conditions)
                + " | how far each run goes, unless its world dies first |")
    table = "\n".join([head, rule, *rows])
    every = quote(table + "\n\nA value in **bold** differs from the baseline "
                  f"{base['id']}.", "[!info]- Every setting of these runs")
    return f"{summary}\n\n{every}"


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


def contents(md: str, _: str) -> str:
    """Every page under its home topic, in reading order: the front page's list."""
    out = []
    for topic in topics():
        out.append(f"\n### {topic['title']}\n\n{topic['about']}\n")
        for c in at_home(topic["id"]):
            if c.get("file") and c["file"] != md:
                out.append(f"- [{c['title']}]({link(md, c['file'])})")
            elif not c.get("file"):
                out.append(f"- {c['title']} — *not written yet*")
    return "\n".join(out).strip("\n")


#: The pages that link to each page, by file, in reading order: what a page says,
#: not how it is found, so the links of the navigation, the contents and the
#: tables of settings are left out. fill() works it out once the other blocks
#: are filled, and the navigation shows it.
BACKLINKS: Dict[str, List[str]] = {}
NAVIGATION = ("turns", "contents", "runs")


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
    and rows, each cell text or links to pages ({text, href} relative to book/).
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
    nothing. Only words outside code, formulas, links, headings, callouts' titles
    and the blocks made here are linked; the longest term first, so that "spectral
    dimension" is not taken for "dimension"; never on the term's own page, nor
    where the page links to it already, nor in the phrases a term lists as `not`
    ("game theory" is no game), nor for a term the page opts out of (`skip`).
    """
    text = TERM_LINK.sub(r"\1", text)
    index = read_json("terms.json")
    here = next((c["id"] for c in pages() if c.get("file") == md), None)
    linked = links_from(md, text)
    out = text
    kept = protected_spans(out) + [m.span() for m in re.finditer(r"^(?:#{1,6} |> \[!)[^\n]*", out, re.M)]

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


MAKERS = {"figure": figure, "thesis": thesis, "runs": runs, "contents": contents, "turns": turns,
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
    navigation, then the navigation, which shows the pages that link to a page
    and so needs every other block filled first. Filling them all also checks
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
        after[md] = link_terms(md, fill_text(md, texts[md], tuple(k for k in KINDS if k != "turns")))
    BACKLINKS.clear()
    BACKLINKS.update(backlinks(after))
    changed = 0
    placed = set()
    for md in asked:
        after[md] = fill_text(md, after[md], ("turns",))
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
