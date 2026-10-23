#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Research Notebook without chapter numbers (the author's request of
2026-10-23): every chapter and note in one folder, book/pages/, under a name
that says what it is about, and the index sorted by topic.

    python3 research/tools/unnumber.py          what it would do: every page's new name and topics
    python3 research/tools/unnumber.py --move   do it, once (done on 2026-10-23)
    python3 research/tools/unnumber.py --words  then the numbers out of the words
    python3 research/tools/unnumber.py --code FILE ...   and out of the strings of the code that writes them

What --move does:
- `git mv` every chapter and note to book/pages/<name>.md;
- rewrites every relative link of every page by path: a link is resolved from
  where its page was, mapped to where its target went, and written from where
  the page is now. (A plain replacement of file names would turn the chapter
  `28-gains-and-losses.md` into the note `gains-and-losses.md`.);
- the same for the links in the results the figures were drawn with
  (book/results/*.json), written for a page in book/chapters/, and for the
  pages a table links to (written from book/);
- writes book.json anew: the topics, and every page in reading order with its
  topics (the first is its home), its kind, its experiments, its status, the day
  it was added and a line that says what it is;
- writes book/legacy.json: every old chapter number and note id with the page
  it became, and, thesis by thesis, the page each chapter number it cites meant
  on the day it was written. Theses are quoted word for word, so they keep the
  numbers of their day.

The pages planned as Meta III, IV and V are dropped: each topic's overview
does what they were to do.

What --words does, after the move: a link that reads "Chapter 13" or "13" is
named by the title of the page it leads to; a plain "Chapter 13", "Meta X" or
"Appendix C" becomes such a link (a page not written yet, its title in italics);
"the book" becomes "the notebook" and "chapter" "page", never in a path, code
or a formula, and never in a thesis. A heading whose words change takes its
links with it. What it cannot word — the parts, ranges of chapters — it lists,
and a person words them. --code does the same in the strings and comments of
the figures' code, so that a figure drawn again says what its page says.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BOOK = os.path.join(ROOT, "book")
sys.path.insert(0, ROOT)

#: The topics, in the order the contents list them: id, title, and what a reader finds there.
TOPICS = [
    ("start", "Start here",
     "What this notebook is about, how to read it, and the question behind it: can evolution in a "
     "simulated world go on without end?"),
    ("simulation", "How the simulation works",
     "The rules of a world, step by step: tokens, agents and links, the game, births and deaths, the "
     "brains, and every setting."),
    ("baselines", "The baselines B1 and B2",
     "The two standard worlds every experiment starts from: B1, the original rules, and B2, a world "
     "that cannot break, and how it was built."),
    ("lives", "Agents' lives: births, deaths, ages",
     "Who is born, who dies and when, how old agents get, and how a world's ages change."),
    ("wealth", "Wealth: who holds the tokens",
     "How unequal a world is, who stays rich, and how gains and losses are shared out."),
    ("game", "The game and the flow of tokens",
     "How the game is played, where tokens flow, which loops they go round, and the groups they stay in."),
    ("network", "The shape and health of the network",
     "Degrees, cores, bridges, power laws and cuts: what shape the links take, and how a world breaks."),
    ("space", "Space and dimension",
     "Whether a world is a space, of how many dimensions, what folds it, and the physics the project hopes for."),
    ("size", "Size and riches of a world",
     "What sets how many agents a world holds, and how rich each is."),
    ("brains", "Brains, and how alike agents are",
     "What the brains are and compute, the families they make, and how different agents really are."),
    ("evolution", "Does evolution keep going?",
     "How to tell whether evolution goes on, why it stops, and what would count as open-ended evolution."),
    ("strategies", "Strategies written by hand",
     "Raiders, turtles and crews written by hand and planted in evolved worlds: is evolution's play the best?"),
    ("groups", "Groups and cooperation",
     "Help, alliances, other contests and patterns that outlast their agents: do agents work together?"),
    ("rules", "Changing the rules",
     "The switches of the engine, alone and in company: what each rule does to a world."),
    ("healthy", "Looking for a healthy, rich world",
     "The search for a world that is decentralised, resilient, about three-dimensional and full of "
     "different agents, and that keeps evolving."),
    ("methods", "Methods and statistics",
     "How runs are recorded and measured, the statistics the notebook uses, and every simulation made."),
]

#: Every page, by its old id, in reading order: (old id, topics with its home first, kind, new id or None).
#: A new id of None is the old file's name without its number.
PAGES: List[Tuple[str, List[str], str, Optional[str]]] = [
    # Start here
    ("about", ["start"], "overview", "README"),
    ("1", ["start"], "overview", "what-this-notebook-is-about"),
    ("6", ["start", "evolution"], "explanation", None),
    # How the simulation works
    ("2", ["simulation"], "explanation", None),
    ("3", ["simulation"], "explanation", None),
    ("4", ["simulation"], "explanation", None),
    ("39", ["simulation", "baselines"], "study", None),
    ("8", ["simulation", "methods"], "experiment", None),
    ("n-settings", ["simulation"], "term", None),
    ("n-starting-ring", ["simulation"], "term", None),
    ("n-random-numbers", ["simulation"], "term", None),
    ("n-share-function", ["simulation"], "term", None),
    ("n-binary-decision", ["simulation"], "term", None),
    ("n-largest-remainder", ["simulation"], "term", None),
    ("n-brain-inputs", ["simulation", "brains"], "term", None),
    ("n-brain-outputs", ["simulation", "brains"], "term", None),
    ("n-messages", ["simulation"], "term", None),
    ("n-revolution", ["simulation"], "term", None),
    ("n-mutation", ["simulation", "brains"], "term", None),
    ("n-extinction", ["simulation"], "term", None),
    # The baselines
    ("26", ["baselines"], "overview", "what-the-baseline-world-is"),
    ("9", ["baselines"], "study", None),
    ("10", ["baselines", "lives"], "experiment", None),
    ("11", ["baselines"], "study", None),
    ("14", ["baselines", "methods"], "experiment", None),
    ("47", ["baselines", "rules"], "experiment", None),
    ("48", ["baselines", "network"], "experiment", None),
    ("49", ["baselines", "lives"], "experiment", None),
    # Agents' lives
    ("12", ["lives"], "study", None),
    ("13", ["lives"], "study", None),
    ("31", ["lives", "network"], "study", None),
    ("n-births-and-deaths", ["lives", "simulation"], "term", None),
    ("n-age-and-lifetime", ["lives"], "term", None),
    ("n-reproduction-statistics", ["lives"], "term", None),
    # Wealth
    ("15", ["wealth"], "experiment", None),
    ("27", ["wealth"], "study", None),
    ("21", ["wealth"], "study", None),
    ("28", ["wealth", "game"], "study", None),
    ("33", ["wealth", "network"], "study", None),
    ("44", ["wealth"], "experiment", None),
    ("n-gini-coefficient", ["wealth"], "term", None),
    ("n-richest-tenth", ["wealth"], "term", None),
    ("n-tokens-per-agent", ["wealth"], "term", None),
    ("n-entropy-and-evenness", ["wealth"], "term", None),
    ("n-gains-and-losses", ["wealth", "game"], "term", "gains-losses-and-the-share-out"),
    # The game and the flow of tokens
    ("75", ["game"], "overview", "where-tokens-go"),
    ("16", ["game"], "study", None),
    ("29", ["game"], "study", None),
    ("30", ["game"], "study", None),
    ("71", ["game", "groups"], "study", None),
    ("72", ["game"], "study", None),
    ("n-home-stake", ["game", "wealth"], "term", None),
    ("n-token-flow", ["game"], "term", None),
    ("n-lightning", ["game"], "term", None),
    ("n-token-curvature", ["game"], "term", None),
    ("n-circulation", ["game"], "term", None),
    ("n-map-equation", ["game", "groups"], "term", None),
    # The shape and health of the network
    ("17", ["network"], "experiment", None),
    ("22", ["network", "lives"], "study", None),
    ("25", ["network"], "study", None),
    ("32", ["network", "methods"], "study", None),
    ("34", ["network"], "study", None),
    ("35", ["network", "space"], "study", None),
    ("36", ["network"], "study", None),
    ("n-degree", ["network"], "term", None),
    ("n-preferential-attachment", ["network", "lives"], "term", None),
    ("n-clustering", ["network"], "term", None),
    ("n-path-length", ["network", "space"], "term", None),
    ("n-core-trees-leaves", ["network"], "term", None),
    ("n-bridges", ["network"], "term", None),
    ("n-loops", ["network"], "term", None),
    ("n-cut-risk", ["network"], "term", None),
    ("n-radius-and-diameter", ["network", "space"], "term", None),
    ("n-spectral-gap", ["network"], "term", None),
    ("n-assortativity", ["network"], "term", None),
    ("n-scaling-relations", ["network"], "term", None),
    # Space and dimension
    ("7", ["space", "start"], "explanation", None),
    ("23", ["space", "network"], "study", None),
    ("24", ["space"], "study", None),
    ("60", ["space", "size"], "experiment", None),
    ("87", ["space", "healthy"], "study", None),
    ("61", ["space"], "study", None),
    ("62", ["space"], "study", None),
    ("63", ["space"], "experiment", None),
    ("64", ["space"], "experiment", None),
    ("65", ["space"], "experiment", None),
    ("66", ["space"], "experiment", None),
    ("n-correlation-function", ["space", "network"], "term", None),
    ("n-ball-dimension-and-curvature", ["space"], "term", None),
    ("n-spectral-dimension", ["space"], "term", None),
    ("n-box-dimension", ["space"], "term", None),
    # Size and riches
    ("84", ["size", "healthy"], "overview", "what-holds-a-worlds-size"),
    ("42", ["size"], "experiment", None),
    ("101", ["size", "healthy"], "experiment", None),
    ("105", ["size"], "experiment", None),
    # Brains
    ("77", ["brains"], "overview", "how-alike-the-agents-are"),
    ("18", ["brains", "evolution"], "experiment", None),
    ("19", ["brains"], "study", None),
    ("20", ["brains"], "study", None),
    ("41", ["brains"], "experiment", None),
    ("43", ["brains"], "experiment", None),
    ("70", ["brains"], "study", None),
    ("53", ["brains", "evolution"], "experiment", None),
    ("n-genotype", ["brains", "simulation"], "term", None),
    ("n-vanishing-signal", ["brains", "simulation"], "term", None),
    ("n-brain-diversity", ["brains"], "term", None),
    ("n-families", ["brains"], "term", None),
    ("n-common-ancestor", ["brains", "evolution"], "term", None),
    # Does evolution keep going?
    ("40", ["evolution", "baselines"], "overview", "what-the-measurements-say-about-open-ended-evolution"),
    ("46", ["evolution", "healthy"], "experiment", None),
    ("50", ["evolution"], "experiment", None),
    ("52", ["evolution", "brains"], "experiment", None),
    ("95", ["evolution", "strategies"], "experiment", None),
    ("69", ["evolution"], "experiment", None),
    ("n-open-ended-test", ["evolution", "start"], "term", None),
    ("n-exploitability", ["evolution", "strategies"], "term", None),
    # Strategies written by hand
    ("51", ["strategies"], "study", None),
    ("94", ["strategies", "brains"], "study", None),
    ("96", ["strategies"], "experiment", None),
    ("97", ["strategies", "evolution"], "experiment", None),
    ("102", ["strategies", "groups"], "experiment", None),
    ("103", ["strategies"], "experiment", None),
    ("104", ["strategies", "baselines"], "experiment", None),
    ("67", ["strategies", "space"], "experiment", None),
    ("n-invasion-test", ["strategies", "methods"], "term", None),
    # Groups and cooperation
    ("76", ["groups"], "overview", "who-helps-whom"),
    ("38", ["groups"], "study", None),
    ("55", ["groups", "rules"], "study", None),
    ("56", ["groups", "rules"], "study", None),
    ("58", ["groups", "game"], "study", None),
    ("n-help", ["groups"], "term", None),
    # Changing the rules
    ("91", ["rules"], "overview", "rules-in-company"),
    ("54", ["rules"], "study", None),
    ("57", ["rules", "baselines"], "study", None),
    ("73", ["rules", "healthy"], "study", None),
    ("78", ["rules", "brains"], "study", None),
    ("83", ["rules", "network"], "study", None),
    ("82", ["rules"], "study", None),
    ("86", ["rules"], "study", None),
    ("88", ["rules"], "study", None),
    ("89", ["rules"], "study", None),
    ("90", ["rules"], "study", None),
    # Looking for a healthy, rich world
    ("79", ["healthy"], "overview", "healthy-and-rich"),
    ("n-what-makes-a-world-interesting", ["healthy"], "term", None),
    ("80", ["healthy"], "experiment", None),
    ("81", ["healthy", "evolution"], "experiment", None),
    ("85", ["healthy"], "experiment", None),
    ("99", ["healthy"], "experiment", None),
    ("92", ["healthy"], "experiment", None),
    ("93", ["healthy"], "experiment", None),
    ("98", ["healthy"], "experiment", None),
    ("100", ["healthy", "space"], "experiment", None),
    ("106", ["healthy"], "experiment", None),
    # Methods and statistics
    ("74", ["methods"], "overview", "every-world-side-by-side"),
    ("5", ["methods"], "method", None),
    ("37", ["methods"], "method", None),
    ("A", ["methods"], "method", "what-a-simulation-costs"),
    ("B", ["methods"], "study", None),
    ("C", ["methods"], "method", None),
    ("n-frames-and-stats", ["methods", "simulation"], "method", None),
    ("n-viewer-colours", ["methods"], "method", None),
    ("n-pilot-worlds", ["methods"], "method", None),
    ("n-median-and-quantiles", ["methods"], "method", None),
    ("n-bands", ["methods"], "method", None),
    ("n-settled-life", ["methods"], "method", None),
    ("n-spread-between-worlds", ["methods"], "method", None),
    ("n-correlation", ["methods"], "method", None),
    ("n-autocorrelation", ["methods"], "method", None),
    ("n-mobility", ["methods", "wealth"], "method", None),
    ("n-between-and-within", ["methods"], "method", None),
    ("n-participation-ratio", ["methods"], "method", None),
    ("n-kaplan-meier", ["methods", "lives"], "method", None),
    ("n-bootstrap", ["methods"], "method", None),
    ("n-permutation-test", ["methods"], "method", None),
    ("n-seeds-needed", ["methods"], "method", None),
    ("n-wilson-interval", ["methods"], "method", None),
    ("n-logarithmic-axes", ["methods"], "method", None),
    ("n-muller-plot", ["methods", "brains"], "method", None),
    ("n-least-squares", ["methods"], "method", None),
    ("n-power-law-fit", ["methods"], "method", None),
    ("n-power-spectrum", ["methods"], "method", None),
    ("n-variance-split", ["methods", "brains"], "method", None),
    ("n-effective-number", ["methods", "brains"], "method", None),
    ("n-fractional-factorial", ["methods", "rules"], "method", None),
]
#: Planned once as the summaries of parts that are now topics.
DROPPED = {"45", "59", "68"}
#: Titles that were numbers: the summaries of parts, now named by what they say.
TITLES = {"26": "What the baseline world is",
          "40": "What the measurements say about open-ended evolution",
          "74": "Every world side by side", "75": "Where tokens go", "76": "Who helps whom",
          "77": "How alike the agents are", "79": "Healthy and rich", "84": "What holds a world's size",
          "91": "Rules in company"}

#: For every thesis that cites a chapter by number, the page each number meant on the day it was
#: written (found from book.json as it stood in the commit that gave the thesis its words; checked by
#: hand, and against the notes the chapters added after four of them).
THESES: Dict[str, Dict[str, str]] = {
    "E06": {"Part II": "topic:evolution"},          # 2026-10-02: "Part II · Towards open-ended evolution"
    "E07": {"Chapter 4": "14"},                     # 2026-10-03: How much does the seed decide?
    "E08": {"Chapter 3": "10"},                     # 2026-10-05: The life of a world
    "E09": {"Chapter 53": "50", "Chapter 43": "52", "Chapter 52": "49"},   # 2026-10-14
    "E10": {"Chapter 6": "6", "Chapter 23": "24", "Chapter 38": "42", "Chapter 3": "3"},   # 2026-10-09
    "E13": {"Chapter 19": "19"},
    "E14": {"Chapter 49": "46"},
    "E16": {"Chapter 49": "46"},
    "E17": {"Meta X": "79", "Chapter 46": "46"},
    "E18": {"Meta X": "79", "Chapter 46": "46"},
    "E19": {"Meta XI": "84"},
    "E20": {"Meta X": "79"},
    "E21": {"Meta X": "79", "Meta XI": "84"},
    "E22": {"Meta XI": "84"},
    "E24": {"Chapter 93": "93", "Meta X": "79"},
    "E28": {"Chapter 81": "81", "Chapter 92": "92"},
    "E29": {"Chapter 51": "51", "Chapter 94": "94"},
    "E30": {"Chapter 95": "95"},
    "E31": {"Chapter 87": "87", "Meta XII": "91"},
    "E34": {"Chapter 51": "51"},
    "E35": {"Chapter 97": "97", "Chapter 102": "102", "Chapter 103": "103"},
    "E36": {"Chapter 101": "101"},
    "E37": {"Chapter 101": "101"},
    "E38": {"Chapter 100": "100", "Chapter 101": "101"},
}

LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def old_index() -> Dict[str, Dict[str, Any]]:
    with open(os.path.join(BOOK, "book.json"), encoding="utf-8") as f:
        book = json.load(f)
    return {c["id"]: c for part in book["parts"] for c in part["chapters"]}


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower().replace("'", "")).strip("-")


def new_ids(old: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Every old id and the page id it becomes."""
    out = {}
    for oid, _, _, nid in PAGES:
        c = old[oid]
        if nid is None:
            nid = (re.sub(r"^(\d+|[A-Z])-", "", os.path.basename(c["file"])[:-3]) if c.get("file")
                   else slugify(c["title"]))
        out[oid] = nid
    return out


def check(old: Dict[str, Dict[str, Any]], ids: Dict[str, str]) -> None:
    listed = [p[0] for p in PAGES]
    assert len(listed) == len(set(listed)), "a page is listed twice"
    missing = set(old) - set(listed) - DROPPED
    assert not missing, f"pages of book.json not placed: {sorted(missing)}"
    assert not (set(listed) - set(old)), f"pages placed that book.json lacks: {sorted(set(listed) - set(old))}"
    assert len(set(ids.values())) == len(ids), "two pages would get one name"
    topics = {t[0] for t in TOPICS}
    for oid, tops, kind, _ in PAGES:
        assert tops and set(tops) <= topics, (oid, tops)
        assert kind in ("overview", "explanation", "experiment", "study", "term", "method"), (oid, kind)
    for name, cited in THESES.items():
        for words, target in cited.items():
            assert target.startswith("topic:") or target in old, (name, words, target)


def added(old: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """The day each page was added: its file's first commit, or the day the index first named it."""
    out = {}
    for oid, c in old.items():
        if c.get("file"):
            days = git("log", "--follow", "--format=%ad", "--date=short", "--", f"book/{c['file']}").split()
        else:
            days = git("log", "--format=%ad", "--date=short", "-S", f'"title": "{c["title"]}"', "--",
                       "book/book.json").split()
        out[oid] = days[-1] if days else ""
    return out


def first_sentence(path: str) -> str:
    """
    A page's summary, until one is written: the questions it opens with, or else the
    first sentence of its first paragraph of prose.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    asks = re.search(r"^> \[!question\][^\n]*\n((?:>[^\n]*\n)+)", text, re.M)
    if asks:
        items = re.split(r"\n> - ", "\n" + asks.group(1).replace("\n>  ", " ").rstrip())
        questions = [" ".join(i.replace("> ", " ").split()) for i in items if i.strip(" >")]
        return " ".join(questions[:3])
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or block[0] in "#>|!-*`$<" or re.match(r"\d+\.", block):
            continue
        prose = " ".join(block.split())
        prose = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", prose)
        m = re.match(r"(.+?[.?!])(\s|$)", prose)
        return (m.group(1) if m else prose)[:400]
    return ""


def rewrite_links(text: str, was: str, now: str, moved: Dict[str, str]) -> str:
    """Every relative link of a text written for book/<was>, as book/<now> must write it."""
    def to(m: "re.Match[str]") -> str:
        href = m.group(2)
        if re.match(r"^[a-z][a-z0-9+.-]*:|^#|^/", href, re.I):
            return m.group(0)
        path, hash_, part = href.partition("#")
        target = os.path.normpath(os.path.join(os.path.dirname(was), path))
        if target.startswith(".."):
            return m.group(0)
        target = moved.get(target, target)
        rel = os.path.relpath(target, os.path.dirname(now) or ".").replace(os.sep, "/")
        return f"{m.group(1)}{rel}{hash_}{part}{m.group(3)}"
    return LINK.sub(to, text)


def rewrite_result(value: Any, moved: Dict[str, str]) -> Any:
    """A results file's text, written for a page in book/chapters/, as a page in book/pages/ writes it."""
    if isinstance(value, str):
        return rewrite_links(value, "chapters/_.md", "pages/_.md", moved)
    if isinstance(value, list):
        return [rewrite_result(v, moved) for v in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "href" and isinstance(v, str) and not re.match(r"^[a-z]+:", v):
                path, hash_, part = v.partition("#")
                out[k] = moved.get(path, path) + hash_ + part
            else:
                out[k] = rewrite_result(v, moved)
        return out
    return value


def index(old: Dict[str, Dict[str, Any]], ids: Dict[str, str], days: Dict[str, str]) -> Dict[str, Any]:
    pages = []
    for oid, tops, kind, _ in PAGES:
        c = old[oid]
        nid = ids[oid]
        page: Dict[str, Any] = {"id": nid, "title": TITLES.get(oid, c["title"])}
        if c.get("file"):
            page["file"] = "README.md" if oid == "about" else f"pages/{nid}.md"
        page["topics"] = tops
        page["kind"] = kind
        if c.get("experiments"):
            page["experiments"] = c["experiments"]
        page["status"] = c.get("status") or ("written" if c.get("file") else "planned")
        page["added"] = days[oid]
        page["summary"] = c.get("note") or (first_sentence(os.path.join(BOOK, c["file"])) if c.get("file") else "")
        pages.append(page)
    return {"title": "Graph of Life — the Research Notebook",
            "topics": [{"id": i, "title": t, "about": a} for i, t, a in TOPICS],
            "pages": pages}


def legacy(old: Dict[str, Dict[str, Any]], ids: Dict[str, str]) -> Dict[str, Any]:
    def page(target: str) -> str:
        return target if target.startswith("topic:") else ids[target]
    return {"about": "The notebook's chapter numbers and note ids until 23 October 2026, and the page each "
                     "became. A thesis is quoted word for word, so it keeps the numbers of its day: `theses` "
                     "says, for each, which page every number it cites meant when it was written.",
            "pages": {oid: ids.get(oid) for oid in old},
            "theses": {name: {words: page(t) for words, t in cited.items()} for name, cited in THESES.items()}}


# ---------------------------------------------------------------------------
# --words: the numbers out of the words
# ---------------------------------------------------------------------------

#: The words of a link that name a chapter by its number, not by what it is.
NUMBERED = re.compile(r"(?:Chapters?\s+\d+|\d{1,3}|Meta\s+[IVXL]+|Appendix\s+[A-C])(?:\s*·.*)?", re.S)
LINKED = re.compile(r"(?:\b(?:[Cc]hapters?|Meta|Appendix)\s+)?\[([^\[\]]+)\]\(([^)\s]+)\)")
PLAIN = re.compile(r"\b[Cc]hapters?\s+(\d+(?:(?:,\s+|,?\s+and\s+|,?\s+or\s+)\d+)*)\b(?!\s*(?:to|–|-)\s*\d)"
                   r"|\bMeta\s+([IVXL]+)\b|\bAppendix\s+([A-C])\b")
#: What is left for a person to word: ranges of chapters, parts, and summaries never written.
LEFT = re.compile(r"\b[Cc]hapters? \d+\s*(?:to|–|-)\s*\d+|\bParts? [IVXL]+(?:(?:, | and | to |–)[IVXL]+)*\b"
                  r"|\bMeta (?:III|IV|V)\b|\bAppendices\b|\b[Cc]hapters? \d+|\b[Cc]hapters? \["
                  r"|\]\([^)\s]*\)(?:,|,? and|,? or| to|–)\s*\d+\b")
WORDS = [
    (r"\b[Tt]he Book tab\b", "the Research Notebook tab"),
    (r"\bBook tab\b", "Research Notebook tab"),
    (r"\b([Tt]his|[Tt]he|[Ww]hole|[Aa])(\s+)book\b", r"\1\2notebook"),
    (r"\bchapters\b", "pages"),
    (r"\bChapters\b", "Pages"),
    (r"\bchapter\b", "page"),
    (r"\bChapter\b(?!\s+\d)", "Page"),
]


def by_file() -> Dict[str, Dict[str, Any]]:
    with open(os.path.join(BOOK, "book.json"), encoding="utf-8") as f:
        return {p["file"]: p for p in json.load(f)["pages"] if p.get("file")}


def target(md: str, href: str, pages: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The page a link written in book/<md> leads to, if it leads to one."""
    path = href.partition("#")[0]
    if re.match(r"^[a-z][a-z0-9+.-]*:", path, re.I):
        return None
    return pages.get(os.path.normpath(os.path.join(os.path.dirname(md), path)) if path else md)


def named(md: str, page: Dict[str, Any], links: Optional[bool]) -> str:
    """
    A page, named by its title: a link from book/<md>, or the title in italics where
    no link can go, or to a page not written yet; with links=None, the title alone.
    """
    if links is None:
        return page["title"]
    if not links or not page.get("file"):
        return f"*{page['title']}*"
    return f"[{page['title']}]({os.path.relpath(page['file'], os.path.dirname(md) or '.').replace(os.sep, '/')})"


def retitle_links(text: str, md: str, pages: Dict[str, Dict[str, Any]]) -> str:
    """Every link whose words are a chapter's number, named instead by the page it leads to."""
    import book_fill
    kept = book_fill.protected_spans(text, links=False)

    def to(m: "re.Match[str]") -> str:
        if any(a <= m.start(1) - 1 < b for a, b in kept) or not NUMBERED.fullmatch(m.group(1).strip()):
            return m.group(0)
        page = target(md, m.group(2), pages)
        ranged = re.search(r"\d\s*(?:to|–)\s*\d", m.group(1)) or re.match(r"\s*(?:to|–)\s*\d", text[m.end():])
        if page is None or ranged:
            return m.group(0)                          # a range, or a page that is not one: left for a person
        return f"[{page['title']}]({m.group(2)})"
    return LINKED.sub(to, text)


def name_numbers(text: str, md: str, pages: Dict[str, Dict[str, Any]], links: Optional[bool] = True) -> str:
    """Every plain "Chapter 13", "Chapters 13 and 14", "Meta X", "Appendix C" as the pages they are."""
    old = legacy_pages()
    with open(os.path.join(BOOK, "book.json"), encoding="utf-8") as f:
        every = {p["id"]: p for p in json.load(f)["pages"]}
    number = {oid: every.get(pid) for oid, pid in old.items() if pid}

    def to(m: "re.Match[str]") -> str:
        if m.group(1):
            ids = re.findall(r"\d+", m.group(1))
            if not all(number.get(i) for i in ids):
                return m.group(0)
            out = m.group(1)
            for i in ids:
                out = re.sub(rf"\b{i}\b", lambda _: named(md, number[i], links), out, count=1)
            return out
        key = META.get(m.group(2)) if m.group(2) else m.group(3)
        page = number.get(key) if key else None
        return named(md, page, links) if page else m.group(0)

    import book_fill
    return book_fill.outside(text, lambda s: PLAIN.sub(to, s))


#: The parts' summaries, by their numerals: the old chapter each was.
META = {"I": "26", "II": "40", "VI": "74", "VII": "75", "VIII": "76", "IX": "77", "X": "79", "XI": "84",
        "XII": "91"}


def legacy_pages() -> Dict[str, Optional[str]]:
    with open(os.path.join(BOOK, "legacy.json"), encoding="utf-8") as f:
        return json.load(f)["pages"]


def reword(text: str) -> str:
    """"The book" and "chapter" in the words of a page, never in a path, code or a formula."""
    import book_fill

    def change(s: str) -> str:
        for pattern, new in WORDS:
            s = re.sub(pattern, new, s)
        return s
    return book_fill.outside(text, change, links=False)


def words_of_page(text: str, md: str, pages: Dict[str, Dict[str, Any]]) -> str:
    return reword(name_numbers(retitle_links(text, md, pages), md, pages))


def words_of_result(value: Any, pages: Dict[str, Dict[str, Any]]) -> Any:
    """A results file's text as words_of_page makes a page's, written as from a page in book/pages/."""
    if isinstance(value, str):
        return words_of_page(value, "pages/_.md", pages)
    if isinstance(value, list):
        return [words_of_result(v, pages) for v in value]
    if isinstance(value, dict):
        if set(value) == {"text", "href"} and NUMBERED.fullmatch(str(value["text"])) and value["href"] in pages:
            return {"text": pages[value["href"]]["title"], "href": value["href"]}
        return {k: words_of_result(v, pages) for k, v in value.items()}
    return value


def anchors_moved(before: str, after: str) -> Dict[str, str]:
    """The ids of a page's headings that its rewording changed: old id -> new id."""
    import book_fill
    old, new = book_fill.anchors(before), book_fill.anchors(after)
    return {a: b for a, b in zip(old, new) if a != b} if len(old) == len(new) else {}


def words() -> None:
    """Rewrite the words of every page, of the index and of the results; list what is left."""
    pages = by_file()
    texts = {}
    for md in sorted(pages):
        with open(os.path.join(BOOK, md), encoding="utf-8") as f:
            texts[md] = f.read()
    new = {md: words_of_page(t, md, pages) for md, t in texts.items()}
    moved = {md: anchors_moved(texts[md], new[md]) for md in new}
    for md, text in new.items():                      # links to a heading whose words changed
        def fix(m: "re.Match[str]") -> str:
            page = target(md, m.group(2), pages)
            path, _, part = m.group(2).partition("#")
            if page and part and part in moved.get(page["file"], {}):
                return f"[{m.group(1)}]({path}#{moved[page['file']][part]})"
            return m.group(0)
        new[md] = re.sub(r"\[([^\]]*)\]\(([^)\s]+#[^)\s]+)\)", fix, text)
    for md, text in new.items():
        if text != texts[md]:
            with open(os.path.join(BOOK, md), "w", encoding="utf-8") as f:
                f.write(text)
    for name in sorted(os.listdir(os.path.join(BOOK, "results"))):
        if name.endswith(".json"):
            path = os.path.join(BOOK, "results", name)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            out = words_of_result(data, pages)
            if out != data:
                with open(path, encoding="utf-8") as f:
                    compact = not f.read(2).startswith("{\n")
                with open(path, "w", encoding="utf-8") as f:   # as gol_stats.write_json wrote it
                    json.dump(out, f, **({"separators": (",", ":")} if compact else {"indent": 1}))
                    f.write("\n")
    with open(os.path.join(BOOK, "book.json"), encoding="utf-8") as f:
        index_ = json.load(f)
    for page in index_["pages"]:
        page["title"] = reword(page["title"])
        page["summary"] = reword(name_numbers(page["summary"], "pages/_.md", pages, links=False))
    with open(os.path.join(BOOK, "book.json"), "w", encoding="utf-8") as f:
        json.dump(index_, f, indent=2, ensure_ascii=False)
        f.write("\n")
    import book_fill
    left = []
    for md in sorted(pages):
        with open(os.path.join(BOOK, md), encoding="utf-8") as f:
            text = " ".join(book_fill.BLOCK.sub("", f.read()).split())
        left += [f"{md}: …{text[max(0, m.start() - 50):m.end() + 30]}…" for m in LEFT.finditer(text)]
    print("\n".join(left))
    print(f"{len(left)} left for a person")


#: The parts, in code: what each was about, as a phrase that reads in a sentence.
PARTS = [(r"\bParts II (?:and|to|–) (?:III|IV)\b", "the study of the baseline"),
         (r"\bPart III\b", "the measurements of the baseline"), (r"\bPart II\b", "the study of the baseline"),
         (r"\bPart IV\b", "the experiments that change one setting"),
         (r"\bPart VII\b", "the long runs"), (r"\bPart VIII\b", "the overviews of every world"),
         (r"\bPart VI\b", "the physics of space"), (r"\bPart IX\b", "the strategies written by hand"),
         (r"\bPart V\b", "the work on open-ended evolution"), (r"\bPart X\b", "the screen of random worlds"),
         (r"\bPart I\b", "how the simulation works")]


def words_of_code(path: str) -> int:
    """
    The numbers out of a module's strings and comments, never its names: a caption or
    a recipe names a page by a link (written as from a page in book/pages/), a
    docstring or a comment by its title; a part by what it was about.
    """
    import io
    import tokenize
    pages = by_file()
    with open(path, encoding="utf-8") as f:
        source = f.read()
    kinds = {tokenize.STRING, tokenize.COMMENT, getattr(tokenize, "FSTRING_MIDDLE", -1)}
    tokens = [t for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type in kinds]
    lines = source.split("\n")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)
    edits = []
    for t in tokens:
        start = offsets[t.start[0] - 1] + t.start[1]
        end = offsets[t.end[0] - 1] + t.end[1]
        text = source[start:end]
        prose = t.type == tokenize.COMMENT or " " in text
        docstring = t.type == tokenize.COMMENT or text.startswith(('"""', "'''"))
        new = re.sub(r"\b(?:Chapter \d+|Appendix [A-C]|Meta [IVXL]+) · ", "", text)
        new = retitle_links(new, "pages/_.md", pages)
        new = name_numbers(new, "pages/_.md", pages, links=None if docstring else True)
        for pattern, phrase in PARTS:
            new = re.sub(pattern, phrase, new)
        if prose:
            for pattern, word in WORDS:
                new = re.sub(pattern, word, new)
        if new != text:
            edits.append((start, end, new))
    for start, end, new in reversed(edits):
        source = source[:start] + new + source[end:]
    if edits:
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
    return len(edits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--move", action="store_true")
    ap.add_argument("--words", action="store_true")
    ap.add_argument("--code", nargs="*")
    a = ap.parse_args()
    if a.words:
        words()
        return
    if a.code is not None:
        for path in a.code:
            print(path, words_of_code(path))
        return
    old = old_index()
    ids = new_ids(old)
    check(old, ids)
    if not a.move:
        for oid, tops, kind, _ in PAGES:
            print(f"{oid:>34} -> {ids[oid]:<55} {kind:<11} {', '.join(tops)}")
        print(f"{len(PAGES)} pages, {len(DROPPED)} dropped")
        return

    days = added(old)
    moved = {}                                         # old path under book/ -> new path
    for oid, _, _, _ in PAGES:
        c = old[oid]
        if c.get("file") and oid != "about":
            moved[c["file"]] = f"pages/{ids[oid]}.md"
    new_index = index(old, ids, days)                  # before the move: summaries read the files where they are
    os.makedirs(os.path.join(BOOK, "pages"), exist_ok=True)
    for was, now in moved.items():
        git("mv", f"book/{was}", f"book/{now}")
    where = {now: was for was, now in moved.items()}
    where["README.md"] = "README.md"
    for now, was in where.items():
        path = os.path.join(BOOK, now)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        new = rewrite_links(text, was, now, moved)
        if new != text:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
    for name in sorted(os.listdir(os.path.join(BOOK, "results"))):
        if name.endswith(".json"):
            path = os.path.join(BOOK, "results", name)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            new = rewrite_result(data, moved)
            if new != data:
                with open(path, "w", encoding="utf-8") as f:   # as gol_stats.write_json writes them
                    json.dump(new, f, indent=1)
                    f.write("\n")
    with open(os.path.join(BOOK, "legacy.json"), "w", encoding="utf-8") as f:
        json.dump(legacy(old, ids), f, indent=1, ensure_ascii=False)
        f.write("\n")
    with open(os.path.join(BOOK, "book.json"), "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2, ensure_ascii=False)
        f.write("\n")
    for folder in ("chapters", "notes"):
        if os.path.isdir(os.path.join(BOOK, folder)) and not os.listdir(os.path.join(BOOK, folder)):
            os.rmdir(os.path.join(BOOK, folder))
    print(f"moved {len(moved)} pages to book/pages/; book.json and book/legacy.json written")


if __name__ == "__main__":
    main()
