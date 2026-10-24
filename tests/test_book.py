#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the book's data layer: how a measure of a run is kept
(book_data), where an iteration's frames are, and the helpers the figures
are drawn with (book_figures, book_graph, book_svg).

    python3 tests/test_book.py

A tiny world is run once, into a runs folder of the tests' own, and every
measure that can be taken of a world that short is taken of it. The others
read a settled world of 3,000 iterations and are listed as such, so that a
new measure has to be put on one list or the other.
"""
from __future__ import annotations

import os
import sys

import runner  # first: the repository on the path, and a runs folder of the tests' own

import numpy as np

import book_chapters  # noqa: F401  (registers every measure and chapter)
import book_data as D
import book_figures as F
import book_graph
import book_svg
import gol_run
import gol_store
from gol_config import SimConfig

#: Measures that read a settled world — from iteration 500, often to 3,000 —
#: and so cannot be taken of the tiny one. Every other measure is.
SETTLED_ONLY = {"partners", "mobility", "fragile", "growth", "alike3", "lifedims2", "star"}


def _tiny_run(name: str = "tiny", iterations: int = 30, **overrides) -> str:
    settings = dict(total_tokens=3000, n_nodes=60, k_neighbors=6, hidden_layers=[14, 12],
                    message_amount=2, random_input_amount=2, seed=5, checkpoint_every=10)
    settings.update(overrides)
    run_id = gol_store.create_run(name, SimConfig(**settings))["id"]
    gol_run.advance(run_id, until=iterations)
    return run_id


TINY = None


def tiny() -> str:
    global TINY
    if TINY is None:
        TINY = _tiny_run()
    return TINY


# ---------------------------------------------------------------------------
# Keeping a measure
# ---------------------------------------------------------------------------

def test_a_measure_is_made_once_and_kept_until_the_run_moves_on():
    made = []

    @D.measure("test-count")
    def count(run_id):
        made.append(run_id)
        return {"frames": gol_store.count_frames(run_id)}

    run_id = _tiny_run("kept", 6)
    first = count(run_id)
    assert count(run_id) == first and len(made) == 1, "a kept measure was made again"
    assert first["stamp"] == 6 and first["frames"] == 12

    gol_run.advance(run_id, until=8)
    again = count(run_id)
    assert len(made) == 2 and again["stamp"] == 8 and again["frames"] == 16, \
        "a run that moved on was handed the measure of where it was"
    leftovers = [n for n in os.listdir(os.path.join(gol_store.BASE_DIR, ".book")) if ".part" in n]
    assert not leftovers, f"a write left {leftovers} behind"


def test_a_strict_cache_makes_nothing():
    @D.measure("test-strict")
    def nothing(run_id):
        return {}

    os.environ["GOL_BOOK_CACHE"] = "strict"
    try:
        nothing(tiny())
    except D.NotKept:
        pass
    else:
        raise AssertionError("a strict cache made a measure")
    finally:
        del os.environ["GOL_BOOK_CACHE"]
    assert nothing(tiny())["stamp"] == 30


def test_arrays_are_kept_whole_and_the_stamp_comes_back_a_number():
    @D.measure("test-arrays", file="{run}.test-arrays.npz")
    def arrays(run_id):
        return {"table": np.arange(6, dtype=float).reshape(2, 3)}

    made = arrays(tiny())
    kept = arrays(tiny())
    assert np.array_equal(made["table"], kept["table"]) and D._plain(kept["stamp"]) == 30


def test_what_a_measure_is_asked_with_is_part_of_its_key():
    calls = []

    def want(run_id, window=None):
        return {"iteration": gol_store.load_meta(run_id)["iteration"], "window": window}

    def fits(kept, wanted):
        return kept["iteration"] == wanted["iteration"] and wanted["window"] in (None, kept["window"])

    @D.measure("test-window", field="key", want=want, fits=fits)
    def windowed(run_id, window=None):
        calls.append(window)
        return {"asked": window}

    windowed(tiny(), 3)
    windowed(tiny())                       # answered by the copy made with a window
    windowed(tiny(), 3)
    windowed(tiny(), 4)                    # another window: made again
    assert calls == [3, 4], calls


def test_two_measures_cannot_share_a_name():
    def a(run_id):
        return {}

    def b(run_id):
        return {}

    D.measure("test-twice")(a)
    D.measure("test-twice")(a)             # the same function, seen again: as book_figures run as a script
    try:
        D.measure("test-twice")(b)
    except ValueError:
        return
    raise AssertionError("a second measure took a name already taken")


def test_every_measure_the_book_keeps_can_be_taken_of_a_world_or_says_why_not():
    real = {name: m for name, m in D.MEASURES.items() if not name.startswith("test-")}
    assert SETTLED_ONLY <= set(real), f"no such measures: {SETTLED_ONLY - set(real)}"
    for name, m in sorted(real.items()):
        if name in SETTLED_ONLY:
            continue
        made = m.make(tiny())
        assert isinstance(made, dict) and made, f"{name} made nothing of a world"


def test_a_measure_of_decisions_refuses_a_run_that_kept_none():
    run_id = _tiny_run("silent", 4, export_decisions=False)
    try:
        D.MEASURES["kin"].make(run_id)
    except ValueError:
        return
    raise AssertionError("kinship read a run without decisions as if nobody had staked")


# ---------------------------------------------------------------------------
# Where an iteration's frames are
# ---------------------------------------------------------------------------

def test_frame_at_finds_both_phases_of_an_iteration():
    for t in (0, 7, 29):
        for phase in (1, 2):
            frame = D.frame_at(tiny(), t, phase)
            assert (frame["iteration"], frame["phase"]) == (t, phase)
    assert D.last_iteration(tiny()) == 29


def test_frame_at_refuses_a_frame_that_is_not_the_one_asked_for():
    run_id = _tiny_run("misfiled", 4)
    wrong = gol_store.read_frame(run_id, 2)
    gol_store.write_frame(run_id, 4, wrong)               # iteration 1's frame where 2's should be
    try:
        D.frame_at(run_id, 2, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("frame_at handed out the wrong iteration's frame")

    sparse = _tiny_run("sparse", 6, export_every=2)
    assert D.frame_at(sparse, 4, 2)["iteration"] == 4
    try:
        D.frame_at(sparse, 3, 2)
    except KeyError:
        return
    raise AssertionError("an iteration that was never recorded was read")


# ---------------------------------------------------------------------------
# The book's lists of settings
# ---------------------------------------------------------------------------

def test_the_book_explains_every_setting_a_plan_can_set():
    # Every chapter's table of settings lists them in book/settings.json's order,
    # explains each from it, and links each to its heading in the settings note.
    # A setting the engine gains and these miss is left out of every table, or
    # linked to nowhere.
    import json
    import re
    import gol_plan
    book = os.path.join(runner.ROOT, "book")
    with open(os.path.join(book, "settings.json"), encoding="utf-8") as f:
        glossary = json.load(f)
    order = glossary["order"]
    assert len(order) == len(set(order)), "a setting is listed twice"
    assert set(order) == gol_plan.SETTABLE, (
        f"settings.json lists {sorted(set(order) - gol_plan.SETTABLE)} that a plan cannot set, "
        f"and misses {sorted(gol_plan.SETTABLE - set(order))}")
    missing = [k for k in order if not (glossary["settings"].get(k) or {}).get("meaning")]
    assert not missing, f"no meaning for {missing}"
    import book_fill
    with open(os.path.join(book, book_fill.page_file("settings")), encoding="utf-8") as f:
        headings = set(re.findall(r"^### (\w+)$", f.read(), re.M))
    assert set(order) <= headings, f"the settings note has no heading for {sorted(set(order) - headings)}"


def test_every_link_between_the_books_pages_arrives():
    # The pages are research notes that point at one another; a link to a page
    # that is not there (renamed, merged, not written yet) is a click that leads
    # nowhere. Every relative link to a Markdown page, from a page or the
    # contents, has to name a file that exists.
    import glob
    import re
    book = os.path.join(runner.ROOT, "book")
    pages = glob.glob(os.path.join(book, "pages", "*.md"))
    pages.append(os.path.join(book, "README.md"))
    broken = []
    for page in pages:
        with open(page, encoding="utf-8") as f:
            text = f.read()
        for href in re.findall(r"\]\(([^)#\s:]+\.md)(?:#[^)]*)?\)", text):
            if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(page), href))):
                broken.append(f"{os.path.relpath(page, book)} -> {href}")
    assert not broken, "links to pages that are not there:\n  " + "\n  ".join(broken)


def test_every_page_the_figures_code_links_to_is_there():
    # Captions and recipes are written by code and kept in book/results/; a
    # figure drawn again writes them anew. So a link in the code to a page that
    # was merged away would come back with the next drawing, even after the
    # results were relinked. A link may be split over two string literals
    # ("[words]" "(page.md)"), so every page name in parentheses counts.
    import glob
    import io
    import re
    import tokenize
    pages = os.path.join(runner.ROOT, "book", "pages")
    kinds = {tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)}
    broken = []
    for path in glob.glob(os.path.join(runner.ROOT, "book_chapters", "*.py")) + [
            os.path.join(runner.ROOT, "book_figures.py")]:
        with open(path, encoding="utf-8") as f:
            source = f.read()
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in kinds:
                broken += [f"{os.path.relpath(path, runner.ROOT)}:{token.start[0]} -> {name}"
                           for name in re.findall(r"\(([\w-]+\.md)(?:#[\w-]*)?\)", token.string)
                           if not os.path.exists(os.path.join(pages, name))]
    assert not broken, "the figures' code links to pages that are not there:\n  " + "\n  ".join(broken)


def test_a_merge_leads_page_ids_only_where_the_code_lists_pages():
    # A merge once renamed every quoted word in the figures' code that matched a
    # merged page's id, so data keys and figure names ("alliances", "clustering")
    # became page ids, and the next drawing would have broken. Only the pilot
    # batches' table names pages by id, and every page it names must be there.
    sys.path.insert(0, os.path.join(runner.ROOT, "research", "tools"))
    import json
    import merge_pages
    from book_chapters import catalogue
    code = 'BATCHES = {\n    "A": ("a batch", ["alliances"]),\n}\ng["alliances"] = 1\n'
    assert merge_pages.lead_batch_pages(code, ["alliances"], "another-game") == (
        'BATCHES = {\n    "A": ("a batch", ["another-game"]),\n}\ng["alliances"] = 1\n')
    with open(os.path.join(runner.ROOT, "book", "book.json"), encoding="utf-8") as f:
        written = {p["id"] for p in json.load(f)["pages"] if p.get("file")}
    unknown = [f"{batch}: {page}" for batch, (_, pages) in catalogue.BATCHES.items() for page in pages
               if page not in written]
    assert not unknown, "pilot batches name pages that are not there:\n  " + "\n  ".join(unknown)


def test_no_page_is_named_by_a_number():
    # The notebook is read by topic and by link, from any page, so a page is
    # named by what it is about: "Chapter 13", "Meta X", "Part V" or a link
    # that reads "13" name a place in an order that is gone. A thesis keeps the
    # numbers of its day, word for word, each linked to the page it meant
    # (book/legacy.json); everything else a reader sees is held to this, and so
    # is the code that writes captions, recipes and labels.
    import glob
    import io
    import json
    import re
    import tokenize
    import book_fill
    numbered = re.compile(r"\bChapters? \d|\bMeta [IVXL]+\b|\bParts? [IVXL]+\b|\bAppendix [A-C]\b|\[\d{1,3}\]\(")
    book = os.path.join(runner.ROOT, "book")
    thesis = re.compile(r"^> \[!quote\] The thesis[^\n]*\n(?:>[^\n]*\n)+(?:\n\*Written while[^\n]*\n)?", re.M)
    found = []

    def check(where, text):
        found.extend(f"{where}: …{text[max(0, m.start() - 40):m.end() + 20]}…" for m in numbered.finditer(text))

    for page in glob.glob(os.path.join(book, "pages", "*.md")) + [os.path.join(book, "README.md")]:
        with open(page, encoding="utf-8") as f:
            text = f.read()
        text = thesis.sub("", book_fill.BLOCK.sub(lambda m: "" if m.group(1) == "thesis" else m.group(0), text))
        check(os.path.relpath(page, book), text)
    with open(os.path.join(book, "book.json"), encoding="utf-8") as f:
        index = json.load(f)
    for page in index["pages"]:
        check(f"book.json {page['id']}", page["title"] + " " + page["summary"])
    for path in glob.glob(os.path.join(book, "results", "*.json")) + [os.path.join(book, "settings.json")]:
        with open(path, encoding="utf-8") as f:
            check(os.path.relpath(path, book), f.read())
    for path in glob.glob(os.path.join(runner.ROOT, "book_chapters", "*.py")) + [
            os.path.join(runner.ROOT, "book_figures.py"), os.path.join(runner.ROOT, "web", "index.html")]:
        with open(path, encoding="utf-8") as f:
            source = f.read()
        if path.endswith(".py"):
            kinds = {tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)}
            source = "\n".join(t.string for t in tokenize.generate_tokens(io.StringIO(source).readline)
                               if t.type in kinds and not t.string.startswith(('"""', "'''")))
        check(os.path.relpath(path, runner.ROOT), source)
    assert not found, "pages named by a number:\n  " + "\n  ".join(found[:40])


def test_every_term_has_a_page_that_explains_it():
    # A term links to the page, and the section, that explains it, with a line
    # that says what it is as the link's title; a name two terms share would be
    # linked to whichever came first.
    import book_fill
    names = {}
    for term in book_fill.terms():
        home = book_fill.page_file(term["page"])
        with open(os.path.join(runner.ROOT, "book", home), encoding="utf-8") as f:
            ids = book_fill.anchors(f.read())
        assert not term.get("anchor") or term["anchor"] in ids, f"{term['name']}: no section {term['anchor']} in {home}"
        assert term["says"] and '"' not in term["says"] and "\n" not in term["says"], term["name"]
        for name in [term["name"], *term.get("also", [])]:
            assert name.lower() not in names, f"{name} names both {names[name.lower()]} and {term['name']}"
            names[name.lower()] = term["name"]


def test_a_term_is_linked_where_it_is_first_used():
    # Once per page, at its first use outside headings, code and links; the
    # longest name first; never where the page links to the term already; and
    # linking a page twice changes nothing.
    import book_fill
    md = "pages/x.md"
    text = ("# Degree and dimension\n\nThe spectral dimension of a world, its dimension by balls, and the degree\n"
            "of each; their degrees again, `degree` in code, and [the Gini](gini-coefficient.md) and the Gini again.\n")
    once = book_fill.link_terms(md, text)
    assert book_fill.link_terms(md, once) == once, "linking twice changed the page"
    assert once.startswith("# Degree and dimension\n"), "a heading was linked"
    def href(name):
        term = next(t for t in book_fill.terms() if t["name"] == name)
        page = os.path.basename(book_fill.page_file(term["page"]))
        return page + (f"#{term['anchor']}" if term.get("anchor") else "")
    assert f'[spectral dimension]({href("spectral dimension")} "' in once, once
    assert f'[dimension]({href("dimension")} "' in once, "a shorter name took the longer one's words"
    assert once.count("](degree.md") == 1 and "`degree`" in once, once
    assert once.count("gini-coefficient.md") == 1, "a term the page links already was linked again"
    assert book_fill.link_terms("pages/degree.md", "the degree of an agent").count("degree.md") == 0, \
        "a term was linked on its own page"


def test_every_page_is_filled_as_the_notebook_says():
    # The blocks of a page — figures, theses, runs, contents, and the foot that
    # names its topic and the pages that link to it — are made from book.json,
    # the plans and the results. A page edited without `python3 book_fill.py`
    # after it shows a foot, a list or a link that no longer holds.
    import book_fill
    stale = book_fill.fill(write=False)
    assert stale == 0, f"{stale} pages are not filled as book_fill would fill them: run python3 book_fill.py"


def test_every_number_a_thesis_cites_has_its_page():
    # A thesis keeps the chapter numbers of the day it was written; each must
    # be mapped to the page it meant, or it would link nowhere.
    import glob
    import json
    import re
    import book_fill
    with open(os.path.join(runner.ROOT, "book", "legacy.json"), encoding="utf-8") as f:
        cited = json.load(f)["theses"]
    missing = []
    for path in glob.glob(os.path.join(runner.ROOT, "book", "experiments", "E*.json")):
        name = os.path.basename(path)[:-5]
        with open(path, encoding="utf-8") as f:
            said = " ".join(str(v) for v in (json.load(f).get("thesis") or {}).values())
        for m in re.finditer(r"\bChapters? (\d+(?:(?:, | and | or )\d+)*)|\b(?:Meta|Part) [IVXL]+\b", said):
            words = [f"Chapter {n}" for n in re.findall(r"\d+", m.group(1))] if m.group(1) else [m.group(0)]
            missing += [f"{name}: {w}" for w in words if w not in cited.get(name, {})]
    assert not missing, "numbers in theses that lead nowhere: " + ", ".join(missing)
    for name in cited:
        book_fill.thesis(f"pages/{name}.md", name)          # every page it maps to exists


def test_every_link_to_a_section_or_a_picture_arrives():
    # A link to a section (`settings.md#k_neighbors`, `#what-is-set-aside`) names
    # a heading by the id GitHub gives it, which the notebook's tab gives it too
    # (markdown.js); a heading reworded, or a picture moved, leaves a link that
    # scrolls nowhere or an empty frame.
    import glob
    import re
    from urllib.parse import unquote
    import book_fill
    book = os.path.join(runner.ROOT, "book")
    pages = glob.glob(os.path.join(book, "**", "*.md"), recursive=True)
    ids = {}

    def headings(path):
        if path not in ids:
            with open(path, encoding="utf-8") as f:
                ids[path] = set(book_fill.anchors(f.read()))
        return ids[path]

    broken = []
    for page in pages:
        with open(page, encoding="utf-8") as f:
            text = re.sub(r"^```.*?^```", "", f.read(), flags=re.S | re.M)
        for href, part in re.findall(r"\]\(([^)#\s:]*\.md|)#([^)\s]+)\)", text):
            target = os.path.normpath(os.path.join(os.path.dirname(page), href)) if href else page
            if os.path.exists(target) and unquote(part) not in headings(target):
                broken.append(f"{os.path.relpath(page, book)} -> {href}#{part}")
        for src in re.findall(r"!\[[^\]]*\]\(([^)\s:]+)\)", text):
            if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(page), unquote(src)))):
                broken.append(f"{os.path.relpath(page, book)} -> picture {src}")
    assert not broken, "links to sections or pictures that are not there:\n  " + "\n  ".join(broken)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def test_classes_say_where_a_value_falls():
    classes = F.Classes((1, 1, "1"), (2, 4, "2–4"), (5, 10 ** 9, "5+"))
    assert classes.labels == ["1", "2–4", "5+"]
    assert [classes.which(v) for v in (0, 1, 3, 4, 70)] == [None, 0, 1, 1, 2]
    assert classes.label_of(3) == "2–4" and classes.label_of(0) is None
    masks = classes.masks(np.array([1, 2, 9]))
    assert [m.tolist() for m in masks] == [[True, False, False], [False, True, False], [False, False, True]]


def test_a_quantile_band_reads_past_gaps_and_drops_thin_points():
    table = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, np.nan], [5.0, 6.0, 7.0]])
    band = F.quantile_band(table, [10, 20, 30])
    assert band["x"] == [10, 20, 30] and band["y"][0] == 3.0 and band["y"][1] == 4.0
    thin = F.quantile_band(table, [10, 20, 30], min_count=2)
    assert thin["x"] == [10, 20], "a point only one world reached was drawn"


def test_the_graph_helpers_agree_on_a_small_graph():
    frame = {"ids": [1, 2, 3, 4, 5], "edges": [[1, 2], [2, 3], [3, 1], [3, 4], [4, 5], [5, 5]]}
    adj = book_graph.graph(frame)
    assert adj[5] == {4}, "a self-loop became a neighbour"
    assert book_graph.degrees(frame["edges"])[5] == 3, "a frame's own count of a self-loop is two"
    assert book_graph.bfs(adj, 1) == {1: 0, 2: 1, 3: 1, 4: 2, 5: 3}
    assert book_graph.bfs(adj, 1, limit=1) == {1: 0, 2: 1, 3: 1}
    assert dict(book_graph.triangles(adj)) == {1: 1, 2: 1, 3: 1}
    assert set(book_graph.core(adj)) == {1, 2, 3}


def test_a_chart_that_says_what_is_not_drawn_is_refused():
    good = {"title": "t", "x": {"label": "x"}, "y": {"label": "y", "min": 0},
            "series": [F.line([1, 2], [3, 4], "a line", 0), F.bars(["a", "b"], [1, 2], 1),
                       F.area([1, 2], [0, 0], [1, 1], 2), F.cells([(0, 1, 0, 1, 5)])],
            "colourbar": {"map": "viridis", "min": 0, "max": 5}}
    assert book_svg.validate(good) == []
    book_svg.render([good])
    bad = dict(good, series=[{"x": [1], "y": [1], "color": 1}])
    try:
        book_svg.render([bad], name="bad")
    except ValueError as e:
        assert "color" in str(e)
        return
    raise AssertionError("a misspelt key was drawn as nothing, without a word")


if __name__ == "__main__":
    sys.exit(runner.main(globals()))
