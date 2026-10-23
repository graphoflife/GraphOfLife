#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merges pages of the Research Notebook into one (the author's request of
2026-10-23: pages that tell one story become one page; the old titles stay
as sections).

    python3 research/tools/merge_pages.py NEW OLD:section [OLD:section ...]           what it would change
    python3 research/tools/merge_pages.py NEW OLD:section [OLD:section ...] --apply   change it

NEW is the page that stays; its file must already hold the merged text, with a
section for each page merged into it. OLD is a page merged into NEW, and
`section` the id of the heading in NEW where its text now is.

What --apply does, so that nothing that pointed at an old page points nowhere:
- every link to OLD — in the pages, the front page, the results the figures were
  drawn with, and the strings of the figures' code — leads to NEW#section; a link
  to OLD#part leads to NEW#part when NEW has that section, else to NEW#section;
- book/terms.json: a term explained on OLD is explained on NEW, at its section;
- book/legacy.json: an old number or note id that became OLD, and every number
  a thesis cites that meant OLD, now leads to NEW (theses keep their words);
- the figures' code that names OLD as a page id (the catalogue's batches) names NEW;
- book.json: OLD's entry is gone; NEW reports OLD's experiments too, in reading
  order, and is about OLD's topics as well;
- `git rm` OLD's file.
Then run `python3 book_fill.py` and the tests.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BOOK = os.path.join(ROOT, "book")
sys.path.insert(0, ROOT)

import book_fill  # noqa: E402

LINK = re.compile(r"\]\(([^)\s]+)((?: \"[^\"]*\")?)\)")


def load(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(path: str, value: Any, compact: bool = False, indent: int = 2, ascii_: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, **({"separators": (",", ":")} if compact else {"indent": indent}), ensure_ascii=ascii_)
        f.write("\n")


def plan(new: str, merged: List[Tuple[str, str]]) -> Dict[str, str]:
    """Old page file (under book/) -> the new target, `pages/NEW.md#section`."""
    with open(os.path.join(BOOK, "pages", f"{new}.md"), encoding="utf-8") as f:
        ids = set(book_fill.anchors(f.read()))
    out = {}
    for old, section in merged:
        if section not in ids:
            raise SystemExit(f"{new}.md has no section {section!r} for {old}")
        if not os.path.exists(os.path.join(BOOK, "pages", f"{old}.md")):
            raise SystemExit(f"no page {old}")
        out[f"pages/{old}.md"] = section
    return out


def retarget(href: str, md: str, moved: Dict[str, str], new: str, ids: set) -> str:
    """A link written in book/<md>, led to NEW if it led to a page merged into it."""
    if re.match(r"^[a-z][a-z0-9+.-]*:", href) or href.startswith("#"):
        return href
    path, _, part = href.partition("#")
    target = os.path.normpath(os.path.join(os.path.dirname(md), path)).replace(os.sep, "/")
    if target not in moved:
        return href
    section = part if part in ids else moved[target]
    rel = os.path.relpath(os.path.join("pages", f"{new}.md"), os.path.dirname(md) or ".").replace(os.sep, "/")
    return f"{rel}#{section}"


def rewrite_links(text: str, md: str, moved: Dict[str, str], new: str, ids: set) -> str:
    return LINK.sub(lambda m: f"]({retarget(m.group(1), md, moved, new, ids)}{m.group(2)})", text)


def rewrite_value(value: Any, moved: Dict[str, str], new: str, ids: set) -> Any:
    """A results file's text (written as from a page in book/pages/), and its table cells (from book/)."""
    if isinstance(value, str):
        return rewrite_links(value, "pages/_.md", moved, new, ids)
    if isinstance(value, list):
        return [rewrite_value(v, moved, new, ids) for v in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "href" and isinstance(v, str):
                out[k] = retarget(v, "_.md", moved, new, ids)
            else:
                out[k] = rewrite_value(v, moved, new, ids)
        return out
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("new")
    ap.add_argument("merged", nargs="+", help="OLD:section")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    merged = [tuple(m.split(":", 1)) for m in a.merged]
    moved = plan(a.new, merged)
    olds = {old for old, _ in merged}
    with open(os.path.join(BOOK, "pages", f"{a.new}.md"), encoding="utf-8") as f:
        ids = set(book_fill.anchors(f.read()))
    changes = []

    # Pages and the front page.
    for md in book_fill.markdown_files():
        if md in moved:
            continue
        path = os.path.join(BOOK, md)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        out = rewrite_links(text, md, moved, a.new, ids)
        if out != text:
            changes.append(md)
            if a.apply:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(out)

    # Results, written as from a page in book/pages/.
    for name in sorted(os.listdir(os.path.join(BOOK, "results"))):
        if not name.endswith(".json"):
            continue
        path = os.path.join(BOOK, "results", name)
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        data = json.loads(raw)
        out = rewrite_value(data, moved, a.new, ids)
        if out != data:
            changes.append(f"results/{name}")
            if a.apply:
                save(path, out, compact=not raw.startswith("{\n"), indent=1, ascii_=True)

    # The strings of the figures' code: links, and page ids in the catalogue.
    code = sorted(os.path.join(ROOT, "book_chapters", n) for n in os.listdir(os.path.join(ROOT, "book_chapters"))
                  if n.endswith(".py")) + [os.path.join(ROOT, "book_figures.py")]
    for path in code:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        out = rewrite_links(text, "pages/_.md", moved, a.new, ids)
        for old in olds:
            out = out.replace(f'"{old}"', f'"{a.new}"')
        if out != text:
            changes.append(os.path.relpath(path, ROOT))
            if a.apply:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(out)

    # Terms, the old numbers, and the index.
    terms = load(os.path.join(BOOK, "terms.json"))
    for term in terms["terms"]:
        if term["page"] in olds:
            term["anchor"] = dict(merged)[term["page"]]
            term["page"] = a.new
            changes.append(f"terms.json: {term['name']}")
    legacy = load(os.path.join(BOOK, "legacy.json"))
    for oid, pid in legacy["pages"].items():
        if pid in olds:
            legacy["pages"][oid] = a.new
            changes.append(f"legacy.json: {oid}")
    for name, cited in legacy["theses"].items():
        for words, pid in cited.items():
            if pid in olds:
                cited[words] = a.new
                changes.append(f"legacy.json: {name} {words}")
    index = load(os.path.join(BOOK, "book.json"))
    keep = next(p for p in index["pages"] if p["id"] == a.new)
    for page in index["pages"]:
        if page["id"] in olds:
            for e in page.get("experiments", []):
                keep.setdefault("experiments", [])
                if e not in keep["experiments"]:
                    keep["experiments"].append(e)
            keep["topics"] += [t for t in page["topics"] if t not in keep["topics"]]
    index["pages"] = [p for p in index["pages"] if p["id"] not in olds]
    changes.append("book.json")

    print("\n".join(changes))
    if a.apply:
        save(os.path.join(BOOK, "terms.json"), terms)
        save(os.path.join(BOOK, "legacy.json"), legacy, indent=1)
        save(os.path.join(BOOK, "book.json"), index)
        for old in sorted(olds):
            subprocess.run(["git", "rm", "-q", f"book/pages/{old}.md"], cwd=ROOT, check=True)
        print(f"merged {', '.join(sorted(olds))} into {a.new}; now run python3 book_fill.py")


if __name__ == "__main__":
    main()
