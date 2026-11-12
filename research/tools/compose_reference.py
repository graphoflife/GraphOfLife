#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composes a reference page of the Research Notebook from short pages (the
author, 2026-11-09: "Simplify the research notes as much as possible ... there
are so many pages"; they chose to merge the term and method pages into a few
reference pages, one section for each old page, with every link still landing
where it did).

    python3 research/tools/compose_reference.py NEW "Title" "intro" OLD [OLD ...]           what it would do
    python3 research/tools/compose_reference.py NEW "Title" "intro" OLD [OLD ...] --apply   do it

It writes book/pages/NEW.md: the title, one question from each old page, an
In short of the intro and a line for each old page, then each old page as a
section under its own title (its In short as the section's first paragraph,
its other sections one level down), and one Where to go next, How our view
changed and References gathered from them all. A heading that two of the old
pages share is renamed with its page's short name, so that the new page has no
two headings alike; links to it, anywhere in the notebook, are pointed at the
new name, and links to an old page's In short at its section. NEW is entered in
book/book.json beside the old pages. Then, with --apply, it runs
research/tools/merge_pages.py, which leads every link, term and old number to
NEW and removes the old pages. Afterwards: `python3 book_fill.py` and the tests.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BOOK = os.path.join(ROOT, "book")
sys.path.insert(0, ROOT)

import book_fill  # noqa: E402

GATHERED = ("in short", "where to go next", "how our view changed", "references")


def read_page(page: str) -> str:
    """A page's own text: everything above the navigation book_fill writes at its foot."""
    with open(os.path.join(BOOK, "pages", f"{page}.md"), encoding="utf-8") as f:
        text = f.read()
    cut = text.find("<!-- turns -->")
    text = text[:cut] if cut >= 0 else text
    return text.rstrip().removesuffix("---").rstrip() + "\n"


def split(text: str) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    """A page as its title, its questions, and its sections (heading, body), outside code fences."""
    lines = text.split("\n")
    title = lines[0][2:].strip() if lines and lines[0].startswith("# ") else ""
    i = 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    questions = []
    if i < len(lines) and lines[i].startswith("> [!question]"):
        i += 1
        while i < len(lines) and lines[i].startswith(">"):
            item = lines[i][1:].strip()
            if item.startswith("- "):
                questions.append(item[2:])
            elif item and questions:
                questions[-1] += " " + item
            i += 1
    sections: List[Tuple[str, List[str]]] = [("", [])]
    fenced = False
    for line in lines[i:]:
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)
    out = [(h, "\n".join(body).strip("\n")) for h, body in sections]
    return title, questions, [s for s in out if s[0] or s[1].strip()]


def demote(body: str) -> str:
    """Every heading of a section one level down, outside code fences."""
    out, fenced = [], False
    for line in body.split("\n"):
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and re.match(r"^#{2,5} ", line):
            line = "#" + line
        out.append(line)
    return "\n".join(out)


def headings(body: str) -> List[str]:
    text = re.sub(r"^```.*?^```", "", body, flags=re.S | re.M)
    return re.findall(r"^#{1,6}\s+(.*?)\s*$", text, re.M)


def short_name(title: str) -> str:
    """An old page's short name, for a heading it shares with another: its title up to a colon or comma."""
    return re.split(r"[:,?]", re.sub(r"\s*\(.*?\)", "", title))[0].strip().lower()


def compose(new: str, title: str, intro: str, olds: List[str]) -> Tuple[str, Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    The new page's text; each old page's section id in it; and, for each old page, its headings
    that were renamed (old id -> new id), its In short included.
    """
    index = {p["id"]: p for p in json.load(open(os.path.join(BOOK, "book.json"), encoding="utf-8"))["pages"]}
    parts = {old: split(read_page(old)) for old in olds}
    # Headings two old pages share, or that the new page's own sections take.
    count: Dict[str, int] = {}
    for old, (_, _, sections) in parts.items():
        names = set()
        for h, body in sections:
            if h.lower() in GATHERED:
                continue
            names.add(book_fill.slug(h))
            names.update(book_fill.slug(x) for x in headings(body))
        for n in names:
            count[n] = count.get(n, 0) + 1
    taken = {book_fill.slug(x) for x in ("In short", "Where to go next", "How our view changed", "References")}
    taken |= {book_fill.slug(parts[old][0]) for old in olds}
    shared = {n for n, c in count.items() if c > 1} | taken

    section_of: Dict[str, str] = {}
    renamed: Dict[str, Dict[str, str]] = {}
    body_parts: List[str] = []
    onward: List[str] = []
    seen_onward = set()
    turns: List[str] = []
    references: List[str] = []
    for old in olds:
        t, _, sections = parts[old]
        sid = book_fill.slug(t)
        section_of[old] = sid
        moved = {"in-short": sid}
        out = [f"## {t}", ""]
        for h, body in sections:
            key = h.lower()
            if key == "in short":
                out += [body.strip(), ""]
                continue
            if key == "where to go next":
                for item in re.findall(r"^- .*?(?=^- |\Z)", body + "\n", re.S | re.M):
                    target = re.search(r"\]\(([\w-]+)\.md", item)
                    page = target.group(1) if target else item
                    if page in olds or page in seen_onward:
                        continue
                    seen_onward.add(page)
                    onward.append(item.rstrip())
                continue
            if key == "how our view changed":
                lines = [ln for ln in body.split("\n") if ln.strip()]
                turns.append(f"- *{t}*:\n" + "\n".join("  " + ln for ln in lines))
                continue
            if key == "references":
                references.append(body.strip())
                continue
            block = (f"### {h}\n\n" if h else "") + demote(body.strip())
            for x in headings(block):
                s = book_fill.slug(x)
                if s in shared:
                    nicer = f"{x} ({short_name(t)})"
                    block = re.sub(rf"^(#+ ){re.escape(x)}\s*$", lambda m: m.group(1) + nicer, block, count=1,
                                   flags=re.M)
                    moved[s] = book_fill.slug(nicer)
            out += [block, ""]
        text = "\n".join(out)
        # Links within the old page to its own renamed headings.
        for a, b in moved.items():
            text = text.replace(f"](#{a})", f"](#{b})")
        renamed[old] = moved
        body_parts.append(text.rstrip() + "\n")

    # One question from each old page that asked any; the In short lists every page anyway.
    questions = [parts[old][1][0] for old in olds if parts[old][1]]
    lines = [f"# {title}", "", "> [!question] Questions of this page"]
    lines += [f"> - {q}" for q in questions]
    # Each old page is a section now; its title says what it holds, and the section opens with the rest.
    lines += ["", "## In short", "", intro.strip(), "", "Its sections:", ""]
    lines += [f"- [{parts[old][0]}](#{section_of[old]})" for old in olds]
    lines.append("")
    text = "\n".join(lines) + "\n" + "\n".join(body_parts)
    if onward:
        text += "\n## Where to go next\n\n" + "\n".join(onward) + "\n"
    if turns:
        text += "\n## How our view changed\n\n" + "\n".join(turns) + "\n"
    if references:
        text += "\n## References\n\n" + "\n\n".join(references) + "\n"
    return text, section_of, renamed


def repoint(olds: List[str], renamed: Dict[str, Dict[str, str]], apply: bool) -> List[str]:
    """Links anywhere to an old page's renamed heading, pointed at its new name, before the merge leads them on."""
    files = [os.path.join(BOOK, md) for md in book_fill.markdown_files()]
    files += [os.path.join(BOOK, "results", n) for n in os.listdir(os.path.join(BOOK, "results")) if n.endswith(".json")]
    files += [os.path.join(ROOT, "book_chapters", n) for n in os.listdir(os.path.join(ROOT, "book_chapters"))
              if n.endswith(".py")]
    files.append(os.path.join(BOOK, "terms.json"))
    changed = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        out = text
        for old in olds:
            for a, b in renamed[old].items():
                if a == b:
                    continue
                out = re.sub(rf"(?<![\w-]){re.escape(old)}\.md#{re.escape(a)}(?=\)| |\\?\")",
                             f"{old}.md#{b}", out)
        if path.endswith("terms.json"):
            data = json.loads(out)
            for term in data["terms"]:
                moved = renamed.get(term.get("page"), {})
                if term.get("anchor") in moved:
                    term["anchor"] = moved[term["anchor"]]
            out = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        if out != text:
            changed.append(os.path.relpath(path, ROOT))
            if apply:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(out)
    return changed


def register(new: str, title: str, olds: List[str], summary: str) -> None:
    """NEW in book.json, just before the first of its old pages: a reference page about their topics."""
    path = os.path.join(BOOK, "book.json")
    index = json.load(open(path, encoding="utf-8"))
    pages = index["pages"]
    if any(p["id"] == new for p in pages):
        return
    at = min(i for i, p in enumerate(pages) if p["id"] in olds)
    topics = ["reference"] + [t for old in olds for p in pages if p["id"] == old for t in p["topics"]]
    entry = {"id": new, "title": title, "file": f"pages/{new}.md",
             "topics": list(dict.fromkeys(topics)), "kind": "reference", "status": "written",
             "added": "2026-11-09", "summary": summary}
    pages.insert(at, entry)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(index, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("new")
    ap.add_argument("title")
    ap.add_argument("intro")
    ap.add_argument("olds", nargs="+")
    ap.add_argument("--summary", default="")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    text, section_of, renamed = compose(a.new, a.title, a.intro, a.olds)
    ids = book_fill.anchors(text)
    doubled = sorted({i for i in ids if ids.count(i) > 1})
    if doubled:
        raise SystemExit(f"{a.new}: headings still alike: {', '.join(doubled)}")
    changed = repoint(a.olds, renamed, a.apply)
    print(f"{a.new}.md: {len(text.splitlines())} lines from {len(a.olds)} pages; renamed headings: "
          f"{sum(len(m) - 1 for m in renamed.values())}; files repointed: {len(changed)}")
    if not a.apply:
        return
    with open(os.path.join(BOOK, "pages", f"{a.new}.md"), "w", encoding="utf-8") as f:
        f.write(text)
    register(a.new, a.title, a.olds, a.summary or a.intro.split(". ")[0] + ".")
    args = [f"{old}:{section_of[old]}" for old in a.olds]
    subprocess.run([sys.executable, os.path.join(ROOT, "research", "tools", "merge_pages.py"), a.new, *args,
                    "--apply"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
