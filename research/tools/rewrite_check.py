#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks a rewrite of the notebook's pages into simple English (the author's
request of 2026-10-23) against what the pages said before.

    python3 research/tools/rewrite_check.py REV PAGE [PAGE ...] [--into NEW]

REV is the commit the pages are compared with; PAGE is a page id (its file
name under book/pages/). With --into, the pages were merged into the page NEW,
and the check reads them together.

For every page (or the pages merged, together) it reports:
- **numbers lost**: every number the hand-written words held before (outside
  the blocks book_fill makes) must still be in the words or the blocks after;
- **blocks lost or added**: every `<!-- kind arg -->` block must survive,
  once for each time it was there;
- **notes added later** still standing as boxes, which a rewrite folds into
  the text and the page's How our view changed;
- the sections every page has: its questions, In short, Where to go next
  (a term page has no questions box and needs none);
- **long sentences**, of more than 25 words, to look at again.
Nothing is written; the report is for the person who rewrote the pages.
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys
from typing import Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import book_fill  # noqa: E402

NUMBER = re.compile(r"(?<![\w.])[−-]?\d+(?:[,.]\d+)*(?:%)?")


def old_text(rev: str, page: str) -> str:
    out = subprocess.run(["git", "show", f"{rev}:book/pages/{page}.md"], cwd=ROOT, capture_output=True,
                         text=True)
    if out.returncode:
        raise SystemExit(f"{page}: not in {rev}")
    return out.stdout


def words(text: str) -> str:
    """The words written by hand: the made blocks and the links' targets taken out."""
    text = book_fill.BLOCK.sub("", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def numbers(text: str) -> collections.Counter:
    found = collections.Counter()
    for n in NUMBER.findall(text):
        n = n.replace("−", "-").rstrip("%")
        if n.replace(",", "").replace(".", "").lstrip("-").isdigit():
            found[n.lstrip("-")] += 1
    return found


def blocks(text: str) -> collections.Counter:
    return collections.Counter(f"{kind}{arg}" for kind, arg in
                               re.findall(r"^<!-- (\w+)((?: [^\n]*?)?) -->$", text, re.M))


def sentences(text: str) -> List[str]:
    """The sentences of a page's prose longer than 25 words, paragraph by paragraph and item by item."""
    prose = re.sub(r"^\s*(?:#|>|\||\$\$|<!--).*$", "", words(text), flags=re.M)
    long = []
    for unit in re.split(r"\n\s*\n|\n\s*(?:[-*]|\d+\.)\s+", prose):
        unit = re.sub(r"\$[^$]*\$|`[^`]*`|\*\*|\*|\[|\]", "", " ".join(unit.split()))
        long += [s for s in re.split(r"(?<=[.?!])\s+(?=[A-Z(])", unit) if len(s.split()) > 25]
    return long


def check(before: Dict[str, str], after: str, kind: str) -> List[str]:
    """
    What a rewrite lost or kept. Pages merged into one need each of their blocks
    once (two pages may have shown the same one) and one navigation, not one each.
    """
    report = []
    old_numbers = collections.Counter()
    old_blocks = collections.Counter()
    for text in before.values():
        old_numbers |= numbers(words(text))
        old_blocks = old_blocks | blocks(text) if len(before) > 1 else old_blocks + blocks(text)
    have = numbers(after)
    lost = sorted(n for n in old_numbers if n not in have)
    if lost:
        report.append("numbers lost: " + ", ".join(lost))
    new_blocks = blocks(after)
    if old_blocks - new_blocks:
        report.append("blocks lost: " + ", ".join(sorted((old_blocks - new_blocks).elements())))
    if new_blocks - old_blocks:
        report.append("blocks added: " + ", ".join(sorted((new_blocks - old_blocks).elements())))
    later = re.findall(r"^> \[!\w+\]-? Added later[^\n]*", after, re.M)
    if later:
        report.append(f"{len(later)} notes added later still standing as boxes")
    if kind != "term" and "> [!question]" not in after:
        report.append("no questions box")
    for section in ("## In short", "## Where to go next"):
        if section not in after:
            report.append(f"no {section[3:]}")
    long = sentences(after)
    if long:
        report.append(f"{len(long)} sentences over 25 words, e.g.: " + " | ".join(s[:120] for s in long[:3]))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rev")
    ap.add_argument("pages", nargs="+")
    ap.add_argument("--into")
    a = ap.parse_args()
    kinds = {p["id"]: p.get("kind") for p in book_fill.pages()}
    groups = {a.into: a.pages} if a.into else {p: [p] for p in a.pages}
    bad = 0
    for new, olds in groups.items():
        with open(os.path.join(ROOT, "book", "pages", f"{new}.md"), encoding="utf-8") as f:
            after = f.read()
        report = check({p: old_text(a.rev, p) for p in olds}, after, kinds.get(new, ""))
        print(f"{new}: " + ("fine" if not report else "\n  " + "\n  ".join(report)))
        bad += bool(report)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
