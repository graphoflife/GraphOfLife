#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A rewritten page's line in book.json, from the first sentences of its In short.

    python3 research/tools/summaries.py PAGE [PAGE ...]

The summary is what the notebook's search reads and what a page not yet written
shows; once a page has an In short, its first sentences say best what it is.
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INDEX = os.path.join(ROOT, "book", "book.json")


def summary(text: str, limit: int = 260) -> str:
    m = re.search(r"^## In short\n(.*?)(?=^## |^<!--)", text, re.M | re.S)
    if not m:
        return ""
    prose = " ".join(m.group(1).split())
    prose = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", prose)
    prose = re.sub(r"\*\*|(?<!\w)\*|\*(?!\w)|`", "", prose)
    out = ""
    for sentence in re.split(r"(?<=[.?!])\s+(?=[A-Z])", prose):
        if out and len(out) + len(sentence) > limit:
            break
        out = f"{out} {sentence}".strip()
    return out


def main() -> None:
    with open(INDEX, encoding="utf-8") as f:
        index = json.load(f)
    by_id = {p["id"]: p for p in index["pages"]}
    for page in sys.argv[1:]:
        with open(os.path.join(ROOT, "book", by_id[page]["file"]), encoding="utf-8") as f:
            line = summary(f.read())
        if line:
            by_id[page]["summary"] = line
            print(f"{page}: {line}")
    with open(INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
