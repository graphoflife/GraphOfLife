#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Renumber the book's chapters: their files, book.json, and every reference.

    python3 research/tools/renumber.py MAPPING.json            # what would change
    python3 research/tools/renumber.py MAPPING.json --apply    # change it

MAPPING is a JSON object from old chapter numbers to new ones, as strings:
{"43": "51", "49": "46", ...}; a chapter mapped to null is dropped, which is
allowed only for a chapter without a file. It must be a permutation of the
numbers it moves (no two old numbers to one new one).

What it rewrites, in one pass each so that a permutation cannot chain:
- the file of every written chapter that moves (git mv), keeping its slug;
- book.json: every chapter's id, and the order of the parts' chapter lists
  is left to the caller (move entries by hand before or after);
- in Markdown, Python, and the JSON of results and of experiments without
  runs: "Chapter N", "Chapters N–M" (and "to", "and", "or", commas between),
  "Ch. N", the file names of moved chapters, and "### N ·" headings in
  research notes.

It never touches the thesis of an experiment that has runs (pre-registered
text is not reworded); it lists those that cite a moved chapter, so a note
can be added after the thesis. It reports every remaining mention of a
dropped chapter.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TEXT_GLOBS = ("book/chapters/*.md", "book/notes/*.md", "book/README.md", "research/*.md",
              "research/oee/pilots/LOG.md", "book_chapters/*.py", "book_figures.py", "book_fill.py",
              "gol_tournament.py", "book/results/*.json")
NUMBER_RUN = r"(\d+)((?:\s*(?:[–-]|to|and|or|,)\s*\d+)*)"
REFERENCE = re.compile(r"\b(Chapters?|chapters?|Ch\.)(\s+)" + NUMBER_RUN)
HEADING = re.compile(r"^(###\s+)(\d+)(\s+·)", re.M)


def load_mapping(path: str) -> Dict[str, Optional[str]]:
    mapping = json.load(open(path))
    targets = [v for v in mapping.values() if v is not None]
    if len(targets) != len(set(targets)):
        raise SystemExit("two chapters would get the same number")
    return mapping


def renumber_text(text: str, mapping: Dict[str, Optional[str]]) -> Tuple[str, int, List[str]]:
    """The text with every reference renumbered, how many changed, and mentions of dropped chapters."""
    changed = 0
    dropped: List[str] = []

    def number(n: str) -> str:
        nonlocal changed
        if n in mapping:
            if mapping[n] is None:
                dropped.append(n)
                return n
            if mapping[n] != n:
                changed += 1
            return mapping[n]
        return n

    def reference(m: re.Match) -> str:
        rest = re.sub(r"\d+", lambda k: number(k.group(0)), m.group(4))
        return m.group(1) + m.group(2) + number(m.group(3)) + rest

    text = REFERENCE.sub(reference, text)
    text = HEADING.sub(lambda m: m.group(1) + number(m.group(2)) + m.group(3), text)
    return text, changed, dropped


def experiments_with_runs() -> set:
    """Experiments whose runs exist: their theses are not to be reworded."""
    out = set()
    for path in glob.glob(os.path.join(ROOT, "book/experiments/E*.json")):
        name = os.path.basename(path)[:-5]
        try:
            sys.path.insert(0, ROOT)
            import gol_plan
            import gol_store as store
            if any(os.path.exists(store.meta_path(s.run_id)) for s in gol_plan.experiment_runs(name)):
                out.add(name)
        except Exception:
            out.add(name)                      # when in doubt, leave it alone
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mapping")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    mapping = load_mapping(a.mapping)
    book_path = os.path.join(ROOT, "book/book.json")
    book = json.load(open(book_path))

    # Files of written chapters that move.
    moves: Dict[str, str] = {}
    for part in book["parts"]:
        for ch in part.get("chapters", []):
            cid = str(ch.get("id", ""))
            if cid in mapping and ch.get("file"):
                if mapping[cid] is None:
                    raise SystemExit(f"chapter {cid} has a file and cannot be dropped")
                old = ch["file"]
                base = os.path.basename(old)
                new = os.path.join(os.path.dirname(old), re.sub(r"^\d+", mapping[cid].zfill(2), base))
                moves[old] = new
    names = {os.path.basename(o): os.path.basename(n) for o, n in moves.items() if o != n}

    def rename_files(text: str) -> Tuple[str, int]:
        count = 0
        for old, new in names.items():
            k = text.count(old)
            if k:
                text = text.replace(old, new)
                count += k
        return text, count

    with_runs = experiments_with_runs()
    report: List[str] = []
    dropped_mentions: List[str] = []
    paths = sorted({p for g in TEXT_GLOBS for p in glob.glob(os.path.join(ROOT, g))})
    paths += sorted(glob.glob(os.path.join(ROOT, "book/experiments/*.json")))
    edits: Dict[str, str] = {}
    for path in paths:
        rel = os.path.relpath(path, ROOT)
        text = open(path, encoding="utf-8").read()
        name = os.path.basename(path)[:-5] if rel.startswith("book/experiments/") else None
        if name in with_runs:
            hits = sorted(set(re.findall(r"Chapter (\d+)", json.dumps(json.load(open(path)).get("thesis", {}),
                                                                     ensure_ascii=False))) & set(mapping))
            if hits:
                report.append(f"  thesis of {name} cites moved chapter(s) {', '.join(hits)}: add a note after it")
            continue
        new, n, dropped = renumber_text(text, mapping)
        new, k = rename_files(new)
        for d in dropped:
            dropped_mentions.append(f"  {rel}: mentions dropped Chapter {d}")
        if new != text:
            edits[path] = new
            report.append(f"  {rel}: {n} numbers, {k} file names")

    # book.json: ids and files.
    for part in book["parts"]:
        for ch in part.get("chapters", []):
            cid = str(ch.get("id", ""))
            if cid in mapping and mapping[cid] is not None:
                ch["id"] = mapping[cid]
            if ch.get("file") in moves:
                ch["file"] = moves[ch["file"]]
            for key in ("note", "title"):
                if isinstance(ch.get(key), str):
                    ch[key], _, _ = renumber_text(ch[key], mapping)
    print("files to move:")
    for o, n in moves.items():
        if o != n:
            print(f"  {o} -> {n}")
    print("texts to change:")
    print("\n".join(report) or "  none")
    if dropped_mentions:
        print("mentions of dropped chapters (fix by hand):")
        print("\n".join(dropped_mentions))
    if not a.apply:
        print("\n(dry run: nothing changed; add --apply)")
        return
    for path, text in edits.items():
        open(path, "w", encoding="utf-8").write(text)
    for o, n in moves.items():
        if o != n:
            subprocess.run(["git", "mv", os.path.join("book", o), os.path.join("book", n)], cwd=ROOT, check=True)
    with open(book_path, "w", encoding="utf-8") as f:
        json.dump(book, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("\napplied.")


if __name__ == "__main__":
    main()
