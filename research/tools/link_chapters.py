"""
Make every mention of a written chapter a link to it.

The book's chapters and notes refer to one another as "Chapter 58" or
"Chapters 48 and 49". Most of those are already links; this finds the rest and
links them, to the chapter's file as book/book.json names it. It changes no
word: "Chapter 58" becomes "[Chapter 58](58-gliders-and-who-helps-whom.md)",
"Chapters 48 and 49" becomes "Chapters [48](…) and [49](…)".

Left alone: a chapter not written yet (book.json has no file for it), a
chapter's mention of itself, headings, code, existing links, and everything
between the markers book_fill.py fills in (figures, theses, runs, turns,
contents, costs), which is written by the build, not by hand.

    python3 research/tools/link_chapters.py          # say what would change
    python3 research/tools/link_chapters.py --write  # change it
"""
import argparse
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOOK = os.path.join(ROOT, "book")

#: What may not be touched: fenced code, marker regions, comments, inline code,
#: existing links and images, and headings.
PROTECTED = re.compile(
    r"```.*?```"
    r"|<!-- (figure|thesis|runs|turns|contents|costs)\b[^>]*-->.*?<!-- /\1 -->"
    r"|<!--.*?-->"
    r"|`[^`\n]*`"
    r"|!?\[[^\]\n]*\]\([^)\n]*\)"
    r"|^#{1,6} [^\n]*$",
    re.S | re.M)
LIST = r"\d+(?:(?:, | and | or |–|-)\d+)*"
MENTION = re.compile(rf"\b(Chapters?) ({LIST})\b")


def chapter_files():
    book = json.load(open(os.path.join(BOOK, "book.json")))
    return {str(c["id"]): os.path.basename(c["file"])
            for part in book["parts"] for c in part["chapters"]
            if c.get("file") and str(c["id"]).isdigit()}


def link_text(text, files, prefix, own):
    """The text with its mentions of chapters linked, and how many were."""
    count = 0

    def target(number):
        name = files.get(str(int(number)))
        return None if name is None or name == own else prefix + name

    def one(match):
        nonlocal count
        word, numbers = match.group(1), match.group(2)
        found = re.findall(r"\d+", numbers)
        if word == "Chapter" and len(found) == 1:
            href = target(found[0])
            if not href:
                return match.group(0)
            count += 1
            return f"[{word} {numbers}]({href})"
        linked = re.sub(r"\d+", lambda m: f"[{m.group(0)}]({target(m.group(0))})" if target(m.group(0)) else m.group(0),
                        numbers)
        if linked != numbers:
            count += 1
        return f"{word} {linked}"

    out, at = [], 0
    for m in PROTECTED.finditer(text):
        out.append(MENTION.sub(one, text[at:m.start()]))
        out.append(m.group(0))
        at = m.end()
    out.append(MENTION.sub(one, text[at:]))
    return "".join(out), count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    files = chapter_files()
    total = 0
    for folder, prefix in (("chapters", ""), ("notes", "../chapters/")):
        for path in sorted(glob.glob(os.path.join(BOOK, folder, "*.md"))):
            text = open(path, encoding="utf-8").read()
            new, n = link_text(text, files, prefix, os.path.basename(path) if folder == "chapters" else None)
            if n:
                total += n
                print(f"{n:3d}  {os.path.relpath(path, ROOT)}")
                if a.write:
                    open(path, "w", encoding="utf-8").write(new)
    print(f"{total} mentions {'linked' if a.write else 'to link'}")


if __name__ == "__main__":
    main()
