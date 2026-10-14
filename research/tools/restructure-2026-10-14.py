"""Move the chapters between parts for the renumbering of 2026-10-14 (run before renumber.py).

Part IV keeps the baseline's one-change experiments; Experiment 9, re-planned on
B2, and the chapter on brain size move to Part V after "Why evolution stops";
"Does size change the dynamics?" is dropped (Chapters 42 and 55 answer it).
"""
import json

path = "book/book.json"
book = json.load(open(path))
parts = {p["title"]: p for p in book["parts"]}
iv = next(p for t, p in parts.items() if t.startswith("Part IV"))
v = next(p for t, p in parts.items() if t.startswith("Part V ·"))
take = {c["title"]: c for c in iv["chapters"]}
rate = take["How fast should brains change?"]
size = take["How big should a brain be?"]
iv["chapters"] = [c for c in iv["chapters"]
                  if c["title"] not in ("How fast should brains change?", "How big should a brain be?",
                                        "Does size change the dynamics?")]
titles = [c["title"] for c in v["chapters"]]
at = titles.index("Why evolution stops") + 1
v["chapters"][at:at] = [rate, size]
with open(path, "w", encoding="utf-8") as f:
    json.dump(book, f, indent=2, ensure_ascii=False)
    f.write("\n")
for p in (iv, v):
    print(p["title"])
    for c in p["chapters"]:
        print("  ", c.get("id"), c["title"])
