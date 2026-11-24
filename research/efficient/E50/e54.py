"""Reads Experiment 54: in the two worlds of equals, the later brains' share against the earlier ones'. A coin is two
contests with one deal, the two sides in swapped places; a score is the mean of its coins, its band two standard errors
either side. Plays nothing.   python3 research/efficient/E50/e54.py [tokens]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e51

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "share"
    played = e51.coins("E54", what=what)
    out = {}

    def show(title, values):
        r = e51.score(values); out[title] = r
        print(f"{title:<44} " + ("no whole coin yet" if r is None else
              f"{r['score']:.3f} ({r['low']:.3f} to {r['high']:.3f}); {r['coins']} coins, {r['won']} above a half, {r['lost']} below"))

    print({"share": "SHARE OF THE AGENTS (what the thesis reads)", "tokens": "SHARE OF THE TOKENS (beside it)"}[what])
    for early, late in ((23000, 25000), (25000, 27500), (23000, 27500)):
        pooled = []
        for seed in (5301, 5302):
            mine = [next(v for (run, era), v in c["shares"].items() if era == late)
                    for c in played if c["set"] == f"seed {seed}: {late:,} against {early:,}"]
            show(f"    seed {seed}: {late:,} against {early:,}", mine); pooled += mine
        show(f"  both runs: {late:,} against {early:,}", pooled)
    json.dump(out, open(os.path.join(HERE, "e54.json" if what == "share" else "e54_tokens.json"), "w"), indent=1)
