"""Reads Experiment 74: brains of two eras of a world of Experiment 73, played in each era's world. The thesis reads
the home era's share: the share of the agents held at the end by the brains of the era in whose world the contest was
played, pooled over the four pairs between which the world changed. A coin is two contests with one deal, the two
sides in swapped places; a score is the mean of its coins, its band two standard errors either side.
Plays nothing.   python3 research/efficient/E50/e74.py [tokens]"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e51

CHANGED = ("30 numbers, seed 6301: 31,500 against 30,000", "30 numbers, seed 6301: 37,500 against 31,500",
           "50 numbers, seed 6302: 36,000 against 30,000", "30 numbers, seed 6302: 37,500 against 36,000")
SAME = ("50 numbers, seed 6303: 37,500 against 30,000", "30 numbers, seed 6302: 33,500 against 31,000",
        "50 numbers, seed 6302: 37,500 against 35,500")


def era(text):
    return int(text.replace(",", ""))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "share"
    played = e51.coins("E74", what=what)
    out = {}

    def show(title, values):
        r = e51.score(values); out[title] = r
        print(f"{title:<72} " + ("no whole coin yet" if r is None else
              f"{r['score']:.3f} ({r['low']:.3f} to {r['high']:.3f}); {r['coins']} coins, {r['won']} above a half, {r['lost']} below"))

    def of(c, which):                                  # the share of one era's brains in a coin
        return next(v for (run, e), v in c["shares"].items() if e == which)

    print({"share": "SHARE OF THE AGENTS (what the thesis reads)", "tokens": "SHARE OF THE TOKENS (beside it)"}[what])
    for title, group in (("the four pairs of a world that changed (the thesis)", CHANGED), ("the three pairs of a world that did not", SAME)):
        print(f"the home era's brains, {title}")
        for name in group:
            show(f"    {name}", [of(c, c["home"]) for c in played if c["set"] == name])
        show(f"  pooled: {title}", [of(c, c["home"]) for c in played if c["set"] in group])
    print("the later era's brains, each pair in each of its two worlds")
    for name in CHANGED + SAME:
        late, early = (era(t) for t in name.split(": ")[1].split(" against "))
        for home in (early, late):
            show(f"    {name}, in the world of {home:,}", [of(c, late) for c in played if c["set"] == name and c["home"] == home])
    json.dump(out, open(os.path.join(HERE, "e74.json" if what == "share" else "e74_tokens.json"), "w"), indent=1)
