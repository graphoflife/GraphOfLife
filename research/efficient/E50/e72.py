"""Reads Experiment 72: in the four worlds Experiment 71 played under a head count, the later brains' share against the
earlier ones'. A coin is two contests with one deal, the two sides in swapped places; a score is the mean of its coins,
its band two standard errors either side. Beside the thesis's reading: each pair by the era whose world it was
played in, and the share of the brains that played in their own era's world. Plays nothing.   python3 research/efficient/E50/e72.py [tokens]"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e51

WORLDS = ("30 numbers, seed 6301", "30 numbers, seed 6302", "50 numbers, seed 6302", "50 numbers, seed 6303")
STILL = "50 numbers, seed 6303"                      # the one world of Experiment 71 that had no births to speak of

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "share"
    played = e51.coins("E72", what=what)
    out = {}

    def show(title, values):
        r = e51.score(values); out[title] = r
        print(f"{title:<56} " + ("no whole coin yet" if r is None else
              f"{r['score']:.3f} ({r['low']:.3f} to {r['high']:.3f}); {r['coins']} coins, {r['won']} above a half, {r['lost']} below"))

    print({"share": "SHARE OF THE AGENTS (what the thesis reads)", "tokens": "SHARE OF THE TOKENS (beside it)"}[what])
    for early, late in ((26000, 30000), (22500, 26000), (22500, 30000)):
        mine = {world: [next(v for (run, era), v in c["shares"].items() if era == late)
                        for c in played if c["set"] == f"{world}: {late:,} against {early:,}"] for world in WORLDS}
        for world in WORLDS:
            show(f"    {world}: {late:,} against {early:,}", mine[world])
        show(f"  the three with births: {late:,} against {early:,}", sum((mine[w] for w in WORLDS if w != STILL), []))
        show(f"  all four worlds: {late:,} against {early:,}", sum(mine.values(), []))
        for home in (early, late):                   # read afterwards: by the era whose world the contest was played in
            show(f"  all four worlds, in the world of {home:,}: {late:,} against {early:,}",
                 [next(v for (run, era), v in c["shares"].items() if era == late) for c in played
                  if c["set"].endswith(f": {late:,} against {early:,}") and c["home"] == home])
    # Read afterwards: the share of the brains whose own era's world the contest was played in, over all three pairs.
    at_home = lambda cs: [next(v for (run, era), v in c["shares"].items() if era == c["home"]) for c in cs]
    for world in WORLDS:
        show(f"    {world}: the home era's brains", at_home([c for c in played if c["set"].startswith(world + ":")]))
    show("  all four worlds: the home era's brains", at_home(played))
    json.dump(out, open(os.path.join(HERE, "e72.json" if what == "share" else "e72_tokens.json"), "w"), indent=1)
