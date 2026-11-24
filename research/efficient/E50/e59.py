"""Reads Experiment 59: the brains of a world of equals against those of the world it fell back to under the engine's
mutation, on both worlds, and the two worlds of equals against each other. A coin is two contests with one deal, the
two sides in swapped places. Plays nothing.   python3 research/efficient/E50/e59.py [tokens]"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e51
import gol_plan

END = 27500

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "share"
    runs = {(s.condition, s.lab["seed"]): s.run_id for s in gol_plan.experiment_runs("E53")}
    played = e51.coins("E59", what=what)
    out = {}

    def show(title, values):
        r = e51.score(values); out[title] = r
        print(f"{title:<66} " + ("no whole coin yet" if r is None else
              f"{r['score']:.3f} ({r['low']:.3f} to {r['high']:.3f}); {r['coins']} coins, {r['won']} above a half, {r['lost']} below"))

    print({"share": "SHARE OF THE AGENTS (what the thesis reads)", "tokens": "SHARE OF THE TOKENS (beside it)"}[what])
    print("the share held by the brains of the world of equals")
    home, away = [], []
    for seed in (5301, 5302):
        eq, churn = runs[("10 numbers", seed)], runs[("the engine's mutation", seed)]
        mine = [c for c in played if c["set"] == f"equals against the engine's world, seed {seed}"]
        at_home = [c["shares"][(eq, END)] for c in mine if c["world"] == eq]
        abroad = [c["shares"][(eq, END)] for c in mine if c["world"] == churn]
        show(f"    seed {seed}, on the world of equals", at_home); show(f"    seed {seed}, on the engine's world", abroad)
        home += at_home; away += abroad
    show("  both seeds, on the worlds of equals", home)
    show("  both seeds, on the engine's worlds", away)
    e1, e2 = runs[("10 numbers", 5301)], runs[("10 numbers", 5302)]
    two = [c for c in played if c["set"] == "two worlds of equals"]
    show("the two worlds of equals: the share of seed 5301's brains, at home", [c["shares"][(e1, END)] for c in two if c["world"] == e1])
    show("the two worlds of equals: the share of seed 5301's brains, away", [c["shares"][(e1, END)] for c in two if c["world"] == e2])
    json.dump(out, open(os.path.join(HERE, "e59.json" if what == "share" else "e59_tokens.json"), "w"), indent=1)
