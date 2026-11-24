"""Reads Experiment 51's contests as Experiment 50's thesis says: a coin is two contests with one deal, the two
sides in swapped places, and scores the mean of the two; a score is the mean of its coins, its band two
standard errors of that mean either side. Plays nothing."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("GOL_RUNS_DIR", "/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns")
sys.path.insert(0, ROOT)
import gol_plan, gol_store

HERE = os.path.dirname(os.path.abspath(__file__))
AT, END = 20000, 22500


def side(trial, place):
    """Who is in a contest's place 'a' or 'b': (run, era)."""
    return (trial.get(place + "_run", trial["run"]), trial[place])


def coins(name="E51", what="share"):
    """Every whole coin: its set, the world it was played on, and the share of each side over its two contests.
    `what` is "share", the share of the agents, which the thesis reads, or "tokens", the share of the tokens
    agents hold, reported beside it: a side that keeps few rich agents has few agents and many tokens."""
    played = {}
    for found in gol_plan.trials_of(name):
        trial = found["trial"]
        try:
            with open(os.path.join(gol_store.run_dir(gol_plan.trial_id_for(trial)), "result.json")) as f:
                share = json.load(f)["result"][what]
        except (OSError, ValueError):
            continue
        played.setdefault((found["set"], trial["run"], trial["home"], trial["seed"]), []).append((trial, share))
    out = []
    for (label, world, home, seed), pair in played.items():
        if len(pair) != 2 or any(share is None for _, share in pair):
            continue
        shares = {}
        for trial, share in pair:
            shares.setdefault(side(trial, "b"), []).append(share)
            shares.setdefault(side(trial, "a"), []).append(1.0 - share)
        out.append({"set": label, "world": world, "home": home, "seed": seed,
                    "shares": {k: float(np.mean(v)) for k, v in shares.items()}})
    return out


def score(values):
    v = np.array(values, dtype=float)
    if v.size == 0:
        return None
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
    return {"coins": int(v.size), "score": float(v.mean()), "low": float(v.mean() - 2 * se), "high": float(v.mean() + 2 * se),
            "won": int((v > 0.5).sum()), "lost": int((v < 0.5).sum())}


if __name__ == "__main__":
    runs = {(s.condition, s.lab["seed"]): s.run_id for s in gol_plan.experiment_runs("E50")}
    arm_of = {run: arm for (arm, seed), run in runs.items()}
    what = sys.argv[1] if len(sys.argv) > 1 else "share"
    all_coins = coins(what=what)
    out = {}
    print({"share": "SHARE OF THE AGENTS (what the thesis reads)", "tokens": "SHARE OF THE TOKENS agents hold (beside it)"}[what])

    def show(title, values):
        r = score(values); out[title] = r
        print(f"{title:<62} " + ("no whole coin yet" if r is None else
              f"{r['score']:.3f} ({r['low']:.3f} to {r['high']:.3f}); {r['coins']} coins, {r['won']} above a half, {r['lost']} below"))

    print("(1) the end against the start: the share of the brains of 22,500 against those of 20,000")
    for arm in ("as it is", "10 numbers", "100 numbers"):
        pooled = []
        for seed in (5001, 5002):
            run = runs[(arm, seed)]
            mine = [c["shares"][(run, END)] for c in all_coins if c["set"].startswith(f"{arm}, seed {seed}:")]
            show(f"    {arm}, seed {seed}", mine); pooled += mine
        show(f"  {arm}, both runs", pooled)
    print("(2) the small arms against the arm as it is, both at 22,500: the small arm's share")
    for arm in ("10 numbers", "100 numbers"):
        pooled = []
        for seed in (5001, 5002):
            run = runs[(arm, seed)]
            mine = [c["shares"][(run, END)] for c in all_coins if c["set"].startswith(f"{arm} against as it is, seed {seed}")]
            for where in ("its own world", "the other's world"):
                show(f"      {arm}, seed {seed}, on {where}",
                     [c["shares"][(run, END)] for c in all_coins if c["set"].startswith(f"{arm} against as it is, seed {seed}")
                      and (c["world"] == run) == (where == "its own world")])
            show(f"    {arm}, seed {seed}", mine); pooled += mine
        show(f"  {arm}, both seeds", pooled)
    print("the world before it was played on: the share of the brains of 20,000")
    source = "B1-40000-bbbccd-s4602"
    for before in (17500, 15000):
        show(f"  against {before:,}", [c["shares"][(source, AT)] for c in all_coins if c["set"].endswith(f"against {before:,}")])
    json.dump(out, open(os.path.join(HERE, "e51.json" if what == "share" else "e51_tokens.json"), "w"), indent=1)
