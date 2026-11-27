"""Experiment 68: the two worlds that died, played on from inside the takeover under each rule.
Reads the sequels the lab played; plays nothing.   python3 research/efficient/E50/e68.py

    python3 research/efficient/E50/e68.py lines     the line's brains at each world played on from,
                                                    read from the frames of the first play (for the plan)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e64
import autopsy_lines
import gol_plan, gol_store

#: The worlds played on from: the run of Experiment 63, the frames its line is traced in, and the iterations.
WORLDS = {"30 numbers": ("B1-40000-8020c3-s6302-on210139", 29100, 29350, (29280, 29300, 29320)),
          "50 numbers": ("B1-40000-3f4b02-s6302-on210139", 23150, 23620, (23400, 23470, 23520))}


def lines():
    """The line's brains at each world, and what the line holds there."""
    out = {}
    for name, (run, first, last, eras) in WORLDS.items():
        origin, born, inline, before, after, start = autopsy_lines.lineage(run, first, last)
        for era in eras:
            f = after[era - 1]
            tokens, line = np.array(f["tokens"]), np.array([inline(b) for b in f["brain_ids"]])
            out[f"{name} at {era}"] = {"run": run, "era": era, "agents": len(tokens), "line_agents": int(line.sum()),
                                      "line_share_of_tokens": round(float(tokens[line].sum() / tokens.sum()), 3),
                                      "line": sorted({b for b in f["brain_ids"] if inline(b)})}
    return out


def read(name="E68"):
    trials = []
    for spec in gol_plan.experiment_runs(name):
        try:
            with open(os.path.join(gol_store.run_dir(spec.run_id), "result.json")) as f:
                result = json.load(f)
        except (OSError, ValueError):
            continue
        rule, world = spec.condition.split(": ")
        trials.append({"set": spec.condition, "rule": rule, "world": world, "place": spec.trial["seed"],
                       "series": result["result"]["series"], "seconds": result["seconds"]})
    return trials


if __name__ == "__main__":
    if sys.argv[1:] == ["lines"]:
        for name, row in lines().items():
            print(f"{name}: {row['agents']} agents, {row['line_agents']} of the line with {row['line_share_of_tokens']:.0%} "
                  f"of the tokens, {len(row['line'])} brains")
        sys.exit()
    trials = read()
    out = {}
    for rule in dict.fromkeys(t["rule"] for t in trials):
        mine = [t for t in trials if t["rule"] == rule]
        rows = out[rule] = [{**e64.reading(t), "world": t["world"], "richest_most": max(p["richest"] for p in t["series"]),
                             "line_end": t["series"][-1]["mine"], "line_tokens_end": t["series"][-1]["tokens"]}
                            for t in mine]
        print(f"\n{rule}: {len(rows)} trials; worlds end {sum(r['world_ends'] for r in rows)} "
              f"(died out {sum(r['died_out_at'] is not None for r in rows)}); lines sweep "
              f"{sum(r['sweeps_at'] is not None for r in rows)}; lines gone {sum(r['line_gone_at'] is not None for r in rows)}; "
              f"minutes {sum(t['seconds'] for t in mine) / 60:.0f}")
        for world in dict.fromkeys(r["world"] for r in rows):
            part = sorted((r for r in rows if r["world"] == world), key=lambda r: r["place"])
            show = lambda key: [r[key] for r in part]
            print(f"  {world}: line at the start {part[0]['at']['0'][1]} agents, {part[0]['start_tokens']} tokens; "
                  f"sweeps at {show('sweeps_at')}; line gone at {show('line_gone_at')}; most copies {show('most_copies')}; "
                  f"most tokens {show('most_tokens')}; line at the end {show('line_end')} agents, {show('line_tokens_end')} tokens; "
                  f"agents {part[0]['agents_start']} -> {show('agents_end')}; halved at {show('halved_at')}; "
                  f"played {show('played')}; richest most {show('richest_most')}; ends {sum(show('world_ends'))}")
    with open(os.path.join(e64.HERE, "e68.json"), "w") as f:
        json.dump(out, f, indent=1)
