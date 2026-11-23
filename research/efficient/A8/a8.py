"""Card A8 (added 23 November 2026): how long do agents live, and how much do they hold, in round 3's 82 living worlds?
Read from research/screen/health.json alone (each world's averages over iterations 100 to 3,000). Plays nothing.

An agent's lifetime is taken as 1 / (births per agent and iteration): in a world whose size holds, as many die as are born.
"""
import json, os, statistics as st
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
h = json.load(open(os.path.join(ROOT, "research", "screen", "health.json")))
rank = {w["id"]: i + 1 for i, w in enumerate(json.load(open(os.path.join(ROOT, "research", "screen", "evolution.json")))["worlds"])}
rows = []
for w in h["worlds"]:
    if not (w.get("averages") and w["fate"] == "lived"):
        continue
    a, s = w["averages"], w["set"]
    rows.append({"world": "s" + str(w["seed"]) + (" " + s["allocation"].split("_")[0] if s.get("allocation", "full") != "full" else ""),
                 "id": w["id"], "lifetime": 1 / a["children"], "tokens_per_agent": 30000 / a["median_agents"],
                 "evolution_rank": rank.get(w["id"]), "health_rank": w["ranks"]["product"], "contest": s.get("contest", "largest"),
                 "estates": s.get("redistribution", "uniform"), "mend_on_death": bool(s.get("mend_on_death", False)),
                 "culled_tokens": a["culled_tokens"], "links": a["links"]})
rows.sort(key=lambda r: -r["lifetime"])
life = [r["lifetime"] for r in rows]; tok = [r["tokens_per_agent"] for r in rows]
print(f"{len(rows)} worlds. Lifetime in iterations: median {st.median(life):.1f}, least {min(life):.1f}, most {max(life):.1f}; "
      f"{sum(l >= 5 for l in life)} worlds of 5 or more, {sum(l >= 10 for l in life)} of 10 or more.")
print(f"Tokens an agent: median {st.median(tok):.1f}, least {min(tok):.1f}, most {max(tok):.1f}.")
for r in rows[:14]:
    print(f"  {r['world']:<16} lives {r['lifetime']:5.1f}  holds {r['tokens_per_agent']:5.1f}  evolution rank {r['evolution_rank']:>2}  health rank {r['health_rank']:>2}  "
          f"{r['contest']:<8} estates to {r['estates']:<9} mend {str(r['mend_on_death']):<5} culled {r['culled_tokens']:.3f}")
by = {}
for r in rows: by.setdefault(r["estates"], []).append(r["lifetime"])
print("median lifetime by where estates go:", {k: (round(st.median(v), 1), len(v)) for k, v in sorted(by.items())})
ra = np.argsort(np.argsort([r["evolution_rank"] for r in rows])); rb = np.argsort(np.argsort(life))
print("rank correlation of the evolution rank (1 = best) with lifetime:", round(float(np.corrcoef(ra, rb)[0, 1]), 2))
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "a8.json"), "w"), indent=1)
