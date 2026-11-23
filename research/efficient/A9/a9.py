"""Card A9 (added 23 November 2026): the wave-1 readings (B1, B2, A1) on round 3's long-lived worlds.
Round 3 kept a world every 250 iterations, so A1 here also says how fast behaviour is reshuffled. Plays nothing."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("", "B1", "B2", "A1"):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), d))
import probe as P
import b1, b2, a1
import gol_store

WORLDS = {"s3004": "B1-30000-7d5208-s3004", "s3043": "B1-30000-02ed17-s3043", "s3065": "B1-30000-ba3afe-s3065",
          "s3019": "B1-30000-b1209a-s3019", "s3080": "B1-30000-625bc7-s3080", "s3058": "B1-30000-4108dd-s3058"}
b1.OPERATORS = ["engine", 1, 10, 100]

if __name__ == "__main__":
    out = {}
    for name, run in WORLDS.items():
        try:
            cfg = P.config(run)
            eras = gol_store.archived(run)
            rules = {k: getattr(cfg, k) for k in ("contest", "redistribution", "mend_on_death", "mutate_on_replication", "mix_at_birth",
                                                   "mutation_probability", "mutation_noise_std", "mutation_sparsity", "keep_home", "allocation")}
            m = b1.card(name, run, eras[-1])
            p = b2.card(name, run, [eras[0], eras[len(eras) // 2], eras[-1]])
            w = a1.card(name, run)
        except Exception as exc:                       # the drive may go away: say so and go on
            print(f"{name}: not read ({type(exc).__name__}: {exc})", flush=True); continue
        out[name] = {"rules": rules, "b1": m, "b2": p, "a1": w}
        last = p["eras"][-1]
        print(f"\n{name}: {rules}")
        print(f"  B1 at {eras[-1]}: {m['agents']} agents, {m['tokens_per_agent']:.1f} tokens each; strangers differ on {m['strangers_differ']['any']:.0%}; "
              + "; ".join(f"{op}: {o['probes_decided_differently']['any']:.1%} ({o['relative_to_strangers_median']:.2f} of strangers)" for op, o in m["operators"].items()))
        print(f"  B2 at {last['era']}: {last['genotypes_among_them']} genotypes in {last['brains']} brains, largest clone {last['largest_clone_share']:.0%}; "
              f"same as most common: first {last['agree_with_most_common_first']:.0%}, child {last['agree_with_most_common_child']:.0%}; home first {last['first_token_home']:.0%}, has a child {last['has_child']:.0%}")
        print(f"  A1: floor first {w['floor']['first']:.3f}, home {w['floor']['home']:.3f}; by lag: "
              + "  ".join(f"{t['lag']}: {t['first']:.2f}/{t['home']:.2f}" for t in w["by_lag"]) + f"; slopes {dict((k, round(v, 2)) for k, v in w['slopes'].items())}", flush=True)
        json.dump(out, open(os.path.join(HERE, "a9.json"), "w"), indent=1, default=str)
