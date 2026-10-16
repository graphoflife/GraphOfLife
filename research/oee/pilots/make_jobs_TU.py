"""
Batches T, U and V (2026-10-16): gifts in combination, and the switches in combination.

The author asked for gifts combined with every other mechanism, and for the
existing settings tested in combination. Every world is a pilot in the engine's
own code (--proto engine=true) at 10,000 tokens for 1,000 iterations, a whole
world every 100, the health rows every 50 — batch S's form, so that batch S's
worlds, the later batches' and the lab's are the controls, seed for seed.

T  gifts (allow_gifting, prune_after = reproduction, inactive_window = iteration,
   exactly batch S's S2-gifts) together with one other switch, seeds 1 and 2.
   Each world is half of a two-by-two: B2 (S0-B2 / P/B2), B2 + gifts (S2-gifts),
   B2 + the switch (the control named below, same seeds), B2 + both (here). The
   one switch without a control at 10,000 tokens, handover, gets one (T/handover).

U  a two-level fractional factorial over eight switches on B2, sixteen worlds
   (seed 1): A gifts, B no revolutions, C every unused link lapses (prune_needs 0),
   D no random inputs, E estates to the newborns, F links hold a token (valence 0),
   G brains mixed at birth, H handover. A-D take every combination; E = BCD,
   F = ACD, G = ABC, H = ABD (a design of resolution IV: every switch's effect is
   free of every interaction of two others).

V  the same design on B1, where revolutions matter (they take most of the nodes
   that change hands): A gifts, B no revolutions, C no handover (B1 allows it),
   D no random inputs, E estates to the newborns, F brains change only when
   copied (B2's rule), G brains mixed at birth, H links hold a token.

The order of the queue (jobs_TUV.txt): every T combination with seed 1, then
U, then V, then T's seed 2 — a first look at everything before the second seeds.

    python3 make_jobs_TU.py         writes jobs_T.txt, jobs_U.txt, jobs_V.txt and jobs_TUV.txt
"""
import itertools
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "../../../GraphOfLifeRuns/.pilots"
RUN = "python3 run.py {out} --iters 1000 --ckpt 100 --health 50 --set {sets} --proto engine=true{proto}"
B2 = ("total_tokens=10000 seed={seed} hidden_layers=16 message_amount=8 message_prepass=false allow_handover=false "
      "mutate_on_replication=true incumbency=1.5 birth_links=2 mend_on_death=true prune_needs=2 valence=8 "
      "keep_home=0.3 max_age=100 min_child=10")
B1 = "total_tokens=10000 seed={seed}"
GIFTS = "allow_gifting=true prune_after=reproduction inactive_window=iteration"

#: name, base, extra settings, extra proto keys, the control of the same seeds (the switch without gifts).
T = [
    ("norev", B2, "allow_revolutions=false", "", "S/S1-norev"),
    ("bytokens", B2, "redistribution=by_tokens", "", "S/S5-bytokens"),
    ("poorest", B2, "redistribution=poorest", "", "R/B2-poorest"),
    ("newborns", B2, "redistribution=newborns", "", "R/B2-newborns"),
    ("parents", B2, "redistribution=parents", "", "R/B2-parents"),
    ("linktoken", B2, "valence=0", "link_token=true", "P/linktoken"),
    ("mix", B2, "", "mix=true", "Q/mix"),
    ("helped", B2, "", "helped_by=true", "P/helped"),
    ("came", B2, "", "came_from=true", "P/came"),
    ("handover", B2, "allow_handover=true", "", "T/handover"),
    ("nodice", B2, "random_input_amount=0", "", "lab B2-10000-bad270"),
    ("noage", B2, "max_age=0", "", "lab B2-10000-3ec24f"),
    ("anychild", B2, "min_child=1", "", "lab B2-10000-185eb7"),
    ("keephalf", B2, "keep_home=0.5", "", "lab B2-10000-df2615"),
    ("keepone", B2, "keep_home=0.01", "", "lab B2-10000-054b50"),
    ("lottery", B2, "allow_revolutions=false", "contest=lottery", "S/G1-lottery"),
    ("pool", B2, "allow_revolutions=false", "pool=genotype", "S/G2-genotype"),
    ("grow", B2, "brain_kind=grow hidden_layers=32", "", "Q/grow"),
    ("unittoken", B2, "brain_kind=grow hidden_layers=32", "unit_token=true", "Q/unittoken"),
    ("mut05", B2, "mutation_probability=0.05", "", "lab B2-10000-bf74a9"),
    ("mut80", B2, "mutation_probability=0.8", "", "lab B2-10000-fe3009"),
    ("B1", B1, "", "", "S/S6-B1"),
    ("B1norev", B1, "allow_revolutions=false", "", "S/S7-B1norev"),
]
#: Worlds without gifts that no batch has made at 10,000 tokens: the controls they need.
T_CONTROLS = [("handover", B2, "allow_handover=true", "")]

#: The eight switches of U, as (settings, proto keys) when on.
U_SWITCHES = {
    "A": (GIFTS, ""),
    "B": ("allow_revolutions=false", ""),
    "C": ("prune_needs=0", ""),
    "D": ("random_input_amount=0", ""),
    "E": ("redistribution=newborns", ""),
    "F": ("valence=0", "link_token=true"),
    "G": ("", "mix=true"),
    "H": ("allow_handover=true", ""),
}


#: The eight switches of V, on B1.
V_SWITCHES = {
    "A": (GIFTS, ""),
    "B": ("allow_revolutions=false", ""),
    "C": ("allow_handover=false", ""),
    "D": ("random_input_amount=0", ""),
    "E": ("redistribution=newborns", ""),
    "F": ("mutate_on_replication=true", ""),
    "G": ("", "mix=true"),
    "H": ("", "link_token=true"),
}


def job(out: str, base: str, sets: str, proto: str, seed: int) -> str:
    settings = " ".join(x for x in (base.format(seed=seed), sets) if x)
    return RUN.format(out=out, sets=settings, proto=(" " + proto) if proto else "")


def u_design():
    """The sixteen rows of the 2^(8-4) design: A-D in every combination, E-H their products."""
    rows = []
    for a, b, c, d in itertools.product((-1, 1), repeat=4):
        rows.append({"A": a, "B": b, "C": c, "D": d, "E": b * c * d, "F": a * c * d, "G": a * b * c, "H": a * b * d})
    return rows


def main():
    lines = ["# Batch T: gifts together with one other switch (make_jobs_TU.py). Controls first, then every combination, seed 1 then 2."]
    for name, base, sets, proto in T_CONTROLS:
        for seed in (1, 2):
            lines.append(job(f"{OUT}/T/{name}-s{seed}", base, sets, proto, seed))
    for name, base, sets, proto, _ in T:
        for seed in (1, 2):
            lines.append(job(f"{OUT}/T/gifts-{name}-s{seed}", base, " ".join(x for x in (GIFTS, sets) if x), proto, seed))
    open(os.path.join(HERE, "jobs_T.txt"), "w").write("\n".join(lines) + "\n")

    screens = {}
    for batch, base, switches in (("U", B2, U_SWITCHES), ("V", B1, V_SWITCHES)):
        lines = [f"# Batch {batch}: eight switches in a 2^(8-4) design of resolution IV (make_jobs_TU.py), seed 1."]
        for row in u_design():
            on = [k for k in "ABCDEFGH" if row[k] > 0]
            sets = " ".join(switches[k][0] for k in on if switches[k][0])
            proto = " ".join(switches[k][1] for k in on if switches[k][1])
            name = f"{batch}-" + ("".join(on) or "none")
            lines.append(job(f"{OUT}/{batch}/{name}-s1", base, sets, proto, 1))
        open(os.path.join(HERE, f"jobs_{batch}.txt"), "w").write("\n".join(lines) + "\n")
        screens[batch] = lines[1:]
    t = [l for l in open(os.path.join(HERE, "jobs_T.txt")).read().splitlines() if l.startswith("python3")]
    queue = [l for l in t if "-s1 " in l] + screens["U"] + screens["V"] + [l for l in t if "-s2 " in l]
    open(os.path.join(HERE, "jobs_TUV.txt"), "w").write("# Batches T, U, V in queue order (make_jobs_TU.py).\n"
                                                        + "\n".join(queue) + "\n")
    print(f"jobs_T.txt: {len(t)} worlds; jobs_U.txt, jobs_V.txt: 16 each; jobs_TUV.txt: {len(queue)}")


if __name__ == "__main__":
    main()
