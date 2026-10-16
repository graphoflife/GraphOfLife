"""
Batches T, U, V, W, X, Y and K (2026-10-16): gifts in combination, the switches
in combination, revolutions, handover and brains mixed at birth in combination,
and links to strangers.

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

W  revolutions in combination on B2 (asked for the same day): B2 without
   revolutions together with one other switch, seeds 1 and 2, each world again
   the fourth corner of a two-by-two whose other three exist (B2, S1-norev, and
   the switch's control). The other contests of batch S (lottery, sides pooled
   by genotype, kin sense, family sense, family pools) were only ever run
   without revolutions; for them the missing corner is the same rule *with*
   revolutions, and that is what W makes. Gifts without revolutions are in T.

X  handover in combination on B2 (asked for the same day): B2 with handover (a
   parent may move some of its links to its newborn; B1's rule, off in B2)
   together with each switch of W and each of batch S's other contests (with
   revolutions, as W makes them), seeds 1 and 2: the fourth corner of a
   two-by-two whose other three are B2, T/handover and the switch's control.
   Gifts with handover are in T, handover without revolutions in W.

Y  brains mixed at birth in combination on B2 (asked for the same day, after
   Meta X found mixed worlds the closest to the author's picture): B2 with
   mix together with each switch of W except mixing itself, and each of batch
   S's other contests with revolutions, seeds 1 and 2: the fourth corner of a
   two-by-two whose other three are B2, Q/mix and the switch's control. Gifts
   with mixing are in T, mixing without revolutions in W, with handover in X.

K  links to strangers on B2 (asked for the same day, once the engine had them):
   every search the engine knows (jump, walk, reach) with every steering
   (random, looking) and every consent (mutual, forced), seeds 1 and 2. The
   control is B2 itself (P/B2, S/S0-B2: the lab's own B2 worlds of the same
   seeds), and each search with random steering and mutual consent is the
   control of the other three of its kind.

M  steadying mixed worlds (2026-10-16, after Experiment 17 found mixed worlds rich
   but unsteady — they swing between a few hundred and two thousand agents, and
   one died out): B2 with brains mixed at birth together with each rule of the
   engine that might keep a world's size — the estate of the dead to the
   newborns, to the parents or to the poorest, half kept at home, links to
   strangers by a free reach or by a scout — and a slower change of brains;
   seeds 1 and 2, 5,000 iterations. Their twins without the rule are Experiment
   17's worlds of the same seeds (B2-10000-4af4a3-s001, s002).

SW mixing switched at a moment (2026-10-16, after Experiment 17 found that a mixed
   world is rich because its agents put off having children — the agents that
   could have one have it about fourteen times less often than in B2): the five
   worlds of Experiment 17 that met every mark, taken from the lab at iteration
   5,000 and played on without mixing; and five of B2's own worlds (Experiment
   16), taken at iteration 2,000 and played on with mixing; 2,000 iterations, a
   whole world every 250. Their twins without the switch are the lab's own worlds
   from the same moment: Experiment 17's runs, and B2's continuations L/B2-long.

The order of the queue (jobs_TUV.txt): every T combination with seed 1, then
U, then V, then T's seed 2 — a first look at everything before the second seeds.
jobs_W.txt, jobs_X.txt and jobs_Y.txt: every combination with seed 1, then seed 2 (queues of their own).

    python3 make_jobs_TU.py         writes jobs_T.txt, jobs_U.txt, jobs_V.txt, jobs_TUV.txt, jobs_W.txt, jobs_X.txt,
                                    jobs_Y.txt, jobs_K.txt, jobs_M.txt and jobs_SW.txt
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
#: W: switches combined with B2 without revolutions — name, extra settings, extra proto keys, control (with revolutions).
W_NOREV = [
    ("bytokens", "redistribution=by_tokens", "", "S/S5-bytokens"),
    ("poorest", "redistribution=poorest", "", "R/B2-poorest"),
    ("newborns", "redistribution=newborns", "", "R/B2-newborns"),
    ("parents", "redistribution=parents", "", "R/B2-parents"),
    ("linktoken", "valence=0", "link_token=true", "P/linktoken"),
    ("mix", "", "mix=true", "Q/mix"),
    ("helped", "", "helped_by=true", "P/helped"),
    ("came", "", "came_from=true", "P/came"),
    ("handover", "allow_handover=true", "", "T/handover"),
    ("nodice", "random_input_amount=0", "", "lab B2-10000-bad270"),
    ("noage", "max_age=0", "", "lab B2-10000-3ec24f"),
    ("anychild", "min_child=1", "", "lab B2-10000-185eb7"),
    ("keephalf", "keep_home=0.5", "", "lab B2-10000-df2615"),
    ("keepone", "keep_home=0.01", "", "lab B2-10000-054b50"),
    ("grow", "brain_kind=grow hidden_layers=32", "", "Q/grow"),
    ("unittoken", "brain_kind=grow hidden_layers=32", "unit_token=true", "Q/unittoken"),
    ("mut05", "mutation_probability=0.05", "", "lab B2-10000-bf74a9"),
    ("mut80", "mutation_probability=0.8", "", "lab B2-10000-fe3009"),
    ("lapse", "prune_needs=0", "", "S/S4-lapse"),
]
#: W: batch S's other contests, run before only without revolutions, now with them — name, proto keys, control.
W_REV = [
    ("lottery", "contest=lottery", "S/G1-lottery"),
    ("pool", "pool=genotype", "S/G2-genotype"),
    ("kinsense", "kin_sense=true", "S/G3-kinsense"),
    ("familysense", "tag_rate=0.05 kin_sense=true", "S/G4-familysense"),
    ("familypool", "tag_rate=0.05 pool=family", "S/G5-familypool"),
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

    lines = ["# Batch W: revolutions in combination on B2 (make_jobs_TU.py), seed 1 then 2."]
    for seed in (1, 2):
        for name, sets, proto, _ in W_NOREV:
            lines.append(job(f"{OUT}/W/norev-{name}-s{seed}", B2, " ".join(x for x in ("allow_revolutions=false", sets) if x),
                             proto, seed))
        for name, proto, _ in W_REV:
            lines.append(job(f"{OUT}/W/rev-{name}-s{seed}", B2, "", proto, seed))
    open(os.path.join(HERE, "jobs_W.txt"), "w").write("\n".join(lines) + "\n")
    print(f"jobs_W.txt: {len(lines) - 1} worlds")

    lines = ["# Batch X: handover in combination on B2 (make_jobs_TU.py), seed 1 then 2."]
    for seed in (1, 2):
        for name, sets, proto, _ in W_NOREV:
            if name == "handover":
                continue
            lines.append(job(f"{OUT}/X/handover-{name}-s{seed}", B2,
                             " ".join(x for x in ("allow_handover=true", sets) if x), proto, seed))
        for name, proto, _ in W_REV:
            lines.append(job(f"{OUT}/X/handover-{name}-s{seed}", B2, "allow_handover=true", proto, seed))
    open(os.path.join(HERE, "jobs_X.txt"), "w").write("\n".join(lines) + "\n")
    print(f"jobs_X.txt: {len(lines) - 1} worlds")

    lines = ["# Batch Y: brains mixed at birth in combination on B2 (make_jobs_TU.py), seed 1 then 2."]
    for seed in (1, 2):
        for name, sets, proto, _ in W_NOREV:
            if name == "mix":
                continue
            lines.append(job(f"{OUT}/Y/mix-{name}-s{seed}", B2, sets, " ".join(x for x in ("mix=true", proto) if x), seed))
        for name, proto, _ in W_REV:
            lines.append(job(f"{OUT}/Y/mix-{name}-s{seed}", B2, "", "mix=true " + proto, seed))
    open(os.path.join(HERE, "jobs_Y.txt"), "w").write("\n".join(lines) + "\n")
    print(f"jobs_Y.txt: {len(lines) - 1} worlds")

    lines = ["# Batch K: links to strangers on B2 (make_jobs_TU.py), every search, steering and consent, seed 1 then 2."]
    for seed in (1, 2):
        for search, steering, consent in itertools.product(("jump", "walk", "reach"), ("random", "looking"),
                                                           ("mutual", "forced")):
            sets = f"link_search={search} link_steering={steering} link_consent={consent}"
            lines.append(job(f"{OUT}/K/{search}-{steering}-{consent}-s{seed}", B2, sets, "", seed))
    open(os.path.join(HERE, "jobs_K.txt"), "w").write("\n".join(lines) + "\n")
    print(f"jobs_K.txt: {len(lines) - 1} worlds")

    lines = ["# Batch M: steadying mixed worlds on B2 (make_jobs_TU.py), seeds 1 and 2, 5,000 iterations."]
    steadying = [("newborns", "redistribution=newborns"), ("parents", "redistribution=parents"),
                 ("poorest", "redistribution=poorest"), ("keephalf", "keep_home=0.5"),
                 ("reach", "link_search=reach link_consent=forced"), ("walk", "link_search=walk"),
                 ("mut05", "mutation_probability=0.05")]
    for seed in (1, 2):
        for name, sets in steadying:
            lines.append(job(f"{OUT}/M/mix-{name}-s{seed}", B2, f"mix_at_birth=true {sets}", "", seed)
                         .replace("--iters 1000", "--iters 5000").replace("--ckpt 100", "--ckpt 500"))
    open(os.path.join(HERE, "jobs_M.txt"), "w").write("\n".join(lines) + "\n")
    print(f"jobs_M.txt: {len(lines) - 1} worlds")

    lines = ["# Batch SW: mixing switched off in five worlds of Experiment 17 at 5,000, on in five of B2's at 2,000 "
             "(make_jobs_TU.py)."]
    switches = ([(f"B2-10000-4af4a3-s{s:03d}", 5000, "false", f"mixoff-s{s:03d}") for s in (2, 4, 5, 6, 10)]
                + [(f"B2-10000-s{s:03d}", 2000, "true", f"mixon-s{s:03d}") for s in (1, 2, 3, 4, 5)])
    for run, at, mix, name in switches:
        lines.append(f"python3 run.py {OUT}/SW/{name} --from {run}:{at} --copy-eras {at} --iters 2000 --ckpt 250 "
                     f"--health 50 --set mix_at_birth={mix} --proto engine=true")
    open(os.path.join(HERE, "jobs_SW.txt"), "w").write("\n".join(lines) + "\n")
    print(f"jobs_SW.txt: {len(lines) - 1} worlds")


if __name__ == "__main__":
    main()
