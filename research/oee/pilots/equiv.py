import sys
sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife")
from gol_config import SimConfig
from GraphOfLifeSimple import new_world
from proto import make_world
cfg = SimConfig.for_new_run(seed=7, total_tokens=3000, hidden_layers=[16])
a = new_world(cfg); b = make_world(SimConfig.for_new_run(seed=7, total_tokens=3000, hidden_layers=[16]), {"mor": "off"})
for t in range(25):
    fa = a.step(record_decisions=False); fb = b.step(record_decisions=False)
    for x, y in zip(fa, fb):
        assert x["ids"] == y["ids"] and x["tokens"] == y["tokens"] and x["brain_ids"] == y["brain_ids"] and x["edges"] == y["edges"], t
print("identical for 25 iterations; nodes", len(fa[1]["ids"]), "counters", b.c)

# The engine mode: the engine's own phases with bookkeeping only, under B2's rules
# and the switches the pilots of the settings screen turn.
B2 = dict(hidden_layers=[16], message_amount=8, message_prepass=False, allow_handover=False,
          mutate_on_replication=True, incumbency=1.5, birth_links=2, mend_on_death=True,
          prune_needs=2, valence=8, keep_home=0.3, max_age=100, min_child=10)
for extra in ({}, {"allow_revolutions": False}, {"allow_gifting": True, "prune_after": "reproduction",
              "inactive_window": "iteration", "prune_needs": 0}, {"redistribution": "by_tokens"},
              {"redistribution": "poorest"}, {"redistribution": "newborns"}, {"redistribution": "parents"}):
    s = {**B2, **extra, "seed": 11, "total_tokens": 3000}
    a = new_world(SimConfig.for_new_run(**s))
    b = make_world(SimConfig.for_new_run(**s), {"engine": True})
    for t in range(25):
        fa = a.step(record_decisions=False); fb = b.step(record_decisions=False)
        for x, y in zip(fa, fb):
            assert x["ids"] == y["ids"] and x["tokens"] == y["tokens"] and x["brain_ids"] == y["brain_ids"] and x["edges"] == y["edges"], (extra, t)
    print("engine mode identical for 25 iterations with", extra or "B2", "; nodes", len(fa[1]["ids"]), "counters", b.c)

# Brains mixed at birth: the engine's mix_at_birth, and the prototype's switch for it (mix),
# which every pilot made with mixing names.
s = {**B2, "seed": 11, "total_tokens": 3000}
a = new_world(SimConfig.for_new_run(**s, mix_at_birth=True))
b = make_world(SimConfig.for_new_run(**s), {"engine": True, "mix": True})
for t in range(25):
    fa = a.step(record_decisions=False); fb = b.step(record_decisions=False)
    for x, y in zip(fa, fb):
        assert x["ids"] == y["ids"] and x["tokens"] == y["tokens"] and x["brain_ids"] == y["brain_ids"] and x["edges"] == y["edges"], ("mix", t)
print("engine mode identical for 25 iterations with mixing at birth; nodes", len(fa[1]["ids"]), "counters", b.c)
# Mates among the parents: the engine's mix_mates, and the prototype's switch for it (mates).
a = new_world(SimConfig.for_new_run(**s, mix_at_birth=True, mix_mates="parents"))
b = make_world(SimConfig.for_new_run(**s), {"engine": True, "mix": True, "mates": "parents"})
for t in range(25):
    fa = a.step(record_decisions=False); fb = b.step(record_decisions=False)
    for x, y in zip(fa, fb):
        assert x["ids"] == y["ids"] and x["tokens"] == y["tokens"] and x["brain_ids"] == y["brain_ids"] and x["edges"] == y["edges"], ("mates", t)
print("engine mode identical for 25 iterations with mates among the parents; nodes", len(fa[1]["ids"]), "counters", b.c)
