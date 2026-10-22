#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The screen of random worlds: every world's settings drawn at random (the
author's plan of 2026-10-22).

A screen is a plan of its own kind (gol_plan.py expands it): rounds of worlds,
each world's settings drawn here, every one played by the lab like any other
run and judged for health afterwards (research/screen/judge.py). Values that
make every world unhealthy are forced off in the next round's plan.

How a setting is drawn:

  switch    on in half the worlds (the rules' on/off switches, and the choices
            between two options)
  choice    each of its options equally often
  number    its default in three worlds of four, otherwise one of a few levels
  zero      0 in one world of ten, otherwise one of a few levels (the size of
            a message and the number of noise inputs)

A setting that can matter only beside another is drawn only there and keeps its
default elsewhere: buying life only with old age, the mate's kind only with
mixing, a search's steering and consent only with a search, messages before the
game and the pooled self-message only with messages. Two draws never make the
same world under two names.

Every value is drawn from a hash of the sampler's version, the round's seed, the
world and the setting, not from one random stream: adding worlds to a round, or
drawing a refused combination again, changes no other world. A draw the engine
refuses is drawn again with the next attempt number. The settings come back as
what differs from B1, whose rules are every mechanic's default, as plain Python
values of B1's own types.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gol_config import SimConfig

#: What a screen plan may say.
SCREEN_PLAN = ("sampler", "seed", "worlds", "tokens", "iterations", "forced", "budget_hours", "record",
               "references", "workers")
#: Each round's worlds have seeds 1000·seed + i; its reference worlds 1000·seed + 900 + j.
SEED_STRIDE = 1000
REFERENCE_OFFSET = 900
MAX_WORLDS = REFERENCE_OFFSET - 1
ATTEMPTS = 64

# A setting is (name, kind, values, needs): for a switch, (off, on); for a choice, its options;
# for a number, (default, *levels); for a zero, its levels besides 0. `needs` names what must hold
# of the world's other settings for it to be drawn at all.
Setting = Tuple[str, str, Tuple[Any, ...], Optional[Tuple[str, Any]]]

SPACES: Dict[int, List[Setting]] = {
    1: [
        # What a brain hears and is, first: messages decide whether two settings below apply.
        ("message_amount", "zero", (4, 8, 16, 30), None),
        ("random_input_amount", "zero", (2, 5, 10), None),
        ("message_prepass", "switch", (False, True), ("messages", True)),
        ("self_message", "switch", ("own", "largest"), ("messages", True)),
        ("hidden_layers", "number", ([16], [8], [32], [16, 16]), None),
        ("mutation_probability", "number", (0.2, 0.1, 0.4, 0.8), None),
        ("mutation_noise_std", "number", (0.2, 0.1, 0.4), None),
        ("mutation_sparsity", "number", (0.1, 0.05, 0.3), None),
        # The rules' switches.
        ("allow_revolutions", "switch", (False, True), None),
        ("allow_handover", "switch", (False, True), None),
        ("allow_gifting", "switch", (False, True), None),
        ("mutate_on_replication", "switch", (False, True), None),
        ("mend_on_death", "switch", (False, True), None),
        ("mix_at_birth", "switch", (False, True), None),
        ("mix_mates", "switch", ("any", "parents"), ("mix_at_birth", True)),
        ("child_floor", "switch", ("fixed", "neighbours"), None),
        ("inactive_window", "switch", ("phase", "iteration"), None),
        # The rules' choices.
        ("redistribution", "choice", ("uniform", "by_tokens", "poorest", "newborns", "parents", "children"), None),
        ("prune_after", "choice", ("blotto", "reproduction", "both"), None),
        ("link_search", "choice", ("off", "jump", "walk", "reach"), None),
        ("link_steering", "switch", ("random", "looking"), ("link_search", True)),
        ("link_consent", "switch", ("mutual", "forced"), ("link_search", True)),
        # The rules' numbers.
        ("incumbency", "number", (1.0, 1.25, 1.5, 2.0), None),
        ("keep_home", "number", (0.0, 0.1, 0.3, 0.5), None),
        ("max_age", "number", (0, 50, 100, 200), None),
        ("buy_life", "switch", (False, True), ("max_age", True)),
        ("min_child", "number", (1, 5, 10, 20), None),
        ("valence", "number", (0, 4, 8, 16), None),
        ("birth_links", "number", (0, 1, 2, 4), None),
        ("prune_needs", "number", (0, 1, 2, 3), None),
    ],
}


def space(version: int) -> List[Setting]:
    if version not in SPACES:
        raise ValueError(f"no sampler {version!r}: there are {', '.join(map(str, SPACES))}")
    return SPACES[version]


def values_of(setting: Setting) -> Tuple[Any, ...]:
    """Every value a setting can take, its default first."""
    name, kind, values, _ = setting
    return (0,) + tuple(values) if kind == "zero" else tuple(values)


def default_of(setting: Setting) -> Any:
    return values_of(setting)[0]


def _uniform(version: int, seed: int, world: int, attempt: int, key: str) -> float:
    digest = hashlib.sha256(f"{version}/{seed}/{world}/{attempt}/{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def _pick(setting: Setting, u: float, barred: Sequence[Any]) -> Any:
    """The value a uniform draw u gives a setting, among the values not barred."""
    name, kind, values, _ = setting
    every = values_of(setting)
    allowed = [v for v in every if v not in barred]
    if not allowed:
        raise ValueError(f"every value of {name} is forced off: a setting keeps at least one")
    if kind in ("switch", "choice") or allowed[0] != every[0]:
        # Options, and a number whose default is forced off, are drawn evenly from what is left.
        return allowed[min(int(u * len(allowed)), len(allowed) - 1)]
    first = 0.75 if kind == "number" else 0.1          # the default's share: 3 in 4, or 0 in 1 of 10
    if u < first or len(allowed) == 1:
        return allowed[0]
    rest = allowed[1:]
    return rest[min(int((u - first) / (1 - first) * len(rest)), len(rest) - 1)]


def _applies(needs: Optional[Tuple[str, Any]], drawn: Dict[str, Any]) -> bool:
    if needs is None:
        return True
    other, _ = needs
    if other == "messages":
        return drawn["message_amount"] > 0
    if other == "max_age":
        return drawn["max_age"] > 0
    if other == "link_search":
        return drawn["link_search"] != "off"
    return bool(drawn[other])


def draw(version: int, seed: int, world: int, forced: Optional[Dict[str, Sequence[Any]]] = None,
         base: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], int]:
    """
    One world's settings: what differs from `base` (B1's settings), and the attempt that gave them.
    `forced` names, for a setting, the values no longer drawn.
    """
    import gol_plan
    base = gol_plan.baseline("B1") if base is None else base
    forced = forced or {}
    names = {s[0] for s in space(version)}
    unknown = sorted(set(forced) - names)
    if unknown:
        raise ValueError(f"forced names {', '.join(unknown)}, which the sampler does not draw")
    for attempt in range(ATTEMPTS):
        drawn: Dict[str, Any] = {}
        for setting in space(version):
            name, kind, values, needs = setting
            if _applies(needs, drawn):
                drawn[name] = _pick(setting, _uniform(version, seed, world, attempt, name), forced.get(name, ()))
            else:
                drawn[name] = default_of(setting)
        if drawn["message_amount"] == 0:
            # No messages: nothing to say before the game, and a talk-only look would cost for nothing.
            drawn["message_prepass"] = False
        changed = {k: (list(v) if isinstance(v, list) else v) for k, v in drawn.items() if base.get(k) != v}
        try:
            SimConfig.from_dict({**base, **changed, "seed": 1}, stored=False)
        except ValueError:
            continue
        return changed, attempt
    raise ValueError(f"no world {world} of sampler {version}, seed {seed}, after {ATTEMPTS} attempts")


def check(screen: Dict[str, Any]) -> None:
    """Refuse a screen plan that says too little, or what no screen reads."""
    unknown = sorted(set(screen) - set(SCREEN_PLAN))
    missing = [k for k in ("sampler", "seed", "worlds", "tokens", "iterations") if k not in screen]
    if unknown or missing:
        raise ValueError("a screen says " + ", ".join(SCREEN_PLAN)
                         + (f"; {', '.join(missing)} missing" if missing else "")
                         + (f"; {', '.join(unknown)} is not something it reads" if unknown else ""))
    if not 1 <= int(screen["worlds"]) <= MAX_WORLDS:
        raise ValueError(f"a screen plays 1 to {MAX_WORLDS} worlds, not {screen['worlds']}")
    space(int(screen["sampler"]))
    for name, refs in (screen.get("references") or {}).items():
        if not 0 <= int(refs) <= SEED_STRIDE - REFERENCE_OFFSET - 1:
            raise ValueError(f"a screen plays 0 to {SEED_STRIDE - REFERENCE_OFFSET - 1} worlds of {name}")


def draws(screen: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every world of a screen: its number, seed, settings (what differs from B1) and attempt."""
    check(screen)
    version, seed = int(screen["sampler"]), int(screen["seed"])
    out = []
    for i in range(1, int(screen["worlds"]) + 1):
        changed, attempt = draw(version, seed, i, screen.get("forced"))
        out.append({"world": i, "seed": SEED_STRIDE * seed + i, "set": changed, "attempt": attempt})
    return out


def references(screen: Dict[str, Any]) -> List[Tuple[str, int]]:
    """The reference worlds a round plays beside its draws: (baseline, seed)."""
    seed = int(screen["seed"])
    return [(name, SEED_STRIDE * seed + REFERENCE_OFFSET + j)
            for name, n in sorted((screen.get("references") or {}).items()) for j in range(1, int(n) + 1)]


def full(changed: Dict[str, Any], version: int = 1, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A world's value of every drawn setting, B1's where the draw left it out."""
    import gol_plan
    base = gol_plan.baseline("B1") if base is None else base
    return {name: changed.get(name, base[name]) for name, *_ in space(version)}
