#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for a GraphOfLife simulation run.

Every knob the web UI can set lives here in one dataclass. A run stores its own
SimConfig on disk, so a resumed run always continues under the exact settings it
started with, and old runs stay readable after the defaults change.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, asdict, field, fields
from typing import Any, ClassVar, Dict, List


@dataclass
class SimConfig:
    """All parameters of a single simulation run."""

    # What a run did before a setting existed.
    #
    # A stored configuration that predates one of these fields has to resolve
    # to the behaviour it actually had, not to whatever the default is today.
    # Handover did not exist, so an old run ran without it; revolutions were
    # unconditional, so an old run ran with them. Getting this wrong would not
    # merely mislabel a card — it would change the brain's shape and make the
    # run's own checkpoint unloadable.
    #
    # Applied only when reading something off disk. A configuration arriving
    # from outside is a request for a new run, and a key it left out means
    # "whatever the default is", not "whatever a run from last year did" —
    # asking the API for a world without naming the pre-pass used to quietly
    # get one without it, which is the opposite of the documented default.
    LEGACY_WHEN_ABSENT: ClassVar[Dict[str, Any]] = {
        # Every run recorded before there was a choice used the float brain.
        "brain_kind": "float",
        "allow_handover": False,
        "allow_revolutions": True,
        # On by default now, but every run recorded before the option existed
        # ran with one pass per phase. Reading those as having used a pre-pass
        # would change what a resumed run does.
        "message_prepass": False,
    }

    # What each kind of brain wants, so that choosing one does not also mean
    # knowing what else to change.
    #
    # A binary unit is on or off — one bit, where a float unit carries many —
    # so a binary brain is given wider layers to hold a comparable amount, on
    # top of the thermometer coding that already widens its input layer by
    # brain_bits. A weight costs a byte instead of eight, so the wider shape is
    # still the cheaper one to store.
    #
    # The sparsity is not a preference. A binary brain's smallest possible move
    # is a whole step against a weight spread of about 0.58, so any mutation
    # shifts a weight by roughly 1.7 times that spread, against 0.15 for the
    # float brain. At the float rate every child is substantially damaged: over
    # nine runs the binary populations died out in five. Dropping it to 0.02
    # turned one death and two small populations into three survivals two to
    # four times larger. Too little is its own failure — at 0.005 a run died
    # again.
    BRAIN_PRESETS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "float":   {"hidden_layers": [50, 45, 40, 35, 30], "mutation_sparsity": 0.1},
        "float16": {"hidden_layers": [50, 45, 40, 35, 30], "mutation_sparsity": 0.1},
        "binary":  {"hidden_layers": [160, 128, 96, 80, 64], "mutation_sparsity": 0.02},
    }

    # ---- Economy ----
    # Total tokens in the world. Conserved unless tokens_created_per_phase > 0.
    total_tokens: int = 10_000
    # Tokens injected at every cleanup. 0 keeps the economy closed.
    tokens_created_per_phase: int = 0

    # ---- Seed graph (Watts-Strogatz) ----
    # 0 means "derive from total_tokens": n = total_tokens / 100.
    n_nodes: int = 0
    # 0 means "derive from n": k = max(n / 100, 5).
    k_neighbors: int = 0
    # Watts-Strogatz: the chance each lattice link is redrawn to a random node
    # instead of a neighbour, which is what makes the starting graph small-world.
    # Nothing to do with agents moving edges — that mechanic no longer exists.
    rewire_p: float = 0.2

    # ---- Brain ----
    hidden_layers: List[int] = field(default_factory=lambda: [50, 45, 40, 35, 30])

    # How a brain stores and computes.
    #
    #   float    64-bit weights, sigmoid hidden layers. The original.
    #   float16  the same arithmetic on 16-bit weights: a quarter of the
    #            memory, and about three decimal digits of precision, which
    #            asks whether evolution needs the other thirteen.
    #   binary   weights of -1, 0 or +1 and hidden units that are on or off.
    #            Inputs arrive thermometer-coded, the output layer emits its
    #            integer sum so magnitudes survive, and no float is involved
    #            anywhere — which also makes a run bit-identical on any machine.
    #
    # A binary brain needs a gentler mutation_sparsity than the others, and
    # not by preference. Its smallest possible move is a whole step, while the
    # spread of its weights is about 0.58 — so any mutation at all shifts a
    # weight by roughly 1.7 times that spread, against 0.15 for the float
    # brain. At the same sparsity every child is substantially damaged: over
    # nine runs the binary populations died out in five. Dropping sparsity from
    # 0.1 to 0.02 turned one death and two small populations into three
    # survivals two to four times larger, across both ladder widths tried.
    # Too little is its own failure: at 0.005 a run died again. A wider ladder
    # does not rescue a high rate — the step size is fixed by the
    # representation, so the rate is the lever — but it does help at a low one.
    brain_kind: str = "float"
    # How many bits each input is spread over, for the binary kind. Every bit
    # is a threshold on the log-scaled value, so this sets how finely an agent
    # can tell one order of magnitude from another.
    brain_bits: int = 16
    message_amount: int = 5
    random_input_amount: int = 5
    exchange_messages: bool = True

    # An extra look at the start of each phase, for talking only.
    #
    # With this off, a phase is one pass: an agent observes, says what it has to
    # say, and acts, all from the same look. Its neighbours' messages are
    # therefore a phase old — written before the births, deaths and conquests
    # that have happened since — so an agent acts on a description of a graph
    # that no longer exists.
    #
    # With it on, everyone observes and writes messages first, those are
    # delivered, and only then does the pass that acts happen. The messages it
    # reads describe the graph as it stands right now. The acting pass still
    # writes messages of its own, so nothing is lost; there is simply a fresher
    # generation available to act on.
    #
    # It costs one extra forward pass per agent per phase, so a run is roughly
    # twice the work. Nothing about the brain changes, which is why — unlike
    # handover and revolutions — this one can be turned on or off for a run
    # that already exists.
    message_prepass: bool = True

    # ---- Rules ----
    # Let a parent hand one of its own connections to the newborn: the edge
    # moves from parent to child rather than being copied.
    allow_handover: bool = True

    # Let a coalition of smaller allocators unseat the largest one. With this
    # off, a node simply goes to whoever allocated the most, ties broken at
    # random, and the revolution fraction head disappears from the brain.
    allow_revolutions: bool = True

    # ---- Mutation ----
    mutation_probability: float = 0.5
    mutation_noise_std: float = 0.2
    mutation_sparsity: float = 0.1

    # ---- Run control ----
    # There is no iteration ceiling: a run goes until it is stopped or the
    # population dies out. How long it is worth running is a judgement made
    # while watching it, not one that can be guessed when it is created.
    # Below this node count the run is declared extinct and stops.
    extinction_threshold: int = 20
    # Write a resume checkpoint every N iterations. 0 disables checkpointing.
    checkpoint_every: int = 20
    # Random seed for reproducibility. Left blank on a new run, one is drawn and
    # written down — see resolve_seed. It fixes the starting graph and every
    # random choice the run goes on to make.
    seed: int | None = None

    # ---- Export ----
    # Record every Nth iteration into the viewer format. 1 = every iteration.
    export_every: int = 1
    # Store per-agent decisions alongside the topology. Costs disk, enables the
    # decision inspector in the UI.
    export_decisions: bool = True

    # --------------------------------------------------------------------
    # Derived values
    # --------------------------------------------------------------------

    def resolved_n(self) -> int:
        return self.n_nodes if self.n_nodes > 0 else int(self.total_tokens / 100)

    def resolved_k(self) -> int:
        if self.k_neighbors > 0:
            return self.k_neighbors
        return max(int(self.resolved_n() / 100), 5)

    # How an observation is laid out, in the order _input_vec builds it. Named
    # because the binary brain has to tell a magnitude from something that is
    # already a single bit, and counting to 29 in three places was how the two
    # would come to disagree.
    FLAG_INPUTS: ClassVar[int] = 1        # is-self
    MAGNITUDE_INPUTS: ClassVar[int] = 28  # own/target tokens and degrees, and quantiles

    def n_inputs(self) -> int:
        return (self.FLAG_INPUTS + self.MAGNITUDE_INPUTS
                + 4 * self.message_amount + self.random_input_amount)

    def bit_inputs(self) -> int:
        """
        Inputs that are already a single bit: the is-self flag, and — in a
        binary world — every message channel and every noise draw.

        Only magnitudes need a ladder. Spreading a value that is only ever 0 or
        1 across sixteen thresholds spends sixteen rows to say one thing, and
        fifteen of them can never change.
        """
        return (self.FLAG_INPUTS
                + 4 * self.message_amount + self.random_input_amount)

    def binary_rows(self) -> int:
        """The width of a binary brain's first layer."""
        return self.MAGNITUDE_INPUTS * self.brain_bits + self.bit_inputs()

    def n_outputs(self) -> int:
        # 9 always-present heads, plus the optional ones, then the message
        # vector. Conditional rather than always present so that a run keeps
        # exactly the architecture it was checkpointed with.
        return (9
                + (2 if self.allow_revolutions else 0)
                + (4 if self.allow_handover else 0)
                + self.message_amount)

    def head_layout(self) -> Dict[str, Any]:
        """Output row layout. Mirrors the engine's heads, for the UI to display."""
        layout = {
            "REPRO_FRACTION": [0, 2],
            "LINK": [2, 4],
            "LINK_MODE": [4, 6],
            "BLOTTO": [6, 7],
            "BLOTTO_MODE": [7, 9],
        }
        nxt = 9
        if self.allow_revolutions:
            layout["REV_FRACTION"] = [nxt, nxt + 2]
            nxt += 2
        if self.allow_handover:
            layout["HANDOVER"] = [nxt, nxt + 2]
            layout["HANDOVER_MODE"] = [nxt + 2, nxt + 4]
            nxt += 4
        layout["MESSAGE"] = [nxt, nxt + self.message_amount]
        return layout

    # --------------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, stored: bool = True) -> "SimConfig":
        """
        Build a config from untrusted input, ignoring unknown keys.

        `stored` says where the input came from. True — the default, because it
        is the answer that cannot corrupt an existing run — means it was read
        back from a run's own metadata, so a missing key resolves to what that
        run actually did. False means it is a fresh request, and a missing key
        resolves to today's default.
        """
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in (data or {}).items() if k in known}

        if stored:
            for key, legacy in cls.LEGACY_WHEN_ABSENT.items():
                clean.setdefault(key, legacy)

        if "hidden_layers" in clean:
            clean["hidden_layers"] = [int(x) for x in clean["hidden_layers"] if int(x) > 0]
            if not clean["hidden_layers"]:
                clean.pop("hidden_layers")

        cfg = cls(**clean)
        cfg.validate()
        return cfg

    def resolve_seed(self) -> "SimConfig":
        """
        Pick a seed if none was given, and write it down.

        "Blank means random" used to mean the run was seeded from entropy and
        nobody ever learned from what — so a run that produced something
        interesting could not be run again, and its settings could not honestly
        be shown, because the number that decided everything was not among
        them. Blank still means a different world every time; it just means a
        recorded one.

        Called when a run is created, not when a stored config is read, so an
        existing run keeps whatever it was given.
        """
        if self.seed is None:
            self.seed = secrets.randbits(32)
        return self

    def validate(self) -> None:
        """Reject values that would produce a broken or unrunnable simulation."""
        if self.total_tokens < 1:
            raise ValueError("total_tokens must be at least 1")
        if self.resolved_n() < 2:
            raise ValueError("seed graph needs at least 2 nodes")
        if self.resolved_k() < 2:
            raise ValueError("k_neighbors must be at least 2")
        if self.resolved_k() >= self.resolved_n():
            raise ValueError("k_neighbors must be smaller than the node count")
        if not 0.0 <= self.rewire_p <= 1.0:
            raise ValueError("rewire_p must be between 0 and 1")
        if self.message_amount < 0 or self.random_input_amount < 0:
            raise ValueError("message_amount and random_input_amount cannot be negative")
        if self.message_prepass and not self.exchange_messages:
            raise ValueError("message_prepass needs exchange_messages: a pass that "
                             "exists only to send messages has nothing to do without them")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between 0 and 1")
        if not 0.0 <= self.mutation_sparsity <= 1.0:
            raise ValueError("mutation_sparsity must be between 0 and 1")
        if self.mutation_noise_std < 0:
            raise ValueError("mutation_noise_std cannot be negative")
        if self.brain_kind not in ("float", "float16", "binary"):
            raise ValueError("brain_kind must be float, float16 or binary")
        if not 2 <= self.brain_bits <= 64:
            raise ValueError("brain_bits must be between 2 and 64")
        if self.export_every < 1:
            raise ValueError("export_every must be at least 1")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every cannot be negative")
        if self.total_tokens < self.resolved_n():
            raise ValueError("total_tokens must be at least the seed node count, or every node starts broke")
