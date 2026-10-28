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
from typing import Any, ClassVar, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Naming an algorithm
# ---------------------------------------------------------------------------
#
# An experiment has to cite the exact algorithm it ran on, and that citation
# has to still resolve in a year when six more mechanics exist. See
# research/Research.md §5 for the scheme and research/strains.md for the
# registry of every strain an experiment has used.
#
#     gol-<SPEC>[+<mechanic>[=<value>]]...
#
# Only mechanics differing from their FROZEN default are named, so adding a new
# mechanic that defaults to off never renames an existing strain. `gol-1` means
# the same thing forever. That is the whole trick, and it only works if the
# defaults below are never edited.

# Bumped ONLY for a change that cannot be expressed as an optional mechanic:
# a bug fix that alters results, or a change to a frozen default below.
SPEC = 1

# Mechanics — settings that change *which code paths can execute*. Part of the
# strain id. The value is the frozen default: what this setting was when SPEC
# was last bumped, so that omitting it reproduces that behaviour.
#
# Reserved but not yet implemented, so not fields and not settable:
# germline, token_colours, mutual_flow_yield, edge_proposal,
# growable_layers, local_rules, contracts, structure_replication. Their names
# and defaults are fixed in research/strains.md so the first experiment to
# turn one on does not also get to choose what it is called.
MECHANICS: Dict[str, Any] = {
    "brain_kind": "float",
    "exchange_messages": True,
    "message_prepass": True,
    "allow_handover": True,
    "allow_revolutions": True,
    "random_decisions": False,
    # A magnitude, but 0 against anything else is the difference between a
    # closed economy and one that mints, which is a different algorithm rather
    # than a different setting of one.
    "tokens_created_per_phase": 0,

    # ---- edge upkeep -------------------------------------------------------
    #
    # Frozen at what the engine did before they existed, like everything else
    # here. What a *new* run is offered is a different question, and it is
    # answered by NEW_RUN_DEFAULTS rather than by editing these.
    "allow_gifting": False,
    "prune_after": "blotto",
    "inactive_window": "phase",
    "redistribution": "uniform",

    # ---- open-ended evolution (Part V of the book) ---------------------------
    #
    # Each of these answers one way the pilots of research/oee/ saw a world
    # fail; research/strains.md says which, and the book's Part V teaches it.
    # All off, so that `gol-1` keeps naming the algorithm it always named.
    "mutate_on_replication": False,
    "incumbency": 1.0,
    "birth_links": 0,
    "mend_on_death": False,
    "prune_needs": 0,
    "valence": 0,
    "keep_home": 0.0,
    "max_age": 0,
    "min_child": 1,
    "child_floor": "fixed",
    "mix_at_birth": False,
    "mix_mates": "any",
    "buy_life": False,
    "self_message": "own",

    # ---- links to strangers ------------------------------------------------
    "link_search": "off",
    "link_steering": "random",
    "link_consent": "mutual",

    # ---- the author's walkthrough of 2026-10-24 --------------------------------
    #
    # Rules the author chose for the screen's next round of random worlds
    # (research/screen/walkthrough.md): a brain whose hidden layers pass a
    # change on, links that cost tokens, the lottery game and a mate that pays,
    # each off here so that `gol-1` keeps naming what it always named.
    "hidden_activation": "sigmoid",
    "contest": "largest",
    "mate_pays": False,
    "link_token": False,
    "link_upkeep": "off",
    "upkeep_to": "end",

    # ---- the original engine's game (2026-10-28) -------------------------------
    #
    # How the stakes are placed: all at once, the rule since the simple engine,
    # or a token at a time with the board in view, as the original engine
    # placed them (GraphOfLife.py of 2026-08-01). Off here, so `gol-1` keeps
    # naming what it always named.
    "allocation": "full",

    # ---- the pilots' rules ---------------------------------------------------
    #
    # Rules first played in the pilots' prototype (research/oee/pilots/proto.py)
    # and carried into the engine for the lab, one dict of switches rather than
    # a field each: see PROTO_SWITCHES. Empty, so no strain gains a name.
    "proto": {},
}

# What an agent reads of each candidate before it places a token, under
# allocation = weakest_first or strongest_first (gol_rounds.py): the original
# engine's blotto features (GraphOfLife.py of 2026-08-01) of the game as it
# stands, each a row of its observation, in this order. "On it" is on the
# candidate's node; "left" is tokens still to be placed this game; a "rival" is
# anyone else who may stake on the candidate's node: the candidate and its
# neighbours, but the observer.
BOARD: Tuple[str, ...] = (
    "staked",           # tokens on it so far, from everyone
    "lead",             # the largest stake on it
    "crossed",          # whether my link to it has carried a token this game (my own node: yes)
    "mine",             # what I have staked on it
    "its_own",          # what it has staked on itself
    "win_now",          # my share of its lead: 1 / how many share the lead, if I am one of them
    "my_left",          # tokens I have left
    "its_left",         # tokens it has left
    "around_left",      # tokens left to everyone who may stake on it
    "my_part",          # my part of those
    "rivals",           # how many rivals I have on it
    "rivals_left",      # tokens my rivals have left
    "rival_most_left",  # the most any one rival has left
    "others_staked",    # what others have staked on it
    "others_most",      # the largest stake on it by another
    "gap",              # how far the lead is ahead of the next stake
    "gap_share",        # that gap over one more than the lead
    "tied",             # how many share the lead
    "leading",          # whether I lead on it
    "behind",           # tokens I need to lead it
    "revolt_staked",    # revolutionary tokens on it
    "my_revolt",        # my revolutionary tokens on it
    "lead_revolts",     # whether its largest staker is also its largest revolutionary
    "would_win",        # whether I would take it if the game ended now (under the lottery, my chance)
)

# The pilots' rules a run may switch on in `proto`, each with the values it may
# take. A switch at its default is left out rather than spelt, so one world has
# one name. What each does is told at the field, SimConfig.proto.
PROTO_SWITCHES: Dict[str, Tuple[Any, ...]] = {
    "came_from": (True,),
    "helped_by": (True,),
    "contest": ("lottery",),
    "pool": ("genotype", "family"),
    "kin_sense": (True,),
    "tag_rate": (float,),
    "mate_pays": (True,),
    "grow": (True,),
    "unit_token": (True,),
    "link_token": (True,),
}

# What the new-simulation form arrives filled in with.
#
# Deliberately not the same thing as the frozen defaults above, and the
# distinction is the whole point. A frozen default is a promise about what
# `gol-1` names, and it can never move. This is a recommendation about what is
# worth running next, and it is free to move whenever the evidence does.
#
# A run started from these is named for every mechanic it differs in, so a
# different algorithm announces itself rather than quietly borrowing the old
# name. Today one differs: a new run is offered `gol-1+brain_kind=float16` —
# brains on half-precision weights, a quarter of the memory, and otherwise the
# frozen mechanics: no gifts, unused links cut after the game, judged on that
# phase alone — with longer messages and a slower rate of mutation than the
# frozen values. Those two are parameters, which a run cites beside its strain
# rather than in it. This is also the book's baseline B1. Until 2026-10-02 the
# form offered float brains with gifting, cutting after reproduction and a
# whole-iteration window instead,
#
#     gol-1+allow_gifting+inactive_window=iteration+prune_after=reproduction
#
# and the runs made then keep that name. Anything omitted here is offered at
# its frozen value. Redistribution stays uniform: weighting the estate of the
# dead by what a survivor already holds turns every cull into a concentration
# event, and that is an experiment rather than a default.
NEW_RUN_DEFAULTS: Dict[str, Any] = {
    "brain_kind": "float16",
    "message_amount": 30,
    "mutation_probability": 0.2,
}

# Parameters — magnitudes. Cited alongside an experiment, but not part of the
# strain: otherwise every seed would be its own algorithm.
PARAMETERS = (
    "total_tokens", "n_nodes", "k_neighbors", "rewire_p", "hidden_layers",
    "brain_bits", "message_amount", "random_input_amount",
    "mutation_probability", "mutation_noise_std", "mutation_sparsity",
    "extinction_threshold", "seed",
)

# Infrastructure — affects only what is recorded, never what happens.
INFRASTRUCTURE = ("checkpoint_every", "archive_every", "export_every", "export_decisions")


def _spell(value: Any) -> str:
    """One canonical spelling per value, so a strain id is comparable as text."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


@dataclass
class SimConfig:
    """All parameters of a single simulation run."""

    # What a run did before a setting existed, where that differs from the
    # frozen default.
    #
    # A stored configuration that predates one of these fields has to resolve
    # to the behaviour it actually had. Getting this wrong would not merely
    # mislabel a card — it would change the brain's shape and make the run's
    # own checkpoint unloadable.
    #
    # Only mechanics whose pre-existing behaviour differs from their frozen
    # default belong here, which is why the list is two entries and not eleven.
    # An absent key already resolves to the frozen default, so naming a
    # mechanic whose legacy behaviour *is* that default writes a line that
    # changes nothing — and buries the two that do. This list had seven such
    # lines, four of them added with a comment claiming they averted a broken
    # checkpoint; the frozen defaults were what averted it, and the entries
    # were never consulted. test_engine.py fails on a redundant entry now.
    #
    # Applied only when reading something off disk. A configuration arriving
    # from outside is a request for a new run, and a key it left out means
    # "whatever the default is", not "whatever a run from last year did" —
    # asking the API for a world without naming the pre-pass used to quietly
    # get one without it, which is the opposite of the documented default.
    LEGACY_WHEN_ABSENT: ClassVar[Dict[str, Any]] = {
        # Handover did not exist, so a run recorded before it ran without it.
        "allow_handover": False,
        # A pre-pass is the default now, and every run recorded before the
        # option existed ran with one pass per phase.
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

    # ---- Modes ----
    # Settings that name one of a few rules, and what each rule is. validate()
    # refuses anything else and the engine looks its rule up here, so a mode
    # the engine does not know is an error rather than a silent fall back onto
    # some other rule under a strain that names this one.
    #
    # After which phases unused connections are cut, by phase number.
    PRUNE_AFTER: ClassVar[Dict[str, Tuple[int, ...]]] = {
        "blotto": (2,), "reproduction": (1,), "both": (1, 2)}
    # How many phases a connection may go without carrying anything.
    INACTIVE_WINDOW: ClassVar[Dict[str, int]] = {"phase": 1, "iteration": 2}
    # How the tokens of removed agents are shared among the survivors.
    REDISTRIBUTIONS: ClassVar[Tuple[str, ...]] = (
        "uniform", "by_tokens", "poorest", "newborns", "parents", "children", "local")
    # How an agent reaches a stranger (link_search), which way it goes there
    # (link_steering), and who has to agree to the link it proposes (link_consent).
    LINK_SEARCHES: ClassVar[Tuple[str, ...]] = ("off", "jump", "walk", "reach")
    LINK_STEERINGS: ClassVar[Tuple[str, ...]] = ("random", "looking")
    LINK_CONSENTS: ClassVar[Tuple[str, ...]] = ("mutual", "forced")
    # Who may be a newborn's mate under mix_at_birth.
    MIX_MATES: ClassVar[Tuple[str, ...]] = ("any", "parents")
    # What an agent writes to itself: what its own column says, or the largest of
    # what it writes to anyone.
    SELF_MESSAGES: ClassVar[Tuple[str, ...]] = ("own", "largest")
    # The least a child may start with: min_child alone, or also the median
    # holding of its parent's neighbours.
    CHILD_FLOORS: ClassVar[Tuple[str, ...]] = ("fixed", "neighbours")
    # The curve a hidden unit applies to its sum.
    HIDDEN_ACTIVATIONS: ClassVar[Tuple[str, ...]] = ("sigmoid", "tanh")
    # Who takes a contested node: the largest stake, or a staker drawn in
    # proportion to its stake.
    CONTESTS: ClassVar[Tuple[str, ...]] = ("largest", "lottery")
    # How the game's stakes are placed: all at once, or a token at a time with
    # the poorest or the richest placing first.
    ALLOCATIONS: ClassVar[Tuple[str, ...]] = ("full", "weakest_first", "strongest_first")
    # Whether links cost upkeep, and what then keeps one alive; and where the
    # token paid for it goes.
    LINK_UPKEEPS: ClassVar[Tuple[str, ...]] = ("off", "paid_or_used", "paid_only")
    UPKEEP_TO: ClassVar[Tuple[str, ...]] = ("end", "estate")
    MODES: ClassVar[Dict[str, Tuple[str, ...]]] = {
        "brain_kind": tuple(BRAIN_PRESETS),
        "prune_after": tuple(PRUNE_AFTER),
        "inactive_window": tuple(INACTIVE_WINDOW),
        "redistribution": REDISTRIBUTIONS,
        "mix_mates": MIX_MATES,
        "self_message": SELF_MESSAGES,
        "child_floor": CHILD_FLOORS,
        "link_search": LINK_SEARCHES,
        "link_steering": LINK_STEERINGS,
        "link_consent": LINK_CONSENTS,
        "hidden_activation": HIDDEN_ACTIVATIONS,
        "contest": CONTESTS,
        "allocation": ALLOCATIONS,
        "link_upkeep": LINK_UPKEEPS,
        "upkeep_to": UPKEEP_TO,
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
    # The curve each hidden unit applies to its sum. A sigmoid passes on at most
    # a quarter of a change, so B1's five layers pass on about a thousandth of
    # what a brain sees, and its brains decide nearly the same whatever they see.
    # tanh is the same S-shape centred on zero, with a slope of one there: three
    # layers of it still pass on about half. The output layer stays linear. Float
    # brains only, since a binary brain's units are on or off.
    hidden_activation: str = "sigmoid"

    # How a brain stores and computes.
    #
    #   float    64-bit weights, sigmoid hidden layers. The original.
    #   float16  the same arithmetic on 16-bit weights: a quarter of the
    #            memory, and about three decimal digits of precision, which
    #            asks whether evolution needs the other thirteen.
    #   binary   weights of -1, 0 or +1 and hidden units that are on or off.
    #            Inputs arrive thermometer-coded, the output layer emits its
    #            integer sum so magnitudes survive, and no weight is a float —
    #            which takes the matrix arithmetic, the largest difference between
    #            machines, out of a run. Its inputs are still floats until laddered.
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
    # How many rows each magnitude is spread over, for the binary kind. Split
    # between the two fields below, so this is the total width, not a count of
    # thresholds on one ladder.
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

    # Who takes a contested node. `largest`, the old rule: the largest stake as
    # the game weighs it, or a coalition that outweighs it. `lottery`: a staker
    # drawn with a chance in proportion to its weighed stake (Tullock's contest),
    # so a small stake can win and a large one lose. A coalition belongs to the
    # largest-stake rule, so the lottery needs revolutions off. The pilots'
    # rule, in the engine.
    contest: str = "largest"

    # How the game's stakes are placed: the original engine's
    # BLOTTO_ALLOCATION_MODE (GraphOfLife.py of 2026-08-01), back in the engine
    # at the author's request (2026-10-28). `full`, the rule since: every agent
    # looks once and places all its tokens at once. `weakest_first` and
    # `strongest_first`: a token at a time, in rounds (gol_rounds.py). Before
    # each round an agent that places one this round looks again, now with the
    # board in view (BOARD: what each candidate has drawn so far and from whom,
    # who leads it and by how much, what is still to be placed), and puts its
    # next token where its brain scores highest. Under weakest_first every
    # agent with a token left places one each round, so the poor are done first
    # and the richest places its last tokens alone; under strongest_first only
    # the agents with the most tokens left place one, so the richest starts
    # alone and everyone places a last token in the last round. The board adds
    # its rows to every observation in these two, and no head is added.
    allocation: str = "full"

    # The control. With this on, an agent never reads its inputs: the numbers
    # its brain would have produced are replaced with noise, and every decision
    # downstream follows from that instead. Nothing else changes — the same
    # heads, the same reproduction rule, the same Blotto, the same mutation of
    # brains that no longer matter — so a run of this against a run without it
    # isolates exactly one thing: whether processing the inputs is doing
    # anything at all.
    random_decisions: bool = False

    # ---- Edge upkeep ----
    #
    # A connection has always had to be used or die. What was missing was any
    # way to *choose* to keep one: the only act that could put tokens on an
    # edge was a Blotto bid, so an edge nobody happened to bid across died
    # whether or not anyone wanted it. These four settings turn upkeep into a
    # decision — what counts as use, when the accounting falls due, and whether
    # an agent may pay to keep a connection it values.

    # Let an agent hand tokens to a neighbour during reproduction, as a gift
    # rather than a bid. The tokens are a real transfer and the act is real
    # flow, so a gift keeps the connection it crosses alive. This adds three
    # output heads and the input flag that makes them worth having, so it
    # cannot be switched on for a run that already exists.
    #
    # Frozen off, like every mechanic here, so that `gol-1` keeps naming one
    # fixed algorithm. What a *new* run is offered is a separate question and
    # is answered by NEW_RUN_DEFAULTS below.
    allow_gifting: bool = False

    # When the accounting falls due: after the reproduction phase, after the
    # Blotto phase, or after both. Pruning after reproduction is what gives a
    # gift somewhere to land — an agent sees a connection about to lapse and
    # can spend to save it in the same phase.
    prune_after: str = "blotto"

    # What "unused" means. `phase` judges an edge on the phase just ended;
    # `iteration` gives it the last two, so a connection that carried a Blotto
    # bid is still in credit when the reproduction phase settles up. A newly
    # created edge counts as active for the phase it was born in, whichever
    # window is in force — a newborn's connections are never killed before
    # anyone has had a chance to use them.
    inactive_window: str = "phase"

    # How the estate of the dead is shared out. `uniform` scatters it evenly
    # over the survivors; `by_tokens` weights each survivor's share by what it
    # already holds, which makes the cull an engine of concentration rather
    # than a leveller; `poorest` raises the poorest to a common level, like
    # water filling the lowest places first; `newborns` gives it to the
    # children born this iteration, and `parents` to their parents, once per
    # child; those two fall back to `uniform` in an iteration without any.
    # `children` leaves each dead agent's estate to its own living children,
    # and what has no heir goes as `uniform`. `local` lets each estate flow
    # through the network to the nearest living agents: from the dead agent to
    # its neighbours one step nearer the living, split evenly, and on until it
    # reaches them, over the network as it stood before the phase's lapses, so
    # that a piece cut off whole still flows out across the link it lost. An
    # estate with no way to the living goes to the dead agent's parent if it
    # lives, and what is no one's estate goes as `uniform`.
    redistribution: str = "uniform"

    # ---- Open-ended evolution ----
    #
    # Nine rules, each the answer to a way a world was seen to fail when brains
    # could see (research/oee/LOG.md, the book's Part V), and a tenth from the
    # pilots that followed (mix_at_birth). Every one is off by default and
    # local: it reads an agent, its links and its neighbours, and nothing
    # further away.

    # Heredity. A brain changes only when it is copied — into a newborn, or into
    # a node its owner won — and no longer after every game wherever it sits.
    # The world-wide change wore every adaptation away within a few dozen
    # iterations of its making.
    mutate_on_replication: bool = False

    # An agent's own stake on its node counts this many times over when the game
    # decides who takes the node. 1 is the old rule; above it, defending is
    # cheaper than attacking.
    incumbency: float = 1.0

    # Space. A newborn is always joined to its parent and to this many of the
    # parent's neighbours, drawn at random, before its own link choices. Every
    # birth then closes triangles; with 0 the newborn has only the links its
    # brain chooses, and is often a leaf.
    birth_links: int = 0
    # When an agent dies, the groups of its former neighbours that are not
    # linked to one another are joined in a ring, through their least-linked
    # members, so a death never splits a neighbourhood — and so never the
    # world — and none of the new links is a bridge.
    mend_on_death: bool = False
    # How many neighbours the two ends of an unused link must have in common
    # for the link to lapse. 0 is the old rule — any unused link lapses — and
    # it cuts bridges as readily as anything else; 2 lets only links lapse
    # whose ends stay joined twice over without them.
    prune_needs: int = 0
    # The most links an agent's brain may choose to hold (0 = no limit). A
    # newborn's links to its parent and the parent's neighbours are exempt, and
    # so is mending. An agent at its limit cannot have a child: it is crowded.
    valence: int = 0
    # A cost for links instead of a cap on them. Every link holds one token for
    # as long as it stands: whoever makes it pays — a child for the links it is
    # born with, a searcher for its find, an agent mending a death for the mend
    # — and a link its maker cannot pay for is not made. A link that lapses
    # gives its token to one of its two ends, drawn at random; a link that goes
    # with a dead agent gives it to the end that lives on, or to the estate when
    # both die; a link handed to a child keeps its token. The tokens agents hold
    # and the links together are the world's tokens, so the links can never
    # outnumber them. The pilots' rule, in the engine.
    link_token: bool = False
    # Upkeep a brain chooses. Every iteration, in the reproduction phase, an
    # agent says of each neighbour whether to pay one token to keep their link
    # (its UPKEEP head, read like the link decision), out of what reproduction
    # left and never its last token; a link is paid for if either end pays.
    # `paid_or_used`: a link lives if it was paid for this iteration or tokens
    # crossed it, payment being another way to stay; `paid_only`: only paid
    # links live, used or not. A new link counts as paid for the iteration it is
    # made in. Off, the old rule: only use keeps a link.
    link_upkeep: str = "off"
    # Where an upkeep token goes: to the link's other end (`end`), or into the
    # estate of the dead (`estate`), shared out with it by the world's
    # redistribution rule.
    upkeep_to: str = "end"

    # Demography. At least this share of an agent's tokens stays on its own
    # node whatever its brain says — rounded up, and never less than one token
    # — and a parent keeps at least one: no agent can stake or breed itself to
    # death. 0 is the old rule. A small share is a floor of one token; a larger
    # one also keeps agents from staking nearly everything on one neighbour,
    # which in the pilots left three agents in four with a single token.
    keep_home: float = 0.0
    # An agent dies when it reaches this age (0 = never). Its tokens are shared
    # out like any other estate.
    max_age: int = 0
    # Under max_age, whether an agent that has reached it may live on by paying
    # one token an iteration into the estate of the dead — never its last, so an
    # agent with one token left dies. Off, every agent dies at max_age.
    buy_life: bool = False
    # The fewest tokens a child can be born with; a smaller endowment means no
    # birth. 1 is the old rule.
    min_child: int = 1
    # With `neighbours`, a child must also start with at least the median holding
    # of its parent's neighbours: a world whose births are cheap (an estate paid
    # to the newborns, say) can then grow only as rich as it is, rather than
    # filling with paupers (Chapter 100). Local and without a number of its own.
    # `fixed`, the old rule, asks only min_child.
    child_floor: str = "fixed"

    # Heredity of parts. A newborn's brain is its parent's mixed with a mate's
    # — one of the parent's neighbours, drawn at random — before it mutates:
    # each hidden unit, with the weights into it, its bias and the weights out
    # of it, comes from one or the other by an even chance, and so does each
    # output's bias (with more hidden layers, each unit of every layer, with
    # its weights in and its bias). A unit that works can then spread without
    # the rest of the brain it arose in. The mixture is a new genotype unless
    # it is the parent's brain again, as it is with a mate of the same
    # genotype; a parent without a neighbour has a copy. Off is the old rule.
    # The pilots of the book's Chapter 78 found it.
    mix_at_birth: bool = False
    # Under mix_at_birth, who may be the mate: `any` neighbour of the parent,
    # or only `parents` — a neighbour with a living child of its own, so that
    # an agent that never breeds does not spread half its brain for free
    # through other agents' children. With no such neighbour the newborn's
    # brain is a copy of its parent's.
    mix_mates: str = "any"
    # Under mix_at_birth, the mate pays the child as many tokens as the parent
    # gave it, as far as it can and never its last, once every birth of the
    # phase is made: a mate's half of the brain costs it something, and a
    # child's price is shared. The pilots' rule (Experiment 22), in the engine.
    mate_pays: bool = False

    # Reach beyond a pair. A brain scores each candidate from that candidate's
    # column alone, so what it writes to one neighbour can never carry news of
    # another, and no message ever says "a stranger is next door to me"
    # (Chapter 94). With `largest`, what an agent writes to itself — which each
    # of its neighbours reads as one of its inputs — is the largest, number by
    # number, of what it writes to all its candidates, itself included: what one
    # column marks reaches every neighbour a phase later. `own`, the old rule,
    # is what its own column says.
    self_message: str = "own"

    # ---- Links to strangers ----
    #
    # Every link an agent can make is to its own neighbourhood: its newborn's,
    # chosen from the parent's neighbours. With link_search an agent can reach
    # further, in jumps and walks paying one token for every step, to the agent
    # a step enters, and never its last token. Off is the old rule.
    #
    #   jump   in one iteration a walk as many steps long as the tokens the
    #          agent spends on it (a SEEK head's share of what reproduction
    #          left), and a link proposed where it ends;
    #   walk   a scout that goes one step further in every iteration the agent
    #          pays for, and a link proposed wherever it stands when the agent
    #          wants one; after a proposal the scout comes home;
    #   reach  a jump that costs nothing: as many steps as the SEEK head's share
    #          of the tokens the agent holds, so a richer agent reaches further.
    link_search: str = "off"
    # Which way each step goes: to a neighbour drawn at random, or to the one
    # the searcher would most like a link with (its ACCEPT head, looking at
    # every neighbour of where it stands).
    link_steering: str = "random"
    # Who has to say yes: both ends, each by its ACCEPT head, or only the one
    # who proposes. Either way no link is made to or from an agent at its
    # valence.
    link_consent: str = "mutual"

    # ---- The pilots' rules ----
    #
    # Rules the pilot worlds of research/oee/pilots/ played before the engine
    # had them, switched on by name (PROTO_SWITCHES lists the values). Only the
    # prototype's world (gol_proto) plays them; the engine's own refuses a run
    # that asks for one.
    #
    #   came_from   one input row per candidate: is this the node my brain
    #               came from — the node whose holder took mine, or my parent
    #               if I was born here?
    #   helped_by   one input row per candidate: what its stake on my node did
    #               in the last game — the chance I kept my node with every
    #               stake, minus the chance without its stake.
    #   contest     "lottery": a node goes to a staker drawn in proportion to
    #               its stake as the game weighs it (Tullock's contest), not to
    #               the largest.
    #   pool        "genotype": the stakes on a node of agents with one brain
    #               count as one side's, the side with most takes the node, and
    #               the owner keeps it if its side wins; "family" pools by
    #               family tag instead. Adds an input row: is this candidate on
    #               my side?
    #   kin_sense   the same input row without the pooling: is this
    #               candidate's brain mine (with tag_rate, my family)?
    #   tag_rate    a family tag carried with every brain: a copy keeps its
    #               parent's tag, and a brain that mutates takes a new tag with
    #               this probability, so a family is all the genotypes
    #               descended from one tag (a mutation parameter).
    #   mate_pays   under mix_at_birth, the mate pays the child as many tokens
    #               as the parent gave it, as far as it can and never its last,
    #               once every birth of the phase is made.
    #   grow        a brain of one hidden layer of which only the first units
    #               take part; a mutation may switch the next one on or the
    #               last one off.
    #   unit_token  under grow, every unit that takes part holds a token, paid
    #               by the node it is in.
    #   link_token  every link holds a token: its maker pays for it, and a link
    #               that lapses returns its token to one of its ends.
    proto: Dict[str, Any] = field(default_factory=dict)

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
    # Keep a whole world every N iterations, beside the run in archive/, for
    # whatever needs the brains of more than one moment — a cross-time
    # tournament above all. The checkpoint is overwritten; these are not.
    # 0 keeps none.
    archive_every: int = 0
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

    def records(self, iteration: int) -> bool:
        """Whether an iteration's two frames are recorded: every export_every-th, from 0."""
        return iteration % self.export_every == 0

    def frames_before(self, iteration: int) -> int:
        """
        How many frames the iterations below `iteration` recorded, two apiece.

        On resume this is where the checkpoint's timeline ends: frames past it
        belong to a future the checkpoint never lived through, and are dropped.
        Both backends resume, and the browser's used to keep them.
        """
        recorded = -(-max(0, iteration) // max(1, self.export_every))  # ceil
        return recorded * 2

    def resolved_n(self) -> int:
        return self.n_nodes if self.n_nodes > 0 else int(self.total_tokens / 100)

    def resolved_k(self) -> int:
        if self.k_neighbors > 0:
            return self.k_neighbors
        return max(int(self.resolved_n() / 100), 5)

    def input_layout(self) -> Dict[str, Tuple[int, int, str]]:
        """
        Every block of rows in an observation, by name and in the order
        _inputs fills them: where it starts, where it stops, and what kind of
        value it holds — a flag, already a single bit; a magnitude (tokens,
        connections and their neighbourhood quantiles), which a binary brain
        spreads over a ladder; or a bit (a message channel or a noise draw,
        single bits in a binary world).

        The one statement of the input, as head_layout() is of the output:
        every count of input rows is read off it.

        Gifting brings a second flag: whether the connection to this candidate
        would lapse if nothing more crossed it. Without it an agent can see a
        neighbour's tokens and degree but not the one fact the decision turns
        on, and paying to save an edge becomes a guess. Conditional rather than
        always present for the same reason the output heads are — a run keeps
        exactly the architecture it was checkpointed with.
        """
        m, noise = self.message_amount, self.random_input_amount
        blocks = [("self", 1, "flag")]
        if self.allow_gifting:
            blocks.append(("at_risk", 1, "flag"))
        blocks += [
            ("own_tokens", 1, "magnitude"), ("its_tokens", 1, "magnitude"),
            ("own_degree", 1, "magnitude"), ("its_degree", 1, "magnitude"),
            ("own_token_quantiles", 6, "magnitude"), ("its_token_quantiles", 6, "magnitude"),
            ("own_degree_quantiles", 6, "magnitude"), ("its_degree_quantiles", 6, "magnitude"),
        ]
        # The game as it stands, read before each token under a step allocation (BOARD).
        if self.allocation != "full":
            blocks.append(("board", len(BOARD), "magnitude"))
        blocks += [
            # What each of the two wrote: the observer to itself, the observer
            # to the candidate, the candidate to the observer, the candidate to
            # itself.
            ("message_to_self", m, "bit"), ("message_to_it", m, "bit"),
            ("message_from_it", m, "bit"), ("message_its_own", m, "bit"),
            ("noise", noise, "bit"),
        ]
        # What the pilots' rules let an agent see more of, after the noise and
        # in the prototype's order, so that a pilot's brain and the lab's read
        # the same rows in the same places.
        if self.proto.get("pool") or self.proto.get("kin_sense"):
            blocks.append(("same_side", 1, "bit"))
        for name in ("came_from", "helped_by"):
            if self.proto.get(name):
                blocks.append((name, 1, "bit"))
        layout, row = {}, 0
        for name, width, kind in blocks:
            layout[name] = (row, row + width, kind)
            row += width
        return layout

    def _input_rows(self, *kinds: str) -> int:
        return sum(stop - start for start, stop, kind in self.input_layout().values()
                   if kind in kinds)

    def input_kinds(self) -> Dict[str, slice]:
        """
        The rows of each kind of input — flags, then magnitudes, then bits —
        which is how a binary brain cuts an observation up before it ladders
        the magnitudes. Each kind is one unbroken run of rows, and they come in
        that order; a layout that broke this would be refused here rather than
        encoded wrongly.
        """
        spans: Dict[str, List[int]] = {}
        for start, stop, kind in self.input_layout().values():
            if stop > start:
                spans.setdefault(kind, []).append(start)
                spans[kind].append(stop)
        out, row = {}, 0
        for kind in ("flag", "magnitude", "bit"):
            edges = spans.get(kind, [row, row])
            if edges[0] != row or any(a != b for a, b in zip(edges[1::2], edges[2::2])):
                raise ValueError(f"the {kind} inputs are not one run of rows after the others")
            out[kind] = slice(edges[0], edges[-1])
            row = edges[-1]
        return out

    def flag_inputs(self) -> int:
        """Single-bit inputs: is-self, and with gifting whether the link is at risk."""
        return self._input_rows("flag")

    def n_inputs(self) -> int:
        return self._input_rows("flag", "magnitude", "bit")

    def ladder_split(self) -> tuple[int, int]:
        """
        How a magnitude's rows divide between its band and its place inside it.

        One ladder resolves as many levels as it has rungs, which over a range
        of e^12 in tokens made a level a factor of 2.23 — an agent could not
        tell a hundred tokens from a hundred and eighty. Two ladders, one for
        which band the value is in and one for where it sits inside that band,
        resolve the product rather than the sum: sixteen rows go from fifteen
        levels to thirty-six.

        Both fields stay monotone in the value, which is the whole constraint.
        A binary unit computes a sum of weights in -1, 0 and +1 and thresholds
        it, and that sum cannot weight a row by two to its position — so an
        encoding has to put the magnitude somewhere a plain sum can find it.
        Measured over four thousand values and four hundred random units, a
        ladder's row count tracks the value at 0.995 and a place-value code's
        at 0.408, and the split ladder gives up almost nothing: 72% of random
        units read it against 73% for the plain one.

        A quarter of the rows to the fine field. Wider than that and
        readability starts to go; narrower and there is nothing gained.
        """
        within = max(1, self.brain_bits // 4)
        return self.brain_bits - within, within

    def bit_inputs(self) -> int:
        """
        Inputs that are already a single bit: the is-self flag, and — in a
        binary world — every message channel and every noise draw.

        Only magnitudes need a ladder. Spreading a value that is only ever 0 or
        1 across sixteen thresholds spends sixteen rows to say one thing, and
        fifteen of them can never change.
        """
        return self._input_rows("flag", "bit")

    def binary_rows(self) -> int:
        """The width of a binary brain's first layer."""
        return self._input_rows("magnitude") * self.brain_bits + self.bit_inputs()

    def n_outputs(self) -> int:
        """
        The rows of the output layer: up to where the last head, the message
        vector, ends. Read off the layout rather than added up again beside
        it, which is how the two came to need a test to hold them together.
        """
        return self.head_layout()["MESSAGE"][1]

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
        if self.allow_gifting:
            # How much of what reproduction left me do I give away at all,
            # then which neighbours receive a share of it. Read exactly like
            # reproduction's fraction and like the link decision, so a gift is
            # not a new kind of choice — only a new thing to choose about.
            layout["GIFT_FRACTION"] = [nxt, nxt + 2]
            layout["GIFT"] = [nxt + 2, nxt + 4]
            layout["GIFT_MODE"] = [nxt + 4, nxt + 6]
            nxt += 6
        if self.link_search != "off":
            # How much of what reproduction left to spend searching, read like
            # the gift's fraction; and yes or no to a link with a stranger, read
            # like the link decision (with LINK_MODE).
            layout["SEEK"] = [nxt, nxt + 2]
            layout["ACCEPT"] = [nxt + 2, nxt + 4]
            nxt += 4
        if self.link_upkeep != "off":
            # Yes or no to paying a token to keep the link to each neighbour,
            # read like the link decision (with LINK_MODE).
            layout["UPKEEP"] = [nxt, nxt + 2]
            nxt += 2
        layout["MESSAGE"] = [nxt, nxt + self.message_amount]
        return layout

    # --------------------------------------------------------------------
    # Naming this algorithm
    # --------------------------------------------------------------------

    @classmethod
    def for_new_run(cls, **overrides: Any) -> "SimConfig":
        """
        The configuration the new-simulation form is offered.

        Not `SimConfig()`, which is frozen at whatever `gol-1` means and has to
        stay there. This is what the project currently thinks is worth running,
        and it names itself accordingly.
        """
        return cls(**{**NEW_RUN_DEFAULTS, **overrides})

    def strain_id(self) -> str:
        """
        Which algorithm this is, as a citable string.

        `gol-1` is every mechanic at its frozen default. Anything else names
        only what differs, alphabetically, so two strains are comparable as
        text and a new mechanic never renames an old one.

        A boolean that is on appears bare — `gol-1+mutate_on_replication` reads
        as a thing that was switched on. One that is off is named with its
        value, because the absence of a name already means "default".

        This does not pin bugs. An experiment record carries the commit too;
        see research/strains.md.
        """
        named = []
        for name in sorted(MECHANICS):
            value = getattr(self, name)
            if value == MECHANICS[name]:
                continue
            if name == "proto":
                # Each of the pilots' rules is named as a mechanic of its own would be.
                named.extend(f"proto.{k}" if v is True else f"proto.{k}={_spell(v)}"
                             for k, v in sorted(value.items()))
                continue
            named.append(name if value is True else f"{name}={_spell(value)}")
        return f"gol-{SPEC}" + "".join(f"+{part}" for part in named)

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
        if self.link_search == "off" and (self.link_steering, self.link_consent) != ("random", "mutual"):
            raise ValueError("link_steering and link_consent say how a search for strangers goes: "
                             "they need link_search")
        if self.mix_mates != "any" and not self.mix_at_birth:
            raise ValueError("mix_mates says who a newborn's brain is mixed with: it needs mix_at_birth")
        if self.buy_life and not self.max_age:
            raise ValueError("buy_life buys a life past max_age: it needs max_age")
        if self.mate_pays and not self.mix_at_birth:
            raise ValueError("mate_pays makes a mate pay for its child: it needs mix_at_birth")
        if self.allocation != "full" and self.proto:
            raise ValueError("allocation places the engine's stakes a token at a time; the prototype's world "
                             "(proto) plays a game of its own")
        if self.contest == "lottery" and self.allow_revolutions:
            raise ValueError("contest = lottery draws a node's taker by chance, and a coalition belongs "
                             "to the largest-stake rule: it needs allow_revolutions off")
        if self.hidden_activation != "sigmoid" and self.brain_kind == "binary":
            raise ValueError("hidden_activation shapes a float brain's units; a binary brain's are on or off")
        if self.upkeep_to != "end" and self.link_upkeep == "off":
            raise ValueError("upkeep_to says where a link's upkeep goes: it needs link_upkeep")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between 0 and 1")
        if not 0.0 <= self.mutation_sparsity <= 1.0:
            raise ValueError("mutation_sparsity must be between 0 and 1")
        if self.mutation_noise_std < 0:
            raise ValueError("mutation_noise_std cannot be negative")
        for setting, allowed in self.MODES.items():
            if getattr(self, setting) not in allowed:
                raise ValueError(f"{setting} must be {', '.join(allowed[:-1])} or {allowed[-1]}")
        if not 2 <= self.brain_bits <= 64:
            raise ValueError("brain_bits must be between 2 and 64")
        if self.export_every < 1:
            raise ValueError("export_every must be at least 1")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every cannot be negative")
        if self.archive_every < 0:
            raise ValueError("archive_every cannot be negative")
        if self.total_tokens < self.resolved_n():
            raise ValueError("total_tokens must be at least the seed node count, or every node starts broke")
        if not self.incumbency > 0:
            raise ValueError("incumbency must be positive")
        for name in ("birth_links", "prune_needs", "valence", "max_age"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0.0 <= self.keep_home <= 1.0:
            raise ValueError("keep_home is a share of an agent's tokens, between 0 and 1")
        if self.min_child < 1:
            raise ValueError("min_child must be at least 1: a child is born with something")
        if self.valence == 1:
            raise ValueError("valence 1 would let no agent choose a second link; use 0 for no limit")
        self._validate_proto()

    def _validate_proto(self) -> None:
        """The pilots' rules: known ones, each at a value it may take, and what each needs."""
        proto = self.proto
        if not isinstance(proto, dict):
            raise ValueError("proto is a table of the pilots' rules, {} for none")
        unknown = sorted(set(proto) - set(PROTO_SWITCHES))
        if unknown:
            raise ValueError(f"proto: {', '.join(unknown)} is none of the pilots' rules "
                             f"({', '.join(PROTO_SWITCHES)})")
        for name, value in proto.items():
            allowed = PROTO_SWITCHES[name]
            if allowed == (float,):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
                    raise ValueError(f"proto.{name} is a probability above 0 and at most 1")
            elif type(value) is not type(allowed[0]) or value not in allowed:
                raise ValueError(f"proto.{name} may be {' or '.join(_spell(v) for v in allowed)}; "
                                 f"a rule at its default is left out")
        if proto.get("mate_pays") and not self.mix_at_birth:
            raise ValueError("proto.mate_pays makes a mate pay for its child: it needs mix_at_birth")
        if proto.get("pool") == "family" and not proto.get("tag_rate"):
            raise ValueError("proto.pool=family pools stakes by family tag: it needs proto.tag_rate")
        if proto.get("unit_token") and not proto.get("grow"):
            raise ValueError("proto.unit_token makes a growing brain's units hold a token: it needs proto.grow")
        if proto.get("link_token") and self.valence:
            raise ValueError("proto.link_token lets the tokens limit the links: it needs valence 0")
        if proto.get("grow") and (self.brain_kind != "float16" or len(self.hidden_layers) != 1):
            raise ValueError("proto.grow grows one hidden layer of half-precision weights: it needs "
                             "brain_kind float16 and one hidden layer")
