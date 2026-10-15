# The switches the world was built with

> [!question] Questions of this chapter
> - The original algorithm came with switches of its own. Does any of them
>   keep evolution going?
> - What do revolutions do, in a world of nearly blind brains and in one of
>   brains that see?
> - Can gifts, tokens paid to keep a connection alive, make the shape of the
>   network a decision instead of a rule with a number in it?

## Why ask

Part V has judged its own rules by the tournaments of
[Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md): does a world keep
beating its own past? The switches the algorithm was built with were never
judged that way. Some were set for other reasons. Handover was taken out
because it breaks the network ([Chapter 48](48-a-world-that-cannot-break.md)).
Revolutions are planned in
Chapter 44, for what they do to wealth.
Gifts have been off in every run of the book. The author asked whether they
had been tested for open-ended evolution. They had not, and each has a reason
it might matter.

**Revolutions** ([How a coalition takes a node](../notes/revolution.md)) are
the one form of collective action the original game has. Stakers who mark
part of their stakes as revolutionary can, together, take a node from the
single largest staker if they outweigh it. A rule that rewards standing
together is the kind of rule [Chapter 50](50-why-evolution-stops.md) said
evolution needs, and B2 has had it on all along.

**Gifts** let an agent hand tokens to a neighbour during reproduction, as a
gift rather than a stake. A gift is a real transfer and it crosses a
connection, so it keeps that connection alive. The author built it to make
upkeep a decision. In the original algorithm a connection that carried no
tokens lapsed, so an agent could keep a neighbour only by staking on it.
With gifts, and with the accounting done after reproduction over a whole
iteration (`prune_after = reproduction`, `inactive_window = iteration`, the
form's default until 2026-10-02), an agent sees which of its connections are
about to lapse and can pay to keep them. B2 made upkeep a rule instead: a link
lapses only if its two ends share two other neighbours (`prune_needs = 2`).
That rule has a number in it. Gifts with every unused link lapsing
(`prune_needs = 0`) would put the network's shape in the brains' hands, which
is the kind of replacement for a number that
[Chapter 57](57-rules-without-numbers.md) looks for.

**Sharing the estate by wealth** (`redistribution = by_tokens`) gives the
tokens of the dead to the survivors in proportion to what they hold, which
makes the cull concentrate wealth instead of spreading it. A world of rich
and poor plays a different game.

## The worlds

Pilots, two seeds each, in the engine's own code (`--proto engine=true`, see
[Pilot worlds](../notes/pilot-worlds.md)), at 10,000 tokens for 1,000
iterations, with a whole world kept every 100 iterations:

| world | what it changes |
|---|---|
| B1 | the baseline of Parts II and III: brains of five layers, revolutions on |
| B1, no revolutions | `allow_revolutions = false` |
| B2 | a world that cannot break ([Chapter 48](48-a-world-that-cannot-break.md)), revolutions on |
| B2, no revolutions | `allow_revolutions = false` |
| B2 with gifts | `allow_gifting`, `prune_after = reproduction`, `inactive_window = iteration` |
| B2 with gifts, every unused link lapsing | the same, and `prune_needs = 0` |
| B2, every unused link lapsing | `prune_needs = 0` alone |
| B2, estates by wealth | `redistribution = by_tokens` |

B2's pilot of seed 1 is the same world as the lab's run `B2-10000-s001` of
[Experiment 16](50-why-evolution-stops.md), agent for agent and token for
token: the pilots in the engine's code reproduce the lab exactly.

<!-- turns -->
---

← [Chapter 52 · How fast should brains change?](52-how-fast-should-brains-change.md) · [Contents](../README.md) · [Chapter 55 · Another game](55-another-game.md) →
<!-- /turns -->
