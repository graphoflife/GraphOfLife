# Reproduction statistics

## In short

Every reproduction phase records each birth. It notes the parent, the [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.")
it held and gave, whom the child was joined to, and the [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") the parent
handed over. The row of statistics written after the phase sums these up in a
few numbers. Two of them are shares that answer different questions. One says
how much of itself a typical parent gives; the other, how much of all the
tokens went into children.

## What a phase records

Think of a register of births, one line per child. The frame written after
the reproduction phase keeps it as `decisions.births`
([What a run records](frames-and-stats.md)):

| key | what it holds |
|---|---|
| `agent`, `child` | the parent's id, and the child's |
| `tokens_before` | the tokens the parent held before it gave any |
| `invested` | the tokens it gave the child |
| `links` | the candidates the child was joined to at birth |
| `handed_over` | the [neighbours](the-world.md#the-network "An agent joined to another by a connection.") whose connections the parent chose to hand to the child |

A parent's candidates are the parent itself and its neighbours. A handover
moves one of the parent's connections to its child
([One iteration, step by step](one-iteration.md#phase-1-reproduction)).

## The fields

The statistics of a phase-1 row summarise the register. In the rows of the
game they are empty (`null`).

| field | definition |
|---|---|
| `births` | the number of children born, also those the cleanup removes at once |
| `meanInvestedShare` | the mean, over the parents, of tokens given ÷ tokens held |
| `reproTokenShare` | all tokens given to children ÷ all tokens in the world |
| `meanChildLinks` | the mean number of candidates a child was joined to |
| `handovers` | the number of connections parents chose to hand to their children |
| `gifts`, `giftTokens`, `giftShare` | how many gifts of tokens [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") made to neighbours, how many tokens they carried, and those tokens' share of all tokens: 0 when gifting is off, as in both [baselines](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") |

## Two shares that answer different questions

`meanInvestedShare` is **per parent**: how much of itself a typical parent
gives. `reproTokenShare` is **per world**: how much of all the tokens went
into children. A world in which a few poor agents each give everything they
have has a high per-parent share and a tiny world share.

**Example.** Three parents holding 2, 4 and 100 tokens give 1, 2 and 10. Each
gives a share of what it holds, ½, ½ and ⅒, and their mean is
`meanInvestedShare` = (½ + ½ + ⅒)/3 ≈ 0.37. Together they give 13 tokens. In
a world of 10,000 tokens, `reproTokenShare` = 13/10,000 = 0.0013.

The baseline's worlds show the same gap. In the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") world of 100,000
tokens, over its [settled life](settled-life.md "A world from iteration 500 to its end, after its youth, where it is measured."), parents give 36% of what they hold. Yet all the
tokens given to children in one phase are 1.3% of the world's
([How births build the network](how-agents-have-children.md)).

## Links of a child

A child's links are chosen among its parent's candidates, **the parent
included**. So `meanChildLinks` counts the link to the parent too, if there is
one. In [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") a newborn is first joined to its parent and to two of the parent's
neighbours, drawn at random ([`birth_links`](settings.md#birth_links)). These
count as well. The connections a child is handed do not.

A child joined to no candidate and handed no connection belongs to no
network. It dies in the cleanup that ends the phase
([Births and the four ways to die](births-and-deaths.md)). In the B1 worlds of
100,000 tokens, 11.5% of children are joined to no candidate at birth. Two in
five of these (39%) are handed a connection by their parent, and live. The
rest, some 7% of all children, die at once
([How births build the network](how-agents-have-children.md)).

## Handovers

`handovers` counts connections that moved from parent to child: the parent
loses them, the child gains them. They never add a connection to the world. A
handover to a neighbour the child is already joined to even removes one: the
parent loses its connection, and the child gains nothing new.

The field counts what the parents chose. A connection that both of its ends
chose to hand over counts twice, but moves only once: by the second move, it
is gone. B2 has no handovers ([`allow_handover`](settings.md#allow_handover)),
so there the field is 0.

## Gifts

With [`allow_gifting`](settings.md#allow_gifting) on, an agent may give some
of what its child left it to its neighbours, in the reproduction phase. The
field `gifts` counts the gifts paid, one for each giver and neighbour. The
field `giftTokens` counts the tokens they carried, and `giftShare` is those
tokens as a share of all tokens. Gifting is off in both baselines, so there
all three are 0. It is on in some [pilots](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.")
([Gifts, with every other rule](gifts-with-every-other-rule.md)), in
Experiment 24 and in half the worlds of
[A screen of random worlds](a-screen-of-random-worlds.md).

## Where it is used

- [How births build the network](how-agents-have-children.md): these statistics
  over a world's life, in the B1 worlds.
- [What a run records](frames-and-stats.md): every field of a run's
  statistics.

## Where to go next

- [One iteration, step by step](one-iteration.md#phase-1-reproduction): the
  reproduction phase, rule by rule.
- [Births and the four ways to die](births-and-deaths.md): how births and
  deaths are counted per hundred agents.
- [How births build the network](how-agents-have-children.md): who has children,
  and how much they give.

## How our view changed

- **8 October 2026.** This page said that gifting was off in every run of the
  notebook, so that the three gift fields were absent.
- **15 October 2026.** Pilots with gifts were played for
  [The switches the world was built with](the-switches-the-world-was-built-with.md).
  More pilots followed, then Experiment 24 and half the worlds of
  [A screen of random worlds](a-screen-of-random-worlds.md).
- **23 October 2026.** Reading the runs' statistics files showed that the gift
  fields are never absent from a row of reproduction. Where gifting is off
  they are 0; only the rows of the game leave them empty. The page now says
  both.

<!-- turns -->
---

← [Age and lifetime](age-and-lifetime.md) · [Agents' lives: births, deaths, ages](../README.md#agents-lives-births-deaths-ages)

> [!info]- Pages that link here (3)
> [How births build the network](how-agents-have-children.md) · [Births and the four ways to die](births-and-deaths.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
