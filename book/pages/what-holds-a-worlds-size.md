# Meta XI · What holds a world's size

> [!question] Questions of this chapter
> - Why are B2's worlds so poor, and why were the mixed worlds rich — and
>   mortal?
> - What sets how many agents a world holds, and what would hold a world at a
>   size where its agents are neither all poor nor in danger of dying out?

The author's picture of a world that might keep evolving has two halves
([Meta X](healthy-and-rich.md)): a healthy network, and agents
that differ — not a world where everyone holds one to three tokens. Across
the last chapters, one quantity decided both halves more than any rule did:
**how many agents a world holds.** With the tokens fixed, a world's riches
are its size: 10,000 tokens among 3,300 agents are three each, among 700
fourteen. Its health follows its size too, in B2's networks
([Chapter 80](mixed-at-birth-ten-long-worlds.md#the-marks-a-world-misses)):
a small world fails the mark for hubs, a large one the dimension. And a world
that grows too small dies. This chapter gathers what the book now knows about
what sets that number.

> [!info] Added later (2026-10-17)
> Why a large world fails the dimension: B2's kind of network is nearly a
> small world, folded by the deaths of its old agents, and a small world
> reads a larger dimension the more room it has
> ([Chapter 87](what-folds-a-world.md)).

## The ledger of a world

In every world built on B2's rules, an agent lives out its hundred iterations
and dies of age; nearly nobody dies sooner
([Chapter 80](mixed-at-birth-ten-long-worlds.md#why-a-mixed-world-is-rich)).
So a world that keeps its size must have one birth per hundred agents every
iteration — and it does: in the 454 moments measured, in B2's worlds, the
mixed worlds and the worlds whose rule was switched, the median is 1.00
births per hundred agents an iteration.

Those births come from the agents that *can* have a child — 11 tokens or
more, fewer than eight links — and from how eagerly they do:

> births per hundred agents = (share of agents able to have a child) ×
> (births per hundred able agents)

The product has to come to one. Every world therefore settles where the share
of able agents makes it so: **eager agents settle a world where few are able,
and reluctant ones where many are.** And since the share of able agents falls
as a world fills up with agents sharing the same tokens, eagerness sets the
size, and the size sets the riches.

## Three kinds of world

- **Eager: B2.** An agent that can afford a child has one at a median rate of
  42 per hundred an iteration, so the world fills until only 2% can afford
  one, and three agents in four hold three tokens or fewer. Nothing drifts:
  the world is held at that size by its tokens, as a jar is filled by what is
  poured in.
- **Reluctant: brains mixed at birth.** The able have a child at 3 per hundred
  an iteration, so the world settles where a third can afford one — small and
  rich. But nothing holds the reluctance where it is: it drifts as the brains
  change, the size follows, the marks of health follow the size, and when
  the eagerness falls far enough that even a world in which everyone can
  afford a child does not replace its dead, the world dies. Three of the five
  mixed worlds played to 50,000 iterations died so
  ([Chapter 81](five-mixed-worlds-fifty-thousand-iterations.md)): in the
  last world kept before each end, a third to three quarters of the agents
  could have had a child, and births had fallen to 0.4 to 0.5 per hundred
  agents an iteration, against the 1.0 that replaces the dead.
- **Paid: rules that pay for births.** The estate of the dead to the newborns
  or to the parents, or a child for any number of tokens, make every birth
  easier or more rewarding; the worlds fill to more agents than B2's and are
  poorer still ([Chapter 73](who-inherits.md)), with mixing as without it
  ([Chapter 82](mixing-with-every-other-rule.md)).

Switching a rule shows the ledger at work. Mixing switched off in five rich
worlds, eagerness rose three- to twentyfold within 1,500 iterations in four of
them, and all five filled up towards B2's size; switched on in five of B2's
worlds, it fell three- to tenfold, and the worlds shrank — one to 81 agents
([Chapter 80](mixed-at-birth-ten-long-worlds.md#why-a-mixed-world-is-rich)).

## Why mixing makes agents reluctant

A brain spreads in two ways: through its agent's children, and through every
node its agent takes in the game, which gets a copy of the winner's brain. In
B2 a child carries all of its parent's brain. With mixing it carries about
half, and the other half comes from a neighbour — the **mate**, who pays
nothing for it. Having a child then spreads half as much of a brain for the
same price, while a mate's brain spreads for free, whether its agent breeds
or not. Selection keeps working in the mixed worlds — later brains beat
earlier ones as in B2, by more at first
([Chapter 80](mixed-at-birth-ten-long-worlds.md#does-evolution-go-on)) —
but what it rewards has changed, and agents that keep their tokens lose
little by it.

In one of the worlds that died, the brains of the world's last good years
beat those of 5,000 iterations before in a lifetime contest, 86% to 14%: the
brains that let it die out were winners
([Chapter 81](five-mixed-worlds-fifty-thousand-iterations.md)). Biologists
call this **evolutionary suicide** — selection driving a trait to where the
population cannot last. In another world, the last brains lost to their
forebears; in a third, they tied. One world in three is not a law, but it is
a warning.

## What would hold a world's size

A world held at a middle size — rich enough that its agents differ, large
enough not to die out, and, in B2's networks, at the size where every mark of
the health bar is met — needs something that pulls eagerness back: **having a
child must pay more when few have them, and less when many do.** B2 has such
a pull, but only at the top, where the tokens run out. The rules tried so far
either pull too hard — an estate shared among the newborns pays each newborn
more when there are fewer of them, and fills the world — or not at all.
Batch M added each rule of the engine that might steady a mixed world: the
estates to the poorest and half kept at home held their worlds steady at
1,300 to 2,300 agents, but six agents in ten were poor; the searches for
strangers and a slower change of brains left them unsteadier than mixing
alone.

Ways to build the pull, and the first of them tried:

1. **The estate to the dead agent's own children**
   ([`redistribution`](settings.md#redistribution) = `children`, new
   on 17 October 2026): breeding pays late, at the parent's death, and only
   within its own line — a line that does not breed loses its wealth to
   everyone else. **With mixing it did not hold the size** (batch H, two
   seeds, 5,000 iterations): one world shrank to between 170 and 750 agents,
   the other fell to two agents by iteration 4,200 and died out — the engine
   started it again from one agent holding every token, which filled the
   world within a hundred iterations. A late reward for a line does nothing
   for a world whose agents do not breed in the first place.
2. **A mate that pays its share.** If both parents give tokens, the free ride
   ends; the price of a child falls too, which pushes the other way. Not
   tried yet.
3. **Mates among the parents.** Mixing only with a neighbour that has a living
   child of its own
   ([`mix_mates`](settings.md#mix_mates) = `parents`, new on 17
   October 2026) turns being a mate from a free gift into something only
   breeders have. In four pilot worlds (batch MP) the agents that could have a
   child had one a little more eagerly than with any mate — 3.5 to 6 per
   hundred an iteration, against about 3 — and each world held its size more
   steadily than Experiment 17's world of the same seed, at 760 to 2,000
   agents, a quarter to a half of them at three tokens or fewer. Whether that
   is enough to keep a mixed world alive is
   [Experiment 19](mixed-at-birth-with-parents-as-mates.md)'s question.
4. **Tokens that buy a longer life.** In B2 an agent dies at 100 whatever it
   holds; with [`buy_life`](settings.md#buy_life) (new on 17 October
   2026) an agent past that age lives on by paying a token an iteration into
   the estate of the dead. **It gives no pull** (batch BL). With mixing, a rich
   agent lives on, fewer births are needed, and the world settles smaller
   still — 100 to 1,100 agents, as unsteady as mixing alone — and in one world
   a single agent ended up holding every token, buying its life without end.
   In B2 the worlds hold fewer agents (about 1,400 and 2,450, against 3,300),
   kept small by a few old agents rich enough to live on — and they keep
   evolving as B2 does (added later: in lifetime contests the brains of 1,000
   take 67% and 64% of a contested world from those of 500, those of 2,000
   take 79% and 77% from those of 1,000; B2: 69% and 67%, then 65% and 76%).

> [!info] Added later (18 October 2026): Experiment 19
> Parents as mates kept every mixed world alive to 20,000 iterations, but not
> rich. The ten worlds settled at about 2,000 agents, six in ten of them
> holding three tokens or fewer, their agents breeding twice as eagerly as
> with any mate and differing as much
> ([Chapter 85](mixed-at-birth-with-parents-as-mates.md)). Taking the free
> mate away holds a world's size, but at one size for every world — larger
> than mixing's, smaller than B2's — and not at a rich one: the last point
> below, that it did so "without making them poor", held for the pilots of
> 5,000 iterations, not for the ten worlds of 20,000.

## What it teaches

- **A world's size is set by how eagerly its agents breed**, and in these
  worlds that is a matter of what selection rewards: every agent lives the
  same span, and the births have to match the deaths.
- **Riches here are reluctance.** A world of agents that put children off is
  small and rich; B2's agents breed as soon as they can, and fill their world
  until most are poor.
- **Nothing tried so far holds a world both rich and steady.** Rules that pay
  for births fill a world; rules that keep agents alive or reward a line late
  let it shrink further; the estate to the poorest and half kept at home hold
  it steady at about two thousand agents, six in ten of them poor. The most
  promising lead is the free mate: taking it away made mixed worlds steadier,
  without making them poor.

<!-- turns -->
---

← [Chapter 79 · Meta X · Healthy and rich](79-meta-10-healthy-and-rich.md) · [Contents](../README.md) · [Chapter 91 · Meta XII · Rules in company](91-meta-12-rules-in-company.md) →
<!-- /turns -->
