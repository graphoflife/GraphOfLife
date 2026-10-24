# What holds a world's size

> [!question] Questions of this page
> - Why are [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s worlds so poor, and why were the mixed worlds rich — and
>   mortal?
> - What sets how many [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") a world holds, and what would hold a world at a
>   size where its agents are neither all poor nor in danger of dying out?

## In short

Every agent of a B2 world dies at the age of 100, so a world keeps its size
only if one agent in a hundred is born each [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). How eagerly the agents
breed decides how full the world gets; with its [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") fixed, a full world is
a poor one. B2's agents breed as soon as they can, and three in four are poor:
they hold three tokens or fewer. When a newborn's [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") is mixed from its
parent's and a [neighbour](the-world.md#the-network "An agent joined to another by a connection.")'s, agents put children off: their worlds are small
and rich, but the reluctance drifts, and some die out. No rule tried so far
holds a world both rich and steady; the steadiest kept every world alive, but
at about 2,000 agents, six in ten of them poor.

## One number behind riches and health

The author's picture of a world that might keep evolving has two halves
([Healthy and rich](healthy-and-rich.md)). Its network is decentralised,
resilient and almost three-dimensional. And its agents differ: it is not a
world where everyone holds one to three tokens.

Across the experiments with brains mixed at birth
([below](#three-kinds-of-world)), one quantity decided both halves more than
any rule did: **how many agents a world holds**.

**Riches are size.** The number of tokens in a world never changes. So the
fewer its agents, the more each one holds: 10,000 tokens among 3,300 agents
are three each, and among 700 fourteen.

**Health follows size**, in B2's networks
([The marks a world misses](mixed-at-birth-ten-long-worlds.md#the-marks-a-world-misses)).
The [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4.") asks seven things of a network, its marks. A small world fails
the mark for [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub."): below about 300 agents, its best-connected agent is joined
to more than 5% of the others. A large world fails the mark for [dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space."),
which asks for 2.3 to 4: above about 1,500 agents, it reads near 4 or more.

The large world fails because B2's kind of network is nearly a small world, in
which every agent is a few steps from every other. Every B2 world starts as a
space, wide like a plane or a room. Then the deaths of its old agents fold it:
each death is mended with a ring of links among the dead agent's neighbours.
And a small world reads a larger dimension the more room it has
([What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world)).

**A world that grows too small dies out.**

So this page gathers what the notebook knows about what sets the number of
agents in a world.

## The ledger of a world

Picture a bathtub with the plug pulled. The fuller it is, the faster it
drains. Now picture a tap that pours more slowly the fuller the tub gets. The
water settles at the level where the tap and the drain match. A world built on
B2's rules is such a tub, with agents for water.

**The drain is old age.** In every world built on B2's rules, an agent lives
out its hundred iterations and then dies of age; nearly nobody dies sooner
([Why a mixed world is rich](mixed-at-birth-ten-long-worlds.md#why-a-mixed-world-is-rich)).
So a world of 1,000 agents loses about 10 of them every iteration, and a world
of 3,000 about 30. To keep its size, a world needs one birth per hundred agents
every iteration.

And it gets one. The notebook measured 454 moments: in B2's worlds, in the
worlds whose brains are mixed at birth, and in worlds where that rule was
switched on or off. The [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") is 1.00 births per hundred agents an iteration. Worlds
read later show the same balance: 99 to 102 agents for each birth an iteration,
averaged over iterations 2,000 to 3,000
([A world's size is its births](room-for-the-steadiest-world.md#a-worlds-size-is-its-births)).
A world's size is a hundred times its births an iteration.

**The tap is the agents that can have a child.** A child takes at least 10
tokens, and its parent keeps one, so a parent needs 11 tokens or more. And an
agent with 8 links, the most its brain may hold, cannot have a child at all.
We call an agent **able** when it holds 11 tokens or more and has fewer than
eight links. And we call how often the able have a child their **eagerness**:
the births per hundred able agents an iteration.

The births a world sees are then its able agents, times how eagerly each of
them breeds:

> births per hundred agents = (share of agents able to have a child) ×
> (births per hundred able agents)

The left side has to come to one. In B2's worlds, a median of 2.3% of the agents
are able, and they have a child at a median rate of 42 per hundred an
iteration: 0.023 × 42 ≈ 1. In the worlds whose brains are mixed at birth, a
third are able, and they breed at 3 per hundred: ⅓ × 3 = 1. The births are the
same; the two factors are not.

The share of able agents falls as a world fills up, because more agents share
the same tokens. That is the tap slowing as the tub fills. So every world
settles where its share of able agents brings the product to one: **eager
agents settle a world where few are able, and reluctant ones where many
are.** Eagerness sets the size, and the size sets the riches.

## Three kinds of world

**Eager: B2.** An agent that can afford a child has one at a median rate of 42
per hundred an iteration, so within two or three iterations. The world fills
until only about 2% of its agents can afford one, and three agents in four
hold three tokens or fewer. Nothing drifts. The world is held at that size by
its tokens, as a jar is filled by what is poured in.

**Reluctant: brains mixed at birth.** With brains mixed at birth
([`mix_at_birth`](settings.md#mix_at_birth)), a newborn's brain is half its
parent's and half a **mate**'s: one of the parent's neighbours, drawn at random
([Brains mixed at birth, in long worlds](mixed-at-birth-ten-long-worlds.md)).
The able agents of these worlds have a child at 3 per hundred an iteration. So
a world settles where a third of its agents can afford one: small and rich.

But nothing holds the reluctance where it is. It drifts as the brains change,
and the size follows. Across the 181 moments kept of Experiment 17's ten mixed
worlds, eagerness and size go together closely: a [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of 0.87 between
their logarithms. The marks of health follow the size.

And the eagerness can fall too far. Suppose every agent were able: the share
in the ledger would be one, and the births per hundred agents would equal the
eagerness. So once the able have fewer than one child per hundred an
iteration, even a world in which everyone can afford a child does not replace
its dead. The world dies.

Three of the five mixed worlds played to 50,000 iterations died so
(Experiment 18,
[Five mixed worlds, fifty thousand iterations](mixed-at-birth-ten-long-worlds.md#five-mixed-worlds-fifty-thousand-iterations)).
In the last world kept before each end, a third to three quarters of the
agents could have had a child. Yet births had fallen to 0.4 to 0.5 per hundred
agents an iteration, against the 1.0 that replaces the dead.

**Paid: rules that pay for births.** Some rules make every birth easier or
more rewarding: the [estate](estate.md "The tokens a dead agent leaves, and who gets them.") of the dead paid to the newborns or to their
parents, or a child allowed to start with any number of tokens. Their worlds
fill to more agents than B2's, and are poorer still. With the estate to the
newborns or to the parents, B2's worlds grew by half, to 5,100 to 5,300
agents, and nine agents in ten held three tokens or fewer
([Who inherits?](who-inherits.md)). Mixing brains at birth does not change
that ([Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md)).
Nor do parents as mates: with them, the estates filled a world to about 5,000
agents, of 1.9 tokens each
([Room for the steadiest world](room-for-the-steadiest-world.md#room-for-the-steadiest-world)).

**Switching a rule shows the ledger at work.** Batch SW, a group of [pilot](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.")
worlds, took worlds from the [lab](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") at a moment and played them on with mixing
switched
([Why a mixed world is rich](mixed-at-birth-ten-long-worlds.md#why-a-mixed-world-is-rich)):
- **Mixing switched off** in five rich worlds: within 1,500 iterations,
  eagerness rose three- to twentyfold in four of them, and all five filled up
  towards B2's size.
- **Mixing switched on** in five of B2's worlds: eagerness fell three- to
  tenfold, and the worlds shrank, one of them to 81 agents.

## Why mixing makes agents reluctant

A brain spreads in two ways. One is through its agent's children. The other is
through the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."): each node, a place in the network, that an agent wins there
gets a copy of the winner's brain. Taking nodes is by far the larger way. In
the median iteration of a mixed world, 7.7 nodes per hundred agents change
hands, against one birth per hundred agents.

In B2 a child carries all of its parent's brain. With mixing it carries about
half, and the other half comes from the mate, who pays nothing for it. So
having a child spreads half as much of a brain for the same price. And a
mate's brain spreads for free, whether its own agent breeds or not. We call
this the **free ride**.

Selection keeps working in the mixed worlds. Later brains beat earlier ones,
as in B2, and by more at first
([Does evolution go on?](mixed-at-birth-ten-long-worlds.md#does-evolution-go-on)).
But what selection rewards has changed, and agents that keep their tokens lose
little by it.

It can even reward the reluctance that ends a world. To see whether brains
still improve, the notebook plays a **lifetime contest**. The brains of two
moments of a world, two [eras](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?"), are seated on alternate agents and played for
100 iterations, an agent's [lifetime](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death."). If the later brains then hold more than
half of the world, they are the better ones.

In one of the worlds that died, the brains of its last good years beat those
of 5,000 iterations before, 86% to 14%. The brains that let the world die out
were winners
([Five mixed worlds, fifty thousand iterations](mixed-at-birth-ten-long-worlds.md#five-mixed-worlds-fifty-thousand-iterations)).
Biologists call this **evolutionary suicide**: selection driving a trait to
where the population cannot last. In another world, the last brains lost to
their forebears; in a third, they tied. One world in three is not a law, but
it is a warning.

Was the free ride the cause? Two later experiments took it away, in two
different ways, and both steadied mixed worlds
([Who may be a mate](mixed-at-birth-with-parents-as-mates.md)). So it was at
least one cause of the reluctance.

## What would hold a world's size

Think of a thermostat. When the room is cold it heats, and when the room is
warm it stops, so the room stays near one temperature. A world held at a
middle size needs the same for births: **having a child must pay more when few
have them, and less when many do.** We call that a **pull**.

Such a world would be rich enough that its agents differ, and large enough not
to die out. In B2's networks it could also sit at the size where every mark of
the health bar is met. Among Experiment 17's mixed worlds, those of 600 to
1,000 agents met every mark at 98% of their moments.

B2 has a pull, but only at the top, where the tokens run out. The rules tried
so far either pull too hard or not at all. An estate shared among the newborns
pulls too hard: it pays each newborn more when there are fewer of them, and so
it fills the world.

The pilot worlds of batch M gave a mixed world, one at a time, each rule of
the engine that might steady it, for 5,000 iterations:
- the estate shared out to the poorest first, and half of an agent's tokens
  kept on its own node, held their worlds steady at 1,300 to 2,300 agents, but
  six agents in ten were poor;
- the searches for links beyond an agent's neighbours, and a slower change of
  brains, left them unsteadier than mixing alone.

Five ways to build a pull have been tried. Here is each, and what came of it.

### The estate to a dead agent's own children

With [`redistribution`](settings.md#redistribution) = `children` (new on
17 October 2026), the tokens a dead agent leaves go to its own living
children. Breeding then pays, but late, at the parent's death, and only within
its own line of descent. A line that does not breed loses its wealth to
everyone else.

**With mixing it did not hold the size** (batch H: two [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world."), 5,000
iterations). One world shrank to between 170 and 750 agents. The other fell to
two agents by iteration 4,200 and died out. The engine then started it again
from one agent holding every token, which filled the world within a hundred
iterations. A late reward for a line does nothing for a world whose agents do
not breed in the first place.

### A mate that pays its share

If the mate gives the child tokens too, the free ride ends. But a child that
two pay for is cheaper to each of them, and that pushes the other way.

Experiment 22 tried it later, in four worlds of 5,000 iterations
([A mate that pays its share](mixed-at-birth-with-parents-as-mates.md#a-mate-that-pays-its-share)).
The mate gave the child as many tokens as the parent did, as far as it could
and never its last. Every world held its size more steadily than Experiment
17's world of the same seed, whose mate pays nothing, and none came near dying
out. But the cheaper child counted too. The paying mates' worlds held more
agents than the free mates' worlds of the same seeds: medians of 1,394 to
2,048, against 586 to 1,051. So each of their agents held fewer tokens.

### Mates among the parents

With [`mix_mates`](settings.md#mix_mates) = `parents` (new on 17 October
2026), only a neighbour with a living child of its own may be the mate. Being
a mate turns from a free gift into something only breeders have. A parent with
no such neighbour gives its newborn a copy of its own brain. This rule was
played twice: first in four pilot worlds, then in the lab's ten.

**The pilots** (batch MP) were four worlds, seeds 1 to 4, played for 5,000
iterations. Each was set beside Experiment 17's world of the same seed, where
any neighbour may be the mate: its *twin*
([What the pilots found](mixed-at-birth-with-parents-as-mates.md#what-the-pilots-found)).
- **More eager.** In every pilot the able agents had a child more often than
  in its twin: 3.5 to 7.5 per hundred an iteration, against 2.1 to 4.3. Each
  number is a world's median moment, read every 500 iterations.
- **Steadier.** Every pilot held its size more steadily than its twin. The
  spread of the logarithm of its number of agents, from iteration 500 to
  5,000, was 0.09 to 0.29, against 0.19 to 0.63. A spread of 0.1 means the
  size strays by about a tenth.
- **Size and riches.** The pilots held about 760 to 2,200 agents from
  iteration 500 to 5,000. At 5,000, 23% to 54% of their agents held three
  tokens or fewer; B2's worlds have 66% to 79%.
- **But** the eagerness rose by only about a half, one pilot still fell from
  2,000 agents to 900, and three of the four ended poorer than their twins.

**The experiment** (Experiment 19) played the same rule in the lab: ten
worlds, seeds 1 to 10, for 20,000 iterations
([Mixed at birth, with parents as mates](mixed-at-birth-with-parents-as-mates.md#mixed-at-birth-with-parents-as-mates)).
Every world lived, and settled: from iteration 15,000 on, each held 2,030 to
2,350 agents on average. Over the first 10,000 iterations, the able had a
child 6.0 times per hundred an iteration, twice the 3.0 with any mate. Their
agents differed as much as in mixed worlds with a free mate: a [diversity score](diversity-score.md "How much the agents of a world differ, in one number from 0 to 1: in tokens, links, ages and genotypes.")
of 0.59 in the median world, against 0.58. But eager agents fill their world.
At iteration 20,000, 55% to 66% of each world's agents held three tokens or
fewer.

So taking the free mate away holds a world's size, but at one size for every
world — larger than mixing's, smaller than B2's — and not at a rich one.

### Tokens that buy a longer life

In B2 an agent dies at 100, whatever it holds. With
[`buy_life`](settings.md#buy_life) (new on 17 October 2026), an agent past
that age lives on by paying a token an iteration into the estate of the dead.

**It gives no pull** (batch BL). With mixing, a rich agent lives on, fewer
births are needed, and the world settles smaller still: 100 to 1,100 agents,
as unsteady as mixing alone. In one world a single agent ended up holding
every token, buying its life without end.

In B2 the worlds hold fewer agents, about 1,400 and 2,450 against 3,300. A few
old agents rich enough to live on keep them small. And they keep evolving as
B2 does. In lifetime contests, the brains of iteration 1,000 take 67% and 64%
of a contested world from those of 500, and those of 2,000 take 79% and 77%
from those of 1,000. In B2 itself: 69% and 67%, then 65% and 76%.

### A child as rich as its neighbours

With [`child_floor`](settings.md#child_floor) = `neighbours`, a child must
start with at least the median holding of its parent's neighbours. Among rich
neighbours a child costs more, among poor ones less. So a birth cannot make a
pauper where the world is rich, and a world can grow only as rich as it
already is: a pull against filling up, with no number of its own.

Experiment 32 tried it with brains mixed at birth and parents as mates, alone
and beside the estates paid to births: two seeds each, for 3,000 iterations.
**It changed nothing the measures can see**, because it hardly ever bound.
Every B2 child already starts with at least 10 tokens. In a poor world a
parent's neighbours hold far less: in the worlds with an estate, the median
agent's neighbours held a median of one token. A rule that measures a world
against itself follows a poor world down
([Births as rich as the neighbours](room-for-the-steadiest-world.md#births-as-rich-as-the-neighbours)).

The same experiment found where the poor come from. In the worlds with an
estate, children start with ten tokens or more, and the game takes most of it
within twenty iterations. In plain B2 the floor is just as idle (Experiment
36): every child already starts with ten tokens there too
([The floor in plain B2 and in B1](room-for-the-steadiest-world.md#the-floor-in-plain-b2-and-in-b1)).
Whether it binds in [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it."), where a child may start with a single token,
Experiment 37 will tell once it is played.

## What we learned

- **A world's size is set by how eagerly its agents breed.** Every agent
  lives the same hundred iterations, so the births have to match the deaths:
  one per hundred agents an iteration. In these worlds, eagerness is a matter
  of what selection rewards.
- **Riches here are reluctance.** With the tokens fixed, a small world is a
  rich one. B2's agents breed as soon as they can, and fill their world until
  most are poor. Agents that put children off keep their world small and
  rich.
- **Reluctance that nothing holds can end a world.** In the mixed worlds it
  drifts, and the size, the riches and the marks of health drift with it.
  Three of five long worlds died when births fell below one per hundred
  agents; in one of them, the winning brains were those that let it die.
- **Nothing tried so far holds a world both rich and steady.** Rules that pay
  for births fill a world. Rules that keep agents alive, or reward a line
  late, let it shrink further. The estate to the poorest and half kept at home
  hold it steady at 1,300 to 2,300 agents, six in ten of them poor. With only
  parents as mates every world lived, at about 2,000 agents, again six in ten
  of them poor; a mate that pays filled its worlds further. A floor on a
  child's tokens, set by its neighbours, hardly binds in a poor world.
- **The pull is still missing:** births that pay more when few are born and
  less when many are, measured against something a poor world does not drag
  down with it.

## Where to go next

- [Brains mixed at birth, in long worlds](mixed-at-birth-ten-long-worlds.md):
  where the ledger was measured, why mixed worlds are rich, and how three of
  them died out.
- [Who may be a mate](mixed-at-birth-with-parents-as-mates.md): the two ways
  of ending the free ride, Experiments 19 and 22.
- [Room and riches for the steadiest world](room-for-the-steadiest-world.md):
  the floor on a child's tokens, and where the poor agents come from.
- [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md):
  the other side of this topic, in B1: a world of twice the tokens holds twice
  the agents.

## How our view changed

- **18 October 2026.** From 17 October, this page called the free mate the
  most promising lead: taking it away "made mixed worlds steadier, without
  making them poor". That held for the four pilot worlds of 5,000 iterations,
  not for Experiment 19's ten worlds of 20,000. All ten lived, but they settled
  at about 2,000 agents, six in ten of them holding three tokens or fewer
  ([Mixed at birth, with parents as mates](mixed-at-birth-with-parents-as-mates.md#mixed-at-birth-with-parents-as-mates)).
- **23 October 2026.** From 17 October, this page said that in the pilots of
  parents as mates (batch MP) the able agents had a child "3.5 to 6 per
  hundred an iteration, against about 3", in worlds of "760 to 2,000 agents, a
  quarter to a half of them at three tokens or fewer". The pilots' own table
  and Experiment 19's thesis, written the same day, read the same four worlds
  at 5,000 iterations. They give 3.5 to 7.5 against 2.1 to 4.3 in their
  twins, 760 to 2,200 agents, and 23% to 54% of them poor
  ([What the pilots found](mixed-at-birth-with-parents-as-mates.md#what-the-pilots-found)).
  This page's ranges match the four worlds as they stood when two of them,
  seeds 3 and 4, had played only about 2,000 of their 5,000 iterations.

<!-- turns -->
---

[Size and riches of a world](../README.md#size-and-riches-of-a-world) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) →

**Also about:** [Looking for a healthy, rich world](../README.md#looking-for-a-healthy-rich-world).

> [!info]- Pages that link here (13)
> [Building B2: a world that cannot break](a-world-that-cannot-break.md) · [The estate of the dead](estate.md) · [Where tokens go](where-tokens-go.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [Why the raiders stopped winning](does-evolution-close-the-door.md) · [Rules in company](rules-in-company.md) · [Eight switches at once](eight-switches-at-once.md) · [Healthy and rich](healthy-and-rich.md) · [Brains mixed at birth, in long worlds](mixed-at-birth-ten-long-worlds.md) · [Who may be a mate](mixed-at-birth-with-parents-as-mates.md) · [The candidates, played on and played again](the-candidates-played-on.md) · [Room and riches for the steadiest world](room-for-the-steadiest-world.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
