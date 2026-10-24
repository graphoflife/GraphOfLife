# How different are the agents?

> [!question] Questions of this page
> - Put two [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") in exactly the same place, with the same random inputs: do
>   they do the same thing?
> - How much of what an agent does comes from where it is, how much from who
>   it is, and how much from chance?
> - How many kinds of behaviour live side by side in a world, and how does that
>   compare with the number of genotypes?
> - Do the kinds last, and do they live in different places?

## In short

We put the brains of a world's agents into the same situations, with the
same random numbers, and compare what they do. In [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") they do not act alike:
nine pairs in ten stake measurably differently, in about 17 effective kinds
of staking. In [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") they mostly do: two agents stake nearly seven times more
alike than two random brains. Either way there are far fewer kinds than
genotypes, and how an agent stakes depends on where it is, often through the
rules. B2's kinds turn over within a few hundred [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") while B1's last,
and kinds sit in somewhat different places: niches, or families.

## Many genotypes, how many ways of playing?

How many different agents does a world hold? One answer counts genotypes.
Every brain carries a number, its genotype, which a copy keeps and a change
replaces ([Genotypes and the family tree](genotype.md)).

- A B1 world of 10,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") has about 1,500 agents at the end of
  its run. They carry about a thousand genotypes, 0.64 for each agent
  ([Genotypes and lineages](genotypes-and-lineages.md)).
- A B2 world of 10,000 tokens has about 3,400 agents at iteration 2,000.
  They carry about 1,600 genotypes, 0.46 for each agent
  ([Brain diversity](genotype.md#brain-diversity)). Counted by how many
  agents carry each, as an effective number, these are worth about 950
  equally common genotypes.

But a genotype is a name. A brain gets a new one whenever it changes. In B1
it may change when it is copied into a child, and after every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."). In B2 it
may change only when it is copied, into a child or into a node its owner has
won (How a brain changes). It gets the new name even when the
change changes nothing the brain does
([A caution about names](genotype.md#a-caution-about-names)).

[What the brains are like](what-the-brains-are-like.md) opened the brains of
B1's worlds. It found that selection had set **constants**, the same in every
world: never have children, spread the stake, mark every token
revolutionary. So is a world a thousand variations on one way of playing? Or
do several ways of playing live side by side?

To tell whether two people play chess differently, you would not watch them
in different games. You would show both the same positions, and compare
their moves. This page does that with brains. It takes a world at one
moment, and puts the brains of its agents into the same situations, down to
the random numbers. Then it compares what each brain decides.

## The experiment

- **Situations.** After one game, so that the [messages](messages.md "Numbers an agent writes to itself and to each neighbour every phase, which their brains read.") in flight are
  fresh, take 200 living agents at random. Record what each of them reads:
  its tokens and links, its [neighbours](the-world.md#the-network "An agent joined to another by a connection.")', and the messages it holds
  ([What a brain sees](brain-inputs.md)). The messages are part of the
  situation: they are what the agent holds at that moment. A brain also
  reads a few random numbers ([The brain](the-brain.md)). Draw those 8 times
  for each situation, and give every brain the same 8 draws.
- **Brains.** Take 400 agents at random, and put each one's brain into every
  situation and draw. A brain carried by many agents is drawn as often as it
  is carried. Beside them go 400 fresh **[founder](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")** brains, which have
  never met selection. They show what brains do by chance.
- **Decisions.** The [engine](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.")'s own rules turn each brain's outputs into
  decisions (`research/oee/checks/diversity.py`):

| decision | what it is |
|---|---|
| the stake | the shares of its tokens the agent puts on its own node and on each neighbour's |
| kept at home | the share on its own node ([Staking at home](how-the-game-is-played.md#staking-at-home-and-keeping-ones-node)) |
| given to a child | the share it would give a child; 0 when no child would be made |
| spread or all in | 1 if it spreads its stake over its [candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides."), 0 if it puts it all on one |
| a newborn's links | how likely a child of its own is to be linked to each of its neighbours |
| marked revolutionary | the share of its stakes on others marked revolutionary ([How a coalition takes a node](revolution.md)) |

The worlds are all of 10,000 tokens:

- **B2**: the ten worlds of Experiment 16, from the archives they keep every
  100 iterations, at 0, 100, 500, 1,000 and 2,000 iterations;
- **B1**: the first ten worlds of Experiment 2, at the end of their 3,000
  iterations;
- **the long runs of B1**: two [pilot worlds](pilot-worlds.md) of the same
  settings, from their start to 10,000 iterations;
- **the long runs of B2**: three of B2's ten worlds, played on from 2,000 to
  10,000.

## Where, who and chance

Why does one agent decide differently from another? There are four possible
reasons:

- **where it is**: every brain does the same in the same place, and agents
  differ because their places differ;
- **who it is**: some brains do more of it than others wherever they are: a
  trait, like a temperament;
- **who it is, there**: brains differ in how they answer a place;
- **chance**: the random inputs decide.

With every brain in every situation and draw, the variance of a decision
splits exactly into these four parts.
[Where, who and chance](variance-split.md) shows how each part is computed.

A small example, made up to show the idea. Two brains, A and B, decide what
share of their tokens to keep at home. Each is put in two places, an agent
with one token and an agent with ten, and each place gets two throws of the
dice:

| | one token, throw 1 | one token, throw 2 | ten tokens, throw 1 | ten tokens, throw 2 |
|---|---|---|---|---|
| brain A | 1 | 1 | 0.3 | 0.5 |
| brain B | 1 | 1 | 0.7 | 0.9 |

With one token, both brains keep everything. With ten, A keeps about 0.4 and
B about 0.8, and each throw moves the share by 0.1. So the variance splits
like this:

- **where**: the two places differ, 1 against 0.6 on average: 62% of the
  variance;
- **who**: B keeps more than A, 0.9 against 0.7 on average: 15%;
- **who, there**: the brains differ only in the place with ten tokens: 15%
  too;
- **chance**: the throws: 8%.

"Do agents behave the same in the same circumstance?" asks whether the two
middle parts are small.

<!-- figure diversity/split -->
![Where, who and chance](../figures/diversity/split.svg)

**Where, who and chance.** How the variance of each decision splits: into where the agent is (the situation), who it is (its brain, wherever it is), who it is in that situation (how its brain answers that situation), and chance (its random inputs). The median over ten worlds of each part. A decision driven by where agents are is one they all take alike; one driven by who they are is a trait that differs between them.

> [!example]- How to make this figure
> **Runs.** The ten B2 worlds of Experiment 16 at iteration 2,000 (their archives) and the first ten B1 worlds of Experiment 2 at the end of their 3,000 iterations.
>
> **Data.** From `GraphOfLifeRuns/.pilots/diversity/B2.json` and `B1.json` (`research/oee/checks/diversity.py`).
>
> 1. Load the world and play one game, so its messages are fresh (`research/oee/checks/diversity.py`).
> 2. Take what 200 living agents read, and draw their random input rows 8 times, the same draws for every brain.
> 3. Put the brains of 400 agents drawn at random, and 400 fresh founder brains, into every situation and draw, and turn their outputs into decisions by the engine's rules.
> 4. Split each decision's variance over brains, situations and draws into the four parts (the note *Where, who and chance*); take the median over worlds.
>
> **To make it again:** `python3 book_figures.py diversity`.
<!-- /figure -->

**Where an agent stakes depends on where it is, and on how its brain answers
that place — hardly on a fixed leaning.** In B2 the situation makes 46% of
the variance of the share kept at home. How a brain answers the situation
makes 23%, and chance 29%. The brain on its own, wherever it is, makes 2%. In
B1 the situation makes 71%.

Where an agent is reaches its decisions in two ways. One is what its brain
reads. The other is the rules, which turn the same outputs into different
stakes for different agents. The split cannot tell the two apart, and here
much of it is the rules.

- **In B2**, about half the agents hold a single token (49% and 57% in two
  of the worlds at 2,000 iterations). B2's rule
  [keep_home](settings.md#keep_home) keeps 30% of an agent's tokens at home,
  rounded up. So a single token stays at home, whatever the brain says. The
  other agents nearly all go **all in** (spreading in only 0.1–4% of cases).
  They put what is not kept on whichever candidate scores highest: their own
  node about two times in three, a neighbour's the third time. Among the
  agents with two tokens or more, the two shares were 63–76% and 21–37% in
  three of the worlds at 2,000 iterations. When scores lie close together,
  the random numbers decide which candidate wins. That is why chance is a
  larger part in B2 than among founders (29% against 17%).
- **In B1** nearly every brain spreads its stake, in 94–99% of cases in the
  ten worlds. Most split it evenly, or nearly evenly
  ([What the brains are like](what-the-brains-are-like.md)). An
  [even split](even-split.md "What the rules do when a brain scores no candidate above zero: the stake is spread evenly.") puts 1/(*k* + 1) of the stake at home for an agent
  with *k* neighbours. That again is where the agent is, not what its brain
  makes of it. It fits the finding of
  [What the brains are like](what-the-brains-are-like.md) that B1's brains are
  nearly blind: the situation reaches B1's stakes mostly through the rules.

**Whether a child is born depends on the place in B2, on the brain in B1.**
In B2, 82% of the variance of a child's share is the situation. B2's rules
refuse a birth to a parent already at its [valence](settings.md#valence) of 8
links. They also refuse one whose child would get fewer than
[10 tokens](settings.md#min_child). Those are facts about where the parent
is. B1 has neither rule, and there 92% is who the agent is. Almost every B1
brain would give a child nothing: the mean share is 0.2–2.7% of its tokens
in the ten worlds. The few that would give more are particular brains. This
is the near abolition of reproduction that
[What the brains are like](what-the-brains-are-like.md) found.

**Two decisions are pure traits, and were so before any selection.** Whether
a brain spreads or goes all in depends almost only on who it is. In B2, 42%
of its variance is *who it is* and 44% *who it is, there*; in B1, 100% is
*who it is*. So does how readily it links a newborn to its neighbours: 89%
*who it is* in B2, and 100% in B1. Founders are no different: 87% and 90%
*who it is*. Each brain has its own answer, and keeps it in nearly every
situation. That is how these decisions are built, not something selection
made.

What selection did was move the answers: B1's brains to spreading, B2's to
going all in. The link stayed a trait on which agents differ widely. Some
brains would link a newborn to nearly every neighbour, others to almost
none. The standard deviation between agents, on a scale from 0 to 1, is
0.34–0.49. That holds in all ten B2 worlds, and in eight of the ten B1
worlds. In the other two B1 worlds it is 0.32, and 0.14 in one where nearly
every brain links a newborn to nearly every neighbour.

## Two agents in the same place

The split looks at decisions one at a time. But the stake, which every agent
makes in every game, is a whole list: a share for its own node and one for
each neighbour. How far apart are two such lists?

Take two agents in the same situation, with the same random numbers. Their
distance is the **total variation**: the share of its tokens one agent would
have to move to stake like the other. It is 0 when they stake alike, and 1
when they stake on no candidate in common. An example: of ten tokens, A
stakes 6 at home and 4 on neighbour X. B stakes 3 at home, 3 on X and 4 on
Y. To stake like A, B would move 3 tokens from Y to home and 1 from Y to X.
That is 4 of its 10: a total variation of 0.4.

Set that beside how far one brain is from *itself* when only the random
numbers change: its own noise.

<!-- figure diversity/life -->
![Through a B2 world's life](../figures/diversity/life.svg)

**Through a B2 world's life.** Left: in the same situation and with the same random inputs, how far apart two agents' stakes are (the total variation: the share of tokens one would have to move to stake like the other), against how far one brain is from itself with other random inputs, and against two fresh founders. Right: how many kinds of staking the agents fall into, counted by the agents in each (an effective number), beside the same for founders. Medians of ten worlds.

> [!example]- How to make this figure
> **Runs.** The ten B2 worlds of Experiment 16, from their archives at iterations 0, 100, 500, 1,000 and 2,000.
>
> **Data.** From `GraphOfLifeRuns/.pilots/diversity/B2.json` (`research/oee/checks/diversity.py`).
>
> 1. Load the world and play one game, so its messages are fresh (`research/oee/checks/diversity.py`).
> 2. Take what 200 living agents read, and draw their random input rows 8 times, the same draws for every brain.
> 3. Put the brains of 400 agents drawn at random, and 400 fresh founder brains, into every situation and draw, and turn their outputs into decisions by the engine's rules.
> 4. Two brains are of one kind while their stakes differ by no more than 95% of brains differ from themselves between two halves of the draws; count the kinds by their carriers (1 / sum of squared shares).
>
> **To make it again:** `python3 book_figures.py diversity`.
<!-- /figure -->

In B2 (the left panel):

- At the start two founders differ by 0.46, and each from itself by 0.26
  (the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") of ten worlds).
- By 2,000 iterations two agents differ by 0.19, and one from itself by 0.13.
  The evolved brains are steadier: their stakes depend less on the random
  numbers.
- Two fresh founders put into the situations of iteration 2,000 differ by
  just as much, 0.19. So the fall from 0.46 comes from the situations as much
  as from the brains. **B2's agents are no more alike than brains drawn at
  random.**

Every one of these numbers is an average over situations. In B2 about half
of the situations are agents with a single token. There every brain stakes
alike, because the rules leave nothing to choose. All the differences come
from the other half.

In B1 two agents differ by 0.075 (the median of ten worlds; from themselves
by 0.035), and two founders in the same situations by 0.51. **B1's agents are
nearly seven times more alike than random brains.**

## Kinds of behaviour

When do two agents behave differently, rather than by chance? Think of two
kitchen scales. Weigh the same apple twice on one scale, and you get two
slightly different numbers. You would call the two scales different only if
they disagree by more than one scale disagrees with itself.

Brains are compared the same way. Each brain's stakes in the 200 situations
are averaged over two halves of the draws, four draws each. The distance
between its two halves is its own noise. Two brains count as acting
differently when they are further apart than 95% of brains are from
themselves. This is a test at the usual five per cent. Brains are then
grouped into **kinds**. The closest groups are merged, as long as they are
no further apart than that line.

How many kinds are there? A plain count would weigh a kind of one agent as
much as a kind of a thousand. So the kinds are counted by their carriers, as
an **effective number**: how many equally common kinds would give the same
spread ([Effective numbers](effective-number.md)). If *p* is the share of the
agents in each kind, the effective number is 1 / Σ *p*². That is one over
the chance that two agents drawn at random are of the same kind.

An example. Ten agents in two kinds of five each count 1 / (0.5² + 0.5²) = 2.
Ten agents in three kinds, of eight, one and one, count
1 / (0.8² + 0.1² + 0.1²) = 1 / 0.66 ≈ 1.5. That is three kinds, but as
varied as one and a half equally common ones. Ten equally common kinds count
10, and one kind with nearly everyone counts close to 1.

**The test checks itself.** [What the brains are like](what-the-brains-are-like.md) found that many
B1 agents score every candidate zero or below. The rules then split their
stake exactly evenly
([Splitting tokens into whole numbers](largest-remainder.md)). Such agents
all stake alike, and should come out as one kind. In nine of the ten B1
worlds, the largest kind of staking holds just the share of such agents, to
within two hundredths. That share is 4% of the agents in one world, and 90%
in another. In the tenth, the largest kind (94%) also takes in agents whose
positive scores are so close that they split nearly evenly too.

At the start, the founders fall into about 4 kinds of staking (the right
panel of *Through a B2 world's life*, above). About half of them spread their
stake, almost evenly, and in every world these make one kind. The other half
go all in, and differ. In B2:

- after 100 iterations the agents fall into about 22 kinds of staking;
  after 2,000, into 17.5 (from 1.6 to 42.5 in the ten worlds);
- nine pairs of agents in ten stake measurably differently (0.90–0.95 at
  every moment after the start);
- counted over every decision, not only the stake, there are about 54 kinds
  (from 6 to 114).

In B1 the median world has 3.2 kinds of staking (from 1.1 to 57), and 9 over
every decision. In four worlds of ten, one kind, built around the even split,
holds 73–94% of the agents.

So B2's agents differ as widely as random brains. But unlike random brains,
their differences are real: they are steady enough to tell apart. They share
one habit, all in and at home two times in three, and differ in *when* and
*on whom* they go all in.

The screen of random worlds uses this count. A world it calls diverse has
more effective kinds of staking than B2's median world, and a higher
diversity score ([What makes a world interesting](what-makes-a-world-interesting.md)).
The screen reads the kinds from the world as kept, with no game played first.
And it averages two moments, since the count swings from one moment to the
next ([The diversity score](diversity-score.md)).

## Genotypes and kinds

<!-- figure diversity/kinds -->
![Genotypes and kinds of behaviour](../figures/diversity/kinds.svg)

**Genotypes and kinds of behaviour.** Each dot is a world at one moment: across, how many genotypes it holds, counted by their carriers; up, how many kinds of staking its agents fall into, counted the same way. Far more genotypes than kinds means most genetic differences change nothing an agent does.

> [!example]- How to make this figure
> **Runs.** Every world measured: B2 over its life, B1 at its end, the long pilot worlds.
>
> **Data.** From `GraphOfLifeRuns/.pilots/diversity/*.json` (`research/oee/checks/diversity.py`).
>
> 1. Load the world and play one game, so its messages are fresh (`research/oee/checks/diversity.py`).
> 2. Take what 200 living agents read, and draw their random input rows 8 times, the same draws for every brain.
> 3. Put the brains of 400 agents drawn at random, and 400 fresh founder brains, into every situation and draw, and turn their outputs into decisions by the engine's rules.
>
> **To make it again:** `python3 book_figures.py diversity`.
<!-- /figure -->

A B2 world of 3,400 agents holds about 950 effective genotypes and 17 kinds
of staking, or 54 over every decision. So most genetic differences change
nothing an agent does. And the number of kinds hardly follows the number of
genotypes. Worlds with about a thousand effective genotypes hold anything
from 1.6 to 43 kinds of staking.

Both counts here are effective numbers of the same kind, 1 / Σ *p*². The
note on [Brain diversity](genotype.md#brain-diversity) counts genotypes by
2^*H* instead, which weighs rare genotypes more and so gives a larger number
([Two effective numbers](effective-number.md#two-effective-numbers)).

The long runs add time:

- **B1's first long world converges.** At 5,000 iterations one kind of
  staking holds 92% of its agents, and its genotypes have fallen to 27
  effective. At 10,000 one kind holds 96%.
- **B1's second long world goes the other way**, to 12 kinds at 10,000.
- **B2's three long worlds** keep 4–19 kinds of staking, and 35–75 over every
  decision, at 5,000 and at 10,000 iterations. B2 does not converge on one
  way of playing.

## Do kinds last?

A kind of one moment lasts if, at a later moment, some agent still behaves
that way. Take 200 agents from each of two moments of one world. Put all
their brains into the situations of the first moment, and group them into
kinds together. These kinds are over the stake and every other decision,
which makes them finer than kinds of staking alone. Then ask: what share of
the first moment's agents belong to a kind that is found among the second
moment's 200?

A sample of 200 agents can miss a rare kind. So even a world that had not
changed at all would not score 1. The **ceiling** is the same moment sampled
twice. In B2 at 1,000 iterations, 0.68 of the agents' kinds are found again
(the median of ten worlds; 0.56–0.75).

<!-- figure diversity/persist -->
![Do kinds of behaviour last?](../figures/diversity/persist.svg)

**Do kinds of behaviour last?** For two moments of the same world: 200 agents drawn from each, their brains put into the situations of the first and grouped into kinds together, over the stake and every other decision. The share of the first moment's agents whose kind is found among the second's, +N iterations later. The *ceiling* draws the second 200 from the same world at the same moment (at iteration 1,000 in Experiment 16's worlds, 5,000 in the long runs): it is as much as a sample of 200 can find, since a rare kind can be missed. Each dot is a world.

> [!example]- How to make this figure
> **Runs.** The ten B2 worlds of Experiment 16 at 1,000 (against themselves, and 100 iterations later), 500→1,000 and 1,000→2,000; three of them played on as pilots, at 5,000 against themselves, 2,000→5,000 and 5,000→10,000; and B1's two long pilot worlds, at 5,000 against themselves, 2,000→5,000 and 5,000→10,000.
>
> **Data.** From `GraphOfLifeRuns/.pilots/diversity/persist.json` and `persist_gaps.json` (`research/oee/checks/diversity.py --persist`).
>
> 1. Load the world and play one game, so its messages are fresh (`research/oee/checks/diversity.py`).
> 2. Take what 200 living agents read, and draw their random input rows 8 times, the same draws for every brain.
> 3. Draw 200 agents from each moment and put their brains into the first moment's situations and draws.
> 4. Group the brains of both moments together, over the stake and the five decisions; ask which kinds of the first are found in the second.
>
> **To make it again:** `python3 book_figures.py diversity`.
<!-- /figure -->

In B2 the share found falls fast, then slowly:

- **100 iterations later**, 0.49 are found: seven tenths of the ceiling.
  B2's agents live at most 100 iterations, so by then every agent of the
  first moment has died. The kinds found again are carried by new agents.
- **500 iterations later**, 0.25, and 1,000 later 0.22: about a third of the
  ceiling.
- **In the long runs**, 3,000 and 5,000 iterations later, 0.145 and 0.17,
  against a ceiling of 0.63 at 5,000: about a quarter.

So most of B2's kinds are gone within a few hundred iterations, and others
take their place. The number of kinds stays about where it was. A core lasts
far longer: about a quarter of what a sample can find is still there 5,000
iterations later.

B1's long worlds keep their kinds. Over the 3,000 iterations from 2,000 to
5,000, the kinds of 0.47 of the agents are found again. Over the 5,000 from
5,000 to 10,000, it is 0.82, against a ceiling of 0.90. B1 has few kinds, and
a sample of 200 finds nearly all of them. The first of the two worlds is the
extreme: from 5,000 iterations on, one kind of staking holds more than nine
tenths of its agents.

## Do kinds live in different places?

Suppose each kind were suited to its own kind of place, as a cactus is to a
desert. Then the agents of one kind would sit in similar places. To test
this, ask how much of the variance of the carriers' tokens and links their
kind accounts for. That share is η²: the variance between the kinds' means,
as a share of all the variance. Suppose every agent of one kind had 3 links,
and every agent of another had 9. Then the kind would account for all of it:
η² = 1. If both kinds had the same mix of links, it would account for none:
η² = 0.

Kinds drawn at random account for some variance by chance alone. So η² is
compared with kinds shuffled among the carriers: groups of the same sizes,
drawn at random.

- **B2** after 2,000 iterations: the kind accounts for 31% of the variance of
  its carriers' number of links, against 16% for shuffled kinds. That is more
  than all 200 shuffles in every one of the ten worlds. It accounts for 22% of
  their tokens, against 16%. Both are counted on a logarithmic scale.
- **B1**: 31% of the tokens, against 16%; but of the links, 15% against 16%:
  no more than chance.

Kinds do sit in somewhat different places: in B2 by how many links their
carriers have, in B1 by how rich they are. That fits niches. But it fits
kinship as well. Relatives carry similar brains and live near one another
([Like next to like](like-next-to-like.md)). So a kind may share a place
because it is a family, not because the place suits it. Telling the two
apart needs an experiment: move a kind to another place, and see whether it
does worse there.

## What we learned

- **In the same place, with the same dice, B2's agents do not act alike.**
  Nine pairs in ten stake measurably differently. There are about 17
  effective kinds of staking, and about 54 over every decision, in worlds of
  about 950 effective genotypes.
- **B1's agents mostly do.** They are nearly seven times more alike than
  random brains. In four worlds of ten, one kind, built around the even
  split, holds 73–94% of the agents.
- **How an agent stakes depends on where it is, hardly on a fixed leaning.**
  The brain on its own makes only 2–3% of the variance of the share kept at
  home. The place, how a brain answers it, and chance make the rest. In B2
  the place also decides a child's share, through its rules. Spreading and
  linking a newborn are traits of each brain, and were before selection
  began. In B1 a child's share is a trait too.
- **Genotypes overstate the variety of behaviour.** B2's worlds hold about
  950 effective genotypes, but 17 kinds of staking (54 over every decision).
  Across worlds, the number of genotypes does not predict the number of
  kinds.
- **B2's kinds turn over; B1's last.** Within 500 iterations, two thirds of
  the kinds a sample of B2 can find are gone, and new ones have taken their
  place. A core lasts thousands of iterations. B1's few kinds are still there
  5,000 iterations later.
- **Kinds sit in somewhat different places**, by links in B2 and by tokens in
  B1: niches or families, not yet told apart.
- **For [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau."), B2 has the raw material and B1 has not**:
  many kinds of behaviour at once, steadily renewed. What B2 lacks is a
  reason for a kind to *stay*: a place where it does better than the others.
  Then new kinds would be added to the old ones, rather than replacing them.
  Whether the turnover is selection or drift is open. Contests as long as an
  agent's life say that later brains beat earlier ones in these worlds
  ([Would we notice if it went on?](how-to-tell-a-world-that-keeps-evolving.md#would-we-notice-if-it-went-on)).
  Those of iteration 2,000 hold 0.73 of a world shared with those of 500. So
  at least some of it is selection. The next measure is to play each kind
  against the kinds that replaced it.

## Where to go next

- [How alike the agents are](how-alike-the-agents-are.md): the same measure
  in every kind of world, and the rule that makes B2's agents as alike as
  B1's.
- [What the brains are like](what-the-brains-are-like.md): why B1's brains
  are nearly blind, and the constants selection set in them.
- [Genotypes and lineages](genotypes-and-lineages.md): how long a genotype
  lasts, and how lines of descent take worlds over.
- [Where, who and chance](variance-split.md) and
  [Effective numbers](effective-number.md): the two methods of this page.

## How this was made

The figures read the measurements of `research/oee/checks/diversity.py`,
kept in `GraphOfLifeRuns/.pilots/diversity/`. Each figure says, folded
beneath it, which worlds it reads and how. `python3 book_figures.py diversity`
draws them all, and writes the numbers they quote to
`book/results/diversity.json`.

Nothing in the measure is particular to B1 or B2. It needs only a world at
one moment: a lab run's checkpoint or archive, or a pilot's. The first
command below measures a lab run's world at iteration 1,000, and the second
a pilot's world. The third asks which kinds of one moment are still found at
a later one.

```
python3 research/oee/checks/diversity.py B2-10000-s001:1000 --label mine
python3 research/oee/checks/diversity.py pilot:GraphOfLifeRuns/.pilots/Q/mix-s1:2000 --label mine
python3 research/oee/checks/diversity.py --persist B2-10000-s001:1000 B2-10000-s001:2000 --label mine
```

Two checks come with it. A world without random inputs (Experiment 16's
second condition) must give chance a share of exactly 0, and does. And a kind
the rules make, the even split, must come out as one kind, and does.
[How alike the agents are](how-alike-the-agents-are.md) applies the measure
to every condition of every experiment, and to the pilot worlds.

## How our view changed

- **24 October 2026.** From 16 October, this page opened with "A world of
  3,000 agents holds about a thousand genotypes". Neither [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") fits those
  words. A B1 world of 10,000 tokens holds about a thousand genotypes, but it
  has about 1,500 agents. A B2 world of 10,000 tokens has about 3,400 agents
  and about 1,600 genotypes, 0.46 for each agent. Only as an effective number
  are B2's genotypes about a thousand (950). The page now names the baseline
  of each count. The counts are those of the worlds measured here
  (`GraphOfLifeRuns/.pilots/diversity/B1.json` and `B2.json`), and the
  genotypes per agent, 0.64 and 0.46, are in `book/results/how_alike.json`.
- **24 October 2026.** From 16 October, the page's findings said that where
  an agent is decides more than who it is "for most decisions in both rule
  sets". The page's own split shows that this holds in B2, for the stake, a
  child's share and the revolutionary mark. In B1 it holds only for the
  stake. There a child's share is 92% who the agent is, and spreading and a
  newborn's links are 100%. The revolutionary mark is 45% who against 36%
  where (`book/results/diversity.json`).
- **24 October 2026.** From 16 October, this page gave the standard deviation
  of a newborn's links between agents as 0.34–0.49, as if for every world. It
  holds in all ten B2 worlds and in eight of the ten B1 worlds. In the other
  two it is 0.32 and 0.14 (`decisions.link.sd_between_agents` in
  `GraphOfLifeRuns/.pilots/diversity/B1.json`).

<!-- turns -->
---

← [Do the brains matter?](do-the-brains-matter.md) · [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are) · [Genotypes, families and the common ancestor](genotype.md) →

> [!info]- Pages that link here (15)
> [Glossary](glossary.md) · [The even split](even-split.md) · [Flow modules: the groups tokens stay in](flow-modules.md) · [How alike the agents are](how-alike-the-agents-are.md) · [Genotypes, families and the common ancestor](genotype.md) · [What would count as open-ended evolution](open-ended-test.md) · [Healthy and rich](healthy-and-rich.md) · [What makes a world interesting](what-makes-a-world-interesting.md) · [The diversity score](diversity-score.md) · [Brains mixed at birth, in long worlds](mixed-at-birth-ten-long-worlds.md) · [A screen of random worlds](a-screen-of-random-worlds.md) · [Every world side by side](every-world-side-by-side.md) · [Every simulation: what was run, and what was measured](every-simulation.md) · [Where, who and chance: splitting a decision's variance](variance-split.md) · [Effective numbers: how many kinds there really are](effective-number.md)
<!-- /turns -->
