# What a brain can compute

> [!question] Questions of this page
> - Could an evolved brain play the [raider written by hand](strategies-written-by-hand.md),
>   or only a rule written in Python?
> - How does a rule written in words become a brain's weights?
> - What can no brain of [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") compute, whatever its size?
> - What is the smallest change to the rules that would let it?

## In short

A raider written in Python took 40% and 62% of two evolved B2 worlds, with
play that evolution had never found. Yet 12 units of a B2 brain, with weights
set by hand, decide as the rule does in all but eight of 62,114 situations. So
evolution missed the raider for want of a way there, not for want of room in a
brain. But no B2 brain of any size can put news of a third [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") into a
message: it reads each [neighbour](the-world.md#the-network "An agent joined to another by a connection.")'s column on its own. One small rule would
lift that: the message an agent writes to itself becomes the largest of
everything it writes.

## Why ask

[Can a strategy written by hand beat evolution?](strategies-written-by-hand.md)
planted rules written in Python into evolved B2 worlds. Each started as a
patch of 5% of a world's agents. A plain one, the **raider**, took 40% of
world 2 and 62% of world 3 within 300 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). It died out in the other
eight of the ten worlds ([Ten worlds](strategies-written-by-hand.md#ten-worlds)).
On a second [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") it took 50% of world 2. The brains it beat had been shaped
by evolution for 2,000 iterations. So the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") has better play than
evolution found.

But the raider was a rule written in Python. It read exactly what a brain
reads, and wrote exactly what a brain writes. Yet nothing showed that a brain
could compute it. A rule in Python may combine what it reads in any way it
likes; a brain has only its own arithmetic.

That matters for the question the author put in October 2026. Are there
strategies that are ever better, with no top? And can the agents' networks
play them? If a strategy is beyond every brain a world can grow, evolution
will never find it, however long it runs. If it is within reach, evolution
failed for another reason: the way there. The question then becomes how to
open the way.

This page answers for the raider by building it: a B2 brain whose weights are
set by hand to play the raider's rule. It then proves what no B2 brain can
compute, and names the smallest change to the rules that would let one.

## What a B2 brain sees and does

When an agent decides anything, it looks at its [candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides."): itself first, then
each of its neighbours. For each candidate its brain gets one column of 66
numbers ([What a brain sees](brain-inputs.md)):
- whether the candidate is the agent itself: 1 or 0;
- the agent's [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") and [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes."), and the candidate's, on a logarithmic
  scale, as ln(1 + count);
- six [quantiles](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") of the tokens of the agent's neighbours, six of their
  connections, and the same for the candidate's neighbours;
- four messages of eight numbers each: the agent's to itself and to the
  candidate, and the candidate's to the agent and to itself;
- five random numbers.

A B2 brain is small. One hidden layer of 16 units reads the column, and 19
outputs read the units ([What a brain says](brain-outputs.md)). The same
weights score every column. They are kept in half precision (float16), which
holds about three significant digits. B2 has one hidden layer where [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") has
five, so that a change in what a brain sees reaches what it does
([Why evolution stops](how-to-tell-a-world-that-keeps-evolving.md#why-evolution-stops)).

The [engine](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") turns the outputs into decisions:
- **The stake.** Everything the agent may stake goes all on the column with
  the highest stake score, or is spread in proportion to the scores. A pair
  of outputs, averaged over the columns, decides which. Under B2 an agent may
  stake 70% of its tokens; 30%, rounded up and at least one token, stays at
  home.
- **The child.** A pair of outputs, averaged over the columns, gives the share
  of its tokens the agent gives a child
  ([The share function](share-function.md)). Under B2 a child needs at least
  10 tokens, or there is none.
- **The child's links.** Each column says yes or no to joining the child to
  that candidate.
- **[Revolutions](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.").** A pair of outputs per column says how much of the stake on
  that candidate is marked revolutionary.
- **The messages.** Each column's eight message outputs, squashed by tanh into
  the range from −1 to 1, are what the agent writes to that candidate.

## A brain's arithmetic

### A unit

A unit is the smallest working part of a brain. Picture a judge who hears
every number of the column, listens to some more than to others, and gives
one verdict between 0 and 1.

Exactly: a unit multiplies each of the 66 inputs *x*₁, …, *x*₆₆ by a weight of
its own. It adds up the products and one more number, its bias *b*, and
squashes the sum with the sigmoid σ:

$$
h = \sigma\bigl(w_1 x_1 + w_2 x_2 + \dots + w_{66} x_{66} + b\bigr),
\qquad \sigma(z) = \frac{1}{1 + e^{-z}} .
$$

A weight says how much an input counts, and which way. A positive weight
pushes the verdict up as its input grows, and a negative one pushes it down.
A weight of zero means the unit ignores that input. The bias sets where the
verdict tips: σ(0) = 0.5, σ(2) = 0.88 and σ(−2) = 0.12.
[The brain](the-brain.md#the-neural-network) describes the same arithmetic
for B1's larger brains.

### Large weights make a switch

Far from zero the sigmoid is flat: σ(20) and σ(−20) differ from 1 and 0 by
about two parts in a billion. So a unit with large weights answers, in
effect, yes or no.

The brain built below has a unit that asks "is this column the agent
itself?". It puts a weight of 40 on the input that is 1 in the agent's own
column and 0 elsewhere. Its bias is −20. In the agent's own column its sum
is 40 − 20 = 20, and its verdict is 1 to eight decimal places. In a
neighbour's column the sum is −20, and the verdict is 0.

### One unit draws one line

A unit's verdict hangs on a single weighted sum. So all it can ask is whether
that sum is above a threshold. With two inputs, the threshold is a line in
the plane of the inputs. Points on one side get yes, points on the other side
no. With more inputs it is a flat sheet, but the idea is the same.

So one unit cannot ask "is this number between 0.88 and 0.92?". The yes lies
between two no's, and no line puts it alone on one side. Two units can: two
walls. One has a weight of 10,000 on the number and a bias of −8,800, so it
is near 1 when the number is above 0.88. The other has the same weight and a
bias of −9,200: near 1 above 0.92. The first minus the second is 1 between
the walls and 0 outside them. Try three numbers:
- at 0.9, the first sum is 9,000 − 8,800 = 200 and the second −200: 1 − 0 = 1;
- at 0.5, both sums are far below zero: 0 − 0 = 0;
- at 0.95, both are above zero: 1 − 1 = 0.

### The outputs add up

An output is a weighted sum of the 16 units, with no sigmoid after it. Each
unit adds its own share, whatever the others say. So an output can add up
evidence, but not multiply it: it cannot make one unit's vote count only when
another unit says yes.
[When one unit must see two things at once](#when-one-unit-must-see-two-things-at-once)
turns on this.

## A raider in a brain's weights

### The rule in words

The [raider](strategies-written-by-hand.md#three-strategies) does three
things:
- **It attacks.** It looks for the poorest neighbour that is not a teammate: a
  *stranger*. If 70% of its own tokens are more than one and a half times
  that stranger's, it stakes them all there. Otherwise it keeps everything at
  home.
- **It breeds.** At 30 tokens or more it gives a child a third of them,
  joined to every teammate next door.
- **It signs.** Every message it writes starts with four fixed numbers, 0.9,
  −0.9, 0.9, −0.9: its password. Teammates are the neighbours whose last
  message to it starts with these numbers, each within 0.02.

### From words to units

A rule in words becomes weights in two steps. First, list what the rule needs
to know about each candidate, and give each question a unit. Then weigh the
answers in the outputs, so that the candidate the rule would choose scores
highest. Twelve of the brain's sixteen units are enough for the raider. The
other four are left at zero.

In the table, *own* is the agent's tokens and *its* the candidate's:

| unit | its weights | is near 1 when |
|---|---|---|
| 0 | 40 on the row that says the column is the agent itself; bias −20 | the column is the agent itself |
| 1–8 | 10,000 on one of the first four numbers the candidate wrote to the agent, two walls per number, at the password's value less 0.02 and plus 0.02; −30,000 on the self row | the lower wall minus the upper is 1 inside the band and 0 outside |
| 9 | 6,000 on ln(1 + own), −6,000 on ln(1 + its); bias −6,000 × 0.69 | ln(1 + own) − ln(1 + its) > 0.69: the candidate can be taken |
| 10 | −1 on ln(1 + its); bias 3 | it falls as the candidate's tokens rise: how weak the candidate is |
| 11 | 1,500 on ln(1 + own), its line halfway between ln 30 and ln 31 | the agent holds 30 tokens or more: rich enough for a child |

### The outputs

The outputs weigh what the units say. A password number is *heard* when the
candidate's last message to the agent carries it, within 0.02: its band
reads 1.

- **Stake score:** 40 × (can be taken) + 1 × (weak) + 5 × (self), less 10 for
  each password number heard. The stake goes all on the highest score.
- **Child:** a share of a third when rich, nothing otherwise. The two child
  outputs read a third and two thirds when unit 11 is on, and 0 and 1 when it
  is off. The share function turns these into a third and nothing.
- **Links:** yes where the column is the agent itself or a teammate. The
  "yes" output is 1 for the agent's own column, plus a quarter for each
  password number heard; the "no" output is a fixed half.
- **Messages:** the password, then zeros. The message outputs are fixed
  numbers, the same in every column, which tanh turns into 0.9, −0.9, 0.9,
  −0.9, 0, 0, 0 and 0.
- **Revolutions:** none. The revolution outputs are fixed at a share of
  nothing.

### Three orders

The stake score's weights keep three orders. Each bracket sets the least the
winner can score against the most the loser can; the weak unit adds between
0 and 1.
- A stranger that can be taken beats home, even when three of its numbers
  match the password (40 − 30 > 5 + 1).
- Home beats a stranger that cannot be taken (5 > 1).
- Home beats every teammate. A teammate's 40 for "can be taken" is cancelled
  by its four password numbers. So home wins: 5 > 40 + 1 − 40, or
  5 + 40 > 40 + 1.

Among strangers that can be taken, the weakest scores highest. So the brain
puts its stake where the rule does.

### One look, worked through

Here is one look of a raider holding 3 tokens. Its neighbours are a stranger
of 1 token, a stranger of 2 and a teammate of 1:

| column | can be taken | weak | self | password numbers heard | stake score |
|---|---|---|---|---|---|
| itself, 3 tokens | 0 | 0.83 | 1 | 0 (shut) | 5 + 0.83 = 5.83 |
| stranger, 1 token | 1 | 0.91 | 0 | 0 | 40 + 0.91 = 40.91 |
| stranger, 2 tokens | 0 | 0.87 | 0 | 0 | 0.87 |
| teammate, 1 token | 1 | 0.91 | 0 | 4 | 40 + 0.91 − 40 = 0.91 |

The stranger of 1 token wins, and the raider stakes there the 2 tokens it may
stake. The rule says the same: 70% of 3 tokens is 2.1, more than one and a
half times 1. The stranger of 2 tokens cannot be taken, since 2.1 is less
than 3. The teammate is never a target.

### Three details

Three details took more than the first try:
- **The agent's own column has to be shut out of the password.** In its own
  column, an agent reads what it wrote to itself a phase ago. A node a raider
  has just won, or a raider just born, has not written anything yet. So its
  own column looked like a stranger's, and home lost to every target. Now the
  self row closes every password wall: its weight of −30,000 keeps all eight
  walls at 0 in the agent's own column.
- **"Can be taken" is a line in ln(1 + tokens), where the rule's line is
  straight in tokens.** No straight line on the one scale is the other's
  line. The rule counts 13 tokens against 6 as a sure take, and 15 against 7
  not. Indeed, 70% of 13 is 9.1, more than 1.5 × 6 = 9. But 70% of 15 is
  10.5, no more than 1.5 × 7 = 10.5. Yet both pairs differ by ln 2 on the
  brain's scale, so unit 9, which reads that difference, must treat them
  alike. But the raiders hold one to a few tokens. Over the 55,599 weakest
  strangers they faced in the three worlds below, a line just below ln 2
  agrees with the rule every time. Both say that 3 tokens take 1, and that 2
  against 1 or 6 against 3 do not.
- **A third is not a third in float16.** The nearest float16 below it,
  0.33325, would give a parent of 30 tokens a child of 9, below B2's minimum
  of 10. The weight is the next float16 above a third, 0.33350, which gives a
  child of 10.

### The check

Raiders were planted as the raider's own tests planted them. A patch of 5%
of the agents went into each of three of
[Experiment 16](how-to-tell-a-world-that-keeps-evolving.md#the-runs-of-experiment-16)'s
B2 worlds, at iteration 2,000. Every situation a raider met in their first 40
iterations was kept. A situation is one look: a column for each candidate, as
the raider read it. That gives 62,114 situations. Each was given to the rule
and to the brain, and the two were compared decision by decision, as the
engine would read them.

| world | situations | every decision the same | the stake on the very same candidate |
|---|---|---|---|
| `B2-10000-s001` | 15,824 | 100% | 99.4% |
| `B2-10000-s002` | 21,024 | 99.98% | 98.8% |
| `B2-10000-s003` | 25,266 | 99.98% | 98.4% |

"Every decision the same" counts a stake as the same if it went to the same
kind of target, holding the same tokens. That is home, or a stranger as weak
as the rule's choice. Where the brain chose another candidate, it was another
equally weak stranger. The rule takes the first of equals; the brain takes
whichever a trace of the password walls puts a hair ahead.

The child, its links, the password and the revolutions matched in every
situation. In eight of the 62,114, the weakest stranger's last message
happened to match one of the password's four numbers. There the brain took
the next weakest stranger instead.

One part of the rule was barely tested. Only one of the 62,114 situations had
a raider with 30 tokens or more, rich enough for a child. So the check shows
the child and its links mostly where there is no child. And the brain would
join a child to a stranger whose message matched three of the password's four
numbers, which the rule never does. No raider that rich met one.

### Planted in its place

Planted in its place, the brain does what the rule did. The raider's
invasion test ([The invasion test](eras-and-contests.md#the-invasion-test))
was played in worlds 1 to 3, on seeds 1 and 2. Each patch was played once
with the rule and once with the hand-set brain, in the same place. Neither
ever mutates.

<!-- figure computable/replays -->
![The rule and the brain, planted in the same places](../figures/computable/replays.svg)

**The rule and the brain, planted in the same places.** One line per planting: three evolved worlds, two seeds each (seed 1 solid, seed 2 dashed). Orange: a patch of raiders that follow the rule of [Strategies written by hand](strategies-written-by-hand.md); blue: the same patch given a B2 brain whose weights were set by hand to play the same rule. Neither ever mutates. A line that stops early is a patch that died out.

> [!example]- How to make this figure
> **Runs.** Three B2 worlds of Experiment 16, `B2-10000-s001` … `-s003`, as their archives keep them at iteration 2,000 (`GraphOfLifeRuns/<run>/archive/`), each with a patch of raiders planted as in [Strategies written by hand](strategies-written-by-hand.md) (`research/oee/pilots/manual.py`: the agent drawn on seed *s* and its nearest neighbours, step by step, until the patch holds 5% of the world).
>
> **Data.** From the worlds the archives keep.
>
> 1. Load the world of iteration 2,000 and plant the patch on seed 1 or 2.
> 2. Give every agent of the patch the raider's rule (`manual.py`), or a frozen copy of the hand-set brain (`research/oee/strategies/raider_B2.npz`, made by `python3 research/oee/strategies/handset.py save`).
> 3. Play 300 iterations under the world's own rules (`handset.py invade B2-10000-s001 B2-10000-s002 B2-10000-s003 --seeds 1,2 --T 300`) and count, after each, the agents carrying the patch's brain, as a share of all agents.
>
> **To make it again:** `python3 book_figures.py computable`.
<!-- /figure -->

In world 1 both died out on both seeds, as the rule had in the raider's own
tests. In world 2 both took 40 to 50% of the world: the rule 40% and 50%,
the brain 47% and 50%. In world 3 the rule took 62% on seed 1, and the brain
63%. On seed 2 the rule died out after 143 iterations, while the brain took
59%.

The two choose the very same candidate in all but one or two situations in a
hundred. So their worlds part ways within a few iterations, and from there it
is chance whether a patch takes hold. That is a warning for every invasion
test: one planting says little. The tests that came after plant each patch in
several places per world
([How large a crowd a raider needs](strategies-written-by-hand.md#how-large-a-crowd-a-raider-needs)).

### So the raider is a B2 brain

It takes 12 of the 16 units B2 gives every agent. Evolution did not fail to
find it for lack of room in a brain.
[How deep is the valley?](strategies-written-by-hand.md#how-deep-is-the-valley)
says why evolution misses it. The raid, the password and the crowd pay only
together, and about a hundred raiders must start at once. [Pilots](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.") that gave
evolution the password and the crowd for free did not lead it there either
([A password and a crowd, for free](another-game.md#a-password-and-a-crowd-for-free)).

## What no brain of B2 can compute

Picture an agent as a clerk who answers its neighbours' letters. It writes
each reply with one file open, the file on that neighbour. The file holds
both agents' tokens and connections, and a summary of each one's
neighbourhood. It also holds the last notes the two wrote to each other and
to themselves. The clerk never has two files open at once. So no reply can
pass on what a third agent wrote.

**The claim.** In B2, what an agent *u* writes to a neighbour *v* depends
only on what *v*'s column in *u*'s look holds:
- the tokens and connections of *u* and *v*;
- the quantiles of their neighbours' tokens and connections;
- the messages *u* and *v* wrote to each other and to themselves the phase
  before;
- the random numbers.

It never depends on what any third agent wrote, nor on which brain a third
agent carries.

**Why.** A brain scores each column from that column alone: the same weights,
column by column, and no unit sees two columns. So *u*'s message to *v* is
made from *v*'s column, which holds the list above. What *u* wrote to itself
is made from *u*'s own column. That column holds *u*'s tokens, connections
and quantiles, and what *u* wrote to itself before. Of *u*'s neighbours it
holds nothing but the quantiles of their tokens and connections.

So if no message written so far carried word of a third agent, none written
now does. At the start nothing has been written. Phase by phase, no message
ever carries such word. The quantiles are the only window onto other agents,
and they show tokens and connections: never what anyone wrote, never which
brain anyone carries.

**What does see every column** is the agent's own decisions:
- the stake goes on the best column, or is spread over all of them;
- the child's share, and the choice between all in and spread, are averages
  over all columns.

News of a third agent can leave an agent only through what it does: the
tokens and connections its decisions change, which its neighbours see. A
message cannot carry it.

**What this rules out.** The
[factory](strategies-written-by-hand.md#three-strategies) had every member
write its distance to the team's frontier. That distance is 0 if a stranger
is next door, and otherwise one more than the nearest teammate's. To a
teammate, whether *u* has a stranger next door is news of a third agent. No
B2 brain, of any size, can write it into a message. It could only show it
through its own tokens or connections, at a cost. Raiders that pool their
stakes would need the same kind of news to choose a target together
([A crew of raiders](does-evolution-close-the-door.md#a-crew-of-raiders)).

The raider needs nothing of the kind. Everything it decides comes from its
own columns, and that is why its weights exist.

## When one unit must see two things at once

Some rules need one fact to count for a candidate in one look and against it
in another, depending on a second fact. The outputs cannot do that, since
they only add. It takes a unit that sees both facts at once.

**A factory without the message.** Suppose the factory gave up its message,
and its members simply fed their poorest teammate when no stranger can be
taken. That needs something the raider's units do not give. The stake score,
one weighted sum per column, would have to rank:
1. a stranger that can be taken;
2. then the poorest teammate, whatever it holds;
3. then home;
4. and a stranger that cannot be taken, below home.

The raider's "can be taken" unit fires for a poor teammate as it does for a
stranger. Picture a teammate and a stranger that hold the same tokens and
differ only in their password. In the raider's units, the password and the
tokens never meet. So being a teammate moves the score by a fixed amount,
whatever anyone holds. Beside a rich agent both can be taken, and the
stranger must score higher: being a teammate must cost points. Beside a poor
agent neither can be taken. Then the teammate must beat home while the
stranger loses to it: being a teammate must earn points. A fixed amount
cannot do both.

**The exact statement.** Take two looks that differ only in the agent's own
tokens: rich in one, poor in the other. In each, a teammate and a stranger
hold the same tokens and differ only in their password. If no unit reads both
the password and the tokens, the teammate's lead over the stranger is the
same in both looks. The rule needs it below zero in the rich look and above
zero in the poor one. So some unit must read the password and the tokens at
once.

One unit cannot do that with the raider's password. A unit is near 1 on one
side of a line. A password number in the middle of a message's range is a
band, two walls a number. And a band cannot be one side of one unit's line. A
password number at the edge of the range — beyond 0.98, say — is a single
wall, since no message number exceeds 1. A single unit can combine such a
wall with tokens. The ladder of strategies to be built on the raider will
need such passwords.

## The smallest change

The obstacle is precise: a brain pools its columns only into its own
decisions, never into what it says. The smallest change that lifts it is a
**pooled message to itself**. The message an agent writes to itself becomes
the largest, number by number, of the messages it writes to all its
candidates, itself included.

Every neighbour already reads that message: it is one of the four messages in
each column. So with the change:
- **One phase:** a column can mark "this neighbour is a stranger". The
  agent's message to itself then says "a stranger is next door", and every
  neighbour reads it a phase later.
- **A few phases:** a distance to the frontier follows. An agent marks a
  stranger 1, and a teammate a fraction of what that teammate says of itself.
  The largest mark is then its own nearness to the frontier. This is a field
  over the team, which settles in as many phases as the team is deep.
- **In general,** the agents' brains become a network that passes messages
  over the graph and pools them. It is a graph neural network, with the
  largest as its pooling. What such networks can tell apart is known
  (Xu et al., 2019). No network that passes messages tells apart more than
  the Weisfeiler–Lehman test does, which colours every node by its
  neighbours' colours, round after round. One that pools by the largest tells
  neighbourhoods apart by which values occur in them, not by how often.

It costs no weights. A brain keeps its shape, and a world saved without the
change loads with it. But an agent's message to itself is no longer free: it
is at least what it tells any neighbour. So the change alters what every
evolved brain does. It is a rule to test in new worlds, off unless a run
turns it on, not a repair for old ones.

The engine has had it since 19 October 2026, as the setting
[`self_message`](settings.md#self_message) = largest; B1 and B2 keep `own`.
No experiment has yet tested it on its own.
[A screen of random worlds](a-screen-of-random-worlds.md) turns it on in
about half of its worlds, among many other changes.

A second way would give every column the largest and smallest of what all
neighbours wrote, as new inputs. That leaves the messages free, and changes
the brain's shape.

A field of nearness over a team is also a first local field on the graph.
Every agent holds a value of it, made only from its neighbours' values a step
before. The notebook's second aim, a physics
([Physical inspiration](physical-inspiration.md)), asks for exactly that kind
of thing.

## What we learned

- **The raider is within B2's reach.** A brain of 12 units plays it, in every
  decision of 62,114 situations but eight. Planted in the raider's place, it
  took 47 to 63% of worlds 2 and 3. The rule took 40 to 62% there, and once
  died out. Evolution's failure to find it is a failure of the way there, not
  of the brain.
- **A brain is private to each pair.** What it hears from one neighbour
  reaches another only through what it does, never through what it says.
  Pipelines and chains of command that pass word by message, and any message
  about a third agent, are beyond every B2 brain, whatever its size.
- **The outputs only add.** Some rules need one fact to count for a candidate
  in one look and against it in another. Those need a unit that sees both
  facts at once. One unit can join a password at the edge of a message's
  range to the tokens, but not a password in the middle.
- **One small rule would open the messages**: a message to itself that pools
  what it says to everyone. It is now a setting, still to be tested on its
  own. [What would count as open-ended evolution](open-ended-test.md) says how
  the long runs will be judged.

## Where to go next

- [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md):
  the raider, the factory, and the crowd a raid needs.
- [Why the raiders stopped winning](does-evolution-close-the-door.md): the
  same raider in older worlds, where its patches grew too poor to raid.
- [What a brain sees](brain-inputs.md) and
  [What a brain says](brain-outputs.md): every input and output row of a
  brain.
- [Messages](messages.md): who writes the four messages, and when they are
  read.

## How this was made

The check and the replays are pilots: no experiment of the lab stands behind
them, so this page has no box of runs. They were made with
`research/oee/strategies/handset.py`, and its records are kept in
`GraphOfLifeRuns/.pilots/strategies/`.
- **The weights** are set by the engine's `gol_strategies.raider_weights`,
  unit by unit as above. `python3 research/oee/strategies/handset.py save`
  writes them to `research/oee/strategies/raider_B2.npz`.
- **The situations:** `handset.py record B2-10000-s001 --era 2000 --T 40 --seed 1`,
  and the same for `-s002` and `-s003`. Each plants the raiders' patch and
  keeps every column a raider read, in `raider_situations_<run>_2000_s1.npz`.
- **The comparison:** `handset.py check` with the three files and
  `--out agreement.json`. It gives every situation to the rule and to the
  brain, and compares their decisions as the engine reads them.
- **The replays:** `handset.py invade`, as the figure's recipe says.
- **The numbers:** the comparison and the replays are in
  `book/results/computable.json`, and `python3 book_figures.py computable`
  draws the figure again. The counts of weakest strangers and of rich raiders
  are read from the situations' files.

## How our view changed

- **24 October 2026.** From 19 October, this page said that the raider took
  "half or more of three worlds in ten". The plantings' records
  (`book/results/handmade.json`) say otherwise. Planted once in each of ten
  worlds, the raider took 40% of world 2 and 62% of world 3. It died out in
  the other eight. The third world, world 6, fell to the factory, with 43%.
  [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md#ten-worlds)
  corrected the same words when its pages were merged that day.

## References

- Xu, K., Hu, W., Leskovec, J. and Jegelka, S. (2019). How powerful are
  graph neural networks? *International Conference on Learning
  Representations*.

<!-- turns -->
---

← [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md) · [Strategies written by hand](../README.md#strategies-written-by-hand) · [Why the raiders stopped winning](does-evolution-close-the-door.md) →

**Also about:** [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are).

> [!info]- Pages that link here (9)
> [Every setting](settings.md) · [What a brain sees](brain-inputs.md) · [What a brain says](brain-outputs.md) · [Messages](messages.md) · [Eras, contests and invasions: does a world keep getting better?](eras-and-contests.md) · [Every era against three](every-era-against-three.md) · [What would count as open-ended evolution](open-ended-test.md) · [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md) · [Why the raiders stopped winning](does-evolution-close-the-door.md)
<!-- /turns -->
