# Do agents cooperate? Help, kin and partners

> [!question] Questions of this page
> - What would cooperation be in a game whose only act is a stake? What
>   does an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") give, at what cost, and when does it help?
> - Are [neighbours](the-world.md#the-network "An agent joined to another by a connection.") related? Do agents treat their relatives differently?
> - When a node changes hands, does it stay in the family?
> - Do agents answer a stake with a stake? How long do the same two agents
>   face each other — long enough for anything like trust?
> - How could cooperation be found and counted?

## In short

Cooperation means paying a cost so that someone else gains. In Graph of Life
the only act is a stake. A stake on a neighbour's node **helps** if, without
it, the neighbour would have been more likely to lose that node. In the
thirty [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") worlds of 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), kin live side by side, but [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") cannot
see kinship: they stake on kin as on strangers. Most pairs that stake on each
other stake exactly the same, a handshake that keeps their [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") alive.
And the same two brains rarely face each other twice, so direct reciprocity
has no future to work in.

## Five ways cooperation can evolve

Why would anyone pay a cost so that someone else gains? That is
**cooperation**, and it is one of the central puzzles of evolution. Selection
favours whatever spreads. Why would it favour helping another to spread?

And yet the great steps in the history of life were steps of cooperation.
Genes joined into genomes, cells into organisms, organisms into colonies and
societies. In these "major transitions" (Maynard Smith and Szathmáry 1995),
units that once competed came to reproduce as one. A world in which evolution
keeps producing new kinds of things will, very likely, have to produce new
kinds of **togetherness**. So for the aim of this notebook,
[open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau."), cooperation is not a side topic.

This page asks what cooperation would mean in Graph of Life, and measures what
is there. Then it sets out how cooperation could be found and counted.

Biologists know five ways in which cooperation can evolve. Martin Nowak sorted
them in *Five rules for the evolution of cooperation* (Science, 2006). Each
comes with a condition. It compares the **benefit** *B* that a helping act
gives the receiver with its **cost** *C* to the helper.

1. **Kin selection** (Hamilton 1964). Help relatives, who carry copies of your
   genes: a gene for helping can spread by helping copies of itself. Helping
   spreads if *r B* > *C*. Here *r* is the **relatedness**: how much more
   likely the receiver is than a random member of the population to carry the
   helper's genes.
2. **Direct reciprocity** (Trivers 1971; Axelrod and Hamilton 1981). Help
   those who help you. It works only if the same two meet again. If *w* is the
   probability of another round with the same partner, cooperation can be
   stable when *w* > *C*/*B*. Without a future, there is nothing to
   reciprocate.
3. **Indirect reciprocity** (Nowak and Sigmund 1998). Help those with a good
   reputation. It needs others to know what you did.
4. **Network reciprocity** (Nowak and May 1992; Ohtsuki, Hauert, Lieberman and
   Nowak 2006). On a network, cooperators can form clusters that help each
   other and resist exploiters. On a network where each has *k* neighbours,
   cooperation can win when *B*/*C* > *k*.
5. **Group selection** (Traulsen and Nowak 2006). Groups of cooperators grow
   and split more than groups of defectors.

## Help

Every one of the five ways needs an act that helps. So what would helping be
in a game whose only act is to stake tokens on nodes? This section first
lists what an agent can give. Then it defines **help** from the rules of the
game alone, without asking what an agent meant, and three signs by which help
becomes **cooperation**.
[Gliders, and who helps whom](gliders-and-who-helps-whom.md) measures both.

### What an agent can give

What can an agent of Graph of Life give, and at what cost?

- **A stake on a neighbour's node.** Every token staked on a neighbour ends
  the game on the neighbour's node ([Gains and losses](gains-and-losses.md)).
  It is a pure gift of tokens: it costs the staker *C*, and the node gains
  *B* = *C*. But a stake is also a **bid** for that node: if it wins, the node
  takes the staker's brain. For the brain, every gift is also an attack.
  This double meaning is built into the game.
- **A child's endowment**: the tokens and connections a parent gives its
  child ([How agents have children](how-agents-have-children.md)).
- **Keeping a connection alive.** A connection survives only if a token
  crosses it ([How a world breaks](how-a-world-breaks.md)). Paying that token
  keeps open a channel both sides need.
- **Joining a coalition.** A stake marked revolutionary can join others to
  take a node from its largest staker
  ([How a coalition takes a node](revolution.md)).
- **Messages**, which cost nothing, and could carry signals
  ([Messages](messages.md)).

Only a stake acts on what happens to another agent's node. And a stake on a
neighbour's node is a gift and a bid at once. So when does it help the
neighbour?

### A stake that saves a node

Every stake an agent $u$ puts on a neighbour $v$'s node changes the chance
that $v$ keeps it. To see how, compare the game as it was with the same game
without $u$'s stake, everyone else's stakes unchanged:

$$\text{effect}(u \to v) = P(v \text{ keeps its node} \mid \text{all stakes}) - P(v \text{ keeps its node} \mid \text{all stakes but } u\text{'s}).$$

In words: the effect is how much more likely $v$ was to keep its node with
$u$'s stake than without it. A positive effect is **help**: without it, $v$
would have been more likely to lose its node. A negative effect is an
**attack**. Most stakes have no effect at all, because the node would have
gone the same way without them.

Both probabilities come from the world's own rule for deciding a node,
counted exactly, with nothing sampled (`research/oee/checks/decisive.py`).
Under the [engine](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.")'s rule ([One iteration, step by step](one-iteration.md)), the
largest stake wins, and ties are drawn at random. In [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") the owner's own
stake counts one and a half times. But a crowd of revolutionary stakes can
win instead. Count the crowd up from its smallest stake. Once the part counted
so far outweighs the largest staker and everyone in the crowd above it, the
crowd wins. The winner is drawn from the step that tipped it. Under the
[pilots](the-lab.md#pilot-worlds "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.")' alliances ([Alliances](another-game.md#alliances)), the side with the
largest total wins.

A tie between $k$ stakers gives each a probability $1/k$. In both cases the
probability is what the rule would give if the game were played many times.
So the effect is a fraction of a node: a stake that turns a lost node into a
tie has the effect $1/2$.

Under these rules, only two things let one agent's stake raise another's
chance to keep its node. Revolutions do, because a stake on $v$'s node
can join $v$'s own in the crowd that overthrows an attacker. Pooled alliances
do, because an ally's stake counts for $v$'s side. Without either, a stake on
someone else's node is a bid against its holder and can only lower that
chance. Help is then impossible by rule.

### A worked example

$U$ stakes 6 tokens on $V$'s node and marks none of them revolutionary. $V$
stakes 4 on its own node, all revolutionary. $V$'s neighbour $W$ stakes 3 on
$V$'s node, all revolutionary.

**Under B1's rule**, with no extra weight for the owner, the largest
stake is $U$'s. So $U$ is the one the crowd must beat. The crowd is $W$ with
3 and $V$ with 4, together 7, more than 6. Count up from the smallest. $W$'s 3
is not yet more than $V$'s 4 plus $U$'s 6. But $W$'s and $V$'s 7 is more than
$U$'s 6 alone. So $V$'s step tips it, and $V$ keeps its node: $P = 1$.
Without $W$'s stake the crowd is $V$ alone, and 4 is not more than 6: $U$
takes the node, $P = 0$. The effect of $W$'s stake is $1 - 0 = +1$: it saved
$V$'s node.

**Under B2's rule**, $V$'s 4 counts as 6 and ties with $U$'s 6. With $W$, $V$
keeps the node whichever of the two is drawn as the largest: $P = 1$. Without
$W$, it keeps it only when it is drawn: $P = 1/2$. The effect of $W$'s stake
is $+1/2$.

The effect of $U$'s stake, given $W$'s, is zero under both rules: $V$ keeps
its node with or without it. And $W$'s 3 tokens go to $V$'s node. So $W$ has
paid 3 tokens for $V$'s keeping it, and $V$ gained them.

### From help to cooperation

Help alone is not yet cooperation. It can be a by-product. An agent that
stakes a little on every neighbour helps some of them, with no more reason to
help one than another. Evolution can favour helping only by one of the
five ways [above](#five-ways-cooperation-can-evolve), and each leaves its own
mark in who helps whom (Nowak 2006). Three of the marks can be read from the
record of the games, each against the level chance would give:

- **Kin.** Help goes to clones, agents carrying the helper's own brain, more
  often than clones are among the helper's neighbours.
- **Direct reciprocity.** Help is returned. Take the pairs in which $u$ helped
  $v$ within a stretch of 50 games, and count the share in which $v$ also
  helped $u$. Its chance level gives every pair the same chance per game they
  spent as neighbours: $1 - (1 - q)^n$ for $n$ such games, $q$ being help per
  neighbour-game. So pairs that were simply neighbours for longer do not pass
  for pairs that return help.
- **Indirect reciprocity.** Help goes round: $u$ helps $v$, $v$ helps $w$, $w$
  helps $u$. The net help between two agents in a stretch is a flow on the
  network. The share of it that circulates is compared with chance. Chance is
  the share that would circulate if each pair's net help had a random
  direction ([Flows that go round](token-flow.md#flows-that-go-round)).

Help that shows none of the three is help without cooperation.

### What the measure can and cannot say

The effect is that of one game, with everyone else's stakes as they were: had
$W$ not staked, $V$ might have staked differently. It is measured one stake at
a time. So when two helpers each would have been enough, neither is counted as
saving the node: help that is redundant is undercounted. The effect is about
keeping a node. The tokens a helper hands over count too, but later, and they
are a transfer that the helper loses. And a stake with a positive effect may
also have been a bid that could have won the node for the helper. Help is what
a stake did, not what it was for.

[Gliders, and who helps whom](gliders-and-who-helps-whom.md) takes this
measure in B1, in B2 and among allies. [Who helps whom](who-helps-whom.md)
sets what it found beside every other page on cooperation. The rest of this
page looks at B1's stakes without it: who is related to whom, who stakes on
whom, and for how long.

## Are neighbours related?

From here on, the worlds are the thirty B1 worlds of 100,000 tokens of
[Thirty worlds](a-worlds-life.md#thirty-worlds). This page was first written
from the thirty of 10,000 tokens; where the two differ, both are given.

Kin selection needs relatives to meet. Do they?

<!-- figure cooperation/related -->
![Are neighbours related?](../figures/cooperation/related.svg)

**Are neighbours related?** One dot per world. For the pairs of agents joined by a connection ("neighbours"), and for pairs of agents drawn at random from the whole world (2,000 per look), the share that carry the same genotype, and the share that are close kin: the same genotype, or one's genotype the parent of the other's, or both with the same parent genotype. At the start of the games of every 100th iteration from 500 on.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. For every 100th iteration t from 500 on, read frame `2·t`: each agent's genotype (`brain_ids`) and its parent genotype (`parent_brain_ids`).
> 2. For every connection of `edges`, test the two relations; also for 2,000 pairs of agents drawn at random (generator seeded with 11).
> 3. Divide the counts by the pairs, per run.
>
> **To make it again:** `python3 book_figures.py cooperation`.
<!-- /figure -->

Yes, strongly. In the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world, **9.6%** of neighbours carry the
same [genotype](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from."), against **0.05%** of two agents drawn at random: 200
times as often. At 10,000 tokens it was 10.6% against 0.28%, 38 times. Counted
over all pairs of a world rather than a sample of them, chance is rarer still:
0.03%. [Like next to like](like-next-to-like.md) puts the ratio near 290.

Count close kin too: one's genotype the parent of the other's, or both sharing
a parent genotype. Then 20% of neighbours are kin, against 0.12% of random
pairs. The share of kin among neighbours is the same at both sizes. What a
larger world changes is how rare kin are far away.

This is the picture of [One world, many colours](one-world-many-colours.md) in
numbers: families live together. A brain's copies sit next to it, for two
reasons. A brain spreads by taking neighbouring nodes
([Where the tokens flow](where-the-tokens-flow.md)), and children are joined
to their parents.

So the precondition of kin selection is met: the relatedness *r* between
neighbours is high.

## Do agents treat their own kind differently?

<!-- figure cooperation/kin-stakes -->
![Does an agent treat its own kind differently?](../figures/cooperation/kin-stakes.svg)

**Does an agent treat its own kind differently?** Only agents that had at least one neighbour of their own genotype and at least one of another, in the games of every 100th iteration from 500 on; one dot per world. Left: the share of its tokens an agent staked on each neighbour, averaged over the neighbours of its own genotype (green) and over the others (grey), then over the agents. Right: of what it staked on a neighbour, the share it marked revolutionary.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. For every 100th iteration t from 500 on, take the genotypes from frame `2·t` and the stakes from `decisions.allocations` of frame `2·t + 1`.
> 2. For each agent with neighbours of both kinds: the mean of `alloc` / `tokens` over each kind, and the mean of `revolt` / `alloc` over the neighbours it staked on.
> 3. Average over the agents of a run; one dot per run.
>
> **To make it again:** `python3 book_figures.py cooperation`.
<!-- /figure -->

No. Take the agents that had neighbours of both kinds. On average, an agent
staked 0.202 of its tokens on each neighbour of its own genotype, and 0.197 on
each other neighbour. Only 31% of these agents staked more on kin than on
others. And they mark 99.7% of what they stake on either kind as
revolutionary. At 10,000 tokens it was 99.7% on kin and 99.6% on others.

That is not surprising once one looks at what a brain can see
([What a brain sees](brain-inputs.md)). It sees tokens and connections: its
own, the candidate's, and their neighbourhoods'. It reads messages and
random numbers. **Not genotypes.**

So a brain cannot tell kin from stranger, except through messages. A
[lineage](genotype.md#families "A line of descent: the agents whose genotypes go back to one ancestor.") could, in principle, evolve a "password": send a particular
pattern, and treat those who send it back differently. Biologists call this a
**green-beard** (Hamilton 1964; Dawkins 1976). Nothing in the stakes says that
this has happened.

The pattern itself exists already, without any evolution.
[What agents say to each other](what-agents-say-to-each-other.md) found that
every message names its writer's genotype. Two agents of one genotype write
the same message, but for a thousandth or so. Kin could tell each other by it.
What is missing is a brain that compares what it reads, and the
[baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.")'s brains barely see
([What the brains are like](what-the-brains-are-like.md)).

## When a node changes hands, does it stay in the family?

<!-- figure cooperation/succession -->
![When a neighbour takes a node, is it the same kind?](../figures/cooperation/succession.svg)

**When a neighbour takes a node, is it the same kind?** Nodes won by a neighbour in the games of every 100th iteration from 500 on; one dot per world. Left: the share in which the winner carried the same genotype as the agent whose node it took — a takeover that changes nothing in the brain. Right: the same share if the winner had been drawn at random from the neighbours that staked on the node.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. For every 100th iteration t from 500 on, read the genotypes in frame `2·t` and `decisions.winners` and `decisions.allocations` in frame `2·t + 1`.
> 2. For each node whose `winner` is not the node: is the winner's genotype the node's? And: of the agents other than the node that staked on it, the share with the node's genotype.
> 3. Per run: the first count, and the sum of the second, over the nodes taken.
>
> **To make it again:** `python3 book_figures.py cooperation`.
<!-- /figure -->

In the sampled games, neighbours took 2,411,000 nodes. In the median world,
the winner carried the node's own genotype in 13.4% of cases. If the winner
had been any of the neighbours that staked on the node, 12.8% would be
expected. Takeovers do not spare kin, nor seek them out.

## Do agents answer a stake with a stake?

<!-- figure cooperation/returned -->
![What comes back for what is given](../figures/cooperation/returned.svg)

**What comes back for what is given.** Pairs of neighbours that both staked on each other in a game: how many tokens each staked on the other (12,000 pairs, up to 400 drawn at random per world, from the games of every 100th iteration from 500 on). A cell's colour is how many pairs fall in it; the diagonal is an even exchange. In the median world 82% of the pairs staked exactly the same on each other.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. From `decisions.allocations` of frame `2·t + 1`, for every 100th iteration t from 500 on, collect every pair (A, B) with a stake of A on B and of B on A.
> 2. Draw up to 400 pairs per run (generator seeded with 11) and count them in cells.
>
> **To make it again:** `python3 book_figures.py cooperation`.
<!-- /figure -->

Take the pairs of neighbours that staked on each other in a game. In the
median world, 82% of them staked **exactly the same** amount on each other.
And **66%** of all such pairs exchanged exactly one token each way. The
[correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of the logarithms of the two stakes is 0.57. (At 10,000 tokens:
79%, 58% and 0.63.)

Is that reciprocity? Two observations say: not in the sense of the theory.
First, an equal exchange cancels: it changes neither balance
([Where the tokens flow](where-the-tokens-flow.md)). What it does is keep the
connection alive at its minimum price, paid by both. Second, it is what the
[even split](even-split.md "What the rules do when a brain scores no candidate above zero: the stake is spread evenly.") gives. Two poor agents, each splitting a few tokens evenly over a
few candidates, put one token on each other. It is a **handshake**, not a
favour returned.

Real reciprocity would mean that one agent stakes more on another *because*
the other staked on it, beyond what the even split gives. That is the third
measure [below](#how-cooperation-could-be-found-and-counted), not yet taken.

## How long partners last

Direct reciprocity needs the same two to meet again: the "shadow of the
future", *w*. How long do the same two brains face each other across a
connection?

<!-- figure cooperation/partners -->
![How long partners last](../figures/cooperation/partners.svg)

**How long partners last.** Of the connections at the start of a game, the share still there k games later (blue); still there with neither of its two nodes won by a neighbour in any of the k games in between, so that the same two brains face each other, changed at most by the small changes offered after every game (green); and still there with exactly the same two genotypes (yellow). Pooled over the 4,010,315 connections at the start of the games of every 200th iteration from 500 on, in the 30 worlds. Both axes are logarithmic.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. For every 200th iteration t from 500 on, take the connections of frame `2·t` and the genotypes of their ends.
> 2. For k = 1, 2, 4, 8, 16: read frame `2·(t + k)` — is the connection still in `edges`, and are the ends' genotypes the same? — and the `decisions.winners` of every game frame `2·(t + j) + 1`, j < k: was either end won by a neighbour?
> 3. Pool the counts over the runs and divide by the connections.
>
> **To make it again:** `python3 book_figures.py cooperation`.
<!-- /figure -->

The connections last: 90% of the connections at the start of a game are still
there a game later, and 60% sixteen games later. But the **brains** at their
ends do not. In only **4%** of connections do the same two brains face each
other one game later, with neither end won by a neighbour in between. After
two games it is 0.8%; after four, 0.2%.

The stage stays; the actors change almost every game. About half of all
nodes, and nearly every [hub](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub."), are taken over in every game
([How the game is played](how-the-game-is-played.md),
[Where the tokens flow](where-the-tokens-flow.md)).

Direct reciprocity needs *w* > *C*/*B*, so a benefit at least 1/*w* times its
cost. With *w* ≈ 0.04, that is a benefit twenty-five times the cost (twenty at
10,000 tokens, where *w* ≈ 0.05). In this world, nobody meets anybody twice.

## How cooperation could be found and counted

Nothing measured here shows cooperation beyond what the even split and kinship
by proximity produce. But "not found" depends on where one looks. Here are six
measures the notebook can use, each computable from the
[frames](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections.") every run already writes:

1. **Kin discrimination.** The ratio of stakes on kin to stakes on others,
   per neighbour (here 1.02; 1.01 at 10,000 tokens). A value well above 1
   would mean that brains recognise kin — through messages, since they cannot
   see genotypes.
2. **Hamilton's accounting.** For every stake: its cost *C* to the staker, its
   benefit *B* to the receiving node, and the relatedness *r* of the pair.
   Summed over a world, Σ(*rB* − *C*) says whether helping as practised pays
   the helping genes. (This is the regression form of Price's equation;
   Queller 1992.)
3. **Reciprocity beyond the split.** The correlation between *a*(*u* → *v*)
   and *a*(*v* → *u*) after subtracting what an even split would give each.
   Real reciprocity is what remains.
4. **Partnerships.** The distribution of how long the same two brains keep
   exchanging stakes, against the null of independent takeovers. Lasting
   partnerships would be the first sign of something above the individual.
5. **Coalitions that repeat.** Whether the same agents join forces to take
   nodes again and again.
6. **Groups that win.** Whether regions held by one family grow, survive or
   split more than mixed regions. That is selection between groups. It can be
   read from the families of [One world, many colours](one-world-many-colours.md)
   and the breaks of [How a world breaks](how-a-world-breaks.md).

None of the six has been computed in the form given here. The work on the
second baseline, B2, judged progress another way. It played a world's
[eras](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?") against each other
([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)).
[Revolutions, handover and gifts: the switches the world was built with](the-switches-the-world-was-built-with.md)
judges the rules made for B2 by that test.

Nearby measures were taken instead. Help itself was counted exactly by the
rules ([Help](#help)), with the three signs of cooperation
[above](#from-help-to-cooperation), in
[Gliders, and who helps whom](gliders-and-who-helps-whom.md).
[Flow modules: the groups tokens stay in](flow-modules.md) followed how long
the groups that tokens keep to last, pairs of partners among them. And in
pilot worlds under B2's rules, connected groups of one genotype and of close
kin shrank over time, in every world.

## What the rules would have to give

For open-ended evolution this page points at rules, not brains. Three things
would open roads to cooperation:

- an agent that keeps its identity when its node is taken;
- a way to recognise kin;
- a way to commit to a partner.

[Questioning the mechanics](questioning-the-mechanics.md) examines the rules
responsible, and
[What the measurements say about open-ended evolution](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution)
proposes changes.

The work on B2 went some way down these roads. B2 gives an incumbent an
advantage, so that a brain keeps its node longer
([Who dies, and why](a-world-that-cannot-break.md#who-dies-and-why)). It
gives agents brains that see, which could read the names of their kin
([Brains that see, and worlds that break](a-world-that-cannot-break.md#brains-that-see-and-worlds-that-break)).
Whether they do has not been measured. A memory of what each neighbour staked
in the last game was tried too. It changed 11 to 24% of decisions, and
evolution stopped all the same.

Two attempts to make helping pay failed
([Why evolution stops](how-to-tell-a-world-that-keeps-evolving.md#why-evolution-stops)).
Stakes that support a neighbour's defence were given up by evolution. A yield
on stakes returned built stars rather than partnerships.

A way to commit to a partner was tested later: alliances by mutual consent
([Alliances](another-game.md#alliances)). The agents took them up at once.
But an alliance protects by peace, not by rescue
([Gliders, and who helps whom](gliders-and-who-helps-whom.md)).
[Who helps whom](who-helps-whom.md) puts this beside every other attempt.

## What we learned

- **Help has an exact meaning here.** A stake helps if, without it, the
  neighbour staked on would have been more likely to lose its node. Only
  revolutions and pooled alliances make help possible. Help is cooperation
  only if it goes to kin, comes back or goes round more than chance would have
  it.
- **The precondition is there; the behaviour is not.** Kin live together, so
  kin selection could work. But brains cannot see kinship, and they spread
  their stakes as evenly over kin as over strangers.
- **There is no future to reciprocate in.** Connections persist, but the
  brains at their ends are replaced almost every game. Of the five roads to
  cooperation, the rules close direct reciprocity almost completely.
- **The equal exchanges are handshakes.** They keep connections alive, at the
  price the rules set, and need no cooperation to explain.
- **For open-ended evolution, the rules have to change, not the brains.** An
  agent would need to keep its identity when its node is taken, to recognise
  kin, and to commit to a partner.

## Where to go next

- [Gliders, and who helps whom](gliders-and-who-helps-whom.md): help counted
  by the rules, in B1, in B2 and among allies.
- [Who helps whom](who-helps-whom.md): what every page found about
  cooperation, side by side.
- [Other contests: a lottery, and alliances](another-game.md): kin that sense
  each other, stakes pooled by family, and alliances.
- [What agents say to each other](what-agents-say-to-each-other.md): messages
  that name their writer's genotype.

## How this was made

Every figure of this page says, folded beneath it, which runs it reads and
how. All five are drawn again with `python3 book_figures.py cooperation`, and
their numbers are kept in `book/results/cooperation.json`.

The numbers for the worlds of 10,000 tokens come from the same measurement on
Experiment 2: thirty worlds, `B1-10000-s001` … `-s030`, of 3,000 iterations
each (`python3 gol_lab.py run E02`). The 26 that lived to the end were read.

### The runs of Experiment 10

Experiment 10 is 30 runs: B1 at 100,000 tokens, [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 to 30, 1,500
iterations each.

<!-- runs E10 -->
> [!info]- The runs behind this page
> **30 runs.** Every condition below is run once for every seed (1 to 30), for 1,500 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it.
>
> - **baseline** — changes nothing: B1 as it is; 100,000 tokens. Runs `B1-100000-s001` … `B1-100000-s030`.
>
> To make these runs again: `python3 gol_lab.py run E10`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | baseline | what it is |
> |---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 100,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | 50, 45, 40, 35, 30 | The widths of the brain's hidden layers, in order from input to output. |
> | [`hidden_activation`](settings.md#hidden_activation) | sigmoid | The curve each hidden unit applies to its sum: `sigmoid`, which passes on at most a quarter of a change, so that deep brains barely respond to what they see, or `tanh`, the same S-shape centred on zero, through which a change still passes in several layers. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 30 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | on | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | on | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`contest`](settings.md#contest) | largest | Who takes a contested node: the `largest` stake as the game weighs it, or a staker drawn by `lottery` with a chance in proportion to its stake (needs revolutions off). |
> | [`allow_gifting`](settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in both baselines; on in a few pilots, in Experiment 24 and in half the worlds of the screen. |
> | [`random_decisions`](settings.md#random_decisions) | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none); `local` lets each estate flow through the network to the nearest living agents, split evenly at every step. |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | off | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.0 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 0 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | off | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 0 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 0 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`link_token`](settings.md#link_token) | off | Whether every link holds one token: its maker pays for it, a lapsing link gives it back to one of its ends, and a link of a dead agent to the end that lives on, so the links can never outnumber the tokens. |
> | [`link_upkeep`](settings.md#link_upkeep) | off | Whether an agent may pay a token an iteration to keep a link, each brain deciding for each neighbour: `paid_or_used` keeps a link that was paid for or used, `paid_only` only one that was paid for; `off`, only use keeps a link. |
> | [`upkeep_to`](settings.md#upkeep_to) | end | Where a link's upkeep goes: to the link's other end (`end`), or into the estate of the dead (`estate`), shared out by the world's redistribution rule. |
> | [`keep_home`](settings.md#keep_home) | 0.0 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 0 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 1 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`mate_pays`](settings.md#mate_pays) | off | Under mix_at_birth, whether the mate pays the child as many tokens as the parent gave it, as far as it can and never its last. |
> | [`self_message`](settings.md#self_message) | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 30 | one run per seed |
> | iterations | 1,500 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

## How our view changed

- **24 October 2026.** Since 11 October, this page said that agents mark
  99.7% of their stakes on neighbours revolutionary at 10,000 tokens too, "to
  the third digit". There it was 99.7% on kin but 99.6% on others. These
  are the medians of the 26 worlds, 0.9965 and 0.9964, in the results of
  9 October
  ([Do agents treat their own kind differently?](#do-agents-treat-their-own-kind-differently)).
- **24 October 2026.** Since 14 October, a note on this page said that two
  agents of one genotype write exactly the same message.
  [What agents say to each other](what-agents-say-to-each-other.md) had
  corrected that the same day: the two messages are the same but for a
  thousandth or so.

<!-- turns -->
---

← [Who helps whom](who-helps-whom.md) · [Groups and cooperation](../README.md#groups-and-cooperation) · [Other contests: a lottery, and alliances](another-game.md) →

> [!info]- Pages that link here (13)
> [Questioning the mechanics](questioning-the-mechanics.md) · [What a brain sees](brain-inputs.md) · [Messages](messages.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [Where tokens flow](where-the-tokens-flow.md) · [Measuring the flow: token flow, lightning and circulation](token-flow.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Like next to like: how alike agents are at a distance](like-next-to-like.md) · [What agents say to each other](what-agents-say-to-each-other.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Who helps whom](who-helps-whom.md) · [Gliders, and who helps whom](gliders-and-who-helps-whom.md) · [One world, many colours: what the viewer can colour by](one-world-many-colours.md)
<!-- /turns -->
