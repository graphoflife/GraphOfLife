# Measuring lives: births, deaths, ages and estates

> [!question] Questions of this page
> - How is an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") born, and in which four ways can it die?
> - Where do the [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") of the dead go?
> - How old do agents get, and how is survival measured?
> - What does every reproduction phase record?

## In short

How the notebook counts the lives of agents: who is born and how, the four ways to die, where the tokens of the dead go, how old agents grow, and what every reproduction phase records.

It gathers what were 4 pages of their own, one section each:

- [Births and the four ways to die](#births-and-the-four-ways-to-die): After a world's founders, every agent enters it the same way: a parent gives it tokens in a reproduction phase, and it is born.
- [The estate of the dead](#the-estate-of-the-dead): What happens to the tokens of an agent that dies, and the other ways of passing them on that the notebook tried.
- [Age, lifetime and survival curves](#age-lifetime-and-survival-curves): An agent's age is how many iterations ago it was born; its lifetime is how many games it lives through.
- [Reproduction statistics](#reproduction-statistics): Every reproduction phase records each birth.

## Births and the four ways to die

After a world's founders, every agent enters it the same way: a parent gives
it tokens in a reproduction phase, and it is born. In B1 it leaves in one of
four ways: it **starves** or it is **cut off**, after reproduction or after
the game. Births and deaths are counted per hundred agents alive when their
phase began. In B2 none of the four can happen, and every agent dies of old
age, at 100 iterations.

### Birth

A birth is a split. A parent gives some of its own tokens to a child. The
child takes a new place nearby, with a copy of the parent's [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.").

Here is the exact rule
([One iteration, step by step](one-iteration.md#phase-1-reproduction)). In
the reproduction phase, every agent holding at least one token decides how
many of them to give a child. If that number is at least 1, a child is born.
An agent has at most one child in a phase. Every agent but the founders a
world starts with ([The starting ring](rules-in-detail.md#the-starting-ring)) was born this way.

The run writes a row of statistics after every phase
([What a run records](runs-and-records.md#what-a-run-records)). The phase-1 row counts the
children as `births`: every child born, also those the cleanup removes at
once. B2 asks more of a birth ([In B2](#in-b2)).

### The four ways to die

After each phase the world is tidied up. Whoever holds nothing leaves it, and
so does every part of the network that has come loose from the main piece.

Exactly: every death happens in a **cleanup**, and a cleanup runs after each
of the two phases ([One iteration, step by step](one-iteration.md#cleanup)).
In B1 it removes an agent for one of two reasons:

- **starved**: it holds no tokens;
- **cut off**: it holds tokens, but is not in the largest connected piece of
  what is left once the starved are gone. The statistics call it `orphaned`.

Two reasons and two phases make four ways to die:

| | after reproduction (phase 1) | after the game (phase 2) |
|---|---|---|
| **starved** | a parent that gave its child every token it had | an agent on whose node nobody staked anything, not even itself |
| **cut off** | mostly a newborn joined to nobody; also any piece cut loose by a handover, or by the death of a parent that gave everything | anyone whose part of the network came loose from the largest piece: when unused [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") were cut, or when a starved agent that joined it to the rest was removed |

A handover is a parent moving one of its own connections to its newborn
child. In B1, only parents can starve in reproduction. Every agent alive after
a game holds at least one token. And in B1's reproduction phase, an agent
loses tokens only by giving them to its child.

Each row of the statistics counts its phase's deaths as `starved` and
`orphaned`. So the four counts are `starved` and `orphaned` in the phase-1
rows and in the phase-2 rows.

The starved hold no tokens. The cut off do. Their tokens are pooled and dealt
out among the survivors, each token to a survivor drawn at random.
[The estate of the dead](measuring-lives.md#the-estate-of-the-dead) says more, and
[Seeds and the random stream](rules-in-detail.md#how-a-draw-becomes-a-decision)
how such a draw is made.

### Rates per hundred

A count alone says little: a world of 2,000 agents has more births than one
of 1,000. To compare, a count is divided by the number of agents there were
**when the phase began**, the row's `nodes_before`, and multiplied by 100:

$$
\text{births per hundred agents} = 100 \cdot \frac{\texttt{births}}{\texttt{nodes\_before}}
\quad\text{(phase-1 rows)},
$$

and the same for each way of dying, in the rows of its phase. For example,
"3.1 births per hundred agents" means: of every hundred agents alive at the
start of a reproduction phase, 3.1 have a child.

Here are the five rates of a settled B1 world. Each is a world's mean over
iterations 500 to 1,499. The table gives the median of the 30 worlds of
100,000 tokens (Experiment 10;
[Births, deaths and ages](b1-lives-and-wealth.md#births-deaths-and-ages)):

| per hundred agents | after reproduction | after the game |
|---|---|---|
| born | 3.5 | — |
| starved | 0.15 | 0.7 |
| cut off | 0.4 | 2.1 |

About as many agents die as are born: the typical settled world hardly
changes in size, while its members come and go
([A world's life](b1-lives-and-wealth.md#a-worlds-life)). Three deaths in four are
cuts. The worlds of 10,000 tokens (Experiment 2) had a little less of
everything: 3.1 births per hundred agents.

### In B2

B2's rules were made so that none of the four can happen
([Building B2: a world that cannot break](a-world-that-cannot-break.md#who-dies-and-why)):

- **Nobody starves.** In the game a share of every agent's tokens stays on
  its own node, at least one token ([`keep_home`](settings.md#keep_home)).
  And a parent always keeps at least one token.
- **Nobody is cut off.** A newborn is always joined to its parent, and to two
  of the parent's [neighbours](the-world.md#the-network "An agent joined to another by a connection.") ([`birth_links`](settings.md#birth_links)). The
  neighbours of an agent that dies are joined up again
  ([`mend_on_death`](settings.md#mend_on_death)). An unused connection is cut
  only if its two ends stay joined through neighbours they share
  ([`prune_needs`](settings.md#prune_needs)). And no parent hands over a
  connection.

Instead, an agent dies of **old age** ([`max_age`](settings.md#max_age)). It
dies in the cleanup after the reproduction phase of the iteration in which it
turns 100. So every agent of B2 lives exactly 100 iterations. Its tokens are
shared out like any other estate. The statistics file has no field for these
deaths: a frame's `cleanup` record counts them as `aged`.

B2 also asks more of a birth. A child needs at least 10 tokens
([`min_child`](settings.md#min_child)), and the parent keeps its last token.
A parent with 8 connections or more has no child
([`valence`](settings.md#valence)).

Since every agent lives 100 iterations, a B2 world keeps its size only if one
agent in a hundred has a child in each iteration. In the ten B2 worlds of
5,000 tokens of Experiment 14, from iteration 200 to 1,000, births ran at
1.00 to 1.05 per hundred agents. Not one agent starved or was cut off.

### Where it is used (births and the four ways to die)
- [One iteration, step by step](one-iteration.md#what-an-iteration-does-on-average):
  what an average iteration of a settled world does.
- [The first hundred iterations](b1-lives-and-wealth.md#the-first-hundred-iterations):
  births and deaths in a world's youth.
- [Births, deaths and ages](b1-lives-and-wealth.md#births-deaths-and-ages): the rates over a
  world's life.
- [Building B2: a world that cannot break](a-world-that-cannot-break.md#who-dies-and-why):
  which deaths a world needs.

## The estate of the dead

When an agent dies, the tokens it still holds are not lost: the
[cleanup](one-iteration.md#cleanup) pools them as the **estate of the dead** and hands them to the living. The setting
[`redistribution`](settings.md#redistribution) says who inherits. In both
[baselines](b1-and-b2.md), B1 and B2, every token goes to a survivor drawn at
random, each with the same chance. The notebook tried five other heirs, and found that who
inherits decides how many agents a world holds, and how poor they are.

### What happens at a death

Every death happens in the cleanup after a phase of an iteration
([Births and the four ways to die](measuring-lives.md#births-and-the-four-ways-to-die)). Agents with no
tokens starve, and agents no longer joined to the largest connected piece of
the network are cut off; under [`max_age`](settings.md#max_age), agents also
die when they reach that age. The tokens the dead still hold are pooled and
dealt out among the survivors, so the world's total never changes.

Who leaves an estate depends on the rules. In B1 the starved hold nothing, so
the estate is the tokens of the cut off, who make three deaths in four
([What the baseline world is](what-the-baseline-world-is.md)). In B2 no agent
can starve or be cut off, and every agent dies at age 100, so the estate is the
tokens of the old ([Who dies, and why](a-world-that-cannot-break.md#who-dies-and-why)).

**The baseline rule**, `uniform`, deals the pool in one random draw, each token
to a survivor drawn with equal chance. In the median settled B1 world of
100,000 tokens that is about 1,300 tokens per game, 1.3% of all tokens
([Where do the tokens go?](b1-lives-and-wealth.md#the-tokens-of-the-dead)).
Rich and poor have the same chance at every token, so the estate levels the
world: in ten B1 worlds, the more of it a game scatters, the more the
[Gini coefficient](measuring-tokens.md#measuring-inequality-the-gini-coefficient-the-richest-tenth-entropy-and-mobility) falls, a correlation of −0.78 between
the changes ([What goes with what](what-goes-with-what.md#a-whole-world-over-time)).
In B2 worlds of 10,000 tokens the estate is small, about seventy tokens an
iteration among 3,600 agents ([Who inherits?](who-inherits.md)).

### Who may inherit

| `redistribution` | the estate goes to |
|---|---|
| `uniform` | every survivor, the same chance at every token (B1 and B2) |
| `by_tokens` | every survivor, with a chance in proportion to what it holds |
| `poorest` | the poorest survivors, raised to one common level as far as the pool reaches |
| `newborns` | the children born this iteration that are still alive, an even chance each |
| `parents` | the parents of those children, once per child |
| `children` | each dead agent's own living children, an even chance each |

`newborns` and `parents` fall back to `uniform` in an iteration without a
birth, and `children` for an estate that finds no living child.

### What each heir did

Each rule was played in [pilot worlds](runs-and-records.md#pilot-worlds) of 10,000 tokens, two
seeds each:

- **By wealth** (B2, 1,000 iterations): the world held 2,450 and 2,800 agents
  against 3,400, and otherwise stayed as it was; its brains went on improving
  ([The switches the world was built with](the-switches-the-world-was-built-with.md)).
- **To the poorest** (2,000 iterations, as below;
  [Who inherits?](who-inherits.md)): B2 barely changed, at 3,600 to 3,700
  agents. B1's worlds grew larger and younger: 2,100 to 2,300 agents against
  the controls' 800 to 1,400, and a mean age of 74 to 76 iterations against
  120 to 210.
- **To the newborns or to the parents:** B2's worlds grew by half, to 5,100 to
  5,300 agents, nine in ten holding three tokens or fewer, and evolution
  stopped: later brains took only 37% to 57% of a world contested with earlier
  ones. B1 became a world of infants, with 55 to 85 agents in a hundred born
  every iteration.
- **To the dead agent's own children:** B2 changed little, and evolution went
  on at least as fast: the brains of iteration 1,000 took 80% and 74% of a
  world contested with those of 500, against 69% and 67% in the controls. B1's
  worlds grew larger and younger, less than with the poorest.

### An example: paying for births

Experiment 31 paid the estate to the newborns, or to the parents, in B2 worlds
whose brains are mixed at birth, with parents as mates
([Room for the steadiest world](room-for-the-steadiest-world.md)). Its worlds
met every mark of the [health bar](measuring-health.md#the-health-bar) and were true spaces, 2.0 to
2.65 times as wide as their [random twins](measuring-space.md#the-random-twin-is-a-world-a-space). But about 5,000
agents shared the 10,000 tokens, 1.9 each, and the
diversity score fell from 0.54–0.75 to 0.32–0.41. Experiment 32 found the
arithmetic ([Births as rich as the neighbours](room-for-the-steadiest-world.md#births-as-rich-as-the-neighbours)):
such a world holds about a hundred agents for every birth an iteration, and the
estates paid for 51 to 53 births an iteration, where the same world without
them had 12 to 23.

### Where it is used (the estate of the dead)
- [Who inherits?](who-inherits.md): four of the heirs, in B1 and in B2.
- [What holds a world's size](what-holds-a-worlds-size.md): estates paid to
  births among the rules that set how many agents a world holds.
- [Where tokens go](b1-the-game-and-its-flows.md#the-estate-of-the-dead): the estate beside
  the other flows of tokens.

## Age, lifetime and survival curves

An agent's **age** is how many iterations ago it was born; its **lifetime** is
how many games it lives through. They answer two different questions: how old
are the agents alive now, and how long does an agent live? The answers differ,
because the living are a sample weighted towards long lives. In the B1 worlds
of 100,000 tokens, about half the lives end within ten iterations, yet the
median age of the living is 54.

### Age

An agent's **age** is the number of iterations since it was born. It is 0 in
the iteration of its birth, 1 in the next, and so on. Founders are born at
iteration 0. Every frame records the age of every living agent, as `ages`
([What a run records](runs-and-records.md#what-a-run-records)).

Age belongs to the agent, not to its brain. When an agent's node is won in the
game, the agent keeps its id and its age; only its brain changes. So age is
the age of the agent, not of its genotype. The brain on an old node may have
arrived in the last game
([Questioning the mechanics](questioning-the-mechanics.md#two-roads-of-inheritance)).

### Lifetime

An agent's **lifetime** is how many games it lived through. Say it was born in
iteration *b*, and the last game after which it was still alive was in
iteration *d*. Then its lifetime counts the games from *b* to *d*, both
included:

$$
L = d - b + 1 .
$$

An agent that survives exactly one game has lifetime 1. It is alive after the
game of its birth iteration, and dies in the next iteration, at age 1. In
general, an agent of lifetime *L* dies in iteration *b* + *L*, at age *L*, in
one of that iteration's two cleanups.

An agent that never outlives a game has lifetime 0, and is not counted. That
is a newborn cut off in the cleanup after its birth, or one that dies in the
cleanup after its first game. There are many: in the settled B1 worlds of
100,000 tokens, nearly a third of all newborns
([Births, deaths and ages](b1-lives-and-wealth.md#births-deaths-and-ages)).

### Lives that have not ended

When a run stops, some agents are still alive. Their lifetime is not known,
only that it is **at least** what they have lived so far. Leaving them out
would make lives look shorter than they are, because the long lives are the
ones most likely to be unfinished. Counting them as if they had ended would do
the same. The right way is the **Kaplan–Meier estimate**
([Survival curves](#survival-curves-kaplanmeier)). It counts an unfinished
life as "survived at least this long", and no further.

### Survival curves (Kaplan–Meier)

*What share of agents live at least L iterations?* If every life had ended by
the end of the run, the answer would be a simple count. But some are still
going, and for them we know only that they lasted **at least** so long. The
**Kaplan–Meier estimate** (Kaplan and Meier 1958) uses exactly that much and
no more.

#### The idea: survive one step at a time

To live at least *L* + 1 iterations, a life must first live at least *L*, and
then not end at *L*. So the share surviving to *L* + 1 is the share surviving
to *L*, times one more factor. That factor is the chance of not ending at *L*,
among the lives that got that far. Chaining the steps gives

$$
S(L) = \prod_{\ell < L} \left( 1 - \frac{d_\ell}{n_\ell} \right),
$$

where, for each length ℓ:

- *n*ℓ is the number of lives **at risk** at ℓ: the lives that lasted at
  least ℓ. They ended at ℓ or later, or are still going after lasting ℓ or
  more;
- *d*ℓ is the number of lives that **ended** at exactly ℓ.

In words: multiply, step by step, the shares of the lives still at risk that
did not end there. A life still going at the end of the run, having lasted
*L*₀, counts in *n*ℓ for every ℓ ≤ *L*₀. It is then dropped, without ever
counting as an end. That is how "at least *L*₀" is used: no more, no less.

#### A worked example of survival

Take six agents' lives, in iterations; a + means still alive at the end of
the run: 1, 1, 2, 3+, 4, 6+.

| ℓ | at risk *n*ℓ | ended *d*ℓ | 1 − *d*/*n* | share living longer than ℓ |
|---|---|---|---|---|
| 1 | 6 | 2 | 4/6 | 0.667 |
| 2 | 4 | 1 | 3/4 | 0.500 |
| 3 | 3 | 0 (the 3+ leaves here) | 1 | 0.500 |
| 4 | 2 | 1 | 1/2 | 0.250 |
| 6 | 1 | 0 (the 6+ leaves here) | 1 | 0.250 |

So half of all lives last at least 3 iterations, and a quarter at least 5.
Ignoring the unfinished lives would have given a quarter for "at least 3":
only the 4, of the four finished lives. Counting them as if they ended would
understate the long lives too.

#### Drawing a survival curve

The curve *S*(*L*) starts at 1 and falls in steps. The notebook draws it on
[logarithmic axes](statistics.md#logarithmic-axes-and-power-laws). Then both
the many short lives and the few long ones are visible
([Births, deaths and ages](b1-lives-and-wealth.md#births-deaths-and-ages),
[Genotypes and lineages](b1-brains-lineages-and-messages.md#genotypes-and-lineages)).

The **median lifetime** is the smallest *L* with *S*(*L* + 1) ≤ ½: half of all
lives end by then. In the code, this is `kaplan_meier(lives)` in
`book_figures.py`, where a life is (first iteration, last iteration, still
going).

### Why the living are older than the dead

Picture a hospital where most patients go home after a day, and a few stay a
month. On any day you visit, the long stays fill far more beds than their
number suggests. A patient who stays thirty days is there on thirty days; one
who stays a day, on one.

Agents are the same. Suppose half of all agents live 1 iteration and half live
99. The **average lifetime** is 50. But look at the world at any moment. An
agent of the second kind is alive at 99 times as many moments as one of the
first. So 99 in 100 of the agents you see are long-lived ones: the **living
are a sample weighted by lifetime**.

That is why the living look so old. In the 30 B1 worlds of 100,000 tokens
(Experiment 10), about half of the agents that live through a game are gone
within ten iterations. Two thirds are, if the newborns that never do are
counted too. Yet the median age of the agents alive at the end is 54
([Births, deaths and ages](b1-lives-and-wealth.md#births-deaths-and-ages)).

In general, take a world in a steady state, where lifetimes *L* occur with
frequency *p*(*L*) and have mean μ. The agents alive at a random moment have
lifetimes with frequency

$$
\frac{L \, p(L)}{\mu} :
$$

each lifetime's frequency multiplied by its length, and divided by the mean so
that the frequencies add up to one. So long lives are over-represented in
proportion to their length, and each such agent is equally likely to be at any
point of its life. This is the *inspection paradox* of renewal theory.

The B1 worlds of 10,000 tokens are not exactly in a steady state: their births
slow down for over a thousand iterations. But the effect is the same in kind.
Those of 100,000 tokens are in one from iteration 500 on
([How ages change over a world's life](b1-lives-and-wealth.md#how-ages-change-over-a-worlds-life)).

### The ages of a steady world

The same idea, turned into a formula for the ages of the living. Suppose that
in every iteration, always, *B* newborns live through their first game.
Suppose too that every agent's life follows the same survival curve *S*(*L*),
the share of lives lasting at least *L* iterations. The agents of age *a* alive
after a game were born *a* iterations earlier, and their lives lasted at least
*a* + 1 games. There are *B* · *S*(*a* + 1) of them. Summed over all ages, the
living number *B* · Σ *S*(*L*), and Σ *S*(*L*) over *L* ≥ 1 is the mean
lifetime μ. So in such a world

$$
\text{share of the living of age } a = \frac{S(a+1)}{\mu} ,
$$

and in particular the newborns, of age 0, are a share 1/μ of the living. This
is the **stationary population** of demography. Its ages follow from the
lifetimes alone. A world whose living are older than its lifetimes say has had
more births in the past than now.

Two examples. In the B1 worlds of 100,000 tokens, the lives begun from
iteration 500 to 1,399 last 41.3 iterations on average. So the newborns should
be 1/41.3 = 2.4% of the living, and the median age of the living 55. Over
iterations 500 to 1,499 the worlds had 2.3 to 2.6% newborns, and a median age
of 55 ([How ages change over a world's life](b1-lives-and-wealth.md#how-ages-change-over-a-worlds-life)).

In B2 every agent lives exactly 100 iterations. So in a steady B2 world, each
age from 0 to 99 holds one agent in a hundred. Such a world keeps its size only
if one agent in a hundred has a child in every iteration. B2's worlds in
Experiment 14 had 1.00 to 1.05 births per hundred agents from iteration 200 to
1,000 ([Building B2: a world that cannot break](a-world-that-cannot-break.md#old-age)).

### The risk of dying at each age

The survival curve *S*(*L*) is built from one number per age. An agent
**reaches** age *a* when its lifetime is at least *a*, and its life **ends
there** when its lifetime is exactly *a*. Of the agents whose lives reach age
*a*, the share whose lives end there is

$$
h(a) = \frac{d(a)}{n(a)} ,
$$

with *n*(*a*) the lives lasting at least *a* (finished or not) and *d*(*a*)
the finished lives of exactly *a*. This is the **hazard**, the chance of dying
at age *a* for an agent that got there. To last at least *L* iterations, a
life must get past every age before *L*, so

$$
S(L) = \prod_{a < L} \bigl(1 - h(a)\bigr) .
$$

If the hazard is the same at every age, an agent's past says nothing about its
future: lives are **memoryless**. Then *S*(*L*) falls along a straight line
when *S* is drawn on a logarithmic axis against a plain axis of *L*.

A hazard that falls with age means that those who survive are safer than the
young. The young die most, as in a population with high infant mortality.
Either each agent grows safer as it lives, or the fragile die first and leave
the robust. A hazard that rises with age is ageing in the usual sense.

The worlds show both ends. In the settled B1 worlds of 100,000 tokens the
hazard falls at every age. It is 21% at age 1, 2.7% at 10 and 1.1% at 100
([How ages change over a world's life](b1-lives-and-wealth.md#how-ages-change-over-a-worlds-life)). In B2 it is 0 at
every age below 100, and every agent that reaches 100 dies at it: ageing at
its sharpest.

### Cohorts

Agents born in the same stretch of iterations are a **cohort**. Following each
cohort's lives says how life expectancy changes over a world's life. Cohorts
born late can only be followed to the end of the run. The Kaplan–Meier
estimate leaves their unfinished lives open, rather than cutting them short.

To compare cohorts on equal terms, each is measured by how many of its first
100 iterations an agent lives on average. That is the **restricted mean
lifetime**,

$$
\sum_{L=1}^{100} S(L) ,
$$

the area under the survival curve from *L* = 1 to 100. Every cohort born at
least 100 iterations before the end can be measured by it in full. In the
median B1 world of 100,000 tokens, an agent born in iterations 1 to 24 lived 5
of its first 100 iterations. One born from iteration 500 on lived 28 to 31
([How ages change over a world's life](b1-lives-and-wealth.md#how-ages-change-over-a-worlds-life)).

### Where it is used (age)
- [Births, deaths and ages](b1-lives-and-wealth.md#births-deaths-and-ages): how long agents live,
  and how old the living are, in the B1 worlds.
- [How ages change over a world's life](b1-lives-and-wealth.md#how-ages-change-over-a-worlds-life): the ages of the
  living, the cohorts and the hazard, iteration by iteration.
- [Building B2: a world that cannot break](a-world-that-cannot-break.md#old-age):
  a world in which every agent dies at 100.

## Reproduction statistics

Every reproduction phase records each birth. It notes the parent, the tokens
it held and gave, whom the child was joined to, and the connections the parent
handed over. The row of statistics written after the phase sums these up in a
few numbers. Two of them are shares that answer different questions. One says
how much of itself a typical parent gives; the other, how much of all the
tokens went into children.

### What a phase records

Think of a register of births, one line per child. The frame written after
the reproduction phase keeps it as `decisions.births`
([What a run records](runs-and-records.md#what-a-run-records)):

| key | what it holds |
|---|---|
| `agent`, `child` | the parent's id, and the child's |
| `tokens_before` | the tokens the parent held before it gave any |
| `invested` | the tokens it gave the child |
| `links` | the candidates the child was joined to at birth |
| `handed_over` | the neighbours whose connections the parent chose to hand to the child |

A parent's candidates are the parent itself and its neighbours. A handover
moves one of the parent's connections to its child
([One iteration, step by step](one-iteration.md#phase-1-reproduction)).

### The fields

The statistics of a phase-1 row summarise the register. In the rows of the
game they are empty (`null`).

| field | definition |
|---|---|
| `births` | the number of children born, also those the cleanup removes at once |
| `meanInvestedShare` | the mean, over the parents, of tokens given ÷ tokens held |
| `reproTokenShare` | all tokens given to children ÷ all tokens in the world |
| `meanChildLinks` | the mean number of candidates a child was joined to |
| `handovers` | the number of connections parents chose to hand to their children |
| `gifts`, `giftTokens`, `giftShare` | how many gifts of tokens agents made to neighbours, how many tokens they carried, and those tokens' share of all tokens: 0 when gifting is off, as in both baselines |

### Two shares that answer different questions

`meanInvestedShare` is **per parent**: how much of itself a typical parent
gives. `reproTokenShare` is **per world**: how much of all the tokens went
into children. A world in which a few poor agents each give everything they
have has a high per-parent share and a tiny world share.

**Example.** Three parents holding 2, 4 and 100 tokens give 1, 2 and 10. Each
gives a share of what it holds, ½, ½ and ⅒, and their mean is
`meanInvestedShare` = (½ + ½ + ⅒)/3 ≈ 0.37. Together they give 13 tokens. In
a world of 10,000 tokens, `reproTokenShare` = 13/10,000 = 0.0013.

The baseline's worlds show the same gap. In the median B1 world of 100,000
tokens, over its settled life, parents give 36% of what they hold. Yet all the
tokens given to children in one phase are 1.3% of the world's
([How births build the network](b1-lives-and-wealth.md#how-births-build-the-network)).

### Links of a child

A child's links are chosen among its parent's candidates, **the parent
included**. So `meanChildLinks` counts the link to the parent too, if there is
one. In B2 a newborn is first joined to its parent and to two of the parent's
neighbours, drawn at random ([`birth_links`](settings.md#birth_links)). These
count as well. The connections a child is handed do not.

A child joined to no candidate and handed no connection belongs to no
network. It dies in the cleanup that ends the phase
([Births and the four ways to die](measuring-lives.md#births-and-the-four-ways-to-die)). In the B1 worlds of
100,000 tokens, 11.5% of children are joined to no candidate at birth. Two in
five of these (39%) are handed a connection by their parent, and live. The
rest, some 7% of all children, die at once
([How births build the network](b1-lives-and-wealth.md#how-births-build-the-network)).

### Handovers

`handovers` counts connections that moved from parent to child: the parent
loses them, the child gains them. They never add a connection to the world. A
handover to a neighbour the child is already joined to even removes one: the
parent loses its connection, and the child gains nothing new.

The field counts what the parents chose. A connection that both of its ends
chose to hand over counts twice, but moves only once: by the second move, it
is gone. B2 has no handovers ([`allow_handover`](settings.md#allow_handover)),
so there the field is 0.

### Gifts

With [`allow_gifting`](settings.md#allow_gifting) on, an agent may give some
of what its child left it to its neighbours, in the reproduction phase. The
field `gifts` counts the gifts paid, one for each giver and neighbour. The
field `giftTokens` counts the tokens they carried, and `giftShare` is those
tokens as a share of all tokens. Gifting is off in both baselines, so there
all three are 0. It is on in some pilots
([Gifts, with every other rule](the-switches-the-world-was-built-with.md#gifts-with-every-other-rule)), in
Experiment 24 and in half the worlds of
[A screen of random worlds](a-screen-of-random-worlds.md).

### Where it is used (reproduction statistics)
- [How births build the network](b1-lives-and-wealth.md#how-births-build-the-network): these statistics
  over a world's life, in the B1 worlds.
- [What a run records](runs-and-records.md#what-a-run-records): every field of a run's
  statistics.

## Where to go next

- [Who inherits?](who-inherits.md): the experiment behind most of this page.
- [Births as rich as the neighbours](room-for-the-steadiest-world.md#births-as-rich-as-the-neighbours): why
  a world paid for its births fills with poor agents.
- [Genotypes and the family tree](measuring-evolution.md#lifetimes-of-genotypes): the
  lifetimes of brains, rather than of agents.
- [One iteration, step by step](one-iteration.md#phase-1-reproduction): the
  reproduction phase, rule by rule.
- [How births build the network](b1-lives-and-wealth.md#how-births-build-the-network): who has children,
  and how much they give.

## How our view changed

- *Births and the four ways to die*:
  - **8 October 2026.** This page said that an agent leaves a world in one of
    four ways. It also said that a parent that gives a child at least one
    token has one.
  - **12 October 2026.** B2 was made
    ([Building B2: a world that cannot break](a-world-that-cannot-break.md)).
    Under its rules nobody starves or is cut off, and every agent dies of age,
    at 100. A birth needs at least ten tokens, from a parent with fewer than
    eight connections. Since 23 October 2026 the page says which rules each
    statement holds for.
- *Reproduction statistics*:
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

## References

- Kaplan, E. L. and Meier, P. (1958). Nonparametric estimation from
  incomplete observations. *Journal of the American Statistical Association*
  53, 457–481.
