# Measuring lives: births, deaths, ages and estates

> [!question] Questions of this page
> - How is an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") born, and in which four ways can it die?
> - Where do the [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") of the dead go?
> - How old do agents get, and how is survival measured?
> - What does every reproduction phase record?

## In short

How the notebook counts the lives of agents: who is born and how, the four ways to die, where the tokens of the dead go, how old agents grow, and what every reproduction phase records.

Its sections:

- [Births and the four ways to die](#births-and-the-four-ways-to-die)
- [The estate of the dead](#the-estate-of-the-dead)
- [Age, lifetime and survival curves](#age-lifetime-and-survival-curves)
- [Reproduction statistics](#reproduction-statistics)

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
([Building B2: a world that cannot break](building-b2-and-the-search-for-a-healthy-rich-world.md#who-dies-and-why)):

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

<!-- used births-and-the-four-ways-to-die -->
- [One iteration, step by step](one-iteration.md#cleanup)
- [Measuring tokens: inequality and flow](measuring-tokens.md#what-each-of-the-four-says)
- [How runs are made, recorded and costed](runs-and-records.md#the-statistics-file)

> [!info]- And in the Archive (2)
> [B1: lives and wealth](b1-lives-and-wealth.md#four-ways-in-and-out) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md#ways-to-die)
<!-- /used -->

## The estate of the dead

When an agent dies, the tokens it still holds are not lost: the
[cleanup](one-iteration.md#cleanup) pools them as the **estate of the dead** and hands them to the living. The setting
[`redistribution`](settings.md#redistribution) says who inherits. In both
[baselines](b1-and-b2.md), B1 and B2, every token goes to a survivor drawn at
random, each with the same chance. The notebook tried other heirs, and found
that who inherits decides how many agents a world holds, and how poor they
are.

### What happens at a death

Who leaves an estate depends on the rules. In B1 the starved hold nothing, so
the estate is the tokens of the cut off, who make three deaths in four
([What the baseline world is](what-the-baseline-world-is.md)). In B2 no agent
can starve or be cut off, and every agent dies at age 100, so the estate is the
tokens of the old ([Who dies, and why](building-b2-and-the-search-for-a-healthy-rich-world.md#who-dies-and-why)).

**The baseline rule**, `uniform`, deals the pool in one random draw, each token
to a survivor drawn with equal chance. In the median settled B1 world of
100,000 tokens that is about 1,300 tokens per game, 1.3% of all tokens
([Where do the tokens go?](b1-lives-and-wealth.md#the-tokens-of-the-dead)).
Rich and poor have the same chance at every token, so the estate levels the
world: in ten B1 worlds, the more of it a game scatters, the more the
[Gini coefficient](measuring-tokens.md#measuring-inequality-the-gini-coefficient-the-richest-tenth-and-entropy) falls, a correlation of −0.78 between
the changes ([What goes with what](measures-atlases-and-assembly.md#a-whole-world-over-time)).
In B2 worlds of 10,000 tokens the estate is small, about seventy tokens an
iteration among 3,600 agents ([Who inherits?](what-the-switches-did-in-pilots.md#who-inherits)).

### Who may inherit

[`redistribution`](settings.md#redistribution) lists every heir a world can
have: seven rules, from `uniform`, the baselines', to `local`, the nearest of
the living, added on 24 October 2026. Round 3 of the screen drew among them,
and [What keeps wealth spread?](what-keeps-wealth-spread.md) reads what each
did there.

### What each heir did

Each rule was played in [pilot worlds](runs-and-records.md#pilot-worlds) of 10,000 tokens, two
seeds each:

- **By wealth** (B2, 1,000 iterations): the world held 2,450 and 2,800 agents
  against 3,400, and otherwise stayed as it was; its brains went on improving
  ([The switches the world was built with](what-the-switches-did-in-pilots.md#revolutions-handover-and-gifts-the-switches-the-world-was-built-with)).
- **To the poorest** (2,000 iterations, as below;
  [Who inherits?](what-the-switches-did-in-pilots.md#who-inherits)): B2 barely changed, at 3,600 to 3,700
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
([Room for the steadiest world](building-b2-and-the-search-for-a-healthy-rich-world.md#room-and-riches-for-the-steadiest-world)). Its worlds
met every mark of the [health bar](measuring-health.md#the-health-bar) and were true spaces, 2.0 to
2.65 times as wide as their [random twins](measuring-space.md#the-random-twin-is-a-world-a-space). But about 5,000
agents shared the 10,000 tokens, 1.9 each, and the
diversity score fell from 0.54–0.75 to 0.32–0.41. Experiment 32 found the
arithmetic ([Births as rich as the neighbours](building-b2-and-the-search-for-a-healthy-rich-world.md#births-as-rich-as-the-neighbours)):
such a world holds about a hundred agents for every birth an iteration, and the
estates paid for 51 to 53 births an iteration, where the same world without
them had 12 to 23.

### Where it is used (the estate of the dead)

<!-- used the-estate-of-the-dead -->
- [A screen of random worlds](a-screen-of-random-worlds.md#why-draw-worlds-at-random)
- [How the drawn settings shape a world](how-the-drawn-settings-shape-a-world.md#in-short)
- [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md#the-shortlist)
- [Questioning the mechanics](questioning-the-mechanics.md#the-share-out-lottery)
- [Every setting](settings.md#redistribution)
- [The rules in detail](rules-in-detail.md#the-two-streams-of-a-run)
- [What keeps wealth spread?](what-keeps-wealth-spread.md#the-question)
- [Physical inspiration](physical-inspiration.md#where-the-speed-limit-is-broken)
- [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md#b1-and-b2-side-by-side)
- [Measuring space: dimension and the random twin](measuring-space.md#examples)
- [Local rules only](local-rules-only.md#local-rules-only)
- [How far does a difference travel?](how-far-does-a-difference-travel.md#in-short)
- [Particles: patterns that outlast what carries them](particles.md#for-the-physics-aim)
- [Brains replaced by their average](brains-replaced-by-their-average.md#what-we-expect)
- [How big should a brain be?](how-big-should-a-brain-be.md#how-it-is-tested)
- [A source and a sink](a-source-and-a-sink.md#where-a-source-could-sit)
- [Rules without numbers](rules-without-numbers.md#links-that-hold-a-token)
- [Measuring health: criteria, the health bar and diversity](measuring-health.md#examples)

> [!info]- And in the Archive (6)
> [What B1 is, and what stood in its way](what-the-baseline-world-is.md#the-road-from-here) · [B1: lives and wealth](b1-lives-and-wealth.md#the-tokens-of-the-dead) · [B1: the game and its flows](b1-the-game-and-its-flows.md#the-estate-of-the-dead) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md#where-the-tail-of-the-tokens-could-come-from) · [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md#where-to-go-next) · [Measures, atlases and assembly (side studies)](measures-atlases-and-assembly.md#a-whole-world-over-time)
<!-- /used -->

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
the same. The right way is the **Kaplan–Meier estimate** (Kaplan and Meier 1958;
[Survival curves](#survival-curves-kaplanmeier)). It counts an unfinished
life as "survived at least this long", and no further.

### Survival curves (Kaplan–Meier)

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

Why the living are older than the dead, the ages of a steady world, the risk
of dying at each age and cohorts were measured only by studies now in the
Archive ([Lives and ages](measures-of-the-archived-studies.md#lives-and-ages)).

### Where it is used (age)

<!-- used age-lifetime-and-survival-curves -->
- [The ideas this builds on](the-ideas-this-builds-on.md#measuring)
- [Did the brains get better? Round 3's eras against each other](round-3-eras-against-each-other.md#why-play-eras-against-each-other)
- [The world: tokens, agents and a network](the-world.md#agents-and-tokens)
- [Measuring evolution: genotypes, eras and the open-ended test](measuring-evolution.md#lifetimes-of-genotypes)
- [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md#playing-against-the-past)
- [How fast should brains change?](how-fast-should-brains-change.md#in-short)
- [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md#is-evolutions-play-the-best-there-is)
- [Other contests: a lottery, and alliances](another-game.md#does-a-world-of-allies-keep-evolving)
- [Rules without numbers](rules-without-numbers.md#did-the-worlds-that-lived-keep-evolving)
- [How worlds are measured](how-worlds-are-measured.md#step-6--how-long-things-last)
- [Every simulation: what was run, and what was measured](every-simulation.md#what-was-measured-on-what)
- [Statistics: the tools of the notebook](statistics.md#distributions-on-loglog-axes)

> [!info]- And in the Archive (5)
> [What B1 is, and what stood in its way](what-the-baseline-world-is.md#the-baseline-world-in-numbers) · [B1: lives and wealth](b1-lives-and-wealth.md#the-picture) · [B1: brains, lineages and messages](b1-brains-lineages-and-messages.md#genotypes-come-and-go) · [Cooperation, help and gliders (side study)](cooperation-help-and-gliders.md#what-was-tried-to-make-helping-pay) · [Measures, atlases and assembly (side studies)](measures-atlases-and-assembly.md#agents)
<!-- /used -->

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
([Gifts, with every other rule](what-the-switches-did-in-pilots.md#gifts-with-every-other-rule)), in
Experiment 24 and in half the worlds of
[A screen of random worlds](a-screen-of-random-worlds.md).

### Where it is used (reproduction statistics)

<!-- used reproduction-statistics -->
- [How runs are made, recorded and costed](runs-and-records.md#the-statistics-file)

> [!info]- And in the Archive (1)
> [B1: lives and wealth](b1-lives-and-wealth.md#reproduction-over-a-worlds-life)
<!-- /used -->

## Where to go next

- [Who inherits?](what-the-switches-did-in-pilots.md#who-inherits): the experiment behind most of this page.
- [Births as rich as the neighbours](building-b2-and-the-search-for-a-healthy-rich-world.md#births-as-rich-as-the-neighbours): why
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
    ([Building B2: a world that cannot break](building-b2-and-the-search-for-a-healthy-rich-world.md#building-b2-a-world-that-cannot-break)).
    Under its rules nobody starves or is cut off, and every agent dies of age,
    at 100. A birth needs at least ten tokens, from a parent with fewer than
    eight connections. Since 23 October 2026 the page says which rules each
    statement holds for.
- *Reproduction statistics*:
  - **8 October 2026.** This page said that gifting was off in every run of the
    notebook, so that the three gift fields were absent.
  - **15 October 2026.** Pilots with gifts were played for
    [The switches the world was built with](what-the-switches-did-in-pilots.md#revolutions-handover-and-gifts-the-switches-the-world-was-built-with).
    More pilots followed, then Experiment 24 and half the worlds of
    [A screen of random worlds](a-screen-of-random-worlds.md).
  - **23 October 2026.** Reading the runs' statistics files showed that the gift
    fields are never absent from a row of reproduction. Where gifting is off
    they are 0; only the rows of the game leave them empty. The page now says
    both.
- **12 November 2026.** Moved to [Measures of the archived
  studies](measures-of-the-archived-studies.md): four sections on age that only
  pages of the Archive use: why the living are older than the dead, the ages of
  a steady world, the risk of dying at each age, and cohorts.

## References

- Kaplan, E. L. and Meier, P. (1958). Nonparametric estimation from
  incomplete observations. *Journal of the American Statistical Association*
  53, 457–481.

<!-- turns -->
---

← [The rules in detail](rules-in-detail.md) · [Reference: the rules and measures in detail](../README.md#reference-the-rules-and-measures-in-detail) · [Measuring tokens: inequality and flow](measuring-tokens.md) →

**Also about:** [How the simulation works](../README.md#how-the-simulation-works), [Changing the rules](../README.md#changing-the-rules), [Methods and statistics](../README.md#methods-and-statistics).

> [!info]- Pages that link here (20)
> [The ideas this builds on](the-ideas-this-builds-on.md) · [The world: tokens, agents and a network](the-world.md) · [One iteration, step by step](one-iteration.md) · [Questioning the mechanics](questioning-the-mechanics.md) · [Every setting](settings.md) · [Glossary](glossary.md) · [What keeps wealth spread?](what-keeps-wealth-spread.md) · [Measuring tokens: inequality and flow](measuring-tokens.md) · [Physical inspiration](physical-inspiration.md) · [Measuring evolution: genotypes, eras and the open-ended test](measuring-evolution.md) · [How runs are made, recorded and costed](runs-and-records.md) · [How worlds are measured](how-worlds-are-measured.md) · [Statistics: the tools of the notebook](statistics.md) · [B1: lives and wealth](b1-lives-and-wealth.md) · [B1: the game and its flows](b1-the-game-and-its-flows.md) · [B1: brains, lineages and messages](b1-brains-lineages-and-messages.md) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) · [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md) · [Measures, atlases and assembly (side studies)](measures-atlases-and-assembly.md) · [Measures of the archived studies](measures-of-the-archived-studies.md)
<!-- /turns -->
