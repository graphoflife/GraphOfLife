# Births, deaths and ages

> [!question] Questions of this page
> - How long does an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") live, and why do most agents die young?
> - How often is an agent born, and which of the four ways of dying is the
>   most common?
> - How old are the living, and why is that a different question from how
>   long agents live?
> - Do the ages settle as a world lives on, and do they settle the same way
>   in a world a tenth the size?
> - At which age is an agent most likely to die? Do agents grow old?
> - Who are the old: the rich, the well connected, or neither?

## In short

From iteration 500 on, a world of 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") keeps about 14,400 agents, but
replaces about 3.5 in a hundred of them in every iteration. Three in four of
those that die are cut off from the network, and the rest starve. Most die
young: nearly a third of all newborns die in the iteration of their birth, and
two thirds within ten iterations. After that, the risk of dying falls with age,
to about one game in a hundred, so agents do not grow old. By iteration 500 the
ages, the lifetimes and the risk at every age have stopped changing. On average
the old are richer and better connected, because the well connected live long;
age itself brings no [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.").

## The picture

A Graph of Life world lives in steps called iterations. In each, its agents may
first have a child. Then all of them play a game, staking their tokens on their
own nodes and their [neighbours](the-world.md#the-network "An agent joined to another by a connection.")' ([One iteration, step by
step](one-iteration.md)). After a youth of a few hundred iterations, a world
settles: it keeps about the same number of agents ([A world's
life](a-worlds-life.md#a-worlds-life)). Yet its members come and go. In every
iteration some agents are born and others die, and the two nearly balance. So
how long does an agent live?

Picture a town where most babies die in their first days. Anyone who gets
through childhood can expect a long life, and nobody dies of old age. A settled
world is such a town. Nearly a third of all newborns die in the very iteration
of their birth. The risk of dying is highest just after birth. It falls with
every iteration an agent survives, down to about one game in a hundred. It
never rises again: agents do not grow old.

Why do most die young? Many newborns never get a hold on the network. A
parent's [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") may join its child to no one, and then the child is removed at
once. Those that do get a hold are safer with every iteration they last, and
the well connected last longest. Living long does not bring an agent
connections; it is the connections that help it live long.

There are two ways to ask how old agents get, and they give different answers.
Follow every agent from its birth to its death, and you get its **lifetime**.
About half of those that live through their first game are gone within ten
iterations. Or stop a world at one moment, and look at the **ages of the
living**. Their [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above."), the age that half of them are below, is about 55
iterations. The living are older because an agent that lives long is alive at
many moments, so it is counted at many of them. A hospital ward is the same. On
any day, long-stay patients fill most of its beds, though most patients go home
within a day or two ([Age and lifetime](age-and-lifetime.md)).

The worlds of this page are the thirty of 100,000 tokens of [Thirty
worlds](a-worlds-life.md#thirty-worlds). They are the first [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2."), [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it."), run
with the [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 to 30 for 1,500 iterations each: Experiment 10. A settled
world holds about 14,400 agents. Unless a passage says otherwise, a number is
measured over a world's [settled life](settled-life.md "A world from iteration 500 to its end, after its youth, where it is measured."), from iteration 500 on. The typical world
is the median of the thirty. This page was first written from the thirty worlds
of 10,000 tokens; [Two sizes](#two-sizes) sets them beside the large ones.

## Four ways in and out

An agent enters a world in one way: it is born in a reproduction phase, with
tokens its parent gives it. It leaves in one of four ways. Every death happens
in a cleanup, at the end of each of the two phases of an iteration ([One
iteration, step by step](one-iteration.md)). An agent is:

- **cut off in reproduction**: mostly a newborn that its parent joined to no
  one;
- **starved in reproduction**: a parent that gave its last token to its child;
- **starved in the game**: no one, not even the agent itself, staked a token
  on its node;
- **cut off in the game**: no tokens crossed the connections that tied its
  part of the network to the rest, so they were cut.

[Births and the four ways to die](births-and-deaths.md) defines each. Each is
counted per hundred agents alive when its phase began:

<!-- figure births/rates -->
![Births and deaths per 100 agents](../figures/births/rates.svg)

**Births and deaths per 100 agents.** Each line is the median over the worlds, in stretches of 25 iterations, of a count divided by the agents present when the phase began, times 100. The first iterations run far above the top of the axis ([The first hundred iterations](a-worlds-life.md#the-first-hundred-iterations)).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. For every row, divide the count (`births` or `orphaned` in rows with `phase` = 1; `starved` or `orphaned` in rows with `phase` = 2) by the row's `nodes_before`, the agents present when that phase began, and multiply by 100.
> 2. Cut the 1,500 iterations into 60 stretches of 25; take each run's mean in each.
> 3. Draw the median over the runs that have a value in the stretch.
>
> **To make it again:** `python3 book_figures.py births`.
<!-- /figure -->

From iteration 500 on, the median world sees 3.5 births per hundred agents in
every reproduction phase. It loses 0.4 per hundred to cutting off in
reproduction, 0.7 to starvation in the game and 2.1 to cutting off in the game.
Starving in reproduction is rarer still, 0.15 per hundred, and it is not drawn.
So of every four agents who die, three die because the network around them came
apart. The fourth starves, most often because nobody staked on its node in the
game.

Each world counts once here. Each rate is a world's mean over its settled life,
and the typical world is the median one. The rates are steady from about
iteration 500 on, after falling from the youth ([The first hundred
iterations](a-worlds-life.md#the-first-hundred-iterations)). And they differ
little between worlds: the middle half of the worlds have between 3.1 and 3.8
births per hundred agents. That is less obvious than it looks. Worlds whose
brains never change end up anywhere from no births at all to about one child
per agent per iteration ([Do the brains matter?](do-the-brains-matter.md)).

## How long agents live

Not every newborn lives long enough to be followed. From iteration 500 on, 15.2
million agents were born in the thirty worlds. Of these, 4.7 million — nearly a
third — died in the very iteration they were born in. Some 2 million were cut
off in the cleanup of the reproduction phase, most of them because they were
joined to no one. The rest died in their first game, starved or cut off.

An agent's **lifetime** counts the games it lived through ([Age and
lifetime](age-and-lifetime.md)). It runs from the iteration of its birth to the
last iteration after whose game it was alive, both included. The 4.7 million
were never alive after a game, so the curves of this page leave them out.

The curve below, a **survival curve**, follows the other 10.4 million. These
are all the agents born at iteration 500 or later that were still alive after
the game of their birth iteration. Some were still alive when their run ended.
Their lives are counted as having lasted *at least* that long. The Kaplan–Meier
estimate does this: it uses what is known of an unfinished life, and no more
([Survival curves (Kaplan–Meier)](kaplan-meier.md)). The share that lived at
least *L* iterations:

<!-- figure births/lifetimes -->
![How long agents live](../figures/births/lifetimes.svg)

**How long agents live.** Of all agents born at iteration 500 or later in the 30 worlds and alive after at least one game — the newborns cut off in the reproduction phase of their birth, or dead in their first game, are left out — the share that lived at least L iterations, for every L. A life is counted from the iteration of birth to the last iteration after whose game the agent was alive, both included. Lives still going when a run ended are counted as far as they went (Kaplan–Meier). Both axes are logarithmic.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read every frame with `phase` = 2. For every agent id note its birth iteration (the frame's `iteration` minus its `ages` entry) and the last iteration it appears.
> 2. Keep agents born at iteration 500 or later. An agent still present in the last frame is censored: its life is known to be at least that long.
> 3. Compute the Kaplan–Meier estimate S(L) = Π over lifetimes ℓ < L of (1 − d(ℓ) / n(ℓ)), with d(ℓ) the lives that ended at ℓ and n(ℓ) those still at risk.
>
> **To make it again:** `python3 book_figures.py births`.
<!-- /figure -->

Both axes are logarithmic: each step along them multiplies by the same factor
([Logarithmic axes and power laws](logarithmic-axes.md)). On them, the curve
shows several things at once:

- 79% of these agents live at least 2 iterations, 58% at least 5, and 48% at
  least 10. **About half of them are gone within ten iterations**, and their
  median lifetime is 8 iterations. Counted over all 15.2 million newborns,
  with the third that died in their birth iteration, **two thirds are gone
  within ten iterations**. The median lifetime of all newborns is 2.
- 25% live at least 50 iterations and 13% at least 100. The curve falls slowly
  there: an agent that has survived its first ten iterations has good chances
  of surviving many more.
- Then it drops: 0.26% live at least 500 iterations, and 0.003% — about one in
  thirty thousand — at least 1,000.

A curve that fell along a straight line on these axes would be a [power law](power-law-fit.md "A quantity whose frequency falls as a power of its size: few huge values, many small ones.").
This one bends: early deaths are common, and long lives exist, but not without
limit. The bend comes from the risk of dying: past an age of about a hundred,
it holds near one game in a hundred. A steady risk makes the curve fall ever
more steeply on these axes ([The risk of dying at each
age](#the-risk-of-dying-at-each-age)).

## How old the living are

The lifetimes above follow every agent that lived through its first game,
whatever became of it later. The living at one moment are a different sample,
weighted towards long lives ([The picture](#the-picture)). So they are older
than the typical agent at death.

<!-- figure births/ages -->
![Ages of the living at the end](../figures/births/ages.svg)

**Ages of the living at the end.** The ages of all 394,056 agents alive after the last game of the 30 worlds, in bins that double in width: age 0 is born in this iteration, age 1 in the one before, and so on.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read each run's last frame (iteration 1,499, `phase` 2) and its `ages`.
> 2. Count the ages in the bins 0, 1, 2–3, 4–7, …, and divide by the number of agents.
>
> **To make it again:** `python3 book_figures.py births`.
<!-- /figure -->

After the last game of the thirty worlds, 394,056 agents were alive. Pooled
over all thirty worlds, their median age was 54 iterations. Of them, 2.4% had
been born in that very iteration, 31% were 100 iterations old or more, and 37
were over a thousand.

Is this how a world's ages always look, or only a moment in a drift? The next
section follows the ages through the whole life of the worlds.

## How ages change over a world's life

A world whose size holds still is not necessarily in a steady state. If births
slow while deaths slow too, the population can stay the same size while it
grows older. And births do slow, over a world's first few hundred iterations
([Births and deaths slow down](a-worlds-life.md#births-and-deaths-slow-down)).
An older population is a different one. Older agents have had more time to
gather connections and tokens, and their brains have survived more games. For
evolution, what matters is how long a brain lasts and what it does in that
time. For a picture of a world as a physical system, what matters is whether it
ever reaches an equilibrium.

So this section follows both views of age from a world's start to its end. How
old are the living at each moment? And how long do the agents born at each
moment live? One number per age connects the two, the **hazard**: of the agents
that reach an age, the share that die at it.

### How old the living are, iteration by iteration

At iteration 0 every agent is a [founder](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") or a founder's first child, and all are
of age 0. From then on, after every game, the run records a frame, the whole
world as it stands, with the age of every agent alive. The figure follows three
ages: the median, the oldest, and the age that nine in ten of the living are
below, the **90th percentile**:

<!-- figure ages/over-time -->
![How old the living are](../figures/ages/over-time.svg)

**How old the living are.** After every game of the 30 worlds. Left: the median age of the agents alive, and the age that nine in ten of them are below (the 90th percentile). Right: the age of the oldest agent alive; the dashed diagonal is the age of an agent alive since iteration 0, which only the oldest can follow. For each, the line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames after every game (frame `2·t + 1`, `gol_store.read_frame`), read in order by `book_chapters.ages.lifelines`: every agent's `ages` entry gives its birth iteration (the frame's iteration minus its age), and the last frame it is in its last iteration alive.
>
> 1. After every game, the median, the 90th percentile and the largest of `ages`.
> 2. Cut the 1,500 iterations into 600 stretches of 2.5 and take each world's mean in each stretch.
> 3. In each stretch, take the median, the 25th and 75th and the 5th and 95th percentiles of those means over the worlds that still have a value there (a world that died has none after its death).
>
> **To make it again:** `python3 book_figures.py ages`.
<!-- /figure -->

The median age climbs for two hundred iterations — 12 at iteration 50, 28 at
100, 48 at 200 — and then stops. It is 46 at iteration 300, 51 at 500, 57 at
1,000 and 51 at the last iteration. That last number is the median world's own;
the 54 of [How old the living are](#how-old-the-living-are) pools the agents of
all thirty worlds. Over the settled stretch from 500 to 1,499, the median
world's median age is **55 iterations**. The middle half of the worlds lie
between 51 and 60. The 90th percentile stops a little later, near 200: it is
182 at iteration 300, 216 at 1,000 and 183 at the end. The ages of the living
stop changing long before a world's life is over.

The oldest agent behaves differently. For three hundred iterations it is
someone alive since the start: at iteration 300, the oldest agent of the median
world is 296 iterations old. Then it falls behind the figure's diagonal, which
marks an agent alive since iteration 0. It is 487 at iteration 500, 703 at 750
and 825 at 1,000. From there it goes no further: 841 at 1,250, 820 at the end.

The agents of the start do not last. Of the agents born in iteration 0, the
median world keeps 37 at iteration 25, 2 or 3 at iteration 100, and none at
200. In the median world the last of them died at iteration 161; in one world
the last died at iteration 24, in another at 702.

### Why the oldest agent stops near 850

Why should the oldest stop near 850? Not because agents wear out: [The risk of
dying at each age](#the-risk-of-dying-at-each-age), below, finds no sign of
that. It is a matter of numbers.

A settled world has about 500 births in every iteration: 3.5 per hundred
agents, of some 14,400. But nearly a third of the newborns die in the iteration
they are born in ([How long agents live](#how-long-agents-live)). So about
*B* = 345 in every iteration live through their first game: 2.4 per hundred agents.
A share *S*(100) = 0.13 of these live to an age of 100 iterations; the curve of
[How long agents live](#how-long-agents-live) counts exactly these agents. And
from then on, about *h* = 1% of those left die in every game, whatever their
age.

Now count the agents alive that are older than *L*. Every past iteration added
*B* lives. Of the lives begun more than *L* iterations ago, a share *S*(100)
reached an age of 100. Every game after that kept a share 1 − *h* of them.
Adding these up over all past iterations gives about

$$
N(L) \approx \frac{B \, S(100)}{h} \, (1-h)^{L-100}
= \frac{345 \times 0.13}{0.01} \, 0.99^{\,L-100}
\approx 4{,}500 \times 0.99^{\,L-100} ,
$$

which falls to one, a single agent older than *L*, at *L* ≈ 100 + ln 4,500 /
−ln 0.99 ≈ 940. That is near the 820 to 850 that the oldest agent reaches,
given that *h* is not quite constant. All 500 births with the same *S*(100)
would give *L* ≈ 975, too high. The share 0.13 is of the agents that live
through their first game, not of all newborns.

A world holds a few agents near that age at any time, and almost never one much
older. The oldest agent is where the long tail of lifetimes runs out of agents.

### The living, by age

Sorting the living into age classes that widen fourfold shows the whole age
structure at a few moments:

<!-- figure ages/pyramids -->
![The living, by age, at six moments](../figures/ages/pyramids.svg)

**The living, by age, at six moments.** All agents alive after the games of 100, 250, 500, 750, 1,000 and 1,475, pooled over the 30 worlds and sorted by age in classes that widen fourfold: born in that very iteration (age 0), 1 to 3, 4 to 15, 16 to 63, 64 to 255, and 256 iterations or more. Each column adds up to 1.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames after every game (frame `2·t + 1`, `gol_store.read_frame`), read in order by `book_chapters.ages.lifelines`: every agent's `ages` entry gives its birth iteration (the frame's iteration minus its age), and the last frame it is in its last iteration alive.
>
> 1. Every 25 iterations, count the agents of each age class; pool the worlds.
> 2. Divide each class by the agents of the moment and stack.
>
> **To make it again:** `python3 book_figures.py ages`.
<!-- /figure -->

At iteration 100, half of the living (51%) are 16 to 63 iterations old. They
were born in iterations 37 to 84, as the boom of [The first hundred
iterations](a-worlds-life.md#the-first-hundred-iterations) turned into its
crash. Few agents born since have lasted. At iteration 250 that generation has
moved on, and 39% of the living are 64 to 255 iterations old. From iteration
500 on, the columns hardly change:

- about 2.5% of the living were born in that very iteration;
- 5% are 1 to 3 iterations old;
- 13 to 15% are 4 to 15;
- about a third are 16 to 63;
- 37 to 39% are 64 to 255;
- and 5 to 8% are older.

The shape of the population is set by iteration 500, and it holds.

### How long the agents of each period live

The ages of the living mix agents born at many times. Lifetimes separate them.
Group the agents that lived through their first game by the period in which
they were born: each group is a **cohort**. Then follow each group to its end,
as in [How long agents live](#how-long-agents-live):

<!-- figure ages/cohorts -->
![How long agents live, by when they were born](../figures/ages/cohorts.svg)

**How long agents live, by when they were born.** Of all agents born in each period of the 30 worlds' lives that were alive after at least one game, the share that lived at least L iterations. A life is counted from the iteration of birth to the last iteration after whose game the agent was alive, both included; lives still going at the end of a run are counted as far as they went (Kaplan–Meier, as in [Births, deaths and ages](births-deaths-and-ages.md)). A period's line stops at the longest life its agents could have had before the run ended. Both axes logarithmic.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames after every game (frame `2·t + 1`, `gol_store.read_frame`), read in order by `book_chapters.ages.lifelines`: every agent's `ages` entry gives its birth iteration (the frame's iteration minus its age), and the last frame it is in its last iteration alive.
>
> 1. For every agent, its birth iteration b and its last iteration alive d; d − b + 1 iterations lived; alive after the run's last game: still going.
> 2. Sort the agents by b into the periods shown, pool the worlds, and estimate S(L) by Kaplan–Meier (`book_chapters.ages.survival`).
>
> **To make it again:** `python3 book_figures.py ages`.
<!-- /figure -->

Agents born in iterations 1 to 49 lived short lives: their median lifetime was
2 iterations, 15% reached 10 and 1.5% reached 100. The median lifetime is the
longest *L* that more than half of the lives reach. Each later period did
better. The median lifetime was 4 iterations for those born in iterations 50 to
99, 6 for 100 to 249, and 7 for 250 to 499. From iteration 500 on, the
improvement stops. The three cohorts born from iteration 500 on have the same
curve. Each has a median lifetime of 8 iterations; 47 to 48% live at least 10,
and 13 to 14% at least 100. Their three lines lie on top of one another.

### Life expectancy, birth by birth

A finer view takes every stretch of 25 birth iterations, in each world on its
own. How many of their first 100 iterations do its agents live, on average?
This is the **restricted mean lifetime**, the area under the survival curve up
to 100 ([Age and lifetime](age-and-lifetime.md)). It can be measured in full
for every agent born at least 100 iterations before the end of a run:

<!-- figure ages/expectancy -->
![Life expectancy over a world's life](../figures/ages/expectancy.svg)

**Life expectancy over a world's life.** Agents born in each stretch of 25 iterations (those born at iteration 0, the founders and their first children, left out). Left: how many of their first 100 iterations they live on average — the sum of the Kaplan–Meier curve up to L = 100, a mean that every stretch born at least 100 iterations before the end can be measured by in full. Right: the median lifetime, the longest L that more than half of them live. For each world separately; the line is the median of the 30 worlds, the darker band the middle half and the paler band nine in ten.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames after every game (frame `2·t + 1`, `gol_store.read_frame`), read in order by `book_chapters.ages.lifelines`: every agent's `ages` entry gives its birth iteration (the frame's iteration minus its age), and the last frame it is in its last iteration alive.
>
> 1. For each world and stretch of 25 birth iterations with at least 20 births, the Kaplan–Meier curve S(L) of their lives.
> 2. Left: Σ S(L) for L = 1 … 100. Right: the largest L with S(L) > ½.
>
> **To make it again:** `python3 book_figures.py ages`.
<!-- /figure -->

In the median world, an agent born in iterations 1 to 24 lived 5 of its first
100 iterations. One born in 50 to 74 lived 18, and one born in 100 to 124 lived
24. By iteration 250 the expectancy is 27, and from iteration 500 on it hovers
between 28 and 31 for the rest of the run. The median lifetime does the same,
rising from 2 iterations to between 8 and 11. So life expectancy rises through
a world's first five hundred iterations, and then it holds still. Those
iterations are its youth ([The first hundred
iterations](a-worlds-life.md#the-first-hundred-iterations)) and the slowing of
its turnover after it ([Births and deaths slow
down](a-worlds-life.md#births-and-deaths-slow-down)).

### The risk of dying at each age

The **hazard** says, for an agent that has reached an age, how likely it is to
die at that age. Exactly: take the agents alive when the iteration in which
they are *a* iterations old begins. The hazard at age *a* is the share of them
that is gone after that iteration's game ([Age and
lifetime](age-and-lifetime.md)):

<!-- figure ages/hazard -->
![The risk of dying, by age](../figures/ages/hazard.svg)

**The risk of dying, by age.** At each age a: of the agents alive after the game at age a − 1, the share that is gone after the game at age a — the agent's chance of dying at that age, pooled over the worlds, for agents born early in a world's life (orange) and in its settled life (blue). An age is drawn only where at least 200 agents reached it.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames after every game (frame `2·t + 1`, `gol_store.read_frame`), read in order by `book_chapters.ages.lifelines`: every agent's `ages` entry gives its birth iteration (the frame's iteration minus its age), and the last frame it is in its last iteration alive.
>
> 1. For agents born in the period, with d − b + 1 iterations lived: at each L, the lives still going at L (lasting at least L) and those ending at L (lasting exactly L, not still going at the end).
> 2. Divide the second by the first (`book_chapters.ages.hazard`).
>
> **To make it again:** `python3 book_figures.py ages`.
<!-- /figure -->

For an agent born in the settled stretch, the risk is 21% at age 1, 14% at 2,
9% at 3 and 5% at 5. It is 2.7% at 10, 1.8% at 20, 1.3% at 50, 1.1% at 100 and
1.0% at 200. It falls at every age. There is **no ageing**: an old agent is
never more at risk than a young one. Beyond an age of about a hundred, the risk
hardly changes. About one game in a hundred is an old agent's last, whether it
is a hundred iterations old or four hundred.

A constant risk makes the survival curve fall exponentially from there on.
Drawn on a logarithmic axis against a plain axis of *L*, it would be a straight
line. That is what lets the estimate of [the oldest
agent](#why-the-oldest-agent-stops-near-850) work.

Agents born early, in iterations 1 to 249, were at more risk at every age, and
most of all when young. Their risk was 31% at age 1, 4.9% at 10 and 1.3% at
100. The difference between the two periods shrinks with age. An agent of the
early period that had lived a hundred iterations was almost as safe as one born
later.

A risk that falls with age can have two causes, and this figure cannot tell
them apart. Either each agent becomes safer as it lives, gathering tokens and
connections. Or agents differ from birth: the fragile ones die first, and those
left are the robust. Then the risk of each agent stays what it was, but the mix
changes. [Who the old are](#who-the-old-are) looks at what the old have.

### Who the old are

Every 25 iterations, the agents of each age class, with their mean tokens and
mean number of connections:

<!-- figure ages/wealth-age -->
![Who the old are](../figures/ages/wealth-age.svg)

**Who the old are.** Every 25 iterations, the agents alive after the game in each age class: their mean tokens (left) and mean number of connections (right); each line is the median over the 30 worlds of the class's mean.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames after every game (frame `2·t + 1`, `gol_store.read_frame`), read in order by `book_chapters.ages.lifelines`: every agent's `ages` entry gives its birth iteration (the frame's iteration minus its age), and the last frame it is in its last iteration alive.
>
> 1. Every 25 iterations, for each age class, the mean of `tokens` and of the connections each agent has in `edges`.
> 2. Median over the worlds.
>
> **To make it again:** `python3 book_figures.py ages`.
<!-- /figure -->

At iteration 1,475, the last of these moments, a newborn of the median world
holds 5.9 tokens after its first game, and has 3.6 connections
([How births build the network](how-agents-have-children.md#how-agents-have-children)
says what parents give). A few iterations later the young hold fewer tokens: 4.7 at ages 1 to 3, and 4.9 at 4
to 15. They have 3.8 and 3.2 connections. From then on, the older the class,
the richer: 6.0 tokens at 16 to 63, 8.5 at 64 to 255, **15.3** at 256 and over.
That is three times what the young hold, with 5.5 connections. On average, the
old are the rich and the well connected.

But the average is not the typical agent. [Connections and
age](how-agents-have-children.md#connections-and-age) sorts the living by age,
and looks at the median instead of the mean. The median agent has 2 connections
at every age, from a few iterations to over 700. The old classes' higher means
come from the [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") among them: agents joined to far more agents than most. Hubs
die less often than leaves, the agents with a single connection. Within ten
games, 18% of the agents with a single connection die. Of those with ten
connections or more, fewer than one in ten die ([The fate of the
richest](where-do-the-tokens-go.md#the-fate-of-the-richest)). So the hubs make
up more of the old than of the young. And an agent's tokens follow its
connections ([How properties scale
together](how-properties-scale-together.md)), so the old classes are richer
too.

So, as far as connections go, the risk falls with age because the mix changes.
An agent does not gather connections by living long; the well connected are the
ones that live long. Connections give an agent age, not age connections. [Do
the rich stay rich?](where-do-the-tokens-go.md#do-the-rich-stay-rich) follows
single agents' tokens from game to game.

One caution about what "old" means. An agent's age is its node's, not its
brain's. When a neighbour wins an agent's node in a game, the agent keeps its
id, its age and its connections; only its brain is replaced. So an old agent
may carry a brain that arrived in the last game ([Questioning the
mechanics](questioning-the-mechanics.md)).

### The ages of a steady world

Suppose a world's births never changed, and every agent's life followed the
same survival curve *S*. Then the ages of the living would follow from *S*
alone. The agents of age *a* alive after a game are those born *a* iterations
earlier whose lives lasted at least *a* + 1 games. With *B* newborns per
iteration that live through their first game, there are *B* · *S*(*a* + 1) of
them. So the share of the living at age *a* is

$$
\frac{S(a+1)}{\sum_{L \ge 1} S(L)} ,
$$

and the sum in the denominator is the mean lifetime. This is the identity of a
**stationary population** in demography: a population whose births and risks of
death never change. It gives a check. If the settled worlds are in such a
state, their ages should follow from their lifetimes.

Pool every life begun from iteration 500 to 1,399, and follow it up to 999
iterations. The mean lifetime is 41.3 iterations. So the newborns should be 1 /
41.3 = **2.4%** of the living, and the median age of the living should be
**55**. The pyramid measured 2.3 to 2.6%, and the median age of the living over
the settled stretch was 55. The large worlds are in a steady state of ages from
about iteration 500 on. Their births per hundred agents are 3.3 to 3.6 in every
stretch of 250 iterations from then on, and the lives follow one curve.

### Two sizes

The study of the baseline was first written from the thirty worlds of 10,000
tokens of Experiment 2. They hold a tenth as many agents, and they ran for
3,000 iterations. They had a little less of everything. From iteration 500 on,
the median small world had 3.1 births per hundred agents. It lost 0.4 per
hundred to cutting off in reproduction, 0.5 to starvation in the game and 1.7
to cutting off in the game. A large world turns its agents over about a tenth
faster. Their ages, beside these:

<!-- figure ages/two-sizes -->
![Ages in worlds of two sizes](../figures/ages/two-sizes.svg)

**Ages in worlds of two sizes.** The median age and the oldest agent after every game, in the 30 worlds of 100,000 tokens (blue, 1,500 iterations) and in the 30 worlds of 10,000 tokens of the first edition, Experiment 2 (yellow, 3,000 iterations). The line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th); a world that died is no longer counted.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`. The 30 baseline runs `B1-10000-s001` … `B1-10000-s030`: the baseline B1 at 10,000 tokens, seeds 1 to 30, 3,000 iterations each (every setting is listed in [A world's life](a-worlds-life.md)). They are made with `python3 gol_lab.py run E02`.
>
> **Data.** From each run's frames after every game (frame `2·t + 1`, `gol_store.read_frame`), read in order by `book_chapters.ages.lifelines`: every agent's `ages` entry gives its birth iteration (the frame's iteration minus its age), and the last frame it is in its last iteration alive.
>
> 1. As in the first figure, for both sets of worlds.
> 2. In each stretch, take the median, the 25th and 75th and the 5th and 95th percentiles of those means over the worlds that still have a value there (a world that died has none after its death).
>
> **To make it again:** `python3 book_figures.py ages`.
<!-- /figure -->

The small worlds' living are younger at first and older later. At iteration
500 their median age was 43 in the median world; at 1,000, 68; and at 1,499,
70. The large worlds stayed between 51 and 57. Over the whole settled stretch
of 500 to 2,999, the median small world's median age was 74, where the large
worlds' is 55. When the small worlds ended, after twice as many iterations,
the median age of all their living was 72. Of these, 39% were a hundred or
older, against 31% at the end of the large worlds. The small worlds also differ
far more from one another: a tenth as many agents makes every number noisier.

Their lives explain part of it. Take the lives begun in the settled stretch,
from iteration 500 until a hundred iterations before the end. At first they
lasted much the same: 48% lived at least 10 iterations in the small worlds, and
47% in the large. And young agents died equally often: 21% at age 1, 2.7% at
age 10. But old agents died less often in the small worlds: 0.96% per game at
age 100 against 1.14%, and 0.84% at age 200 against 1.02%. So the small worlds'
curve has a longer tail. Of their lives, 15% reached 100 iterations against 13%
in the large worlds, and 0.6% reached 500 against 0.26%. More long lives give a
mean lifetime of 46.7 iterations instead of 41.3, and a steady small world
would have a median age of 65.

The rest is that the small worlds were not steady. Their births per hundred
agents went on falling after iteration 500. They fell from 3.4 in iterations
500 to 749 to between 2.7 and 2.9 from iteration 1,250 on. So their living grew
older than a steady state with those lives would be, while the large worlds'
births held still.

Why old agents die less often in a small world is open. One candidate is the
cuts. An agent is cut off in a game about a fifth more often in a world of
100,000 tokens than in one of 10,000. The rates are 2.1 and 1.7 per hundred.
Being old is no protection when every connection between an agent's region and
the rest of the world carries no tokens in the same game. Then all of them are
cut together, and the region dies ([How a world breaks](how-a-world-breaks.md)
calls it a border going quiet).

## What we learned

- **A settled world replaces about 3.5 of every 100 agents in every
  iteration.** Three in four of those that die are cut off from the network
  rather than starved. The rates hold steady from iteration 500 on, and differ
  little from world to world.
- **Most agents die young, and none grows old.** Nearly a third of all
  newborns die in the iteration they are born in, and two thirds within ten
  iterations. A few live more than a thousand. The risk of dying falls with
  age and then holds near one game in a hundred, so long lives are almost
  memoryless. The oldest agent is set by how many agents there are, not by any
  limit on a life.
- **The living are older than the dead.** They are a sample weighted towards
  long lives, older than the typical agent ever gets. Half of the agents that
  live through their first game are gone within ten iterations, yet half of
  the living are older than 55.
- **In a world of 100,000 tokens, ages settle by iteration 500.** By then a
  world's age structure, life expectancy and risk at every age are those it
  will still have at iteration 1,500. About 2.4% of its agents are newborns,
  and half are younger than 55. An agent that lives through its first game can
  expect some 41 iterations. The first five hundred iterations, the youth and
  the slowing of turnover after it, are the only time the ages change. The
  ages of the living follow from the lifetimes, as those of a stationary
  population do.
- **A world a tenth the size settles later, and grows older.** Its births kept
  falling after iteration 500, and its old agents died less often.
- **The old are the rich, on average, because the well connected live long.**
  The typical agent has 2 connections at every age; the hubs, which die less
  often, fill the old classes. Connections give an agent age, not age
  connections.
- **For [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau."), a world's demography is finished early.** From
  iteration 500 on, nothing about who lives how long changes. Evolution, if it
  acts, has to act inside that steady turnover, not through how long agents
  last. It would act through which brains the 2.4% of newborns carry, and
  which brains take over nodes in the games
  ([Genotypes and lineages](genotypes-and-lineages.md)). And the brains settle
  even earlier than the ages. In ten worlds of 5,000 tokens, the brains of each
  moment were played against those of earlier moments. They improve over a
  world's first 50 to 100 iterations, and not after. From then on, the brains
  of each moment tie with those of the moment before
  ([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)).

## Where to go next

- [Age and lifetime](age-and-lifetime.md): age, lifetime, the hazard and the
  stationary population, made exact.
- [How a world breaks](how-a-world-breaks.md): the cuts that kill three in
  four of the agents that die.
- [How births build the network](how-agents-have-children.md): who has a
  child, what a newborn is given, and how births wire the network.
- [Genotypes and lineages](genotypes-and-lineages.md): how long brains and
  their lines last, rather than agents.

## How this was made

Every figure of this page reads the thirty runs of Experiment 10, listed below.
The figure of [Two sizes](#two-sizes) also reads the thirty runs of 10,000
tokens of Experiment 2, made with `python3 gol_lab.py run E02`. From the runs,
`python3 book_figures.py births` draws the figures of [Four ways in and
out](#four-ways-in-and-out), [How long agents live](#how-long-agents-live) and
[How old the living are](#how-old-the-living-are). And `python3 book_figures.py
ages` draws those of [How ages change over a world's
life](#how-ages-change-over-a-worlds-life). Each figure says, folded beneath
it, which runs it reads and how.

### The runs of Experiment 10

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
> | [`brain_kind`](settings.md#brain_kind) | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 30 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | on | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | on | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in both baselines; on in a few pilots, in Experiment 24 and in half the worlds of the screen. |
> | [`random_decisions`](settings.md#random_decisions) | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | off | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.0 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 0 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | off | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 0 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 0 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](settings.md#keep_home) | 0.0 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 0 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 1 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
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

- **8 and 11 October 2026.** The page was first written, on 8 October, from
  the thirty worlds of 10,000 tokens. It followed "every agent born at
  iteration 500 or later", and found that about half of all agents are gone
  within ten iterations. On 11 October it was rewritten from the thirty worlds
  of 100,000 tokens. It took the 10.4 million agents of its curve to be every
  agent born from iteration 500 on. Half of all agents, it said, are gone
  within ten iterations, and the median lifetime is 8.
  [How ages change over a world's life](#how-ages-change-over-a-worlds-life),
  written beside it, estimated the oldest agent from all 500 births of an
  iteration, at *L* ≈ 930. It also said that the old are the rich and the
  well connected.
- **14 October 2026.** A review of the whole notebook found that the curves
  leave out nearly a third of all newborns. They are the 4.7 million of 15.2
  million that die in the iteration of their birth, never alive after a game.
  Half gone within ten iterations, and a median lifetime of 8, hold only for
  the agents that live through their first game. Of all newborns, two thirds
  are gone within ten iterations, and the median lifetime is 2. The oldest
  agent was estimated again from the 345 newborns per iteration that live
  through their first game: *L* ≈ 940. The same day, the page took in what
  [How the network grows](how-agents-have-children.md#how-the-network-grows) had found: the median
  agent has 2 connections at every age. The old are well connected only on
  average, because hubs die less often than leaves: connections give an agent
  age, not age connections.
- **23 October 2026.** When the two pages were merged, three slips were
  corrected against the data and the analysis code. The hazard at age *a* had
  been described as a share of the agents alive after the game at age *a*. It
  is a share of those alive when the iteration in which they are *a* begins,
  after the game at age *a* − 1. Its numbers were right. The tokens and connections of
  [Who the old are](#who-the-old-are) are from iteration 1,475, the last of the
  moments drawn, not from the last iteration. And agents aged 1 to 3 have 3.8
  connections on average, not three.

<!-- turns -->
---

[Agents' lives: births, deaths, ages](../README.md#agents-lives-births-deaths-ages) · [How births build the network](how-agents-have-children.md) →

> [!info]- Pages that link here (13)
> [Questioning the mechanics](questioning-the-mechanics.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [The life of a B1 world](a-worlds-life.md) · [How births build the network](how-agents-have-children.md) · [Births and the four ways to die](births-and-deaths.md) · [Age and lifetime](age-and-lifetime.md) · [How the game is played](how-the-game-is-played.md) · [How a world breaks](how-a-world-breaks.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [Genotypes and lineages](genotypes-and-lineages.md) · [What the brains are like](what-the-brains-are-like.md) · [The settled life of a world](settled-life.md) · [Survival curves (Kaplan–Meier)](kaplan-meier.md)
<!-- /turns -->
