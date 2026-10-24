# Like next to like: how alike agents are at a distance

> [!question] Questions of this page
> - How is likeness at a distance measured, and what is a correlation length?
> - Are [neighbours](the-world.md#the-network "An agent joined to another by a connection.") alike in their [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), their age, their [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.")?
> - How far does likeness reach — two steps, ten, across the world?
> - How much of it comes from an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")'s place in the network, and how much
>   from closeness itself?
> - Do kin live together, and in territories?

## In short

How alike are two agents a given number of steps apart, and up to how far?
The **correlation function** and the **correlation length** measure it. In the
thirty worlds of 100,000 tokens, neighbours are a little unlike in tokens and
connections, mostly because [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") sit among poor leaves. They are a little
alike in age, as cohorts that arrived together. Every such likeness fades
within three steps, so the correlation length is one to two steps: a world is
far from critical. Agents of one [genotype](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from.") cluster within two or three steps,
and lines of descent in patches 7 to 12 steps wide.

## How alike at a distance

Is a world the same everywhere, or does it have regions? A rich quarter and a
poor one, an old town and new suburbs, the territory of one family and of
another? Physics answers such questions by asking how far the state of one
place reaches. Are two places some distance apart more alike than two places
drawn at random? And up to what distance?

Think of a magnet. It is made of tiny atomic magnets, called spins, each
pointing up or down. A spin pointing up makes its neighbours more likely to
point up, and their neighbours a little more likely, and so on, fading with
distance. How fast it fades is one of the most telling facts about a physical
system. The tool that measures it is the **correlation function** (Chaikin
and Lubensky 1995). On a network, distance is counted in steps along
connections, and the same tool applies.

This is part of the notebook's second aim: to learn whether simple, local
rules grow something like physical space
([Physical inspiration](physical-inspiration.md)). In a space, a difference at
one place shows in the places near it, and fades farther away. The
**correlation length** says how far it reaches.

### The correlation function

Give every agent *u* a value *x*(*u*), such as its tokens, its age or its
number of connections. **Standardise** the values within the world: subtract
their mean, and divide by their standard deviation. A standardised value says
how far an agent stands above or below the typical one. Then take every pair
of agents exactly *r* steps apart, and ask whether their two values go up and
down together. That is the [correlation](correlation.md) over all such pairs,
and it is the correlation function:

$$
C(r) = \operatorname{corr}\bigl(x(u), x(v)\bigr) \quad \text{over pairs with } d(u, v) = r .
$$

Here *d*(*u*, *v*) is the number of steps on the shortest path from *u* to
*v*. The sign of *C*(*r*) says which way the likeness goes:

- *C*(*r*) > 0: agents *r* steps apart tend to be alike;
- *C*(*r*) < 0: unlike, such as a rich agent beside poor ones;
- *C*(*r*) = 0: no relation at that distance.

At one step, *C*(1) takes all connected pairs: it is the assortativity of the
value. For the connections themselves, it is the assortativity of the network
([Assortativity](degree.md#assortativity)).

Pairs *r* steps apart are found by **breadth-first search**. Start from an
agent drawn at random. Its neighbours are one step away. Their neighbours not
yet reached are two steps away, and so on, outward. When pairs from several
moments of a world are pooled, each moment's values are standardised on their
own. Otherwise, pooling would add a false likeness: at a moment when agents
hold more on average, any two would both look rich.

### How fast it fades

In most systems *C*(*r*) falls off **exponentially**: each further step keeps
the same fraction of the likeness left at the step before. In symbols,
*C*(*r*) ∝ e^(−*r*/ξ). The **correlation length** ξ is the distance over which
likeness fades by a factor e, about 2.7. A system has structure on scales up
to ξ, and looks uniform beyond it.

At a **critical point** this changes. A magnet exactly at its Curie
temperature, above which it loses its magnetism, is at one. There ξ grows
without bound, and *C*(*r*) falls only as a power of *r*. Likeness then
reaches across the whole system. That is the sense in which critical systems
have "no scale": there is no distance beyond which they look uniform
([No scale at all](physical-inspiration.md#no-scale-at-all)).

### A trap: place versus closeness

On a network, a correlation at distance *r* can arise without any agent
influencing another. Suppose tokens follow connections:
hubs are rich, and **leaves**, agents with a single connection,
are poor. Suppose too that hubs sit among leaves. Then neighbours have
*unlike* tokens. And two leaves of one hub, two steps apart, have *like*
tokens. Both come purely from where hubs and leaves sit.

To see how much of *C*(*r*) is just **place**, shuffle the values among agents
that have about as many connections, and measure again. The shuffle keeps how
values follow connections, and destroys everything else. What survives it was
there because of place; what disappears was **closeness**.

## Like next to like

Now ask it of the worlds of Graph of Life. Are neighbours more alike than
strangers, in their tokens, their connections, their ages and their
genotypes? And how far does the likeness reach?

The worlds are the thirty of 100,000 tokens of
[Thirty worlds](a-worlds-life.md#thirty-worlds), about 14,000 agents each.
This page was first written from the worlds of 10,000 tokens, the 26 of
thirty that lived to the end, about 1,300 agents each. Where the two sizes
differ, both are given. Every value is read at five moments of each world's
[settled life](settled-life.md "A world from iteration 500 to its end, after its youth, where it is measured.").

### Neighbours

Start with the closest pairs. Take every connection of a world and the two
agents at its ends, and ask whether their values go together. That is
*C*(1): the correlation, over all connections, of the value at one end with
the value at the other. A positive correlation means like beside like; a
negative one means the opposite, such as rich beside poor. Each value is
first taken as a logarithm: of one plus the tokens, of one plus the age, and
of the number of connections. So a few very rich or very connected agents
cannot decide a correlation alone.

<!-- figure alike/neighbours -->
![Are neighbours alike?](../figures/alike/neighbours.svg)

**Are neighbours alike?** For every connection of a world, the two agents at its ends: the correlation of their tokens, of their ages and of their numbers of connections (tokens and age as ln(1 + x), connections as ln k), counting every connection both ways round. One dot per world, the mean over five moments (after the games of iterations 500, 750, 1,000, 1,250 and 1,499); the bar is the median. The same values shuffled among a world's agents give correlations within 0.01 of zero in the median world, and within 0.09 in every world; the last column is the assortativity of [How properties scale together](how-properties-scale-together.md), taken of ln k instead of k.

> [!example]- How to make this figure
> **Runs.** The 30 baseline worlds that lived to the end (`B1-100000-s001` … `-s030`, [Thirty worlds](a-worlds-life.md#thirty-worlds); `python3 gol_lab.py run E10`).
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read frames 2·t + 1 for t = 500, 750, 1,000, 1,250, 1,499: `ids`, `tokens`, `ages`, `brain_ids` and `edges`.
> 2. Pearson's correlation of the value at one end of a connection with the value at the other, over every connection in both directions; the null shuffles the values among agents first (`book_chapters.space.likeness`).
>
> **To make it again:** `python3 book_figures.py alike`.
<!-- /figure -->

- **Tokens: unlike.** Neighbours' tokens are correlated at **−0.16** in the
  [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world (−0.10 at 10,000 tokens). The rich sit among the poor.
- **Age: like.** **+0.14** (+0.15). The old sit among the old, the young among
  the young.
- **Connections: unlike.** **−0.20** (−0.19): hubs sit among agents with few
  connections. This is the assortativity of
  [How properties scale together](how-properties-scale-together.md), taken of
  the logarithm of the connections. Of the connections themselves, which the
  hubs sway more, it is about −0.07 ([Assortativity](degree.md#assortativity)).

All three are small in the median world. Neighbours are alike or unlike only
a little. And the worlds differ: the age correlation ranges from −0.09 to
0.30 between them. The worlds with a giant hub
([How much does the seed decide?](how-much-does-the-seed-decide.md)) reach
−0.57 for tokens and −0.64 for connections. Such a world is a star: one rich,
connected agent among thousands of poor leaves.

### Place, or closeness?

A correlation between neighbours need not mean that neighbours influence each
other. Tokens follow connections
([How properties scale together](how-properties-scale-together.md)), and hubs
sit among leaves. So neighbours' tokens can be unlike without any agent doing
anything to another: the trap of
[place versus closeness](#a-trap-place-versus-closeness). How much of the
likeness is just place?

Shuffle each agent's tokens, and then its age, among the agents with about as
many connections as it has, and measure again.

- For **tokens**, the shuffled worlds give −0.12, most of the −0.16 of the
  real ones. Rich beside poor is mostly place: hubs are rich, leaves are poor,
  and hubs sit among leaves. In the worlds of 10,000 tokens it was all place:
  the shuffle gave −0.13, even more than the real −0.10.
- For **age**, the shuffled worlds give −0.01. Old beside old is not place. It
  is something about the agents themselves.

Why would the old sit beside the old? A likely reason: agents die together,
when a piece of the world is cut off
([How a world breaks](how-a-world-breaks.md)). And they are born into their
parents' neighbourhoods
([How the network grows](how-agents-have-children.md#how-the-network-grows)).
So the young arrive where the old are gone, side by side. Agents that arrived
together live together, as **cohorts**.

### How far likeness reaches

Now measure the same correlations between agents 2, 3, … up to 15 steps
apart: the correlation function *C*(*r*) of the worlds. The pairs come from
breadth-first searches out of agents drawn at random. At one step they are
the neighbours again. But they are a sample, pooled over the five moments, so
their values differ a little from those above.

<!-- figure alike/distance -->
![How alike, how far apart](../figures/alike/distance.svg)

**How alike, how far apart.** For pairs of agents r steps apart, the correlation of their tokens, ages and connections (as in the figure above), for r = 1 to 15. For each world, pairs from breadth-first searches out of 150 agents drawn at random at each of the five moments; the coloured line is the median of the 30 worlds, the darker band the middle half of them and the paler band nine in ten. Grey dashes: the median when each agent's tokens (or age) are first shuffled among the agents with about as many connections (1, 2, 3–4, 5–9, 10–49, 50 and more) — what an agent's place in the network explains without any closeness of its own.

> [!example]- How to make this figure
> **Runs.** The 30 baseline worlds that lived to the end (`B1-100000-s001` … `-s030`, [Thirty worlds](a-worlds-life.md#thirty-worlds); `python3 gol_lab.py run E10`).
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. At each moment, a breadth-first search out of 150 agents drawn at random (generator seeded with 22), to 15 steps; every agent reached at r steps gives the pair (source, agent).
> 2. Standardise each value within its moment (subtract the mean, divide by the standard deviation); per r, pooled over sources and moments, Pearson's correlation of the two values.
> 3. The grey null: the same pairs, with tokens and ages permuted at random within each class of connections (`book_chapters.space.likeness`).
>
> **To make it again:** `python3 book_figures.py alike`.
<!-- /figure -->

Every curve is close to zero from **three steps** on, as in the worlds of
10,000 tokens.

- **Tokens:** −0.13 at one step, **+0.08** at two, and nothing from three on.
  Two agents two steps apart are often two leaves of one hub, poor alike. The
  shuffle explains most of that (+0.06).
- **Age:** +0.13 at one step, +0.08 at two, +0.03 at three, +0.02 at four, and
  nothing from five on. Cohorts span two or three steps.
- **Connections:** −0.17 at one step, +0.12 at two (two leaves of one hub
  again), and close to zero beyond.

In the language of physics, the correlation length of a world is one to two
steps. Within that distance, a world has structure: hubs and their leaves,
cohorts of the same age. Beyond it, every part of a world looks like every
other.

### Kin, near and far

The last question is about families. Do agents of one
genotype live together? A genotype is the identity of a [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions."): a copy
keeps it, and a change gives the brain a new one.

<!-- figure alike/kin -->
![Kin, near and far](../figures/alike/kin.svg)

**Kin, near and far.** For pairs of agents r steps apart (as in the figure above), the share that carry the same genotype, divided by the share among all pairs of agents of the world, on a logarithmic axis: a straight falling line would be a relatedness that halves over a fixed number of steps. The line is the median of the 30 worlds, the darker band the middle half of them and the paler band nine in ten. From 9 steps on, at least half of the worlds have no such pair at all. In the others such pairs are rare: at most an eighth of what chance would give at 9 steps, and less than a fortieth from 11 steps on.

> [!example]- How to make this figure
> **Runs.** The 30 baseline worlds that lived to the end (`B1-100000-s001` … `-s030`, [Thirty worlds](a-worlds-life.md#thirty-worlds); `python3 gol_lab.py run E10`).
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. As above; for every pair, whether the two `brain_ids` are equal.
> 2. Any two agents: Σ c(c − 1) / (n(n − 1)) over the genotypes' counts c among the n agents.
>
> **To make it again:** `python3 book_figures.py alike`.
<!-- /figure -->

Among agents one step apart, **9.2%** carry the same genotype. That is 290
times as many as among any two agents of the world (0.03%).
[Do agents cooperate?](do-agents-cooperate.md) finds the same, counting along
every connection instead of from agents drawn at random. Two steps apart the
share is 4.1% (100 times chance), three steps 0.5% (12 times), and four steps
0.09% (2.6 times). From **five steps on, it is lower than chance would give**:
0.55 times at five steps, 0.1 at six. Beyond eight steps, almost no pair
shares a genotype.

At 10,000 tokens the shares were 9.9% at one step and 5.0% at two. But two
agents drawn at random shared a genotype ten times as often (0.30%). So the
share at one step was 33 times chance, and it fell below chance from four
steps on. The kin clusters are the same size in both. What a larger world
changes is only how rare the same genotype is far away.

So kin live together, but only as small clusters: a parent, its children, and
the nodes it has just conquered, within two or three steps. There are no
territories of genotypes. A genotype is a few agents, not a region.

Two reasons keep it so. A genotype lasts about two
[iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") before a change renames it
([Genotypes and lineages](genotypes-and-lineages.md)), so a cluster of one
genotype has little time to grow. And more than half of all nodes change hands
in every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.")
([How the game is played](how-the-game-is-played.md)). So any territory is
broken up as soon as it forms.

[Lineages](genotype.md#families "A line of descent: the agents whose genotypes go back to one ancestor."), the lines of descent, last much longer than genotypes. A line is all
the agents whose genotypes go back to one ancestor, some iterations earlier.
And lines do hold regions
([Where the lines live](how-does-a-worlds-size-follow-its-tokens.md#where-the-lines-live)).
Give every agent the genotype it descends from 25 iterations back: its
family. Two neighbours share a family 99 times as often as chance would have
them, and the excess reaches about 7 steps. For families of 200 iterations
back it reaches about 12 steps (4 to 17 across the worlds). As a line ages,
its patch widens the way a random walker spreads on these worlds, about as
*t*^0.34 after *t* iterations. A front moving at a steady speed would widen
in proportion to *t*. [One world, many colours](one-world-many-colours.md)
draws such families in colour. So a world has territories after all: of
lines, not of genotypes.

## What we learned

- **Likeness at a distance has a measure.** The correlation function *C*(*r*)
  says how alike agents *r* steps apart are, and the correlation length how
  far likeness reaches. Shuffling values among agents with as many
  connections tells place from closeness.
- **A world has structure only within one or two steps.** Its correlation
  length is one to two connections. Beyond three steps, a world is
  statistically the same everywhere: no rich regions, no old towns, no
  territories of genotypes.
- **Most of the local structure is place, not influence.** Rich beside poor
  is mostly hubs among leaves. Only age is alike beside alike beyond what
  place explains, as cohorts born and dying together.
- **Kin cluster, and lines spread.** Agents of one genotype cluster within two
  or three steps: one step apart, two agents carry the same genotype 290 times
  as often as chance. Lines of descent hold patches of about 7 to 12 steps, which widen
  as a random walker spreads.
- **A world is far from critical.** At a critical point the correlation
  length grows without bound, and likeness falls off with distance only as a
  power. Here it is gone within three steps. That fits
  [Power laws, real and apparent](power-laws-real-and-apparent.md), which
  finds no avalanches of every size and no 1/*f* noise. The space itself may
  have no scale: the room within *r* steps grows as a power of *r*
  ([Is a world a space?](how-many-dimensions-does-a-world-have.md)). But what
  the agents hold is alike only within three steps.
- **For [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau."), nothing yet lets a second kind of world form
  inside the first.** Matter in a vacuum, a cell in its medium, a species in
  its habitat: each is a region that keeps differing from its surroundings.
  A world has room for such regions, for it is a space. But what
  its agents hold differs only within three steps, and the patches of lines
  keep widening. [Space and dimension](../README.md#space-and-dimension) looks
  for rules that would let such regions form.

## Where to go next

- [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md):
  the space the agents live in, with a [dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.") and room for regions.
- [Where the lines live](how-does-a-worlds-size-follow-its-tokens.md#where-the-lines-live):
  lines of descent, whose patches widen as they age.
- [One world, many colours](one-world-many-colours.md): one world drawn by
  tokens, age and family, where cohorts and families can be seen.
- [Physical inspiration](physical-inspiration.md): the physics behind the
  notebook's second aim, and a test of how far a difference travels.

## How this was made

Every figure of this page says, folded beneath it, which runs it reads and
how. All three are drawn again with `python3 book_figures.py alike`. The
measurement is the function `likeness` in `book_chapters/space.py`, and its
numbers are kept in `book/results/alike.json`.

The numbers for the worlds of 10,000 tokens come from the same measurement on
Experiment 2: thirty worlds, `B1-10000-s001` … `-s030`, of 3,000 iterations
each (`python3 gol_lab.py run E02`). The 26 that lived to the end were read at
five moments, from iteration 1,000 to 2,999.

### The runs of Experiment 10

Experiment 10 is 30 runs: [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") at 100,000 tokens, [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 to 30, 1,500
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

- **11 October 2026.** From 9 October, when it was written from the worlds of
  10,000 tokens, this page said that rich beside poor is entirely place.
  There, tokens shuffled among agents with as many connections gave −0.13,
  even more than the real −0.10. The worlds of 100,000 tokens give −0.12 of the real
  −0.16. Rich beside poor is mostly place, but not all
  ([Place, or closeness?](#place-or-closeness)).
- **14 October 2026.** From 9 October, this page said that a world has no
  territories at all, no regions that differ from their surroundings beyond
  three steps.
  [Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens)
  then measured lines of descent. They hold patches, about 7 steps wide for
  families of 25 iterations back and about 12 for 200. So a world has
  territories after all, of lines rather than of genotypes
  ([Kin, near and far](#kin-near-and-far)). Each patch keeps widening as its
  line ages.
- **24 October 2026.** Since 11 October, three numbers of this page had been
  rounded the wrong way. In `book/results/alike.json`, the most negative
  correlation of neighbours' connections, in one world, is −0.6446: −0.64,
  not −0.65. In the median world, the age correlation four steps apart is
  0.0248: +0.02, not +0.03. And agents five steps apart share a genotype
  0.5499 times as often as chance: 0.55, not 0.6.

## References

- Chaikin, P. M. and Lubensky, T. C. (1995). *Principles of Condensed Matter
  Physics*. Cambridge University Press.

<!-- turns -->
---

← [Physical inspiration](physical-inspiration.md) · [Space and dimension](../README.md#space-and-dimension) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) →

**Also about:** [The shape and health of the network](../README.md#the-shape-and-health-of-the-network).

> [!info]- Pages that link here (10)
> [Glossary](glossary.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [The life of a B1 world](a-worlds-life.md) · [An agent's connections: degree, clustering, assortativity, attachment and scaling](degree.md) · [Physical inspiration](physical-inspiration.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [How different are the agents?](how-different-are-the-agents.md) · [Do agents cooperate?](do-agents-cooperate.md) · [One world, many colours](one-world-many-colours.md)
<!-- /turns -->
