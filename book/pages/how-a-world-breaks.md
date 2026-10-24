# How a world breaks

> [!question] Questions of this page
> - How much of a world hangs on single [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.")?
> - Does a world in which a large part hangs on one connection lose more
>   [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")?
> - When a game cuts off a thousand agents, what severed them — and why there?

## In short

Only the largest connected piece of a world's network survives each phase, so
a whole region can die in one game. In a typical settled world one connection
holds a ninth of it to the rest, yet such weak points almost never give way.
Every game that cut off a thousand agents or more did it across a whole
border: ten connections or more, falling quiet at once. The agents on both
sides hold fewer [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") than they have connections, and a connection that
carries no token in a game is cut. So a world breaks along its poor regions,
not at its weakest bridge.

## Whole regions die at once

Picture a world as islands joined by roads. A road stays open only while
traffic crosses it. An island that no open road joins to the mainland is
abandoned, with everyone on it. An island on a single road hangs by that road.
An island on twenty roads is lost only if all twenty fall silent together.
Which of the two is how a world loses its regions?

The rules are just like that. In [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it."), a connection is cut at the end of a game
if no token crossed it in that game, in either direction
([`prune_after`](settings.md#prune_after)). And after every phase the cleanup
keeps only the **largest connected piece** of the network
([One iteration, step by step](one-iteration.md#cleanup)). Every agent outside
it dies, however many tokens it holds, and its tokens are dealt out among the
survivors. This is the **cull**.

Three deaths in four are of this kind
([Births, deaths and ages](births-deaths-and-ages.md)). And they come in
lumps: half of them in games that cut off hundreds of agents at once
([How many agents one game removes](power-laws-real-and-apparent.md#how-many-agents-one-game-removes)).
This page asks how fragile a world is, whether its breaks can be foreseen, and
what actually severs a region from the rest.

The worlds are the thirty B1 worlds of 100,000 tokens of Experiment 10
([Thirty worlds](a-worlds-life.md#thirty-worlds)), read over their [settled
life](settled-life.md "A world from iteration 500 to its end, after its youth, where it is measured."), from iteration 500 on. The page was first written from the thirty worlds
of 10,000 tokens; where those differ, their numbers are given beside. A
**big** cut is counted against the size of a world: a thousand agents or more
here, a hundred in the worlds of 10,000 tokens. In both, that is one agent for
every hundred tokens of the world.

## Bridges and the worst cut

A **bridge** is a connection that lies on no loop: cut it, and the network
falls in two ([Bridges](bridges.md)). A settled world has about 6,400 of them,
26% of its connections
([Core, leaves and bridges](what-shape-does-the-network-take.md#core-leaves-and-bridges)).
Most hold a single leaf, an agent with one connection.

How much hangs on the worst of them? Take away one bridge, and the network
falls into two pieces; the smaller piece is what the bridge holds to the rest.
The **cut risk** is the largest such piece, over all bridges, as a share of
all agents ([Cut risk](bridges.md#cut-risk)). It is 0 when there is no bridge, and at
most a half.

In the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world the cut risk is **11%** (12% at 10,000 tokens). Somewhere,
a single connection is all that joins a ninth of the world to the rest.

## Does a fragile world lose more?

If the worst bridge failed, the game would cut off that ninth. Does it?

<!-- figure breaking/risk -->
![Does a fragile world lose more?](../figures/breaking/risk.svg)

**Does a fragile world lose more?** Every game from iteration 500 on of the 30 worlds. x: before the game, the largest share of the world that one bridge held to the rest — if that one connection were cut, that many would be cut off ([Cut risk](bridges.md#cut-risk)). y: the share of agents the game really cut off. White: the mean of y in classes of x 0.05 wide; dashed: y = x.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 1,499: `cutRiskBefore`, and `orphaned` / `nodes_before`.
> 2. Plot one against the other; average y in classes of x.
>
> **To make it again:** `python3 book_figures.py breaking`.
<!-- /figure -->

Over 30,000 settled games, the cut risk before a game and the share of agents
it cut off have a [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of **0.09**: almost nothing. The mean risk is
11%, and the mean share cut off is 2.0%. In 32% of games more than a tenth of
the world hung on one connection. In only 1.1% of games was more than a tenth
cut off. The worst bridge almost never fails. (At 10,000 tokens: 0.06, 41% and
1.25%.)

The cut risk here is the one a run records in each game's row: read after the
game's unused connections are cut, just before the cleanup. Read at the start
of the game instead, in the worlds of 10,000 tokens, it gives the same
picture: a correlation of 0.067, with more than a tenth of the world on one
connection in 40.5% of games.

The rules say why. A connection is cut only if **no** token crosses it, in
either direction, during the game. A bridge with a hundred agents behind it
is, for the same reason that makes it important, a busy connection. Whatever
flows between the branch and the rest flows across it. Like the only road to
an island, it is almost never empty.

## One world, six hundred iterations

<!-- figure breaking/one-world -->
![600 iterations of one world](../figures/breaking/one-world.svg)

**600 iterations of one world.** The world with seed 26, iterations 600 to 1,200. Top: its number of agents after every game. Middle: how many agents each game cut off. Bottom: before each game, the largest share of the world held to the rest by a single bridge, measured on the network as it stood before the cleanup.

> [!example]- How to make this figure
> **Runs.** `B1-100000-s026`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take `nodes`, `orphaned` and `cutRiskBefore` of the rows with `phase` = 2 and 600 ≤ `iteration` ≤ 1,200.
>
> **To make it again:** `python3 book_figures.py breaking`.
<!-- /figure -->

Follow one world closely: the world of [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 26, from iteration 600 to 1,200.
It shrinks from about 22,800 agents to 19,000 by iteration 734. Then a single
game, that of iteration 735, cuts off **9,460** agents. That is half the
world, and the largest cut of all thirty worlds.

Before that game, 49% of the world hung on one connection. The game cut off a
region of about that size, across 79 connections that all carried nothing in
it. The world grows back to 15,000 within a hundred iterations, and wanders
there, losing a thousand or two in a few games.

The bottom panel shows how the worst cut behaves. It lives in **plateaus**:
the same bridge holds a few percent to a tenth of the world for tens of
iterations. Then the plateau ends abruptly. Either a new connection closed a
loop around the bridge, or the branch was lost. From iteration 1,060 on, a
third to almost half of the world hung on one connection, for 140 iterations.
None of it was lost through it.

## Where the cut-off deaths happen

How large are the games that do the killing?

<!-- figure breaking/deaths-by-size -->
![Where the cut-off deaths happen](../figures/breaking/deaths-by-size.svg)

**Where the cut-off deaths happen.** Of all agents cut off in games from iteration 500 on of the 30 worlds, the share that died in games which cut off at least n agents, for every n. Where the line is at 0.5, half of all such deaths happened in games at least that large.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take `orphaned` of the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 1,499.
> 2. For every n, add up `orphaned` over the rows with `orphaned` ≥ n and divide by the total.
>
> **To make it again:** `python3 book_figures.py breaking`.
<!-- /figure -->

Half of all agents cut off died in games that cut off at least 353. The
largest single loss was 9,460 agents. One game in fifty cuts off a thousand
agents or more, and these games account for a sixth of all deaths by cutting.

Counted against the size of a world, it is much as at 10,000 tokens. There,
half died in games that cut off at least 42. One game in forty cut off a
hundred or more, and these games held a fifth of the deaths.

## What severs a region

Which rule cuts a region off? There is a clean way to tell. Take a game, the
agents it cut off, and the survivors. Suppose a cut-off agent had a [neighbour](the-world.md#the-network "An agent joined to another by a connection.")
among the survivors before the game. Then it must have **lost that connection
to pruning**, the cut of every connection that carried nothing. Otherwise it
would still be joined to that neighbour, and would have survived with it.

So the connections between the lost and the kept, as they stood before the
game, are exactly the connections pruned across the cut. They are the region's
**border**, and counting them tells what kind of break it was. A bridge that
fails leaves a border of one connection. A border of ten means that ten
connections went quiet in the same game.

Of the 601 games that cut off a thousand agents or more (30 worlds, settled
life), **every one** was a whole border going quiet. Each had ten or more
connections between the region and the rest of the world, all carrying nothing
in the same game. Together these games cut off 1.4 million agents. Not one big
cut was a bridge failing.

At 10,000 tokens it was nine in ten. Of the 1,614 games that cut off a hundred
or more, 1,421 went through ten connections or more. Another 161 went through
three to nine, and 32 through one or two.

There, once, a single line of [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") cut itself off. In the world of seed 1 of
10,000 tokens, a game at iteration 541 cut off 319 agents. Of them, 60%
carried one [genotype](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from."), and the typical one staked all its tokens on its own
node. An agent that stakes everything at home sends nothing across its
connections; they stay alive only if its neighbours stake on it.

In the large worlds no such cut shows. Take the six largest cuts of the world
of seed 1 of 100,000 tokens, from 2,177 to 4,608 agents, lost across 202 to
843 connections. In none of them did one genotype hold more than 4% of the
lost.

## Poverty disconnects

Why would ten connections go quiet at once? Look at the agents on the two
sides of the quiet borders, in the games with ten or more border connections:

| | median tokens | median connections | hold fewer tokens than connections |
|---|---|---|---|
| lost side of the border | 1 | 2 | 52% |
| kept side of the border | 3 | 4 | 72% |
| every agent, same games | 3 | 2 | 13% |

The agents on both sides of a quiet border are **poor for their connections**.
Half of those on the lost side, and nearly three in four on the kept side,
hold fewer tokens than they have connections. Among all the agents of the same
games, only 13% do.

Here is the arithmetic that makes that fatal. A connection survives a game
only if at least one token crosses it. An agent with τ tokens can send a token
across at most τ of its connections, and fewer if it keeps any at home. An
agent with one token and two connections can keep at most one of them alive by
itself. The other depends on the neighbour's stake.

So every connection costs at least one token of flow per game, paid by one
side or the other. A poor region cannot pay for all its connections. When the
richer agents beyond the border spend their stakes elsewhere, the border goes
quiet. Then the region dies, all of it, and its tokens are handed out by
lottery to the survivors ([The lottery](gains-and-losses.md#the-lottery)).

The same price shows connection by connection
([Which connections carry nothing](where-the-tokens-flow.md#which-connections-carry-nothing)).
A leaf's only connection is empty in 1.3% of games. A connection between two
agents with ten connections or more is empty in 13% to 14%.

## Worlds that cannot break

A world that loses regions in every iteration is poor ground for evolution
that should go on. In B1's worlds of 5,000 tokens, the cull removes about 2%
of the agents in every iteration. That is four times as many as starve
([Brains that see, and worlds that break](a-world-that-cannot-break.md#brains-that-see-and-worlds-that-break)).
No such world passes the [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4.")
([The network that holds, in numbers](how-to-tell-a-world-that-keeps-evolving.md#the-network-that-holds-in-numbers)).
One of its marks is a cut risk of at most 1%.

The second [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2."), [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death."), answers with three local rules
([A world that cannot break](a-world-that-cannot-break.md#a-world-that-cannot-break)).
A newborn is joined to two of its parent's neighbours. A death is mended:
where the dead agent's neighbours would be left apart, they are joined up in a
ring. And only redundant links lapse: an unused link is cut only where its two
ends share at least two neighbours. Under these rules neither a death nor a
lapsed link can split a world. In the ten B2 worlds of Experiment 14, not one
agent was cut off from iteration 200 to 1,000
([Nobody cut off](a-world-that-cannot-break.md#nobody-cut-off)). Letting a
separated piece live on as a world of its own is planned for the experiment
*Local rules only*.

## What we learned

- **A world breaks along its poor regions, not at its weakest bridge.** The
  most fragile-looking connection almost never fails. The big losses are
  borders of agents too poor to keep their connections in use.
- **Connections have a price.** One token of flow per game, per connection,
  paid by one side or the other. That caps how many connections a world of *T*
  tokens can hold. It may be part of why a world's size follows its tokens
  ([Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md)).
- **The cull is global, and blind.** A region that is cut off dies whatever
  its brains did, even if it was doing well inside. Selection here acts on
  regions through their borders, not on agents through what they do. And a
  world can never split into two populations that go their own ways, which in
  nature is how new species begin.
  [One rule reaches across the world](questioning-the-mechanics.md#one-rule-reaches-across-the-world)
  questions this rule, and
  [Fourth: open the world](what-the-baseline-world-is.md#fourth-open-the-world)
  what changing it might do. B2 took the other road: rules under which a world
  cannot split at all.

## Where to go next

- [The shape of the network](what-shape-does-the-network-take.md): the core,
  the bottlenecks, and why a world mixes slowly.
- [Where tokens flow](where-the-tokens-flow.md#what-crosses-a-connection):
  what crosses each connection in a game, and which connections carry nothing.
- [Building B2: a world that cannot break](a-world-that-cannot-break.md): the
  rules under which nobody is cut off.
- [Questioning the mechanics](questioning-the-mechanics.md#one-rule-reaches-across-the-world):
  the cull, and what could replace it.

## How this was made

Every figure of this page reads the thirty runs of Experiment 10, listed
below. They are worlds of the baseline B1 at 100,000 tokens, seeds 1 to 30,
run for 1,500 iterations each. The command `python3 gol_lab.py run E10` makes
them. The figure of one world reads the world of seed 26 alone. The numbers at
10,000 tokens come from the thirty worlds of Experiment 2, of which 26 lived
to the end of their 3,000 iterations (`python3 gol_lab.py run E02`). Those of
[Worlds that cannot break](#worlds-that-cannot-break), and of a connection's
price connection by connection, are quoted from the pages linked there.

The command `python3 book_figures.py breaking` draws every figure of this
page. It also writes the numbers of the big cuts to
`book/results/breaking.json`. For every game that cut off a thousand agents or
more, it compares the world before the game with the world after it. Then it
counts the border. Each figure says, folded beneath it, which runs it reads
and how.

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

- **11 October 2026.** This page was first written, on 8 October, from the
  thirty worlds of 10,000 tokens. On this day it was rewritten from the thirty
  of 100,000 tokens, with the small worlds' numbers beside the new ones. Two
  claims changed with the size of a world:
  - The page had said that one big cut in fifty is a bridge failing. In the
    small worlds, 32 of 1,614 went through one connection or two. So did two
    of the six largest cuts of the world of seed 1. In the large worlds not
    one did: every big cut was a border of ten connections or more.
  - It had said that a line of brains that keeps everything at home can cut
    its whole region off, and die with it. One did, in the world of seed 1. In
    the large worlds no such cut shows.

<!-- turns -->
---

← [How properties scale together](how-properties-scale-together.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [An agent's connections: degree, clustering, assortativity, attachment and scaling](degree.md) →

> [!info]- Pages that link here (15)
> [Questioning the mechanics](questioning-the-mechanics.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [The life of a B1 world](a-worlds-life.md) · [Births, deaths and ages](births-deaths-and-ages.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [Where tokens flow](where-the-tokens-flow.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Power laws, real and apparent](power-laws-real-and-apparent.md) · [Distances in a network: path length, radius and diameter, the spectral gap](path-length.md) · [Bridges, cut risk and loops: how a network can be cut](bridges.md) · [Physical inspiration](physical-inspiration.md) · [Like next to like](like-next-to-like.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [Do agents cooperate?](do-agents-cooperate.md) · [One world, many colours](one-world-many-colours.md)
<!-- /turns -->
