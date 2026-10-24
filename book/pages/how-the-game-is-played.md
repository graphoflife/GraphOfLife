# How the game is played: what agents stake, and who wins the nodes

> [!question] Questions of this page
> - How much of their tokens do agents stake on their own node?
> - Do they spread their stakes or put everything on one candidate?
> - Who wins the nodes — the agent living there, or a neighbour; the largest
>   staker, or a coalition?
> - Do the answers change between the [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")' first game and a settled
>   world, and do the worlds agree?

## In short

In every game, each agent stakes all its tokens on its own node and its
neighbours' nodes. A node goes to its largest staker or to a coalition, and
the winner's [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") is copied into it. The founders' random brains play in
every way there is. Within a few hundred iterations, every world but the
[stars](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") plays the same game. There, 98% of the agents spread their stakes, and
97% of the staked tokens are marked revolutionary. About 30% of the staked
tokens stay at home.
An agent then keeps its node in 44% of games, and coalitions decide about
half of all nodes. So brains spread mostly by conquest, not by children.

## The game, and what the rules leave open

The game is where tokens and brains change hands. Every agent stakes all its
tokens on itself and its neighbours. Every node goes to the largest staker or
to a coalition, and the winner's brain is copied into the node
([One iteration, step by step](one-iteration.md)). The rules leave
everything else to the brains. This page looks at what the brains of the
[baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") actually do.

The worlds are the thirty of 100,000 tokens of
[Thirty worlds](a-worlds-life.md#thirty-worlds). This page was first written
from the thirty worlds of 10,000 tokens; where they differ, both are given.

Each agent takes three decisions in every game
([What a brain says](brain-outputs.md)):

1. **Spread or all in.** Either the agent splits its tokens over its
   candidates in proportion to its stake scores. Or it puts all of them on
   the candidate with the highest score.
2. **How much at home.** Its own node is one of its candidates. So the share
   it stakes there follows from the scores: anything from nothing to
   everything ([Staking at home, and keeping one's node](#staking-at-home-and-keeping-ones-node)).
3. **How much is revolutionary.** Of what it stakes on each candidate, the
   agent marks a share as revolutionary. That share counts for a coalition
   against the largest staker
   ([How a coalition takes a node](revolution.md)).

## Staking at home, and keeping one's node

Two questions run through this page. How much do agents stake **at home**,
on their own node? And how often does an agent **keep** its node, by winning
it itself? Every game row of a run records the statistics that answer them
([What a run records](frames-and-stats.md)).

**How much is staked at home.** `selfAllocationShare` is, of all the tokens
staked in the game, the share that agents staked on their own node:

$$
\text{selfAllocationShare} = \frac{\sum_u a_{u \to u}}{\sum_u \tau(u)} .
$$

Here *a*ᵤ→ᵤ is what agent *u* staked on its own node, and τ(*u*) everything
it staked, which is all its tokens. The sums run over all agents. It is a
share of **tokens**, so a rich agent counts more than a poor one.

The notebook also looks at single agents. For an agent *u*, the share of its
own stake that it put at home is *a*ᵤ→ᵤ / τ(*u*). This is read from the
decisions kept in the frames.

**Example.** Agent *A* holds 10 tokens and stakes 3 of them at home. Agent
*B* holds 2 and stakes both at home. Their own shares are 0.3 and 1, a mean
of 0.65. But of the 12 tokens staked, 5 went home: `selfAllocationShare` is
5/12 ≈ 0.42. The rich agent weighs more, so the token share leans toward
what the rich do.

**How often an agent keeps its node.** `heldHomeShare` is, of all nodes on
which anyone staked anything, the share won by **their own agent**. A node
that is not kept was won by a neighbour, alone or with a coalition. Its agent
now carries a copy of that neighbour's brain.

**How often a coalition wins.** `revolutions` is the number of nodes won by a
coalition ([How a coalition takes a node](revolution.md)). The notebook
divides it by the number of agents at the start of the game:
`revolutions/nodes_before`.

Together these say what kind of contest the game is. Suppose agents staked
everything at home and always kept their nodes. Then the game would change
nothing, and brains would be passed on only to children. In fact, in a
settled baseline world, agents stake about 30% of their tokens at home and
keep their node in about 45% of games. About half of all nodes are won by a
coalition rather than by their largest staker. The rest of this page shows
how the worlds get there.

## How much is staked at home

<!-- figure game/home-stake -->
![How much of its stake an agent puts on its own node](../figures/game/home-stake.svg)

**How much of its stake an agent puts on its own node.** For every agent that staked in a game: the share of its tokens it put on its own node. '0' and '1' are exact: nothing at home, everything at home. Grey: the game of iteration 0, played by the founders and their first children, whose brains no selection has touched yet. Blue: the game of iteration 1,499.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. In the frames of iteration 0 and 1,499 with `phase` 2, read `decisions.allocations`: for each agent, `alloc[0]` is what it staked on its own node (its first target is itself) and `tokens` all it staked.
> 2. Put `alloc[0] / tokens` into the bins; divide each count by the number of agents.
>
> **To make it again:** `python3 book_figures.py game`.
<!-- /figure -->

The founders and their first children have brains that nobody selected, and
they are extreme. In the first game, 43% of them put **nothing** on their own
node, and 7% put **everything** there. By the last game both extremes have
almost gone: 5.5% stake nothing at home, and 4.2% everything. Most agents put
between a tenth and two thirds of their tokens on their own node. The mean
share rose from 0.14 to 0.38.

The first game differs from that of the worlds of 10,000 tokens. There, 38%
staked nothing at home, 12% everything, and the mean share was 0.25. The
reason is the starting ring
([The world: tokens, agents and a network](the-world.md)). Each of the 1,000
founders here starts with *k* = max(⌊1,000 / 100⌋, 5) = 10 neighbours. The
100 founders of a small world had 4: there *k* = 5, and an odd *k* is wired
as *k* − 1. A brain that splits its tokens between more candidates puts a
smaller share on any one of them, its own node included. By the last game the
two sizes agree (0.39 at 10,000 tokens).

Why do the bars of the last game have peaks? Agents hold few tokens: half of
all agents hold 5 or fewer
([How unequal is a world, and who stays rich](where-do-the-tokens-go.md)).
And they stake whole tokens. An agent with 3 tokens can put only 0, ⅓, ⅔ or
all of them at home; one with 2, only 0, ½ or all. So the shares cluster at
simple fractions.

## How agents stake

<!-- figure game/doctrine -->
![How agents stake, over a world's life](../figures/game/doctrine.svg)

**How agents stake, over a world's life.** After every game of the 30 baseline worlds. Blue: the share of the agents that staked who spread their tokens over several candidates rather than putting all on one (`spreadShare`). Yellow: of all tokens staked, the share staked by agents on their own node (`selfAllocationShare`). Red: of all tokens staked, the share marked revolutionary (`revoltShare`). For each, the line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th) ([Bands](bands.md)).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take the rows with `phase` = 2 and the statistics `spreadShare`, `selfAllocationShare`, `revoltShare` (`a/b` is the row's `a` divided by its `b`; [Staking at home](how-the-game-is-played.md#staking-at-home-and-keeping-ones-node) defines each).
> 2. Cut the 1,500 iterations into 600 stretches of 2.5 and take each world's mean in each stretch.
> 3. In each stretch, take the median, the 25th and 75th and the 5th and 95th percentiles of those means over the worlds that still have a value there (a world that died has none after its death).
>
> **To make it again:** `python3 book_figures.py game`.
<!-- /figure -->

Three shares change over the life of the worlds:

- **Spreading becomes the rule.** In the first game, 50% of the agents that
  staked spread their tokens over several candidates, in the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world.
  That is about what random brains give, since the choice turns on which of
  two random outputs is larger. From iteration 500 on, it is 98%.
- **Nearly every token is revolutionary.** The share of staked tokens marked
  revolutionary rose from 49% in the first game to 97%. So nearly every stake
  on a neighbour's node counts for a coalition against that node's largest
  staker.
- **Tokens at home**, counted as a share of all staked *tokens*, rose from 13%
  in the first game to 29%. That is lower than the average agent's share
  (0.38 in the last game, above), because here each agent counts by its
  tokens. The richer an agent, the smaller the share it keeps at home.

At 10,000 tokens the same three shares went from 49%, 50% and 25% to 98%,
97% and 30%: the same game, from a different start.

## Who wins the nodes

A node on which anyone staked has four possible outcomes. It is won by the
agent living on it or by a neighbour. And the winner is either the largest
single staker, or a member of a coalition that outweighed the largest staker.

<!-- figure game/winners -->
![Who wins a node](../figures/game/winners.svg)

**Who wins a node.** Every node on which anyone staked in the game, by who won it: the agent living on it (green and cyan) or one of its neighbours (orange and violet), and whether the winner was the largest single staker (green, orange) or the member of a coalition that outweighed the largest staker (cyan, violet; see [How a coalition takes a node](revolution.md)). Each column adds up to 1. Left: the game of iteration 0; right: the game of iteration 1,499.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read frame 1 of each of the 30 runs (the game of iteration 0) and frame 2,999 of the 30 that reached it (the game of iteration 1,499).
> 2. For every entry of `decisions.winners`, note whether `winner` is the `node` itself, and whether `revolt` is true (the node was won by a coalition).
> 3. Count the four kinds over all runs of a column, divide by the number of entries, and stack them.
>
> **To make it again:** `python3 book_figures.py game`.
<!-- /figure -->

In the **first game**, 82% of the nodes went to a neighbour. Most of them
(60% of all nodes) went to a neighbour that simply staked the most there, and
23% to a coalition. The founders' random brains, each with ten neighbours to
stake on, left their own nodes open. At 10,000 tokens, with four neighbours
each, 72% went to a neighbour.

In the **last game**, the agent living on a node won it 45% of the time. It
won 33% of the nodes as the largest staker, and 12% with a coalition.
Neighbours won 55%, most of
them (40% of all nodes) as members of a coalition. **Coalitions decided half
of all nodes** (12% + 40%), as at 10,000 tokens.

<!-- figure game/kept-taken -->
![Who wins the nodes, over a world's life](../figures/game/kept-taken.svg)

**Who wins the nodes, over a world's life.** After every game of the 30 baseline worlds. Green: of all nodes on which anyone staked, the share won by the agent living on it (`heldHomeShare`). Violet: the number of nodes won by a coalition, divided by the number of agents at the start of the game (`revolutions/nodes_before`). For each, the line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th) ([Bands](bands.md)).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take the rows with `phase` = 2 and the statistics `heldHomeShare`, `revolutions/nodes_before` (`a/b` is the row's `a` divided by its `b`; [Staking at home](how-the-game-is-played.md#staking-at-home-and-keeping-ones-node) defines each).
> 2. Cut the 1,500 iterations into 600 stretches of 2.5 and take each world's mean in each stretch.
> 3. In each stretch, take the median, the 25th and 75th and the 5th and 95th percentiles of those means over the worlds that still have a value there (a world that died has none after its death).
>
> **To make it again:** `python3 book_figures.py game`.
<!-- /figure -->

The change happened in the first few hundred iterations, and then held. From
iteration 500 on, the median world's agents kept their node in 44% of games.
The number of nodes won by a coalition was 51% of the number of agents (45%
and 52% at 10,000 tokens).

That 45% is an average over very different places.
[Where tokens flow](where-the-tokens-flow.md) sorts the nodes by their
connections. An agent with one connection keeps its node in 74% of games, an
agent with fifty or more in only 2%. On a node with *d* connections, the
agent living there is one of *d* + 1 stakers. Agents spread their stakes
almost evenly over everyone they can reach. So on a hub, the agent living
there wins about as often as any one of its stakers would. **Hubs are places,
not individuals**: the brain on a hub is replaced in nearly every game.

## Do the worlds agree?

<!-- figure game/settled -->
![How the game is played in each settled world](../figures/game/settled.svg)

**How the game is played in each settled world.** One dot per world: its mean over iterations 500 to 1,499 of each statistic of the two figures above; the bar is the median of the 30. From left: the share of stakers who spread their stake, the share of staked tokens marked revolutionary, the share of nodes kept by their own agent, nodes won by a coalition per agent, and the share of staked tokens put at home. The dots are spread sideways only so that they do not hide each other.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. For each run, average each statistic over the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 1,499.
> 2. Draw one dot per run and statistic, and a bar at the median.
>
> **To make it again:** `python3 book_figures.py game`.
<!-- /figure -->

Remarkably well, with one kind of exception. Over the [settled life](settled-life.md "A world from iteration 500 to its end, after its youth, where it is measured."), 23 of the
30 worlds have between 96% and 99% of their stakers spreading. Their agents
keep their node in 37% to 48% of games. Coalitions win nodes numbering 47% to
61% of the agents, and 26% to 31% of the staked tokens go home.

In the other seven, one agent became a hub joined to a tenth of the world or
more
([How much does the seed decide?](how-much-does-the-seed-decide.md)). They
lie apart, in the order of their hubs. Take the world with [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 28, whose hub
is joined to 58% of all agents. There, agents keep their node in 67% of
games, and coalitions win nodes numbering only 16% of the agents. And 45% of
the staked tokens go home, while 90% of stakers spread. A leaf of a star has
two candidates, its own node and the hub's. With a few tokens, it outstakes
the single token the hub puts on its node
([The largest and the smallest](the-largest-and-the-smallest.md)).

At 10,000 tokens, where hubs were smaller, all 26 surviving worlds lay
together. They had 95% to 99% spreading, 39% to 52% of nodes kept, and 41% to
56% won by coalitions per agent. Apart from its stars, every world ends up
playing the game in much the same way, whatever its founders started with.

## What we learned

- **Brains spread mostly by conquest.** In a settled world of about 14,400
  agents, about 56% of nodes, some 8,000, go to a neighbour in every game.
  Each takes on the winner's brain. Against that, about 500 children are
  born per iteration
  ([Births, deaths and ages](births-deaths-and-ages.md)). A brain's
  descendants are mostly the nodes it conquers, not its children.
- **The game is a game of coalitions.** 97% of staked tokens are marked
  revolutionary. So a node goes, nearly always, to one of its other stakers
  when together they put more on it than its largest single staker did. That
  winner may be the agent living there.
- **The stakes are nearly an even split.**
  [Where tokens flow](where-the-tokens-flow.md) shows that agents stake on
  almost all of their candidates. They keep a little more than an even share
  at home, which is the rules' rounding of whole tokens, not the brains. A
  game moves tokens almost as a random walk would.
- **The behaviour converges, by selection, and early.** The founders' random
  brains play every way there is; every world but the stars ends up at
  nearly the same shares. Selection set them:
  [What the brains are like](what-the-brains-are-like.md) finds the brains'
  constants (few children, the stake spread, every token revolutionary)
  made alike in every world.
  [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md)
  plays the brains of each [era](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?") against their founders' in ten worlds of
  5,000 tokens. By iteration 50 they hold nine tenths of a contested world,
  and no later era does better than the one before it.
  [Do the brains matter?](do-the-brains-matter.md) compares worlds whose
  brains never change.

## Where to go next

- [Where tokens flow](where-the-tokens-flow.md): where the staked tokens go,
  connection by connection, and why the game is nearly a random walk.
- [How a coalition takes a node](revolution.md): the rule behind half of all
  nodes.
- [The even split](even-split.md): what the rules do with a stake no brain
  decided.
- [What the brains are like](what-the-brains-are-like.md): the brains that
  learned to play this way.

## How this was made

Every figure of this page says, folded beneath it, which runs it reads and
how. To make every figure of this page again: `python3 book_figures.py game`.

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

<!-- turns -->
---

← [Where tokens go](where-tokens-go.md) · [The game and the flow of tokens](../README.md#the-game-and-the-flow-of-tokens) · [The even split](even-split.md) →

**Also about:** [Wealth: who holds the tokens](../README.md#wealth-who-holds-the-tokens).

> [!info]- Pages that link here (23)
> [Glossary](glossary.md) · [The world: tokens, agents and a network](the-world.md) · [One iteration, step by step](one-iteration.md) · [The brain](the-brain.md) · [What a brain says](brain-outputs.md) · [How a coalition takes a node](revolution.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [The life of a B1 world](a-worlds-life.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [Gains and losses: who gains, who loses, and the share-out](gains-and-losses.md) · [The largest and the smallest](the-largest-and-the-smallest.md) · [Flow modules: the groups tokens stay in](flow-modules.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Stars and hubs](stars-and-hubs.md) · [Like next to like: how alike agents are at a distance](like-next-to-like.md) · [How alike the agents are](how-alike-the-agents-are.md) · [Genotypes and lineages](genotypes-and-lineages.md) · [What the brains are like](what-the-brains-are-like.md) · [How different are the agents?](how-different-are-the-agents.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [Do agents cooperate?](do-agents-cooperate.md) · [What a run records](frames-and-stats.md) · [Muller plots](muller-plot.md)
<!-- /turns -->
