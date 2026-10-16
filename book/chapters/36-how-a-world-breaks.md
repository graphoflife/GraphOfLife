# How a world breaks

At the end of every phase the cleanup keeps only the **largest connected
piece** of the network; every agent outside it dies, however many tokens it
holds, and its tokens are dealt out among the survivors
([Chapter 3](03-one-iteration.md)). Three deaths in four are of this kind
([Chapter 12](12-births-deaths-and-ages.md)), and [Chapter 32](32-power-laws-real-and-apparent.md)
found that they come in lumps: half of them in games that cut off hundreds of
agents at once. This chapter asks how fragile a world is, whether its breaks
can be foreseen, and what actually severs a region from the rest.

> [!question] Questions of this chapter
> - How much of a world hangs on single connections?
> - Does a world in which a large part hangs on one connection lose more
>   agents?
> - When a game cuts off a thousand agents, what severed them — and why there?

The worlds are the thirty of 100,000 tokens of [Chapter 9](09-thirty-worlds.md);
where the thirty of 10,000 tokens this chapter was first written from differ,
both are given. A "big" cut is counted against the size of a world: a
thousand agents or more here, a hundred in the worlds of 10,000 tokens — a
hundredth of the world's tokens' worth.

<!-- runs E10 -->
> [!info] The runs behind this chapter
> **30 runs.** Every condition below is run once for every seed (1 to 30), for 1,500 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it.
>
> - **baseline** — changes nothing: B1 as it is; 100,000 tokens. Runs `B1-100000-s001` … `B1-100000-s030`.
>
> To make these runs again: `python3 gol_lab.py run E10`, or ▶ in the Book tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | baseline | what it is |
> |---|---|---|
> | [`total_tokens`](../notes/settings.md#total_tokens) | 100,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](../notes/settings.md#n_nodes) | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](../notes/settings.md#k_neighbors) | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](../notes/settings.md#rewire_p) | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](../notes/settings.md#hidden_layers) | 50, 45, 40, 35, 30 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](../notes/settings.md#brain_kind) | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](../notes/settings.md#brain_bits) | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](../notes/settings.md#message_amount) | 30 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](../notes/settings.md#random_input_amount) | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](../notes/settings.md#exchange_messages) | on | Whether agents send and read messages at all. |
> | [`message_prepass`](../notes/settings.md#message_prepass) | on | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](../notes/settings.md#allow_handover) | on | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](../notes/settings.md#allow_revolutions) | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](../notes/settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](../notes/settings.md#random_decisions) | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](../notes/settings.md#prune_after) | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](../notes/settings.md#inactive_window) | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](../notes/settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births). |
> | [`tokens_created_per_phase`](../notes/settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](../notes/settings.md#mutate_on_replication) | off | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](../notes/settings.md#incumbency) | 1.0 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](../notes/settings.md#birth_links) | 0 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](../notes/settings.md#mend_on_death) | off | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](../notes/settings.md#prune_needs) | 0 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](../notes/settings.md#valence) | 0 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](../notes/settings.md#keep_home) | 0.0 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](../notes/settings.md#max_age) | 0 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`min_child`](../notes/settings.md#min_child) | 1 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`mix_at_birth`](../notes/settings.md#mix_at_birth) | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mutation_probability`](../notes/settings.md#mutation_probability) | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](../notes/settings.md#mutation_noise_std) | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](../notes/settings.md#mutation_sparsity) | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](../notes/settings.md#extinction_threshold) | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 30 | one run per seed |
> | iterations | 1,500 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

## Bridges and the worst cut

A **bridge** is a connection on no loop: cut it, and the network falls in two
([Bridges](../notes/bridges.md)). A settled world has about 6,400 of them — 26%
of its connections ([Chapter 17](17-what-shape-does-the-network-take.md)). Most
hold a single leaf. The **cut risk** asks about the worst one: the largest
share of the world that one bridge holds to the rest ([Cut risk](../notes/cut-risk.md)).
In the median world it is **11%** (12% at 10,000 tokens): somewhere, a single
connection is all that joins a ninth of the world to the rest.

## Does a fragile world lose more?

If the worst bridge failed, the game would cut off that ninth. Does it?

<!-- figure breaking/risk -->
![Does a fragile world lose more?](../figures/breaking/risk.svg)

**Does a fragile world lose more?** Every game from iteration 500 on of the 30 worlds. x: before the game, the largest share of the world that one bridge held to the rest — if that one connection were cut, that many would be cut off ([Cut risk](../notes/cut-risk.md)). y: the share of agents the game really cut off. White: the mean of y in classes of x 0.05 wide; dashed: y = x.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Chapter 9](09-thirty-worlds.md)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](../notes/frames-and-stats.md)).
>
> 1. Take the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 1,499: `cutRiskBefore`, and `orphaned` / `nodes_before`.
> 2. Plot one against the other; average y in classes of x.
>
> **To make it again:** `python3 book_figures.py breaking`.
<!-- /figure -->

Over 30,000 settled games, the cut risk before a game and the share of agents
the game cut off have a correlation of **0.09**: almost nothing. The mean risk
is 11%; the mean share cut off is 2.0%. In 32% of games more than a tenth of
the world hung on one connection; in only 1.1% of games was more than a tenth
cut off. The worst bridge almost never fails. (At 10,000 tokens: 0.06, 41%
and 1.25%.)

The rules say why. A connection is cut only if **no** token crosses it, in
either direction, during the game ([`prune_after`](../notes/settings.md#prune_after)).
A bridge with a hundred agents behind it is, for the same reason that makes it
important, a busy connection: whatever flows between the branch and the rest
flows across it.

## One world, six hundred iterations

<!-- figure breaking/one-world -->
![600 iterations of one world](../figures/breaking/one-world.svg)

**600 iterations of one world.** The world with seed 26, iterations 600 to 1,200. Top: its number of agents after every game. Middle: how many agents each game cut off. Bottom: before each game, the largest share of the world held to the rest by a single bridge, measured on the network as it stood before the cleanup.

> [!example]- How to make this figure
> **Runs.** `B1-100000-s026`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](../notes/frames-and-stats.md)).
>
> 1. Take `nodes`, `orphaned` and `cutRiskBefore` of the rows with `phase` = 2 and 600 ≤ `iteration` ≤ 1,200.
>
> **To make it again:** `python3 book_figures.py breaking`.
<!-- /figure -->

The world of seed 26 between iterations 600 and 1,200. It shrinks from about
22,800 agents to 19,000 by iteration 734. Then a single game, that of
iteration 735, cuts off **9,460** agents — half the world, the largest cut of
all thirty worlds. Before that game 49% of the world hung on one connection;
the game cut off a region of about that size, across 79 connections that all
carried nothing in it. The world grows back to 15,000 within a hundred
iterations and wanders there, losing a thousand or two in a few games. The
worst single cut (bottom) lives in **plateaus**: the same bridge holds a few
percent to a tenth of the world for tens of iterations, until the plateau
ends abruptly — because a new connection closed a loop around the bridge, or
because the branch was lost. From iteration 1,060 on, a third to almost half
of the world hung on one connection, for 140 iterations, and none of it was
lost through it.

## Where the cut-off deaths happen

<!-- figure breaking/deaths-by-size -->
![Where the cut-off deaths happen](../figures/breaking/deaths-by-size.svg)

**Where the cut-off deaths happen.** Of all agents cut off in games from iteration 500 on of the 30 worlds, the share that died in games which cut off at least n agents, for every n. Where the line is at 0.5, half of all such deaths happened in games at least that large.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Chapter 9](09-thirty-worlds.md)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](../notes/frames-and-stats.md)).
>
> 1. Take `orphaned` of the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 1,499.
> 2. For every n, add up `orphaned` over the rows with `orphaned` ≥ n and divide by the total.
>
> **To make it again:** `python3 book_figures.py breaking`.
<!-- /figure -->

Half of all agents cut off died in games that cut off at least 353; the
largest single loss was 9,460 agents. One game in fifty cuts off a thousand or
more agents, and these games account for a sixth of all deaths by cutting.
Counted against the size of a world, it is much as at 10,000 tokens, where
half died in games that cut off at least 42 and one game in forty cut off a
hundred or more — a fifth of the deaths.

## What severs a region

Which rule cuts a region off? There is a clean way to tell. Take a game, the
agents it cut off, and the survivors. A cut-off agent that had a neighbour
among the survivors before the game must have **lost that connection to
pruning** — otherwise it would still be joined to them, and would have
survived. So the connections between the lost and the kept, as they stood
before the game, are exactly the connections pruned across the cut: the
region's **border**. Counting them tells what kind of break it was.

Of the 601 games that cut off a thousand agents or more (30 worlds, settled
life), **every one** was a whole border going quiet: ten or more connections
between a region and the rest of the world, all carrying nothing in the same
game — 1.4 million agents cut off in them. Not one big cut was a bridge
failing.

At 10,000 tokens it was nine in ten: of the 1,614 games that cut off a
hundred or more, 32 went through one or two connections, 161 through three
to nine, and 1,421 through ten or more. There, too, a single line of brains
once cut itself off — in the world of seed 1 at iteration 541, 60% of the 319
lost agents carried one genotype, and the typical one staked all its tokens
on its own node. In the large worlds no such cut shows: in the six largest
cuts of the world of seed 1, from 2,177 to 4,608 agents, lost across 202 to
843 connections, no genotype held more than 4% of the lost.

## Poverty disconnects

Why would ten connections go quiet at once? Look at the agents on the two
sides of the quiet borders (the games with ten or more border connections):

| | median tokens | median connections | hold fewer tokens than connections |
|---|---|---|---|
| lost side of the border | 1 | 2 | 52% |
| kept side of the border | 3 | 4 | 72% |
| every agent, same games | 3 | 2 | 13% |

The agents on both sides of a quiet border are **poor for their connections**.
Here is the arithmetic that makes that fatal. A connection survives a game only
if at least one token crosses it. An agent with τ tokens can send a token
across at most τ of its connections — fewer, if it keeps any at home. An agent
with one token and two connections can keep at most one of them alive by
itself; the other depends on the neighbour's stake. Every connection costs at
least one token of flow per game, and a poor region cannot pay for its
connections. When the richer agents beyond the border spend their stakes
elsewhere, the border goes quiet — and the region dies, all of it, with its
tokens handed out by lottery to the survivors.

## What this means

- **A world breaks along its poor regions, not at its weakest bridge.** The
  most fragile-looking connection almost never fails; the big losses are
  borders of agents too poor to keep their connections in use.
- **Connections have a price.** One token of flow per game, per connection,
  paid by one side or the other. That caps how many connections a world of
  *T* tokens can hold, which may be part of why a world's size follows its
  tokens ([Chapter 42](42-how-does-a-worlds-size-follow-its-tokens.md)).
- **The cull is global, and blind.** A region that is cut off dies whatever its
  brains did, even if it was doing well inside. Selection here acts on regions
  through their borders, not on agents through what they do — and a world can
  never split into two populations that go their own ways, which in nature is
  how new species begin. [Chapter 39](39-questioning-the-mechanics.md)
  questions this rule, and [Meta II](40-meta-2.md) what changing it might do.

> [!info] Added later: worlds that cannot break
> In the baseline's worlds of 5,000 tokens the cull removes about 2% of the
> agents in every iteration, four times as many as starve, and no world
> passes the health bar of
> [Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md), one of whose
> lines is a cut risk of at most 1%
> ([Chapter 47](47-brains-that-see-and-worlds-that-break.md)).
> [Chapter 48](48-a-world-that-cannot-break.md) answers with local rules —
> two links at birth, a death mended in a ring, only redundant links
> lapsing — under which neither a death nor a lapsed link can split a world.
> Letting separated populations live on is planned for Chapter 63, *Local
> rules only*.

To make every figure of this chapter: `python3 book_figures.py breaking`.

<!-- turns -->
---

← [Chapter 35 · The geometry of a world](35-the-geometry-of-a-world.md) · [Contents](../README.md) · [Chapter 37 · One world, many colours](37-one-world-many-colours.md) →
<!-- /turns -->
