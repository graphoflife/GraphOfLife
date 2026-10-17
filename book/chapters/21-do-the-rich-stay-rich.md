# Do the rich stay rich?

[Chapter 15](15-where-do-the-tokens-go.md) found the tokens of a world held
unequally, with a Gini coefficient that hardly moves over a world's settled
life. But an unchanging inequality can hide two very different worlds. In
one, the same agents are rich game after game. In the other, wealth is as
unequal at every moment, but who holds it keeps changing. This chapter
follows single agents through time to tell which world Graph of Life is.

> [!question] Questions of this chapter
> - If an agent is rich now, how likely is it to be rich ten, a hundred,
>   two hundred games later?
> - How far does an agent move up or down the order of wealth?
> - What becomes of the very richest?

The worlds are the thirty of 100,000 tokens of [Chapter 9](09-thirty-worlds.md);
where the thirty of 10,000 tokens this chapter was first written from differ,
both are given.

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
> | [`redistribution`](../notes/settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](../notes/settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](../notes/settings.md#mutate_on_replication) | off | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](../notes/settings.md#incumbency) | 1.0 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](../notes/settings.md#birth_links) | 0 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](../notes/settings.md#mend_on_death) | off | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](../notes/settings.md#prune_needs) | 0 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](../notes/settings.md#valence) | 0 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](../notes/settings.md#keep_home) | 0.0 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](../notes/settings.md#max_age) | 0 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](../notes/settings.md#buy_life) | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](../notes/settings.md#min_child) | 1 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`mix_at_birth`](../notes/settings.md#mix_at_birth) | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](../notes/settings.md#mix_mates) | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`link_search`](../notes/settings.md#link_search) | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](../notes/settings.md#link_steering) | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](../notes/settings.md#link_consent) | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`mutation_probability`](../notes/settings.md#mutation_probability) | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](../notes/settings.md#mutation_noise_std) | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](../notes/settings.md#mutation_sparsity) | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](../notes/settings.md#extinction_threshold) | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 30 | one run per seed |
> | iterations | 1,500 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

## How the order of wealth is remembered

Line up the agents of a world by their tokens, the poorest first, and do it
again some games later, for the agents still alive. If the order is the
same, the rich stayed rich. Spearman's **rank correlation** ρ measures how
alike the two orders are, from 1 (exactly the same) through 0 (unrelated)
([Mobility](../notes/mobility.md)).

<!-- figure rich/memory -->
![How long wealth lasts](../figures/rich/memory.svg)

**How long wealth lasts.** Left: the agents alive after the game of an iteration t, and the same agents k games later, if they are still alive: Spearman's rank correlation of their tokens then and later ([Correlation](../notes/correlation.md)) — 1 if every agent kept its place in the order of wealth, 0 if the later order has nothing to do with the earlier. Right: the share of the agents still alive k games later. For each world, the mean over 8 moments t = 500, 600, …, 1,200 (leaving out, for the correlation, the few moments after which fewer than three of the agents were alive); the line is the median of the 30 worlds, the darker band the middle half of them and the paler band nine in ten.

> [!example]- How to make this figure
> **Runs.** The 30 baseline worlds that lived to the end (`B1-100000-s001` … `-s030`, [Chapter 9](09-thirty-worlds.md); `python3 gol_lab.py run E10`).
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. For t = 500, 600, …, 1,200 and k = 1, 2, 5, 10, 20, 50, 100, 200: read frames 2·t + 1 and 2·(t + k) + 1; the agents (`ids`) in both, and their `tokens` in each.
> 2. Spearman's ρ: the correlation of the ranks of the two token counts (ties share their mean rank); alive: agents in both over agents in the first.
> 3. Mean over t for each world; median and quantiles over worlds.
>
> **To make it again:** `python3 book_figures.py rich`.
<!-- /figure -->

After one game, ρ = **0.86**. After 10 games, 0.77; after 50, 0.57; after
100, 0.46; after 200, 0.33 — almost exactly as at 10,000 tokens. Meanwhile the
agents themselves die, a little faster than in the small worlds: 84% are
still alive after 10 games, 50% after 50, 30% after 100, 10% after 200 (87%,
58%, 36% and 16% at 10,000 tokens).

The order of wealth is forgotten slowly, and in a particular way. A process
that forgets a fixed fraction every game would have ρ falling like
0.86, 0.86², 0.86³, … — to 0.22 after 10 games and to nothing after 50. In
the worlds, instead, **ρ falls by about 0.1 every time the lag doubles**:
0.07 from 10 to 20 games, 0.11 from 50 to 100 and 0.13 from 100 to 200.
That is how memory fades when it is held by many things that last for very
different times. Here the likely holders are the connections. An agent's tokens follow its number of
connections plus one ([Chapter 34](34-how-properties-scale-together.md)),
and a hub's connections outlast many games, while a leaf's change all the
time.

## A two-game rhythm

The left panel has a small surprise at its start. After **two** games the
order is *closer* to the first than after one: ρ = 0.88 against 0.86. The
richest show it too (below): 76% of the richest hundredth are still in the
richest tenth two games later, but only 73% one game later.

[Chapter 29](29-where-the-tokens-flow.md) shows that the brains move tokens
almost as an even split would: each agent stakes the same share on itself
and on each of its neighbours. Take a hub with *k* neighbours that have no
other connections. The hub keeps 1/(*k* + 1) of its tokens and gives each
leaf as much; each leaf keeps half of its tokens and gives the hub half.
After one game,

$$
\tau_h' = \frac{\tau_h}{k + 1} + \frac{k}{2}\,\tau_l, \qquad
\tau_l' = \frac{\tau_h}{k + 1} + \frac{1}{2}\,\tau_l .
$$

The total τ_h + *k* τ_l does not change. The world is at rest when the hub
holds (*k* + 1) shares and each leaf 2. Any departure from rest is
multiplied, game by game, by the second eigenvalue of this map,

$$
\lambda = \frac{1}{k + 1} - \frac{1}{2},
$$

which is **negative** for every hub with two neighbours or more: −0.17 for
*k* = 2, −0.4 for *k* = 9. A negative factor flips the sign of the
departure every game. A hub above its resting share falls below it in the
next game and comes back above it in the one after, each time by less than
half as much. Tokens slosh between hubs and leaves, and the order of wealth
swings with a period of two games. The even split is the step of a diffusion
([Chapter 7](07-physical-inspiration.md)), and this rhythm is its overshoot
on the star-shaped parts of the network.

## Moving up and down

<!-- figure rich/moves -->
![From one fifth to another, ten games later](../figures/rich/moves.svg)

**From one fifth to another, ten games later.** All agents alive after the game of t = 500, 600, …, 1,200 in the 30 worlds, sorted into fifths of the world by their tokens (ranks, ties shared), and where each one was ten games later: in which fifth, or dead. Each row adds up to 1; a cell's colour is the share of its row.

> [!example]- How to make this figure
> **Runs.** The 30 baseline worlds that lived to the end (`B1-100000-s001` … `-s030`, [Chapter 9](09-thirty-worlds.md); `python3 gol_lab.py run E10`).
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. As in the figure above, with k = 10: each agent's fifth is ⌈rank / n · 5⌉ among the n agents of its frame.
> 2. Count the agents in every pair of fifths, and those not in the later frame as dead; divide each row by its total.
>
> **To make it again:** `python3 book_figures.py rich`.
<!-- /figure -->

Ten games on, where are the agents of each fifth of the order of wealth?

- Of the **richest fifth**, 68% are still in it, 12% have fallen one fifth,
  and 12% are dead.
- Of the **middle fifths**, 37% to 44% are where they were, and most of the
  others are one fifth up or down.
- Of the **poorest fifth**, 34% are still in it, 24% have risen one fifth,
  and **29% are dead**. Poverty is where agents die:
  [Chapter 36](36-how-a-world-breaks.md) shows why — a region too poor to
  keep its connections in use is cut off whole.

Among those still alive, the **Shorrocks index** of the moves — 0 for a
world in which nobody changes place, 1 for one that forgets the order
completely in ten games ([Mobility](../notes/mobility.md)) — is **0.59**
(0.57 at 10,000 tokens).
The top is the stickiest part of the order: an agent at the top has the
most connections to hold it there.

## The fate of the richest

<!-- figure rich/richest -->
![The fate of the richest](../figures/rich/richest.svg)

**The fate of the richest.** The agents in the richest hundredth of their world after the game of t = 500, 600, …, 1,200 (36,656 in all, pooled over the 30 worlds), followed k games on. Left: the share of them still in the richest tenth of their world, and the share dead. Right: for those alive, their tokens then divided by their tokens at t — the median in each world, and the median of the worlds.

> [!example]- How to make this figure
> **Runs.** The 30 baseline worlds that lived to the end (`B1-100000-s001` … `-s030`, [Chapter 9](09-thirty-worlds.md); `python3 gol_lab.py run E10`).
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. As in the first figure: the richest hundredth are the agents with tokens at or above the 99th percentile of their frame; the richest tenth later, at or above the 90th.
>
> **To make it again:** `python3 book_figures.py rich`.
<!-- /figure -->

Follow the richest hundredth of a world — 36,656 agents over all the moments
and worlds. One game later, 73% are still in the richest tenth and 4% are
dead. Ten games later, 56% and 25%; a hundred games later, 19% and 70%. The
survivors keep almost all they had for ten games (87% in the median), then
lose it steadily: a hundred games later they hold a quarter of what they held.

So the very richest **die faster** at first than agents in general: 25%
within ten games, against 16% of all agents. That is odd. Nearly half of
the richest are hubs with ten connections or more, and a hub is safe. Which
of them die?

<!-- figure rich/fragile -->
![Which of the richest die](../figures/rich/fragile.svg)

**Which of the richest die.** The agents alive after the game of t = 500, 600, …, 1,200 in the 30 worlds, in classes of their connections at t: the share no longer alive ten games later, for the richest hundredth of their world (red) and for every agent (grey). Pooled over moments and worlds: 7,069 of the richest and 1,367,720 in all with 1, 3,674 of the richest and 818,555 in all with 2, 8,721 of the richest and 1,223,956 in all with 3–9, 11,114 of the richest and 148,486 in all with 10–49, 6,078 of the richest and 6,405 in all with 50+ connections.

> [!example]- How to make this figure
> **Runs.** The 30 baseline worlds that lived to the end (`B1-100000-s001` … `-s030`, [Chapter 9](09-thirty-worlds.md); `python3 gol_lab.py run E10`).
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. For t = 500, 600, …, 1,200: read frames 2·t + 1 and 2·(t + 10) + 1; each agent's connections at t from `edges`; the richest hundredth as above.
> 2. Per class of connections, the agents of the first frame missing from the second, over all agents of the class (`book_chapters.dynamics.fragile`).
>
> **To make it again:** `python3 book_figures.py rich`.
<!-- /figure -->

Not the hubs. Among the richest with ten connections or more, 7% to 9% die
within ten games, as many as among other agents with that many connections.
The excess is all among the **rich with few connections**. Of the richest
agents with a single connection, **54%** are dead ten games later, against
18% of all agents with a single connection. With two connections it is 44%
against 17%.

Wealth on a dead end does not last. An agent with one connection hangs on
it: if that connection is cut, or the piece it leads to is cut off from the
rest of the world, the agent dies, and dies holding its tokens. The rules
then deal those tokens out among all survivors
([Births and the four ways to die](../notes/births-and-deaths.md)). A rich
leaf is a large pile of tokens on a single thread. When the thread breaks,
its wealth returns to everyone at once. Concentrated wealth that is not also
connectedness is taxed away in this manner, a little at a time.

## What this means

- **The rich stay rich for a while, and then they don't.** The order of
  wealth is half forgotten in fifty to a hundred games and mostly forgotten
  in 200.
  It is remembered as long as the connections that carry it, because tokens
  follow connections.
- **Wealth without connections is fragile.** The richest agents on dead ends
  die three times as often as other agents on dead ends. Wealth is safe only
  where the network is thick.
- **Tokens move like a diffusion, overshoot included.** The two-game rhythm
  is what an even split does on a network of hubs and leaves. It is one more
  sign that the game, as the brains play it, is close to a random walk of
  tokens ([Chapter 29](29-where-the-tokens-flow.md)). The rich are those
  that the walk's resting state favours, not those that played well.
- **For open-ended evolution**, this world rewards position, not strategy.
  A lineage cannot bank an advantage as tokens for long. Wealth drains back
  towards the resting share of its connections within tens of games.
  Selection can only act on what persists: connections, and the brains that
  sit on them.

To make every figure of this chapter: `python3 book_figures.py rich`.

<!-- turns -->
---

← [Chapter 20 · What agents say to each other](20-what-agents-say-to-each-other.md) · [Contents](../README.md) · [Chapter 22 · How the network grows](22-how-the-network-grows.md) →
<!-- /turns -->
