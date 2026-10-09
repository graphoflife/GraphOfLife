# Worlds of 100,000 tokens

> [!question] Questions of this chapter
> - How does a line take over a world ten times the size: ten times slower,
>   as chance would, about as fast, as a better brain in a well-mixed world
>   would, or at the pace of a front that has twice as far to go?
> - Is a world of 100,000 tokens ten copies of a world of 10,000, or does
>   anything appear only at size: longer-lived regions, coexisting lines,
>   structure that a small world has no room for?
> - Do the measures of space of Part II — dimension, correlation length,
>   the width of a world — read the same at ten times the size?

<!-- experiment E10 -->

> [!warning] Waiting for the runs
> The runs of this chapter are still to be made. The plan, the thesis and
> the method below were written before any of them existed; the results
> follow when they are done.

## Why ask

Every finding of Parts II and III was made on worlds of 10,000 tokens, about
1,300 agents each. That is small. A world of 1,300 agents is some 30 steps
across ([Chapter 31](31-the-geometry-of-a-world.md)), and its structure
fades within three ([Chapter 22](22-like-next-to-like.md)): there is room
for a dozen regions side by side, no more. Questions about space, about
scale, and about whether different kinds of agents can live side by side
need worlds with more room.

[Chapter 38](38-how-does-a-worlds-size-follow-its-tokens.md) showed that a
bigger world is, on average, more of the same: agents, connections and
births all follow the tokens in proportion. It ran only three worlds of each
size, for 600 iterations. This experiment makes a **second baseline**:
thirty worlds of 100,000 tokens, ten times the baseline, each run for 1,500
iterations. Part VI builds on it — Chapter 52 is to measure its dimensions
at scale, and Chapter 53 to look for properties without a scale (both
planned, see [the contents](../README.md)) — and it answers one question of
its own: how a line takes over a world.

**How a line takes over.** [Chapter 17](17-genotypes-and-lineages.md) found
that whole lines of descent take the baseline worlds over again and again,
about once every 350 iterations. Is that chance or merit? Three answers give
three different paces, and the size of the world tells them apart:

1. **Chance.** If no line is better than another, all but one die out by
   chance in the end, in a time that grows in proportion to the number of
   agents: ten times the agents, ten times as long (the coalescent,
   [Chapter 6](06-the-ideas-this-builds-on.md)).
2. **Merit, in a well-mixed world.** A better line grows by a fixed factor
   every generation, and needs only a time that grows with the logarithm of
   the number of agents: ten times the agents, about a third longer.
3. **Merit, in space.** A better line spreads outwards from where it arose,
   as a front at a steady speed (Fisher 1937). The time grows with the width
   of the world.

A world is a space: it has a finite dimension
([Chapter 23](23-how-many-dimensions-does-a-world-have.md)), new connections
are only ever made inside a neighbourhood
([Chapter 21](21-how-the-network-grows.md)), and a node can only be taken by
a neighbour. Ten times the agents make a world only about twice as wide: its
diameter grows as agents^0.30 ([Chapter 38](38-how-does-a-worlds-size-follow-its-tokens.md)).
So the third answer predicts two to three times as long, the first ten, the
second barely longer.

[Chapter 18](18-what-the-brains-are-like.md) gives a reason to expect the
first answer: the brains are nearly blind and agree on every decision that
matters, which leaves little for one line to be better at. The worlds of
Chapter 38 point to the third: at 102,400 tokens, all three came from a
single founder by iterations 400, 400 and 525, where the median baseline
world needed 175.

## The thesis

<!-- thesis E10 -->
> [!quote] The thesis of Experiment 10, written down before any of its runs existed
> **The claim.** A line takes a world over the way a front crosses a country: at a steady pace, so that the time it needs grows with the world's width, not with its number of agents. A world of 100,000 tokens holds ten times the agents of the baseline but is only about twice as wide, so a line takes it over in two to five times as long as at 10,000 tokens — not ten times, as chance alone would need, and not about as fast, as a better brain spreading through a well-mixed world would.
>
> **Why it would be so.** Three paces are possible. If no line is better than another, all but one die out by chance, in a time that grows in proportion to the number of agents (the coalescent, Chapter 6). If a better line spreads through a well-mixed population, it grows by a fixed factor every generation, and the time grows only with the logarithm of the number of agents. If it spreads through space, it does so as a front at a steady speed (Fisher 1937), and the time grows with the width of the world. The worlds are spaces: they have a finite dimension (Chapter 23), their diameter grows as agents^0.30 (Chapter 38), and a node can only be taken by a neighbour (Chapter 3). At 10,000 tokens (Experiment 2) the living of the median world all came from a single founder by iteration 175, and of every world by 750. The worlds of Experiment 8 point to the front: at 102,400 tokens, all three worlds got there by iterations 400, 400 and 525.
>
> **It holds if:** Counting each world of 100,000 tokens that reaches iteration 1,500 by the first iteration at which all its living descend from a single founder — and a world that never gets there as later than all of them — the median lies between 350 and 875: two to five times the baseline's 175.
>
> **It fails if:** The median is below 350: a large world is taken over about as fast as a small one, as a better brain spreading through a well-mixed population would take it. Or it is above 875, or most worlds never get there by iteration 1,500: then the pace is set by the number of agents, as chance alone would set it.
<!-- /thesis -->

## Method

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
> | [`redistribution`](../notes/settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds. |
> | [`tokens_created_per_phase`](../notes/settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutation_probability`](../notes/settings.md#mutation_probability) | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](../notes/settings.md#mutation_noise_std) | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](../notes/settings.md#mutation_sparsity) | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](../notes/settings.md#extinction_threshold) | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 30 | one run per seed |
> | iterations | 1,500 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

Thirty worlds of the baseline B1 at 100,000 tokens, seeds 1 to 30, for
1,500 iterations each. Nothing else changes. A world of 100,000 tokens
starts with 1,000 founders in a ring and settles at about 13,000 agents.

**What is measured.**

- **How lines take over.** After every 25th iteration, how far back all the
  living share one ancestor, and the first iteration at which they all
  descend from a single founder ([The common ancestor](../notes/common-ancestor.md)):
  the thesis. Against the baseline's 30 worlds over the same iterations, and
  against the time a front would need, from the worlds' diameters.
- **Everything of Chapter 9**, over the settled life from iteration 500 to
  1,499: agents, connections, births, deaths, inequality, genotypes and
  families, each per token or per agent and set beside the baseline.
- **Space**, in the frames of iterations 500, 1,000 and 1,499: the two
  rulers of [Chapter 23](23-how-many-dimensions-does-a-world-have.md), the
  correlation functions of [Chapter 22](22-like-next-to-like.md), the
  domains of lines — how far apart two agents can be that share an ancestor
  of 25, 50 and 100 iterations back — and the widths of the worlds.

**What it costs.** The lab estimates about 40 hours on four workers, 52 GB
of disk (frames and checkpoints) and up to 3.7 GB of memory per run. At
this size a world needs about 12 seconds per iteration, and a run about
five hours. Together with Experiment 9 (48 GB), it fits on the disk as it is now
(185 GB free on 9 October 2026), with room to spare.

## Results

*Waiting for the runs.*

<!-- turns -->
---

← [Chapter 39 · How fast should brains change?](39-how-fast-should-brains-change.md) · [Contents](../README.md) · [Appendix A · What a simulation costs](A-costs.md) →
<!-- /turns -->
