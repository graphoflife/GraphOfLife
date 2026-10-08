# How does a world's size follow its tokens?

> [!question] Questions of this chapter
> - Does a world of twice the tokens hold twice the agents and twice the
>   connections?
> - Is the number of agents a power law of the tokens — and if so, with what
>   exponent?
> - How small can a world be and still live?

<!-- experiment E08 -->

> [!warning] Waiting for the runs
> The runs of this chapter are still being made. The plan, the thesis and
> the method below were written before any of them existed; the results
> follow when they are done.

## Why ask

Every chapter so far looked at worlds of 10,000 tokens. Whether what they
found holds at other sizes rests first on this question. If agents and
connections grow in proportion to the tokens, a bigger world is more of the
same, and the size of a world can be chosen for what an experiment can
afford. If they do not, size changes what a world is, and every result
belongs to the size it was found at.

## The thesis

<!-- thesis E08 -->
> [!quote] The thesis of Experiment 8, written down before any of its runs existed
> **The claim.** A world's size follows its tokens in proportion. From a few thousand tokens up to 409,600, a settled world holds about 0.13 agents and 0.2 connections for every token, whatever its size: agents and connections are each a power law of the tokens with exponent 1. A world of twice the tokens is two worlds of the same kind.
>
> **Why it would be so.** Everything an agent does is local. It plays the game with its neighbours, has its children beside itself, and dies when its own tokens run out; nothing in the rules reaches across the world except the sharing out of the tokens of the dead, which gives every survivor the same chance whatever the size of the world. So a bigger world should be more of the same, with as many agents per token and as many connections per agent. At 10,000 tokens the settled worlds of Chapter 3 held 0.13 agents and 0.20 connections per token, from iteration 500 on.
>
> **It holds if:** For the baseline worlds of 3,200 to 409,600 tokens that live to iteration 600, each measured by its mean over iterations 500 to 599: the exponent fitted to agents against tokens lies between 0.9 and 1.1 and its 95% interval contains 1; the same holds for connections; and neither set of points bends away from a straight line on logarithmic axes — the interval of the bend contains 0.
>
> **It fails if:** The exponent's interval leaves out 1, so that bigger worlds hold clearly fewer or more agents (or connections) per token than smaller ones, or the points bend on logarithmic axes. Then a world's size is more than its tokens, something in the dynamics reaches across the whole world, and every result in this book belongs to the size it was found at.
<!-- /thesis -->

("Chapter 3" in the thesis is today's [Chapter 9](09-a-worlds-life.md). And its
reasoning missed a rule: besides the sharing out, the cleanup also reaches
across the world, since it keeps only the largest connected piece —
[Chapter 28](28-questioning-the-mechanics.md) — which is one way the thesis
could fail.)

## Method

<!-- runs E08 -->
> [!info] The runs behind this chapter
> **42 runs.** Every condition below is run once for every seed (1 to 3) and every number of tokens it lists, for 600 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it.
>
> - **baseline** — changes nothing: B1 as it is; 800, 1,600, 3,200, 6,400, 12,800, 25,600, 51,200, 102,400, 204,800, 409,600 tokens. Runs `B1-800-s001` … `B1-409600-s003`.
> - **stopped only when empty** — changes `extinction_threshold` = 0; 800, 1,600, 3,200, 6,400 tokens. Runs `B1-800-57b358-s001` … `B1-6400-57b358-s003`.
>
> To make these runs again: `python3 gol_lab.py run E08`, or ▶ in the Book tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | baseline | stopped only when empty | what it is |
> |---|---|---|---|
> | [`total_tokens`](../notes/settings.md#total_tokens) | 800, 1,600, 3,200, 6,400, 12,800, 25,600, 51,200, 102,400, 204,800, 409,600 | 800, 1,600, 3,200, 6,400 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](../notes/settings.md#n_nodes) | 0 | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](../notes/settings.md#k_neighbors) | 0 | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](../notes/settings.md#rewire_p) | 0.2 | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](../notes/settings.md#hidden_layers) | 50, 45, 40, 35, 30 | 50, 45, 40, 35, 30 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](../notes/settings.md#brain_kind) | float16 | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](../notes/settings.md#brain_bits) | 16 | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](../notes/settings.md#message_amount) | 30 | 30 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](../notes/settings.md#random_input_amount) | 5 | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](../notes/settings.md#exchange_messages) | on | on | Whether agents send and read messages at all. |
> | [`message_prepass`](../notes/settings.md#message_prepass) | on | on | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](../notes/settings.md#allow_handover) | on | on | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](../notes/settings.md#allow_revolutions) | on | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](../notes/settings.md#allow_gifting) | off | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](../notes/settings.md#random_decisions) | off | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](../notes/settings.md#prune_after) | blotto | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](../notes/settings.md#inactive_window) | phase | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](../notes/settings.md#redistribution) | uniform | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds. |
> | [`tokens_created_per_phase`](../notes/settings.md#tokens_created_per_phase) | 0 | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutation_probability`](../notes/settings.md#mutation_probability) | 0.2 | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](../notes/settings.md#mutation_noise_std) | 0.2 | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](../notes/settings.md#mutation_sparsity) | 0.1 | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](../notes/settings.md#extinction_threshold) | 20 | **0** | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 3 | 1 to 3 | one run per seed |
> | iterations | 600 | 600 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

Worlds of the baseline B1 from 800 to 409,600 tokens, each size double the
one before — ten sizes, three seeds each, 600 iterations each. A world starts
with one founder for every hundred tokens, so the founders go from 8 to
4,096.

**What is measured.** For every run, the mean number of agents alive after
the game, and of connections, over iterations 500 to 599 — well past the
youth of [Chapter 10](10-the-first-hundred-iterations.md). The same means over iterations 400 to 499 say whether a
world had settled: a world still growing or shrinking would show it as a
difference between the two.

**How it is read.** On logarithmic axes, on both sides, a power law —
agents = *a* × tokens^*b*, some number *a* times the tokens to the power
*b* — is a straight line, and its slope is the exponent *b*
([Logarithmic axes and power laws](../notes/logarithmic-axes.md)). An
exponent of 1 means in proportion; below 1, bigger worlds hold fewer agents
per token; above 1, more. The line is fitted to the baseline
worlds of 3,200 tokens and more, every run a point. The interval of its slope
comes from resampling the runs of each size, and a second fit with a squared
term says whether the points bend away from a straight line. The exponent
between each size and the next shows where along the range any bend lies.
Connections per agent are read the same way: in proportion means a slope
of 0.

**The small end.** Two rules of the baseline limit it, both met in pilots
run to plan this chapter.

- *No world below 600 tokens.* The founders' starting ring needs more
  founders than neighbours, at least six, so the baseline cannot build a
  world of 500 tokens or fewer. The doubling starts at 800, not at 50.
- *A run stops at 20 agents or fewer.* Worlds of 800 and 1,600 tokens start
  with 8 and 16 founders, below that line. In the pilots, every 800-token
  world and two of three 1,600-token worlds were stopped as extinct after
  their first iteration; worlds of 3,200 tokens lived, with 282 and 481
  agents over iterations 500 to 599. The baseline worlds of 800 and 1,600
  tokens are run anyway, as the baseline does it. Beside them, a second
  condition — *stopped only when empty*, `extinction_threshold` 0 — lets
  worlds of 800 to 6,400 tokens go on as long as anyone is alive. A world
  that never falls to twenty agents is then exactly the baseline's world of
  the same seed, so where both run they must agree, and below that it shows
  how small a world can live. It has no thesis.

**The starting ring.** Each founder starts joined to its nearest founders,
four of them in worlds below 60,000 tokens and more in bigger ones — a
hundredth of the founders, so 40 at 409,600 tokens. That is the baseline,
and it is kept. The first games cut every connection that carries no
tokens, so by iteration 500 a world should have long forgotten how it
started.

**One worker.** The worlds run one after another on a single worker,
whatever else the lab is doing: the biggest needs several gigabytes of
memory. The lab estimates about two days in all, almost half of it the three
worlds of 409,600 tokens — an estimate that itself assumes what this chapter
tests, that a world's size grows in proportion to its tokens.

## Results

*Waiting for the runs.*

<!-- turns -->
---

← [Chapter 30 · Do the brains matter?](30-do-the-brains-matter.md) · [Contents](../README.md) · [Appendix A · What a simulation costs](A-costs.md) →
<!-- /turns -->
