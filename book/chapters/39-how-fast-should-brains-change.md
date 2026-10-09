# How fast should brains change?

> [!question] Questions of this chapter
> - Do the worlds reproduce at the same rate whatever the rate of mutation —
>   is three births per hundred agents something selection finds?
> - Is there a rate of mutation beyond which selection can no longer keep
>   what it has found?
> - How does the rate of mutation change how long genotypes and lines last —
>   the heredity open-ended evolution needs?

<!-- experiment E09 -->

> [!warning] Waiting for the runs
> The runs of this chapter are still to be made. The plan, the thesis and
> the method below were written before any of them existed; the results
> follow when they are done.

## Why ask

Evolution needs **variation** ([Chapter 1](01-what-this-book-is-about.md)).
In Graph of Life a brain is offered a change at two moments — when it is
copied into a child, and after every game — and each time it changes with
probability *p* = 0.2 ([How a brain changes](../notes/mutation.md)). Is that
a lot or a little?

Too little, and a world is stuck with what it started with.
[Chapter 37](37-do-the-brains-matter.md) took change away altogether, and
worlds whose brains never change ended up anywhere: some frozen, with no one
born and no one dying, some teeming, with a child for every agent in every
iteration. Too much, and selection cannot keep what it finds. Between the two
lies the range in which evolution works — but even inside it, the rate of
change can leave its mark. Theory offers three pictures of how:

1. **Selection finds a rate.** If one way of behaving is best, any world that
   varies enough to find it ends up there, however fast it varies: the rate of
   mutation changes how fast a world arrives, not where.
2. **Mutation–selection balance** (Haldane 1927). Changes are blind, so they
   keep breaking what selection has built, and selection keeps repairing it.
   Where the two balance — how much is broken at any moment — rises steadily
   with how often things break: twice the mutation, about twice the damage.
3. **An error threshold** (Eigen 1971; Eigen and Schuster 1977). Past a
   critical rate of mutation, a well-adapted type melts into a cloud of
   variants faster than selection can favour it, and what was learned is lost
   all at once — an "error catastrophe". Below the threshold little changes;
   above it, everything does.

Two findings of this book make the question sharper.

- **Three births per hundred agents.** The 26 evolving worlds of
  [Chapter 37](37-do-the-brains-matter.md) all settled near three births per
  hundred agents per iteration, while worlds without change spread from none
  to a hundred. Is three what selection finds — a rate any evolving world
  arrives at — or only the churn of mutation, which keeps brains from
  settling anywhere extreme? If it is churn, it should move with the amount
  of churn.
- **Change in the living.** [Chapter 35](35-questioning-the-mechanics.md)
  found that 97% of new genotypes arise in brains that are not being copied,
  after a game. The rate of mutation decides how fast a successful brain is
  eroded in place, and so how long a line stays like its ancestor — the
  heredity on which the first rung of the ladder towards open-ended evolution
  rests (P1 in [Chapter 1](01-what-this-book-is-about.md)).

So this experiment changes one number, the probability *p* that a brain
offered a change takes it, and leaves everything else — how large a change
is, what it touches, when it is offered — as in the baseline. Births per
hundred agents are the test between the three pictures: flat across the rates
for the first, rising steadily for the second, flat and then a jump for the
third.

## The thesis

<!-- thesis E09 -->
> [!quote] The thesis of Experiment 9, written down before any of its runs existed
> **The claim.** The rate of births is set by how much brains change, not fixed by selection. Births per hundred agents rise steadily with the mutation probability: settled worlds at 0.05 have fewer than the baseline's three per hundred agents per iteration, worlds at 0.8 at least twice as many. Selection holds reproduction down; every change of a brain undoes a little of that, and where the two balance moves with how often brains change.
>
> **Why it would be so.** Founders, whose brains no selection has touched, have children some twenty times as often as settled agents (Chapter 28): selection holds reproduction down. A change to a brain is blind, so it can undo that as easily as anything else. When selection removes a trait that mutation keeps making, the trait settles at a level that rises with the rate of mutation — mutation–selection balance, the oldest result of population genetics about variation (Haldane 1927). Two pilots run to plan this chapter — seed 1 at 10,000 tokens for 600 iterations, at mutation probabilities 0.05 and 0.8 — point that way. Over iterations 500 to 599, births per hundred agents were 2.0 at 0.05, 2.9 in the baseline's world of the same seed at 0.2, and 8.0 at 0.8; genotypes per agent were 0.32, 0.60 and 0.98.
>
> **It holds if:** Over iterations 500 to 2,999 of the worlds that live to the end: the median world's mean births per hundred agents per iteration rises at every step from mutation probability 0.05 to 0.1, 0.2, 0.4 and 0.8; the difference from the baseline, paired by seed, is below 0 at 0.05 and above 0 at 0.8, both with 95% intervals that leave out 0; and the median at 0.8 is at least twice the baseline's 3.1.
>
> **It fails if:** Births per hundred agents stay within 30% of the baseline's (2.2 to 4.1) at every mutation probability, or do not rise with it. Then the baseline's three births per hundred agents is a rate selection finds, whatever the churn of mutation.
<!-- /thesis -->

## Method

<!-- runs E09 -->
> [!info] The runs behind this chapter
> **150 runs.** Every condition below is run once for every seed (1 to 30), for 3,000 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it.
>
> - **baseline** — changes nothing: B1 as it is; 10,000 tokens. Runs `B1-10000-s001` … `B1-10000-s030`.
> - **mutation 0.05** — changes `mutation_probability` = 0.05; 10,000 tokens. Runs `B1-10000-bf74a9-s001` … `B1-10000-bf74a9-s030`.
> - **mutation 0.1** — changes `mutation_probability` = 0.1; 10,000 tokens. Runs `B1-10000-87318c-s001` … `B1-10000-87318c-s030`.
> - **mutation 0.4** — changes `mutation_probability` = 0.4; 10,000 tokens. Runs `B1-10000-56c74d-s001` … `B1-10000-56c74d-s030`.
> - **mutation 0.8** — changes `mutation_probability` = 0.8; 10,000 tokens. Runs `B1-10000-fe3009-s001` … `B1-10000-fe3009-s030`.
>
> To make these runs again: `python3 gol_lab.py run E09`, or ▶ in the Book tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | baseline | mutation 0.05 | mutation 0.1 | mutation 0.4 | mutation 0.8 | what it is |
> |---|---|---|---|---|---|---|
> | [`total_tokens`](../notes/settings.md#total_tokens) | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](../notes/settings.md#n_nodes) | 0 | 0 | 0 | 0 | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](../notes/settings.md#k_neighbors) | 0 | 0 | 0 | 0 | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](../notes/settings.md#rewire_p) | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](../notes/settings.md#hidden_layers) | 50, 45, 40, 35, 30 | 50, 45, 40, 35, 30 | 50, 45, 40, 35, 30 | 50, 45, 40, 35, 30 | 50, 45, 40, 35, 30 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](../notes/settings.md#brain_kind) | float16 | float16 | float16 | float16 | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](../notes/settings.md#brain_bits) | 16 | 16 | 16 | 16 | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](../notes/settings.md#message_amount) | 30 | 30 | 30 | 30 | 30 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](../notes/settings.md#random_input_amount) | 5 | 5 | 5 | 5 | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](../notes/settings.md#exchange_messages) | on | on | on | on | on | Whether agents send and read messages at all. |
> | [`message_prepass`](../notes/settings.md#message_prepass) | on | on | on | on | on | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](../notes/settings.md#allow_handover) | on | on | on | on | on | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](../notes/settings.md#allow_revolutions) | on | on | on | on | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](../notes/settings.md#allow_gifting) | off | off | off | off | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](../notes/settings.md#random_decisions) | off | off | off | off | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](../notes/settings.md#prune_after) | blotto | blotto | blotto | blotto | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](../notes/settings.md#inactive_window) | phase | phase | phase | phase | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](../notes/settings.md#redistribution) | uniform | uniform | uniform | uniform | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds. |
> | [`tokens_created_per_phase`](../notes/settings.md#tokens_created_per_phase) | 0 | 0 | 0 | 0 | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutation_probability`](../notes/settings.md#mutation_probability) | 0.2 | **0.05** | **0.1** | **0.4** | **0.8** | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](../notes/settings.md#mutation_noise_std) | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](../notes/settings.md#mutation_sparsity) | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](../notes/settings.md#extinction_threshold) | 20 | 20 | 20 | 20 | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 30 | 1 to 30 | 1 to 30 | 1 to 30 | 1 to 30 | one run per seed |
> | iterations | 3,000 | 3,000 | 3,000 | 3,000 | 3,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

Five rates of mutation, each double the one before: *p* = 0.05, 0.1, 0.2,
0.4 and 0.8 — from a change every twenty offers to one in almost every
offer. The baseline is *p* = 0.2, and its thirty worlds are those of
[Chapter 9](09-thirty-worlds.md): nothing is run again for it. Every other
rate is run with the same thirty seeds, at 10,000 tokens, for 3,000
iterations. Two worlds with the same seed start from the same founders, the
same brains and the same ring; they part at the first change one of them
takes and the other does not.

**What is measured.** Over the settled life of every world that lives to the
end, iterations 500 to 2,999 ([The settled life](../notes/settled-life.md)):
births per agent in a reproduction phase, the number of agents, inequality,
the share of nodes kept by their own agent, genotypes per agent and families
([Families](../notes/families.md)); and, from the frames, how long genotypes
last and how often one branch of the family tree replaces all others
([The common ancestor](../notes/common-ancestor.md)), as in
[Chapter 37](37-do-the-brains-matter.md). Each rate is compared with the
baseline seed by seed: the difference of the means, an interval for it from
resampling the seeds, and how often flipping the signs of the differences at
random gives one as large ([Permutation tests](../notes/permutation-test.md)).
Worlds that fall to twenty agents are stopped, as in the baseline, and counted
apart ([When a world ends](../notes/extinction.md)) — remembering, from
[Chapter 38](38-how-does-a-worlds-size-follow-its-tokens.md), that not every
one of them would have died.

**What each rate means.** Every brain is offered a change after every game,
so the expected number of games between two changes of one brain is 1/*p*:
twenty games at 0.05, five at the baseline's 0.2, about one at 0.8. A newborn
is offered one more change at birth.

**What it costs.** 120 new runs: the lab estimates about 31 hours on four
workers and 48 GB of disk for the whole experiment, frames and checkpoints,
the baseline's existing runs included. (Changing brains more often costs little: in the
pilots, an iteration took about 1.0 milliseconds per agent at 0.8 and 0.6 at
0.05 — and the baseline's own runs took 1.1.)

## Results

*Waiting for the runs.*

<!-- turns -->
---

← [Chapter 38 · How does a world's size follow its tokens?](38-how-does-a-worlds-size-follow-its-tokens.md) · [Contents](../README.md) · [Chapter 51 · Worlds of 100,000 tokens](51-worlds-of-100000-tokens.md) →
<!-- /turns -->
