# How fast should brains change?

> [!question] Questions of this chapter
> - Does the rate at which brains change decide where evolution stops, or
>   only how soon it gets there?
> - Is there a rate past which selection can no longer keep what it has found?
> - What does the rate do to a world — to how many kinds of brain it holds, and
>   how often its nodes change hands?

<!-- experiment E09 -->

## Why ask

Evolution needs **variation** ([Chapter 1](01-what-this-book-is-about.md)).
In B2 a brain changes only when it is copied — into a newborn, or into a node
its owner has just won — and then with probability *p* = 0.2
([How a brain changes](../notes/mutation.md);
[`mutate_on_replication`](../notes/settings.md#mutate_on_replication)). One
copy in five is a new genotype. Is that a lot or a little?

[Chapter 50](50-why-evolution-stops.md) proposed that evolution stops because
the game has an answer: a contest for nodes is best played at random, and
once a population plays it well, nothing better exists to be selected. That
reading makes a prediction about *p*. Theory offers three pictures of what the
rate of change can do:

1. **Selection finds the answer.** If one way of playing is best, any world
   that varies enough to find it ends up there, however fast it varies: the
   rate of change decides how soon a world arrives, not where.
2. **Mutation–selection balance** (Haldane 1927). Changes are blind, so they
   keep breaking what selection has built, and selection keeps repairing it.
   How much is broken at any moment rises with how often things break: the
   brains of a faster-changing world are a little worse, the more the faster.
3. **An error threshold** (Eigen 1971; Eigen and Schuster 1977). Past a
   critical rate, a well-adapted type melts into a cloud of variants faster
   than selection can favour it, and what was learned is lost all at once.
   Below the threshold little changes; above it, everything does.

If the game has an answer, the first picture should hold over a wide range of
rates. If evolution stops for another reason — because damage keeps pace with
repair — the second or third should.

## Why the question moved

This experiment was first planned on the baseline B1, to ask whether three
births per hundred agents is a rate selection finds or the churn of
mutation. Two pilots at 10,000 tokens (seed 1, 600 iterations) pointed to
churn: births per hundred agents were 2.0 at *p* = 0.05, 2.9 at the baseline's
0.2 and 8.0 at 0.8, and genotypes per agent 0.32, 0.60 and 0.98 — a
mutation–selection balance, in a world whose brains are nearly blind.

In B2 that test is gone. Every agent dies at age 100, so births are held at
one in a hundred agents per iteration whatever the brains want
([Chapter 49](49-who-dies-and-why.md)). What the rate of change can still
move is how good the brains become — and that is the question Part V needs
answered.

## How the rates are compared

Five rates, each double the one before: *p* = 0.05, 0.1, 0.2 (B2), 0.4 and
0.8 — from one copy in twenty changed to four in five. Each is run with ten
seeds at 10,000 tokens for 1,500 iterations, archiving the whole world every
100. Worlds with the same seed start from the same founders, the same brains
and the same ring; they part at the first copy one of them changes and the
other does not.

Two kinds of contest are played, both as in
[Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md) — a world whose
agents each get a brain from one side or the other by a coin, mutation off,
twenty iterations, and the share each side holds at the end:

- **Across rates.** For each seed and each pair of rates, the brains of
  iteration 1,500 of one rate play those of the other, once in each rate's
  world at iteration 1,500. Ten pairs, ten seeds, two worlds: 200 contests.
  Under the first picture every pair ties; under the second the slower rate
  wins, the more the further apart; under the third the fastest rate loses to
  all the others.
- **Within a rate.** Each world plays its own eras, as in Chapter 46 — to see
  when each rate stops, and whether the slowest has stopped at all by
  iteration 1,500.

Beside them, the world statistics of every run: agents, births, genotypes per
agent, and the share of nodes kept by their own agent in a game.

## The thesis

<!-- thesis E09 -->
> [!quote] The thesis of Experiment 9, written down before any of its runs existed
> **The claim.** How fast brains change decides how soon evolution stops, not where. In B2 at 10,000 tokens, with brains changed only when they are copied, the brains of iteration 1,500 tie whatever the rate of change they evolved under: for every pair of the mutation probabilities 0.05, 0.1, 0.2, 0.4 and 0.8, the brains of one rate hold between 45% and 55% of a world contested with the other's, in the median seed. And every rate has stopped by then: the brains of iteration 1,500 hold between 45% and 55% against those of iteration 1,000 of the same world.
>
> **Why it would be so.** Chapter 53 reads the end of evolution as the end of a game: a contest for nodes has a best way to play — be unpredictable — and once a population plays it, nothing better exists to be selected. If that is right, every world that varies enough to find it ends in the same place, and the rate of change only sets how fast it gets there (selection finds the answer). The two other pictures of Chapter 43 predict otherwise: in a mutation-selection balance (Haldane 1927) the brains of a faster-changing world carry more damage at any moment and lose to those of a slower one, the more the faster; past an error threshold (Eigen 1971) the fastest world loses what it learned altogether. No pilot has varied the rate in B2. In B1 two pilots at 10,000 tokens (seed 1, 600 iterations) found births of 2.0, 2.9 and 8.0 per hundred agents at 0.05, 0.2 and 0.8 — a balance in a world whose brains are nearly blind. In B2 births are held at one in a hundred by old age (Chapter 52), so the question moves from how often agents breed to how good their brains become.
>
> **It holds if:** For each of the ten pairs of rates, in each seed, the brains of iteration 1,500 of one rate play those of the other twice — once in each rate's world of that seed at iteration 1,500 (gol_tournament.contest); a pair's score is the mean share of the second rate's brains; the median over the ten seeds lies between 0.45 and 0.55 for all ten pairs. And in each of the five conditions, the median world's score of iteration 1,500 against iteration 1,000 (two contests) lies between 0.45 and 0.55.
>
> **It fails if:** A pair of rates outside 0.45 to 0.55 in the median seed: the rate of change decides how good the brains become — by damage that grows with the rate if the slower rate wins, by too little variation if the faster wins. Or a condition still improving after iteration 1,000 (above 0.55): that rate has not stopped.
<!-- /thesis -->

("Chapter 53", "Chapter 43" and "Chapter 52" in the thesis are today's [Chapter 50](50-why-evolution-stops.md), this chapter, and [Chapter 49](49-who-dies-and-why.md): the chapters were renumbered on 2026-10-14.)

## The runs

<!-- runs E09 -->
> [!info] The runs behind this chapter
> **50 runs.** Every condition below is run once for every seed (1 to 10), for 1,500 iterations — or until its world dies out. Every setting is that of the baseline **B2** (a world that cannot break) unless the condition changes it.
>
> - **mutation 0.05** — changes `mutation_probability` = 0.05; 10,000 tokens. Runs `B2-10000-bf74a9-s001` … `B2-10000-bf74a9-s010`.
> - **mutation 0.1** — changes `mutation_probability` = 0.1; 10,000 tokens. Runs `B2-10000-87318c-s001` … `B2-10000-87318c-s010`.
> - **B2** — changes nothing: B2 as it is; 10,000 tokens. Runs `B2-10000-s001` … `B2-10000-s010`.
> - **mutation 0.4** — changes `mutation_probability` = 0.4; 10,000 tokens. Runs `B2-10000-56c74d-s001` … `B2-10000-56c74d-s010`.
> - **mutation 0.8** — changes `mutation_probability` = 0.8; 10,000 tokens. Runs `B2-10000-fe3009-s001` … `B2-10000-fe3009-s010`.
>
> To make these runs again: `python3 gol_lab.py run E09`, or ▶ in the Book tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | mutation 0.05 | mutation 0.1 | B2 | mutation 0.4 | mutation 0.8 | what it is |
> |---|---|---|---|---|---|---|
> | [`total_tokens`](../notes/settings.md#total_tokens) | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](../notes/settings.md#n_nodes) | 0 | 0 | 0 | 0 | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](../notes/settings.md#k_neighbors) | 0 | 0 | 0 | 0 | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](../notes/settings.md#rewire_p) | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](../notes/settings.md#hidden_layers) | 16 | 16 | 16 | 16 | 16 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](../notes/settings.md#brain_kind) | float16 | float16 | float16 | float16 | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](../notes/settings.md#brain_bits) | 16 | 16 | 16 | 16 | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](../notes/settings.md#message_amount) | 8 | 8 | 8 | 8 | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](../notes/settings.md#random_input_amount) | 5 | 5 | 5 | 5 | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](../notes/settings.md#exchange_messages) | on | on | on | on | on | Whether agents send and read messages at all. |
> | [`message_prepass`](../notes/settings.md#message_prepass) | off | off | off | off | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](../notes/settings.md#allow_handover) | off | off | off | off | off | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](../notes/settings.md#allow_revolutions) | on | on | on | on | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](../notes/settings.md#allow_gifting) | off | off | off | off | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](../notes/settings.md#random_decisions) | off | off | off | off | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](../notes/settings.md#prune_after) | blotto | blotto | blotto | blotto | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](../notes/settings.md#inactive_window) | phase | phase | phase | phase | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](../notes/settings.md#redistribution) | uniform | uniform | uniform | uniform | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds. |
> | [`tokens_created_per_phase`](../notes/settings.md#tokens_created_per_phase) | 0 | 0 | 0 | 0 | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](../notes/settings.md#mutate_on_replication) | on | on | on | on | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](../notes/settings.md#incumbency) | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](../notes/settings.md#birth_links) | 2 | 2 | 2 | 2 | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](../notes/settings.md#mend_on_death) | on | on | on | on | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](../notes/settings.md#prune_needs) | 2 | 2 | 2 | 2 | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](../notes/settings.md#valence) | 8 | 8 | 8 | 8 | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](../notes/settings.md#keep_home) | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](../notes/settings.md#max_age) | 100 | 100 | 100 | 100 | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`min_child`](../notes/settings.md#min_child) | 10 | 10 | 10 | 10 | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`mutation_probability`](../notes/settings.md#mutation_probability) | **0.05** | **0.1** | 0.2 | **0.4** | **0.8** | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](../notes/settings.md#mutation_noise_std) | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](../notes/settings.md#mutation_sparsity) | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](../notes/settings.md#extinction_threshold) | 20 | 20 | 20 | 20 | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 10 | 1 to 10 | 1 to 10 | 1 to 10 | 1 to 10 | one run per seed |
> | iterations | 1,500 | 1,500 | 1,500 | 1,500 | 1,500 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B2.
<!-- /runs -->

> [!warning] Waiting for the runs
> The plan, the thesis and the method above were written before any run of
> Experiment 9 existed; the results follow when they are done. The B1 plan
> this chapter had before was never run.

<!-- turns -->
---

← [Chapter 50 · Why evolution stops](50-why-evolution-stops.md) · [Contents](../README.md) · [Chapter 54 · Worlds of 100,000 tokens](54-worlds-of-100000-tokens.md) →
<!-- /turns -->
