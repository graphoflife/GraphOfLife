# Why evolution stops

> [!question] Questions of this chapter
> - Does a world that cannot break keep evolving, now that its brains can see?
> - Which of the changes that should give evolution more to work with —
>   memory, larger brains, a reward for cooperating, hidden information — keep
>   it going?
> - What is it about the game that lets evolution finish so soon?
> - What would it take to give evolution room to go on?

<!-- experiment E16 -->

## Why ask

The baseline learns what it learns in its first fifty to a hundred iterations
and then stops ([Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md)).
Its brains are nearly blind, so that was no surprise: evolution could only set
a few constants. Chapters 47 to 49 then built a world in which brains can see
without the world breaking — B2. The question now is the one Part V exists
for: in such a world, does evolution keep going?

## What the pilots tried

Every pilot below played its eras against each other as in Chapter 46 — the
brains of iterations 0, 100, 200, 500 and 1,000, two contests per pair — and
reports the rating of each era against the founders and the **step score**,
each era against the one before it. A world still improving keeps its step
scores above one half.

| the world (pilot) | rating against the founders | step scores after iteration 100 |
|---|---|---|
| B1, the baseline (5,000 tokens) | +283 at 100, then +320 to +345 to 600 | 0.45–0.59 |
| B2's rules, the last token kept (5,000) | +133 at 100, then +137 to +171 to 1,000 | a plateau from 200 |
| the same, with brains that can grow | +113 to +130 throughout | a plateau from 100 |
| 30% kept at home, memory of the last game's stakes (3,000) | +176 to +210 from 100 | 0.44–0.54 |
| … and a memory of six games, with brains that can grow | +87 to +246 | 0.48–0.69 |
| … and a yield on stakes returned (3,000) | +161 to +212 from 100 | 0.43–0.76 |
| … and fog: neighbours' tokens hidden (3,000) | +43 to +100 | 0.53–0.63 |
| … with support stakes (5,000) | +155, +194, +319, +337 (seed 1); +332 to +368 from 100 (seed 2) | 0.59–0.72; 0.47–0.52 |
| … without random inputs (5,000) | +134 at 100, then +166 to +184 | 0.49–0.63 |

Two lines stand out, and neither survives a second look. One world with
support stakes kept beating its predecessors to iteration 1,000; its twin seed
stopped at once. And the hidden-information worlds kept their step scores a
little above a half because they barely evolved at all: with neighbours' tokens
hidden, attacks stopped — 0.2 to 0.4% of the nodes changed hands in a game —
and with them nearly every copying of a brain, and so nearly every mutation.

Two more things were measured along the way.

**Brains that can grow did not grow for a reason.** A growable brain gains or
loses a hidden unit by mutation, and carries a *shadow* — a size that is
inherited and mutated exactly like the real one but used by nothing
([Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md#numbers-that-rise-and-mean-nothing)).
In four worlds without handover, the real size followed its shadow: what
rose, rose because a size cannot fall below its floor — the left wall — not
because larger brains won.

**Support was given up.** A stake marked as support adds to its target's
defence instead of contesting the node. It is a gift — the tokens go to the
node's owner — and it buys nothing a home stake would not buy better, since a
home stake counts one and a half times and support once. Evolution gave it
up: support fell from 200–300 tokens a game to about 20 by iteration 600.

## The game has an answer

Why should every world stop? Look at the game an agent plays
([Chapter 16](16-how-the-game-is-played.md)): it divides its tokens between
its own node and its neighbours', and each node goes to whoever stakes most on
it. That is, very nearly, the oldest game in game theory with a name of its
own — **Colonel Blotto** (Borel, 1921): two colonels divide their soldiers
between battlefields, and each battlefield goes to whoever sends more.

Blotto has a peculiar property: **no division is best.** Take three
battlefields and six soldiers each, and count the fields each side wins:

- (3, 3, 0) beats the even division (2, 2, 2) — it wins the first two fields;
- (4, 0, 2) beats (3, 3, 0) — the first and the third;
- (1, 1, 4) beats (4, 0, 2) — the second and the third;
- and the even division (2, 2, 2) beats (1, 1, 4) — the first and the second.

The beatings go round in a circle, and there is no way out of it: whatever
division a colonel picks, some other division beats it, so a colonel who is
predictable is beaten. The best a colonel can do is to be
**unpredictable** — to draw a division at random from a well-chosen
distribution, so that no opponent can aim at it. For two colonels such a
*mixed strategy* exists, and against it no other strategy does better (Gross
and Wagner, 1950; Roberson, 2006). The agents' game is not exactly Blotto —
many players, a node goes with every token staked on it, and the battlefields
are where the network puts them — but the circle is the same: a predictable
agent is one its neighbours can aim at.

Evolution finds that. In pilot worlds under B2's rules, the brains' decisions were probed
input by input: take an evolved agent with at least three tokens to divide,
redraw one kind of its inputs, and see whether its division changes. Redrawing
its **random inputs** changed 25 to 58% of the divisions; its own state, 2 to
7%; the neighbour's state, 4 to 23%; the messages it read, 2 to 23%; its
memory of the last game's stakes, 11 to 24%. The evolved brains use their
random inputs as dice. And once a population plays a good mixed strategy,
there is nothing left to select: a better strategy against it does not exist.
That is the reading this chapter proposes, and Experiment 16 tests: the game
has an answer, evolution finds it in a few hundred iterations, and then it
stops — not because the brains are too simple to do better, but because there
is no better to do.

It would explain why more memory, larger brains or hidden information
changed little: each gives a brain more to compute *with*, but the game asks
for nothing that needs it. Taking the dice away did not help either (the last
line of the table) — presumably because a brain without random inputs can
read randomness off a world that is noisy enough.

## Room to go on

If competition for nodes has an answer, evolution needs a game without one.
Games without a final answer are games whose payoffs depend on what the
others have become — and the richest kind are those in which agents can do
better *together*, so that whom to trust, whom to join and how to coordinate
become questions with answers that change as the population changes. Two
things the pilots tried were meant to be such games, and both lacked the one
thing cooperation needs, a **surplus**: support counted less than a home
stake, and the yield on returned stakes went to whoever had most partners, so
it built stars rather than partnerships. What a surplus could look like is
the subject of the next part of the work.

## The experiment

Experiment 16 tests the claim in the world built for it: B2 at 10,000 tokens,
ten seeds, for 2,000 iterations, with a whole world archived every 100, and
the same without random inputs. Its tournaments play the eras 0, 100, 200,
500, 1,000, 1,500 and 2,000.

## The thesis

<!-- thesis E16 -->
> [!quote] The thesis of Experiment 16, written down before any of its runs existed
> **The claim.** A world that cannot break still stops improving. In B2 the brains of iteration 2,000 hold between 45% and 55% of a contested world against those of iteration 500, in the median world of ten; and taking the random inputs away does not change that.
>
> **Why it would be so.** In every pilot the cross-time tournament of Chapter 49 flattened within a few hundred iterations: the baseline at +320 to +345 from iteration 200, a B2-like world at +160 to +170 from 200 to 1,000, one with 30% kept at home at +162 to +180 from 50 to 800. Redrawing an evolved agent's random inputs changed 25–58% of its stakes, its neighbour's state 4–23%, its own 2–7%: what evolution found is a way to be unpredictable, which is the best play in a Colonel Blotto contest and cannot be bettered. Without random inputs a pilot flattened all the same (+141 to +188 from 50 to 1,000).
>
> **It holds if:** In the median world of the ten B2 worlds, and again of the ten without random inputs, the score of iteration 2,000 against iteration 500 — the mean of its two contests — lies between 0.45 and 0.55.
>
> **It fails if:** Above 0.55 in either condition's median world: evolution in that world does not stop by iteration 500.
<!-- /thesis -->

("Chapter 49" in the thesis is today's [Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md): the chapters were renumbered on 2026-10-14.)

## The runs

<!-- runs E16 -->
> [!info] The runs behind this chapter
> **20 runs.** Every condition below is run once for every seed (1 to 10), for 2,000 iterations — or until its world dies out. Every setting is that of the baseline **B2** (a world that cannot break) unless the condition changes it.
>
> - **B2** — changes nothing: B2 as it is; 10,000 tokens. Runs `B2-10000-s001` … `B2-10000-s010`.
> - **no random inputs** — changes `random_input_amount` = 0; 10,000 tokens. Runs `B2-10000-bad270-s001` … `B2-10000-bad270-s010`.
>
> To make these runs again: `python3 gol_lab.py run E16`, or ▶ in the Book tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | B2 | no random inputs | what it is |
> |---|---|---|---|
> | [`total_tokens`](../notes/settings.md#total_tokens) | 10,000 | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](../notes/settings.md#n_nodes) | 0 | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](../notes/settings.md#k_neighbors) | 0 | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](../notes/settings.md#rewire_p) | 0.2 | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](../notes/settings.md#hidden_layers) | 16 | 16 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](../notes/settings.md#brain_kind) | float16 | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](../notes/settings.md#brain_bits) | 16 | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](../notes/settings.md#message_amount) | 8 | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](../notes/settings.md#random_input_amount) | 5 | **0** | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](../notes/settings.md#exchange_messages) | on | on | Whether agents send and read messages at all. |
> | [`message_prepass`](../notes/settings.md#message_prepass) | off | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](../notes/settings.md#allow_handover) | off | off | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](../notes/settings.md#allow_revolutions) | on | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](../notes/settings.md#allow_gifting) | off | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](../notes/settings.md#random_decisions) | off | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](../notes/settings.md#prune_after) | blotto | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](../notes/settings.md#inactive_window) | phase | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](../notes/settings.md#redistribution) | uniform | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds. |
> | [`tokens_created_per_phase`](../notes/settings.md#tokens_created_per_phase) | 0 | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](../notes/settings.md#mutate_on_replication) | on | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](../notes/settings.md#incumbency) | 1.5 | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](../notes/settings.md#birth_links) | 2 | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](../notes/settings.md#mend_on_death) | on | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](../notes/settings.md#prune_needs) | 2 | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](../notes/settings.md#valence) | 8 | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](../notes/settings.md#keep_home) | 0.3 | 0.3 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](../notes/settings.md#max_age) | 100 | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`min_child`](../notes/settings.md#min_child) | 10 | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`mutation_probability`](../notes/settings.md#mutation_probability) | 0.2 | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](../notes/settings.md#mutation_noise_std) | 0.2 | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](../notes/settings.md#mutation_sparsity) | 0.1 | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](../notes/settings.md#extinction_threshold) | 20 | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 10 | 1 to 10 | one run per seed |
> | iterations | 2,000 | 2,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B2.
<!-- /runs -->

> [!warning] Waiting for the runs
> The plan, the thesis and the method above were written before any run of
> Experiment 16 existed; the results follow when they are done.

<!-- turns -->
---

← [Chapter 49 · Who dies, and why](49-who-dies-and-why.md) · [Contents](../README.md) · [Chapter 51 · How fast should brains change?](51-how-fast-should-brains-change.md) →
<!-- /turns -->
