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
and then stops ([Chapter 46](how-to-tell-a-world-that-keeps-evolving.md)).
Its brains are nearly blind, so that was no surprise: evolution could only set
a few constants. Chapters [47](brains-that-see-and-worlds-that-break.md) to 49 then built a world in which brains can see
without the world breaking — B2. The question now is the one Part V exists
for: in such a world, does evolution keep going?

## What the pilots tried

Every pilot below played its eras against each other as in [Chapter 46](how-to-tell-a-world-that-keeps-evolving.md) — the
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
([Chapter 46](how-to-tell-a-world-that-keeps-evolving.md#numbers-that-rise-and-mean-nothing)).
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
([Chapter 16](how-the-game-is-played.md)): it divides its tokens between
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

> [!info] Added later: the dice were inherited, and the game is not solved
> Two later tests do not support this reading, though the experiment's thesis,
> that evolution stops, held. Random founder brains already change their stakes
> as often when their random inputs are redrawn, and in a lottery contest,
> where dice are worthless, evolved brains use them as much. A strategy written by
> hand took half of one of these worlds. See
> [What the probes say about why](#what-the-probes-say-about-why) below and
> [Chapter 51](strategies-written-by-hand.md).

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
the subject of [Chapter 56](alliances.md), and other contests are that of
[Chapter 55](another-game.md).

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

("[Chapter 49](who-dies-and-why.md)" in the thesis is today's [Chapter 46](how-to-tell-a-world-that-keeps-evolving.md): the chapters were renumbered on 2026-10-14.)

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
> | [`total_tokens`](settings.md#total_tokens) | 10,000 | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | 16 | 16 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 8 | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | **0** | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | off | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | off | off | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](settings.md#allow_gifting) | off | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](settings.md#random_decisions) | off | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | on | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.5 | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 2 | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | on | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 2 | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 8 | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](settings.md#keep_home) | 0.3 | 0.3 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 100 | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 10 | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | off | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`self_message`](settings.md#self_message) | own | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | off | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | mutual | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | none | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | 20 | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 10 | 1 to 10 | one run per seed |
> | iterations | 2,000 | 2,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B2.
<!-- /runs -->

## What the twenty worlds did

All twenty worlds lived their 2,000 iterations. Every one improved at first,
and none went on improving.

<!-- figure stopping/elo -->
![How much better than the founders, over 2,000 iterations](../figures/stopping/elo.svg)

**How much better than the founders, over 2,000 iterations.** Thin lines are single worlds, thick ones the median world of each condition.

> [!example]- How to make this figure
> **Runs.** The 20 runs of Experiment 16 — B2 at 10,000 tokens, seeds 1 to 10, 2,000 iterations, a whole world archived every 100 — with and without the random inputs. They are made with `python3 gol_lab.py run E16`.
>
> **Data.** From each run's archive, `GraphOfLifeRuns/<run>/archive/world_<iteration>.npz`: the whole world as a checkpoint writes it, kept every 100 iterations.
>
> 1. For every pair of the eras 0, 100, 200, 500, 1000, 1500, 2000 play two contests, one in each era's world (`gol_tournament.contest`), and fit one rating per era (`gol_tournament.ratings`).
>
> **To make it again:** `python3 book_figures.py stopping`.
<!-- /figure -->

<!-- figure stopping/steps -->
![Each era against the one before](../figures/stopping/steps.svg)

**Each era against the one before.** One dot per world: the later era's share of the agents, the mean of its two contests with the era before, one in each era's world. The bars are the medians. The contests of the yardstick last 20 iterations; the third colour is the same steps of B2 in contests of 100 iterations, an agent's life.

> [!example]- How to make this figure
> **Runs.** The 20 runs of Experiment 16 — B2 at 10,000 tokens, seeds 1 to 10, 2,000 iterations, a whole world archived every 100 — with and without the random inputs. They are made with `python3 gol_lab.py run E16`.
>
> **Data.** From each run's archive, `GraphOfLifeRuns/<run>/archive/world_<iteration>.npz`: the whole world as a checkpoint writes it, kept every 100 iterations.
>
> 1. Of the contests of the figure above, take those between consecutive eras and average each pair's two.
> 2. For the contests a lifetime long: the same pairs of eras of the ten B2 worlds, played with T = 100, once in each era's world, seed 17 (`python3 research/oee/checks/long_contests.py --T 100 --pairs 0:100,100:200,200:500,500:1000,1000:1500,1500:2000,1000:2000 --out GraphOfLifeRuns/.pilots/long_steps.json`).
>
> **To make it again:** `python3 book_figures.py stopping`.
<!-- /figure -->

**The first hundred iterations do almost everything.** Against the founders,
the brains of iteration 100 held 0.68 of a contested world in the median B2
world, and 0.69 without random inputs. A little more came between 100 and 200:
the step score there was 0.56 in both conditions. From 200 to 500 it was
0.52, and after that a tie: 0.49, 0.50 and 0.50 for the steps from 500 to
2,000 in B2, and 0.50, 0.49 and 0.49 without random inputs. In ratings, the
median B2 world stood at +90 at iteration 100, +124 at 200, +126 at 500, +128
at 1,000 and +120 at 1,500 and 2,000. Without random inputs it stood at +86,
+121, +130, +162, +139 and +123.

> [!info] Added later: the third colour
> The green dots were added after the rest of this chapter was written: the
> same steps in contests a lifetime long, 100 iterations. None of them is a
> tie. See [Would we notice if it went on?](#would-we-notice-if-it-went-on)
> below.

**Later is a little worse.** The brains of iteration 2,000 held 0.482 of a
world against those of iteration 500 in the median B2 world (0.471 to 0.502),
and less than half in nine of the ten worlds. Without random inputs the
median was 0.477 (0.452 to 0.519), less than half in seven. The loss is small,
about two points in a hundred, but it is nearly always a loss. After its first
few hundred iterations a world does not only stop climbing. It slips back a
little, as a world would whose brains are worn by mutation faster than
selection can repair them, once there is little left to select for.

**The dominance tournament agrees.** It crowns the founders, the brains of
iteration 100, and one or two later eras in every world. The last era crowned
was iteration 200 in four of the ten B2 worlds, 500 in three, 1,000 in two and
2,000 in one. Without random inputs it was 100, 200, 500 or 1,000. Contests
in worlds of 10,000 tokens are less noisy than those of
[Chapter 46](how-to-tell-a-world-that-keeps-evolving.md). One contest's
share varies around its pair's mean by about 0.05, against 0.12 there. A
tournament of these seven eras would crown 2.3 champions on average if every
era after the founders were equally good, and the worlds crowned 3.2. The
difference fits the real step from iteration 100 to 200; it needs no later one.

## The verdict

The thesis holds. In the median world the brains of iteration 2,000 held
0.482 of a world contested with those of iteration 500, and 0.477 without
random inputs. Both lie inside 0.45 to 0.55. A world that cannot break still
stops improving, within its first few hundred iterations, and taking the
random inputs away does not change that.


> [!warning] Added later: the verdict stands, the reading does not
> The thesis was judged as it was written, with contests of 20 iterations,
> and it holds. But a contest of 20 iterations cannot see what pays off over
> an agent's life of 100. In contests of 100 iterations the brains of
> iteration 2,000 beat those of 500 in all ten worlds (0.73), and those of 200
> (0.82). Evolution did not stop at iteration 200 to 500: it went on improving
> what a short contest cannot show. See
> [Would we notice if it went on?](#would-we-notice-if-it-went-on) below.

## Would we notice if it went on?

A tie with the past is what a world that has stopped looks like. Before
believing that these worlds have stopped, it is fair to ask what the yardstick
would miss. The author of the simulation asked exactly that, and there are at
least four ways it could.

**Improvement too slow to see.** One contest's share varies by about 0.046
around the mean of its pair, so a single world's two contests pin a score to
within about 0.03. But the ten worlds agree with each other closely: the
median score of iteration 2,000 against 500 is 0.482, with a standard error of
0.005, and even the top of its 95% range, 0.492, lies below a tie. So between
iterations 500 and 2,000 these worlds did not improve, not even slowly. They
lost a little. What happens after iteration 2,000 this experiment cannot say.

**Change that goes round in circles.** In a *Red Queen* race every era would
be adapted to its own world. It would win at home and lose away, while no era
was better than another overall. The contests are played once in each era's
world, so this would show as a home advantage of the later era, but there is
none. For pairs from iteration 200 on, the later era's share at home minus
its share away averaged −0.006, positive in 46 of 100 pairs.

**Traits a short contest cannot show.** A contest lasts 20 iterations and an
agent lives 100. What a brain does for its children over a whole life barely
shows in 20 iterations, so the late eras were played again in contests five
times as long, 100 iterations, once in each era's world.

The answer is unambiguous. In contests of 100 iterations the brains of
iteration 2,000 held 0.73 of the world against those of iteration 500 in the
median world, and more than half in all ten worlds. They held 0.70 in
iteration 500's world and 0.77 in their own. Against the brains of iteration
200 they held 0.82, and those of 200 held 0.75 against those of 100. Two
controls make sure the contest is fair. The brains of iteration 2,000 played
against themselves held 0.48 (0.44 to 0.54), a tie. With the labels swapped,
the brains of 500 counted as the later era, those of 500 held 0.26 against
those of 2,000.

So the twenty worlds did not stop evolving at iteration 200 or 500. In a
contest of 20 iterations the later brains start a little behind (0.48); over
a life they overtake. What evolution kept improving pays off slowly, and the
yardstick of [Chapter 46](how-to-tell-a-world-that-keeps-evolving.md) was
too short to see it.

Played step by step, a lifetime long, every era beats the one before it in all
ten worlds: the third colour of the figure *Each era against the one before*
above. Iteration 100 holds 0.79 of the world against the founders, 200 holds
0.75 against 100, 500 holds 0.79 against 200, and from there each 500
iterations win a little less: 0.68 for 1,000 against 500, 0.62 for 1,500
against 1,000, and 0.60 for 2,000 against 1,500, the lowest world still at
0.52. Where the contests of 20 iterations see a tie from iteration 500 on, the
contests of 100 see evolution going on all the way, more slowly as it goes.

How far it goes was measured by going on: three of the ten worlds were
continued from iteration 2,000 to 10,000 under the same rules, exactly as the
lab would have run them (`research/oee/pilots/run.py --from`), and their eras
played against each other the same way. All three kept improving to iteration
5,000: its brains held 0.69, 0.59 and 0.63 of the world against those of
2,000. After that the gains all but stopped. The brains of 10,000 held 0.51,
0.57 and 0.55 against those of 5,000, inside or at the edge of the 0.44 to
0.58 that eras reach against themselves, and against those of 2,000 they held
0.61, 0.62 and 0.65, no more than the brains of 5,000 had. Each 500 iterations
won a step of 0.6 to 0.7 up to iteration 2,000; the 5,000 iterations from 5,000
to 10,000 together won barely one. Evolution in B2 slows towards a plateau. The
same three worlds are being continued to iteration 50,000 to see whether the
plateau is where it ends.

**Distance to the best play.** A tie between eras says nothing about how good
both are. [Exploitability](exploitability.md) measures it from
outside. For each agent it asks how much more it would hold after a game if
it alone played its best reply to everyone else.

<!-- figure stopping/exploit -->
![How far from an equilibrium](../figures/stopping/exploit.svg)

**How far from an equilibrium.** For a sample of 300 agents with at least three tokens: how many more nodes each would hold after one game with its best reply to everyone else's play than with its own, averaged. Zero would be an equilibrium. Thin lines are single worlds, thick ones the median world of each condition.

> [!example]- How to make this figure
> **Runs.** The 20 runs of Experiment 16 — B2 at 10,000 tokens, seeds 1 to 10, 2,000 iterations, a whole world archived every 100 — with and without the random inputs. They are made with `python3 gol_lab.py run E16`.
>
> **Data.** From each run's archive, `GraphOfLifeRuns/<run>/archive/world_<iteration>.npz`: the whole world as a checkpoint writes it, kept every 100 iterations.
>
> 1. Load the world of each era, play one game so that its messages are fresh, and draw the random inputs eight times; for each draw, turn every agent's brain outputs into stakes as the engine does (`research/oee/checks/exploitability.py`).
> 2. Score a way of staking by the nodes an agent holds after the game (its own, and every neighbour's it takes), everyone else as drawn and each node decided by the engine's rule, averaged over the eight draws.
> 3. An agent's best reply is the best of: everything at home; for each draw, the cheapest stakes that keep its node and then take the cheapest neighbours (one token above the largest rival offer), or that only take them. Its exploitability is the best reply's score minus its own; average over the agents (see the note *Exploitability*).
>
> **To make it again:** `python3 book_figures.py stopping`.
<!-- /figure -->

The founders could have gained 0.87 nodes a game by their best replies, in
the median B2 world. The brains of iteration 200 could have gained 1.49, and
those of 1,000 to 2,000 about 1.2. At iteration 2,000, 87% of agents could
have held at least half a node more. Evolution made the brains better at
holding what they have: 1.23 nodes a game at iteration 100 against the
founders' 1.00. But it did not move them towards an equilibrium. It moved
them away from one, and from iteration 1,000 on they stayed where they were.
Without random inputs the picture is the same, with a peak of 1.75 at
iteration 200 and 1.33 at 2,000.

## What the probes say about why

> [!info] Added later: it slows, it does not stop
> This section was written while the contests of 20 iterations were the
> yardstick. A lifetime long, every step to iteration 2,000 is won by the later
> era ([above](#would-we-notice-if-it-went-on)). Read "stops" below as "stops
> showing in a short contest": what the probes say about the dice and the
> senses holds for the brains as they are, but evolution goes on improving
> them, more slowly as it goes.

The thesis said *that* evolution stops, and it does. The reason this chapter
gave, that evolution finds the best play and the best play is to roll dice,
did not survive the tests that came after it.

**The dice were there from the start.** Redraw the random inputs of a
*founder*, a brain no selection has touched, and its division of stakes
changes in 49 to 66% of agents. Redrawing them in evolved brains changes 38
to 59%. Evolution did not find the dice; random brains already roll them. What
evolution does is turn the senses down. In all but one of the worlds probed,
sensitivity to an agent's own state falls from 14–22% to 1–9%, to the
neighbour's state from 13–20% to 5–10%, and to messages from 34–48% to
11–20%. The exception is a world that pooled the stakes of kin
(`research/oee/pilots/sense.py`; `LOG.md`, *The dice were there from the
start*). And in a contest where dice are worthless, a lottery, evolved brains
roll them as much as under Blotto: 48 to 67%, against 59% in the same world
under Blotto, both without revolutions ([Chapter 55](another-game.md)). The
brains keep the founders' dice because nothing selects against them, not
because the game rewards them.

**The game is not solved.** A strategy written by hand, which stakes what it
can on the poorest neighbour it is sure to take and otherwise keeps
everything at home, took half of the B2 world of seed 2 from a patch of 5% in
300 iterations, and it or its cousin, the factory, took 40 to 62% of three of
the ten worlds ([Chapter 51](strategies-written-by-hand.md)). The residents
were the very brains of iteration 2,000 played here. So there is better play
than the play these worlds end with, and evolution does not reach it, or does
not keep it.

That changes the question. Evolution stops in B2 not because the game has run
out of answers, but because the search has. Whether the cause is how fast
brains change ([Chapter 52](how-fast-should-brains-change.md)), how large
they are (Chapter 53), what the game asks of them
([Chapters 54](the-switches-the-world-was-built-with.md) and
[55](another-game.md)) or what they could do together
([Chapter 56](alliances.md)) is what the rest of Part V asks.

<!-- turns -->
---

← [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md) · [Does evolution keep going?](../README.md#does-evolution-keep-going) · [How fast should brains change?](how-fast-should-brains-change.md) →
<!-- /turns -->
