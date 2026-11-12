# Rounds 1 and 2 of the screen, and the criteria they were judged by

> [!question] Questions of this page
> - What did round 1 of the screen claim, and why was round 2 never played?
> - By which criteria were the screen's worlds judged before 9 November 2026?
> - When was a value of a setting marked bad, and drawn no more?

## In short

Rounds 1 and 2 of the [screen of random worlds](a-screen-of-random-worlds.md)
drew their worlds from an older list of settings. Round 1 (Experiment 38)
found no healthy world among fifty, and round 2 (Experiment 39) was withdrawn
unplayed. Both were to be judged by six criteria read at [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 2,000. Round
3 was first judged by six criteria averaged over each world's run, with looser
lines, until the author defined health anew on 9 November 2026. This page keeps
both sets of criteria, and the rule that marked a value bad, as they were.

## Round 1

<!-- experiment E38 -->

Round 1 is Experiment 38: fifty worlds of [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")'s rules, changed as the first
sampler drew them, with two worlds of [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") beside them. What it found is told in
[Round 1](a-screen-of-random-worlds.md#round-1). Its tables, its rule and its
worlds one by one are in the notebook's history: `git show
9ec755d:book/pages/a-screen-of-random-worlds.md`.

<!-- thesis E38 -->
> [!quote] The thesis of Experiment 38, written down before any of its runs existed
> **The claim.** Among fifty worlds whose settings are drawn at random — every on/off rule on in half of them, every number at its default in three of four, messages and noise absent in one of ten — at least one is healthy at iteration 2,000: alive; decentralised and resilient by every mark of the health bar but the dimension; a true space, of 2 to 4 dimensions and at least 1.5 times as wide as its random twin; as diverse as B2's median world in its agents and in its decisions; and no more even than B2's least even world.
>
> **Why it would be so.** Every healthy-looking world so far was found by changing B2 one rule at a time, and each paid for one line with another: the estates make spaces of paupers, mixed brains rich worlds that fold, no mending worlds that can be cut (Chapters [100](building-b2-and-the-search-for-a-healthy-rich-world.md#room-and-riches-for-the-steadiest-world) and [101](building-b2-and-the-search-for-a-healthy-rich-world.md#births-as-rich-as-the-neighbours)). Settings drawn together reach combinations no single change would, and the round is judged by every line at once. B2's own rules come together too rarely for a random draw to meet them, so two worlds of B2 at 20,000 tokens are played beside the fifty for reference. A healthy world is rare: of 45 worlds of B2's kind read at 2,000, ten met the health bar's marks at a dimension of 2 to 4, before width, diversity and evenness were asked for. The thesis asks for one in fifty.
>
> **It holds if:** At least one of the fifty drawn worlds meets all five lines at 2,000 (research/screen/judge.py, with B2's lines from Experiment 16's ten worlds).
>
> **It fails if:** None does: the chapter says which lines the worlds missed, and which values the next round forces off because every world they appeared in was unhealthy, beyond what chance would explain.
>
> *Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*
<!-- /thesis -->

## Round 2

<!-- experiment E39 -->

Round 2 is Experiment 39, round 1's plan again with a new round [seed](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world."). It was
withdrawn unplayed on 24 October 2026
([Round 2](a-screen-of-random-worlds.md#round-2)).

<!-- thesis E39 -->
> [!quote] The thesis of Experiment 39, written down before any of its runs existed
> **The claim.** Fifty more worlds drawn as round 1's were, with a new round seed, meet the same two walls: at iteration 2,000 no drawn world is at once resilient, a true space, as diverse as B2's median world in its agents and its decisions, and no more even than B2's least even world.
>
> **Why it would be so.** In round 1 (Experiment 38), of the 22 drawn worlds that lived to 2,000, 15 were spaces that a single loss could cut, 2 were resilient but folded, and 4 were neither. The one world both resilient and a space (world 46, the estate paid to the newborns) had agents alike in their staking and tokens too even. One rule at a time showed the same: mending keeps a world whole and folds it, and estates that pay for births keep it a space and fill it with paupers. A round drawn the same way should meet the same walls. The author chose to draw round 2 as round 1 was drawn and to judge both rounds together, since round 1 had a single interesting world to set values aside by.
>
> **It holds if:** No drawn world of round 2 meets the four criteria resilient, almost three-dimensional, diverse and not too even together at 2,000 (research/screen/judge.py E38 E39, with B2's lines from Experiment 16's ten worlds).
>
> **It fails if:** At least one does: the page says which world, what else it misses if anything, and which of its settings set it apart.
<!-- /thesis -->

## What makes a world interesting

*This is how health was judged from 24 October to 7 November 2026, as round 3
was first judged. [The definition in
force](measuring-health.md#the-definition-in-force-health-of-9-november-2026) replaced it on 9
November.*

We are looking for worlds in which evolution could go on for a very long
time. Such a world has to stay alive. It must not hang on a few [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."), and it
must not fall apart when one agent dies. It should look like a space of about
three [dimensions](measuring-space.md#three-ways-to-measure-a-networks-dimension "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space."), as our own world does. And its agents should differ from
one another, in [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), links, age, genes and decisions, without the tokens
being spread too evenly.

We turn this picture into six criteria. From round 3 of the screen on, every
number behind them is an **average over a world's whole run**, from iteration
100 to its end. The first hundred iterations are ignored: every world starts
the same way, and a world shows what it is only once it has settled.
- A world that meets all six criteria is **healthy**.
- What makes a world **interesting** is decided with the author once round 3
  has run. Until then a world counts as interesting, provisionally, when it is
  alive and misses at most two of the other five.

Each criterion is made of one or more **lines**: a number, and which side of
it passes. The author controls every line, sees every world's value beside
it, and sees how many worlds would pass if it moved. Rounds 1 and 2 were
judged by stricter criteria, at a single moment; they are kept at the
[below](#the-criteria-of-rounds-1-and-2).

### The six criteria

<!-- table criteria/criteria -->
| criterion | what it asks |
|---|---|
| alive | played to the end of its run (3,000 iterations in round 3) and never down to 20 agents |
| decentralised | no agent linked to more than 5% of the others, and 3 to 12 links per agent |
| resilient | at least 80% of the agents in the 2-core, and no single agent's loss cuts off more than 15% of the world, nor a single link's more than 15% (the rules keep a world in one piece) |
| almost three-dimensional | a typical distance at least 1.3 times its random twin's, and a dimension of 2 to 5 |
| diverse | statistic by statistic: fewer than 80% of the agents hold the same number of tokens, and some agent holds more than 3; fewer than 80% of the 2-core's agents have the same number of links; fewer than 80% have the same age, and fewer than 80% the same genotype; and fewer than 95% make the same choice, to have a child or not |
| not too even | a Gini of the tokens of at least 0.46 (B2's most even world in Experiment 16) |
<!-- /table -->

Each criterion stands for one part of the picture.

1. **Alive.** A world that died tells us nothing about long evolution. A
   world dies when it falls to 20 agents. In round 3 a world must live through
   all 3,000 of its iterations.
2. **Decentralised.** If one agent holds links to many others, the world
   depends on it: when that [hub](measuring-networks.md#stars-and-hubs "An agent joined to far more agents than most; a star is a world held together by a single hub.") dies, everything changes at once. We ask that
   no agent links to more than 5% of the world, and that agents have 3 to 12
   links each, as a network that is neither a tree nor a tangle.
3. **Resilient.** A healthy world does not break apart when one agent or one
   link disappears. We ask that most agents (80%) sit on [loops](measuring-networks.md#loops "A path through the network that comes back to where it started without using a connection twice.") (the
   *[2-core](measuring-networks.md#core-trees-and-leaves "What is left of a network when agents with one connection or none are peeled away again and again: the agents on loops or between them.")*), and that no single loss cuts off more than 15% of the agents.
   The worlds are small, a few thousand agents, so the author allows a larger
   share than the [health bar](how-to-tell-a-world-that-keeps-evolving.md)'s 2%
   and 1%. A world is always in one piece: whatever is cut off from its
   largest piece dies at once.
4. **Almost three-dimensional.** The physical side of the project asks
   whether a space can grow out of the agents' links
   ([Physical inspiration](physical-inspiration.md)). We measure a world's
   dimension by how fast the number of agents within a given distance grows,
   and ask for 2 to 5. A tangled network can also read "three dimensions", so
   we ask a second thing: a typical distance between two agents at least 1.3
   times that in a *[random twin](measuring-space.md#the-random-twin-is-a-world-a-space "A network with the same agents and as many links each, wired at random: what a world would be if it were no space at all.")*, a network with the same number of links per
   agent wired at random. A folded world fails this; a true space passes
   ([What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world)).
5. **Diverse.** Evolution needs differences to choose from. Each statistic is
   judged on its own, not mixed into one score:
   - **tokens:** not diverse if 80% or more of the agents hold the same number
     of tokens, or if no agent holds more than 3;
   - **links:** counted in the 2-core only, since leaves do not make a network
     bad: not diverse if 80% or more of the core's agents have the same number
     of links;
   - **ages** and **[genotypes](measuring-evolution.md#genotypes-families-and-the-common-ancestor "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from."):** not diverse if 80% or more of the agents share
     one age, or carry one genotype;
   - **decisions:** not diverse if 95% or more of the agents make the same
     choice, to have a child or not. A world where every agent has a child in
     every iteration, or none does, is sorted out.
6. **Not too even.** If almost every agent holds the same few tokens, there
   is little to win and little to select. We ask for a
   [Gini coefficient](measuring-tokens.md#measuring-inequality-the-gini-coefficient-the-richest-tenth-and-entropy) of at least 0.46, the Gini of B2's
   most even world in Experiment 16. Round 3 has no B2 worlds of its own, so
   this line is a fixed number.

### The lines

Every line, with the criterion it belongs to, where it is drawn, and the
values the sensitivity table moves it to:

<!-- table criteria/lines -->
| what is measured | criterion | passes | the sensitivity table tries |
|---|---|---|---|
| links of the best-linked agent, as a share of the others | decentralised | at most 0.05 | 0.02, 0.05, 0.1, 0.2 |
| links per agent | decentralised | at least 3 | 2, 3, 4 |
| links per agent | decentralised | at most 12 | 8, 12, 20 |
| share of the agents in the 2-core | resilient | at least 0.8 | 0.6, 0.7, 0.8, 0.9 |
| most of the world one agent's loss cuts off | resilient | at most 0.15 | 0.02, 0.05, 0.1, 0.15, 0.2, 0.3 |
| most of the world one link's loss cuts off | resilient | at most 0.15 | 0.01, 0.05, 0.1, 0.15, 0.2, 0.3 |
| typical distance, over the random twin's | almost three-dimensional | at least 1.3 | 1, 1.1, 1.2, 1.3, 1.5, 2 |
| dimension, by how balls grow | almost three-dimensional | at least 2 | 1.5, 2, 2.3 |
| dimension, by how balls grow | almost three-dimensional | at most 5 | 4, 5, 6 |
| share of the agents holding the most common number of tokens | diverse | below 0.8 | 0.6, 0.7, 0.8, 0.9, 0.95 |
| the largest holding | diverse | above 3 | 2, 3, 5, 10 |
| share of the 2-core's agents with its most common number of links | diverse | below 0.8 | 0.6, 0.7, 0.8, 0.9, 0.95 |
| share of the agents of the most common age | diverse | below 0.8 | 0.6, 0.7, 0.8, 0.9, 0.95 |
| share of the agents carrying the most common genotype | diverse | below 0.8 | 0.6, 0.7, 0.8, 0.9, 0.95 |
| share of the agents making the more common choice, to have a child or not | diverse | below 0.95 | 0.7, 0.8, 0.9, 0.95, 0.99, 1 |
| Gini of the tokens | not too even | at least 0.46 | 0.3, 0.4, 0.46, 0.5, 0.6 |
<!-- /table -->

**Where the numbers come from.** A world records a row of statistics every
iteration, a [frame](runs-and-records.md#what-a-run-records "A whole world as a run records it after one phase: every agent, its tokens and connections.") of every phase, and the whole world every 250 iterations.
The judge reads each number where it is kept, as often as it is kept, and
averages it from iteration 100 to the end:
- every iteration: the number of agents, their links, the hub, the largest
  holding, the Gini, and the share of agents that had a child;
- every 25th iteration: the dimension, and from the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.")'s frame how alike
  the agents are, the 2-core, and the worst cuts;
- every world kept (every 250 iterations): the width against the random
  twin, and the two numbers of the next section.

**How alike.** For tokens, ages, genotypes and the core's links, "how alike"
is the share of agents that share the most common value: 1 when all are the
same. For the choice to have a child it is the share making the more common
choice, at each iteration: 0.5 when half the agents have a child, 1 when all
of them do or none does.

**The sensitivity table.** For each line, the judge moves that line alone
through the values in the last column. For each value it counts how many
living worlds pass the line, meet its criterion, are healthy and are
interesting. Where a count jumps between two values, the verdict depends on
exactly where we draw the line.

### Healthy, and interesting

A world that meets all six criteria is healthy.

Worlds made with the same settings but different seeds differ. A setting
that gives a healthy world in one seed may miss a line in another, and we do
not want to throw it away for that. So a world may miss some criteria and
still be **interesting**. How many, and which, will be decided with the
author once round 3 has run and we have looked at its worlds. Until then the
judge counts a world as interesting when it is alive and misses at most two of
criteria 2 to 6, the author's first answer.

Round 3 has run. Instead of a verdict, the author then asked (7 November
2026) for an order of every world drawn at random, from the most healthy to
the least, by a definition they can check
([Which worlds are healthy?](which-worlds-are-healthy.md)). Its definition of
9 November is now
[the definition in force](measuring-health.md#the-definition-in-force-health-of-9-november-2026),
and the lines of this section are an earlier version.

### When a setting is marked bad

In the [screen of random worlds](a-screen-of-random-worlds.md),
every world draws its settings at random, fifty or a hundred worlds at a time.
After each round we look at every value a setting took. A value is **marked
bad**, and drawn no more, only when both of these hold:
- it was tried in **at least ten worlds** where it could make a difference;
- **every one of them died**, or **none of them was interesting**.

Why ten? One or two dull worlds can be bad luck. Ten worlds that all failed
are a pattern. Beside each value marked bad, the judge also prints how likely
such a run of failures would be by chance alone, if the value made no
difference at all. That number is a warning, not part of the rule: when
interesting worlds are rare overall, ten failures in a row come easily. While
interesting is only provisional, so is every value the judge marks bad.

The judge is `research/screen/judge.py`, and it reads the numbers with
`research/screen/measures.py`. The criteria and their lines live in one list,
`research/screen/criteria.py`, which this page's tables are made from, so the
page and the judge always agree.

### The criteria of rounds 1 and 2

Rounds 1 and 2 were judged at one moment, iteration 2,000, from the world kept
then, by these criteria (`research/screen/criteria_1.py`):

<!-- table criteria/criteria-1 -->
| criterion | what it asked |
|---|---|
| alive | more than 20 agents at iteration 2,000, and not stopped as too slow |
| decentralised | no agent linked to more than 5% of the others, and 3 to 12 links per agent |
| resilient | one piece, with at least 80% of the agents in the 2-core; no single agent's loss cuts off more than 2% of the world, and no single link's more than 1% |
| almost three-dimensional | a dimension of 2 to 4, and a typical distance at least 1.5 times its random twin's: a true space, not a folded small world |
| diverse | tokens, links, ages and genotypes spread more widely than in B2's median world, and its agents' decisions vary more (effective kinds of staking) |
| not too even | a Gini of the tokens no lower than B2's most even world |
<!-- /table -->

A world was healthy if it met all six, and interesting if it was alive and
missed at most one. Too slow meant it did not reach iteration 2,000 within
four hours of computing. The kinds of staking swing from moment to moment, so
the judge took their mean over iterations 1,500 and 2,000.

#### The lines we compare with

The lines for criteria 5 and 6 came from B2's ten worlds of Experiment 16,
read in the same way and at the same iterations.

<!-- table criteria/references -->
| the line | at iteration 1,000 | at iteration 2,000 |
|---|---|---|
| diversity score of B2's median world | 0.53 | 0.51 |
| effective kinds of staking of B2's median world | 19.7 | 16.5 |
| Gini of the tokens of B2's most even world | 0.49 | 0.46 |
<!-- /table -->

## Where to go next

- [A screen of random worlds](a-screen-of-random-worlds.md): round 3, the
  round the notebook reads.
- [The definition in force](measuring-health.md#the-definition-in-force-health-of-9-november-2026):
  health as the author defined it on 9 November 2026, used only as a filter
  since 11 November.
- [Which worlds are healthy?](which-worlds-are-healthy.md): round 3's worlds
  scored, and put in three orders.

## How this was made

Round 1 is the plan of Experiment 38 (`book/experiments/E38.json`): sampler 1,
round seed 1, fifty worlds of B1's rules changed as drawn, at 20,000 tokens,
for 2,000 iterations or four hours, and two worlds of B2 beside them. World *i*
of a round with seed *s* has seed 1000·*s* + *i*, so round 1's worlds have
seeds 1001 to 1050, and the worlds of B2 1901 and 1902.

The judge plays nothing: it reads every world from its archives, the whole
world, [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") included ([`archive_every`](settings.md#archive_every)). `python3
research/screen/judge.py E38` judges round 1.

### The worlds of round 1

<!-- runs E38 -->
> [!info]- The worlds behind this page
> **50 worlds**, each with its settings drawn at random (sampler 1, round seed 1): B1's rules, changed as the draw says, at 20,000 tokens, played for 2,000 iterations — or until the world dies out, or until 4 hours of playing have gone by: a world that slow stops as too slow.
>
> - **How a setting is drawn:** an on/off switch is on in half the worlds, and a choice among several options takes each equally often; a number keeps its default in three worlds of four, and otherwise takes one of a few levels; the size of a message and the number of noise inputs are 0 in one world of ten (`gol_screen.py`).
> - **Only where it can matter:** buying life only with old age, the mate's kind only with mixing, a search's steering and consent only with a search, messages before the game and the pooled self-message only with messages.
> - **Beside them, for reference:** B2 at seed 1901 (`B2-20000-s1901`), B2 at seed 1902 (`B2-20000-s1902`).
>
> To make these worlds again: `python3 gol_lab.py run E38`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every world's settings
> | world | seed | run | what differs from B1 |
> |---|---|---|---|
> | 1 | 1001 | `B1-20000-8c796a-s1001` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 8, [`inactive_window`](settings.md#inactive_window) = iteration, [`incumbency`](settings.md#incumbency) = 1.25, [`message_amount`](settings.md#message_amount) = 8, [`message_prepass`](settings.md#message_prepass) = off, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.1, [`prune_after`](settings.md#prune_after) = reproduction, [`prune_needs`](settings.md#prune_needs) = 1, [`redistribution`](settings.md#redistribution) = newborns |
> | 2 | 1002 | `B1-20000-ad33ae-s1002` | [`allow_revolutions`](settings.md#allow_revolutions) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, 16, [`link_search`](settings.md#link_search) = walk, [`message_amount`](settings.md#message_amount) = 8, [`message_prepass`](settings.md#message_prepass) = off, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.4, [`prune_after`](settings.md#prune_after) = both, [`valence`](settings.md#valence) = 4 |
> | 3 | 1003 | `B1-20000-63be3b-s1003` | [`allow_handover`](settings.md#allow_handover) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`incumbency`](settings.md#incumbency) = 1.25, [`keep_home`](settings.md#keep_home) = 0.5, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = jump, [`link_steering`](settings.md#link_steering) = looking, [`message_amount`](settings.md#message_amount) = 4, [`message_prepass`](settings.md#message_prepass) = off, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.3, [`prune_after`](settings.md#prune_after) = both, [`prune_needs`](settings.md#prune_needs) = 1, [`self_message`](settings.md#self_message) = largest |
> | 4 | 1004 | `B1-20000-7ea3c8-s1004` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`link_search`](settings.md#link_search) = walk, [`link_steering`](settings.md#link_steering) = looking, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_amount`](settings.md#message_amount) = 8, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`prune_after`](settings.md#prune_after) = both, [`random_input_amount`](settings.md#random_input_amount) = 0, [`redistribution`](settings.md#redistribution) = by_tokens, [`self_message`](settings.md#self_message) = largest |
> | 5 | 1005 | `B1-20000-69be0c-s1005` | [`hidden_layers`](settings.md#hidden_layers) = 16, [`keep_home`](settings.md#keep_home) = 0.3, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = walk, [`max_age`](settings.md#max_age) = 50, [`message_amount`](settings.md#message_amount) = 0, [`message_prepass`](settings.md#message_prepass) = off, [`min_child`](settings.md#min_child) = 20, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_probability`](settings.md#mutation_probability) = 0.1, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.05, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = parents, [`valence`](settings.md#valence) = 16 |
> | 6 | 1006 | `B1-20000-d30991-s1006` | [`allow_handover`](settings.md#allow_handover) = off, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`hidden_layers`](settings.md#hidden_layers) = 8, [`inactive_window`](settings.md#inactive_window) = iteration, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = reach, [`link_steering`](settings.md#link_steering) = looking, [`message_prepass`](settings.md#message_prepass) = off, [`min_child`](settings.md#min_child) = 10, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_probability`](settings.md#mutation_probability) = 0.1, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.3, [`prune_needs`](settings.md#prune_needs) = 3, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = parents |
> | 7 | 1007 | `B1-20000-b54298-s1007` | [`allow_handover`](settings.md#allow_handover) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`keep_home`](settings.md#keep_home) = 0.1, [`link_search`](settings.md#link_search) = jump, [`link_steering`](settings.md#link_steering) = looking, [`message_amount`](settings.md#message_amount) = 8, [`min_child`](settings.md#min_child) = 10, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mutation_probability`](settings.md#mutation_probability) = 0.4, [`prune_after`](settings.md#prune_after) = both, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = by_tokens, [`self_message`](settings.md#self_message) = largest, [`valence`](settings.md#valence) = 4 |
> | 8 | 1008 | `B1-20000-1ba6bd-s1008` | [`allow_gifting`](settings.md#allow_gifting) = on, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`incumbency`](settings.md#incumbency) = 1.5, [`message_amount`](settings.md#message_amount) = 4, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mutation_probability`](settings.md#mutation_probability) = 0.4, [`prune_after`](settings.md#prune_after) = both, [`prune_needs`](settings.md#prune_needs) = 3, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = by_tokens |
> | 9 | 1009 | `B1-20000-fc88c6-s1009` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`keep_home`](settings.md#keep_home) = 0.3, [`message_amount`](settings.md#message_amount) = 4, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mix_mates`](settings.md#mix_mates) = parents, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`prune_after`](settings.md#prune_after) = both, [`random_input_amount`](settings.md#random_input_amount) = 0, [`redistribution`](settings.md#redistribution) = poorest, [`self_message`](settings.md#self_message) = largest, [`valence`](settings.md#valence) = 8 |
> | 10 | 1010 | `B1-20000-28be4f-s1010` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`incumbency`](settings.md#incumbency) = 1.5, [`message_amount`](settings.md#message_amount) = 4, [`message_prepass`](settings.md#message_prepass) = off, [`min_child`](settings.md#min_child) = 10, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`prune_after`](settings.md#prune_after) = both, [`random_input_amount`](settings.md#random_input_amount) = 0, [`redistribution`](settings.md#redistribution) = parents, [`valence`](settings.md#valence) = 4 |
> | 11 | 1011 | `B1-20000-f7d2f4-s1011` | [`allow_handover`](settings.md#allow_handover) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 32, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_amount`](settings.md#message_amount) = 0, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_probability`](settings.md#mutation_probability) = 0.1, [`prune_after`](settings.md#prune_after) = reproduction, [`prune_needs`](settings.md#prune_needs) = 3, [`redistribution`](settings.md#redistribution) = poorest |
> | 12 | 1012 | `B1-20000-3dcf03-s1012` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`buy_life`](settings.md#buy_life) = on, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`incumbency`](settings.md#incumbency) = 2.0, [`link_search`](settings.md#link_search) = jump, [`max_age`](settings.md#max_age) = 50, [`message_amount`](settings.md#message_amount) = 8, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_probability`](settings.md#mutation_probability) = 0.1, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.3, [`prune_after`](settings.md#prune_after) = reproduction, [`self_message`](settings.md#self_message) = largest |
> | 13 | 1013 | `B1-20000-5695c2-s1013` | [`allow_handover`](settings.md#allow_handover) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`keep_home`](settings.md#keep_home) = 0.3, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = walk, [`link_steering`](settings.md#link_steering) = looking, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.1, [`mutation_probability`](settings.md#mutation_probability) = 0.1, [`prune_after`](settings.md#prune_after) = reproduction, [`prune_needs`](settings.md#prune_needs) = 1, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = children, [`self_message`](settings.md#self_message) = largest |
> | 14 | 1014 | `B1-20000-3414ae-s1014` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = walk, [`message_amount`](settings.md#message_amount) = 4, [`message_prepass`](settings.md#message_prepass) = off, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.4, [`mutation_probability`](settings.md#mutation_probability) = 0.1, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = newborns, [`self_message`](settings.md#self_message) = largest, [`valence`](settings.md#valence) = 4 |
> | 15 | 1015 | `B1-20000-fdd440-s1015` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = walk, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`prune_after`](settings.md#prune_after) = reproduction, [`random_input_amount`](settings.md#random_input_amount) = 10 |
> | 16 | 1016 | `B1-20000-9bf2a7-s1016` | [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = reach, [`link_steering`](settings.md#link_steering) = looking, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_amount`](settings.md#message_amount) = 8, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mix_mates`](settings.md#mix_mates) = parents, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.1, [`prune_after`](settings.md#prune_after) = reproduction, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = by_tokens |
> | 17 | 1017 | `B1-20000-d97eea-s1017` | [`hidden_layers`](settings.md#hidden_layers) = 32, [`message_amount`](settings.md#message_amount) = 8, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_probability`](settings.md#mutation_probability) = 0.4, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.3, [`redistribution`](settings.md#redistribution) = newborns |
> | 18 | 1018 | `B1-20000-0f6e84-s1018` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.4, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = poorest |
> | 19 | 1019 | `B1-20000-4c2347-s1019` | [`allow_handover`](settings.md#allow_handover) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`incumbency`](settings.md#incumbency) = 1.5, [`link_search`](settings.md#link_search) = jump, [`link_steering`](settings.md#link_steering) = looking, [`max_age`](settings.md#max_age) = 200, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_amount`](settings.md#message_amount) = 0, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mix_mates`](settings.md#mix_mates) = parents, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_probability`](settings.md#mutation_probability) = 0.8, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.05, [`redistribution`](settings.md#redistribution) = by_tokens |
> | 20 | 1020 | `B1-20000-5e0717-s1020` | [`allow_handover`](settings.md#allow_handover) = off, [`birth_links`](settings.md#birth_links) = 2, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_amount`](settings.md#message_amount) = 16, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_probability`](settings.md#mutation_probability) = 0.1, [`prune_after`](settings.md#prune_after) = both, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = children |
> | 21 | 1021 | `B1-20000-1cdee2-s1021` | [`allow_handover`](settings.md#allow_handover) = off, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`birth_links`](settings.md#birth_links) = 4, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`link_search`](settings.md#link_search) = reach, [`link_steering`](settings.md#link_steering) = looking, [`mend_on_death`](settings.md#mend_on_death) = on, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mix_mates`](settings.md#mix_mates) = parents, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.4, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = children |
> | 22 | 1022 | `B1-20000-c049bc-s1022` | [`allow_revolutions`](settings.md#allow_revolutions) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`mend_on_death`](settings.md#mend_on_death) = on, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mix_mates`](settings.md#mix_mates) = parents, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.3, [`prune_after`](settings.md#prune_after) = both, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = children, [`self_message`](settings.md#self_message) = largest |
> | 23 | 1023 | `B1-20000-fae518-s1023` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`birth_links`](settings.md#birth_links) = 1, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`link_search`](settings.md#link_search) = walk, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mix_mates`](settings.md#mix_mates) = parents, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.3, [`prune_after`](settings.md#prune_after) = both, [`prune_needs`](settings.md#prune_needs) = 3, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = newborns |
> | 24 | 1024 | `B1-20000-9760a2-s1024` | [`allow_revolutions`](settings.md#allow_revolutions) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`max_age`](settings.md#max_age) = 200, [`message_amount`](settings.md#message_amount) = 8, [`message_prepass`](settings.md#message_prepass) = off, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.4, [`mutation_probability`](settings.md#mutation_probability) = 0.8, [`prune_after`](settings.md#prune_after) = reproduction, [`random_input_amount`](settings.md#random_input_amount) = 2 |
> | 25 | 1025 | `B1-20000-f570c2-s1025` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`keep_home`](settings.md#keep_home) = 0.3, [`message_amount`](settings.md#message_amount) = 16, [`message_prepass`](settings.md#message_prepass) = off, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.3, [`prune_after`](settings.md#prune_after) = reproduction, [`prune_needs`](settings.md#prune_needs) = 2, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = parents, [`self_message`](settings.md#self_message) = largest, [`valence`](settings.md#valence) = 4 |
> | 26 | 1026 | `B1-20000-34e831-s1026` | [`allow_handover`](settings.md#allow_handover) = off, [`birth_links`](settings.md#birth_links) = 2, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`max_age`](settings.md#max_age) = 200, [`message_prepass`](settings.md#message_prepass) = off, [`prune_after`](settings.md#prune_after) = both, [`redistribution`](settings.md#redistribution) = poorest |
> | 27 | 1027 | `B1-20000-b53ad5-s1027` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = walk, [`link_steering`](settings.md#link_steering) = looking, [`message_amount`](settings.md#message_amount) = 8, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.4, [`mutation_probability`](settings.md#mutation_probability) = 0.4, [`prune_needs`](settings.md#prune_needs) = 1, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = children, [`self_message`](settings.md#self_message) = largest |
> | 28 | 1028 | `B1-20000-b4f079-s1028` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, 16, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = jump, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_amount`](settings.md#message_amount) = 16, [`prune_needs`](settings.md#prune_needs) = 1, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = poorest, [`self_message`](settings.md#self_message) = largest, [`valence`](settings.md#valence) = 16 |
> | 29 | 1029 | `B1-20000-a6f392-s1029` | [`allow_handover`](settings.md#allow_handover) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = walk, [`message_amount`](settings.md#message_amount) = 0, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mutation_probability`](settings.md#mutation_probability) = 0.8, [`prune_needs`](settings.md#prune_needs) = 3, [`redistribution`](settings.md#redistribution) = by_tokens, [`valence`](settings.md#valence) = 16 |
> | 30 | 1030 | `B1-20000-f578dc-s1030` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`keep_home`](settings.md#keep_home) = 0.5, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = reach, [`message_amount`](settings.md#message_amount) = 16, [`message_prepass`](settings.md#message_prepass) = off, [`prune_after`](settings.md#prune_after) = both, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = newborns, [`self_message`](settings.md#self_message) = largest |
> | 31 | 1031 | `B1-20000-fca269-s1031` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 32, [`incumbency`](settings.md#incumbency) = 1.5, [`message_amount`](settings.md#message_amount) = 0, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mix_mates`](settings.md#mix_mates) = parents, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`prune_needs`](settings.md#prune_needs) = 3, [`random_input_amount`](settings.md#random_input_amount) = 2, [`valence`](settings.md#valence) = 4 |
> | 32 | 1032 | `B1-20000-6853f3-s1032` | [`allow_gifting`](settings.md#allow_gifting) = on, [`buy_life`](settings.md#buy_life) = on, [`hidden_layers`](settings.md#hidden_layers) = 16, 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`incumbency`](settings.md#incumbency) = 2.0, [`keep_home`](settings.md#keep_home) = 0.1, [`max_age`](settings.md#max_age) = 100, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`prune_after`](settings.md#prune_after) = reproduction, [`prune_needs`](settings.md#prune_needs) = 3, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = by_tokens |
> | 33 | 1033 | `B1-20000-518d0c-s1033` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`max_age`](settings.md#max_age) = 100, [`message_amount`](settings.md#message_amount) = 16, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mix_mates`](settings.md#mix_mates) = parents, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_probability`](settings.md#mutation_probability) = 0.1, [`prune_needs`](settings.md#prune_needs) = 3, [`random_input_amount`](settings.md#random_input_amount) = 2 |
> | 34 | 1034 | `B1-20000-26aeaf-s1034` | [`allow_revolutions`](settings.md#allow_revolutions) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`message_amount`](settings.md#message_amount) = 16, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.1, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.05, [`prune_after`](settings.md#prune_after) = reproduction, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = parents |
> | 35 | 1035 | `B1-20000-63a28e-s1035` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, 16, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = reach, [`link_steering`](settings.md#link_steering) = looking, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_amount`](settings.md#message_amount) = 16, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`prune_after`](settings.md#prune_after) = reproduction, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = parents, [`self_message`](settings.md#self_message) = largest |
> | 36 | 1036 | `B1-20000-7a52ea-s1036` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`buy_life`](settings.md#buy_life) = on, [`hidden_layers`](settings.md#hidden_layers) = 16, [`link_search`](settings.md#link_search) = reach, [`max_age`](settings.md#max_age) = 50, [`mend_on_death`](settings.md#mend_on_death) = on, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mix_mates`](settings.md#mix_mates) = parents, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.4, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.05, [`prune_after`](settings.md#prune_after) = reproduction, [`prune_needs`](settings.md#prune_needs) = 1, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = poorest, [`self_message`](settings.md#self_message) = largest |
> | 37 | 1037 | `B1-20000-92700f-s1037` | [`allow_gifting`](settings.md#allow_gifting) = on, [`buy_life`](settings.md#buy_life) = on, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = jump, [`max_age`](settings.md#max_age) = 50, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.1, [`prune_after`](settings.md#prune_after) = reproduction, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = parents, [`valence`](settings.md#valence) = 4 |
> | 38 | 1038 | `B1-20000-ab3fda-s1038` | [`allow_handover`](settings.md#allow_handover) = off, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`keep_home`](settings.md#keep_home) = 0.1, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = jump, [`link_steering`](settings.md#link_steering) = looking, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_amount`](settings.md#message_amount) = 4, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = poorest, [`self_message`](settings.md#self_message) = largest, [`valence`](settings.md#valence) = 8 |
> | 39 | 1039 | `B1-20000-5cc34b-s1039` | [`allow_handover`](settings.md#allow_handover) = off, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`keep_home`](settings.md#keep_home) = 0.5, [`link_search`](settings.md#link_search) = walk, [`message_prepass`](settings.md#message_prepass) = off, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`prune_after`](settings.md#prune_after) = reproduction, [`redistribution`](settings.md#redistribution) = newborns, [`self_message`](settings.md#self_message) = largest |
> | 40 | 1040 | `B1-20000-1e8af7-s1040` | [`allow_handover`](settings.md#allow_handover) = off, [`birth_links`](settings.md#birth_links) = 1, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`link_search`](settings.md#link_search) = reach, [`link_steering`](settings.md#link_steering) = looking, [`max_age`](settings.md#max_age) = 200, [`message_amount`](settings.md#message_amount) = 4, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mix_mates`](settings.md#mix_mates) = parents, [`prune_after`](settings.md#prune_after) = both, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = newborns, [`self_message`](settings.md#self_message) = largest |
> | 41 | 1041 | `B1-20000-e55747-s1041` | [`allow_handover`](settings.md#allow_handover) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`link_search`](settings.md#link_search) = reach, [`max_age`](settings.md#max_age) = 200, [`message_amount`](settings.md#message_amount) = 0, [`message_prepass`](settings.md#message_prepass) = off, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.3, [`prune_after`](settings.md#prune_after) = both, [`redistribution`](settings.md#redistribution) = children |
> | 42 | 1042 | `B1-20000-b1bdf8-s1042` | [`birth_links`](settings.md#birth_links) = 2, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_amount`](settings.md#message_amount) = 16, [`message_prepass`](settings.md#message_prepass) = off, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.1, [`mutation_probability`](settings.md#mutation_probability) = 0.4, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.05, [`prune_after`](settings.md#prune_after) = both, [`redistribution`](settings.md#redistribution) = newborns, [`self_message`](settings.md#self_message) = largest |
> | 43 | 1043 | `B1-20000-a9101a-s1043` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_handover`](settings.md#allow_handover) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`keep_home`](settings.md#keep_home) = 0.5, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_amount`](settings.md#message_amount) = 8, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mix_mates`](settings.md#mix_mates) = parents, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = poorest |
> | 44 | 1044 | `B1-20000-cd59a3-s1044` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`incumbency`](settings.md#incumbency) = 1.5, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = jump, [`link_steering`](settings.md#link_steering) = looking, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_amount`](settings.md#message_amount) = 8, [`message_prepass`](settings.md#message_prepass) = off, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.1, [`mutation_probability`](settings.md#mutation_probability) = 0.8, [`prune_after`](settings.md#prune_after) = both, [`redistribution`](settings.md#redistribution) = parents |
> | 45 | 1045 | `B1-20000-276640-s1045` | [`allow_gifting`](settings.md#allow_gifting) = on, [`hidden_layers`](settings.md#hidden_layers) = 16, [`link_search`](settings.md#link_search) = jump, [`message_prepass`](settings.md#message_prepass) = off, [`min_child`](settings.md#min_child) = 20, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = poorest, [`self_message`](settings.md#self_message) = largest |
> | 46 | 1046 | `B1-20000-a5e709-s1046` | [`allow_handover`](settings.md#allow_handover) = off, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`incumbency`](settings.md#incumbency) = 1.5, [`keep_home`](settings.md#keep_home) = 0.1, [`link_search`](settings.md#link_search) = walk, [`link_steering`](settings.md#link_steering) = looking, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_amount`](settings.md#message_amount) = 0, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mix_mates`](settings.md#mix_mates) = parents, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.4, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = newborns |
> | 47 | 1047 | `B1-20000-9c15e0-s1047` | [`allow_handover`](settings.md#allow_handover) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = walk, [`link_steering`](settings.md#link_steering) = looking, [`min_child`](settings.md#min_child) = 20, [`mutation_noise_std`](settings.md#mutation_noise_std) = 0.1, [`mutation_probability`](settings.md#mutation_probability) = 0.1, [`redistribution`](settings.md#redistribution) = poorest, [`self_message`](settings.md#self_message) = largest, [`valence`](settings.md#valence) = 16 |
> | 48 | 1048 | `B1-20000-efdfec-s1048` | [`allow_gifting`](settings.md#allow_gifting) = on, [`allow_revolutions`](settings.md#allow_revolutions) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`keep_home`](settings.md#keep_home) = 0.5, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = jump, [`mend_on_death`](settings.md#mend_on_death) = on, [`message_amount`](settings.md#message_amount) = 4, [`message_prepass`](settings.md#message_prepass) = off, [`mix_at_birth`](settings.md#mix_at_birth) = on, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`prune_after`](settings.md#prune_after) = both, [`random_input_amount`](settings.md#random_input_amount) = 2, [`redistribution`](settings.md#redistribution) = newborns, [`self_message`](settings.md#self_message) = largest, [`valence`](settings.md#valence) = 16 |
> | 49 | 1049 | `B1-20000-d3809d-s1049` | [`allow_gifting`](settings.md#allow_gifting) = on, [`buy_life`](settings.md#buy_life) = on, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`link_consent`](settings.md#link_consent) = forced, [`link_search`](settings.md#link_search) = reach, [`max_age`](settings.md#max_age) = 100, [`message_amount`](settings.md#message_amount) = 8, [`message_prepass`](settings.md#message_prepass) = off, [`mutate_on_replication`](settings.md#mutate_on_replication) = on, [`prune_after`](settings.md#prune_after) = both, [`prune_needs`](settings.md#prune_needs) = 2, [`random_input_amount`](settings.md#random_input_amount) = 10, [`redistribution`](settings.md#redistribution) = poorest, [`self_message`](settings.md#self_message) = largest |
> | 50 | 1050 | `B1-20000-f777c0-s1050` | [`allow_handover`](settings.md#allow_handover) = off, [`child_floor`](settings.md#child_floor) = neighbours, [`hidden_layers`](settings.md#hidden_layers) = 16, [`inactive_window`](settings.md#inactive_window) = iteration, [`message_amount`](settings.md#message_amount) = 4, [`min_child`](settings.md#min_child) = 5, [`mutation_sparsity`](settings.md#mutation_sparsity) = 0.3, [`random_input_amount`](settings.md#random_input_amount) = 0, [`redistribution`](settings.md#redistribution) = children, [`self_message`](settings.md#self_message) = largest |
<!-- /runs -->

### The worlds of round 2

<!-- runs E39 -->
> [!info] The worlds behind this page
> None: withdrawn unplayed on 24 October 2026, at the author's word: the author went through every setting and criterion of the screen again (research/screen/walkthrough.md), and the next round, Experiment 40, is drawn from the settings chosen then.
<!-- /runs -->

## How our view changed

- **23 October 2026.** From 22 October, this page set a value aside by a
  statistical rule. A value was set aside when every world it appeared in was
  unhealthy, and chance could not explain it. With about eighty values judged
  after a round of fifty, some harmless value would be unhealthy in all its
  worlds by luck. So the judge asked how likely that would be, if the round's
  healthy worlds had fallen among its worlds at random. A value was set aside
  only if that chance was below one in twenty divided by the number of values
  judged. On made-up rounds in which no value mattered, chance alone set a
  value aside in fewer than one round in a hundred. While healthy worlds were
  rare, the same was asked of each of the five lines. So a value whose worlds
  all died, or all grew hubs, or all folded, was set aside too, if other
  worlds did not. Before round 1 was judged, the author replaced this rule with
  the simpler one above. The same day the line *decentralised and resilient*
  was split in two, so the five lines became six criteria
  ([What makes a world interesting](rounds-1-and-2-of-the-screen.md#what-makes-a-world-interesting)). The
  judge still prints the chance, as a warning. And round 1's draws in fact
  hold all 92 values of the sampler, not about eighty.
- **23 October 2026.** From 22 October, this page said that a diverse world's
  agents spread at least as widely as those of B2's [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") world, and decide
  at least as variously. The judge has always asked for more than B2's median
  world, and the page has said so since 23 October.
- **23 October 2026.** From 22 October, this page said that every number's
  default is B1's. That holds for the rules' numbers and for [mutation](rules-in-detail.md#how-a-brain-changes "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype."), but not
  for the brain's size. The sampler's default brain is one hidden layer of 16
  neurons, and B1's five layers are never drawn (`gol_screen.py`,
  `book/experiments/B1.json`). So no drawn world has B1's brain, and every row
  of the table of settings lists `hidden_layers`.
- **24 October 2026.** This page called the line of evenness "B2's least
  even world". It is B2's most even world, the one with the lowest Gini
  ([The lines we compare with](#the-lines-we-compare-with)).
  The theses of Experiments 38 and 39 keep the old words, as theses do, and
  mean this world.
- **24 October 2026.** Round 2 (Experiment 39) was to be judged together with
  round 1 by the criteria above. That day the author went through every
  setting and criterion again (`research/screen/walkthrough.md`): each
  measure averaged over iterations 100 to 3,000, looser lines for cuts and
  for space, diversity judged statistic by statistic, and what is interesting
  decided after the round. Round 2 was withdrawn unplayed, and round 3
  (Experiment 40) draws its worlds from the author's settings instead.
- **24 October 2026.** Since 19 October 2026, the notebook called the line of
  the sixth criterion "B2's least even world". The line the judge compares
  with is the lowest Gini among B2's worlds: 0.49 at iteration 1,000 and 0.46
  at 2,000 ([The lines we compare with](#the-lines-we-compare-with)). That is
  B2's **most** even world. The words are set right on this page and on every
  page that draws the line. The line itself, and every verdict judged by it,
  are unchanged. The theses of Experiments 31, 32, 38 and 39 keep the old
  words, as a thesis does, and mean this line.
- **24 October 2026.** The author found the criteria too harsh, and revised
  them for round 3 (`research/screen/walkthrough.md`): averages over
  iterations 100 to 3,000, cuts up to 15%, a width of 1.3 and a dimension of 2
  to 5, and diversity statistic by statistic instead of one score against
  B2's median world. Even B2's own ten worlds met the old criteria only twice
  in ten as interesting, and never as healthy. Round 1 keeps its verdict by
  the criteria it was judged by.
- **25 October 2026.** The judge for the revised criteria was built. This page
  now states them, with every line and its sensitivity table, and keeps the
  first criteria at its end. Two choices were made in building it. The choice
  between spreading a stake and staking it all on one place is reported, not
  judged: in the first world of round 3, 98.5% of the agents spread, which says
  nothing about how alike their stakes are. And the half cut is the cheapest of
  ten cuts, since the slowest way to vibrate alone varied by a factor of two
  between ways that vibrate at nearly the same rate.
- **7 November 2026.** Round 3 has run, and the author asked for an order of
  the worlds rather than a verdict. The proposal
  ([Which worlds are healthy?](which-worlds-are-healthy.md)) keeps these lines
  but reads them as shares, draws lines for the half-cut exponent and the
  same-position share, and would replace B2's Gini line by one drawn by chance.
  This page's lines stand until the author decides.
- **12 November 2026.** Gathered here: rounds 1 and 2, with their theses,
  runs and entries of this list, from
  [A screen of random worlds](a-screen-of-random-worlds.md); and the criteria
  of October 2026, with their entries, from
  [Measuring health](measuring-health.md).

<!-- turns -->
---

← [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) · [Building B2, the healthy and rich search, and the pilots](../README.md#building-b2-the-healthy-and-rich-search-and-the-pilots) · [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md) →

**Also about:** [Round 3 of the random screen](../README.md#round-3-of-the-random-screen).

> [!info]- Pages that link here (8)
> [A screen of random worlds](a-screen-of-random-worlds.md) · [Which worlds are healthy? Three orders](which-worlds-are-healthy.md) · [Glossary](glossary.md) · [Measuring a network: connections, hubs, distances and cuts](measuring-networks.md) · [Measuring health: criteria, the health bar and diversity](measuring-health.md) · [B1: brains, lineages and messages](b1-brains-lineages-and-messages.md) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) · [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md)
<!-- /turns -->
