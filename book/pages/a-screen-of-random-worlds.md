# A screen of random worlds

> [!question] Questions of this page
> - Is there a combination of the notebook's rules that makes a healthy world: a
>   resilient, decentralised network of about three [dimensions](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space."), whose [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")
>   differ in almost everything?
> - Which rules make a world unhealthy wherever they appear, so that they can
>   be set aside?

<!-- experiment E38 -->

## In short

Changing [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") one rule at a time has made worlds that are healthy in some ways,
but each bought one quality with the loss of another. So the author planned
a screen: rounds of fifty worlds with every setting drawn at random, each
judged at [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 2,000 by six criteria. A world is healthy when it meets
all six, and interesting when it is alive and misses at most one. A value is
set aside once it was tried in at least ten worlds where it could matter, and
all of them died or none was interesting. Round 1, Experiment 38, is being
played in the [lab](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world."); its results are still to come.

## Why draw worlds at random

Picture a machine with thirty knobs and switches. You want it to make one
particular thing. But each time you turn a single knob, one fault goes away
and another appears. You could go on turning one knob at a time. Or you could
set all thirty at random, many times over, and learn which positions never
make anything good.

That is where the notebook's search for a healthy world stood in October
2026. Every healthy-looking world so far had been found by changing one rule
of B2 at a time, and each paid for one quality with another:
- the [estates](estate.md "The tokens a dead agent leaves, and who gets them.") of the dead make spaces of paupers;
- mixed [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") make rich worlds that fold;
- worlds without mending can be cut
  ([Room for the steadiest world](room-for-the-steadiest-world.md#room-for-the-steadiest-world)
  and
  [Births as rich as the neighbours](room-for-the-steadiest-world.md#births-as-rich-as-the-neighbours)).

The settings decide what grows, and one at a time is slow. The screen draws
30 settings, which take 92 values between them. Those combine into some 25
million million different worlds: far too many to try one by one.

So this page follows the author's plan of 22 October 2026: a **screen**, as
biologists screen many random mutants for a trait instead of changing one
gene at a time. A round is fifty worlds whose settings are all drawn at
random, each judged for health. After each round, every value that came only
with dead or dull worlds is set aside. The next fifty are drawn from the
values left. Rounds go on until some combination of settings makes a healthy
world. Only then are strategies tested in it: hand-written ones, and the
growth of groups ([Strategies written by hand](strategies-written-by-hand.md)).

## How a world's settings are drawn

Each world is dealt its settings like a hand of cards. For every setting the
sampler draws afresh, by a rule that depends only on the kind of setting. The
sampler is `gol_screen.py`. It starts from [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")'s rules and changes what the
draw says.

- **Every on/off rule** is on in half the worlds: [revolutions](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.")
  ([`allow_revolutions`](settings.md#allow_revolutions)), gifts
  ([`allow_gifting`](settings.md#allow_gifting)), handover
  ([`allow_handover`](settings.md#allow_handover)), mending
  ([`mend_on_death`](settings.md#mend_on_death)), mixing at birth
  ([`mix_at_birth`](settings.md#mix_at_birth)), [messages](messages.md "Numbers an agent writes to itself and to each neighbour every phase, which their brains read.") before the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.")
  ([`message_prepass`](settings.md#message_prepass)), brains that change only
  when copied ([`mutate_on_replication`](settings.md#mutate_on_replication)),
  the pooled self-message ([`self_message`](settings.md#self_message)), and a
  child as rich as its [neighbours](the-world.md#the-network "An agent joined to another by a connection.") ([`child_floor`](settings.md#child_floor)).
- **A choice among several options** takes each equally often: who inherits
  the [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") of the dead ([`redistribution`](settings.md#redistribution), six
  options), after which phase unused links are cut
  ([`prune_after`](settings.md#prune_after), three), how long a link may go
  unused ([`inactive_window`](settings.md#inactive_window), two), and whether
  and how an agent searches for links to strangers
  ([`link_search`](settings.md#link_search), four).
- **Every number** keeps its default in three worlds of four, and otherwise
  takes one of a few levels, each equally often. The table below lists them.
- **The size of a message and the number of noise inputs** are 0 in one world
  of ten, and otherwise one of a few levels, each equally often: 4, 8, 16 or
  30 numbers in a message ([`message_amount`](settings.md#message_amount)),
  and 2, 5 or 10 noise inputs
  ([`random_input_amount`](settings.md#random_input_amount)). B1's 30 and 5
  are among them.
- **A setting that can matter only beside another** is drawn only there:
  buying life ([`buy_life`](settings.md#buy_life)) only with old age, the
  mate's kind ([`mix_mates`](settings.md#mix_mates)) only with mixing at
  birth, a search's steering ([`link_steering`](settings.md#link_steering))
  and consent ([`link_consent`](settings.md#link_consent)) only with a search,
  messages before the game and the pooled self-message only with messages.
  Elsewhere it keeps B1's value, with one exception: without messages there
  is nothing to say before the game, so messages before the game are off. So
  two draws never make one world under two names.

These are the numbers and their levels:

| what is drawn | setting | its default, in three worlds of four | its other levels |
|---|---|---|---|
| the holder's bonus | [`incumbency`](settings.md#incumbency) | 1: no bonus | 1.25, 1.5, 2 |
| the share kept at home | [`keep_home`](settings.md#keep_home) | 0 | 0.1, 0.3, 0.5 |
| old age, in iterations | [`max_age`](settings.md#max_age) | 0: never | 50, 100, 200 |
| the least child, in tokens | [`min_child`](settings.md#min_child) | 1: any child | 5, 10, 20 |
| the cap on links | [`valence`](settings.md#valence) | 0: no cap | 4, 8, 16 |
| links at birth | [`birth_links`](settings.md#birth_links) | 0 | 1, 2, 4 |
| when unused links lapse: the neighbours their two ends must share | [`prune_needs`](settings.md#prune_needs) | 0: every unused link is cut | 1, 2, 3 |
| the brain's size: its hidden layers | [`hidden_layers`](settings.md#hidden_layers) | one of 16 | one of 8, one of 32, two of 16 |
| [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype."): how often a brain changes | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | 0.1, 0.4, 0.8 |
| mutation: how large a change is | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | 0.1, 0.4 |
| mutation: the share of a brain a change touches | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | 0.05, 0.3 |

For the rules' numbers, the default is B1's, under which the rule does not
act at all. Mutation's defaults are B1's too. The brain's size is the
exception. Its default is one hidden layer of 16 neurons, B2's brain. B1's
five layers, of 50, 45, 40, 35 and 30 neurons, are never drawn, so no drawn
world has B1's brain.

Each world has 20,000 tokens and plays 2,000 iterations. That is long enough
for the first generation to die, and for the network to fold if it is going
to ([What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world)). A world that would take more
than four hours stops as **too slow**, since a [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") that slow could never
be played for long. B2's own rules almost never come together in a random
draw. So two worlds of B2, at 20,000 tokens, are played beside every round
for reference.

## When a world is healthy, and when it is interesting

What would a world look like in which evolution could go on for a very long
time? It stays alive. It does not hang on a few agents, and it does not fall
apart when one of them dies. It is a space of about three dimensions, like
ours, and not a tangle in which everyone is a few steps from everyone else.
And its agents differ, in what they hold and in what they do, without their
tokens being spread too evenly.

The judge, `research/screen/judge.py`, makes this picture exact in six
criteria, read at iteration 2,000:

1. **Alive:** more than 20 agents, and not stopped as too slow.
2. **Decentralised:** no agent linked to more than 5% of the others, and 3 to
   12 links per agent.
3. **Resilient:** one piece, with at least 80% of the agents in the [2-core](core-trees-leaves.md "What is left of a network when agents with one connection or none are peeled away again and again: the agents on loops or between them."). No
   single agent's loss cuts off more than 2% of the world, and no single
   link's loss more than 1%.
4. **Almost three-dimensional:** a dimension of 2 to 4, and a typical
   distance at least 1.5 times its [random twin](random-twin.md "A network with the same agents and as many links each, wired at random: what a world would be if it were no space at all.")'s. A folded small world reads
   three dimensions too, so both are asked.
5. **Diverse:** its agents' tokens, links, ages and [genotypes](genotype.md "The identity of a brain: a number that a brain keeps until it changes, and that records whose copy it was.") spread more
   widely than in B2's [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world, by the [diversity score](diversity-score.md "How much the agents of a world differ, in one number from 0 to 1: in tokens, links, ages and genotypes."). And its agents'
   decisions vary more. Put in the same situations, they fall into more
   effective kinds of staking than the agents of B2's median world
   ([How different are the agents?](how-different-are-the-agents.md)). The
   kinds are read from the world as kept, with no game played. They swing
   from one moment to the next, so they are averaged over iteration 1,500 and
   iteration 2,000.
6. **Not too even:** a [Gini](gini-coefficient.md "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all.") of its tokens no lower than B2's least even
   world.

Criteria 2 and 3 are the marks of the [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4."), all but its dimension
([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)).
Criterion 4 stands in for that mark, because the bar's dimension of 2.3 to 4
cannot tell a space from a small world. B2's lines, in criteria 5 and 6, are
those of Experiment 16's ten worlds of B2, at 10,000 tokens, read the same
way;
[What makes a world interesting](what-makes-a-world-interesting.md#the-lines-we-compare-with)
gives them.

A world that meets all six criteria is **healthy**. A world that is alive and
misses at most one of the other five is **interesting**. Worlds of the same
settings differ from [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") to seed, and one miss should not rule a setting
out ([What makes a world interesting](what-makes-a-world-interesting.md)).

## When a value is set aside

A value that makes worlds die, or makes them dull, wherever it appears is not
worth drawing again. But one or two dull worlds can be bad luck. Ten worlds
that all failed are a pattern.

So the author's rule, set on 23 October 2026 before round 1 was judged, is
this. A value is **marked bad**, and drawn no more, only when both of these
hold:
- it was tried in **at least ten worlds** where it could matter;
- **every one of them died**, or **none of them was interesting**.

A dead world is never interesting, so the first case is part of the second;
the judge names the two apart. "Where it could matter" means like with like:
a value is judged only among the worlds where it could have made a
difference. Buying life, for example, is judged only among the worlds with
old age. The worlds are counted over every round judged so far. Only drawn
worlds that have finished count, not B2's. A setting always keeps at least
one value, and the next round draws among the values left.

Ten worlds come at different speeds. A switch is on in about 25 worlds of
fifty, so both of its values can reach ten in one round. But a level of a
number other than its default comes in only about four to six worlds of
fifty, so it needs two or three rounds. In round 1's draws, 50 of the 92
values came in ten worlds or more where they could matter. Every level of a
number but its default came in one to nine.

Beside each value marked bad, the judge prints how likely such a run of
failures would be by chance alone, if the value made no difference. Say *n*
worlds were judged where the value could matter, and *g* of them were
interesting. The value came in *k* of them, and none of its *k* was
interesting. If the interesting worlds had fallen among the *n* at random,
the chance that none of them fell among the value's *k* would be

$$
P = \binom{n-g}{k} \Big/ \binom{n}{k} .
$$

The top counts the ways to pick *k* worlds from the *n* − *g* that were not
interesting; the bottom counts the ways to pick *k* from all *n*. For a value
whose worlds all died, the worlds that did not die take the place of the
interesting ones. With *n* = 50 and *k* = 10, *P* is about 0.08 if *g* = 10,
but about 0.64 if *g* = 2. When interesting worlds are rare, ten failures in a
row come easily. So the chance is a warning printed beside the mark, not a
part of the rule.

## What we expected

We expected at least one of the fifty worlds to be healthy. Settings drawn
together reach combinations that no single change of B2 would, and the round
judges every criterion at once. But a healthy world is rare. Of 45 worlds of
B2's kind read at iteration 2,000, ten met the health bar's marks at a
dimension of 2 to 4, and that was before width, diversity and [evenness](entropy-and-evenness.md "How evenly something is shared out among the agents, from all in one place to spread perfectly evenly.") were
asked for. So the thesis asks for only one healthy world in fifty. It was
written down on 22 October 2026, before any of the round's runs existed:

<!-- thesis E38 -->
> [!quote] The thesis of Experiment 38, written down before any of its runs existed
> **The claim.** Among fifty worlds whose settings are drawn at random — every on/off rule on in half of them, every number at its default in three of four, messages and noise absent in one of ten — at least one is healthy at iteration 2,000: alive; decentralised and resilient by every mark of the health bar but the dimension; a true space, of 2 to 4 dimensions and at least 1.5 times as wide as its random twin; as diverse as B2's median world in its agents and in its decisions; and no more even than B2's least even world.
>
> **Why it would be so.** Every healthy-looking world so far was found by changing B2 one rule at a time, and each paid for one line with another: the estates make spaces of paupers, mixed brains rich worlds that fold, no mending worlds that can be cut (Chapters [100](room-for-the-steadiest-world.md) and [101](room-for-the-steadiest-world.md#births-as-rich-as-the-neighbours)). Settings drawn together reach combinations no single change would, and the round is judged by every line at once. B2's own rules come together too rarely for a random draw to meet them, so two worlds of B2 at 20,000 tokens are played beside the fifty for reference. A healthy world is rare: of 45 worlds of B2's kind read at 2,000, ten met the health bar's marks at a dimension of 2 to 4, before width, diversity and evenness were asked for. The thesis asks for one in fifty.
>
> **It holds if:** At least one of the fifty drawn worlds meets all five lines at 2,000 (research/screen/judge.py, with B2's lines from Experiment 16's ten worlds).
>
> **It fails if:** None does: the chapter says which lines the worlds missed, and which values the next round forces off because every world they appeared in was unhealthy, beyond what chance would explain.
>
> *Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*
<!-- /thesis -->

The thesis speaks of five lines. Its second line, *decentralised and
resilient*, became two criteria when the author set the rule above; its other
four lines are criteria 1, 4, 5 and 6. A world meets all five lines exactly
when it meets all six criteria, so the thesis's test is unchanged. Its last
sentence speaks of the earlier rule for setting values aside, which asked
what chance would explain ([How our view changed](#how-our-view-changed)).
Round 1 is judged by the author's rule, with the chance printed beside each
value marked bad.

## Results still to come

The fifty worlds of round 1, and the two worlds of B2 beside them, are being
played in the lab. The round has not been judged yet.
Its results are still to come: until the judge has read every world, the
thesis is neither confirmed nor refuted, and no value is set aside. Then this
page will say which worlds were healthy or interesting, which criteria the
others missed, and which values the next round draws no more.

## Where to go next

- [What makes a world interesting](what-makes-a-world-interesting.md): the
  six criteria, B2's lines and the rule for setting a value aside, in full.
- [Room and riches for the steadiest world](room-for-the-steadiest-world.md):
  the search one rule at a time that this screen takes over, and what each
  rule cost.
- [Healthy and rich](healthy-and-rich.md): the author's picture of a world
  that might keep evolving, and every world scored against it.
- [Strategies written by hand](strategies-written-by-hand.md): what will be
  tried in a healthy world, once one is found.

## How this was made

Round 1 is the plan of Experiment 38 (`book/experiments/E38.json`): sampler
1, round seed 1, fifty worlds of B1's rules changed as drawn, at 20,000
tokens, for 2,000 iterations or four hours, and two worlds of B2 beside them.
World *i* of a round with seed *s* has seed 1000·*s* + *i*, so round 1's
worlds have seeds 1001 to 1050. The worlds of B2 have seeds
1000·*s* + 900 + *j*: 1901 and 1902.

Every value is drawn from a hash of the sampler's version, the round's seed,
the world's number and the setting's name, not from one random stream. So
adding worlds to a round, or drawing a refused combination again, changes no
other world. A draw the engine refuses is drawn again with the next attempt
number; none of round 1's draws was refused. A world's settings are kept as
what differs from B1. That is why every row of the table of settings below
lists the brain's hidden layers.

The 25 million million worlds are counted from the sampler's list in
`gol_screen.py`. The count multiplies every setting's number of values. A
setting that can matter only beside another is counted only where it can:
messages, for example, give 17 ways, none or one of 4 sizes with each of its
two companion switches on or off.

The judge plays nothing. It reads every world from its archives: the whole
world, brains included, kept every 500 iterations
([`archive_every`](settings.md#archive_every)). It judges a world at
iteration 2,000, and reads iteration 1,000 the same way.
`python3 research/screen/judge.py --reference` reads B2's lines from
Experiment 16's worlds, and `python3 research/screen/judge.py E38` judges the
round.

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
  all died, or all grew [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub."), or all folded, was set aside too, if other
  worlds did not. Before round 1 was judged, the author replaced this rule with
  the simpler one above. The same day the line *decentralised and resilient*
  was split in two, so the five lines became six criteria
  ([What makes a world interesting](what-makes-a-world-interesting.md)). The
  judge still prints the chance, as a warning. And round 1's draws in fact
  hold all 92 values of the sampler, not about eighty.
- **23 October 2026.** From 22 October, this page said that a diverse world's
  agents spread at least as widely as those of B2's median world, and decide
  at least as variously. The judge has always asked for more than B2's median
  world, and the page has said so since 23 October.
- **23 October 2026.** From 22 October, this page said that every number's
  default is B1's. That holds for the rules' numbers and for mutation, but not
  for the brain's size. The sampler's default brain is one hidden layer of 16
  neurons, and B1's five layers are never drawn (`gol_screen.py`,
  `book/experiments/B1.json`). So no drawn world has B1's brain, and every row
  of the table of settings lists `hidden_layers`.

<!-- turns -->
---

← [Room and riches for the steadiest world](room-for-the-steadiest-world.md) · [Looking for a healthy, rich world](../README.md#looking-for-a-healthy-rich-world)

> [!info]- Pages that link here (9)
> [What this notebook is about](what-this-notebook-is-about.md) · [Reproduction statistics](reproduction-statistics.md) · [The random twin: is a world a space?](random-twin.md) · [Healthy and rich](healthy-and-rich.md) · [What makes a world interesting](what-makes-a-world-interesting.md) · [The health bar](health-bar.md) · [The diversity score](diversity-score.md) · [Room and riches for the steadiest world](room-for-the-steadiest-world.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
