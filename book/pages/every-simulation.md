# Every simulation: what was run, and what was measured

> [!question] Questions of this page
> - Which worlds has the notebook made, under which rules, and how far did
>   they run?
> - Which page reports each experiment and each pilot batch?
> - Which measures have been taken on which worlds?

## In short

This page lists every world the notebook has made. The lab's experiments were
planned before they ran, and are judged against a thesis. The pilot
batches are smaller sets of worlds that try an idea quickly. For each, the
page gives the settings, how many worlds, and how far they ran. It names the
page that reports them, and which measures have been taken on them. The tables are made
from the records themselves, so they are always up to date.

## How the tables are made

The tables are made from the records themselves. They read the lab's plans,
in `book/experiments/`, and the settings file of every pilot world. And they
read the lists of jobs still to run, and the result files of every measure. The command
`python3 book_figures.py catalogue` makes them. A batch still running shows
how far it has got; run that command again, and the tables catch up.

## The two rule sets

Every world starts from one of two sets of rules. **B1** is the algorithm
exactly as a new run is offered it. The study of the baseline, and the
experiments of October 2026 that changed one setting at a time, read it. **B2** is the world
built for [open-ended evolution](measuring-evolution.md#what-would-count-as-open-ended-evolution "Evolution that keeps producing new things without end, not only up to a plateau."), built so that its network cannot break apart
([Building B2: a world that cannot break](building-b2-and-the-search-for-a-healthy-rich-world.md#building-b2-a-world-that-cannot-break)).
They differ in a few settings, listed in
[B2: a world that cannot break](b1-and-b2.md#b2-a-world-that-cannot-break).
Every setting, and what each one means, is in [Every setting](settings.md).

## The lab's experiments

An experiment of the lab is a plan written before any of its runs existed.
It holds a question, a thesis, the conditions, the [seeds](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") and how far to run. The runs
are made with `python3 gol_lab.py run E…`, or with ▶ in the Research Notebook
tab. Every one of them writes its settings and engine version beside its
record. Several experiments answer their questions from the runs of another:
Experiments 3 to 6 read Experiment 2's. Two kinds of plan have no conditions of
their own. A **screen** draws every setting of each of its worlds at random
(Experiments 38 to 42). A plan of **trials** plays contests between a world's
[eras](measuring-evolution.md#eras-contests-and-invasions-does-a-world-keep-getting-better "Brains of a world at two moments, its eras, played against each other: do the later ones win?"), or plants strategies written by hand, in worlds another run kept
(Experiments 28 to 30, 33 to 35 and 44). The table lists every plan of the lab.

<!-- table catalogue/lab -->
|  | question | on | conditions (and what each changes) | seeds | tokens | iterations | runs on disk | page |
|---|---|---|---|---|---|---|---|---|
| E01 | Is a run reproducible? | B1 | *straight*; *stopped at 10*; *cut off at 15*; *all BLAS threads* | 1–3 | 10,000 | 30 | 12 | [Is a run reproducible?](is-a-run-reproducible.md) |
| E02 | The life of a world | B1 | *baseline* | 1–30 | 10,000 | 3,000 | 30 | [B1: lives and wealth](b1-lives-and-wealth.md) |
| E03 | How much does the seed decide? | B1 | the runs of E02; *baseline* | 1–30 | 10,000 | 3,000 | 30 | [B1: the network, size and scaling](b1-the-network-size-and-scaling.md) |
| E04 | Where do the tokens go? | B1 | the runs of E02; *baseline* | 1–30 | 10,000 | 3,000 | 30 | [B1: lives and wealth](b1-lives-and-wealth.md) |
| E05 | What shape does the graph take? | B1 | the runs of E02; *baseline* | 1–30 | 10,000 | 3,000 | 30 | [B1: the network, size and scaling](b1-the-network-size-and-scaling.md) |
| E06 | Who wins, and for how long? | B1 | the runs of E02; *baseline* | 1–30 | 10,000 | 3,000 | 30 | [B1: brains, lineages and messages](b1-brains-lineages-and-messages.md) |
| E07 | Do the brains matter at all? | B1 | *baseline*; *brains never change* (mutation_sparsity = 0); *decisions by chance* (random_decisions = on) | 1–30 | 10,000 | 3,000 | 90 | [Do the brains matter?](do-the-brains-matter.md) |
| E08 | How does a world's size follow its tokens? | B1 | *baseline*; *stopped only when empty* (extinction_threshold = 0) | 1–3 | 800–409,600 | 600 | 42 | [B1: the network, size and scaling](b1-the-network-size-and-scaling.md) |
| E09 | How fast should brains change? | B2 | *mutation 0.05* (mutation_probability = 0.05); *mutation 0.1* (mutation_probability = 0.1); *B2*; *mutation 0.4* (mutation_probability = 0.4); *mutation 0.8* (mutation_probability = 0.8) | 1–10 | 10,000 | 1,500 | 50 | [How fast should brains change?](how-fast-should-brains-change.md) |
| E10 | Worlds of 100,000 tokens | B1 | *baseline* | 1–30 | 100,000 | 1,500 | 30 | [B1: the network, size and scaling](b1-the-network-size-and-scaling.md) |
| E12 | Does the baseline keep improving? | B1 | *baseline* | 1–10 | 5,000 | 1,000 | 10 | [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) |
| E13 | Brains that see, and worlds that break | B1 | *baseline*; *one hidden layer* (hidden_layers = 16); *one hidden layer, change only when copied* (hidden_layers = 16, mutate_on_replication = on) | 1–10 | 5,000 | 1,500 | 30 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E14 | A world that cannot break | B2 | *all the rules*; *no mending* (mend_on_death = off); *any unused link lapses* (prune_needs = 0); *no links at birth* (birth_links = 0); *no valence* (valence = 0); *handover* (allow_handover = on) | 1–10 | 5,000 | 1,000 | 60 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E15 | Who dies, and why | B2 | *B2*; *the last token kept* (keep_home = 0.01); *nothing kept* (keep_home = 0.0); *half kept* (keep_home = 0.5); *children of any size* (min_child = 1); *no old age* (max_age = 0) | 1–10 | 10,000 | 1,000 | 60 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E16 | Why evolution stops | B2 | *B2*; *no random inputs* (random_input_amount = 0) | 1–10 | 10,000 | 2,000 | 20 | [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) |
| E17 | Mixed at birth: ten long worlds | B2 | *mixed at birth* (mix_at_birth = on) | 1–10 | 10,000 | 10,000 | 10 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E18 | Five mixed worlds, fifty thousand iterations | B2 | *mixed at birth* (mix_at_birth = on) | 2,4,5,6,10 | 10,000 | 50,000 | 5 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E19 | Mixed at birth, with parents as mates | B2 | *mixed at birth, parents as mates* (mix_at_birth = on, mix_mates = parents) | 1–10 | 10,000 | 20,000 | 10 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E20 | The candidates, played on: two worlds of engine rules | B2 | *links to strangers, a free reach, forced* (link_search = reach, link_consent = forced); *mixed at birth, the last token kept* (mix_at_birth = on, keep_home = 0.01) | 1 | 10,000 | 10,000 | 2 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E21 | Five more seeds for the rules that held | B2 | *mixed at birth* (mix_at_birth = on); *mixed at birth, fast brain change* (mix_at_birth = on, mutation_probability = 0.8); *mixed at birth, no random inputs* (mix_at_birth = on, random_input_amount = 0); *mixed at birth, no revolutions* (mix_at_birth = on, allow_revolutions = off) | 3–7 | 10,000 | 10,000 | 20 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E22 | A mate that pays its share | B2 | *mixed at birth, the mate pays* (mix_at_birth = on, extinction_threshold = 0, proto = {'mate_pays': True}) | 1–4 | 10,000 | 5,000 | 4 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E24 | The search's best world, in new seeds | B2 | *the search's configuration* (hidden_layers = 32, random_input_amount = 0, message_prepass = on, allow_gifting = on, prune_after = reproduction, inactive_window = iteration, incumbency = 1.0, birth_links = 0, mend_on_death = off, prune_needs = 0, keep_home = 0.0, max_age = 0, link_search = walk, mutation_probability = 0.4, extinction_threshold = 0, proto = {'came_from': True, 'contest': 'lottery', 'pool': 'genotype', 'grow': True}); *the search's own world* (hidden_layers = 32, random_input_amount = 0, message_prepass = on, allow_gifting = on, prune_after = reproduction, inactive_window = iteration, incumbency = 1.0, birth_links = 0, mend_on_death = off, prune_needs = 0, keep_home = 0.0, max_age = 0, link_search = walk, mutation_probability = 0.4, extinction_threshold = 0, proto = {'came_from': True, 'contest': 'lottery', 'pool': 'genotype', 'grow': True}) | 1–5 | 10,000 | 2,000 | 6 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E28 | Every era against three | trials | 240 contests from 2 worlds: *against 2,500*; *against 10,000*; *against 25,000*; *each against itself* | — | as their worlds | 100 each | 240 played | [Every era against three](every-era-against-three.md) |
| E29 | How large a crowd a raider needs | trials | 320 invasions from 10 worlds: *raiders, worlds 2 and 3*; *controls, worlds 2 and 3*; *raiders, the other eight worlds*; *controls, the other eight worlds* | — | as their worlds | 300 each | 320 played | [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md) |
| E30 | Does evolution close the door? | trials | 120 invasions from 5 worlds: *raiders*; *controls* | — | as their worlds | 300 each | 120 played | [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md) |
| E31 | Room for the steadiest world | B2 | *estate to the newborns* (mix_at_birth = on, mix_mates = parents, redistribution = newborns); *estate to the parents* (mix_at_birth = on, mix_mates = parents, redistribution = parents); *no old age* (mix_at_birth = on, mix_mates = parents, max_age = 0); *deaths not mended* (mix_at_birth = on, mix_mates = parents, mend_on_death = off) | 1–2 | 10,000 | 3,000 | 8 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E32 | Births as rich as the neighbours | B2 | *estate to the newborns, a child as rich as its neighbours* (mix_at_birth = on, mix_mates = parents, redistribution = newborns, child_floor = neighbours); *estate to the parents, a child as rich as its neighbours* (mix_at_birth = on, mix_mates = parents, redistribution = parents, child_floor = neighbours); *a child as rich as its neighbours* (mix_at_birth = on, mix_mates = parents, child_floor = neighbours) | 1–2 | 10,000 | 3,000 | 6 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E33 | A crew of raiders | trials | 135 invasions from 5 worlds: *crews*; *lone raiders*; *controls* | — | as their worlds | 300 each | 135 played | [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md) |
| E34 | Raiders that breed | trials | 180 invasions from 5 worlds: *breeding raiders*; *breeding turtles*; *lone raiders*; *controls* | — | as their worlds | 300 each | 180 played | [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md) |
| E35 | Hand-written strategies in B1 | trials | 90 invasions from 2 worlds: *crews*; *breeding raiders*; *breeding turtles*; *lone raiders*; *controls* | — | as their worlds | 300 each | 90 played | [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md) |
| E36 | The floor in plain B2 | B2 | *a child as rich as its neighbours* (child_floor = neighbours) | 1–2 | 10,000 | 3,000 | 2 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E37 | The floor in B1 | B1 | *a child as rich as its neighbours* (child_floor = neighbours) | 1–2 | 10,000 | 3,000 | 2 | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| E38 | A screen of random worlds, round 1 | B1, as drawn | 50 worlds, each with its settings drawn at random (sampler 1, round seed 1); beside them 2 worlds of B2 | 1001–1050 | 20,000 | 2,000 | 52 | [A screen of random worlds](a-screen-of-random-worlds.md) |
| E39 | A screen of random worlds, round 2 | B1, as drawn | 50 worlds, each with its settings drawn at random (sampler 1, round seed 2); beside them 2 worlds of B2; withdrawn unplayed | 2001–2050 | 20,000 | 2,000 | 0 | [A screen of random worlds](a-screen-of-random-worlds.md) |
| E40 | A screen of random worlds, round 3: the author's settings | B1, as drawn | 84 worlds, each with its settings drawn at random (sampler 2, round seed 3); 16 of the 100 drawn given up unplayed | 3001–3100 | 30,000 | 3,000 | 84 | [A screen of random worlds](a-screen-of-random-worlds.md) |
| E41 | Round 3, a token at a time: weakest first | B1, as drawn | 8 worlds, each with its settings drawn at random (sampler 2, round seed 4) | 4001–4008 | 30,000 | 3,000 | 8 | [A screen of random worlds](a-screen-of-random-worlds.md) |
| E42 | Round 3, a token at a time: strongest first | B1, as drawn | 8 worlds, each with its settings drawn at random (sampler 2, round seed 4) | 4001–4008 | 30,000 | 3,000 | 8 | [A screen of random worlds](a-screen-of-random-worlds.md) |
| E43 | Round 3's best worlds, again: new seeds and more tokens | B1 | 25 conditions, each a world with its own settings (every setting on the page): *s3004*, *s3008*, *s3016*, *s3018*, *s3019*, *s3022*, *s3023*, *s3024*, *s3026*, *s3042*, *s3045*, *s3051*, *s3054*, *s3057*, *s3058*, *s3080*, *s3087*, *s4002 weakest first*, *s4004 weakest first*, *s4006 strongest first*, *s4008 weakest first*, *s3005*, *s3006*, *s3064*, *s3093* | 4301–4302 | 50,000 | 5,000 | 14 | [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md) |
| E44 | Did the brains get better? Round 3's eras against each other | trials | 1,512 contests from 21 worlds: *against 250*; *against 1,500*; *against 3,000*; *each against itself* | — | as their worlds | 100 each | 0 played | [Did the brains get better? Round 3's eras against each other](round-3-eras-against-each-other.md) |
<!-- /table -->
Experiment 11 is planned and has not run. Its plan of 9 October 2026, one world
of B1 at 204,800 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") for 100,000 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."), lies in
`book/experiments/drafts/`, which this table does not read; since 9 November
it is proposed again on three of round 3's candidates
([A hundred thousand iterations](a-hundred-thousand-iterations.md)).
Experiment 1's runs are thirty iterations long: they test that a run can be
repeated exactly.

## The pilot batches

A pilot world is run by `research/oee/pilots/run.py` rather than by the lab
([Pilot worlds](runs-and-records.md#pilot-worlds)). It is quicker to set up, has no
thesis written in advance, and usually two seeds of each condition. Since
batch S, pilots play the engine's own [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") (`--proto engine=true`). So a
pilot with a lab world's settings and seed *is* that lab world, game for
game. The earlier batches (A, AL, N) ran in the prototype's own code. What
each batch asked:

<!-- table catalogue/pilots -->
| batch | what it asks | conditions | worlds | tokens | iterations | status | page |
|---|---|---|---|---|---|---|---|
| A3k | Alliances, kin pooling and a stake gain, first tried at 3,000 tokens (kept only as hints: the notebook's worlds have at least 10,000 tokens since) | 4 | 8 | 3,000 | 1,000 | 5 done, 3 stopped early | [Other contests: a lottery, and alliances](another-game.md) |
| A10k | Cooperation with a surplus: alliances by mutual consent, sides pooled by kinship, support that counts twice, against a control | 4 | 8 | 10,000 | 1,000 | done | [Other contests: a lottery, and alliances](another-game.md) |
| AL | Alliances over a long time and at 20,000 tokens | 2 | 4 | 10,000, 20,000 | 2,000, 3,000 | done | [Other contests: a lottery, and alliances](another-game.md) |
| N | Rules without numbers (B3, with and without allies) and alliances without the home advantage | 4 | 8 | 10,000 | 1,000 | done | [Other contests: a lottery, and alliances](another-game.md), [Rules without numbers](rules-without-numbers.md) |
| S | The switches the world was built with (revolutions, gifts, lapsing links, estates by wealth) and other contests (a lottery, sides pooled by genotype or family, kin and family senses) | 13 | 26 | 10,000 | 1,000 | done | [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md), [Other contests: a lottery, and alliances](another-game.md) |
| L | Long runs: Experiment 16's B2 worlds played on to 10,000 iterations, B1 to 10,000, B1 without revolutions to 3,000 | 3 | 13 | 10,000 | 3,000, 10,000, 50,000 | 10 done, 3 stopped early | [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md), [A hundred thousand iterations](a-hundred-thousand-iterations.md) |
| P | Two inputs (where my brain came from, who helped me) and links that hold a token | 6 | 12 | 10,000 | 2,000 | done | [Rules without numbers](rules-without-numbers.md), [Cooperation, help and gliders (side study)](cooperation-help-and-gliders.md) |
| Q | Brains mixed at birth, brain units that hold a token, an eye for raids on kin, and links that hold a token without old age | 5 | 10 | 10,000 | 2,000 | done | [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md) |
| R | Who inherits the estate of the dead: the poorest, the newborns, their parents | 9 | 18 | 10,000 | 2,000 | done | [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md) |
| T | Gifts together with each other switch | 24 | 48 | 10,000 | 1,000 | done | [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md) |
| U | Eight switches in combination on B2: a fractional factorial design | 16 | 16 | 10,000 | 1,000 | done | [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md) |
| V | Eight switches in combination on B1: a fractional factorial design | 16 | 16 | 10,000 | 1,000 | done | [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md) |
| W | No revolutions together with each other switch, on B2 | 24 | 48 | 10,000 | 1,000 | done | [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md) |
| X | Handover together with each other switch, on B2 | 23 | 46 | 10,000 | 1,000 | done | [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md) |
| Y | Brains mixed at birth together with each other switch, on B2 | 23 | 46 | 10,000 | 1,000 | done | [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md) |
| K | Links to strangers on B2: every search (jump, walk, reach) with every steering and consent | 12 | 24 | 10,000 | 1,000 | done | [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md) |
| Z | Settings drawn at random from every value the notebook has used, the values that make worlds worse dropped round after round (research/oee/search.md) | 64 | 64 | 10,000 | 1,000 | 62 done, 2 stopped early | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| LC | The candidates, the worlds closest to the author's picture of one that might keep evolving, played on to 10,000 iterations | 20 | 25 | 10,000 | 1,000, 10,000 | 23 done, 2 stopped early | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| SW | Mixing switched at a moment: off in the five worlds of Experiment 17 that met every mark, at 5,000; on in five of B2's worlds, at 2,000 | 2 | 10 | 10,000 | 4,000, 7,000 | done | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| M | Steadying a mixed world: each rule of the engine that might hold its size (estates, half kept at home, searches, a slower change of brains), and the estate to a dead agent's own children (H), parents as mates (MP) and a bought life (BL) | 11 | 28 | 10,000 | 5,000 | 24 done, 2 queued, 2 stopped early | [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) |
| YB | Who breeds: only the young or only the old, in B2 and in B2 with brains mixed at birth | 5 | 7 | 10,000 | 2,000 | done | [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) |
| BIG | B2 at 100,000 tokens, one world to 1,000 iterations | 1 | 1 | 100,000 | 1,000 | done | [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) |
<!-- /table -->
Batch L's iterations are those each job ran. B2's worlds were played on from
iteration 2,000 to 10,000. Three of them were begun on the way to 50,000, and
stopped at 12,000.

## Every pilot condition

What each condition changes, against the rule set it starts from: the
settings that differ from B1 or B2, and the prototype's own switches. The
table holds the batches reported outside the Archive. The conditions of the
batches the Archive reports (K, LC, M, Q, R, S, SW, T, U, V, W, X, Y and Z)
are in
[Every condition of the archived batches](what-the-switches-did-in-pilots.md#every-condition-of-the-archived-batches).

<!-- table catalogue/pilot_conditions -->
| batch | condition | on | what it changes | seeds | iterations |
|---|---|---|---|---|---|
| A3k | allies | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, allies = on, birth_links = 2, incumbency = 1.5, keep_frac = 0.3, max_age = 100, memory = on, min_child = 10, mor = all, repair = on, safe_prune = 2, valence = 8, valence_mode = optional | 1, 2 | 1,000 |
| A3k | ctrl | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, birth_links = 2, incumbency = 1.5, keep_frac = 0.3, max_age = 100, memory = on, min_child = 10, mor = all, repair = on, safe_prune = 2, valence = 8, valence_mode = optional | 1, 2 | 1,000 |
| A3k | gain | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, birth_links = 2, incumbency = 1.5, keep_frac = 0.3, max_age = 100, memory = on, min_child = 10, mor = all, repair = on, safe_prune = 2, support = on, support_gain = 2.0, valence = 8, valence_mode = optional | 1, 2 | 1,000 |
| A3k | kin | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, birth_links = 2, incumbency = 1.5, keep_frac = 0.3, kin_pool = 2, max_age = 100, memory = on, min_child = 10, mor = all, repair = on, safe_prune = 2, valence = 8, valence_mode = optional | 1, 2 | 1,000 |
| A10k | allies | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, allies = on, birth_links = 2, incumbency = 1.5, keep_frac = 0.3, max_age = 100, memory = on, min_child = 10, mor = all, repair = on, safe_prune = 2, valence = 8, valence_mode = optional | 1, 2 | 1,000 |
| A10k | ctrl | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, birth_links = 2, incumbency = 1.5, keep_frac = 0.3, max_age = 100, memory = on, min_child = 10, mor = all, repair = on, safe_prune = 2, valence = 8, valence_mode = optional | 1, 2 | 1,000 |
| A10k | gain | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, birth_links = 2, incumbency = 1.5, keep_frac = 0.3, max_age = 100, memory = on, min_child = 10, mor = all, repair = on, safe_prune = 2, support = on, support_gain = 2.0, valence = 8, valence_mode = optional | 1, 2 | 1,000 |
| A10k | kin | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, birth_links = 2, incumbency = 1.5, keep_frac = 0.3, kin_pool = 2, max_age = 100, memory = on, min_child = 10, mor = all, repair = on, safe_prune = 2, valence = 8, valence_mode = optional | 1, 2 | 1,000 |
| AL | allies10k | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, allies = on, birth_links = 2, incumbency = 1.5, keep_frac = 0.3, max_age = 100, memory = on, min_child = 10, mor = all, repair = on, safe_prune = 2, valence = 8, valence_mode = optional | 3, 4 | 3,000 |
| AL | allies20k | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, allies = on, birth_links = 2, incumbency = 1.5, keep_frac = 0.3, max_age = 100, memory = on, min_child = 10, mor = all, repair = on, safe_prune = 2, valence = 8, valence_mode = optional | 1, 2 | 2,000 |
| N | B3a | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, allies = on, memory = on, mor = all, repair = on | 1, 2 | 1,000 |
| N | B3b | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, allies = on, birth_allies = on, memory = on, mor = all, repair = on | 1, 2 | 1,000 |
| N | B3c | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, memory = on, mor = all, repair = on | 1, 2 | 1,000 |
| N | N0 | B1 (the prototype's own code) | allow_handover = off, allow_revolutions = off, hidden_layers = 16, message_amount = 8, message_prepass = off, allies = on, birth_links = 2, incumbency = 1.0, keep_frac = 0.3, max_age = 100, memory = on, min_child = 10, mor = all, repair = on, safe_prune = 2, valence = 8, valence_mode = optional | 1, 2 | 1,000 |
| L | B1-long | B1 | nothing: B1 itself | 1, 2 | 10,000 |
| L | B1norev-long | B1 | allow_revolutions = off | 3 | 3,000 |
| L | B2-long | B2 | nothing: B2 itself | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 | 50,000 |
| P | B2 | B2 | nothing: B2 itself | 1, 2 | 2,000 |
| P | both | B2 | came_from = on, helped_by = on | 1, 2 | 2,000 |
| P | came | B2 | came_from = on | 1, 2 | 2,000 |
| P | helped | B2 | helped_by = on | 1, 2 | 2,000 |
| P | linkseek | B2 | valence = 0, link_token = on, seek = on, seek_max = 0, toll = path | 1, 2 | 2,000 |
| P | linktoken | B2 | valence = 0, link_token = on | 1, 2 | 2,000 |
| YB | mix-young | B2 | breeders = young, mix = on | 1, 2 | 2,000 |
| YB | old | B2 | breeders = old | 1 | 2,000 |
| YB | old-from500 | B2 | breeders = old | 1 | 2,000 |
| YB | young | B2 | breeders = young | 1, 2 | 2,000 |
| YB | young-from500 | B2 | breeders = young | 1 | 2,000 |
| BIG | B2-100000 | B2 | nothing: B2 itself | 1 | 1,000 |
<!-- /table -->
## What was measured on what

Not every measure has been taken on every world. Each number below is how
many worlds of that experiment or batch the measure has been taken on:

- **the record of every game**: the statistics every run keeps, game by game.
  That is a lab run's `stats.jsonl`, or a pilot's `series.jsonl`
  ([What a run records](runs-and-records.md#what-a-run-records));
- **lifetime contests (T = 100)**: the [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") of a later moment against those
  of an earlier one. They play for an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")'s [lifetime](measuring-lives.md#age-lifetime-and-survival-curves "How many iterations an agent lives, from its birth to its death.")
  ([Would we notice if it went on?](how-to-tell-a-world-that-keeps-evolving.md#would-we-notice-if-it-went-on));
- **games recorded and read again**: takeover chains, help, and flows that
  go round ([Gliders, and who helps whom](cooperation-help-and-gliders.md#gliders-and-who-helps-whom)).
  Where there are gifts, also who gives to whom;
- **[flow modules](measuring-tokens.md#flow-modules-and-the-map-equation "A group of agents that a token, following the flow, tends to stay inside.")**: the groups tokens stay in
  ([Flow modules: the groups tokens stay in](b1-the-game-and-its-flows.md#flow-modules-the-groups-tokens-stay-in));
- **how different the agents are**: every brain in the same situations
  ([The effective kinds of staking](measuring-health.md#decisions-the-effective-kinds-of-staking));
- **the table of every world**: every statistic over the same stretches of
  iterations, side by side
  ([Every world side by side](measures-atlases-and-assembly.md#every-world-side-by-side)).

<!-- table catalogue/measures -->
|  | the record of every game | lifetime contests (T = 100) | games recorded and read again | flow modules | how different the agents are | the table of every world |
|---|---|---|---|---|---|---|
| E01 | 12 | 0 | 0 | 0 | 0 | 12 |
| E02 | 30 | 0 | 10 | 10 | 10 | 30 |
| E03 | 30 | 0 | 10 | 10 | 10 | 30 |
| E04 | 30 | 0 | 10 | 10 | 10 | 30 |
| E05 | 30 | 0 | 10 | 10 | 10 | 30 |
| E06 | 30 | 0 | 10 | 10 | 10 | 30 |
| E07 | 90 | 0 | 10 | 20 | 13 | 90 |
| E08 | 42 | 0 | 0 | 37 | 4 | 42 |
| E09 | 50 | 30 | 10 | 25 | 22 | 50 |
| E10 | 30 | 0 | 0 | 5 | 3 | 30 |
| E12 | 10 | 10 | 0 | 5 | 3 | 10 |
| E13 | 30 | 10 | 0 | 15 | 3 | 30 |
| E14 | 60 | 0 | 0 | 30 | 18 | 60 |
| E15 | 60 | 30 | 10 | 25 | 22 | 55 |
| E16 | 20 | 10 | 10 | 10 | 13 | 20 |
| E17 | 10 | 10 | 0 | 0 | 0 | 0 |
| E18 | 5 | 5 | 0 | 0 | 0 | 0 |
| E19 | 10 | 10 | 0 | 0 | 0 | 0 |
| E20 | 2 | 2 | 0 | 0 | 0 | 0 |
| E21 | 20 | 20 | 0 | 0 | 0 | 0 |
| E22 | 0 | 0 | 0 | 0 | 0 | 0 |
| E24 | 0 | 0 | 0 | 0 | 0 | 0 |
| E28 | — | — | — | — | — | — |
| E29 | — | — | — | — | — | — |
| E30 | — | — | — | — | — | — |
| E31 | 8 | 0 | 0 | 0 | 0 | 0 |
| E32 | 6 | 0 | 0 | 0 | 0 | 0 |
| E33 | — | — | — | — | — | — |
| E34 | — | — | — | — | — | — |
| E35 | — | — | — | — | — | — |
| E36 | 2 | 0 | 0 | 0 | 0 | 0 |
| E37 | 2 | 0 | 0 | 0 | 0 | 0 |
| E38 | 52 | 0 | 0 | 0 | 0 | 0 |
| E39 | 0 | 0 | 0 | 0 | 0 | 0 |
| E40 | 84 | 0 | 0 | 0 | 0 | 0 |
| E41 | 8 | 0 | 0 | 0 | 0 | 0 |
| E42 | 8 | 0 | 0 | 0 | 0 | 0 |
| E43 | 14 | 0 | 0 | 0 | 0 | 0 |
| E44 | — | — | — | — | — | — |
| A3k | 8 | 0 | 0 | 0 | 5 | 8 |
| A10k | 8 | 8 | 4 | 0 | 8 | 8 |
| AL | 4 | 2 | 2 | 0 | 0 | 4 |
| N | 8 | 3 | 0 | 0 | 8 | 8 |
| S | 26 | 26 | 13 | 26 | 26 | 26 |
| L | 13 | 5 | 0 | 9 | 5 | 7 |
| P | 12 | 12 | 12 | 12 | 12 | 12 |
| Q | 10 | 10 | 10 | 10 | 10 | 10 |
| R | 18 | 18 | 12 | 2 | 0 | 0 |
| T | 48 | 48 | 10 | 0 | 7 | 0 |
| U | 16 | 16 | 0 | 0 | 0 | 0 |
| V | 16 | 16 | 0 | 0 | 0 | 0 |
| W | 48 | 48 | 6 | 0 | 5 | 0 |
| X | 46 | 46 | 6 | 0 | 5 | 0 |
| Y | 46 | 46 | 2 | 0 | 1 | 0 |
| K | 24 | 24 | 0 | 0 | 0 | 0 |
| Z | 64 | 0 | 0 | 0 | 0 | 0 |
| LC | 25 | 17 | 0 | 0 | 0 | 0 |
| SW | 10 | 0 | 0 | 0 | 0 | 0 |
| M | 26 | 0 | 0 | 0 | 0 | 0 |
| YB | 7 | 0 | 0 | 0 | 0 | 0 |
| BIG | 1 | 0 | 0 | 0 | 0 | 0 |
<!-- /table -->
Older yardsticks are not counted here:
- the contests of 20 iterations that built B2: the lab's Experiments 12 to 16,
  and the pilot batches of that time;
- the health bar of
  [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md),
  taken on every world from Experiment 12 on;
- and the [correlations](statistics.md#two-numbers-together-correlation-a-fitted-line-and-logarithmic-axes "Whether two quantities go up and down together, from −1 to 1.") among agents of
  [What goes with what](measures-atlases-and-assembly.md#what-goes-with-what): ten B1 and ten B2 worlds, and
  any run in the Research tab.

## Where to go next

- [The lab, the engine and pilots](runs-and-records.md#the-lab-the-engine-and-pilots): how runs and pilots are
  made.
- [B1 and B2: the two baselines](b1-and-b2.md): the two rule sets in plain
  words.
- [Every world side by side](measures-atlases-and-assembly.md#every-world-side-by-side): every statistic of
  every world, in one table.


## How our view changed

- **12 November 2026.** The conditions of the pilot batches that the Archive
  reports moved to
  [What the switches did in pilots](what-the-switches-did-in-pilots.md#every-condition-of-the-archived-batches),
  and the settings in which B1 and B2 differ are listed on
  [B1 and B2](b1-and-b2.md#b2-a-world-that-cannot-break) alone.
- **11 November 2026.** The tables were made again from the records. Until then
  they stopped at Experiment 21, because the catalogue could not read a
  screen's plan. Now they list every plan of the lab to Experiment 44, a screen
  or a plan of trials in a row of its own, and name the pages that report each
  since the Archive took the frozen ones.

<!-- turns -->
---

← [How worlds are measured](how-worlds-are-measured.md) · [Methods and statistics](../README.md#methods-and-statistics)

> [!info]- Pages that link here (10)
> [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Questioning the mechanics](questioning-the-mechanics.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [What the brains are like](what-the-brains-are-like.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [Rules without numbers](rules-without-numbers.md) · [How runs are made, recorded and costed](runs-and-records.md) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) · [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md) · [Measures, atlases and assembly (side studies)](measures-atlases-and-assembly.md)
<!-- /turns -->
