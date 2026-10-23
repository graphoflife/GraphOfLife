# Every simulation: what was run, and what was measured

This appendix lists every world the notebook has made: the lab's experiments,
which were planned before they ran and are judged against a thesis, and the
pilot batches, smaller sets of worlds that try an idea quickly. For each it
gives the settings, how many worlds, how far they ran, the page that reports
them, and which measures have been taken on them.

The tables are made from the records themselves — the lab's plans
(`book/experiments/`), the settings file of every pilot world, the lists of
jobs still to run, and the result files of every measure — by
`python3 book_figures.py catalogue`. A batch still running shows how far it has
got; run that command again and the tables catch up.

## The two rule sets

Every world starts from one of two sets of rules. **B1** is the algorithm
exactly as a new run is offered it; the study of the baseline and the experiments that change one
setting at a time read it. **B2** is the world built for open-ended evolution, built so that its network cannot break apart
([A world that cannot break](a-world-that-cannot-break.md)). They differ in these settings,
and in no other:

<!-- table catalogue/baselines -->
| setting | B1 | B2 | what it is |
|---|---|---|---|
| [`hidden_layers`](settings.md#hidden_layers) | 50, 45, 40, 35, 30 | 16 | The widths of the brain's hidden layers, in order from input to output. |
| [`message_amount`](settings.md#message_amount) | 30 | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
| [`message_prepass`](settings.md#message_prepass) | on | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
| [`allow_handover`](settings.md#allow_handover) | on | off | Whether a parent may move some of its own connections to its newborn child. |
| [`mutate_on_replication`](settings.md#mutate_on_replication) | off | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
| [`incumbency`](settings.md#incumbency) | 1.0 | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
| [`birth_links`](settings.md#birth_links) | 0 | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
| [`mend_on_death`](settings.md#mend_on_death) | off | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
| [`prune_needs`](settings.md#prune_needs) | 0 | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
| [`valence`](settings.md#valence) | 0 | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
| [`keep_home`](settings.md#keep_home) | 0.0 | 0.3 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
| [`max_age`](settings.md#max_age) | 0 | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
| [`min_child`](settings.md#min_child) | 1 | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
<!-- /table -->

Every other setting, and what each one means, is in the note
[Every setting](settings.md).

## The lab's experiments

An experiment of the lab is a plan written before any of its runs existed: a
question, a thesis, the conditions, the seeds and how far to run. The runs are
made with `python3 gol_lab.py run E…` or ▶ in the Research Notebook tab; every one of them
writes its settings and engine version beside its record. Several experiments
answer their questions from the runs of another (Experiments 3 to 6 read
Experiment 2's).

<!-- table catalogue/lab -->
|  | question | on | conditions (and what each changes) | seeds | tokens | iterations | runs on disk | page |
|---|---|---|---|---|---|---|---|---|
| E01 | Is a run reproducible? | B1 | *straight*; *stopped at 10*; *cut off at 15*; *all BLAS threads* | 1–3 | 10,000 | 30 | 12 | [Is a run reproducible?](is-a-run-reproducible.md) |
| E02 | The life of a world | B1 | *baseline* | 1–30 | 10,000 | 3,000 | 30 | [A world's life](a-worlds-life.md) |
| E03 | How much does the seed decide? | B1 | the runs of E02; *baseline* | 1–30 | 10,000 | 3,000 | 30 | [How much does the seed decide?](how-much-does-the-seed-decide.md) |
| E04 | Where do the tokens go? | B1 | the runs of E02; *baseline* | 1–30 | 10,000 | 3,000 | 30 | [Where do the tokens go?](where-do-the-tokens-go.md) |
| E05 | What shape does the graph take? | B1 | the runs of E02; *baseline* | 1–30 | 10,000 | 3,000 | 30 | [What shape does the network take?](what-shape-does-the-network-take.md) |
| E06 | Who wins, and for how long? | B1 | the runs of E02; *baseline* | 1–30 | 10,000 | 3,000 | 30 | [Genotypes and lineages](genotypes-and-lineages.md) |
| E07 | Do the brains matter at all? | B1 | *baseline*; *brains never change* (mutation_sparsity = 0); *decisions by chance* (random_decisions = on) | 1–30 | 10,000 | 3,000 | 90 | [Do the brains matter?](do-the-brains-matter.md) |
| E08 | How does a world's size follow its tokens? | B1 | *baseline*; *stopped only when empty* (extinction_threshold = 0) | 1–3 | 800–409,600 | 600 | 42 | [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md) |
| E09 | How fast should brains change? | B2 | *mutation 0.05* (mutation_probability = 0.05); *mutation 0.1* (mutation_probability = 0.1); *B2*; *mutation 0.4* (mutation_probability = 0.4); *mutation 0.8* (mutation_probability = 0.8) | 1–10 | 10,000 | 1,500 | 50 | [How fast should brains change?](how-fast-should-brains-change.md) |
| E10 | Worlds of 100,000 tokens | B1 | *baseline* | 1–30 | 100,000 | 1,500 | 30 | [Worlds of 100,000 tokens](worlds-of-100000-tokens.md) |
| E12 | Does the baseline keep improving? | B1 | *baseline* | 1–10 | 5,000 | 1,000 | 10 | [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md) |
| E13 | Brains that see, and worlds that break | B1 | *baseline*; *one hidden layer* (hidden_layers = 16); *one hidden layer, change only when copied* (hidden_layers = 16, mutate_on_replication = on) | 1–10 | 5,000 | 1,500 | 30 | [Brains that see, and worlds that break](brains-that-see-and-worlds-that-break.md) |
| E14 | A world that cannot break | B2 | *all the rules*; *no mending* (mend_on_death = off); *any unused link lapses* (prune_needs = 0); *no links at birth* (birth_links = 0); *no valence* (valence = 0); *handover* (allow_handover = on) | 1–10 | 5,000 | 1,000 | 60 | [A world that cannot break](a-world-that-cannot-break.md) |
| E15 | Who dies, and why | B2 | *B2*; *the last token kept* (keep_home = 0.01); *nothing kept* (keep_home = 0.0); *half kept* (keep_home = 0.5); *children of any size* (min_child = 1); *no old age* (max_age = 0) | 1–10 | 10,000 | 1,000 | 60 | [Who dies, and why](who-dies-and-why.md) |
| E16 | Why evolution stops | B2 | *B2*; *no random inputs* (random_input_amount = 0) | 1–10 | 10,000 | 2,000 | 20 | [Why evolution stops](why-evolution-stops.md) |
| E17 | Mixed at birth: ten long worlds | B2 | *mixed at birth* (mix_at_birth = on) | 1–10 | 10,000 | 10,000 | 10 | [Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md) |
| E18 | Five mixed worlds, fifty thousand iterations | B2 | *mixed at birth* (mix_at_birth = on) | 2,4,5,6,10 | 10,000 | 50,000 | 5 | [Five mixed worlds, fifty thousand iterations](five-mixed-worlds-fifty-thousand-iterations.md) |
| E19 | Mixed at birth, with parents as mates | B2 | *mixed at birth, parents as mates* (mix_at_birth = on, mix_mates = parents) | 1–10 | 10,000 | 20,000 | 10 | [Mixed at birth, with parents as mates](mixed-at-birth-with-parents-as-mates.md) |
| E20 | The candidates, played on: two worlds of engine rules | B2 | *links to strangers, a free reach, forced* (link_search = reach, link_consent = forced); *mixed at birth, the last token kept* (mix_at_birth = on, keep_home = 0.01) | 1 | 10,000 | 10,000 | 2 | [The candidates, played on](the-candidates-played-on.md) |
| E21 | Five more seeds for the rules that held | B2 | *mixed at birth* (mix_at_birth = on); *mixed at birth, fast brain change* (mix_at_birth = on, mutation_probability = 0.8); *mixed at birth, no random inputs* (mix_at_birth = on, random_input_amount = 0); *mixed at birth, no revolutions* (mix_at_birth = on, allow_revolutions = off) | 3–7 | 10,000 | 10,000 | 20 | [Five more seeds for the rules that held](five-more-seeds-for-the-rules-that-held.md) |
<!-- /table -->

Experiment 11 — one world of 204,800 tokens for 100,000 iterations
([A hundred thousand iterations](a-hundred-thousand-iterations.md)) — is planned and has not
run. Experiment 1's runs are thirty iterations long: they test that a run can
be repeated exactly.

## The pilot batches

A pilot world is run by `research/oee/pilots/run.py` rather than by the lab:
quicker to set up, without a thesis written in advance, usually two seeds of
each condition. Since batch S, pilots play the engine's own game
(`--proto engine=true`), so a pilot with a lab world's settings and seed *is*
that lab world, game for game. The earlier batches (A, AL, N) ran in the
prototype's own code. What each batch asked:

<!-- table catalogue/pilots -->
| batch | what it asks | conditions | worlds | tokens | iterations | status | page |
|---|---|---|---|---|---|---|---|
| A3k | Alliances, kin pooling and a stake gain, first tried at 3,000 tokens (kept only as hints: the notebook's worlds have at least 10,000 tokens since) | 4 | 8 | 3,000 | 1,000 | 5 done, 3 stopped early | [Alliances](alliances.md) |
| A10k | Cooperation with a surplus: alliances by mutual consent, sides pooled by kinship, support that counts twice, against a control | 4 | 8 | 10,000 | 1,000 | done | [Alliances](alliances.md) |
| AL | Alliances over a long time and at 20,000 tokens | 2 | 4 | 10,000, 20,000 | 2,000, 3,000 | done | [Alliances](alliances.md) |
| N | Rules without numbers (B3, with and without allies) and alliances without the home advantage | 4 | 8 | 10,000 | 1,000 | done | [Alliances](alliances.md), [Rules without numbers](rules-without-numbers.md) |
| S | The switches the world was built with (revolutions, gifts, lapsing links, estates by wealth) and other contests (a lottery, sides pooled by genotype or family, kin and family senses) | 13 | 26 | 10,000 | 1,000 | done | [The switches the world was built with](the-switches-the-world-was-built-with.md), [Another game](another-game.md) |
| L | Long runs: Experiment 16's B2 worlds played on to 10,000 iterations, B1 to 10,000, B1 without revolutions to 3,000 | 3 | 13 | 10,000 | 3,000, 10,000, 50,000 | 10 done, 3 stopped early | [Why evolution stops](why-evolution-stops.md), [A hundred thousand iterations](a-hundred-thousand-iterations.md) |
| P | Two inputs (where my brain came from, who helped me) and links that hold a token | 6 | 12 | 10,000 | 2,000 | done | [Rules without numbers](rules-without-numbers.md), [Gliders, and who helps whom](gliders-and-who-helps-whom.md) |
| Q | Brains mixed at birth, brain units that hold a token, an eye for raids on kin, and links that hold a token without old age | 5 | 10 | 10,000 | 2,000 | done | [Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md) |
| R | Who inherits the estate of the dead: the poorest, the newborns, their parents | 9 | 18 | 10,000 | 2,000 | done | [Who inherits?](who-inherits.md) |
| T | Gifts together with each other switch | 24 | 48 | 10,000 | 1,000 | done | [Gifts, with every other rule](gifts-with-every-other-rule.md) |
| U | Eight switches in combination on B2: a fractional factorial design | 16 | 16 | 10,000 | 1,000 | done | [Eight switches at once](eight-switches-at-once.md) |
| V | Eight switches in combination on B1: a fractional factorial design | 16 | 16 | 10,000 | 1,000 | done | [Eight switches at once](eight-switches-at-once.md) |
| W | No revolutions together with each other switch, on B2 | 24 | 48 | 10,000 | 1,000 | done | [Revolutions, with every other rule](revolutions-with-every-other-rule.md) |
| X | Handover together with each other switch, on B2 | 23 | 46 | 10,000 | 1,000 | done | [Handover, with every other rule](handover-with-every-other-rule.md) |
| Y | Brains mixed at birth together with each other switch, on B2 | 23 | 46 | 10,000 | 1,000 | done | [Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md) |
| K | Links to strangers on B2: every search (jump, walk, reach) with every steering and consent | 12 | 24 | 10,000 | 1,000 | done | [Links to strangers](links-to-strangers.md) |
| Z | Settings drawn at random from every value the notebook has used, the values that make worlds worse dropped round after round (research/oee/search.md) | 64 | 64 | 10,000 | 1,000 | 62 done, 2 stopped early | — |
| LC | The candidates, the worlds closest to the author's picture of one that might keep evolving, played on to 10,000 iterations | 20 | 25 | 10,000 | 1,000, 10,000 | 23 done, 2 stopped early | [Healthy and rich](healthy-and-rich.md), [The candidates, played on](the-candidates-played-on.md) |
| SW | Mixing switched at a moment: off in the five worlds of Experiment 17 that met every mark, at 5,000; on in five of B2's worlds, at 2,000 | 2 | 10 | 10,000 | 4,000, 7,000 | done | [Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md) |
| M | Steadying a mixed world: each rule of the engine that might hold its size (estates, half kept at home, searches, a slower change of brains), and the estate to a dead agent's own children (H), parents as mates (MP) and a bought life (BL) | 11 | 28 | 10,000 | 5,000 | 24 done, 2 queued, 2 stopped early | [What holds a world's size](what-holds-a-worlds-size.md), [Mixed at birth, with parents as mates](mixed-at-birth-with-parents-as-mates.md) |
| YB | Who breeds: only the young or only the old, in B2 and in B2 with brains mixed at birth | 5 | 7 | 10,000 | 2,000 | done | [What folds a world](what-folds-a-world.md) |
| BIG | B2 at 100,000 tokens, one world to 1,000 iterations | 1 | 1 | 100,000 | 1,000 | done | [What folds a world](what-folds-a-world.md) |
<!-- /table -->

Batch L's iterations are those each job ran: B2's worlds were played on from
iteration 2,000 to 10,000, and three of them were begun on the way to 50,000
and stopped at 12,000.

## Every pilot condition

What each condition changes, against the rule set it starts from — the
settings that differ from B1 or B2, and the prototype's own switches:

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
| S | G1-lottery | B2 | allow_revolutions = off, contest = lottery | 1, 2 | 1,000 |
| S | G2-genotype | B2 | allow_revolutions = off, pool = genotype | 1, 2 | 1,000 |
| S | G3-kinsense | B2 | allow_revolutions = off, kin_sense = on | 1, 2 | 1,000 |
| S | G4-familysense | B2 | allow_revolutions = off, kin_sense = on, tag_rate = 0.05 | 1, 2 | 1,000 |
| S | G5-familypool | B2 | allow_revolutions = off, pool = family, tag_rate = 0.05 | 1, 2 | 1,000 |
| S | S0-B2 | B2 | nothing: B2 itself | 1, 2 | 1,000 |
| S | S1-norev | B2 | allow_revolutions = off | 1, 2 | 1,000 |
| S | S2-gifts | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction | 1, 2 | 1,000 |
| S | S3-giftslapse | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, prune_needs = 0 | 1, 2 | 1,000 |
| S | S4-lapse | B2 | prune_needs = 0 | 1, 2 | 1,000 |
| S | S5-bytokens | B2 | redistribution = by_tokens | 1, 2 | 1,000 |
| S | S6-B1 | B1 | nothing: B1 itself | 1, 2 | 1,000 |
| S | S7-B1norev | B1 | allow_revolutions = off | 1, 2 | 1,000 |
| L | B1-long | B1 | nothing: B1 itself | 1, 2 | 10,000 |
| L | B1norev-long | B1 | allow_revolutions = off | 3 | 3,000 |
| L | B2-long | B2 | nothing: B2 itself | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 | 50,000 |
| P | B2 | B2 | nothing: B2 itself | 1, 2 | 2,000 |
| P | both | B2 | came_from = on, helped_by = on | 1, 2 | 2,000 |
| P | came | B2 | came_from = on | 1, 2 | 2,000 |
| P | helped | B2 | helped_by = on | 1, 2 | 2,000 |
| P | linkseek | B2 | valence = 0, link_token = on, seek = on, seek_max = 0, toll = path | 1, 2 | 2,000 |
| P | linktoken | B2 | valence = 0, link_token = on | 1, 2 | 2,000 |
| Q | grow | B2 | brain_kind = grow, hidden_layers = 32 | 1, 2 | 2,000 |
| Q | linknoage | B2 | max_age = 0, valence = 0, link_token = on | 1, 2 | 2,000 |
| Q | mix | B2 | mix = on | 1, 2 | 2,000 |
| Q | tookkin | B2 | took_kin = on | 1, 2 | 2,000 |
| Q | unittoken | B2 | brain_kind = grow, hidden_layers = 32, unit_token = on | 1, 2 | 2,000 |
| R | B1-children | B1 | redistribution = children | 1, 2 | 2,000 |
| R | B1-newborns | B1 | redistribution = newborns | 1, 2 | 2,000 |
| R | B1-parents | B1 | redistribution = parents | 1, 2 | 2,000 |
| R | B1-poorest | B1 | redistribution = poorest | 1, 2 | 2,000 |
| R | B2-buylife | B2 | buy_life = on | 1, 2 | 2,000 |
| R | B2-children | B2 | redistribution = children | 1, 2 | 2,000 |
| R | B2-newborns | B2 | redistribution = newborns | 1, 2 | 2,000 |
| R | B2-parents | B2 | redistribution = parents | 1, 2 | 2,000 |
| R | B2-poorest | B2 | redistribution = poorest | 1, 2 | 2,000 |
| T | gifts-B1 | B1 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction | 1, 2 | 1,000 |
| T | gifts-B1norev | B1 | allow_gifting = on, allow_revolutions = off, inactive_window = iteration, prune_after = reproduction | 1, 2 | 1,000 |
| T | gifts-anychild | B2 | allow_gifting = on, inactive_window = iteration, min_child = 1, prune_after = reproduction | 1, 2 | 1,000 |
| T | gifts-bytokens | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, redistribution = by_tokens | 1, 2 | 1,000 |
| T | gifts-came | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, came_from = on | 1, 2 | 1,000 |
| T | gifts-grow | B2 | allow_gifting = on, brain_kind = grow, hidden_layers = 32, inactive_window = iteration, prune_after = reproduction | 1, 2 | 1,000 |
| T | gifts-handover | B2 | allow_gifting = on, allow_handover = on, inactive_window = iteration, prune_after = reproduction | 1, 2 | 1,000 |
| T | gifts-helped | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, helped_by = on | 1, 2 | 1,000 |
| T | gifts-keephalf | B2 | allow_gifting = on, inactive_window = iteration, keep_home = 0.5, prune_after = reproduction | 1, 2 | 1,000 |
| T | gifts-keepone | B2 | allow_gifting = on, inactive_window = iteration, keep_home = 0.01, prune_after = reproduction | 1, 2 | 1,000 |
| T | gifts-linktoken | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, valence = 0, link_token = on | 1, 2 | 1,000 |
| T | gifts-lottery | B2 | allow_gifting = on, allow_revolutions = off, inactive_window = iteration, prune_after = reproduction, contest = lottery | 1, 2 | 1,000 |
| T | gifts-mix | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, mix = on | 1, 2 | 1,000 |
| T | gifts-mut05 | B2 | allow_gifting = on, inactive_window = iteration, mutation_probability = 0.05, prune_after = reproduction | 1, 2 | 1,000 |
| T | gifts-mut80 | B2 | allow_gifting = on, inactive_window = iteration, mutation_probability = 0.8, prune_after = reproduction | 1, 2 | 1,000 |
| T | gifts-newborns | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, redistribution = newborns | 1, 2 | 1,000 |
| T | gifts-noage | B2 | allow_gifting = on, inactive_window = iteration, max_age = 0, prune_after = reproduction | 1, 2 | 1,000 |
| T | gifts-nodice | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, random_input_amount = 0 | 1, 2 | 1,000 |
| T | gifts-norev | B2 | allow_gifting = on, allow_revolutions = off, inactive_window = iteration, prune_after = reproduction | 1, 2 | 1,000 |
| T | gifts-parents | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, redistribution = parents | 1, 2 | 1,000 |
| T | gifts-pool | B2 | allow_gifting = on, allow_revolutions = off, inactive_window = iteration, prune_after = reproduction, pool = genotype | 1, 2 | 1,000 |
| T | gifts-poorest | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, redistribution = poorest | 1, 2 | 1,000 |
| T | gifts-unittoken | B2 | allow_gifting = on, brain_kind = grow, hidden_layers = 32, inactive_window = iteration, prune_after = reproduction, unit_token = on | 1, 2 | 1,000 |
| T | handover | B2 | allow_handover = on | 1, 2 | 1,000 |
| U | U-ABCDEFGH | B2 | allow_gifting = on, allow_handover = on, allow_revolutions = off, inactive_window = iteration, prune_after = reproduction, prune_needs = 0, random_input_amount = 0, redistribution = newborns, valence = 0, link_token = on, mix = on | 1 | 1,000 |
| U | U-ABCG | B2 | allow_gifting = on, allow_revolutions = off, inactive_window = iteration, prune_after = reproduction, prune_needs = 0, mix = on | 1 | 1,000 |
| U | U-ABDH | B2 | allow_gifting = on, allow_handover = on, allow_revolutions = off, inactive_window = iteration, prune_after = reproduction, random_input_amount = 0 | 1 | 1,000 |
| U | U-ABEF | B2 | allow_gifting = on, allow_revolutions = off, inactive_window = iteration, prune_after = reproduction, redistribution = newborns, valence = 0, link_token = on | 1 | 1,000 |
| U | U-ACDF | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, prune_needs = 0, random_input_amount = 0, valence = 0, link_token = on | 1 | 1,000 |
| U | U-ACEH | B2 | allow_gifting = on, allow_handover = on, inactive_window = iteration, prune_after = reproduction, prune_needs = 0, redistribution = newborns | 1 | 1,000 |
| U | U-ADEG | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, random_input_amount = 0, redistribution = newborns, mix = on | 1 | 1,000 |
| U | U-AFGH | B2 | allow_gifting = on, allow_handover = on, inactive_window = iteration, prune_after = reproduction, valence = 0, link_token = on, mix = on | 1 | 1,000 |
| U | U-BCDE | B2 | allow_revolutions = off, prune_needs = 0, random_input_amount = 0, redistribution = newborns | 1 | 1,000 |
| U | U-BCFH | B2 | allow_handover = on, allow_revolutions = off, prune_needs = 0, valence = 0, link_token = on | 1 | 1,000 |
| U | U-BDFG | B2 | allow_revolutions = off, random_input_amount = 0, valence = 0, link_token = on, mix = on | 1 | 1,000 |
| U | U-BEGH | B2 | allow_handover = on, allow_revolutions = off, redistribution = newborns, mix = on | 1 | 1,000 |
| U | U-CDGH | B2 | allow_handover = on, prune_needs = 0, random_input_amount = 0, mix = on | 1 | 1,000 |
| U | U-CEFG | B2 | prune_needs = 0, redistribution = newborns, valence = 0, link_token = on, mix = on | 1 | 1,000 |
| U | U-DEFH | B2 | allow_handover = on, random_input_amount = 0, redistribution = newborns, valence = 0, link_token = on | 1 | 1,000 |
| U | U-none | B2 | nothing: B2 itself | 1 | 1,000 |
| V | V-ABCDEFGH | B1 | allow_gifting = on, allow_handover = off, allow_revolutions = off, inactive_window = iteration, mutate_on_replication = on, prune_after = reproduction, random_input_amount = 0, redistribution = newborns, link_token = on, mix = on | 1 | 1,000 |
| V | V-ABCG | B1 | allow_gifting = on, allow_handover = off, allow_revolutions = off, inactive_window = iteration, prune_after = reproduction, mix = on | 1 | 1,000 |
| V | V-ABDH | B1 | allow_gifting = on, allow_revolutions = off, inactive_window = iteration, prune_after = reproduction, random_input_amount = 0, link_token = on | 1 | 1,000 |
| V | V-ABEF | B1 | allow_gifting = on, allow_revolutions = off, inactive_window = iteration, mutate_on_replication = on, prune_after = reproduction, redistribution = newborns | 1 | 1,000 |
| V | V-ACDF | B1 | allow_gifting = on, allow_handover = off, inactive_window = iteration, mutate_on_replication = on, prune_after = reproduction, random_input_amount = 0 | 1 | 1,000 |
| V | V-ACEH | B1 | allow_gifting = on, allow_handover = off, inactive_window = iteration, prune_after = reproduction, redistribution = newborns, link_token = on | 1 | 1,000 |
| V | V-ADEG | B1 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, random_input_amount = 0, redistribution = newborns, mix = on | 1 | 1,000 |
| V | V-AFGH | B1 | allow_gifting = on, inactive_window = iteration, mutate_on_replication = on, prune_after = reproduction, link_token = on, mix = on | 1 | 1,000 |
| V | V-BCDE | B1 | allow_handover = off, allow_revolutions = off, random_input_amount = 0, redistribution = newborns | 1 | 1,000 |
| V | V-BCFH | B1 | allow_handover = off, allow_revolutions = off, mutate_on_replication = on, link_token = on | 1 | 1,000 |
| V | V-BDFG | B1 | allow_revolutions = off, mutate_on_replication = on, random_input_amount = 0, mix = on | 1 | 1,000 |
| V | V-BEGH | B1 | allow_revolutions = off, redistribution = newborns, link_token = on, mix = on | 1 | 1,000 |
| V | V-CDGH | B1 | allow_handover = off, random_input_amount = 0, link_token = on, mix = on | 1 | 1,000 |
| V | V-CEFG | B1 | allow_handover = off, mutate_on_replication = on, redistribution = newborns, mix = on | 1 | 1,000 |
| V | V-DEFH | B1 | mutate_on_replication = on, random_input_amount = 0, redistribution = newborns, link_token = on | 1 | 1,000 |
| V | V-none | B1 | nothing: B1 itself | 1 | 1,000 |
| W | norev-anychild | B2 | allow_revolutions = off, min_child = 1 | 1, 2 | 1,000 |
| W | norev-bytokens | B2 | allow_revolutions = off, redistribution = by_tokens | 1, 2 | 1,000 |
| W | norev-came | B2 | allow_revolutions = off, came_from = on | 1, 2 | 1,000 |
| W | norev-grow | B2 | allow_revolutions = off, brain_kind = grow, hidden_layers = 32 | 1, 2 | 1,000 |
| W | norev-handover | B2 | allow_handover = on, allow_revolutions = off | 1, 2 | 1,000 |
| W | norev-helped | B2 | allow_revolutions = off, helped_by = on | 1, 2 | 1,000 |
| W | norev-keephalf | B2 | allow_revolutions = off, keep_home = 0.5 | 1, 2 | 1,000 |
| W | norev-keepone | B2 | allow_revolutions = off, keep_home = 0.01 | 1, 2 | 1,000 |
| W | norev-lapse | B2 | allow_revolutions = off, prune_needs = 0 | 1, 2 | 1,000 |
| W | norev-linktoken | B2 | allow_revolutions = off, valence = 0, link_token = on | 1, 2 | 1,000 |
| W | norev-mix | B2 | allow_revolutions = off, mix = on | 1, 2 | 1,000 |
| W | norev-mut05 | B2 | allow_revolutions = off, mutation_probability = 0.05 | 1, 2 | 1,000 |
| W | norev-mut80 | B2 | allow_revolutions = off, mutation_probability = 0.8 | 1, 2 | 1,000 |
| W | norev-newborns | B2 | allow_revolutions = off, redistribution = newborns | 1, 2 | 1,000 |
| W | norev-noage | B2 | allow_revolutions = off, max_age = 0 | 1, 2 | 1,000 |
| W | norev-nodice | B2 | allow_revolutions = off, random_input_amount = 0 | 1, 2 | 1,000 |
| W | norev-parents | B2 | allow_revolutions = off, redistribution = parents | 1, 2 | 1,000 |
| W | norev-poorest | B2 | allow_revolutions = off, redistribution = poorest | 1, 2 | 1,000 |
| W | norev-unittoken | B2 | allow_revolutions = off, brain_kind = grow, hidden_layers = 32, unit_token = on | 1, 2 | 1,000 |
| W | rev-familypool | B2 | pool = family, tag_rate = 0.05 | 1, 2 | 1,000 |
| W | rev-familysense | B2 | kin_sense = on, tag_rate = 0.05 | 1, 2 | 1,000 |
| W | rev-kinsense | B2 | kin_sense = on | 1, 2 | 1,000 |
| W | rev-lottery | B2 | contest = lottery | 1, 2 | 1,000 |
| W | rev-pool | B2 | pool = genotype | 1, 2 | 1,000 |
| X | handover-anychild | B2 | allow_handover = on, min_child = 1 | 1, 2 | 1,000 |
| X | handover-bytokens | B2 | allow_handover = on, redistribution = by_tokens | 1, 2 | 1,000 |
| X | handover-came | B2 | allow_handover = on, came_from = on | 1, 2 | 1,000 |
| X | handover-familypool | B2 | allow_handover = on, pool = family, tag_rate = 0.05 | 1, 2 | 1,000 |
| X | handover-familysense | B2 | allow_handover = on, kin_sense = on, tag_rate = 0.05 | 1, 2 | 1,000 |
| X | handover-grow | B2 | allow_handover = on, brain_kind = grow, hidden_layers = 32 | 1, 2 | 1,000 |
| X | handover-helped | B2 | allow_handover = on, helped_by = on | 1, 2 | 1,000 |
| X | handover-keephalf | B2 | allow_handover = on, keep_home = 0.5 | 1, 2 | 1,000 |
| X | handover-keepone | B2 | allow_handover = on, keep_home = 0.01 | 1, 2 | 1,000 |
| X | handover-kinsense | B2 | allow_handover = on, kin_sense = on | 1, 2 | 1,000 |
| X | handover-lapse | B2 | allow_handover = on, prune_needs = 0 | 1, 2 | 1,000 |
| X | handover-linktoken | B2 | allow_handover = on, valence = 0, link_token = on | 1, 2 | 1,000 |
| X | handover-lottery | B2 | allow_handover = on, contest = lottery | 1, 2 | 1,000 |
| X | handover-mix | B2 | allow_handover = on, mix = on | 1, 2 | 1,000 |
| X | handover-mut05 | B2 | allow_handover = on, mutation_probability = 0.05 | 1, 2 | 1,000 |
| X | handover-mut80 | B2 | allow_handover = on, mutation_probability = 0.8 | 1, 2 | 1,000 |
| X | handover-newborns | B2 | allow_handover = on, redistribution = newborns | 1, 2 | 1,000 |
| X | handover-noage | B2 | allow_handover = on, max_age = 0 | 1, 2 | 1,000 |
| X | handover-nodice | B2 | allow_handover = on, random_input_amount = 0 | 1, 2 | 1,000 |
| X | handover-parents | B2 | allow_handover = on, redistribution = parents | 1, 2 | 1,000 |
| X | handover-pool | B2 | allow_handover = on, pool = genotype | 1, 2 | 1,000 |
| X | handover-poorest | B2 | allow_handover = on, redistribution = poorest | 1, 2 | 1,000 |
| X | handover-unittoken | B2 | allow_handover = on, brain_kind = grow, hidden_layers = 32, unit_token = on | 1, 2 | 1,000 |
| Y | mix-anychild | B2 | min_child = 1, mix = on | 1, 2 | 1,000 |
| Y | mix-bytokens | B2 | redistribution = by_tokens, mix = on | 1, 2 | 1,000 |
| Y | mix-came | B2 | came_from = on, mix = on | 1, 2 | 1,000 |
| Y | mix-familypool | B2 | mix = on, pool = family, tag_rate = 0.05 | 1, 2 | 1,000 |
| Y | mix-familysense | B2 | kin_sense = on, mix = on, tag_rate = 0.05 | 1, 2 | 1,000 |
| Y | mix-grow | B2 | brain_kind = grow, hidden_layers = 32, mix = on | 1, 2 | 1,000 |
| Y | mix-handover | B2 | allow_handover = on, mix = on | 1, 2 | 1,000 |
| Y | mix-helped | B2 | helped_by = on, mix = on | 1, 2 | 1,000 |
| Y | mix-keephalf | B2 | keep_home = 0.5, mix = on | 1, 2 | 1,000 |
| Y | mix-keepone | B2 | keep_home = 0.01, mix = on | 1, 2 | 1,000 |
| Y | mix-kinsense | B2 | kin_sense = on, mix = on | 1, 2 | 1,000 |
| Y | mix-lapse | B2 | prune_needs = 0, mix = on | 1, 2 | 1,000 |
| Y | mix-linktoken | B2 | valence = 0, link_token = on, mix = on | 1, 2 | 1,000 |
| Y | mix-lottery | B2 | contest = lottery, mix = on | 1, 2 | 1,000 |
| Y | mix-mut05 | B2 | mutation_probability = 0.05, mix = on | 1, 2 | 1,000 |
| Y | mix-mut80 | B2 | mutation_probability = 0.8, mix = on | 1, 2 | 1,000 |
| Y | mix-newborns | B2 | redistribution = newborns, mix = on | 1, 2 | 1,000 |
| Y | mix-noage | B2 | max_age = 0, mix = on | 1, 2 | 1,000 |
| Y | mix-nodice | B2 | random_input_amount = 0, mix = on | 1, 2 | 1,000 |
| Y | mix-parents | B2 | redistribution = parents, mix = on | 1, 2 | 1,000 |
| Y | mix-pool | B2 | mix = on, pool = genotype | 1, 2 | 1,000 |
| Y | mix-poorest | B2 | redistribution = poorest, mix = on | 1, 2 | 1,000 |
| Y | mix-unittoken | B2 | brain_kind = grow, hidden_layers = 32, mix = on, unit_token = on | 1, 2 | 1,000 |
| K | jump-looking-forced | B2 | link_consent = forced, link_search = jump, link_steering = looking | 1, 2 | 1,000 |
| K | jump-looking-mutual | B2 | link_search = jump, link_steering = looking | 1, 2 | 1,000 |
| K | jump-random-forced | B2 | link_consent = forced, link_search = jump | 1, 2 | 1,000 |
| K | jump-random-mutual | B2 | link_search = jump | 1, 2 | 1,000 |
| K | reach-looking-forced | B2 | link_consent = forced, link_search = reach, link_steering = looking | 1, 2 | 1,000 |
| K | reach-looking-mutual | B2 | link_search = reach, link_steering = looking | 1, 2 | 1,000 |
| K | reach-random-forced | B2 | link_consent = forced, link_search = reach | 1, 2 | 1,000 |
| K | reach-random-mutual | B2 | link_search = reach | 1, 2 | 1,000 |
| K | walk-looking-forced | B2 | link_consent = forced, link_search = walk, link_steering = looking | 1, 2 | 1,000 |
| K | walk-looking-mutual | B2 | link_search = walk, link_steering = looking | 1, 2 | 1,000 |
| K | walk-random-forced | B2 | link_consent = forced, link_search = walk | 1, 2 | 1,000 |
| K | walk-random-mutual | B2 | link_search = walk | 1, 2 | 1,000 |
| Z | every world its own | B2 | each world its settings drawn at random from the values the notebook has used (research/oee/search.md) | 1001 to 2032 | 1,000 |
| LC | K-reach-random-forced | B2 | link_consent = forced, link_search = reach | 1 | 10,000 |
| LC | M-mix-children | B2 | mix_at_birth = on, redistribution = children | 2 | 10,000 |
| LC | Q-mix | B2 | mix = on | 1, 2 | 10,000 |
| LC | S-G4-familysense | B2 | allow_revolutions = off, kin_sense = on, tag_rate = 0.05 | 2 | 10,000 |
| LC | U-U-BDFG | B2 | allow_revolutions = off, random_input_amount = 0, valence = 0, link_token = on, mix = on | 1 | 10,000 |
| LC | W-norev-linktoken | B2 | allow_revolutions = off, valence = 0, link_token = on | 1 | 10,000 |
| LC | W-norev-mix | B2 | allow_revolutions = off, mix = on | 1, 2 | 10,000 |
| LC | Y-mix-came | B2 | came_from = on, mix = on | 1, 2 | 1,000 |
| LC | Y-mix-familysense | B2 | kin_sense = on, mix = on, tag_rate = 0.05 | 1 | 1,000 |
| LC | Y-mix-grow | B2 | brain_kind = grow, hidden_layers = 32, mix = on | 1 | 10,000 |
| LC | Y-mix-helped | B2 | helped_by = on, mix = on | 1 | 10,000 |
| LC | Y-mix-keepone | B2 | keep_home = 0.01, mix = on | 1, 2 | 10,000 |
| LC | Y-mix-kinsense | B2 | kin_sense = on, mix = on | 2 | 10,000 |
| LC | Y-mix-linktoken | B2 | valence = 0, link_token = on, mix = on | 1 | 10,000 |
| LC | Y-mix-lottery | B2 | contest = lottery, mix = on | 2 | 1,000 |
| LC | Y-mix-mut80 | B2 | mutation_probability = 0.8, mix = on | 1, 2 | 10,000 |
| LC | Y-mix-nodice | B2 | random_input_amount = 0, mix = on | 2 | 10,000 |
| LC | Y-mix-pool | B2 | mix = on, pool = genotype | 1 | 10,000 |
| LC | Y-mix-poorest | B2 | redistribution = poorest, mix = on | 1 | 10,000 |
| LC | Y-mix-unittoken | B2 | brain_kind = grow, hidden_layers = 32, mix = on, unit_token = on | 2 | 1,000 |
| SW | mixoff | B2 | nothing: B2 itself | 2, 4, 5, 6, 10 | 7,000 |
| SW | mixon | B2 | mix_at_birth = on | 1, 2, 3, 4, 5 | 4,000 |
| M | mix-buylife | B2 | buy_life = on, mix_at_birth = on | 1, 2, 3, 4 | 5,000 |
| M | mix-children | B2 | mix_at_birth = on, redistribution = children | 1, 2 | 5,000 |
| M | mix-keephalf | B2 | keep_home = 0.5, mix_at_birth = on | 1, 2 | 5,000 |
| M | mix-mateparents | B2 | mates = parents, mix = on | 1, 2, 3, 4 | 5,000 |
| M | mix-matepays | B2 | mate_pays = on, mix = on | 1, 2 | 5,000 |
| M | mix-mut05 | B2 | mix_at_birth = on, mutation_probability = 0.05 | 1, 2 | 5,000 |
| M | mix-newborns | B2 | mix_at_birth = on, redistribution = newborns | 1, 2 | 5,000 |
| M | mix-parents | B2 | mix_at_birth = on, redistribution = parents | 1, 2 | 5,000 |
| M | mix-poorest | B2 | mix_at_birth = on, redistribution = poorest | 1, 2 | 5,000 |
| M | mix-reach | B2 | link_consent = forced, link_search = reach, mix_at_birth = on | 1, 2 | 5,000 |
| M | mix-walk | B2 | link_search = walk, mix_at_birth = on | 1, 2 | 5,000 |
| YB | mix-young | B2 | breeders = young, mix = on | 1, 2 | 2,000 |
| YB | old | B2 | breeders = old | 1 | 2,000 |
| YB | old-from500 | B2 | breeders = old | 1 | 2,000 |
| YB | young | B2 | breeders = young | 1, 2 | 2,000 |
| YB | young-from500 | B2 | breeders = young | 1 | 2,000 |
| BIG | B2-100000 | B2 | nothing: B2 itself | 1 | 1,000 |
<!-- /table -->

## What was measured on what

Not every measure has been taken on every world. Each number below is how many
worlds of that experiment or batch the measure has been taken on:

- **the record of every game** — the statistics every run keeps, game by game
  (a lab run's `stats.jsonl`, a pilot's `series.jsonl`;
  [What a run records](frames-and-stats.md));
- **lifetime contests (T = 100)** — the brains of a later moment against those
  of an earlier one, for an agent's lifetime
  ([Why evolution stops](why-evolution-stops.md#would-we-notice-if-it-went-on));
- **games recorded and read again** — takeover chains, help, flows that go
  round ([Gliders, and who helps whom](gliders-and-who-helps-whom.md)) and, where there are
  gifts, who gives to whom;
- **flow modules** — the groups tokens stay in
  ([Flow modules: the groups tokens stay in](flow-modules.md));
- **how different the agents are** — every brain in the same situations
  ([How different are the agents?](how-different-are-the-agents.md));
- **the table of every world** — every statistic over the same stretches of
  iterations, side by side ([Every world side by side](every-world-side-by-side.md)).

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

Older yardsticks are not counted here: the contests of 20 iterations that built B2 (the lab's Experiments 12 to 16 and the pilot batches of
that time), the health bar of [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)
(every world from Experiment 12 on), and the correlations among agents of
[What goes with what](what-goes-with-what.md) (ten B1 and ten B2 worlds, and any run
in the Research tab).

<!-- turns -->
---

← [What goes with what](what-goes-with-what.md) · [Methods and statistics](../README.md#methods-and-statistics) · [What a run records](frames-and-stats.md) →
<!-- /turns -->
