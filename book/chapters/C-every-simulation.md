# Every simulation: what was run, and what was measured

This appendix lists every world the book has made: the lab's experiments,
which were planned before they ran and are judged against a thesis, and the
pilot batches, smaller sets of worlds that try an idea quickly. For each it
gives the settings, how many worlds, how far they ran, the chapter that reports
them, and which measures have been taken on them.

The tables are made from the records themselves — the lab's plans
(`book/experiments/`), the settings file of every pilot world, the lists of
jobs still to run, and the result files of every measure — by
`python3 book_figures.py catalogue`. A batch still running shows how far it has
got; run that command again and the tables catch up.

## The two rule sets

Every world starts from one of two sets of rules. **B1** is the algorithm
exactly as a new run is offered it; the book's Parts II to IV study it.
**B2** is the world of Part V, built so that its network cannot break apart
([Chapter 48](48-a-world-that-cannot-break.md)). They differ in these settings,
and in no other:

<!-- table catalogue/baselines -->
| setting | B1 | B2 | what it is |
|---|---|---|---|
| [`hidden_layers`](../notes/settings.md#hidden_layers) | 50, 45, 40, 35, 30 | 16 | The widths of the brain's hidden layers, in order from input to output. |
| [`message_amount`](../notes/settings.md#message_amount) | 30 | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
| [`message_prepass`](../notes/settings.md#message_prepass) | on | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
| [`allow_handover`](../notes/settings.md#allow_handover) | on | off | Whether a parent may move some of its own connections to its newborn child. |
| [`mutate_on_replication`](../notes/settings.md#mutate_on_replication) | off | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
| [`incumbency`](../notes/settings.md#incumbency) | 1.0 | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
| [`birth_links`](../notes/settings.md#birth_links) | 0 | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
| [`mend_on_death`](../notes/settings.md#mend_on_death) | off | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
| [`prune_needs`](../notes/settings.md#prune_needs) | 0 | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
| [`valence`](../notes/settings.md#valence) | 0 | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
| [`keep_home`](../notes/settings.md#keep_home) | 0.0 | 0.3 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
| [`max_age`](../notes/settings.md#max_age) | 0 | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
| [`min_child`](../notes/settings.md#min_child) | 1 | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
<!-- /table -->

Every other setting, and what each one means, is in the note
[Every setting](../notes/settings.md).

## The lab's experiments

An experiment of the lab is a plan written before any of its runs existed: a
question, a thesis, the conditions, the seeds and how far to run. The runs are
made with `python3 gol_lab.py run E…` or ▶ in the Book tab; every one of them
writes its settings and engine version beside its record. Several experiments
answer their questions from the runs of another (Experiments 3 to 6 read
Experiment 2's).

<!-- table catalogue/lab -->
|  | question | on | conditions (and what each changes) | seeds | tokens | iterations | runs on disk | chapter |
|---|---|---|---|---|---|---|---|---|
| E01 | Is a run reproducible? | B1 | *straight*; *stopped at 10*; *cut off at 15*; *all BLAS threads* | 1–3 | 10,000 | 30 | 12 | [8](08-is-a-run-reproducible.md) |
| E02 | The life of a world | B1 | *baseline* | 1–30 | 10,000 | 3,000 | 30 | [10](10-a-worlds-life.md) |
| E03 | How much does the seed decide? | B1 | the runs of E02; *baseline* | 1–30 | 10,000 | 3,000 | 30 | [14](14-how-much-does-the-seed-decide.md) |
| E04 | Where do the tokens go? | B1 | the runs of E02; *baseline* | 1–30 | 10,000 | 3,000 | 30 | [15](15-where-do-the-tokens-go.md) |
| E05 | What shape does the graph take? | B1 | the runs of E02; *baseline* | 1–30 | 10,000 | 3,000 | 30 | [17](17-what-shape-does-the-network-take.md) |
| E06 | Who wins, and for how long? | B1 | the runs of E02; *baseline* | 1–30 | 10,000 | 3,000 | 30 | [18](18-genotypes-and-lineages.md) |
| E07 | Do the brains matter at all? | B1 | *baseline*; *brains never change* (mutation_sparsity = 0); *decisions by chance* (random_decisions = on) | 1–30 | 10,000 | 3,000 | 90 | [41](41-do-the-brains-matter.md) |
| E08 | How does a world's size follow its tokens? | B1 | *baseline*; *stopped only when empty* (extinction_threshold = 0) | 1–3 | 800–409,600 | 600 | 42 | [42](42-how-does-a-worlds-size-follow-its-tokens.md) |
| E09 | How fast should brains change? | B2 | *mutation 0.05* (mutation_probability = 0.05); *mutation 0.1* (mutation_probability = 0.1); *B2*; *mutation 0.4* (mutation_probability = 0.4); *mutation 0.8* (mutation_probability = 0.8) | 1–10 | 10,000 | 1,500 | 50 | [52](52-how-fast-should-brains-change.md) |
| E10 | Worlds of 100,000 tokens | B1 | *baseline* | 1–30 | 100,000 | 1,500 | 30 | [60](60-worlds-of-100000-tokens.md) |
| E12 | Does the baseline keep improving? | B1 | *baseline* | 1–10 | 5,000 | 1,000 | 10 | [46](46-how-to-tell-a-world-that-keeps-evolving.md) |
| E13 | Brains that see, and worlds that break | B1 | *baseline*; *one hidden layer* (hidden_layers = 16); *one hidden layer, change only when copied* (hidden_layers = 16, mutate_on_replication = on) | 1–10 | 5,000 | 1,500 | 30 | [47](47-brains-that-see-and-worlds-that-break.md) |
| E14 | A world that cannot break | B2 | *all the rules*; *no mending* (mend_on_death = off); *any unused link lapses* (prune_needs = 0); *no links at birth* (birth_links = 0); *no valence* (valence = 0); *handover* (allow_handover = on) | 1–10 | 5,000 | 1,000 | 60 | [48](48-a-world-that-cannot-break.md) |
| E15 | Who dies, and why | B2 | *B2*; *the last token kept* (keep_home = 0.01); *nothing kept* (keep_home = 0.0); *half kept* (keep_home = 0.5); *children of any size* (min_child = 1); *no old age* (max_age = 0) | 1–10 | 10,000 | 1,000 | 58 | [49](49-who-dies-and-why.md) |
| E16 | Why evolution stops | B2 | *B2*; *no random inputs* (random_input_amount = 0) | 1–10 | 10,000 | 2,000 | 20 | [50](50-why-evolution-stops.md) |
<!-- /table -->

Experiment 11 — one world of 204,800 tokens for 100,000 iterations
([Chapter 69](69-a-hundred-thousand-iterations.md)) — is planned and has not
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
| batch | what it asks | conditions | worlds | tokens | iterations | status | chapter |
|---|---|---|---|---|---|---|---|
| A3k | Alliances, kin pooling and a stake gain, first tried at 3,000 tokens (kept only as hints: the book's worlds have at least 10,000 tokens since) | 4 | 8 | 3,000 | 1,000 | 5 done, 3 stopped early | [56](56-alliances.md) |
| A10k | Cooperation with a surplus: alliances by mutual consent, sides pooled by kinship, support that counts twice, against a control | 4 | 8 | 10,000 | 1,000 | done | [56](56-alliances.md) |
| AL | Alliances over a long time and at 20,000 tokens | 2 | 4 | 10,000, 20,000 | 2,000, 3,000 | done | [56](56-alliances.md) |
| N | Rules without numbers (B3, with and without allies) and alliances without the home advantage | 4 | 8 | 10,000 | 1,000 | done | [56](56-alliances.md), [57](57-rules-without-numbers.md) |
| S | The switches the world was built with (revolutions, gifts, lapsing links, estates by wealth) and other contests (a lottery, sides pooled by genotype or family, kin and family senses) | 13 | 26 | 10,000 | 1,000 | done | [54](54-the-switches-the-world-was-built-with.md), [55](55-another-game.md) |
| L | Long runs: Experiment 16's B2 worlds played on to 10,000 iterations, B1 to 10,000, B1 without revolutions to 3,000 | 3 | 13 | 10,000 | 3,000, 8,000, 10,000, 40,000 | 12 done, 1 running | [50](50-why-evolution-stops.md), [69](69-a-hundred-thousand-iterations.md) |
| P | Two inputs (where my brain came from, who helped me) and links that hold a token | 6 | 12 | 10,000 | 2,000 | done | [57](57-rules-without-numbers.md), [58](58-gliders-and-who-helps-whom.md) |
| Q | Brains mixed at birth, brain units that hold a token, an eye for raids on kin, and links that hold a token without old age | 5 | 10 | 10,000 | 2,000 | done | [78](78-mixed-brains-costly-units.md) |
| R | Who inherits the estate of the dead: the poorest, the newborns, their parents | 6 | 12 | 10,000 | 2,000 | 8 done, 4 running | — |
| T | Gifts together with each other switch | 24 | 48 | 10,000 | 1,000 | 2 done, 4 running, 42 queued | — |
| U | Eight switches in combination on B2: a fractional factorial design | 16 | 16 | 10,000 | 1,000 | 0 done, 16 queued | — |
| V | Eight switches in combination on B1: a fractional factorial design | 16 | 16 | 10,000 | 1,000 | 0 done, 16 queued | — |
| W | No revolutions together with each other switch, on B2 | 24 | 48 | 10,000 | 1,000 | 0 done, 2 running, 46 queued | — |
| X | Handover together with each other switch, on B2 | 23 | 46 | 10,000 | 1,000 | 0 done, 2 running, 44 queued | — |
<!-- /table -->

Batch L's iterations are those each job ran: B2's worlds were played on from
iteration 2,000 to 10,000, and three of them are being played on to 50,000.

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
| L | B2-long | B2 | nothing: B2 itself | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 | 8,000 |
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
| R | B1-newborns | B1 | redistribution = newborns | 1, 2 | 2,000 |
| R | B1-parents | B1 | redistribution = parents | 1, 2 | 2,000 |
| R | B1-poorest | B1 | redistribution = poorest | 1, 2 | 2,000 |
| R | B2-newborns | B2 | redistribution = newborns | 1, 2 | 2,000 |
| R | B2-parents | B2 | redistribution = parents | 1, 2 | 2,000 |
| R | B2-poorest | B2 | redistribution = poorest | 1, 2 | 2,000 |
| T | gifts-bytokens | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, redistribution = by_tokens | 1 | 1,000 |
| T | gifts-newborns | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, redistribution = newborns | 1 | 1,000 |
| T | gifts-norev | B2 | allow_gifting = on, allow_revolutions = off, inactive_window = iteration, prune_after = reproduction | 1 | 1,000 |
| T | gifts-parents | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, redistribution = parents | 1 | 1,000 |
| T | gifts-poorest | B2 | allow_gifting = on, inactive_window = iteration, prune_after = reproduction, redistribution = poorest | 1 | 1,000 |
| T | handover | B2 | allow_handover = on | 1 | 1,000 |
| W | norev-bytokens | B2 | allow_revolutions = off, redistribution = by_tokens | 1 | 1,000 |
| W | norev-poorest | B2 | allow_revolutions = off, redistribution = poorest | 1 | 1,000 |
| X | handover-bytokens | B2 | allow_handover = on, redistribution = by_tokens | 1 | 1,000 |
| X | handover-poorest | B2 | allow_handover = on, redistribution = poorest | 1 | 1,000 |
<!-- /table -->

## What was measured on what

Not every measure has been taken on every world. Each number below is how many
worlds of that experiment or batch the measure has been taken on:

- **the record of every game** — the statistics every run keeps, game by game
  (a lab run's `stats.jsonl`, a pilot's `series.jsonl`;
  [What a run records](../notes/frames-and-stats.md));
- **lifetime contests (T = 100)** — the brains of a later moment against those
  of an earlier one, for an agent's lifetime
  ([Chapter 50](50-why-evolution-stops.md#would-we-notice-if-it-went-on));
- **games recorded and read again** — takeover chains, help, flows that go
  round ([Chapter 58](58-gliders-and-who-helps-whom.md)) and, where there are
  gifts, who gives to whom;
- **flow modules** — the groups tokens stay in
  ([Chapter 71](71-flow-modules.md));
- **how different the agents are** — every brain in the same situations
  ([Chapter 70](70-how-different-are-the-agents.md));
- **the table of every world** — every statistic over the same stretches of
  iterations, side by side ([Meta VI](74-meta-6-every-world.md)).

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
| E15 | 58 | 19 | 10 | 25 | 22 | 55 |
| E16 | 20 | 10 | 10 | 10 | 13 | 20 |
| A3k | 8 | 0 | 0 | 0 | 0 | 8 |
| A10k | 8 | 8 | 4 | 0 | 0 | 8 |
| AL | 4 | 2 | 2 | 0 | 0 | 4 |
| N | 8 | 3 | 0 | 0 | 0 | 8 |
| S | 26 | 26 | 0 | 26 | 6 | 26 |
| L | 13 | 3 | 0 | 9 | 5 | 7 |
| P | 12 | 12 | 12 | 12 | 12 | 12 |
| Q | 10 | 10 | 10 | 10 | 10 | 10 |
| R | 12 | 2 | 0 | 2 | 0 | 0 |
| T | 6 | 0 | 0 | 0 | 0 | 0 |
| U | 0 | 0 | 0 | 0 | 0 | 0 |
| V | 0 | 0 | 0 | 0 | 0 | 0 |
| W | 2 | 0 | 0 | 0 | 0 | 0 |
| X | 2 | 0 | 0 | 0 | 0 | 0 |
<!-- /table -->

Older yardsticks are not counted here: the contests of 20 iterations of
Chapters [46](46-how-to-tell-a-world-that-keeps-evolving.md) to 56 (the lab's Experiments 12 to 16 and the pilot batches of
those chapters), the health bar of [Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md)
(every world from Experiment 12 on), and the correlations among agents of
[Appendix B](B-what-goes-with-what.md) (ten B1 and ten B2 worlds, and any run
in the Research tab).

<!-- turns -->
---

← [Appendix B · What goes with what](B-what-goes-with-what.md) · [Contents](../README.md)
<!-- /turns -->
