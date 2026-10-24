# The author's walkthrough of the screen (2026-10-24)

The author's decisions, setting by setting, for the next rounds of random
worlds, and the criteria they are judged by. Written down as they are made;
the sampler (`gol_screen.py`) and the criteria (`research/screen/criteria.py`)
change only after the walkthrough is complete.

## The criteria

Every number is an **average over iterations 100 to 2,000** (the first 100
iterations are always ignored), not a reading at one moment. Every threshold is
a visible number the author controls, shown with each world's value, with a
sensitivity table (how many worlds pass as each threshold moves).

- **Resilient:** one piece and 80% in the 2-core as before; a single agent's or
  link's loss may cut off up to **15%** ("we are dealing with small networks").
- **Almost three-dimensional:** the looser lines: width at least **1.3** times
  the random twin's, dimension **2 to 5**.
- **Diverse:** not one combined score; each statistic on its own line, "not as
  extreme":
  - tokens: not diverse if 80% or more of the agents hold the same amount, or
    if no agent holds more than 3;
  - links: within the core only (leaves do not make a network bad): not
    diverse if 80% or more of the core's agents have the same number of links;
  - ages: not diverse if 80% or more have the same age;
  - genotypes: not diverse if 80% or more carry the same genotype;
  - decisions: sort out worlds where (nearly) every agent makes the same
    choice, e.g. all reproduce or none do.
- **Interesting:** to be defined together **after the round has run** (the
  author, 2026-10-24). Until then the judge reports every measure of every
  world; "at most two misses" was the author's first answer.
- **Asked for:** another number for health — a network that on the largest
  scale looks like 3-D space, really distributed — while very different
  behaviours survive (agents that decide differently in the same position).
  Proposed: the cost of cutting the world in half (its exponent: 2/3 for a 3-D
  body, near 0 for a bottleneck, near 1 for a folded small world), and the
  share of pairs of agents that decide differently in the same situation.
  The author: **report both first**, on every world of the next round, and set
  their lines together afterwards.

## The settings

Every setting that can be set when running a simulation, not only those the
screen draws (the author, 2026-10-24).

| setting | decision |
|---|---|
| hidden_layers | **three to five layers, widths 16 to 50, drawn per world, with tanh**; the curve (sigmoid or tanh) becomes a setting, also in the simulation manager (sigmoid stays the default, so old runs stay the same) |
| mutation_probability | keep drawing 0.1, 0.2, 0.4, 0.8 |
| mutation_noise_std | keep drawing 0.1, 0.2, 0.4 |
| mutation_sparsity | keep drawing 0.05, 0.1, 0.3 |
| random_input_amount | keep drawing 0 (one world in ten), 2, 5, 10 |
| brain_kind, brain_bits | float16, as now (brain_bits unused) |
| random_decisions | never in random worlds (it is the control) |
| exchange_messages | keep messages (one world in ten still draws none: length 0) |
| message_amount | keep drawing 0, 4, 8, 16, 30 |
| message_prepass | keep drawing, where there are messages |
| self_message | keep drawing own and largest (the door note: its own column, or the largest mark of all the notes written that phase), where there are messages |
| mutate_on_replication | keep drawing (half the worlds change brains only when copied) |
| mix_at_birth | drawn in **one world in three** (in those, every birth mixes); copies of the parent's brain in the other two |
| mix_mates | parents only, where brains are mixed |
| min_child | **fixed at 1**: "they would have to learn over time how many are good to give" |
| child_floor | **off** (fixed): "the parents need to decide how many tokens to allocate to children, and need to learn" |
| birth_links | **0**, the brain's choice only |
| allow_handover | keep drawing |
| max_age | **never**: no age limit |
| buy_life | never (moot without an age limit) |
| redistribution | drop by_tokens; keep uniform, poorest, newborns, parents, children; **add a local rule**: the estate flows to the nearest living agents, through dead neighbours if need be ("the token needs to flow to the nearest alive node"); one of the six drawn per world. Design note: for a piece cut off whole, "nearest" is measured in the network as it was just before the cut |
| mend_on_death | keep drawing |
| valence | **off in 75% of the worlds; in 25% one of 2, 4, 8, 16** (the author dislikes a hard limit: hubs should not arise because there is no game-theoretic reason for them). A cap of 2 leaves chains and rings, and an agent at its cap has no child: to confirm in step 2 |
| prune_needs | **0: no exception** — every link no stake crossed in the game dies |
| prune_after | after the game only (blotto) |
| inactive_window | phase only |
| link_token (new in the engine) | **port the pilots' rule** (every link locks one token, paid by whoever makes it, returned when it lapses) and **draw it in half the worlds** |
| link upkeep (new rule) | **implement and draw it**. The brains decide: a new yes/no output per neighbour, "pay one token to keep this link"; the link stays if either end pays ("the link turns active with that, no additional token flow needed"). Where the token goes, **both, drawn**: to the other end, or into the estate of the dead, shared by whatever estate rule the world has. Whether flow still keeps a link alive, **both, drawn**: paid or used, or paid only |
| link_search | **draw all four** (off, jump, walk, reach); the author prefers a search that costs nothing, which is reach: as many steps as the brain's chosen share of the tokens it holds, none spent |
| link_steering | keep drawing random and looking, where there is a search |
| link_consent | keep drawing mutual and forced, where there is a search |
| incumbency | **1: no home advantage** |
| keep_home | **0: none**; the brains decide what to keep |
| allow_revolutions | keep drawing |
| allow_gifting | keep drawing |
| total_tokens | **30,000** in every world |
| tokens_created_per_phase | 0: a closed world |
| n_nodes | **drawn: one founder per 50, 100 or 200 tokens** (600, 300 or 150 founders at 30,000 tokens) |
| k_neighbors | automatic, as now |
| rewire_p | 0.2, as now |
| iterations | **3,000** per world (the criteria average iterations 100 to 3,000) |
| budget_hours | **no limit**: no world is stopped as too slow |
| seed | one seed per world |
| extinction_threshold | 20, as now |
| export_every | every iteration (frames of every phase, for the viewer and the criteria) |
| export_decisions | kept |
| archive_every | **250** (12 whole worlds per run, for contests and the behaviour measure) |
| checkpoint_every | 100 |
| proto: came_from | left out |
| proto: helped_by | left out |
| proto: contest = lottery | **port it and draw it** (half the worlds play the lottery: a node goes to a staker drawn in proportion to its stake) |
| proto: pool (genotype, family) | left out |
| proto: kin_sense, tag_rate | left out (tag_rate unused) |
| proto: mate_pays | **port it and draw it** in half the mixing worlds |
| proto: grow, unit_token | left out (built for one hidden layer) |
| proto: link_token | port and draw (above) |

## The next round

- **100 drawn worlds, no control worlds.** "We just look at them, as if we had
  not seen any simulation before. But we use already established analysis
  methods to analyse them and search for things we want to see."
- **E39 is replaced**: withdrawn unplayed (its plan and thesis stay on record as
  never run); the next round is planned with the settings and criteria above.

## Open points, with the defaults I will use unless the author says otherwise

- Not too even: with no B2 control worlds, its line is B2's most even world as
  Experiment 16 read it (a Gini of 0.46), a fixed number shown and adjustable
  like the others.
- The decisions line: 95% of the agents making the same choice (the author's
  example was "100% always"); the sensitivity table shows 70% to 100%.
- The local estate: the tokens go to the nearest living agents, in the network
  as it was just before the cut, split evenly among all at that distance.

## Built (24 October 2026)

- **Engine** (commit d86b9e7), every rule off by default so that old runs
  replay as they did: `hidden_activation`, `contest`, `mate_pays`,
  `link_token`, `link_upkeep`, `upkeep_to`, and `redistribution = local`.
  Their entries are on the settings page, and the new-simulation form offers
  each (the curve, sigmoid or tanh, as the author asked).
- **Round 3** (commit 394185b): sampler 2 draws the table above; Experiment 40
  plays a hundred of its worlds; Experiment 39 is withdrawn unplayed.

Choices made in building, open to change:

- Upkeep never takes an agent's last token, as a search and a bought life
  never do. A link is paid for if either end pays; a new link counts as paid
  for the iteration it is made in, so a newborn's links are not cut at once.
- Under `link_token`, a dead agent's links give their tokens to the agents
  they joined (and pay for the links that mend its death), rather than to a
  common pool; a link handed to a child keeps its token, and one that merges
  with a link the child already has leaves its token with the child.
- The local estate: an estate with no path to the living goes to the dead
  agent's parent if it lives; tokens that are no one's estate (upkeep paid
  into the estate, for one) are shared out as `uniform`.
- The lottery needs revolutions off, so the sampler draws revolutions only in
  the largest-stake game.
- The brain's width is drawn once per world, every layer as wide.

