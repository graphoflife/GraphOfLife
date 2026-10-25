# The author's walkthrough of the screen (2026-10-24)

The author's decisions, setting by setting, for the next rounds of random
worlds, and the criteria they are judged by. Written down as they are made;
the sampler (`gol_screen.py`) and the criteria (`research/screen/criteria.py`)
change only after the walkthrough is complete.

## The criteria

Every number is an **average over iterations 100 to the end of the run**
(3,000 in round 3; the first 100 iterations are always ignored), not a reading
at one moment. Every threshold is a visible number the author controls, shown
with each world's value, with a sensitivity table (how many worlds pass as
each threshold moves).

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


## The judge for round 3 (built 25 October 2026)

Round 3 (Experiment 40) started on 25 October. The judge reads finished worlds
only, plays none, and keeps every reading as it is made (one file a world in
`GraphOfLifeRuns/.pilots/screen/worlds/`), so it can be run again as worlds
finish: `python3 research/screen/judge.py E40 --jobs 3`.

- **Files**: the lines and verdicts in `criteria.py` (read by the page too);
  the first criteria, for rounds 1 and 2, kept as they were in
  `criteria_1.py`; the readings in `measures.py`; `judge.py` judges a round by
  its sampler's criteria (re-judging round 1 gives its committed verdict
  exactly); tests in `tests/test_judge.py` (the judge's tests moved there from
  `test_lab.py`), in CI.
- **Where each number comes from**, every one averaged from iteration 100 to
  the end:
  - every iteration, from `stats.jsonl`: the share of agents having a child
    (`births / nodes_before`, phase 1; the line uses the larger of the share
    and its complement), links per agent (`meanDegree`), the hub
    (`maxDegree / (nodes - 1)`), the largest holding (`maxTokens`), the Gini;
    the spread-or-all-in doctrine (`spreadShare`) is reported, not judged (in
    the first world 98.5% of agents spread, which says nothing about how alike
    their stakes are);
  - every 25th iteration (the heavy rows), from the game's frame (found
    through each row's `_frame`): how alike the agents are in tokens, age and
    genotype (`brain_ids` name genotypes in the engine); the 2-core, and how
    alike its agents' numbers of links are; the worst agent and link cut
    (`research/atlas/profile.py`); the dimension from the heavy rows;
  - every kept world (250 to 3,000): the width over the random twin
    (`book_chapters/space.measure`, `book_graph.degree_preserving`), the
    half-cut exponent of the world and its twin, and the same-position share
    (`diversity.stake_kinds(..., game=False)["pairs_apart"]`, with the
    effective kinds).
- **The half cut**: the cheapest of the median cuts of the ten smoothest
  eigenvectors of the normalised Laplacian (numpy Lanczos, 400 steps; scipy is
  not installed). One eigenvector alone is not enough: the smallest
  eigenvalues lie close together, and on B1-30000-10c9f1-s3047 at 3,000 the
  Fiedler vector's median cut 121 links where the third eigenvector's cut 62.
  At 200 steps the vectors had not converged (one world: 216 links against an
  exact 33); at 400 the ten cuts match an exact eigensolver's on four worlds.
- **Cost**: a frame 0.03 s; the width 2.4 s; a half cut about 10 s; the
  same-position share about a minute per kept world. Some 15 minutes a world,
  6 hours for the round on 3 processes, one numpy thread each (several threads
  a process loaded the machine to 30 on 8 cores beside the lab).
- **The lines** (each a number the author controls, with a sensitivity
  table): hub ≤ 0.05; links 3–12; 2-core ≥ 0.8; agent and link cut ≤ 0.15;
  width ≥ 1.3; dimension 2–5; tokens, core links, ages, genotypes each
  < 0.8 alike; largest holding > 3; children < 0.95 alike; Gini ≥ 0.46.
  "One piece" is kept by the rules (cleanup removes everything cut off).
  Interesting stays undefined: "at most two misses" is shown as provisional,
  and so is every value it marks bad.
