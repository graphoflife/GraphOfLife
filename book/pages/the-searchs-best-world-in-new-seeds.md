# The search's best world, in new seeds

> [!question] Questions of this page
> - The random search over [settings](settings.md "One number or switch of a run's rules, fixed before it starts.") found one world that met every mark,
>   with [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") more diverse than any other world's. Was it the settings, or
>   the luck of one draw?
> - Can the [lab](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") now play the [pilots](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.")' rules, and play them as the pilots did?

<!-- experiment E24 -->

## Why ask

The search by elimination drew every setting at random, round after round,
and kept the values that did not make worlds worse
(`research/oee/search.md`). In its second round one world stood out,
`Z/r2-25`:
- At [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 1,000 it met every mark of the [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4.").
- Its agents were more diverse than in any world the notebook had measured, a
  [diversity score](diversity-score.md "How much the agents of a world differ, in one number from 0 to 1: in tokens, links, ages and genotypes.") of 0.90 against 0.49 for [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world.
- It was a space, twice as wide as its [random twin](random-twin.md "A network with the same agents and as many links each, wired at random: what a world would be if it were no space at all.").

It plays with:
- [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") that can grow;
- a lottery for each node;
- stakes pooled by [genotype](genotype.md "The identity of a brain: a number that a brain keeps until it changes, and that records whose copy it was.");
- an input that tells an agent where its brain came from;
- no mending and no old age;
- every unused link lapsing;
- gifts;
- a walk to find strangers.

The best of 63 draws is also the luckiest. In
[Five more seeds for the rules that held](five-more-seeds-for-the-rules-that-held.md), rules whose
single world had held over 10,000 iterations held no more often than mixing
alone when they were given new [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world."). Only new seeds can tell a
configuration from a lucky draw.

Four of these rules exist only in the research prototype. The lab could not
play them until now. It now hands a run that asks for them to the prototype
itself, frozen beside the engine (`gol_pilot.py`), so the world of seed 2025
should be the pilot's again, game for game.

Read by [Healthy and rich](healthy-and-rich.md)'s clarified test
([Healthy and rich](healthy-and-rich.md)), the world is more even than
any of B2's, with a [Gini](gini-coefficient.md "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all.") of 0.44 against 0.49 and more. So even if the
configuration holds, it is not yet the author's picture. This page
reports how even each new world is.

## The thesis

<!-- thesis E24 -->
> [!quote] The thesis of Experiment 24, written down before any of its runs existed
> **The claim.** The random search's best world was a configuration, not a winner's curse. Played from five new seeds, at least three of the five worlds meet every mark of the health bar at iteration 1,000, have a diversity score above that of B2's median world (0.49) and are spaces: a typical agent reaches half the world in at least 1.5 times the steps its random twin needs. And the lab plays the search's own world, seed 2025, as the pilot did: its agents and links after every game match the pilot's to iteration 1,000.
>
> **Why it would be so.** In the search's second round, the world of Z/r2-25 met every mark at iteration 1,000 with a diversity score of 0.90 and was twice as wide as its random twin: the most promising of 63 worlds whose every setting was drawn at random. It plays with brains that can grow, a lottery for each node, stakes pooled by genotype, an input that tells an agent where its brain came from, no mending, every unused link lapsing, no old age, gifts and a walk to find strangers. The best of 63 draws is also the luckiest, and in Chapter [93](five-more-seeds-for-the-rules-that-held.md) rules whose single world held did not hold in new seeds; only new seeds can tell. The lab now plays the pilots' rules with the prototype itself, frozen beside the engine, so its world of seed 2025 should be the pilot's again. Read by [Meta X](healthy-and-rich.md)'s clarified test the world is more even than any of B2's (a Gini of 0.44 against 0.49 and more): even a configuration that holds would not yet be the author's picture, and the chapter says how even each new world is.
>
> **It holds if:** At least three of the five new worlds (seeds 1 to 5) pass at iteration 1,000 — every mark, a diversity score above 0.49, a typical distance at least 1.5 times the twin's — and the world of seed 2025 matches the pilot's agents and links after every game to iteration 1,000.
>
> **It fails if:** Two or fewer of the five pass, or the world of seed 2025 differs from the pilot's.
>
> *Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*
<!-- /thesis -->

## The runs

<!-- runs E24 -->
> [!info] The runs behind this page
> **6 runs.** Every condition below is run once for each of its seeds, for 2,000 iterations — or until its world dies out. Every setting is that of the baseline **B2** (a world that cannot break) unless the condition changes it.
>
> - **the search's configuration** — changes `hidden_layers` = 32, `random_input_amount` = 0, `message_prepass` = on, `allow_gifting` = on, `prune_after` = reproduction, `inactive_window` = iteration, `incumbency` = 1.0, `birth_links` = 0, `mend_on_death` = off, `prune_needs` = 0, `keep_home` = 0.0, `max_age` = 0, `link_search` = walk, `mutation_probability` = 0.4, `extinction_threshold` = 0, `proto` = came_from, contest = lottery, grow, pool = genotype; 10,000 tokens; seeds 1 to 5. Runs `B2-10000-d2b0d2-s001` … `B2-10000-d2b0d2-s005`.
> - **the search's own world** — changes `hidden_layers` = 32, `random_input_amount` = 0, `message_prepass` = on, `allow_gifting` = on, `prune_after` = reproduction, `inactive_window` = iteration, `incumbency` = 1.0, `birth_links` = 0, `mend_on_death` = off, `prune_needs` = 0, `keep_home` = 0.0, `max_age` = 0, `link_search` = walk, `mutation_probability` = 0.4, `extinction_threshold` = 0, `proto` = came_from, contest = lottery, grow, pool = genotype; 10,000 tokens; seeds 2025. Runs `B2-10000-d2b0d2-s2025` … `B2-10000-d2b0d2-s2025`.
>
> To make these runs again: `python3 gol_lab.py run E24`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.
>
> Runs that ask for the pilots' rules are played by the research prototype, frozen beside the engine (`gol_pilot.py`): they keep no frames, but one row of the world's statistics every iteration (`series.jsonl`) and a whole world at every checkpoint.

> [!info]- Every setting of these runs
> | setting | the search's configuration | the search's own world | what it is |
> |---|---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 10,000 | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | **32** | **32** | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 8 | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | **0** | **0** | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | **on** | **on** | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | off | off | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](settings.md#allow_gifting) | **on** | **on** | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](settings.md#random_decisions) | off | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | **reproduction** | **reproduction** | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | **iteration** | **iteration** | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | on | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | **1.0** | **1.0** | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | **0** | **0** | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | **off** | **off** | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | **0** | **0** | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 8 | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](settings.md#keep_home) | **0.0** | **0.0** | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | **0** | **0** | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 10 | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | off | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`self_message`](settings.md#self_message) | own | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | **walk** | **walk** | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | mutual | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | **came_from, contest = lottery, grow, pool = genotype** | **came_from, contest = lottery, grow, pool = genotype** | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | **0.4** | **0.4** | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | **0** | **0** | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 5 | 2025 | one run per seed |
> | iterations | 2,000 | 2,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B2.
<!-- /runs -->

## What the worlds did

**The lab plays the pilots' rules as the pilots did.** The world of seed
2025, played by the prototype frozen beside the engine, matched the pilot's
agents and links after every one of its 1,000 [games](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."). Every world of the
pilots' rules the lab plays from now on is the world a pilot would have
been.

<!-- figure searchbest/measures -->
![The search's best settings, in six worlds](../figures/searchbest/measures.svg)

**The search's best settings, in six worlds.** Each dot is one world at iteration 1,000 (orange) or 2,000 (violet): the five new seeds small, the search's own seed large. The dashed line is what the author's picture asks: every mark of the health bar, a diversity score above B2's median world (0.49), a typical distance at least 1.5 times the random twin's, and a Gini no lower than B2's least even world (0.47).

> [!example]- How to make this figure
> **Runs.** Experiment 24's six runs (`python3 gol_lab.py run E24`): the settings of `Z/r2-25` played by the research prototype in the lab (`gol_pilot.py`), seeds 1 to 5 (`B2-10000-d2b0d2-s001` … `-s005`) and the search's own seed (`B2-10000-d2b0d2-s2025`), each to 2,000 iterations, with a whole world kept every 100.
>
> **Data.** From each run's checkpoints at 1,000 and 2,000 (`ckpt_001000.npz`, `ckpt_002000.npz`).
>
> 1. Read each world as `research/atlas/profile.py` reads every world: the seven marks, the four spreads ranked among the frozen reference worlds, the Gini of the tokens.
> 2. Its typical distance over its random twin's as `research/atlas/space.py` measures it.
>
> **To make it again:** `python3 book_figures.py searchbest`.
<!-- /figure -->

**The new worlds are what the search's world was, except for its marks.**
At iteration 1,000 every one of the five new worlds was:
- **diverse,** with diversity scores of 0.80 to 0.89 against B2's 0.49,
  among the highest the notebook has measured;
- **a wide space,** 2.6 to 2.75 times as wide as its random twin;
- **rich,** with 11 to 17 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") per agent.

None met more than four marks of the health bar:
- **One agent's loss** cut off 19% to 47% of the world, and one link's
  loss up to 43%.
- **Their [dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.")** was 2.0 to 2.2, below the bar's 2.3: thin, about
  two-dimensional networks.
- **Their tokens** were too even, with Ginis of 0.33 to 0.38, against B2's
  least even world at 0.47.

At 2,000 the picture was much the same; one world met five marks.

**The search's own world was a lucky moment of the same thing.** At 1,000
it met every mark, as the search found. By 2,000 it met four, with a
dimension of 2.15. An agent's loss could cut off a quarter of it, and its
Gini fell to 0.33. It is one of these worlds, caught in a moment when it
happened to have no bottleneck.

## The verdict

**The thesis fails.** None of the five new worlds passed at 1,000; three
were asked for. Every one missed the marks of cuts and dimension. Its
second part holds: the lab's world of seed 2025 is the pilot's, game for
game.

## What it teaches

- **A search judged at one moment finds lucky moments.** The best of 63
  worlds was best at iteration 1,000 by a bottleneck that happened to be
  missing. Its settings make diverse, wide, rich worlds every time. They
  also make thin, cuttable and even ones every time, and the search's
  measure at one moment could not see that.
- **These settings give exactly what Experiment 19's world lacks.** The
  world of mixing with parents as mates is healthy, diverse and uneven, but
  folded. This one is a wide space, but thin and even. Among its settings,
  every unused link lapsing is the one [Rules in company](rules-in-company.md) found to make worlds rich,
  about two-dimensional and short of marks
  ([Rules in company](rules-in-company.md)). That is where the two
  configurations meet, and where [Room for the steadiest world](room-for-the-steadiest-world.md)
  looks for room.
- **The lab now plays the pilots' rules.** Rules that so far lived only in
  the prototype — a lottery for each node, stakes pooled by genotype,
  brains that grow, an input of where a brain came from — can be put to
  the test like any other, beside the pages that ask for them.


<!-- turns -->
---

← [Five more seeds for the rules that held](five-more-seeds-for-the-rules-that-held.md) · [Looking for a healthy, rich world](../README.md#looking-for-a-healthy-rich-world) · [Room for the steadiest world](room-for-the-steadiest-world.md) →

> [!info]- Pages that link here (2)
> [The random twin: is a world a space?](random-twin.md) · [The diversity score](diversity-score.md)
<!-- /turns -->
