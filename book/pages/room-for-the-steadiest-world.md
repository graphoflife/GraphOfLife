# Room for the steadiest world

> [!question] Questions of this page
> - Which world the notebook has made comes closest to one that might keep
>   evolving, and what does it still miss?
> - Can a rule that keeps a world's room give it the one thing it misses,
>   without taking what it has?

<!-- experiment E31 -->

## Why ask

The author's picture of a world that might keep evolving
([Healthy and rich](healthy-and-rich.md)) asks four things of it at
once:
- **Healthy:** it meets every mark of the health bar.
- **Diverse:** its agents differ, more than B2's do.
- **A space:** it is wide, at least half again as wide as a random network
  with the same links ([What folds a world](what-folds-a-world.md)).
- **Uneven:** its tokens are no more even than B2's
  ([Healthy and rich](healthy-and-rich.md), added later).

Read by all four, few worlds pass even for a moment:
- **At 1,000 iterations:** three of the 466 worlds measured, each with
  brains mixed at birth and one other rule.
- **At 10,000:** one of 64. This is Experiment 21's world without random
  inputs, seed 6. The same rule's other worlds failed, and two of them died.

The steadiest configuration the notebook has is
[Experiment 19](mixed-at-birth-with-parents-as-mates.md)'s: brains mixed
at birth, with only parents as mates.
- **What it has:** all ten of its worlds lived to 20,000 iterations, at
  about 2,000 agents each. They stay diverse (0.52 to 0.69 against B2's
  0.49) and uneven (Ginis of 0.47 to 0.57), and evolution goes on in them,
  later brains beating earlier ones by less and less.
- **What it misses:** room. A typical agent reaches half its world in 1.13
  to 1.32 times the steps its random twin needs. Nine of the ten miss the
  dimension mark, which a folded world of 2,000 agents does. These are
  small worlds, not spaces.

[What folds a world](what-folds-a-world.md) found what folds a world: a death
mended with a ring of links. It also found what keeps a world's room:
- the estate of the dead paid to births;
- no old age;
- no mending.

Each pays for the room in its own coin
([Rules in company](rules-in-company.md)):
- **The estates** fill a world with alike agents. With plain mixing they
  gave spaces of about 5,000 agents whose diversity scores fell to 0.35 to
  0.40.
- **No old age** all but stops births.
- **No mending** lets one loss cut a world.

Parents as mates held a mixed world's size, so with them the estates might
not fill it. This experiment gives each of the four rules to Experiment
19's world, in two seeds each, for 3,000 iterations.

A world of more tokens would not help. B2's kind of world stays folded at
every size: a world of 100,000 tokens was 1.14 times as wide as its twin.
Room has to come from a rule.

## The thesis

<!-- thesis E31 -->
> [!quote] The thesis of Experiment 31, written down before any of its runs existed
> **The claim.** Experiment 19's world, brains mixed at birth with only parents as mates, is steady, diverse and uneven, but folded. One of the rules that keep a world's room unfolds it without costing the rest. For at least one of four rules (the estate of the dead paid to the newborns, or to the parents, no old age, or deaths not mended) both of its worlds, at iteration 3,000, meet every mark of the health bar, are more diverse than B2's median world, are at least 1.5 times as wide as their random twins, and are no more even than B2's least even world, B2's ten worlds played on being read at the same iteration.
>
> **Why it would be so.** All ten of Experiment 19's worlds lived to 20,000 iterations. At 10,000 they held 1,429 to 2,385 agents, with diversity scores of 0.55 to 0.65 against B2's median of 0.49 and Ginis of 0.47 to 0.57 against B2's least of 0.47. Yet a typical agent reached half its world in 1.13 to 1.31 times the steps its random twin needs, and nine of ten missed the dimension mark: they are folded small worlds, not spaces. Chapter [87](what-folds-a-world.md) found what folds a world, a death mended with a ring of links, and what keeps a world's room: the estate paid to births (2.0 to 2.6 times the twin), no old age (1.8 to 1.9) and no mending (2.0). Each pays for the room in its own coin ([Meta XII](rules-in-company.md)): estates fill a world with alike, poor agents, no old age all but stops births, and without mending one loss can cut a world. With plain mixing, the estates gave spaces of about 5,000 agents with diversity scores of only 0.35 to 0.40. With parents as mates a mixed world held its size, which may keep the diversity the estates cost.
>
> **It holds if:** For at least one rule, both worlds (seeds 1 and 2) at iteration 3,000 meet every mark, have a diversity score above the median of B2's ten worlds played on and read at 3,000, a typical distance at least 1.5 times their twin's, and a Gini of their agents' tokens no lower than the least of those ten worlds'.
>
> **It fails if:** No rule has both its worlds pass all four at 3,000: the room and the rest pull apart even in the steadiest mixed world, and the chapter says what each rule cost.
>
> *Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*
<!-- /thesis -->

## The runs

<!-- runs E31 -->
> [!info] The runs behind this page
> **8 runs.** Every condition below is run once for every seed (1 to 2), for 3,000 iterations — or until its world dies out. Every setting is that of the baseline **B2** (a world that cannot break) unless the condition changes it.
>
> - **estate to the newborns** — changes `mix_at_birth` = on, `mix_mates` = parents, `redistribution` = newborns; 10,000 tokens. Runs `B2-10000-06b876-s001` … `B2-10000-06b876-s002`.
> - **estate to the parents** — changes `mix_at_birth` = on, `mix_mates` = parents, `redistribution` = parents; 10,000 tokens. Runs `B2-10000-b8e9fc-s001` … `B2-10000-b8e9fc-s002`.
> - **no old age** — changes `mix_at_birth` = on, `mix_mates` = parents, `max_age` = 0; 10,000 tokens. Runs `B2-10000-89f5b1-s001` … `B2-10000-89f5b1-s002`.
> - **deaths not mended** — changes `mix_at_birth` = on, `mix_mates` = parents, `mend_on_death` = off; 10,000 tokens. Runs `B2-10000-c58a16-s001` … `B2-10000-c58a16-s002`.
>
> To make these runs again: `python3 gol_lab.py run E31`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | estate to the newborns | estate to the parents | no old age | deaths not mended | what it is |
> |---|---|---|---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 10,000 | 10,000 | 10,000 | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | 0 | 0 | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | 0 | 0 | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | 0.2 | 0.2 | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | 16 | 16 | 16 | 16 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | float16 | float16 | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | 16 | 16 | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 8 | 8 | 8 | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | 5 | 5 | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | on | on | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | off | off | off | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | off | off | off | off | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | on | on | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](settings.md#allow_gifting) | off | off | off | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](settings.md#random_decisions) | off | off | off | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | blotto | blotto | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | phase | phase | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | **newborns** | **parents** | uniform | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | 0 | 0 | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | on | on | on | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.5 | 1.5 | 1.5 | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 2 | 2 | 2 | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | on | on | on | **off** | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 2 | 2 | 2 | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 8 | 8 | 8 | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](settings.md#keep_home) | 0.3 | 0.3 | 0.3 | 0.3 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 100 | 100 | **0** | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | off | off | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 10 | 10 | 10 | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | fixed | fixed | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | **on** | **on** | **on** | **on** | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | **parents** | **parents** | **parents** | **parents** | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`self_message`](settings.md#self_message) | own | own | own | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | off | off | off | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | random | random | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | mutual | mutual | mutual | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | none | none | none | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | 0.2 | 0.2 | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | 0.2 | 0.2 | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | 0.1 | 0.1 | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | 20 | 20 | 20 | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 2 | 1 to 2 | 1 to 2 | 1 to 2 | one run per seed |
> | iterations | 3,000 | 3,000 | 3,000 | 3,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B2.
<!-- /runs -->

## What the worlds did

<!-- figure room/measures -->
![Experiment 19's world, given room](../figures/room/measures.svg)

**Experiment 19's world, given room.** Each dot is one world at iteration 3,000: Experiment 19's ten (grey) and Experiment 31's two for each rule. The dashed line is what the author's picture asks: every mark, a diversity score above B2's median world at 3,000 (0.54), a typical distance at least 1.5 times the random twin's, a Gini no lower than B2's least even world at 3,000 (0.47).

> [!example]- How to make this figure
> **Runs.** Experiment 31's eight runs (`python3 gol_lab.py run E31`): B2 at 10,000 tokens with brains mixed at birth and parents as mates, and the estate of the dead paid to the newborns (`B2-10000-06b876-…`) or to the parents (`-b8e9fc-…`), no old age (`-89f5b1-…`) or deaths not mended (`-c58a16-…`), seeds 1 and 2, to 3,000 iterations; and Experiment 19's ten worlds (`B2-10000-bcc1bc-s001` … `-s010`), all read at iteration 3,000 from their archives.
>
> **Data.** From each world's archive at iteration 3,000.
>
> 1. Read each world as `research/atlas/profile.py` reads every world: the seven marks, the four spreads ranked among the frozen reference worlds (their mean is the diversity score), the Gini of the tokens.
> 2. Its typical distance over its random twin's, as `research/atlas/space.py` measures it.
> 3. The lines: every mark; the median diversity, and the least Gini, of B2's ten worlds played on and read at 3,000; a typical distance of 1.5 times the twin's.
>
> **To make it again:** `python3 book_figures.py room`.
<!-- /figure -->

**Every rule gave the world room, and every rule took something else.**
At iteration 3,000:
- **Experiment 19's own worlds** are as before:
  - 6 or 7 marks;
  - diverse, 0.54 to 0.75;
  - uneven, Ginis of 0.44 to 0.57;
  - folded, 1.16 to 1.29 times their twins.
- **The estate to the newborns, or to the parents,** made true spaces:
  - 2.0 to 2.65 times as wide as their twins, at a dimension of about 3.5;
  - every mark of the health bar;
  - about 5,000 agents.

  But the diversity scores fell to 0.32 to 0.41, and the Ginis to 0.42 to
  0.45, a little more even than B2's least even world.
- **No old age** made spaces too, 1.7 to 1.9 times their twins, with every
  mark and very uneven tokens (Ginis of 0.70 and 0.73). Diversity fell to
  0.29 and 0.30.
- **Deaths not mended** went two ways.
  - **Seed 1** came nearest of all: 1.89 times its twin, a diversity score
    of 0.57, a Gini of 0.48. It missed one mark: one agent's loss could cut
    off too much.
  - **Seed 2** missed two marks (a cut and a hub), its diversity (0.39) and
    its unevenness (0.35).

<!-- figure room/parts -->
![What each rule costs of the diversity](../figures/room/parts.svg)

**What each rule costs of the diversity.** The four parts of the diversity score, each world's percentile among the frozen reference worlds: how many doubling classes its agents' tokens, links and ages fill, and its effective number of genotypes per agent. Each dot is one world at iteration 3,000.

> [!example]- How to make this figure
> **Runs.** Experiment 31's eight runs (`python3 gol_lab.py run E31`): B2 at 10,000 tokens with brains mixed at birth and parents as mates, and the estate of the dead paid to the newborns (`B2-10000-06b876-…`) or to the parents (`-b8e9fc-…`), no old age (`-89f5b1-…`) or deaths not mended (`-c58a16-…`), seeds 1 and 2, to 3,000 iterations; and Experiment 19's ten worlds (`B2-10000-bcc1bc-s001` … `-s010`), all read at iteration 3,000 from their archives.
>
> **Data.** From each world's archive at iteration 3,000.
>
> 1. Read each world as `research/atlas/profile.py` reads every world: the seven marks, the four spreads ranked among the frozen reference worlds (their mean is the diversity score), the Gini of the tokens.
>
> **To make it again:** `python3 book_figures.py room`.
<!-- /figure -->

**What each rule cost.** The diversity score is the mean of four parts,
and each rule took a different one:
- **The estates take the tokens.** Paying the estate to births fills the
  world with children, about 5,000 agents for 10,000 tokens, 1.9 tokens
  each. The agents' tokens fill about two doubling classes, against four or
  five in Experiment 19's world, and their links vary little. Almost every
  agent holds the same few tokens: the case the author called unhealthy.
- **No old age takes the ages.** With no deaths of age there are few births,
  and almost every agent is old.
- **No mending takes the ages too,** and leaves the world cuttable, but
  keeps its tokens as spread as Experiment 19's.

## The verdict

**The thesis fails.** No rule had both its worlds meet all four at 3,000.
One world without mending missed by one mark. Room and the rest still pull
apart, even in the steadiest mixed world. What each rule pays is now known:
- the estates pay with poor, alike agents;
- no old age pays with one age;
- no mending pays with cuts.

## What it teaches

- **Half the picture is solved.** With the estates to births, a world of
  mixed brains and parents as mates is a healthy space of about three
  dimensions, with every mark: the network the author's picture asks for.
  What it lacks is in its agents. There are too many of them, each too poor
  for its tokens to differ.
- **The cost is the price of a child.** An estate paid to the newborns makes
  children cheap to have and the world fills with them. The next rule to
  try keeps the estate but asks more of a birth, without a number: a child
  must be born with at least what its parent's neighbours hold, so a world
  can grow only as rich as it is. It is local, needs no constant, and is
  [What holds a world's size](what-holds-a-worlds-size.md)'s missing pull — births that pay more when they would make the
  world poorer.

> [!info] Added later (2026-10-21, [Births as rich as the neighbours](births-as-rich-as-the-neighbours.md))
> The price of a child was not the cost. With the floor, the estate worlds
> were the same ([Births as rich as the neighbours](births-as-rich-as-the-neighbours.md)):
> in a world of paupers a parent's neighbours hold a token or two, so the
> floor asked no more than the ten tokens every child already gets. The
> paupers are not born poor. Children start with ten tokens or more, and
> the game takes most of it within twenty iterations. A world's size is a
> hundred times its births an iteration, and the estates pay for three
> times as many births.

- **The nearest world is one without mending.** Seed 1 missed only a cut. A
  world whose deaths are not mended keeps its tokens and its room; what it
  needs is to stay whole. Mending without folding is a second rule to try
  ([What folds a world](what-folds-a-world.md) traced the folding to the rings a
  death closes).


<!-- turns -->
---

← [The search's best world, in new seeds](the-searchs-best-world-in-new-seeds.md) · [Looking for a healthy, rich world](../README.md#looking-for-a-healthy-rich-world) · [A screen of random worlds](a-screen-of-random-worlds.md) →

**Also about:** [Space and dimension](../README.md#space-and-dimension).

> [!info]- Pages that link here (5)
> [Every setting](settings.md) · [Births as rich as the neighbours](births-as-rich-as-the-neighbours.md) · [A mate that pays its share](a-mate-that-pays-its-share.md) · [The search's best world, in new seeds](the-searchs-best-world-in-new-seeds.md) · [A screen of random worlds](a-screen-of-random-worlds.md)
<!-- /turns -->
