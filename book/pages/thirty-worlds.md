# Thirty worlds

Part II looks at one kind of world, the baseline B1, from every side. It does
so with thirty worlds that differ only in their seed. This chapter introduces
them: every chapter up to [Chapter 26](what-the-baseline-world-is.md), and every chapter of
Part III up to [Chapter 40](what-the-measurements-say-about-open-ended-evolution.md), reads these same thirty runs.

> [!info] The second edition of Parts II and III
> Parts II and III were first written from thirty worlds of 10,000 tokens,
> about 1,300 agents each (Experiment 2). They are written here from thirty
> worlds of 100,000 tokens, about 14,000 agents each (Experiment 10,
> [Chapter 60](worlds-of-100000-tokens.md)): worlds big enough for space, for
> scale and for rare events to show. Where the two sizes differ, the chapters
> give both. The worlds of 10,000 tokens remain the ones Part I teaches with and
> the baseline the experiments of Part IV compare against.

## The runs

<!-- runs E10 -->
> [!info] The runs behind this chapter
> **30 runs.** Every condition below is run once for every seed (1 to 30), for 1,500 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it.
>
> - **baseline** — changes nothing: B1 as it is; 100,000 tokens. Runs `B1-100000-s001` … `B1-100000-s030`.
>
> To make these runs again: `python3 gol_lab.py run E10`, or ▶ in the Book tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | baseline | what it is |
> |---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 100,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | 50, 45, 40, 35, 30 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 30 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | on | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | on | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](settings.md#random_decisions) | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | off | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.0 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 0 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | off | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 0 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 0 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](settings.md#keep_home) | 0.0 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 0 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 1 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`self_message`](settings.md#self_message) | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 30 | one run per seed |
> | iterations | 1,500 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

Thirty runs of the baseline at 100,000 tokens — so a thousand founders of a
hundred tokens each — with seeds 1 to 30, each run for 1,500 iterations.
Every frame was kept, with every decision; every statistic was recorded for
every frame, and the statistics that walk the whole network (bridges, the
core, clustering, distances, the flow's loops) every 25 iterations
([What a run records](frames-and-stats.md)).

Together they took 211 hours of computing, done in a little over two days on
four processor cores, and fill 58 GB. All thirty worlds lived to the end:
where four of the thirty smaller worlds died, none of these did
([Chapter 10](a-worlds-life.md)).

To replicate a figure of Parts II and III, these are the runs you need: on a
computer with Python 3, numpy and networkx and a copy of this repository,
`python3 gol_lab.py run E10` makes them into the folder `GraphOfLifeRuns/`,
and `python3 book_figures.py` then redraws the figures from them. The few
figures that set worlds of different sizes side by side — in Chapters [13](how-ages-change.md),
24, 25 and 33 — also read the thirty worlds of 10,000 tokens
(`python3 gol_lab.py run E02`) or the worlds of every size of
[Chapter 42](how-does-a-worlds-size-follow-its-tokens.md)
(`python3 gol_lab.py run E08`). Each figure says, folded beneath it, which of
the runs it reads and how. A run of 1,500 iterations at this size takes about
seven hours on one core; the thirty worlds of 10,000 tokens take an eighth of
the computing in all, and show most of the same things.

## A first look

One world after 1,500 iterations — every agent a dot, every connection a
line, the dots placed so that joined agents lie near each other. A world of
15,000 agents drawn whole is a tangle, so the picture holds a piece of it:
every agent within a few steps of the world's oldest agent.

<!-- figure shape/world-end -->
![World of seed 1 after the game of iteration 1,499](../figures/shape/world-end.svg)

**World of seed 1 after the game of iteration 1,499.** The world with seed 1 after the game of iteration 1,499, the 2,495 of its 14,870 agents within 25 steps of agent 660,973, the oldest, 806 iterations old — a piece, since a world this large is a tangle when drawn whole: every agent drawn is a dot, every connection between two of them a line. Blue: agents in the core, which is what is left when agents with one connection or none are removed again and again until none remains ([Core, trees and leaves](core-trees-leaves.md)); yellow: the rest of the hanging trees; red: leaves, agents with a single connection. Where a dot is drawn means nothing in itself: a layout (networkx's ForceAtlas2, seed 1) pulls joined agents together and pushes the rest apart.

> [!example]- How to make this figure
> **Runs.** `B1-100000-s001`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read frame `2999` (iteration 1,499, after the game): `ids` and `edges`.
> 2. Peel: remove every agent with one connection or none, again and again, until none is left; the agents never removed are the core. Leaves are agents with exactly one connection.
> 3. In a world of more than 2,500 agents, keep the agents within r steps of the oldest agent, for the largest r that keeps at most 2,500 (`book_chapters.society.shown`).
> 4. Lay the graph out with `networkx.forceatlas2_layout(G, max_iter=200, seed=1)`.
>
> **To make it again:** `python3 book_figures.py shape`.
<!-- /figure -->

The thousand founders on their ring ([Chapter 2](the-world.md)) have become
a network of dense clusters joined by thin paths, with many agents that hang
on by a single connection. [Chapter 17](what-shape-does-the-network-take.md)
takes this shape apart, and [Chapter 25](the-core-of-a-world.md) its core.

Any of the thirty can be opened in the **Viewer** of the Graph of Life app,
which replays a run frame by frame.

## The questions of Part II

- [Chapter 10](a-worlds-life.md) — What does a world do over 1,500
  iterations? Does it settle?
- [Chapter 11](the-first-hundred-iterations.md) — What happens in its first
  hundred iterations?
- [Chapter 12](births-deaths-and-ages.md) — How often are agents born and
  how do they die? How old do they get?
- [Chapter 13](how-ages-change.md) — How do ages and lifespans change over a
  world's life?
- [Chapter 14](how-much-does-the-seed-decide.md) — How much of what a world
  does is decided by its seed?
- [Chapter 15](where-do-the-tokens-go.md) — Where do the tokens go — how
  unequal does a world become?
- [Chapter 16](how-the-game-is-played.md) — How is the game played: where
  do agents stake, who wins the nodes?
- [Chapter 17](what-shape-does-the-network-take.md) — What shape does the
  network take?
- [Chapter 18](genotypes-and-lineages.md) — How long do genotypes last, and
  do lineages take over?
- [Chapter 19](what-the-brains-are-like.md) and
  [Chapter 20](what-agents-say-to-each-other.md) — What are the brains like,
  and what do agents say to each other?
- [Chapter 21](do-the-rich-stay-rich.md) — Do the rich stay rich?
- [Chapter 22](how-the-network-grows.md) — How does the network grow?
- [Chapter 23](like-next-to-like.md) — Are neighbours alike?
- [Chapter 24](how-many-dimensions-does-a-world-have.md) and
  [Chapter 25](the-core-of-a-world.md) — How many dimensions does a world
  have, and its core?
- [Chapter 26](what-the-baseline-world-is.md) — What the baseline world is, in sum.

Five of these chapters (10, 14, 15, 17 and 18) were planned as experiments,
with a thesis written down before the runs of 10,000 tokens; they quote it,
say whether it held there, and test it again on these worlds. The others only
ask.

<!-- turns -->
---

← [What the baseline world is](what-the-baseline-world-is.md) · [The baselines B1 and B2](../README.md#the-baselines-b1-and-b2) · [A world's life](a-worlds-life.md) →
<!-- /turns -->
