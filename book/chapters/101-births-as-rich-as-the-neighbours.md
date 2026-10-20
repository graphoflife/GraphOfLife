# Births as rich as the neighbours

> [!question] Questions of this chapter
> - The estates of the dead paid to births make a world a healthy space but
>   fill it with paupers. Can a rule of birth keep the space and stop the
>   paupers?
> - What does a world do when a child may never be poorer than the agents it
>   is born among?

<!-- experiment E32 -->

## Why ask

[Chapter 100](100-room-for-the-steadiest-world.md) gave Experiment 19's
world each rule that keeps a world's room. The estates of the dead, paid to
the newborns or to the parents, came nearest the network the author's
picture asks for:
- every mark of the health bar;
- a space 2.0 to 2.65 times as wide as its twin, at about three
  dimensions.

But they filled it with children: about 5,000 agents for 10,000 tokens,
1.9 tokens each. Almost every agent held the same few tokens, and the
diversity score fell from Experiment 19's 0.54 to 0.75 to 0.32 to 0.41.

An estate paid to births makes a birth cheap, and nothing stops a world
from making paupers.
[Meta XI](84-meta-11-what-holds-a-worlds-size.md) asked for a pull: births
that cost more when they would make the world poorer. The rule tried here
is that pull, with no number of its own. A child must start with at least
the median holding of its parent's neighbours
([`child_floor`](../notes/settings.md#child_floor) = `neighbours`).
- A parent among rich agents can have a child only by giving it as much as
  they hold.
- A parent among poor ones may have a small child.

The world can grow only as rich as it is.

## The thesis

<!-- thesis E32 -->
> [!quote] The thesis of Experiment 32, written down before any of its runs existed
> **The claim.** Asking a child to start with at least what its parent's neighbours hold keeps the healthy space the estates make and gives back the diversity they cost. With brains mixed at birth, parents as mates and the estate of the dead paid to births, and a child never poorer than the median holding of its parent's neighbours, both worlds of at least one of the two estates (to the newborns, to the parents), at iteration 3,000, meet every mark of the health bar, are more diverse than B2's median world, are at least 1.5 times as wide as their random twins, and are no more even than B2's least even world, B2's ten worlds played on being read at the same iteration.
>
> **Why it would be so.** In Experiment 31 the estates gave Experiment 19's world every mark and a space of about three dimensions, 2.0 to 2.65 times as wide as its twin. But they filled it with children: about 5,000 agents for 10,000 tokens, 1.9 tokens each, whose tokens fill about two doubling classes. The diversity scores fell to 0.32 to 0.41, and the tokens became a little more even than B2's (Ginis of 0.42 to 0.45). The estate makes a birth cheap, and nothing stops a world from making paupers. A child that must start as rich as the agents it is born among stops that, with no number of its own: the world can grow only as rich as it is. Experiment 31's estate worlds of the same seeds are the control, as is Experiment 19's world for the floor alone.
>
> **It holds if:** For at least one estate rule, both worlds (seeds 1 and 2) at 3,000 meet every mark, have a diversity score above the median of B2's ten worlds read at 3,000 (0.54), a typical distance at least 1.5 times their twin's, and a Gini no lower than the least of those ten worlds' (0.47).
>
> **It fails if:** Neither estate rule has both its worlds pass all four at 3,000; the chapter says what the floor changed and what it did not.
<!-- /thesis -->

## The runs

<!-- runs E32 -->
> [!info] The runs behind this chapter
> **6 runs.** Every condition below is run once for every seed (1 to 2), for 3,000 iterations — or until its world dies out. Every setting is that of the baseline **B2** (a world that cannot break) unless the condition changes it.
>
> - **estate to the newborns, a child as rich as its neighbours** — changes `mix_at_birth` = on, `mix_mates` = parents, `redistribution` = newborns, `child_floor` = neighbours; 10,000 tokens. Runs `B2-10000-12215f-s001` … `B2-10000-12215f-s002`.
> - **estate to the parents, a child as rich as its neighbours** — changes `mix_at_birth` = on, `mix_mates` = parents, `redistribution` = parents, `child_floor` = neighbours; 10,000 tokens. Runs `B2-10000-86df9e-s001` … `B2-10000-86df9e-s002`.
> - **a child as rich as its neighbours** — changes `mix_at_birth` = on, `mix_mates` = parents, `child_floor` = neighbours; 10,000 tokens. Runs `B2-10000-b289dd-s001` … `B2-10000-b289dd-s002`.
>
> To make these runs again: `python3 gol_lab.py run E32`, or ▶ in the Book tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | estate to the newborns, a child as rich as its neighbours | estate to the parents, a child as rich as its neighbours | a child as rich as its neighbours | what it is |
> |---|---|---|---|---|
> | [`total_tokens`](../notes/settings.md#total_tokens) | 10,000 | 10,000 | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](../notes/settings.md#n_nodes) | 0 | 0 | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](../notes/settings.md#k_neighbors) | 0 | 0 | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](../notes/settings.md#rewire_p) | 0.2 | 0.2 | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](../notes/settings.md#hidden_layers) | 16 | 16 | 16 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](../notes/settings.md#brain_kind) | float16 | float16 | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](../notes/settings.md#brain_bits) | 16 | 16 | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](../notes/settings.md#message_amount) | 8 | 8 | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](../notes/settings.md#random_input_amount) | 5 | 5 | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](../notes/settings.md#exchange_messages) | on | on | on | Whether agents send and read messages at all. |
> | [`message_prepass`](../notes/settings.md#message_prepass) | off | off | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](../notes/settings.md#allow_handover) | off | off | off | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](../notes/settings.md#allow_revolutions) | on | on | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](../notes/settings.md#allow_gifting) | off | off | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](../notes/settings.md#random_decisions) | off | off | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](../notes/settings.md#prune_after) | blotto | blotto | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](../notes/settings.md#inactive_window) | phase | phase | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](../notes/settings.md#redistribution) | **newborns** | **parents** | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](../notes/settings.md#tokens_created_per_phase) | 0 | 0 | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](../notes/settings.md#mutate_on_replication) | on | on | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](../notes/settings.md#incumbency) | 1.5 | 1.5 | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](../notes/settings.md#birth_links) | 2 | 2 | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](../notes/settings.md#mend_on_death) | on | on | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](../notes/settings.md#prune_needs) | 2 | 2 | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](../notes/settings.md#valence) | 8 | 8 | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](../notes/settings.md#keep_home) | 0.3 | 0.3 | 0.3 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](../notes/settings.md#max_age) | 100 | 100 | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](../notes/settings.md#buy_life) | off | off | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](../notes/settings.md#min_child) | 10 | 10 | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](../notes/settings.md#child_floor) | **neighbours** | **neighbours** | **neighbours** | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](../notes/settings.md#mix_at_birth) | **on** | **on** | **on** | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](../notes/settings.md#mix_mates) | **parents** | **parents** | **parents** | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`self_message`](../notes/settings.md#self_message) | own | own | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](../notes/settings.md#link_search) | off | off | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](../notes/settings.md#link_steering) | random | random | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](../notes/settings.md#link_consent) | mutual | mutual | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](../notes/settings.md#proto) | none | none | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](../notes/settings.md#mutation_probability) | 0.2 | 0.2 | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](../notes/settings.md#mutation_noise_std) | 0.2 | 0.2 | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](../notes/settings.md#mutation_sparsity) | 0.1 | 0.1 | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](../notes/settings.md#extinction_threshold) | 20 | 20 | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 2 | 1 to 2 | 1 to 2 | one run per seed |
> | iterations | 3,000 | 3,000 | 3,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B2.
<!-- /runs -->

<!-- turns -->
---

← [Chapter 100 · Room for the steadiest world](100-room-for-the-steadiest-world.md) · [Contents](../README.md) · [Chapter 60 · Worlds of 100,000 tokens](60-worlds-of-100000-tokens.md) →
<!-- /turns -->
