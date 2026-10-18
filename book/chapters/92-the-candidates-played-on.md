# The candidates, played on

> [!question] Questions of this chapter
> - The worlds that came nearest the author's picture at iteration 1,000 —
>   do they stay there over 10,000 iterations?
> - Which of them live, which fall back to B2's poverty, and which keep
>   evolving?

<!-- experiment E20 -->

## Why ask

A world that meets every mark of the health bar at iteration 1,000, with a
diversity score above B2's median world, is a **candidate**
([Meta X](79-meta-10-healthy-and-rich.md)). One moment says little: mixed
worlds drift in size, and their riches drift with it
([Meta XI](84-meta-11-what-holds-a-worlds-size.md)). So every candidate is
played on to 10,000 iterations, and read by **Meta X's test of a long run**:
at 10,000 it is still alive — it never fell to 20 agents or fewer, where the
lab calls a world extinct — it meets every mark, it is more diverse than B2's
median world at 10,000, and it holds a smaller share of its agents at three
tokens or fewer than the least poor of B2's ten worlds.

Twenty-five candidates were played on as pilot worlds (batch LC), each in a
copy of its own. Nineteen reached 10,000 iterations. Then, on 18 October 2026,
the computer was restarted: the pilots had been running out of sight, in no
chapter of the book, and from then on every simulation of this book runs
where it can be seen — in the lab, started with ▶ beside the chapter that
asks for it.

Six candidates were left. Two of them use only the engine's own settings:
links to strangers by a free reach, forced on the other end
([Chapter 83](83-links-to-strangers.md)), and brains mixed at birth with only
the last token kept at home
([Chapter 82](82-mixing-with-every-other-rule.md)). **Experiment 20** plays
them in the lab from iteration 0. A pilot plays the engine's own game, so a
pilot with a lab world's settings and seed is that lab world — the two mixed
worlds of batch Q turned out to be Experiment 17's worlds of seeds 1 and 2,
agent for agent ([Chapter 80](80-mixed-at-birth-ten-long-worlds.md)) — and
the lab's two worlds should be the pilots' two worlds again, game for game.
The other four candidates use rules that so far exist only in the pilots'
prototype — contests decided by lottery, brain units that cost a token, an
input that tells where a brain came from, a family tag — and follow once those
rules are part of the engine.

## The thesis

<!-- thesis E20 -->
> [!quote] The thesis of Experiment 20, written down before any of its runs existed
> **The claim.** Played by the lab from iteration 0, these two worlds are the pilot worlds of batch LC, game for game: in each, the agents alive and the links between them after every game match the pilot's, iteration by iteration, up to where the pilot stopped (iteration 4,344 for links to strangers, 6,522 for mixing with the last token kept). Played on to 10,000, each is then read by Meta X's test of a long run.
>
> **Why it would be so.** A pilot plays the engine's own game, so a pilot with a lab world's settings and seed is that lab world: the two mixed worlds of batch Q are Experiment 17's worlds of seeds 1 and 2, agent for agent. These two candidates of Meta X were being played on to 10,000 iterations as pilots when the computer was restarted on 18 October 2026; pilots now run only where they can be seen, and both use engine settings only, so the lab can finish them. At iteration 1,000 both met every mark of the health bar: links to strangers by a free reach, forced on the other end (batch K), with 1,931 agents, 49% of them at three tokens or fewer and a diversity score of 0.54; mixing with only the last token kept at home (batch Y), with 1,791 agents, 79% poor and 0.62. Read where the pilots stopped, they held 2,584 agents, 59% poor, at 4,000, and 2,066, 78% poor, at 6,500 — this plan was written after those readings, so the claim is only that the lab plays the same worlds.
>
> **It holds if:** Both worlds' agents and links match their pilot's after every game up to where the pilot stopped.
>
> **It fails if:** A world's agents or links differ from its pilot's at some iteration: the lab plays another world than the pilot did, and the chapter says from where.
<!-- /thesis -->

## The runs

<!-- runs E20 -->
> [!info] The runs behind this chapter
> **2 runs.** Every condition below is run once for every seed (1), for 10,000 iterations — or until its world dies out. Every setting is that of the baseline **B2** (a world that cannot break) unless the condition changes it.
>
> - **links to strangers, a free reach, forced** — changes `link_search` = reach, `link_consent` = forced; 10,000 tokens. Runs `B2-10000-e10e0d-s001` … `B2-10000-e10e0d-s001`.
> - **mixed at birth, the last token kept** — changes `mix_at_birth` = on, `keep_home` = 0.01; 10,000 tokens. Runs `B2-10000-82bd1e-s001` … `B2-10000-82bd1e-s001`.
>
> To make these runs again: `python3 gol_lab.py run E20`, or ▶ in the Book tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | links to strangers, a free reach, forced | mixed at birth, the last token kept | what it is |
> |---|---|---|---|
> | [`total_tokens`](../notes/settings.md#total_tokens) | 10,000 | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](../notes/settings.md#n_nodes) | 0 | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](../notes/settings.md#k_neighbors) | 0 | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](../notes/settings.md#rewire_p) | 0.2 | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](../notes/settings.md#hidden_layers) | 16 | 16 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](../notes/settings.md#brain_kind) | float16 | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](../notes/settings.md#brain_bits) | 16 | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](../notes/settings.md#message_amount) | 8 | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](../notes/settings.md#random_input_amount) | 5 | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](../notes/settings.md#exchange_messages) | on | on | Whether agents send and read messages at all. |
> | [`message_prepass`](../notes/settings.md#message_prepass) | off | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](../notes/settings.md#allow_handover) | off | off | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](../notes/settings.md#allow_revolutions) | on | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](../notes/settings.md#allow_gifting) | off | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](../notes/settings.md#random_decisions) | off | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](../notes/settings.md#prune_after) | blotto | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](../notes/settings.md#inactive_window) | phase | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](../notes/settings.md#redistribution) | uniform | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](../notes/settings.md#tokens_created_per_phase) | 0 | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](../notes/settings.md#mutate_on_replication) | on | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](../notes/settings.md#incumbency) | 1.5 | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](../notes/settings.md#birth_links) | 2 | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](../notes/settings.md#mend_on_death) | on | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](../notes/settings.md#prune_needs) | 2 | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](../notes/settings.md#valence) | 8 | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](../notes/settings.md#keep_home) | 0.3 | **0.01** | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](../notes/settings.md#max_age) | 100 | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](../notes/settings.md#buy_life) | off | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](../notes/settings.md#min_child) | 10 | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`mix_at_birth`](../notes/settings.md#mix_at_birth) | off | **on** | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](../notes/settings.md#mix_mates) | any | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`link_search`](../notes/settings.md#link_search) | **reach** | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](../notes/settings.md#link_steering) | random | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](../notes/settings.md#link_consent) | **forced** | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`mutation_probability`](../notes/settings.md#mutation_probability) | 0.2 | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](../notes/settings.md#mutation_noise_std) | 0.2 | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](../notes/settings.md#mutation_sparsity) | 0.1 | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](../notes/settings.md#extinction_threshold) | 20 | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 | 1 | one run per seed |
> | iterations | 10,000 | 10,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B2.
<!-- /runs -->

## What else will be read

- **Where each candidate went**, from iteration 1,000 to 10,000, on the map
  of riches and diversity, beside B2's worlds and Experiment 17's.
- **Which candidates held**, which fell back, and which died out on the way —
  three of the nineteen finished pilots fell to a single agent and refilled
  their world from it, which the lab would have stopped as extinct.
- **Room**: the typical distance over the random twin's at 1,000 and at
  10,000 ([Chapter 87](87-what-folds-a-world.md)).
- **Does evolution go on?** The brains of 5,000 against those of 2,000, and
  of 10,000 against those of 5,000, a lifetime long.

**Related notes:** [Meta X · Healthy and rich](79-meta-10-healthy-and-rich.md) ·
[Meta XI · What holds a world's size](84-meta-11-what-holds-a-worlds-size.md) ·
[Chapter 80 · Mixed at birth: ten long worlds](80-mixed-at-birth-ten-long-worlds.md) ·
[Chapter 82 · Brains mixed at birth, with every other rule](82-mixing-with-every-other-rule.md) ·
[Chapter 83 · Links to strangers](83-links-to-strangers.md) ·
[Chapter 87 · What folds a world](87-what-folds-a-world.md) ·
[Chapter 93 · Five more seeds for the rules that held](93-five-more-seeds-for-the-rules-that-held.md)

<!-- turns -->
---

← [Chapter 81 · Five mixed worlds, fifty thousand iterations](81-five-mixed-worlds-fifty-thousand-iterations.md) · [Contents](../README.md) · [Chapter 93 · Five more seeds for the rules that held](93-five-more-seeds-for-the-rules-that-held.md) →
<!-- /turns -->
