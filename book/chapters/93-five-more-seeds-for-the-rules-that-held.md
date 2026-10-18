# Five more seeds for the rules that held

> [!question] Questions of this chapter
> - A few candidate worlds stayed healthy, diverse and richer than B2's for
>   10,000 iterations. Was it their rules, or their luck?
> - Does any rule make a mixed world hold more often than mixing alone?

<!-- experiment E21 -->

## Why ask

Of the candidates played on to 10,000 iterations
([Chapter 92](92-the-candidates-played-on.md)), a few passed Meta X's test of
a long run — alive, every mark of the health bar met, more diverse than B2's
median world, and poorer than none of B2's worlds. Three rules stood out, all
of them with brains mixed at birth:

| rule, with mixing | seed | agents at 10,000 | agents with 3 tokens or fewer | diversity score |
|---|---|---|---|---|
| a fast change of brains (mutation probability 0.8) | 1 | 1,244 | 25% | 0.66 |
| a fast change of brains (mutation probability 0.8) | 2 | 407 | 2% | 0.75 |
| no random inputs | 2 | 372 | 4% | 0.63 |
| no revolutions | 2 | 948 | 24% | 0.61 |

But a mixed world's size drifts, and its riches drift with it
([Meta XI](84-meta-11-what-holds-a-worlds-size.md)): plain mixing passed the
same test in five of Experiment 17's ten worlds
([Chapter 80](80-mixed-at-birth-ten-long-worlds.md)), and no revolutions
failed it in its own seed 1 (2,221 agents, 64% poor). One world can be luck.
So Meta X's second step gives each rule that held **five seeds more, from the
start** — seeds 3 to 7, which no batch has used for these rules — and sets
them beside Experiment 17's worlds of the same seeds, plain mixing, which
passed in three of the five.

A fourth rule held in one seed too: mixing with an input that tells each
agent where its brain came from. That input exists so far only in the pilots'
prototype; it joins this experiment once it is part of the engine.

## The thesis

<!-- thesis E21 -->
> [!quote] The thesis of Experiment 21, written down before any of its runs existed
> **The claim.** A rule whose candidate held over 10,000 iterations holds again in new worlds: for each of three rules — brains mixed at birth with a fast change of brains (a mutation probability of 0.8), with no random inputs, and with no revolutions — at least four of five new worlds (seeds 3 to 7) live to iteration 10,000 and there meet every mark of the health bar, have a diversity score above that of B2's median world at 10,000 (0.49), and hold a smaller share of their agents at three tokens or fewer than the least poor of B2's ten worlds (0.66).
>
> **Why it would be so.** Meta X's candidates were played on to 10,000 iterations as pilots (batch LC), and by Meta X's test — it lives, meets every mark, is more diverse than B2's median world and poorer than none of B2's — some held: mixing with a fast change of brains in both its seeds (1,244 agents, 25% at three tokens or fewer, a diversity score of 0.66; 407 agents, 2%, 0.75), mixing without random inputs in seed 2 (372 agents, 4%, 0.63), mixing without revolutions in seed 2 (948 agents, 24%, 0.61; its seed 1 fell back, 2,221 agents, 64%). Plain mixing held in five of Experiment 17's ten worlds, three of them among seeds 3 to 7. One world can be luck: Meta X's second step gives each rule that held five seeds more, from the start. Mixed worlds' size drifts, and their riches go with it (Meta XI), so a rule earns its place only if it holds more often than mixing alone. The control is Experiment 17's worlds of the same seeds.
>
> **It holds if:** For a rule: at least four of its five worlds live to 10,000 and pass Meta X's test there.
>
> **It fails if:** For a rule: three or fewer pass — it does no better than mixing alone, which passed in three of the same five seeds.
<!-- /thesis -->

## The runs

<!-- runs E21 -->
> [!info] The runs behind this chapter
> **20 runs.** Every condition below is run once for every seed (3 to 7), for 10,000 iterations — or until its world dies out. Every setting is that of the baseline **B2** (a world that cannot break) unless the condition changes it.
>
> - **mixed at birth** — changes `mix_at_birth` = on; 10,000 tokens. Runs `B2-10000-4af4a3-s003` … `B2-10000-4af4a3-s007`.
> - **mixed at birth, fast brain change** — changes `mix_at_birth` = on, `mutation_probability` = 0.8; 10,000 tokens. Runs `B2-10000-0df8be-s003` … `B2-10000-0df8be-s007`.
> - **mixed at birth, no random inputs** — changes `mix_at_birth` = on, `random_input_amount` = 0; 10,000 tokens. Runs `B2-10000-1109f3-s003` … `B2-10000-1109f3-s007`.
> - **mixed at birth, no revolutions** — changes `mix_at_birth` = on, `allow_revolutions` = off; 10,000 tokens. Runs `B2-10000-1f9b27-s003` … `B2-10000-1f9b27-s007`.
>
> To make these runs again: `python3 gol_lab.py run E21`, or ▶ in the Book tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | mixed at birth | mixed at birth, fast brain change | mixed at birth, no random inputs | mixed at birth, no revolutions | what it is |
> |---|---|---|---|---|---|
> | [`total_tokens`](../notes/settings.md#total_tokens) | 10,000 | 10,000 | 10,000 | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](../notes/settings.md#n_nodes) | 0 | 0 | 0 | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](../notes/settings.md#k_neighbors) | 0 | 0 | 0 | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](../notes/settings.md#rewire_p) | 0.2 | 0.2 | 0.2 | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](../notes/settings.md#hidden_layers) | 16 | 16 | 16 | 16 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](../notes/settings.md#brain_kind) | float16 | float16 | float16 | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](../notes/settings.md#brain_bits) | 16 | 16 | 16 | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](../notes/settings.md#message_amount) | 8 | 8 | 8 | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](../notes/settings.md#random_input_amount) | 5 | 5 | **0** | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](../notes/settings.md#exchange_messages) | on | on | on | on | Whether agents send and read messages at all. |
> | [`message_prepass`](../notes/settings.md#message_prepass) | off | off | off | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](../notes/settings.md#allow_handover) | off | off | off | off | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](../notes/settings.md#allow_revolutions) | on | on | on | **off** | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](../notes/settings.md#allow_gifting) | off | off | off | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](../notes/settings.md#random_decisions) | off | off | off | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](../notes/settings.md#prune_after) | blotto | blotto | blotto | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](../notes/settings.md#inactive_window) | phase | phase | phase | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](../notes/settings.md#redistribution) | uniform | uniform | uniform | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](../notes/settings.md#tokens_created_per_phase) | 0 | 0 | 0 | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](../notes/settings.md#mutate_on_replication) | on | on | on | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](../notes/settings.md#incumbency) | 1.5 | 1.5 | 1.5 | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](../notes/settings.md#birth_links) | 2 | 2 | 2 | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](../notes/settings.md#mend_on_death) | on | on | on | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](../notes/settings.md#prune_needs) | 2 | 2 | 2 | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](../notes/settings.md#valence) | 8 | 8 | 8 | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](../notes/settings.md#keep_home) | 0.3 | 0.3 | 0.3 | 0.3 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](../notes/settings.md#max_age) | 100 | 100 | 100 | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](../notes/settings.md#buy_life) | off | off | off | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](../notes/settings.md#min_child) | 10 | 10 | 10 | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`mix_at_birth`](../notes/settings.md#mix_at_birth) | **on** | **on** | **on** | **on** | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](../notes/settings.md#mix_mates) | any | any | any | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`link_search`](../notes/settings.md#link_search) | off | off | off | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](../notes/settings.md#link_steering) | random | random | random | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](../notes/settings.md#link_consent) | mutual | mutual | mutual | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`mutation_probability`](../notes/settings.md#mutation_probability) | 0.2 | **0.8** | 0.2 | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](../notes/settings.md#mutation_noise_std) | 0.2 | 0.2 | 0.2 | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](../notes/settings.md#mutation_sparsity) | 0.1 | 0.1 | 0.1 | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](../notes/settings.md#extinction_threshold) | 20 | 20 | 20 | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 3 to 7 | 3 to 7 | 3 to 7 | 3 to 7 | one run per seed |
> | iterations | 10,000 | 10,000 | 10,000 | 10,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B2.
<!-- /runs -->

## What else will be read

- **Each world through its life**: its size and riches, and when it passes
  Meta X's test and when it does not.
- **Room**: the typical distance over the random twin's
  ([Chapter 87](87-what-folds-a-world.md)) — whether any rule that holds a
  world rich also keeps it a space.
- **Does evolution go on?** The brains of 1,000, 5,000 and 10,000 iterations
  against those before them, a lifetime long.

**Related notes:** [Chapter 92 · The candidates, played on](92-the-candidates-played-on.md) ·
[Meta X · Healthy and rich](79-meta-10-healthy-and-rich.md) ·
[Meta XI · What holds a world's size](84-meta-11-what-holds-a-worlds-size.md) ·
[Chapter 80 · Mixed at birth: ten long worlds](80-mixed-at-birth-ten-long-worlds.md) ·
[Chapter 82 · Brains mixed at birth, with every other rule](82-mixing-with-every-other-rule.md)

<!-- turns -->
---

← [Chapter 92 · The candidates, played on](92-the-candidates-played-on.md) · [Contents](../README.md) · [Chapter 74 · Meta VI · Every world side by side](74-meta-6-every-world.md) →
<!-- /turns -->
