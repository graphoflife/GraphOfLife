# Who dies, and why

> [!question] Questions of this chapter
> - In how many ways can an agent die, and which of them does a world need?
> - What happens when agents can stake themselves to death — and when they
>   cannot?
> - What does protecting an agent's last token do to who is left to play?
> - Why should every agent die of age?

<!-- experiment E15 -->

## Why ask

[Chapter 47](47-brains-that-see-and-worlds-that-break.md) found that a world
lives as long as its births replace its deaths, and that most of its deaths
were agents cut off by the cull. [Chapter 48](48-a-world-that-cannot-break.md)
built rules under which nobody is cut off. What is left is the other way to
die — starving, by staking one's last token away — and a question the
baseline never had to ask: which deaths does a world *need*?

Deaths matter twice. They make room: a world of fixed tokens has new agents
only where old ones have gone. And they are where selection can act on whole
agents rather than on the brains they carry. But a death can also carry no
information at all — an agent that dies because a link lapsed somewhere, or
because it was born with one token and spent it — and such deaths only add
noise. This chapter asks what each rule of death does, using the pilots that
led to B2, and plans Experiment 15 to test it.

## Ways to die

In the baseline an agent dies in one of four ways
([Births and the four ways to die](../notes/births-and-deaths.md)): it
**starves** — holds no tokens — or it is **cut off** from the largest piece,
after either of the two phases. B2 adds a fifth: it dies of **age**, when it
has lived [`max_age`](../notes/settings.md#max_age) iterations. Winning a
node does not kill anyone. It replaces the node's brain; the agent, its age
and its links stay
([Chapter 39](39-questioning-the-mechanics.md#two-roads-of-inheritance) calls
agents bodies and brains what moves between them).

## Children as missiles

The first pilots of [Chapter 48](48-a-world-that-cannot-break.md)'s rules joined every newborn to its parent and
two of its neighbours and mended every death. The worlds never split — and
still shrank, with 30 to 60% of their agents starving in a single game. The
starving were children: 46% of them were born that very iteration, 83% at
most two iterations before, and they held one or two tokens, which they staked
in full on a neighbour.

A parent that gives a child one token and lets it stake everything spends the
child as a **missile**. If the stake wins the neighbour's node, the node goes
to the child's brain — which is the parent's lineage. If it loses, the lineage
has lost a token. It is a cheap attack, and evolution found it. Mending made
it worse: every starved child's neighbours were joined up, so degrees climbed
to 11–14 links, every agent's stake was split over more neighbours, and more
of them starved. Letting a newborn keep everything at home in its first game
only delayed the missile by a game.

Two rules remove the missile. A child must be worth having:
[`min_child`](../notes/settings.md#min_child) = 10 gives no child to a parent
who would give it fewer than ten tokens. And an agent must not be able to
stake itself to death.

## The last token

The simplest way to stop agents starving is to keep their last token at
home, whatever the brain says. In a pilot with that rule, nobody starved —
and three agents in four held exactly one token (75% at iteration 1,500, and
8% two tokens), against 1% in the baseline's blind world. The game was played
by a fifth of the world.

The reason is that the rule makes an all-in attack free. An agent that
cannot lose its last token can stake everything else on one neighbour and
lose nothing that matters to it; seeing brains do exactly that, and the
agents they attack are left with the token they could not lose. Such
**zombies** cannot have children, and still each take a place.

Without the rule, the same worlds starved: both pilot seeds were down to a
single agent holding all 5,000 tokens by iteration 300.

## A share kept at home

The middle way is to keep a *share* at home:
[`keep_home`](../notes/settings.md#keep_home) = *f* keeps ⌈*f t*⌉ of an
agent's *t* tokens on its own node — never fewer than one — and lets its
brain stake the rest. With *f* = 0.3 an agent with 10 tokens stakes at most
7, one with 3 tokens at most 2, one with a single token nothing. Nobody
starves, because nobody can stake its last token; and an all-in attack is no
longer free, because a third of what an agent has stays where it can defend
it. In pilots with *f* = 0.3, 35 to 61% of the agents held one token — fewer
than with the last token kept, but still many.

A parent keeps at least one token too, so a birth can never starve it.

## Holding one's place

A brain dies too — when the node it sits on is won. In the baseline that
happens to 40 to 90% of the nodes in every game: a brain lasts a game or two
wherever it is, and an agent's body outlives a long series of brains
([Chapter 39](39-questioning-the-mechanics.md#two-roads-of-inheritance)). Meta II
counted it among the obstacles — conquest erases, and nothing can be held
([Chapter 40](40-meta-2.md#what-stands-in-the-way)).

[`incumbency`](../notes/settings.md#incumbency) weighs an agent's stake on its
own node more than anyone else's: with *h* = 2 an attacker must stake more
than twice what the defender keeps at home. In pilots with brains that see,
conquest fell to 15–25% of the nodes in a game, and births came back as the
way brains spread. But the worlds found an exploit: one agent gathered 57% of
all the tokens, staked all of them at home where nobody could outbid them, and
had child after child, while everyone else held a single token — a **queen**.
With *h* = 1.5, as in B2, and a share kept at home, no queen formed in the
pilots (the Gini coefficient of tokens stayed near 0.5), and conquest settled
between a few and about ten per cent of the nodes a game.

## Old age

With a share kept at home and the rules of [Chapter 48](48-a-world-that-cannot-break.md), nothing kills an agent
any more: it cannot starve, it cannot be cut off, and winning its node changes
only its brain. An agent that cannot die holds its place forever — and in a
world of fixed tokens, a world without deaths has no room for births.

So in B2 every agent dies when it reaches age 100
([`max_age`](../notes/settings.md#max_age)); its tokens are shared out like any
estate and its neighbourhood is mended. That makes a world's demography
simple. Every agent lives exactly 100 iterations, so a world keeps its size
only if one agent in a hundred has a child in every iteration — and in the ten
B2 worlds of Experiment 14, from iteration 200 to 1,000, births ran at 1.00
to 1.05 per hundred agents and iteration, while not one agent starved or was
cut off. The population, 800 to 1,800 agents at 5,000 tokens, settles where
enough agents are rich enough to give a child ten tokens.

Old age also answers what [Chapter 47](47-brains-that-see-and-worlds-that-break.md)
left open: there, seeing brains could give up having children, and a world
whose births fell short of its losses drained slowly. Under old age a line
that stops having children loses a body every time one of its agents turns
100, and keeps its place only by winning nodes from lines that still breed.

## The rules of death in B2

| rule | B2 | what it removes |
|---|---|---|
| [`incumbency`](../notes/settings.md#incumbency) | 1.5 | a brain lost in almost every game |
| [`keep_home`](../notes/settings.md#keep_home) | 0.3 | starving; the free all-in attack |
| [`min_child`](../notes/settings.md#min_child) | 10 | children spent as missiles |
| [`max_age`](../notes/settings.md#max_age) | 100 | agents that hold a place forever; a world without births |

## The experiment

Experiment 15 runs B2 at 10,000 tokens for 1,000 iterations, with ten seeds,
and changes one rule of death at a time: only the last token kept
(`keep_home` 0.01, which keeps one token for anyone with up to a hundred),
nothing kept, half kept, children of any size, and no old age.

## The thesis

<!-- thesis E15 -->
> [!quote] The thesis of Experiment 15, written down before any of its runs existed
> **The claim.** How agents die decides who is left to play. In B2, with at least 30% of an agent's tokens kept at home, fewer than half the agents hold a single token after iteration 300; with only the last token kept, more than two in three do; with nothing kept, worlds starve: at least half fall below 100 agents within 1,000 iterations.
>
> **Why it would be so.** An agent that cannot lose its last token can stake everything else on one neighbour at no risk to itself, and brains that see do: in a pilot with only the last token kept, 75% of the agents held exactly one token at iteration 1,500, against 1% in the baseline's blind world. Kept to 30% of its tokens, an agent's stake stays spread: 35–61% at one token in pilots. With nothing kept, agents stake themselves to death and worlds starved to a single agent within 300 iterations in both pilot seeds.
>
> **It holds if:** Median over iterations 300 to 1,000 of the share of agents holding one token (from the frames): below 0.5 in at least 7 of the 10 B2 worlds, above 0.67 in at least 7 of the 10 worlds with keep_home 0.01; and at least 5 of the 10 worlds with keep_home 0 have fewer than 100 agents after some game.
>
> **It fails if:** Any of the three fails: the share at one token does not follow what is kept at home, or worlds without it do not starve.
<!-- /thesis -->

## The runs

<!-- runs E15 -->
> [!info] The runs behind this chapter
> **60 runs.** Every condition below is run once for every seed (1 to 10), for 1,000 iterations — or until its world dies out. Every setting is that of the baseline **B2** (a world that cannot break) unless the condition changes it.
>
> - **B2** — changes nothing: B2 as it is; 10,000 tokens. Runs `B2-10000-s001` … `B2-10000-s010`.
> - **the last token kept** — changes `keep_home` = 0.01; 10,000 tokens. Runs `B2-10000-054b50-s001` … `B2-10000-054b50-s010`.
> - **nothing kept** — changes `keep_home` = 0.0; 10,000 tokens. Runs `B2-10000-575a1d-s001` … `B2-10000-575a1d-s010`.
> - **half kept** — changes `keep_home` = 0.5; 10,000 tokens. Runs `B2-10000-df2615-s001` … `B2-10000-df2615-s010`.
> - **children of any size** — changes `min_child` = 1; 10,000 tokens. Runs `B2-10000-185eb7-s001` … `B2-10000-185eb7-s010`.
> - **no old age** — changes `max_age` = 0; 10,000 tokens. Runs `B2-10000-3ec24f-s001` … `B2-10000-3ec24f-s010`.
>
> To make these runs again: `python3 gol_lab.py run E15`, or ▶ in the Book tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | B2 | the last token kept | nothing kept | half kept | children of any size | no old age | what it is |
> |---|---|---|---|---|---|---|---|
> | [`total_tokens`](../notes/settings.md#total_tokens) | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](../notes/settings.md#n_nodes) | 0 | 0 | 0 | 0 | 0 | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](../notes/settings.md#k_neighbors) | 0 | 0 | 0 | 0 | 0 | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](../notes/settings.md#rewire_p) | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](../notes/settings.md#hidden_layers) | 16 | 16 | 16 | 16 | 16 | 16 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](../notes/settings.md#brain_kind) | float16 | float16 | float16 | float16 | float16 | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](../notes/settings.md#brain_bits) | 16 | 16 | 16 | 16 | 16 | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](../notes/settings.md#message_amount) | 8 | 8 | 8 | 8 | 8 | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](../notes/settings.md#random_input_amount) | 5 | 5 | 5 | 5 | 5 | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](../notes/settings.md#exchange_messages) | on | on | on | on | on | on | Whether agents send and read messages at all. |
> | [`message_prepass`](../notes/settings.md#message_prepass) | off | off | off | off | off | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](../notes/settings.md#allow_handover) | off | off | off | off | off | off | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](../notes/settings.md#allow_revolutions) | on | on | on | on | on | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](../notes/settings.md#allow_gifting) | off | off | off | off | off | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](../notes/settings.md#random_decisions) | off | off | off | off | off | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](../notes/settings.md#prune_after) | blotto | blotto | blotto | blotto | blotto | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](../notes/settings.md#inactive_window) | phase | phase | phase | phase | phase | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](../notes/settings.md#redistribution) | uniform | uniform | uniform | uniform | uniform | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births). |
> | [`tokens_created_per_phase`](../notes/settings.md#tokens_created_per_phase) | 0 | 0 | 0 | 0 | 0 | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](../notes/settings.md#mutate_on_replication) | on | on | on | on | on | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](../notes/settings.md#incumbency) | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](../notes/settings.md#birth_links) | 2 | 2 | 2 | 2 | 2 | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](../notes/settings.md#mend_on_death) | on | on | on | on | on | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](../notes/settings.md#prune_needs) | 2 | 2 | 2 | 2 | 2 | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](../notes/settings.md#valence) | 8 | 8 | 8 | 8 | 8 | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](../notes/settings.md#keep_home) | 0.3 | **0.01** | **0.0** | **0.5** | 0.3 | 0.3 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](../notes/settings.md#max_age) | 100 | 100 | 100 | 100 | 100 | **0** | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`min_child`](../notes/settings.md#min_child) | 10 | 10 | 10 | 10 | **1** | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`mix_at_birth`](../notes/settings.md#mix_at_birth) | off | off | off | off | off | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mutation_probability`](../notes/settings.md#mutation_probability) | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](../notes/settings.md#mutation_noise_std) | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](../notes/settings.md#mutation_sparsity) | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](../notes/settings.md#extinction_threshold) | 20 | 20 | 20 | 20 | 20 | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 10 | 1 to 10 | 1 to 10 | 1 to 10 | 1 to 10 | 1 to 10 | one run per seed |
> | iterations | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B2.
<!-- /runs -->

> [!warning] Waiting for the runs
> The plan, the thesis and the method above were written before any run of
> Experiment 15 existed; the results follow when they are done.

<!-- turns -->
---

← [Chapter 48 · A world that cannot break](48-a-world-that-cannot-break.md) · [Contents](../README.md) · [Chapter 50 · Why evolution stops](50-why-evolution-stops.md) →
<!-- /turns -->
