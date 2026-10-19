# Mixed at birth, with parents as mates

> [!question] Questions of this chapter
> - Mixed worlds die out because their agents stop having children. Is that
>   because a mate spreads its brain for free?
> - If only an agent that has a child of its own may be a mate, does a mixed
>   world stay alive — and stay rich?

<!-- experiment E19 -->

## Why ask

Brains mixed at birth made the richest healthy worlds the book has found
([Chapter 80](80-mixed-at-birth-ten-long-worlds.md)) — and mortal ones: three
of the five that met every mark at 10,000 iterations died out before 40,000
([Chapter 81](81-five-mixed-worlds-fifty-thousand-iterations.md)). Each died
the same way. Every agent lives its hundred iterations, a world has to see one
birth per hundred agents every iteration to keep its size, and these worlds
fell below that line and stayed below, though many of their agents could have
afforded a child.

[Meta XI](84-meta-11-what-holds-a-worlds-size.md) offers a reason. A newborn's
mixed brain is half its parent's and half a mate's — a neighbour, drawn at
random, who pays nothing for the child. So an agent that never breeds still
spreads its brain through other agents' children, while the parent pays the
whole price for half a brain's worth of offspring. Selection then rewards
keeping tokens over spending them on children, and nothing stops the
reluctance at the point where the world can no longer replace its dead.

If that is the reason, it can be taken away: let only a parent — a neighbour
with a living child of its own — be a mate. Being a mate is then no free ride
for an agent that never breeds.

## What the pilots found

The engine's setting [`mix_mates`](../notes/settings.md#mix_mates) = `parents`
does this; a parent with no such neighbour gives its newborn a copy of its own
brain. Four pilot worlds of batch MP played it, seeds 1 to 4, beside
Experiment 17's mixed worlds of the same seeds:

| seed | agents at 5,000, any mate / parents | 3 tokens or fewer at 5,000 | births per hundred able agents an iteration | spread of the logarithm of the size, 500 to 5,000 |
|---|---|---|---|---|
| 1 | 1,117 / 1,531 | 35% / 38% | 3.3 / 4.7 | 0.19 / 0.09 |
| 2 | 893 / 932 | 31% / 23% | 2.1 / 3.5 | 0.34 / 0.29 |
| 3 | 285 / 1,709 | 6% / 49% | 2.9 / 5.3 | 0.63 / 0.25 |
| 4 | 744 / 1,983 | 38% / 54% | 4.3 / 7.5 | 0.47 / 0.12 |

With parents as mates the agents bred more eagerly in every world, and every
world held its size more steadily — but the eagerness rose only by a half or
so, three of the four worlds ended poorer than their twins, and one still fell
from 2,000 agents to 900. Taking the free mate away does what the reason says
it should; whether it does enough to keep a world alive over many thousands of
iterations, four pilots of 5,000 cannot say.

## The experiment

Experiment 19 runs B2 at 10,000 tokens with brains mixed at birth and parents
as mates, ten seeds, for 20,000 iterations each — twice as long as Experiment
17, long enough for the deaths of Experiment 18 to have begun. It records as
the lab records every run: the graph statistics every 25 iterations and a
whole world every 500, so that each world can be read at any of those moments
as Meta X reads every world, and its brains of one era played against those of
another. Experiment 17's ten worlds, with any neighbour as a mate, are the
comparison.

## The thesis

<!-- thesis E19 -->
> [!quote] The thesis of Experiment 19, written down before any of its runs existed
> **The claim.** Mixed worlds die out because a mate spreads its brain for free. When only a neighbour with a living child of its own may be a mate, a world whose brains are mixed at birth does not breed itself to death: of ten worlds played for 20,000 iterations, none dies out, and the median world ends richer than any of B2's.
>
> **Why it would be so.** In Experiment 17 one mixed world of ten died out by iteration 10,000, and in Experiment 18 three of the five that had met every mark died out between 17,078 and 37,722: births fell below the one per hundred agents an iteration that replaces the dead, while a third to three quarters of the agents could have had a child. A mate passes half of its brain to a child it pays nothing for, so an agent that never breeds still spreads through other agents' children, and selection rewards keeping tokens over spending them on children (Meta XI). With only parents as mates (batch MP: four pilot worlds, seeds 1 to 4, 5,000 iterations), the agents that could have a child had one at 3.5 to 7.5 per hundred an iteration, against 2.1 to 4.3 in Experiment 17's worlds of the same seeds; each world held its size more steadily than its twin (the spread of the logarithm of its size over iterations 500 to 5,000: 0.09, 0.29, 0.25 and 0.12, against 0.19, 0.34, 0.63 and 0.47), at 760 to 2,200 agents, and 23% to 54% of its agents held three tokens or fewer at 5,000, against 66% to 79% in B2. But one pilot world fell from 2,000 agents to 900, the eagerness rose only a little, and three of the four ended poorer than their twins: the free mate may be one cause among others, and taking it away costs some riches.
>
> **It holds if:** No world of the ten ends before iteration 20,000 (the lab stops a world when 20 agents or fewer are left), and at iteration 20,000 the median of the ten worlds' shares of agents holding three tokens or fewer is below 0.66 — the least of B2's ten worlds at 10,000 iterations.
>
> **It fails if:** A world dies out, or the median share is 0.66 or more: taking away the free mate does not keep a mixed world alive and rich.
<!-- /thesis -->

## The runs

<!-- runs E19 -->
> [!info] The runs behind this chapter
> **10 runs.** Every condition below is run once for every seed (1 to 10), for 20,000 iterations — or until its world dies out. Every setting is that of the baseline **B2** (a world that cannot break) unless the condition changes it.
>
> - **mixed at birth, parents as mates** — changes `mix_at_birth` = on, `mix_mates` = parents; 10,000 tokens. Runs `B2-10000-bcc1bc-s001` … `B2-10000-bcc1bc-s010`.
>
> To make these runs again: `python3 gol_lab.py run E19`, or ▶ in the Book tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | mixed at birth, parents as mates | what it is |
> |---|---|---|
> | [`total_tokens`](../notes/settings.md#total_tokens) | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](../notes/settings.md#n_nodes) | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](../notes/settings.md#k_neighbors) | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](../notes/settings.md#rewire_p) | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](../notes/settings.md#hidden_layers) | 16 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](../notes/settings.md#brain_kind) | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](../notes/settings.md#brain_bits) | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](../notes/settings.md#message_amount) | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](../notes/settings.md#random_input_amount) | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](../notes/settings.md#exchange_messages) | on | Whether agents send and read messages at all. |
> | [`message_prepass`](../notes/settings.md#message_prepass) | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](../notes/settings.md#allow_handover) | off | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](../notes/settings.md#allow_revolutions) | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](../notes/settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](../notes/settings.md#random_decisions) | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](../notes/settings.md#prune_after) | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](../notes/settings.md#inactive_window) | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](../notes/settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](../notes/settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](../notes/settings.md#mutate_on_replication) | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](../notes/settings.md#incumbency) | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](../notes/settings.md#birth_links) | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](../notes/settings.md#mend_on_death) | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](../notes/settings.md#prune_needs) | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](../notes/settings.md#valence) | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](../notes/settings.md#keep_home) | 0.3 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](../notes/settings.md#max_age) | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](../notes/settings.md#buy_life) | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](../notes/settings.md#min_child) | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`mix_at_birth`](../notes/settings.md#mix_at_birth) | **on** | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](../notes/settings.md#mix_mates) | **parents** | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`link_search`](../notes/settings.md#link_search) | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](../notes/settings.md#link_steering) | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](../notes/settings.md#link_consent) | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](../notes/settings.md#proto) | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](../notes/settings.md#mutation_probability) | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](../notes/settings.md#mutation_noise_std) | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](../notes/settings.md#mutation_sparsity) | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](../notes/settings.md#extinction_threshold) | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 10 | one run per seed |
> | iterations | 20,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B2.
<!-- /runs -->

## What the ten worlds did

<!-- figure mixparents/sizes -->
![Agents alive, world by world](../figures/mixparents/sizes.svg)

**Agents alive, world by world.** Each line is one world, its agents after each game (means over 50 iterations). Orange: the ten worlds of Experiment 19, whose mates are drawn only among neighbours with a living child; grey: Experiment 17's ten, any neighbour a mate, to 10,000 iterations, and the five of them Experiment 18 played on.

> [!example]- How to make this figure
> **Runs.** The ten runs of Experiment 19 — B2 at 10,000 tokens with brains mixed at birth and the mate drawn only among neighbours with a living child (`mix_mates` = `parents`), seeds 1 to 10, 20,000 iterations (`python3 gol_lab.py run E19`) — beside Experiment 17's ten runs with any neighbour as mate, to 10,000 iterations, and the five of them Experiment 18 played on, read to 20,000.
>
> **Data.** From each run's `stats.jsonl`.
>
> 1. Read the agents after each game (`nodes`, phase 2).
> 2. Average over stretches of 50 iterations.
>
> **To make it again:** `python3 book_figures.py mixparents`.
<!-- /figure -->

**Every world lived, and settled.** None of the ten came near dying out. After
their first 2,000 iterations they held between 760 and 2,670 agents, and from
iteration 15,000 on each held 2,030 to 2,350 on average. Experiment 17's
worlds, with any neighbour as a mate, wandered between 20 and 2,500 agents,
and two of them had died out by iteration 20,000: one before 10,000, and one
of the five that Experiment 18 played on. Two grey lines run with the orange
ones from 10,000 on: the two worlds of Experiment 18 that lived to 50,000. By
20,000 they held 2,170 and 1,880 agents, 62% and 54% of them poor — where
these ten are. Parents as mates bring every world, and soon, to where only the
survivors of Experiment 18 arrived.

<!-- figure mixparents/poor -->
![Agents holding three tokens or fewer](../figures/mixparents/poor.svg)

**Agents holding three tokens or fewer.** Each band is the worlds of one experiment — the median, the middle half and nine in ten — every 500 iterations: the share of their agents holding one, two or three tokens. Grey: Experiment 17's ten worlds to 10,000, and the five Experiment 18 played on; a world that died out leaves the band when it does.

> [!example]- How to make this figure
> **Runs.** The ten runs of Experiment 19 — B2 at 10,000 tokens with brains mixed at birth and the mate drawn only among neighbours with a living child (`mix_mates` = `parents`), seeds 1 to 10, 20,000 iterations (`python3 gol_lab.py run E19`) — beside Experiment 17's ten runs with any neighbour as mate, to 10,000 iterations, and the five of them Experiment 18 played on, read to 20,000.
>
> **Data.** From each run's whole worlds kept every 500 iterations (`archive/world_<iteration>.npz`).
>
> 1. Read every agent's tokens in each whole world.
> 2. Count those holding three or fewer, as a share of all.
>
> **To make it again:** `python3 book_figures.py mixparents`.
<!-- /figure -->

**Six in ten are poor.** At iteration 20,000, 55% to 66% of each world's
agents hold three tokens or fewer. That is richer than any of B2's worlds —
the least poor of its ten held 66% at 10,000 iterations — but far poorer
than Experiment 17's mixed worlds, whose median world held 15% at 10,000,
and as poor as only the poorest two of them.

**As diverse as mixed worlds.** Their diversity score at 20,000 is 0.59 at
the median (0.52 to 0.69) — as high as Experiment 17's worlds at 10,000
(0.58), and above B2's ten worlds played on to 10,000 (0.49). Part of that is
built in: every mixed birth makes a new genotype
([Chapter 82](82-mixing-with-every-other-rule.md)).

<!-- figure mixparents/breeding -->
![Who could have a child, and who has one](../figures/mixparents/breeding.svg)

**Who could have a child, and who has one.** Left: the share of the agents that could have a child — 11 tokens or more and fewer than 8 links. Right: the births of the next 50 iterations, per iteration, per hundred such agents, on a logarithmic axis. Each band is an experiment's worlds every 500 iterations — the median, the middle half and nine in ten.

> [!example]- How to make this figure
> **Runs.** The ten runs of Experiment 19 — B2 at 10,000 tokens with brains mixed at birth and the mate drawn only among neighbours with a living child (`mix_mates` = `parents`), seeds 1 to 10, 20,000 iterations (`python3 gol_lab.py run E19`) — beside Experiment 17's ten runs with any neighbour as mate, to 10,000 iterations, and the five of them Experiment 18 played on, read to 20,000.
>
> **Data.** From each world kept whole every 500 iterations, and each run's births (`research/oee/checks/e17_breeding.py`, `GraphOfLifeRuns/.pilots/e17_breeding.json` and `e19_breeding.json`).
>
> 1. In each whole world, count the agents with 11 tokens or more and fewer than 8 links.
> 2. Read the births of the 50 reproduction phases that follow.
> 3. Divide their mean per iteration by the agents counted.
>
> **To make it again:** `python3 book_figures.py mixparents`.
<!-- /figure -->

**More eager, fewer able.** With parents as mates, an agent that could
afford a child had one 6.0 times per hundred an iteration over the first
10,000 iterations (the median of every world and moment kept), against 3.0
with any mate — the rise the pilots showed — and 6.5 over the next 10,000.
And the eager fill their world: only a sixth of the agents could afford a
child at the median moment, against a third with any mate. This is Meta XI's
rule at work: how eagerly agents breed sets how full a world becomes, and a
full world is a poor one.

<!-- figure mixparents/evolution -->
![Do later brains beat earlier ones?](../figures/mixparents/evolution.svg)

**Do later brains beat earlier ones?** Each dot is a world: the share of a world the later era's brains hold after 100 iterations against the earlier era's, the mean of two contests, one in each era's world; the bar is the median world. 1,000/500: the brains of iteration 1,000 against those of 500. Orange: Experiment 19; grey: Experiment 17, any neighbour a mate.

> [!example]- How to make this figure
> **Runs.** The ten runs of Experiment 19 — B2 at 10,000 tokens with brains mixed at birth and the mate drawn only among neighbours with a living child (`mix_mates` = `parents`), seeds 1 to 10, 20,000 iterations (`python3 gol_lab.py run E19`) — beside Experiment 17's ten runs with any neighbour as mate, to 10,000 iterations, and the five of them Experiment 18 played on, read to 20,000.
>
> **Data.** From each run's whole worlds (`GraphOfLifeRuns/.pilots/e19_T100.json` and `e17_T100.json`).
>
> 1. Seat the brains of the two eras on alternate agents of each era's world and play it on for 100 iterations (`research/oee/checks/rejudge.py`).
> 2. Count the agents the later era's brains hold at the end, as a share of all.
>
> **To make it again:** `python3 book_figures.py mixparents`.
<!-- /figure -->

**Evolution goes on, as it did with any mate.** Played for a lifetime, the
brains of 1,000 beat those of 500 in every world (a median of 0.78 of a
contested world, against 0.84 with any mate), those of 5,000 beat those of
2,000 in every world (0.65, against 0.66), those of 10,000 beat those of 5,000
in eight of the ten (0.56, against 0.54), and those of 20,000 beat those of
10,000 in nine, the tenth all but tied at 0.495 (a median of 0.55): slowing,
but not stopping.

## The verdict

<!-- figure mixparents/verdict -->
![Experiment 19 at iteration 20,000](../figures/mixparents/verdict.svg)

**Experiment 19 at iteration 20,000.** Each dot is a world at iteration 20,000, read from the whole world kept then. Left: the share of its agents holding three tokens or fewer; the thesis asked that the median of the ten be below 0.66, the least of B2's ten worlds at 10,000 iterations. Right: how many of the seven marks of [Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md)'s health bar each world meets.

> [!example]- How to make this figure
> **Runs.** The ten runs of Experiment 19 — B2 at 10,000 tokens with brains mixed at birth and the mate drawn only among neighbours with a living child (`mix_mates` = `parents`), seeds 1 to 10, 20,000 iterations (`python3 gol_lab.py run E19`) — beside Experiment 17's ten runs with any neighbour as mate, to 10,000 iterations, and the five of them Experiment 18 played on, read to 20,000.
>
> **Data.** From each run's whole world kept at 20,000 (`archive/world_020000.npz`) and `GraphOfLifeRuns/.pilots/e19_health.json` (`research/oee/checks/e17_health.py --family B2-10000-bcc1bc`).
>
> 1. Count the agents holding three tokens or fewer, as a share of all.
> 2. Measure the seven marks as `research/atlas/profile.py` does.
>
> **To make it again:** `python3 book_figures.py mixparents`.
<!-- /figure -->

**The thesis holds.** No world ended before iteration 20,000, and at 20,000
the median world holds 60% of its agents at three tokens or fewer, below the
line of 0.66. Every world meets six of the seven marks of the health bar — and
did at every moment kept, from 500 on, seven at nearly a quarter of them — and
misses the one B2's own worlds miss: a dimension between 2.3 and 4 (they read
4.1 to 4.3).

Two things the thesis did not ask should be said beside it. Experiment 18's
worlds died between iterations 17,078 and 37,722, so 20,000 iterations cover
only the start of the span in which mixed worlds died; but none of these ten
shows what came before those deaths — a world that shrinks while many of its
agents could breed. And the riches the thesis's line asked for are B2's
riches, not mixing's: these worlds are richer than any of B2's, not rich.

## What it teaches

- **The free mate was a cause.** Take it away, and agents breed twice as
  eagerly as with any mate, and no world of ten breeds itself towards its
  end in 20,000 iterations.
- **But reluctance was also what made mixed worlds rich.** The eager fill
  their world to about 2,000 agents, six in ten of them poor, and its network
  becomes B2's, with six marks of the bar and a dimension above four — though
  its agents still differ as much as in Experiment 17's worlds. Life and
  riches pull apart again.
- **What Meta XI asked for is still missing**: a pull that makes having a
  child pay more when few have them and less when many do. Parents as mates
  raise eagerness everywhere, at every size, and so set the world at one size
  — larger than mixing's, smaller than B2's.

**Related notes:** [Chapter 80 · Mixed at birth: ten long worlds](80-mixed-at-birth-ten-long-worlds.md) ·
[Chapter 81 · Five mixed worlds, fifty thousand iterations](81-five-mixed-worlds-fifty-thousand-iterations.md) ·
[Meta XI · What holds a world's size](84-meta-11-what-holds-a-worlds-size.md) ·
[Every setting](../notes/settings.md#mix_mates)

<!-- turns -->
---

← [Chapter 82 · Brains mixed at birth, with every other rule](82-mixing-with-every-other-rule.md) · [Contents](../README.md) · [Chapter 86 · Eight switches at once](86-eight-switches-at-once.md) →
<!-- /turns -->
