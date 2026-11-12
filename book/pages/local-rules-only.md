# Local rules only

> [!question] Questions of this page
> - Which rules of a world act on all of it at once, and why does that matter?
> - What would each of them be if it could act only where it is?
> - What did round 3 of the screen of random worlds already show about two
>   local rules it drew: mending a death, and [estates](measuring-lives.md#the-estate-of-the-dead "The tokens a dead agent leaves, and who gets them.") that flow to the nearest
>   living?
> - How would the three local rules be tried on round 3's best worlds, and
>   what do we expect?

## In short

Almost every rule of Graph of Life acts only where it is, but three act on a
whole world at once: the cull, the [share-out](b1-lives-and-wealth.md#gains-losses-and-the-share-out "How a cleanup deals the tokens of the dead out among the living: in the baseline, each token to a survivor drawn at random.") and the one random stream. This
page plans to make each of them local, one at a time, on six of round 3's
candidate worlds, each against its own run. Round 3 already drew two local
rules. Mending a death cut the [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") a [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") culls from 11.4% to 3.5%, but not
the big breaks, and it folded the worlds; estates that flow to the nearest
living pooled wealth, so that the richest [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") held 2.7% of all tokens against
0.6%. The draft thesis expects random streams of their own to change nothing,
local estates to make worlds more unequal, and pieces that live on to break
round 3's worlds into islands, since nothing can ever join two pieces again.

<!-- figure localrules/roundthree -->
![What round 3 says about a local repair and a local share-out](../figures/localrules/roundthree.svg)

**What round 3 says about a local repair and a local share-out.** Each dot is one of the 70 worlds of Experiment 40 that lived to the end of their 3,000 iterations, read from iteration 100 on; the short bar is the median of a column. Top left, on a logarithmic axis: the share of all the world's tokens a game culls, the tokens of agents cut off from the largest piece of the network, for the worlds drawn with mending (a dying agent's former neighbours joined up again in a ring) and those without. Top right: the most of its agents a single game culled. Bottom left: how wide the world is, its typical distance over that of a random network with the same links (on the dashed line, no wider than one). Bottom right, on a logarithmic axis: what the richest agent holds, by where each world's rule sends the tokens of the dead: evenly over all, to the poorest, to this iteration's newborns, to their parents, to the dead agent's own children, or to the nearest living. Every other setting of each world was drawn at random too, so each column mixes many kinds of world.

> [!example]- How to make this figure
> **Runs.** the worlds of round 3 of the screen of random worlds that lived to the end of their 3,000 iterations (Experiment 40: 30,000 tokens, every setting drawn at random)
>
> **Data.** From each world's averages over iterations 100 to the end, `research/screen/health.json` (from the screen's readings, `research/screen/measures.py`), and each run's own statistics, `GraphOfLifeRuns/<run>/stats.jsonl` (`nodes_before`, `starved`, `orphaned`).
>
> 1. Read each living world's averages from `health.json`: `culled_tokens` (the share of all tokens a game's cleanup shares out, every one a culled agent's), `width` and `richest`.
> 2. From its `stats.jsonl`, for every game from iteration 100 on, `orphaned` over `nodes_before`: the share of the agents the game culled. Keep the largest.
> 3. Group the worlds by their drawn `mend_on_death`, and by `redistribution`.
> 4. One dot per world; the bar is the median.
>
> **To make it again:** `python3 book_figures.py localrules`.
<!-- /figure -->

> [!quote] Draft thesis (to be fixed before any run)
> **The claim.** Each of the three rules made local changes round 3's candidate worlds in its own way. Random streams of their own change nothing but the luck of the draw. Estates that flow to the nearest living make the worlds more unequal and cull no less. Pieces that live on break the worlds into islands: within 1,000 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") the largest piece of most worlds holds less than half of all tokens. All three together do as pieces that live on do.
>
> **Why it would be so.** A counter-based stream draws numbers as random as the one stream, only from other addresses, so a world lives another of the histories its rules allow. In round 3, the 9 living worlds drawn with estates to the nearest living had a [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") [Gini](measuring-tokens.md#measuring-inequality-the-gini-coefficient-the-richest-tenth-entropy-and-mobility "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all.") of 0.53, the highest of six rules (about 1 in 10,000 by chance), and a richest agent with 2.7% of all tokens against 0.6%, while culling no less (2.9% of the tokens a game against 5.6%). A piece that separates can never be joined again, since every link is made inside a neighbourhood; and in these worlds no agent dies of age and an agent alone cannot starve, so a piece holding a token lives for ever. The six worlds culled 0.3% to 12.7% of their tokens in a game, and each such game would leave a piece behind with its tokens. Against this, the regions that break off are the poorest, and the largest piece may hold together better once they are gone.
>
> **It holds if:** against each world's own run of round 3, from iteration 100 to 3,000: (1) with streams of their own, health by the product moves by less than the median difference between the two [seeds](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") of one [setting](settings.md "One number or switch of a run's rules, fixed before it starts.") in Experiment 43, in at least five of the six worlds; (2) with local estates, the Gini coefficient and the richest agent's share both rise, and the share of tokens culled in a game does not fall by more than a third, in at least five of the six; (3) with pieces that live on, the largest piece holds less than half of all tokens by iteration 1,000 in at least five of the six; (4) with all three, the same as (3).
>
> **It fails if:** (1) streams of their own move health by more than that difference in three worlds or more: then the stream shapes worlds, or the build changed a rule; (2) local estates leave the Gini coefficient as it was, or lower it, in three worlds or more; (3) or (4) the largest piece still holds more than half of all tokens at iteration 3,000 in three worlds or more. Six worlds are few: either way the result is indicative.

The full draft, with the rules or [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") it would build, the experiment, what
has to be built first and what it would cost, is in the notebook's history:
`git show 9ec755d:book/pages/local-rules-only.md`.

## Where to go next

- [How far does a difference travel?](how-far-does-a-difference-travel.md):
  the light cone of a world, measured, once its rules are local.
- [A source and a sink](a-source-and-a-sink.md): a world with a flow running
  through it, planned on the local world of this page.
- [Physical inspiration](physical-inspiration.md#where-the-speed-limit-is-broken):
  why physics asks for local rules.
- [How a world breaks](b1-the-network-size-and-scaling.md#how-a-world-breaks): what the cull removes, and
  where.

## How our view changed

- **11 November 2026.** Shrunk to its questions, one paragraph and its draft
  thesis, as the owner approved in a review of the whole notebook: a plan not
  yet chosen need not carry its whole draft. The full draft is in the
  notebook's history, as the note above says, and nothing of it is lost.

## References

- Mayr, E. (1942). *Systematics and the Origin of Species*. New York: Columbia University Press.
- Salmon, J. K., Moraes, M. A., Dror, R. O. and Shaw, D. E. (2011). Parallel random numbers: as easy as 1, 2, 3. *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC11)*.

<!-- turns -->
---

← [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [Space and dimension](../README.md#space-and-dimension) · [How far does a difference travel?](how-far-does-a-difference-travel.md) →

**Also about:** [Changing the rules](../README.md#changing-the-rules).

> [!info]- Pages that link here (7)
> [Questioning the mechanics](questioning-the-mechanics.md) · [Physical inspiration](physical-inspiration.md) · [How far does a difference travel?](how-far-does-a-difference-travel.md) · [A source and a sink](a-source-and-a-sink.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md) · [Space at scale (side study)](space-at-scale.md)
<!-- /turns -->
