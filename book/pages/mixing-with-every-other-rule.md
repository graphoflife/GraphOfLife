# Brains mixed at birth, with every other rule

> [!question] Questions of this page
> - Mixing [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") at birth made [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s worlds smaller, richer and more diverse.
>   Does it still, whatever else the rules say?
> - Which rules cancel it, which make it stronger, and why those?

## In short

Mixing a newborn's brain from its parent's and a [neighbour](the-world.md#the-network "An agent joined to another by a connection.")'s makes B2's
worlds rich, because [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") put off having children. Batch Y played mixing
together with 23 other rules, one at a time. With sixteen of them, mixing
keeps its riches. Seven cancel them, mostly rules that pay for births or make
them cheap, so that the world fills up again. Mixing raises how much the
agents differ with nearly every rule. And the later brains beat the earlier
ones in 42 of the 46 worlds, fastest where the agents stay rich. The richest
combinations are also the smallest worlds, and small mixed worlds are the
ones that die out.

## Mixing, with the other rules

Of every rule the notebook has tried, one came closest to the author's
picture of a world that might keep evolving
([Healthy and rich](healthy-and-rich.md)). It is mixing a newborn's brain
from its parent's and a neighbour's.
[Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md) found
out why it makes a world rich. Agents put off having children, so the world
never fills up to the point where hardly anyone can afford one. The author
asked for mixing to be tested together with the other rules. This page does
that, for 23 of them at once.

## The worlds

Batch Y plays B2 at 10,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") with brains mixed at birth and one other
rule. Each world has [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 and 2 and runs for 1,000 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."), as pilot
worlds ([Pilot worlds](the-lab.md#pilot-worlds)). Each world is the fourth
corner of a two-by-two whose other three corners already existed:

| corner | the worlds |
|---|---|
| B2 | the lab's Experiment 16 worlds, seeds 1 and 2 |
| B2 with mixing | the lab's Experiment 17 worlds, seeds 1 and 2 |
| B2 with the other rule | the batch or lab experiment that made it: batches P, Q, R, S, T and W, Experiments 9, 15 and 16 |
| B2 with both | batch Y |

The other rules are every switch of the notebook's pilots and lab experiments
with B2:
- who inherits the [estate](estate.md "The tokens a dead agent leaves, and who gets them.") ([Who inherits?](who-inherits.md));
- links that hold a token;
- inputs that say who helped, or where an agent came from;
- handover, no random inputs, no old age, and children of any size;
- how much is kept at home;
- brains that grow, or whose units hold a token;
- a slower or faster change of brains;
- unused links that lapse;
- and the five other ways of deciding a node that batch S tried.

Every corner is read at iteration 1,000, the same moment in each, as
[Healthy and rich](healthy-and-rich.md) reads every world
(`research/atlas/at.py`).

## Riches

<!-- figure mixcombos/riches -->
![Agents holding three tokens or fewer](../figures/mixcombos/riches.svg)

**Agents holding three tokens or fewer.** Each grey dot is a world of B2 with one other rule, each orange dot the same rule with brains mixed at birth as well (batch Y), seeds 1 and 2, read at iteration 1,000; the short bars are the means. The dashed lines are B2 itself and B2 with mixing alone, the means of the same seeds. The rules are in the order of the orange means.

> [!example]- How to make this figure
> **Runs.** Batch Y's 46 pilot worlds (B2 at 10,000 tokens, brains mixed at birth and one other rule, seeds 1 and 2, `GraphOfLifeRuns/.pilots/Y/`) and their controls: the same rule alone, from the batch or lab experiment that made it, and, of the same seeds, B2 itself (Experiment 16's lab worlds) and B2 with brains mixed at birth alone (Experiment 17's lab worlds).
>
> **Data.** From each world's checkpoint at iteration 1,000, or the lab's whole world kept then (`research/atlas/at.py --at 1000`, `GraphOfLifeRuns/.pilots/atlas/at1000.json`).
>
> 1. Read each world as `research/atlas/profile.py` reads every world.
> 2. Rank its four spreads among [Healthy and rich](healthy-and-rich.md)'s worlds; the diversity score is the mean.
>
> **To make it again:** `python3 book_figures.py mixcombos`.
<!-- /figure -->
**With sixteen of the rules, mixing keeps its riches.** The orange dots sit
far below the grey ones, near or below B2 with mixing alone. With some rules
it does more than alone:
- with a sense of family, 18% of the agents hold three tokens or fewer;
- with an input saying who helped, 26%;
- without random inputs, 28% (one world at 2%);
- with brains that grow, 28%;
- with units that hold a token, 30%.

With mixing alone it is 34%. With three of the sixteen, one seed of the two
stayed nearly as poor as without mixing. They are estates by wealth, estates
to the poorest, and a slower change of brains.

**Seven rules cancel it:**
- the estate of the dead to the newborns, or to the parents;
- children of any size;
- no old age;
- handover;
- only the last token kept at home;
- and stakes pooled by family.

Their worlds are as poor with mixing as
without: 73% to 94% of the agents hold three tokens or fewer.

[Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md) says
why for most of them. Mixing makes a world rich by making its agents slow to
have children. Three of these rules pay for a birth or make it cheap. An
estate lands on the newborns or on their parents, or a child is had for any
number of tokens. Then the world fills up again. Without old age nobody needs
replacing at all, and the world fills with the old. Why handover, the last
token kept and stakes pooled by family cancel mixing too, this batch does not
show.

## How much the agents differ

<!-- figure mixcombos/diversity -->
![How much the agents differ](../figures/mixcombos/diversity.svg)

**How much the agents differ.** Each grey dot is a world of B2 with one other rule, each orange dot the same rule with brains mixed at birth as well (batch Y), seeds 1 and 2, read at iteration 1,000; the short bars are the means. The dashed lines are B2 itself and B2 with mixing alone, the means of the same seeds. The rules are in the order of the orange means.

> [!example]- How to make this figure
> **Runs.** Batch Y's 46 pilot worlds (B2 at 10,000 tokens, brains mixed at birth and one other rule, seeds 1 and 2, `GraphOfLifeRuns/.pilots/Y/`) and their controls: the same rule alone, from the batch or lab experiment that made it, and, of the same seeds, B2 itself (Experiment 16's lab worlds) and B2 with brains mixed at birth alone (Experiment 17's lab worlds).
>
> **Data.** From each world's checkpoint at iteration 1,000, or the lab's whole world kept then (`research/atlas/at.py --at 1000`, `GraphOfLifeRuns/.pilots/atlas/at1000.json`).
>
> 1. Read each world as `research/atlas/profile.py` reads every world.
> 2. Rank its four spreads among [Healthy and rich](healthy-and-rich.md)'s worlds; the diversity score is the mean.
>
> **To make it again:** `python3 book_figures.py mixcombos`.
<!-- /figure -->
Mixing raises the [diversity score](diversity-score.md "How much the agents of a world differ, in one number from 0 to 1: in tokens, links, ages and genotypes.") with nearly every rule, even with most of
those that cancel its riches. With children of any size it rises from 0.25 to
0.54, and with only the last token kept from 0.43 to 0.66. Part of that is
built in. The score counts [genotypes](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from."), and every mixed birth makes a new set of
weights, so a mixed world always has more genotypes per agent. But tokens,
links and ages spread wider too where the world is rich.

The most diverse worlds of the batch are these:
- mixing with a faster change of brains (0.78 and 0.79), more than all but a
  few worlds built on B2's rules;
- mixing with a sense of kin or family (0.70, 0.71);
- mixing with an input saying where an agent came from (0.66);
- and mixing alone (0.70, 0.58).

## Healthy?

Nearly every world meets six or seven marks of the [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4."), as B2's and
the mixed worlds do. The exceptions are the rules that broke the network
before mixing came. With every unused link lapsing, worlds meet four or five
marks; with handover, three.

The marks hide a risk. [Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md)
found it, and
[Five mixed worlds, fifty thousand iterations](mixed-at-birth-ten-long-worlds.md#five-mixed-worlds-fifty-thousand-iterations)
saw it end in death: a mixed world's size drifts. Several of the richest
combinations are small at 1,000 iterations. A world has 270 agents with a
sense of family, and 291 without random inputs. It has 325 with units that
hold a token, and 366 with brains that grow. And small mixed worlds are the ones that die out.

## Does evolution go on?

<!-- figure mixcombos/evolution -->
![Do the brains of 1,000 beat those of 500?](../figures/mixcombos/evolution.svg)

**Do the brains of 1,000 beat those of 500?** Each dot is a world of batch Y: the share of a contested world its brains of iteration 1,000 hold after 100 iterations against those of 500, the mean of two contests, one in each era's world; the bars are the medians, the rules in the order of the figures above. The dashed lines are the medians of B2's ten lab worlds and of the ten mixed worlds of Experiment 17, played the same way.

> [!example]- How to make this figure
> **Runs.** Batch Y's pilot worlds (`GraphOfLifeRuns/.pilots/Y/`); B2: Experiment 16's ten runs; mixing alone: the ten mixed worlds of Experiment 17.
>
> **Data.** From `GraphOfLifeRuns/.pilots/batch_TUV_T100.json`, `rejudge_T100.json`, `long_steps.json` and `e17_T100.json` (`research/oee/checks/rejudge.py`).
>
> 1. Seat the brains of the two eras on alternate agents of each era's world and play 100 iterations.
> 2. Count the agents the later era's brains hold, as a share of all.
>
> **To make it again:** `python3 book_figures.py mixcombos`.
<!-- /figure -->
With almost every rule, yes. In 42 of the 46 worlds, the brains of iteration
1,000 took more than half of a contested world from those of 500. The
contests last an agent's [lifetime](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death.")
([A contest a lifetime long](eras-and-contests.md#a-contest-a-lifetime-long)).
The [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world took 69%, where B2's take 68% and those of mixing alone 84%.
The steps are largest with links that hold a token (93%), half kept at home
(91%) and units that hold a token (90%). Next come the estate to the poorest
and brains that grow (89% each).

Four worlds' later brains were no better. They came from stakes pooled by
family (35%), handover and a slower change of brains (47% each), and no
random inputs (50%). Two of them are rules that cancel mixing's riches. The
other two are worlds that stayed poorer than mixing leaves most worlds, with
81% and 53% of their agents at three tokens or fewer. Mixing does not stop
evolution, whatever else the rules say; it speeds it most where the agents
stay rich.

## What we learned

- **Mixing's riches are robust, but not to rules that pay for births.** With
  sixteen rules of twenty-three the worlds stay rich. Of the seven that cancel
  it, four bring children back. They pay for them, make them cheap, or take
  away the deaths they replace.
- **Evolution goes on with nearly every combination**, fastest where the
  agents stay rich.
- **The richest combinations are also the smallest.** Riches are size where
  the tokens are fixed
  ([Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md)), and
  a small mixed world is a fragile one. None of these combinations was played
  long enough to say whether it holds its size. Experiment 18's five worlds of
  mixing alone did not.
- **[Candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides.").** Some worlds of the batch come closest to the author's
  picture, by riches and diversity together. They are mixing with:
  - a faster change of brains;
  - a sense of kin or of family;
  - an input saying who helped, or where an agent came from;
  - nodes decided by lottery.

  Those that meet every mark join the candidates of
  [Healthy and rich](healthy-and-rich.md) (`research/oee/candidates.md`).

## Where to go next

- [Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md): why
  mixing makes a world rich, and what became of its worlds over 50,000
  iterations.
- [Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md):
  where mixing was first tried.
- [Who inherits?](who-inherits.md): the rules for the estate of the dead,
  several of which cancel mixing.
- [Eight switches at once](eight-switches-at-once.md): mixing among seven
  other switches thrown together.

## How this was made

Every world of batch Y is a pilot, kept in `GraphOfLifeRuns/.pilots`. No
experiment of the lab stands behind it, so the page has no box of runs. Each
corner was read at iteration 1,000 with `research/atlas/at.py`. Every figure
says, folded beneath it, which worlds it reads and how. The setting is
[`mix_at_birth`](settings.md#mix_at_birth).

## How our view changed

- **17 October 2026.** Diversity scores had been ranked among all the worlds
  read so far, so they moved a little whenever worlds were added. Since that
  day they are ranked among a frozen set of worlds
  ([Two halves](healthy-and-rich.md#two-halves)), so that they stay put. Three
  diversity figures of this page moved by 0.01. The figures also gained the
  second seed of every rule alone that had only one before: batch W's worlds
  with [revolutions](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake."), and handover alone.

<!-- turns -->
---

← [Links to strangers](links-to-strangers.md) · [Changing the rules](../README.md#changing-the-rules) · [Eight switches at once, and how sixteen worlds can test them](eight-switches-at-once.md) →

> [!info]- Pages that link here (9)
> [What holds a world's size](what-holds-a-worlds-size.md) · [Rules in company](rules-in-company.md) · [Revolutions, handover and gifts: the switches the world was built with](the-switches-the-world-was-built-with.md) · [Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md) · [Eight switches at once, and how sixteen worlds can test them](eight-switches-at-once.md) · [Healthy and rich](healthy-and-rich.md) · [Who may be a mate](mixed-at-birth-with-parents-as-mates.md) · [The candidates, played on and played again](the-candidates-played-on.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
