# Eight switches at once

> [!question] Questions of this page
> - The notebook has thrown its switches into B2 one at a time. What does each
>   do when several are thrown together?
> - Which switches move a world's riches, its diversity, its size and its
>   network the most?
> - Do their effects add up, or does one switch change what another does?

Most experiments on B2 threw one switch into it and compared the world with
B2 itself. That tells what a switch does *in B2*. Whether it does the same
once other switches are thrown as well — whether the effects add up — it
cannot tell. The author asked for the existing settings tested in
combination (16 October 2026). Every combination of eight switches would be
256 worlds; a **fractional factorial design** plays sixteen of them, chosen so
that each switch is on in eight and off in eight, and every pair of switches
meets in all four ways equally often
([Many switches in few worlds](fractional-factorial.md)).

## The sixteen worlds

The eight switches, each one the notebook has played alone in B2 before:

| | switch | the setting | alone in |
|---|---|---|---|
| A | gifts | [`allow_gifting`](settings.md#allow_gifting), with unused links cut after reproduction once unused for an iteration | batch S ([The switches the world was built with](the-switches-the-world-was-built-with.md)) |
| B | no revolutions | [`allow_revolutions`](settings.md#allow_revolutions) off | batch S ([The switches the world was built with](the-switches-the-world-was-built-with.md)) |
| C | every unused link lapses | [`prune_needs`](settings.md#prune_needs) = 0 | batch S ([The switches the world was built with](the-switches-the-world-was-built-with.md)) |
| D | no random inputs | [`random_input_amount`](settings.md#random_input_amount) = 0 | Experiment 16 ([Why evolution stops](why-evolution-stops.md)) |
| E | estates to the newborns | [`redistribution`](settings.md#redistribution) = `newborns` | [Who inherits?](who-inherits.md) |
| F | links hold a token | a link costs a token to keep, with no cap on links | batch P ([Rules without numbers](rules-without-numbers.md)) |
| G | brains mixed at birth | a newborn's brain mixed from its parent's and a neighbour's | [Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md) |
| H | handover | [`allow_handover`](settings.md#allow_handover): a parent may pass links to its newborn | batch T's control (`T/handover`) |

A, B, C and D are on and off in all sixteen combinations; E, F, G and H are
set from them (E = BCD, F = ACD, G = ABC, H = ABD — on where the product of
the three is +1). Each world is B2 at 10,000 tokens, seed 1, played for
1,000 iterations (pilot batch U), and read at iteration 1,000 as
[Healthy and rich](healthy-and-rich.md) reads every world. The world with
no switch on is B2's own world of seed 1, game for game.

| switches on | agents | agents with 3 tokens or fewer | diversity score | marks (of 7) | dimension | links per agent |
|---|---|---|---|---|---|---|
| C, D, G, H | 931 | 1% | 0.61 | 5 | 2.31 | 3.7 |
| B, C, F, H | 1,128 | 2% | 0.10 | 4 | 1.97 | 3.4 |
| A, B, C, G | 1,023 | 2% | 0.49 | 4 | 1.73 | 3.6 |
| C, E, F, G | 1,279 | 13% | 0.63 | 4 | 2.43 | 3.1 |
| B, C, D, E | 1,502 | 17% | 0.52 | 3 | 2.23 | 3.3 |
| A, C, D, F | 1,334 | 17% | 0.36 | 2 | 2.14 | 3.0 |
| B, D, F, G | 652 | 25% | 0.68 | 7 | 3.67 | 5.4 |
| A, C, E, H | 1,470 | 28% | 0.46 | 3 | 2.45 | 2.7 |
| all eight | 1,408 | 54% | 0.81 | 3 | 2.14 | 3.4 |
| A, F, G, H | 1,456 | 61% | 0.72 | 4 | 3.68 | 3.2 |
| none (B2) | 3,598 | 77% | 0.51 | 6 | 4.39 | 5.5 |
| A, B, D, H | 3,031 | 80% | 0.53 | 4 | 3.95 | 3.7 |
| A, B, E, F | 1,945 | 85% | 0.24 | 7 | 3.46 | 5.3 |
| D, E, F, H | 2,906 | 88% | 0.42 | 3 | 2.39 | 2.5 |
| B, E, G, H | 5,196 | 91% | 0.49 | 4 | 3.40 | 3.1 |
| A, D, E, G | 5,313 | 93% | 0.39 | 7 | 2.86 | 4.6 |

## What each switch does

<!-- figure eightswitches/effects -->
![What each switch does, among the sixteen and alone](../figures/eightswitches/effects.svg)

**What each switch does, among the sixteen and alone.** Orange: each switch's main effect among the sixteen worlds of batch U — the mean of the eight worlds with it on, minus the mean of the eight with it off — at iteration 1,000. Grey: the same switch thrown alone in B2, the world with it minus its twin without it, seeds 1 and 2. The grey band round zero is the seed's own noise: one standard deviation of the measure over the lab's ten B2 worlds, twice the noise of a main effect.

> [!example]- How to make this figure
> **Runs.** Batch U's sixteen pilot worlds (`GraphOfLifeRuns/.pilots/U/`): B2 at 10,000 tokens, seed 1, 1,000 iterations, with each of eight switches on in eight of them (`research/oee/pilots/make_jobs_TU.py`); each switch alone in B2, seeds 1 and 2, from the batch or lab experiment that made it; and the lab's ten B2 worlds (Experiment 16) for the seed's own noise.
>
> **Data.** From each world's checkpoint at iteration 1,000, or the lab's whole world kept then (`research/atlas/at.py --at 1000`, `GraphOfLifeRuns/.pilots/atlas/at1000.json`).
>
> 1. Read each world as `research/atlas/profile.py` reads every world.
> 2. A switch's main effect: the mean of the eight worlds with it on, minus the mean of the eight with it off.
> 3. Thrown alone: the world with the switch minus its twin without it, each seed.
> 4. The seed noise: the standard deviation of the lab's ten B2 worlds at iteration 1,000.
>
> **To make it again:** `python3 book_figures.py eightswitches`.
<!-- /figure -->

A switch's **main effect** is the mean of the eight worlds with it on minus
the mean of the eight with it off. How large is large: the lab's ten B2
worlds at iteration 1,000 differ only in their seed, and their spread (the
grey band) is twice the noise of a main effect. The seed moves the share of
poor agents by 0.03, the diversity score by 0.06, the dimension by 0.2 and
the agents by 270; it never moves the marks — all ten B2 worlds meet six.

- **Every unused link lapsing (C) is the strongest switch.** It takes 1,750
  agents off a world, 0.58 off the share of agents holding three tokens or
  fewer, 1.3 off the dimension, 0.9 links off each agent and 1.75 marks off
  the health bar. Seven of the eight worlds with it hold less than a third of
  their agents at three tokens or fewer; seven of the eight without it, more
  than three fifths.
- **The estate to the newborns (E) fills a world**: 980 agents more, and 0.26
  more of them poor.
- **Links that hold a token (F) shrink it**, by 1,240 agents.
- **Mixing (G) raises diversity the most**, by 0.21 — almost four times the
  seed's noise — and it adds 0.75 marks.
- **Handover (H) costs links and health**: a link per agent and 1.25 marks.
- **Gifts (A) make worlds poorer** (0.13 more at three tokens or fewer), and no
  revolutions (B) leave them a third of a link richer per agent; no random
  inputs (D) raise diversity by 0.09, just above the noise.

## Alone and together

The grey dots of the same figure are the switches thrown alone into B2, two
seeds each. For several switches the two disagree:

- **Mixing** alone takes about 2,600 agents off B2's world and 0.36 to 0.49
  off its share of poor agents ([Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md));
  among the sixteen, it takes 40 agents *on* and 0.07 off. Its rise in
  diversity is the same alone (0.18 and 0.20) as among the sixteen (0.21),
  and so, nearly, is its gain in marks (one alone, 0.75 among the sixteen).
- **Links that hold a token** alone take 0.29 and 0.33 off B2's share of
  the poor; among the sixteen, 0.06.
- **Every unused link lapsing** alone takes 2.5 and 2.7 off the dimension;
  among the sixteen, 1.3.
- **The estate to the newborns** alone adds 0.13 and 0.15 to the share of
  the poor; among the sixteen, twice that.
- **Handover** alone costs three marks and 1.7 and 2.4 links per agent, and
  takes 660 and 1,040 agents off the world; among the sixteen, 1.25 marks, a
  link, and no agents.

A switch does on average among the sixteen what it does alone in B2 only if
no pair of switches acts together. Here pairs do:

<!-- figure eightswitches/pairs -->
![Switches that act together](../figures/eightswitches/pairs.svg)

**Switches that act together.** The seven chains of pairs of batch U (book/notes/fractional-factorial.md): for each, the mean of the eight worlds where the product of the pair's two switches is +1, minus the mean of the other eight — the sum of the four pairs' effects that one chain cannot tell apart. The grey band: the seed's own noise, as in the figure above.

> [!example]- How to make this figure
> **Runs.** Batch U's sixteen pilot worlds (`GraphOfLifeRuns/.pilots/U/`): B2 at 10,000 tokens, seed 1, 1,000 iterations, with each of eight switches on in eight of them (`research/oee/pilots/make_jobs_TU.py`); each switch alone in B2, seeds 1 and 2, from the batch or lab experiment that made it; and the lab's ten B2 worlds (Experiment 16) for the seed's own noise.
>
> **Data.** From each world's checkpoint at iteration 1,000, or the lab's whole world kept then (`research/atlas/at.py --at 1000`, `GraphOfLifeRuns/.pilots/atlas/at1000.json`).
>
> 1. Read each world as `research/atlas/profile.py` reads every world.
> 2. A chain's effect: the mean of the eight worlds where the product of a pair's switches is +1, minus the mean of the other eight.
>
> **To make it again:** `python3 book_figures.py eightswitches`.
<!-- /figure -->

Each chain adds up four pairs the sixteen worlds cannot tell apart. Three of
them stand far outside the seed's noise: the chain AD + BH + CF + EG adds
1,300 agents and 0.15 to the share of the poor; BD + AH + CE + FG adds 0.22
to the diversity and 0.59 to the dimension; AE + BF + CH + DG adds a link
per agent and 1.75 marks. Which pair in a chain does it, the design cannot
say — but the worlds of batch Y, each mixing with one other rule beside the
rule alone, say it for two of them:

- **Mixing with the estate to the newborns (EG).** B2 with the estate to the
  newborns holds 90% and 91% of its agents at three tokens or fewer; with
  mixing as well, 91% and 90%. Mixing takes nothing off — where alone it
  takes 0.36 to 0.49. Among the sixteen, likewise, mixing takes 0.22 off the
  share of the poor in the worlds whose estate is shared evenly, and adds
  0.08 in those whose estate goes to the newborns: the rise of the chain
  AD + BH + CF + EG, though A with D, B with H and C with F could hide in the
  same difference.
- **Mixing with lapsing links (CG).** With every unused link lapsing, 1% and
  9% are poor; with mixing as well, 6% and 4%. A world already rich is not
  made richer.

So mixing's riches, which [Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md) found surviving sixteen of 23 other
rules one at a time, mostly vanish among the sixteen, where most worlds have
four switches on: of the eight mixed worlds, four hold the estate to the
newborns, which cancels it, and four the lapse, which leaves nothing to add.

## Does evolution go on?

> [!info] Added later (17 October 2026)
> The contests below were played after this page was first published.

<!-- figure eightswitches/evolution -->
![Do the brains of 1,000 beat those of 500?](../figures/eightswitches/evolution.svg)

**Do the brains of 1,000 beat those of 500?** Above: each world of batch U, the share of a contested world its brains of iteration 1,000 hold after 100 iterations against those of 500, the mean of two contests, one in each era's world; the dashed line marked B2 is the median of the lab's ten B2 worlds, played the same way. Below: each switch's main effect on that share; the grey band is the spread of the ten B2 worlds (one standard deviation), twice the noise of a main effect.

> [!example]- How to make this figure
> **Runs.** Batch U's sixteen pilot worlds (`GraphOfLifeRuns/.pilots/U/`): B2 at 10,000 tokens, seed 1, 1,000 iterations, with each of eight switches on in eight of them (`research/oee/pilots/make_jobs_TU.py`); each switch alone in B2, seeds 1 and 2, from the batch or lab experiment that made it; and the lab's ten B2 worlds (Experiment 16) for the seed's own noise.
>
> **Data.** From `GraphOfLifeRuns/.pilots/batch_TUV_T100.json` and, for B2, `long_steps.json` (`research/oee/checks/rejudge.py`).
>
> 1. Seat the brains of the two eras on alternate agents of each era's world and play 100 iterations.
> 2. Count the agents the later era's brains hold, as a share of all.
> 3. A switch's main effect: the mean of the eight worlds with it on, minus the other eight.
>
> **To make it again:** `python3 book_figures.py eightswitches`.
<!-- /figure -->

Played against the brains of iteration 500 for a lifetime, the brains of
1,000 hold more than half of a contested world in fifteen of the sixteen
worlds: from 0.47 (C, D, G and H on) to 1.0 (all eight on, where the later
brains took the whole world in both eras' worlds). The median is 0.60, below
B2's own 0.69 (the lab's ten B2 worlds: a median of 0.68, with a spread of
0.04). Gifts, no revolutions, lapsing links and the estate to the newborns
each speed evolution a little — 0.07 to 0.09, about twice the spread of B2's
worlds — and the other four switches do nothing a contest can tell. Pairs
act together here too: three chains add 0.11 to 0.14.

> [!info] Added later (17 October 2026)
> Diversity scores are now ranked among a frozen set of worlds
> ([Healthy and rich](healthy-and-rich.md#two-halves)), so that they stay put
> as worlds are added; two figures above moved by 0.01 (no random inputs: 0.08
> to 0.09; mixing alone: 0.17 and 0.19 to 0.18 and 0.20).

## The same eight switches on B1

> [!info] Added later (18 October 2026)
> Batch V, promised below, has run.

B1 is the world that can break: its revolutions take most of the nodes that
change hands, a parent may pass links to its newborn, and pieces cut off are
culled ([The switches the world was built with](the-switches-the-world-was-built-with.md)). Batch V
plays the same design on it, sixteen worlds of 10,000 tokens, seed 1, 1,000
iterations, with eight switches that mean something there: A gifts, B no
revolutions, C no handover, D no random inputs, E estates to the newborns,
F brains that change only when copied (B2's rule), G brains mixed at birth,
H links that hold a token.

| switches on | agents | agents with 3 tokens or fewer | diversity score | marks (of 7) | dimension | links per agent |
|---|---|---|---|---|---|---|
| none (B1) | 871 | 2% | 0.72 | 3 | 2.56 | 2.7 |
| A, B, D, H | 264 | 9% | 0.67 | 2 | 1.65 | 2.2 |
| B, C, F, H | 501 | 11% | 0.69 | 3 | 2.28 | 3.3 |
| A, B, C, G | 1,374 | 21% | 0.56 | 4 | 2.25 | 6.8 |
| C, D, G, H | 336 | 25% | 0.88 | 3 | 2.75 | 4.9 |
| D, E, F, H | 1,560 | 36% | 0.31 | 3 | 2.49 | 2.3 |
| B, D, F, G | 1,140 | 37% | 0.66 | 2 | 2.19 | 2.7 |
| B, E, G, H | 1,718 | 40% | 0.35 | 3 | 2.35 | 2.4 |
| all eight | 1,567 | 44% | 0.60 | 4 | 2.66 | 3.2 |
| C, E, F, G | 2,584 | 57% | 0.68 | 3 | 2.14 | 3.2 |
| A, B, E, F | 3,619 | 72% | 0.20 | 3 | 2.43 | 2.9 |
| A, C, D, F | 2,933 | 75% | 0.62 | 4 | 3.02 | 9.6 |
| A, C, E, H | 1,841 | 82% | 0.26 | 1 | 4.32 | 2.5 |
| A, D, E, G | 4,413 | 84% | 0.35 | 6 | 2.44 | 3.7 |
| A, F, G, H | died out by iteration 341, after at most 398 agents | | | | | |
| B, C, D, E | died out after iteration 781, after 3,483 agents | | | | | |

**Two worlds died out**, and they are each other's mirror: every switch on in
one is off in the other. That has a use. Every switch still has seven
living worlds with it and seven without, so a main effect is still the
difference of two means of seven. And it has a cost a student should know:
with a world and its mirror gone, one combination of the eight main effects
can no longer be told apart from the rest — no fit can recover it.

<!-- figure eightswitches/b1effects -->
![The same eight switches on B1](../figures/eightswitches/b1effects.svg)

**The same eight switches on B1.** Orange: each switch's main effect among the worlds of batch V that lived to iteration 1,000 (14 of 16; V-AFGH-s1 and V-BCDE-s1 died out, and are each other's mirror, so every switch keeps seven worlds on and seven off): the mean of the seven with it on, minus the mean of the seven with it off. Grey dots: the switch thrown alone in B1, the world with it minus its twin without it, seeds 1 and 2, where the notebook has them. The grey band round zero is twice the spread of the seven chains of pairs, read the same way: what a main effect would scatter by if pairs of switches did not act together.

> [!example]- How to make this figure
> **Runs.** Batch V's sixteen pilot worlds (`GraphOfLifeRuns/.pilots/V/`): B1 at 10,000 tokens, seed 1, 1,000 iterations, with each of eight switches on in eight of them (`research/oee/pilots/make_jobs_TU.py`): gifts, no revolutions, no handover, no random inputs, estates to the newborns, brains that change only when copied, brains mixed at birth, links that hold a token; and the switches the notebook has played alone in B1, seeds 1 and 2 (batches S, T and R).
>
> **Data.** From each world's checkpoint at iteration 1,000 (`research/atlas/at.py --at 1000`, `GraphOfLifeRuns/.pilots/atlas/at1000.json`).
>
> 1. Read each world as `research/atlas/profile.py` reads every world; leave out the worlds with fewer than 20 agents.
> 2. A switch's main effect: the mean of the worlds with it on, minus the mean of those with it off.
> 3. The band: twice the root mean square of the seven chains of pairs, read the same way.
> 4. Thrown alone: the world with the switch minus its twin without it, each seed.
>
> **To make it again:** `python3 book_figures.py eightswitches`.
<!-- /figure -->

B1's worlds are far less alike than B2's — the two seeds of B1 itself differ
in diversity by 0.34 — and on B1 pairs of switches act together more, so the
yardstick here is the spread of the seven chains of pairs, which would
scatter as the main effects do if no pair acted together. Against it, few
switches stand out:

- **The estate to the newborns moves B1 as it moves B2**: 1,410 more agents,
  0.33 more of them at three tokens or fewer, a diversity score 0.29 lower —
  as when thrown alone in B1 ([Who inherits?](who-inherits.md)).
- **Without handover, a world keeps more links** — two more per agent — the
  other side of what handover did in B2
  ([Handover, with every other rule](handover-with-every-other-rule.md)).
- **Gifts make B1 larger and poorer** (1,040 more agents, 0.25 more of them
  poor), and links that hold a token make it smaller (1,310 fewer).
- **No revolutions does not do in company what it did alone.** Thrown alone,
  B1 without revolutions became a world of mayflies, larger by 990 and 1,320
  agents and poorer (0.27 and 0.41 more at three tokens or fewer); among the
  fourteen its effect is within the chains' spread, and if anything the other
  way (620 agents fewer, 0.18 fewer poor).

> [!info] Added later (18 October 2026)
> The contests below were played after this section was first published.

<!-- figure eightswitches/b1evolution -->
![On B1: do the brains of 1,000 beat those of 500?](../figures/eightswitches/b1evolution.svg)

**On B1: do the brains of 1,000 beat those of 500?** Above, in orange: each world of batch V that lived, the share of a contested world its brains of iteration 1,000 hold after 100 iterations against those of 500, the mean of two contests, one in each era's world; in grey, B1 itself played the same way — two pilot worlds of 10,000 tokens, and the lab's ten worlds of 5,000. Below: each switch's main effect on that share, seven worlds with it against seven without; the grey band is twice the spread of the seven chains of pairs.

> [!example]- How to make this figure
> **Runs.** Batch V's sixteen pilot worlds (`GraphOfLifeRuns/.pilots/V/`): B1 at 10,000 tokens, seed 1, 1,000 iterations, with each of eight switches on in eight of them (`research/oee/pilots/make_jobs_TU.py`): gifts, no revolutions, no handover, no random inputs, estates to the newborns, brains that change only when copied, brains mixed at birth, links that hold a token. For B1 itself: the pilots `GraphOfLifeRuns/.pilots/L/B1-long-s1` and `-s2` (10,000 tokens), and the lab's ten B1 worlds of 5,000 tokens (Experiment 12, `B1-5000-s001` … `-s010`).
>
> **Data.** From `GraphOfLifeRuns/.pilots/batch_TUV_T100.json`, `batch_TUVWX4_T100.json`, `batch_TUVWX5_T100.json` and `rejudge_T100.json` (`research/oee/checks/rejudge.py`).
>
> 1. Seat the brains of the two eras on alternate agents of each era's world and play 100 iterations.
> 2. Count the agents the later era's brains hold, as a share of all.
> 3. A switch's main effect: the mean of the worlds with it on, minus those with it off.
>
> **To make it again:** `python3 book_figures.py eightswitches`.
<!-- /figure -->

**Evolution goes on no better than in B1 itself.** Played against the
brains of iteration 500 for a lifetime, the brains of 1,000 hold more than
half of a contested world in six of the fourteen worlds that lived, with a
median of 0.46 — as in B1, whose ten lab worlds of 5,000 tokens gave a
median of 0.49 ([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md))
and whose two pilots of 10,000 tokens 0.37 and 0.68. But the fourteen scatter
far more: from nothing at all — in two worlds the brains of 500 took the
whole world in both eras' worlds, one of them the world with all eight
switches on, where on B2 the brains of 1,000 had taken it all — to the whole
world in two others. No switch moves the share beyond the spread of the
chains of pairs; the largest effect, of links that hold a token (−0.29), is
within it. Two of the eight are B2's own rules — no handover and brains that
change only when copied — and neither does it either (−0.02 and +0.14).

## What it teaches

- **A switch measured alone tells what it does in B2, not what it does.**
  Among combinations, mixing's riches and the riches of links that hold a
  token mostly vanish, and the lapse's fall in dimension halves. Pairs of
  switches act together often and strongly.
- **Three effects hold whatever else is on**: every unused link lapsing makes
  worlds rich, small and low-dimensional, and costs marks; the estate to the
  newborns makes them large and poor; mixing makes the agents differ more.
- **One of the sixteen meets the author's picture better than any**: B, D,
  F and G — no revolutions, no random inputs, links that hold a token, and
  mixing. It meets all seven marks at a dimension of 3.7, holds a quarter of
  its 652 agents at three tokens or fewer, with a diversity score of 0.68.
  It is one world of one seed, and among the candidates that are being
  played on to 10,000 iterations ([Healthy and rich](healthy-and-rich.md)).
- **Evolution goes on** in fifteen of the sixteen worlds, in most of them
  more slowly than in B2 itself; four switches speed it a little.
- **On B1** (added later) few switches stand out against how much pairs of
  switches act together there: the estate to the newborns makes a world
  larger, poorer and more alike, as on B2; no handover keeps more links;
  and no revolutions, which alone made B1 a world of mayflies, does not in
  company. Two of the sixteen worlds died out. And no switch makes B1's
  evolution go on: the later brains won in six of the fourteen worlds that
  lived, about as often as in B1 itself.

**Related notes:** [Many switches in few worlds](fractional-factorial.md) ·
[Every setting](settings.md) ·
[Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md) ·
[Who inherits?](who-inherits.md) ·
[Rules without numbers](rules-without-numbers.md) ·
[Healthy and rich](healthy-and-rich.md) ·
[What holds a world's size](what-holds-a-worlds-size.md)

<!-- turns -->
---

← [Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md) · [Changing the rules](../README.md#changing-the-rules) · [Revolutions, with every other rule](revolutions-with-every-other-rule.md) →

> [!info]- Pages that link here (6)
> [Rules in company](rules-in-company.md) · [Revolutions, with every other rule](revolutions-with-every-other-rule.md) · [Handover, with every other rule](handover-with-every-other-rule.md) · [Gifts, with every other rule](gifts-with-every-other-rule.md) · [Every simulation: what was run, and what was measured](every-simulation.md) · [Many switches in few worlds: fractional factorial designs](fractional-factorial.md)
<!-- /turns -->
