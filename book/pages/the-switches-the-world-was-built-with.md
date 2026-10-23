# The switches the world was built with

> [!question] Questions of this page
> - The original algorithm came with switches of its own. Does any of them
>   keep evolution going?
> - What do revolutions do, in a world of nearly blind brains and in one of
>   brains that see?
> - Can gifts, tokens paid to keep a connection alive, make the shape of the
>   network a decision instead of a rule with a number in it?

## Why ask

The rules made for B2 have been judged by the tournaments of
[How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md): does a world keep
beating its own past? The switches the algorithm was built with were never
judged that way. Some were set for other reasons. Handover was taken out
because it breaks the network ([A world that cannot break](a-world-that-cannot-break.md)).
Revolutions are planned in
*What keeps wealth spread?*, for what they do to wealth.
Gifts have been off in every run of the notebook. The author asked whether they
had been tested for open-ended evolution. They had not, and each has a reason
it might matter.

**Revolutions** ([How a coalition takes a node](revolution.md)) are
the one form of collective action the original game has. Stakers who mark
part of their stakes as revolutionary can, together, take a node from the
single largest staker if they outweigh it. A rule that rewards standing
together is the kind of rule [Why evolution stops](why-evolution-stops.md) said
evolution needs, and B2 has had it on all along.

**Gifts** let an agent hand tokens to a neighbour during reproduction, as a
gift rather than a stake. A gift is a real transfer and it crosses a
connection, so it keeps that connection alive. The author built it to make
upkeep a decision. In the original algorithm a connection that carried no
tokens lapsed, so an agent could keep a neighbour only by staking on it.
With gifts, and with the accounting done after reproduction over a whole
iteration (`prune_after = reproduction`, `inactive_window = iteration`, the
form's default until 2026-10-02), an agent sees which of its connections are
about to lapse and can pay to keep them. B2 made upkeep a rule instead: a link
lapses only if its two ends share two other neighbours (`prune_needs = 2`).
That rule has a number in it. Gifts with every unused link lapsing
(`prune_needs = 0`) would put the network's shape in the brains' hands, which
is the kind of replacement for a number that
[Rules without numbers](rules-without-numbers.md) looks for.

**Sharing the estate by wealth** (`redistribution = by_tokens`) gives the
tokens of the dead to the survivors in proportion to what they hold, which
makes the cull concentrate wealth instead of spreading it. A world of rich
and poor plays a different game.

## The worlds

Pilots, two seeds each, in the engine's own code (`--proto engine=true`, see
[Pilot worlds](pilot-worlds.md)), at 10,000 tokens for 1,000
iterations, with a whole world kept every 100 iterations:

| world | what it changes |
|---|---|
| B1 | the baseline: brains of five layers, revolutions on |
| B1, no revolutions | `allow_revolutions = false` |
| B2 | a world that cannot break ([A world that cannot break](a-world-that-cannot-break.md)), revolutions on |
| B2, no revolutions | `allow_revolutions = false` |
| B2 with gifts | `allow_gifting`, `prune_after = reproduction`, `inactive_window = iteration` |
| B2 with gifts, every unused link lapsing | the same, and `prune_needs = 0` |
| B2, every unused link lapsing | `prune_needs = 0` alone |
| B2, estates by wealth | `redistribution = by_tokens` |

B2's pilot of seed 1 is the same world as the lab's run `B2-10000-s001` of
[Experiment 16](why-evolution-stops.md), agent for agent and token for
token: the pilots in the engine's code reproduce the lab exactly.

## What the switches did to the world

<!-- figure switches/life -->
![How each switch changes a world's life](../figures/switches/life.svg)

**How each switch changes a world's life.** One dot per world, the mean or median over iterations 500 to 1,000; the bar is the median of the two. Labels as in the figure below.

> [!example]- How to make this figure
> **Runs.** Pilot worlds of `research/oee/pilots/run.py` in the engine's own code (`--proto engine=true`), kept in `GraphOfLifeRuns/.pilots/S`, two seeds each (`-s1`, `-s2`), at 10,000 tokens for 1,000 iterations with a whole world every 100 (`--ckpt 100`): B1 (`S6-B1`, the settings of the study of the baseline), B1 without revolutions (`S7-B1norev`), B2 (`S0-B2`: `--set hidden_layers=16 message_amount=8 message_prepass=false allow_handover=false mutate_on_replication=true incumbency=1.5 birth_links=2 mend_on_death=true prune_needs=2 valence=8 keep_home=0.3 max_age=100 min_child=10`), B2 without revolutions (`S1-norev`), B2 with gifts (`S2-gifts`: `allow_gifting=true prune_after=reproduction inactive_window=iteration`), the same with every unused link lapsing (`S3-giftslapse`: also `prune_needs=0`), B2 with every unused link lapsing (`S4-lapse`) and B2 with estates shared by wealth (`S5-bytokens`: `redistribution=by_tokens`). The jobs are listed in `research/oee/pilots/jobs_S.txt`.
>
> **Data.** From each world's `series.jsonl`.
>
> 1. Of the rows from iteration 500 to 1,000: the nodes taken (`conquests`) and the births per 100 agents, averaged; the mean age (`meanAge`), averaged; the agents (`nodes`), median.
>
> **To make it again:** `python3 book_figures.py switches`.
<!-- /figure -->

<!-- figure switches/bar -->
![Each switch against the health bar](../figures/switches/bar.svg)

**Each switch against the health bar.** One dot per world, at its median over iterations 500 to 1,000; the bar is the median of the two; dashed lines are the health bar of [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md). *−rev*: no revolutions; *gifts, lapse*: gifts, and every unused link lapses; *lapse*: every unused link lapses; *by wealth*: estates shared by wealth. Cut shares and the best-connected agent's share are drawn on logarithmic axes, a cut of nothing at 0.0001.

> [!example]- How to make this figure
> **Runs.** Pilot worlds of `research/oee/pilots/run.py` in the engine's own code (`--proto engine=true`), kept in `GraphOfLifeRuns/.pilots/S`, two seeds each (`-s1`, `-s2`), at 10,000 tokens for 1,000 iterations with a whole world every 100 (`--ckpt 100`): B1 (`S6-B1`, the settings of the study of the baseline), B1 without revolutions (`S7-B1norev`), B2 (`S0-B2`: `--set hidden_layers=16 message_amount=8 message_prepass=false allow_handover=false mutate_on_replication=true incumbency=1.5 birth_links=2 mend_on_death=true prune_needs=2 valence=8 keep_home=0.3 max_age=100 min_child=10`), B2 without revolutions (`S1-norev`), B2 with gifts (`S2-gifts`: `allow_gifting=true prune_after=reproduction inactive_window=iteration`), the same with every unused link lapsing (`S3-giftslapse`: also `prune_needs=0`), B2 with every unused link lapsing (`S4-lapse`) and B2 with estates shared by wealth (`S5-bytokens`: `redistribution=by_tokens`). The jobs are listed in `research/oee/pilots/jobs_S.txt`.
>
> **Data.** From each world's `series.jsonl`.
>
> 1. Of the rows from iteration 500 to 1,000 that measured the network (every 10th), take the median of the worst share of the others one agent's loss cuts off (`nodeCutRisk`), the same for one link (`cutRisk`), the 2-core (`coreShare`), links per agent (`meanDegree`), the ball dimension and the best-connected agent's share (`maxDegreeShare`).
>
> **To make it again:** `python3 book_figures.py switches`.
<!-- /figure -->

**Revolutions shape B1 and do nothing to B2.** In B1 a revolution is common:
in every game, revolutions took 28 to 39 nodes per hundred agents, most of the
46 to 55 that changed hands. The world they make is calm and old. The average agent was 116 to 137 iterations
old, 3 agents in a hundred were born in an iteration, and the network grew
stars, one agent joined to 9 to 32% of the others. Take revolutions away and
B1 becomes a world of mayflies. Nodes changed hands 71 to 84 times per hundred
agents a game, 14 to 28 agents in a hundred were born in every iteration, and
the average agent was 7 to 13 iterations old. Its network had no stars (no
agent held more than 1% of the others), but it was stretched and fragile: a
ball dimension of 2.1 to 2.2, and a single agent whose loss would cut off 36
to 37% of the world. In B2, revolutions won about one node in three hundred a
game, and B2 with and without them were the same world in every measure.

**Gifts changed nothing that the network shows,** as long as B2's rule for
lapsing links held: a link lapses only if its ends share two neighbours. Then
links seldom lapse and a gift has little to save.

**Links that must be used break the network, gifts or no gifts.** With every
unused link lapsing, as in the original algorithm, B2's world held 1,300 to
1,450 agents instead of 3,400. One to three agents in a hundred were cut off
in every iteration, its ball dimension was about 2, and one agent's loss
could cut off a fifth of it. Gifts did not change that: the brains did not pay to keep
the links that mattered.

**Estates shared by wealth** made the world smaller, 2,450 and 2,800 agents
against 3,400, and otherwise left it as it was.

## Does any switch keep evolution going?

> [!warning] Added later: these contests were 20 iterations long
> Every tournament in this page was played with contests of 20 iterations.
> In B2's worlds such contests miss what pays off over an agent's life of 100
> iterations: in contests of 100 the brains of iteration 2,000 beat those of
> 500 (0.73) where in contests of 20 they tied
> ([Why evolution stops](why-evolution-stops.md#would-we-notice-if-it-went-on)). Where
> this page says that evolution stops, read: it stops *as far as a short
> contest can see*.
>
> The longer contests have since been played, 100 iterations each, once in
> each era's world (`research/oee/checks/rejudge.py`, seed 17); eras played
> against themselves that way hold 0.44 to 0.58. The brains of iteration 1,000
> against those of 500, in the two seeds: B2 itself 0.69 and 0.67; without
> revolutions 0.76 and 0.61; with gifts 0.87 and 0.83; with every unused link
> lapsing 0.65 and 0.53; with the estate shared by wealth 0.65 and 0.62; with
> gifts and lapsing links 0.25 and 0.62, in worlds whose networks were breaking.
> So **in B2 and its variants evolution went on** between iterations 500 and
> 1,000; the switches do not change that, they only change how fast. In B1
> the short contests were right: with revolutions 0.37 and 0.68, without them
> 0.35 and 0.16, where the later brains lost.

<!-- figure switches/elo -->
![Does any switch keep evolution going?](../figures/switches/elo.svg)

**Does any switch keep evolution going?** One dot per world. Top: the rating of the brains of iteration 1,000 against the founders'. Bottom: their share of a world contested with the brains of iteration 500 of the same world.

> [!example]- How to make this figure
> **Runs.** Pilot worlds of `research/oee/pilots/run.py` in the engine's own code (`--proto engine=true`), kept in `GraphOfLifeRuns/.pilots/S`, two seeds each (`-s1`, `-s2`), at 10,000 tokens for 1,000 iterations with a whole world every 100 (`--ckpt 100`): B1 (`S6-B1`, the settings of the study of the baseline), B1 without revolutions (`S7-B1norev`), B2 (`S0-B2`: `--set hidden_layers=16 message_amount=8 message_prepass=false allow_handover=false mutate_on_replication=true incumbency=1.5 birth_links=2 mend_on_death=true prune_needs=2 valence=8 keep_home=0.3 max_age=100 min_child=10`), B2 without revolutions (`S1-norev`), B2 with gifts (`S2-gifts`: `allow_gifting=true prune_after=reproduction inactive_window=iteration`), the same with every unused link lapsing (`S3-giftslapse`: also `prune_needs=0`), B2 with every unused link lapsing (`S4-lapse`) and B2 with estates shared by wealth (`S5-bytokens`: `redistribution=by_tokens`). The jobs are listed in `research/oee/pilots/jobs_S.txt`.
>
> **Data.** From each world's checkpoints, `ckpt_<iteration>.npz`.
>
> 1. For every pair of the eras 0, 100, 200, 500 and 1,000 play four contests, two in each era's world: every agent gets a brain drawn at random from one era or the other, mutation is switched off, and 20 iterations are played under the world's own rules (`research/oee/pilots/tourney.py --T 20 --reps 2`).
> 2. Fit one rating per era (Bradley–Terry, `research/oee/pilots/elo.py`), the founders at 0.
>
> **To make it again:** `python3 book_figures.py switches`.
<!-- /figure -->

**In B2, none does.** In every variant of B2 the brains of iteration 1,000
held between 0.47 and 0.57 of a world contested with those of iteration 500;
B2 itself held 0.50 in both worlds. Some switches give evolution more to learn
at first. With gifts, the brains of iteration 100 rated +263 and +157 against
the founders, against +102 and +85 in B2. With every unused link lapsing they
rated +520 and +256: in a world whose links break, the founders' careless play
is costly, so the first improvements are large. Then evolution stops as it
does in B2.

**In B1, revolutions decide how long it goes on.** With revolutions, B1's
brains rated about +350 at iteration 100 and changed little after. Iteration
1,000 held 0.54 and 0.41 against iteration 500. Without revolutions,
evolution went on longer. In one world the brains of iteration 1,000 beat
those of 500 in all four contests (0.73) and those of 200 in all four (0.82).
In the other they improved until iteration 500, to +383, and then fell back
(0.40). B1 without revolutions is a world of mayflies, with 15 to 28 times as
many births per agent as B2, and every brain changes after every game, so a
thousand of its iterations hold many more generations than a thousand of
B2's. Its contests are noisy too: single contests between its eras ended
anywhere from 0.09 to 1.00.

A third world, of seed 3, ran for 3,000 iterations to see whether its
evolution stops later or not at all. It stopped later, and then the world
itself nearly died. Its brains improved until iteration 500: the brains of
iteration 500 beat those of 0, 100 and 200 in eleven of twelve contests. Then
they lost ground: iteration 1,000 held 0.32 against iteration 500, and
iteration 1,500 held 0.11. Meanwhile the world shrank, from about a thousand
agents between iterations 1,000 and 1,750 to a few dozen after 2,000: 13 at
its smallest and 44 at iteration 3,000. A lab run would have counted it as
extinct. Contests played in its later eras' worlds often ended with fewer
than 50 agents, so their scores say nothing.

## What it teaches

**The original switches do not keep evolution going in a world that cannot
break.** Revolutions are too rare in B2 to matter, gifts and estates by wealth
change nothing, and links that must be used give the first generations more
to learn and then stop like the rest.

**Revolutions matter where they are common.** In B1 they are most of the
fighting, and they decide whether a world is old and calm, with stars, or
young and turbulent. With them, B1's brains stopped improving by iteration
200 in both worlds. Without them, two worlds improved until iteration 500
and then lost ground, and the third was still improving at iteration 1,000,
where its run ended. So evolution lasts longer in B1 without revolutions, but
not without end. The turbulent world it
lasts in can also die, as the world of 3,000 iterations nearly did.

**Gifts do not make the network a decision.** The author built gifts so that
an agent could pay to keep a connection it values. With every unused link
lapsing, the brains did not pay for the links that held their world together,
and the network broke as it does without gifts. So the candidate of
[Rules without numbers](rules-without-numbers.md), links that cost, does not replace
B2's rule for lapsing links, at least not as gifts stand.



<!-- turns -->
---

← [Rules in company](rules-in-company.md) · [Changing the rules](../README.md#changing-the-rules) · [Rules without numbers](rules-without-numbers.md) →
<!-- /turns -->
