# Brains replaced by their average

> [!question] Questions of this page
> - Does a world need what its brains respond to, or only how they behave on
>   average?
> - If every [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") answered with the crowd's brains instead of its own, would
>   the world still be itself?
> - Of what brains give a world (habits, answers to what they see,
>   individuals that last, heirs, new variation), which does it need to stay
>   as it is?
> - How could we find out, and what would it cost?

## In short

This page is a plan that has not been played: a draft for the author, written
on 9 November 2026. Decisions by chance killed every world of [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") within three
[iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). But chance is a crude control: it takes away the brains' habits
along with everything else. The plan replaces it with a ladder of six ways to
decide, from dice to the world as it is, each taking away a little less than
the one below it. Every way is played on from the world kept at iteration
1,500 of five of round 3's shortlisted worlds, and its health is set beside
the health of the world's own replays. The draft thesis: chance kills; answers
borrowed from other agents keep a world alive but make it another world; and a
brain drawn afresh from the world's own at every look makes it itself again.
Three of the six ways need building, and all six need a run that can start
from a kept world. The runs would take about a day and a half on the
[lab](runs-and-records.md#the-lab-the-engine-and-pilots "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.")'s three workers.

## Why chance is not enough

Experiment 7 took the brains of B1's worlds away by chance
([Do the brains matter?](do-the-brains-matter.md)). With
[`random_decisions`](settings.md#random_decisions) on, every number a brain
would give is drawn from a bell curve instead, and the rules read their
decisions from that noise as usual
([The control: decisions by chance](the-brain.md#the-control-decisions-by-chance)).
All thirty worlds died, 25 after their second iteration and 5 after their
third. In each world's last [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), a [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") of 62% of the agents were cut off
and another 15% starved
([Chance cannot keep a world](do-the-brains-matter.md#chance-cannot-keep-a-world)).

That shows that the brains matter. It does not show what about them matters.
Chance takes everything away at once, even the habits every evolved world
shares. Drawn by chance, a parent gives its child all its [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") one time in
four, and starves: far more often than in any settled world. And brains whose
weights nobody chose, the founders', keep a world alive where chance kills it.
A brain drawn at random is still a function of what it sees: what it does on
one [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") hangs together with what it does on the next.

So [Do the brains matter?](do-the-brains-matter.md#chance-cannot-keep-a-world)
ended by proposing a sharper control. Draw every decision from the
distribution of that decision in the evolved worlds themselves: first as it is
overall, then given the agent's own tokens and connections. Agents that decide
like that behave, on average, like evolved ones, but respond to nothing. This
page plans that control, and the steps between it and the world as it is.

## What the brains of round 3 are like

In B1 the answer would be easy to guess. Its brains are nearly blind: a change
in what a brain sees reaches what it does about a thousand times smaller, a
gain of 0.0015
([Does a brain see what it is shown?](what-the-brains-are-like.md#does-a-brain-see-what-it-is-shown)).
Selection set only constants in them, the same in every world: few children,
the stake spread, every token revolutionary
([What evolution changed](what-the-brains-are-like.md#what-evolution-changed)).
Two of its agents in the same place stake nearly seven times more alike than
two brains drawn at random
([Two agents in the same place](b1-brains-lineages-and-messages.md#two-agents-in-the-same-place)).
A world of such agents needs little more than their average.

But since 7 November 2026 the notebook plays only worlds drawn at random and
worlds that follow round 3's most promising ones
([What could be played next](candidates-for-open-ended-evolution.md#what-could-be-played-next)).
Their brains are another matter. The five worlds this plan would play show it:

- **Their brains can see.** Every one squashes with tanh, through which a
  change still passes in several layers, where B1's sigmoid passes on at most
  a quarter of it in each ([`hidden_activation`](settings.md#hidden_activation)).
- **Their agents differ.** Put two agents of one world in the same situation,
  with the same random numbers, and they act measurably differently in two
  pairs in three (seed 3057) to nearly every pair (seed 3026): the
  [same-position share](measuring-health.md#the-same-position-share "Of two agents put in the same situation, how often they stake their tokens measurably differently.").
- **Children are only partly like their parents.** The heredity index of the
  five lies between 0.14 and 0.34, where 1 is a child that acts as its parent
  does and 0 one that acts like a stranger
  ([How each pillar is read](candidates-for-open-ended-evolution.md#how-each-pillar-is-read)).
- **Few lines carry the future.** The effective population of the five is
  between 11 and 263, so chance more than merit decides which line goes on
  ([Why it is not obvious](did-evolution-happen.md#why-it-is-not-obvious)).
- **Yet their health holds.** Scored over the first and the second half of
  their runs, 41 of round 3's 82 living worlds stayed as healthy as they were,
  and the median change was nil
  ([Healthier or sicker with time?](which-worlds-are-healthy.md#healthier-or-sicker-with-time)).
  And how unequal a world is depends, more than on any other rule drawn, on
  where the tokens of the dead go: a median [Gini](measuring-tokens.md#measuring-inequality-the-gini-coefficient-the-richest-tenth-entropy-and-mobility "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all.") of 0.32 where they go to the
  poorest, 0.53 where they flow to the nearest living
  ([What keeps wealth spread?](what-keeps-wealth-spread.md)).

So in these worlds the brains see, and differ, and their lines come and go by
chance, while the shape of the world holds still. What, of all that the brains
do, holds it?

## A ladder of six ways to decide

Picture a class sitting an exam in which every pupil gets questions of her
own. How much of the class's marks comes from each pupil's own mind? You can
take it away in steps:

- every pupil throws dice;
- every pupil copies the answers that a classmate, drawn at random, gave to
  her own questions;
- every pupil copies from a classmate who sits in a seat like hers;
- every question a pupil gets is answered by a classmate drawn at random, who
  reads the question itself: the class's knowledge, with nobody's own;
- every pupil answers for herself, but nobody learns anything new;
- the class as it is.

Each step gives back one thing: habits, a sense of one's own seat, a sense of
the whole question, a mind of one's own that lasts, and learning.

In a world, a pupil's questions are an agent's **look**: one column of inputs
for each of its [candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides."), itself and each [neighbour](the-world.md#the-network "An agent joined to another by a connection.")
([One look: a column per candidate](the-brain.md#one-look-a-column-per-candidate)).
Her answers are the columns of outputs a brain gives back, which the rules
turn into a child, its links, the stakes and the messages
([What a brain says](rules-in-detail.md#what-a-brain-says)). The six ways differ only in where
those outputs come from. The rules of the world stay as they are. That is how
the control by chance works too: a brain speaks in one place in the engine,
and only that place changes.

| | way to decide | where an agent's answers come from | what it keeps | what it takes away | exists? |
|---|---|---|---|---|---|
| (a) | by chance | a bell curve | nothing | everything | yes: `random_decisions` |
| (b) | a stranger's answers | the answers another agent's brain gave to its own look | the world's habits, in their right proportions | any answer to the agent's own situation | to be built |
| (c) | the answers of an agent like it | the same, lent by an agent with as many neighbours and the nearest tokens | habits, and an answer to its own tokens and links | any answer to its neighbours and to what they say | to be built |
| (d) | the crowd's brain | a brain drawn afresh at every look from the world's own, frozen, reading the agent's own inputs | everything the brains answer, as the crowd answers it on average | individuals that last, heirs, selection | to be built |
| (e) | its own brain, frozen | its own brain, which never changes again | individuals and heirs; lines can still win and lose | new variation | yes: mutation off |
| (f) | its own brain | the world as it is, played again from the same moment with a new seed | everything | nothing | yes |

Every way also needs one thing the lab cannot do yet: start a run from the
world another run kept ([What has to be built first](#what-has-to-be-built-first)).

**(a) By chance.** As in Experiment 7.

**(b) A stranger's answers.** At iteration 1,500, where a world is taken up
again, every living agent's own brain is run once on its own look, as
[How different are the agents?](b1-brains-lineages-and-messages.md#the-experiment)
runs brains in situations taken from the world itself. That gives a library of
looks: for each agent, a column of answers for itself and one for each of its
neighbours. From then on every look of every agent is answered from the
library. An agent drawn at random lends its own column for the borrower's own,
and for each of the borrower's neighbours one of its neighbour columns, drawn
at random. Each column is taken whole, so the two numbers that make a share
stay a pair, and the rows the rules average over a look come from one brain.
The rules then apply the borrowed answers to the borrower's own tokens and
links: the share a parent gives its child is the world's, the tokens it gives
are that share of its own. What is lost is any answer to the borrower's
situation.

**(c) The answers of an agent like it.** The same, but the lender is drawn
from the agents with as many neighbours as the borrower, and among them those
whose tokens are nearest the borrower's, on the scale a brain reads them,
ln(1 + tokens). Ties are drawn at random. No class needs a cut-off: the
lenders are the borrower's nearest in the world itself. With as many
neighbours, the lender's columns fit the borrower's one to one, and they are
dealt to its neighbours in a random order. Where no agent had as many
neighbours, the lenders are those with the nearest number.

**(d) The crowd's brain.** When the world is taken up again, every living
agent's brain is copied into a pool, which never changes again. From then on
every look of every agent is answered by a brain drawn from the pool at
random, by the world's own random stream, reading the agent's own inputs: its
tokens and links, its neighbours', and the messages they wrote. So an agent
answers its situation as the crowd would, on average over its brains. But
nobody has a brain of its own. The same agent may answer its next look with
another brain, a child inherits nothing, and no brain can spread, since every
look draws from the same pool. Physicists call such a stand-in a mean field:
every agent acts as the crowd does on average.

**(e) Its own brain, frozen.**
[`mutation_probability`](settings.md#mutation_probability) = 0, as in every
contest ([A contest, step by step](measuring-evolution.md#a-contest-step-by-step)).
Every agent keeps its brain, a child inherits its parent's, a node taken in a
game takes on the winner's, and lines can still win and lose. But no mutation
makes anything new. In the three worlds below that mix brains at birth
([`mix_at_birth`](settings.md#mix_at_birth)), a child's brain is still its
parent's mixed with a mate's, so new mixtures of old brains still arise. This
way is the fourteenth test of
[Did evolution happen?](did-evolution-happen.md#14--freeze-evolution-halfway),
freeze evolution halfway.

**(f) Its own brain.** The world as it is, played again from the same moment
with a new random stream: a replay, the twelfth test of the same page
([Replay the tape](did-evolution-happen.md#12--replay-the-tape)). The world as
it was played, the run's own iterations 1,500 to 3,000, is one replay more,
made already.

Each step up the ladder gives back one thing. So the difference between two
neighbouring ways says what that thing does for a world:

- from (a) to (b), the world's habits;
- from (b) to (c), an answer to one's own tokens and links;
- from (c) to (d), an answer to the neighbours and to what they say;
- from (d) to (e), individuals that last, and heirs: with them selection can
  work, so they bring the second and the third of the three things evolution
  needs, heredity and selection (Lewontin 1970);
- from (e) to (f), new variation, which keeps the first of them, variation,
  going.

## Five worlds, from iteration 1,500

Every run of round 3 kept its whole world every 250 iterations, brains
included ([Eras](measuring-evolution.md#eras)). The plan takes five of the 24
shortlisted worlds ([The shortlist](candidates-for-open-ended-evolution.md#the-shortlist))
as they stood at iteration 1,500, halfway through their runs. They are the
shortlist of 9 November 2026, when this plan was written; since health also
counts agents on [loops](measuring-networks.md#loops "A path through the network that comes back to where it started without using a connection twice.") per link, seed 3026 has left it.

| world | why this one | health's ranks (product, sum, worst) | agents | median tokens | heredity index | effective population | seconds an iteration | 1,500 iterations |
|---|---|---|---|---|---|---|---|---|
| seed 3042 | the healthiest by the product, and among the first 13 of all three orders | 1, 3, 2 | 12,390 | 2 | 0.14 | 247 | 12.0 | 5.0 hours |
| seed 3018 | third by the product; mixes brains at birth, and allows 8 links an agent at most | 3, 18, 3 | 2,853 | 7 | 0.16 | 263 | 4.7 | 2.0 hours |
| seed 3080 | the healthiest of the fast worlds | 4, 19, 4 | 2,154 | 8 | 0.34 | 11 | 0.9 | 23 minutes |
| seed 3057 | fifth by the product; the one of the five that does not mend its dead | 5, 26, 5 | 4,915 | 5 | 0.14 | 11 | 3.0 | 1.2 hours |
| seed 3026 | evolves the most of the round: 2,159 generations, and its line never comes back | 27, 41, 13 | 2,273 | 9 | 0.30 | 78 | 1.9 | 49 minutes |

The agents and their median tokens are those of the world kept at iteration
1,500. The seconds are each world's own, averaged over its iterations 1,500 to
2,999. The ranks come from [Which worlds are healthy?](which-worlds-are-healthy.md),
the heredity index and the effective population from
[Candidates for open-ended evolution](candidates-for-open-ended-evolution.md#the-shortlist).

Why iteration 1,500, and not the end of the runs?

- **One replay is free.** The world as it was played goes on from there to
  3,000. Its own second half is a replay to set every way beside.
- **The runs are shared.** Tests 12 and 14 of
  [Did evolution happen?](did-evolution-happen.md) start there too, so (e)
  and (f) are their runs, made once for both pages.
- **The worlds have settled.** Health does not drift over a run, so the
  second half of a world is what its first half promises.

## The experiment

From the world kept at iteration 1,500, each way to decide plays each world on
to iteration 3,000, under the world's own rules and the one change the way
names:

- (a) once, with seed 1: we expect it to die within a few iterations, and if
  it lives, it is played once more;
- (b), (c), (d) and (e) twice each, with seeds 1 and 2;
- (f) three times, with seeds 1, 2 and 3, beside the world as it was played.

A seed here starts a fresh random stream at iteration 1,500. Every
continuation is recorded as round 3 recorded its runs: statistics every
iteration, the heavier ones every 25, and a whole world every 250. So each can
be measured as the worlds of round 3 were measured, and opened in the viewer.
That makes 12 continuations a world, 60 in all, five of which are expected to
die at once.

## What is measured

**Health, over the last thousand iterations.** The six criteria of health
(decentralised, resilient, almost three-dimensional, diverse, not too even,
few tokens culled) are read from each continuation over its iterations 2,000
to 3,000, as [What healthy could mean](which-worlds-are-healthy.md#what-healthy-could-mean)
reads them. Each is scored against round 3's 82 living worlds over the same
iterations, as the halves of a run are scored there. The first 500 iterations
after the change are left for a world to settle into its new way.

One measure is left out of *diverse*: the same-position share. It puts agents'
own brains in the same situation, and in ways (a) to (d) no agent has a brain
of its own, so it would measure the pool rather than the agents. It is left
out of every continuation alike, the world as it was played included.

**The distance between two continuations** is the root mean square of the
differences of their six scores. It is 0 when the two score alike in every
criterion, and near 1 when one does better than nearly every world of the
round wherever the other does better than nearly none.

**Chance's yardstick.** The world as it was played and its three replays
differ only by chance. So the distances of the three replays from the world as
it was played say how far chance alone takes a world in 1,500 iterations. A
way to decide is **as close as chance** when the median distance of its
continuations from the world as it was played is no larger than the largest
replay's. It is **further than chance** when each of its continuations is
further than every replay.

**Plain numbers beside them.** For every continuation: its agents; the Gini
coefficient of its tokens; births per agent; nodes taken in a game; the share
of its tokens culled in a game; and whether and when it died, counted apart
([When a world ends](rules-in-detail.md#when-a-world-ends)).

**How it looks.** Every continuation is a run the author can open in the
viewer, beside the world as it was played.

## What we expect

**(a) should die,** as in B1. Round 3's rules spare more than B1's: four of
the five worlds mend the network where an agent dies, and in two a link holds
a token. But noise still gives a child everything one time in four, and leaves
most connections without tokens to keep them; that is how B1 lost three
quarters of its agents in a game.

**(b) and (c) should live,** since their agents spread their stakes as often
as the world's own do, so their connections carry tokens and stay. But they
answer nothing of what the agents around them are and do, which the brains of
these worlds see. So the worlds should become other worlds.

**(d) is the bet.** It keeps everything the brains answer, as the crowd
answers it on average, and takes away only who answers. If what holds a
world's shape is its rules and its crowd's habits of answering, (d) is the
world itself again. If it is its individuals (agents that keep one way of
acting for life, sit where their way suits them, and pass it to their heirs),
(d) drifts away. Agents of one kind sit in somewhat different places from
another's ([Do kinds live in different places?](b1-brains-lineages-and-messages.md#do-kinds-live-in-different-places)),
and agents that keep their ways can hold a pattern that a crowd mixed afresh
at every turn cannot, as cooperators hold out in space against defectors in
the games of Nowak and May (1992). So the bet could well be lost.

**For (e) we make no claim.** In Experiment 7, a world whose brains never
changed was filled by one founder's line within about a hundred iterations,
and became whatever that founder's brain made of it
([Lines by chance](do-the-brains-matter.md#lines-by-chance)). A world frozen
at 1,500 with as few as 11 effective ancestors could likewise come down to a
few brains, and become another world. But in three of the five worlds a
child's brain mixes its parent's with a mate's, and keeps making new brains
from old ones.

> [!quote] Draft thesis (to be fixed before any run)
> **Claim.** What a world needs from its brains is the way the crowd of them
> answers what each agent sees, not who answers. Played on from iteration
> 1,500, a world whose agents decide by chance dies. A world whose agents
> borrow another agent's answers, any agent's or one like their own, lives,
> but becomes another world. A world whose agents answer every look with a
> brain drawn afresh from the world's own is the world itself again, as close
> to it as its own replays.
>
> **Why.** Chance killed every world of B1 within three iterations
> (Experiment 7). The brains of these worlds can see, and two of their agents
> in the same situation mostly act differently, so answers that ignore the
> situation cannot stand in for them. But what holds a world's shape is
> steadier than its individuals: health does not drift over a run, the
> [share-out](b1-lives-and-wealth.md#gains-losses-and-the-share-out "How a cleanup deals the tokens of the dead out among the living: in the baseline, each token to a survivor drawn at random.") of the tokens of the dead sets how unequal a world is, and with 11
> to 263 effective ancestors, chance more than merit decides which line
> carries the brains.
>
> **Confirm.** In at least four of the five worlds, all three hold: (1) the
> continuation by chance has 20 agents or fewer before iteration 1,600; (2)
> (b) and (c) are each further than chance; (3) (d) is as close as chance.
> "Further than chance" and "as close as chance" are as defined above, from
> the six criteria of health over iterations 2,000 to 3,000.
>
> **Refute.** Any of the three fails in two or more worlds. If (3) fails, a
> world needs its individuals, or their lines. If (2) fails, it needs less of
> its brains than all they answer: their habits, or an answer to an agent's
> own tokens and links, are enough. If (1) fails, round 3's rules keep a world
> alive even when its agents decide by chance.

(e) and (f) carry no claim here: (f) is the yardstick, and (e) is the question
of test 14, which its own page leaves open.

## What has to be built first

Three things. None touches a run already made: every new rule is a setting,
off by default, so that every world played so far keeps its name and its
history.

1. **A run that starts from a kept world.** Today a run starts from its
   founders, or goes on from its own last checkpoint (`gol_run.advance`).
   Every way above needs a run that starts from the world another run kept:
   here, iteration 1,500 of a round-3 run, under that run's settings with one
   change, with a random stream of its own and a name that says where it
   began. The lab's plans need a way to say it. The recorder must begin at the
   kept iteration: the kept world holds every agent's parent and birth, so
   families can be counted on from there. And the viewer must show such a run
   from where it begins. **Its check:** started from the kept world with
   nothing changed and the kept world's own random stream, it must make the
   original run's [frames](runs-and-records.md#what-a-run-records "A whole world as a run records it after one phase: every agent, its tokens and connections.") again, frame for frame, to iteration 3,000, as a run
   resumed from a checkpoint must ([Is a run reproducible?](is-a-run-reproducible.md)).
   The same piece serves tests 12 and 14 of
   [Did evolution happen?](did-evolution-happen.md).
2. **Where an agent's answers come from**: a new setting whose default is
   the agent's own brain, as now, with three new values: the crowd's brain
   (d), a stranger's answers (b), and the answers of an agent like it (c). A
   brain speaks in two places in the engine: in the look of every phase
   (`_observe` in `GraphOfLife.py`), and in the look before each token
   when tokens are placed one at a time (`gol_rounds.py`). Both already give
   way to noise under `random_decisions`, and the new values plug in at the
   same two places. The pool of (d), and the library of (b) and (c), are made
   once, when the world is taken up again, and kept beside the run, so that it
   resumes with them, and so that a trial can start from any world it keeps.
   No new rule has a number in it: every draw is uniform, and the lenders of
   (c) are the nearest in the world itself. One limit: a library holds one
   look an agent, made as the world stood, so it has nothing to lend for the
   looks of a game played a token at a time. None of the five worlds plays
   that way.
3. **The analysis**: health over a stretch of iterations, scored against the
   round over the same stretch; the distances and chance's yardstick; and a
   table and a figure for this page.

(d) is the least to build, and it is also the sea of
[One agent in a sea of average agents](one-agent-in-a-sea-of-average-agents.md).

## What it would cost

**Computing.** From each world's own seconds an iteration (the table above), a
continuation of 1,500 iterations takes 5.0 hours for seed 3042, 2.0 for seed
3018, 1.2 for seed 3057, 49 minutes for seed 3026 and 23 minutes for seed
3080: 9.4 hours for one of each. The eleven continuations a world that should
live, all but chance's, make about 103 hours of computing: about a day and a
half on the lab's three workers. Seed 3042 alone is more than half of it;
without it, the rest take about 16 hours.

These are the worlds' own costs. Ways (a) to (c) run no brain, so an
iteration costs less; (d) runs one brain a look, as the world does. A world
that grows under a new way costs more, and one that shrinks less.

**Disk.** A continuation keeps its frames, from 0.15 MB an iteration (seed
3080) to 1.15 MB (seed 3042), and six whole worlds, from 8 MB each (seed 3080)
to 400 MB (seed 3042). That is about 7 GB for one continuation of each world,
and about 75 GB for the 55 that should live, 45 GB of it seed 3042's. The
external drive that holds round 3's runs had 408 GB free on 9 November 2026,
but Experiment 43, still playing, is expected to fill about 340 GB of it. The
disk inside the computer had 166 GB free.

**Building.** The run that starts from a kept world is the largest piece, as
it touches the lab, the recorder and the viewer. The three new ways are small,
as `random_decisions` shows: one switch at the place where a brain speaks.

## Choices for the author

- **Which worlds.** All five, about a day and a half on three workers; or the
  four without seed 3042, about 16 hours.
- **Where to start.** At iteration 1,500, as proposed, where the world's own
  second half is a replay and tests 12 and 14 share the runs; or at 3,000,
  where the runs ended. Then the world as it was played must be played on
  too, as
  [A hundred thousand iterations](a-hundred-thousand-iterations.md#the-long-run-planned-again-on-round-3s-candidates)
  proposes for three worlds.
- **Which ways first.** (a), (e) and (f) need only the run that starts from
  a kept world, and (d) the pool besides; (d) also makes the sea of the next
  page. (b) and (c) need the library.
- **How often (d) draws.** At every look, as proposed, the most thorough
  mixing; or once an agent each phase, so that what an agent says and what it
  then does in one phase come from one brain.
- **(c)'s lenders.** As many neighbours and the nearest tokens, as proposed,
  with no cut-off; or classes by the world's own fifths, the steps in which a
  brain reads its neighbourhood.
- **(e) in the worlds that mix brains at birth.** Mutation off alone, as
  proposed, one change; or mixing off too.
- **How many seeds.** Two a way and three replays, as proposed; or more, for
  a sharper yardstick.
- **A check where the answer is known.** B1's brains are nearly blind, so in
  B1 even (b) should be as close as chance. One B1 world played down the
  ladder would test the method itself. The author's rule of 7 November 2026
  leaves B1 out, unless the author wants this check.
- **The lines of the thesis.** Four worlds of five, iteration 1,600 for chance
  and the replays as chance's yardstick are proposals.

## Where to go next

- [One agent in a sea of average agents](one-agent-in-a-sea-of-average-agents.md):
  a world of (d)'s agents as a sea, and one agent with a brain of its own
  planted in it.
- [Do the brains matter?](do-the-brains-matter.md): the control by chance, and
  worlds whose brains never change.
- [Did evolution happen?](did-evolution-happen.md): fourteen tests, two of
  which would share this plan's runs.
- [How different are the agents?](b1-brains-lineages-and-messages.md#how-different-are-the-agents): how much of
  what an agent does comes from where it is, who it is, and chance.

## How this was made

No run of this plan exists, so this page has no box of runs, and no plan file:
the plan lives on this page until the author picks it. Then it becomes a plan
in `book/experiments/`, with its thesis as written here, fixed before any run.

The numbers come from round 3's own records, read on 9 November 2026 without
playing anything: health's ranks from `research/screen/health.json`; the
heredity index, the effective population and the generations from
[Candidates for open-ended evolution](candidates-for-open-ended-evolution.md#the-shortlist);
the seconds an iteration from each run's `stats.jsonl`, the field `_seconds`
of both phases of an iteration added, over iterations 1,500 to 2,999; the
agents and their median tokens from each run's kept world of iteration 1,500
(`archive/world_001500.npz`); and the disk from the frames and kept worlds of
each run. The same-position shares are the five worlds' averages in
`research/screen/health.json`.

**References.** Lewontin, R. C. (1970), The units of selection. Nowak, M. A.
and May, R. M. (1992), Evolutionary games and spatial chaos.

<!-- turns -->
---

← [Do the brains matter?](do-the-brains-matter.md) · [Brains: what they are, and whether they matter](../README.md#brains-what-they-are-and-whether-they-matter) · [How big should a brain be?](how-big-should-a-brain-be.md) →

> [!info]- Pages that link here (2)
> [Do the brains matter?](do-the-brains-matter.md) · [One agent in a sea of average agents](one-agent-in-a-sea-of-average-agents.md)
<!-- /turns -->
