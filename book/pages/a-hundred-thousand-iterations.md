# A hundred thousand iterations

> [!question] Questions of this page
> - Have the [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") simply not had the time to learn to cooperate? What does
>   a world do over a hundred thousand [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."), ten times longer than any
>   world of [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") has lived?
> - Does a world settle, or does it keep wandering? Does anything appear late
>   that was not there early?
> - How can a run that lasts a month be watched, so that it stops when
>   something goes wrong and says so when something stands out?

## In short

This experiment has not been played: it is a plan, written on 9 October
2026. One world of B1, the first [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2."), at 204,800 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), would be left
for 100,000 iterations and checked every 500. It asks whether B1's agents
fail to cooperate only because they lack the time; its thesis expects that
time is not what is missing. It needs three things the lab could not do:
keep frames only in short windows, advance a run in blocks, and keep
checkpoints. Since then, contests showed that B1's [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") stop learning
within 50 to 100 iterations, so a run made now would be planned on [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.").

## A plan that has not been played

In the worlds of B1 measured so far, nothing shows that agents cooperate.
They stake on kin as on strangers, and the same two brains rarely face each
other twice ([Do agents cooperate? Help, kin and partners](do-agents-cooperate.md)).
Is cooperation impossible under these rules, or only slow? The plainest way
to find out is to wait: leave one world alone for a very long time, and
look.

This page plans that test, Experiment 11. **It has not been played.** The
question, the thesis and the method below were written on 9 October 2026,
before any run of it existed. The plan is the file
`book/experiments/drafts/E11.json`. The lab is the program that plays an
experiment's runs ([The lab, the engine and pilots](the-lab.md)). It reads
only the plans in `book/experiments/` itself, so it cannot start this one.

Nor could it run this one yet. The run needs three things the lab could not
do on 9 October 2026 ([What has to be built first](#what-has-to-be-built-first)).
The lab must keep frames, the pictures of the whole world a run writes after
every phase, only in short windows. It must advance a run in blocks of 500
iterations, with a sentinel that judges each block. And it must keep
checkpoints for later. A **checkpoint** is a run's save: the whole world at
one iteration. From it the run goes on exactly as if it had never stopped
([Is a run reproducible?](is-a-run-reproducible.md#why-it-could-be-true)).
Most of the third has been built since; the first two have not. The plan
file says that, once they exist, it moves to `book/experiments/E11.json`
with its thesis unchanged.

Later work has also answered part of the question, and changed which world
the run would be made on ([What later work found](#what-later-work-found)).

## Why ask

Measuring the baseline world in every way turned a vague hope into a list.
The hope was that interesting things might emerge if we watch long enough.
The list is of rules that stand in the way
([What the measurements say about open-ended evolution](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution)).
That is a claim about time. When this plan was written, it had never been
tested.

No world of the notebook had then lived longer than 3,000 iterations. That
is how long the first thirty worlds of B1, at 10,000 tokens, were run
([The life of a B1 world](a-worlds-life.md)). The largest worlds, of up to
409,600 tokens, had lived 600
([How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)).
A reader may fairly ask whether cooperation is impossible here, or only slow.
Perhaps the brains need longer to learn it.

Three findings, on 9 October, said the question deserved a month of
computing.

- **The brains were still changing.** A brain's **gain** is how much of a
  change in what it sees reaches what it does. In the worlds of 10,000
  tokens, evolution raised the gain 2.5 times in 3,000 iterations
  ([What the brains are like](what-the-brains-are-like.md#does-a-brain-see-what-it-is-shown)).
  If it went on rising, the brains would one day see what is around them.
  Then every road to cooperation that needs seeing would open: recognising
  kin, answering a partner ([Do agents cooperate?](do-agents-cooperate.md)).
  The worlds of 100,000 tokens, examined later, had reached the same gain,
  0.0015, within 1,500 iterations: 2.4 times a [founder](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")'s.
- **The worlds wander, slowly.** Think of the level of a lake. The wind
  makes it jitter from minute to minute, and the seasons move it slowly.
  Average it over a day, and the jitter is gone, but the seasons are not.
  For this plan, the same was done to the 26 worlds of Experiment 2 that
  lived to the end. Their statistics were averaged over blocks of
  iterations, from iteration 500 on. Over blocks of 500 iterations the
  averages still keep between a third and two thirds of the spread of single
  iterations. A quantity that only jittered around a fixed level would keep
  about a twentieth (1/√500 ≈ 0.045). So the worlds wander, slowly, at every
  block length up to 500 iterations.

  The **Hurst exponent** *H* puts a number on this. The spread of a mean
  over *B* iterations shrinks with *B* as *B*^(*H* − 1). Noise, which
  forgets at once, has *H* = 0.5: its mean shrinks as 1/√*B*. A random walk,
  which never forgets, has *H* = 1: its mean does not shrink at all. These
  worlds have 0.8 to 0.9, measured over blocks of 10 to 500 iterations.

  Over longer stretches the worlds forget. A world's level is still much
  where it was a hundred iterations later, but five hundred later it is
  nearly independent of where it was. A world's spectrum says how much of
  its wander happens at each rhythm. For its number of agents it levels off
  for periods of 800 iterations and more
  ([How long a world remembers](autocorrelation.md#the-spectrum-in-the-baseline)).
  On those scales a world is pulled back to its level, like a random walk
  held near it by a spring. Whether a slower drift lies beneath this, on a
  scale of ten thousand iterations, is what 3,000 iterations cannot show.
- **Births fall as a world ages**, long after it has settled. In the worlds
  of 10,000 tokens they fell by 41% from iterations 100–200 to the end
  ([Births and deaths slow down](a-worlds-life.md#births-and-deaths-slow-down)).
  That was the clearest sign of an adaptation that the study of the
  baseline saw. An adaptation that takes ten thousand iterations would not
  have been seen at all. Later work dated the fall. In those worlds, births
  per agent fell until about iteration 1,250 and then held still. In the
  worlds of 100,000 tokens, most of the fall came before iteration 500. If
  it is an adaptation, it is finished early, and does not keep accumulating
  ([A turnover that settles late](what-the-baseline-world-is.md#a-turnover-that-settles-late)).

And two things said that nothing will change.

- **The obstacles are rules.** A node's brain is replaced almost every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."),
  so the same two brains rarely meet twice. Brains cannot see who is kin.
  And nothing an agent does creates anything that was not there before
  ([What the measurements say about open-ended evolution](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution)).
  Time does not change a rule.
- **The brains' rise has a ceiling.** Mutation jitters a brain's weights,
  and so spreads them; its rarer resets draw them back to a founder's. Like
  a bath with the tap running and the plug a little loose, the weights
  settle where the two balance. In the worlds of 10,000 tokens, the
  first-layer weights already sat at that balance by iteration 3,000. In
  those of 100,000 tokens they sat there by iteration 1,500
  ([What the brains are like](what-the-brains-are-like.md#what-a-brain-listens-to)).
  The rise of the gain looks like the approach to that balance, not a
  climb. The larger worlds, which reached the same gain within half the
  time, fit this picture.

## What later work found

Since this plan was written, other work has asked its question of other
worlds, and of B1 in another way.

**B1 stops learning early.** Contests play the brains of one moment of a
world, an era, against those of another
([Eras, contests and invasions](eras-and-contests.md)). A contest's score is
the share of the world the later brains hold at the end, and 0.5 is a tie.
In ten worlds of B1 at 5,000 tokens, nothing is learned after the first 50
to 100 iterations
([Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md)).
Every later era ties with the one before it. The brains of iteration 1,000
hold 0.48 of a world contested with those of iteration 300. Contests five
times as long agree
([The verdict of Experiment 12](how-to-tell-a-world-that-keeps-evolving.md#the-verdict-of-experiment-12)).
That speaks for the thesis below, before any run.

**A world whose brains see.** The work on [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau.") also built a
second baseline, B2
([Building B2: a world that cannot break](a-world-that-cannot-break.md)).
Its brains have one hidden layer instead of five, and a founder's gain is
about 0.2, where B1's is 0.0006. Brains that see had broken worlds under
B1's other rules; B2's rules keep its network whole
([Brains that see, and worlds that break](a-world-that-cannot-break.md#brains-that-see-and-worlds-that-break)).
In B2, evolution goes on for thousands of iterations, more and more slowly
([Why evolution stops](how-to-tell-a-world-that-keeps-evolving.md#why-evolution-stops)).
So if this run is made, it would now be planned on B2. It would need a plan
and a thesis of its own: this thesis expects a gain below 0.005, and B2's
founders start at about 0.2.

**Longer worlds.** Worlds have since lived far longer than 3,000
iterations. B2's worlds of Experiment 16 were played on to 10,000 as
pilots, and the eras of three of them were played against each other.
Their brains kept improving until about iteration 5,000, and then all but
stopped
([Would we notice if it went on?](how-to-tell-a-world-that-keeps-evolving.md#would-we-notice-if-it-went-on)).
Two worlds of B1 were played to 10,000 as pilots
([Every simulation](every-simulation.md#the-pilot-batches)). The longest
worlds of the notebook are two of B2 with brains mixed at birth, played to
50,000
([Five mixed worlds, fifty thousand iterations](mixed-at-birth-ten-long-worlds.md#five-mixed-worlds-fifty-thousand-iterations)).
Their brains climbed for about 20,000 to 25,000 iterations, then stopped
climbing and wandered ([Every era against three](every-era-against-three.md)).

The search for a healthy, rich world plans a long run of its own. Ten [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.")
would go to 10,000 iterations, and the best world on to a hundred thousand.
Brains mixed at birth went first and did not stay healthy, and no other rule
has yet held more often
([Healthy and rich](healthy-and-rich.md#candidates-and-long-runs)).

So for B1, part of the question is answered: its brains stop getting better
early. But no world of B1 has lived a hundred thousand iterations. Whether
its agents would learn to cooperate in that time is still open.

## How much evolution is a hundred thousand iterations?

A world of 204,800 tokens holds about 27,000 agents. After every game each
brain is changed with probability 0.2 ([How a brain changes](mutation.md)).
So some 5,400 changed brains are tried in every iteration, and some 500
million over 100,000 iterations.

How often does one line of descent take the whole world over? In the worlds
of 10,000 tokens, one branch of the family tree replaced all the others
about every 350 iterations
([Genotypes and lineages](genotypes-and-lineages.md)). A world twenty times
larger is slower. By how much, the plan left to
[Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens)
to measure, and it allowed for ten times as long.

That has been measured since. The time a world needs to come down to a
single founder's line grows as the tokens to the power 0.77. Ten times the
tokens take 5.9 times as long
([Verdict: slower than a front, faster than chance](how-does-a-worlds-size-follow-its-tokens.md#verdict-slower-than-a-front-faster-than-chance)).
If that holds up to 204,800 tokens, a world of that size needs about ten
times as long as one of 10,000. That is just what the plan allowed for.

Even so, one line would take the world over again and again in 100,000
iterations, dozens of times. Whatever is one change away from what the
brains do now would be found. What needs a rule changed would not. In a
settled world, the lines that take over carry no brains that play better.
They spread as a random walk does
([Selection, or chance?](genotypes-and-lineages.md#selection-or-chance)).

## The thesis

We expected time not to help. The obstacles to cooperation are rules, and
time does not change a rule. B1's brains are nearly blind, and the rise of
their gain looks like the approach to a balance. A hundred thousand
iterations try so many changed brains that anything one step away would be
found early on. Against this, the worlds wander slowly, and the gain had
risen. So the thesis says, in numbers, what would count against it. It was
written when no world had lived longer than 3,000 iterations, so its
"thirty times longer" is the count of that day.

> [!quote] The thesis of Experiment 11, written down before any of its runs existed
> **The claim.** Time is not what is missing. A world of 204,800 tokens left for 100,000 iterations, thirty times longer than any world of this book has lived, goes on wandering, but within bounds it reaches in its first few thousand iterations, and nothing appears late that was not there early. Its agents do not learn to cooperate: they stake on kin as they stake on strangers, the same two brains still rarely face each other twice, and the brains stay nearly blind.
>
> **Why it would be so.** Part III found what stands in the way of cooperation in the rules, not in a lack of time ([Chapter 40](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution)): a node's brain is replaced almost every game, so nobody meets anybody twice ([Chapter 38](do-agents-cooperate.md)); brains cannot see who is kin; and nothing an agent does creates anything. [Chapter 19](what-the-brains-are-like.md) found the brains nearly blind, with a gain of 0.0015 and their first-layer weights already, by iteration 3,000, at the balance between the jitters of mutation and its resets, so that selection can only set constants, and it has set them alike in every world. A hundred thousand iterations are a great deal of evolution: some 27,000 agents, each brain changed with probability 0.2 after every game, try some 500 million changed brains, and the living come to descend from a single founder every few hundred iterations ([Chapter 18](genotypes-and-lineages.md)), so one line takes the world over again and again, dozens of times at the least. What is one step away will be found; what needs the rules changed will not. Against this: evolution raised the gain 2.5 times in 3,000 iterations ([Chapter 19](what-the-brains-are-like.md)), and the baseline worlds are still moving at the longest scale yet measured: a mean over 500 iterations keeps between a third and two thirds of the spread of single iterations (a Hurst exponent of 0.8 to 0.9, in the 26 worlds of Experiment 2), so no world has yet been seen at rest.
>
> **It holds if:** From iteration 2,000 on, over at least 50,000 iterations, all three hold. (1) The wander levels off: for at least seven of the eight statistics (agents; [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") per agent; [Gini](gini-coefficient.md "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all."); births per agent; the share of stakes placed at home; the share of nodes kept; genotypes per agent; families), the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") difference between the means of 500-iteration blocks 20,000 to 50,000 iterations apart is at most twice the median difference between blocks 2,000 to 5,000 iterations apart. (2) In no three successive windows from iteration 5,000 on does the ratio of stakes on kin to stakes on others leave 0.85 to 1.15 (in the baseline worlds 0.89 to 1.10), or the share of connections whose two brains face each other again one game later rise above 10% (in the baseline worlds 3% to 9%). (3) At every kept checkpoint the median gain of the brains stays below 0.005 (in the baseline worlds 0.0011 to 0.0017).
>
> **It fails if:** Any one of: for two or more of the eight statistics, the difference at 20,000 to 50,000 iterations is more than twice that at 2,000 to 5,000, so the world goes on moving on scales the shorter runs could not see; or the kin ratio or the share of lasting partners crosses its line in three successive windows, so cooperation has appeared; or the median gain passes 0.005, so the brains have begun to see. A world that dies, or is stopped, before iteration 52,000 decides nothing.

*Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*

### Why "twice"?

The first test compares how far apart a world's block means are at two
lags. Blocks 2,000 to 5,000 iterations apart are about 3,500 apart on
average; blocks 20,000 to 50,000 apart, about 35,000, ten times as far.
Three kinds of world set the scale.

- **A world at rest** gives differences that no longer grow at all: the
  ratio is 1.
- **A random walk**, which never forgets, gives differences that grow as
  the square root of the lag: √10 ≈ 3.2 times.
- **A world that wanders as the baseline does**, slowly but pulled back to
  its level, stays close to 1. Its differences stop growing at lags beyond
  its memory of a few hundred iterations. Even a wander with *H* ≈ 0.85 at
  every scale would give only about 1.1 times.

Twice lies between a world at rest and a random walk. So the first test
asks whether a new, slower drift appears, one that 3,000 iterations could
not show.

## Method

### One world

The baseline B1 at 204,800 tokens, seed 1, for up to 100,000 iterations.
One world, because the question is about time within a world. If something
stands out, seeds 2 and 3 are run to the same point. They tell whether it
belongs to this world or to the rules.

Why 204,800 tokens, rather than an even 200,000? Experiment 8 already ran
this world, seed for seed, for 600 iterations (`B1-204800-s001`,
[How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)).
The first frames the long run keeps, those of iterations 483 to 499, must
be that run's frames again. That is a free check that a run is still the
same run a month and several program versions later
([Is a run reproducible?](is-a-run-reproducible.md)).

### In blocks of 500, with a sentinel

The lab advances the world 500 iterations at a time. After each block a
**sentinel**, a program that keeps watch over the run, reads the block's
statistics and asks two questions.

*Is the run healthy?* If not, the run stops. It can be taken up again from
its last checkpoint, once someone has looked.

| the sentinel stops the run when | because |
|---|---|
| the world dies (20 agents or fewer) | every run ends there ([When a world ends](extinction.md)) |
| the tokens no longer add up to 204,800, or a statistic is missing or not a number where it never was | that is a fault in the program, not a finding |
| an iteration takes more than twice as long as in the ten blocks before, two blocks in a row | something has gone wrong with the machine, or the world has fallen into a shape the program handles badly |
| memory or disk would run out within the next ten blocks | the run would fail half-way through a block |

*Does anything stand out?* That never stops the run: a world doing
something new is what this experiment is for. It is flagged instead.

| the sentinel flags | when |
|---|---|
| **outside the known** | a statistic per agent or a share lies outside the band of the thirty worlds of [Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens), for three blocks in a row |
| **a new level** | a statistic's block mean lies beyond the range of all its earlier blocks by more than half that range, for five blocks in a row |
| **cooperation** | the kin ratio or the share of lasting partners leaves the baseline worlds' range, in a window (below) |
| **brains that see** | the median gain at a kept checkpoint passes 0.005 |

When something is flagged, the lab keeps the checkpoint of the block before
it. The flag appears beside this page in the Research Notebook tab, and in
the lab's log.

The rule for a new level asks for so much because the worlds wander. For
this plan, a simpler sentinel was tried on the 26 baseline worlds. It
measured each block of 100 iterations from iteration 1,500 on against the
blocks of iterations 500 to 1,499 of the same world. The unit was how much
those earlier blocks spread. Six such units sounds safe. Yet 5% of all
blocks went beyond it, and runs of three such blocks in a row were common.
They were most common in the largest [degree](degree.md "The number of an agent's connections: how many neighbours it has."), the median degree and the
number of leaves (agents with a single connection). Before the run starts,
the rule above is tried on the same worlds, and it must stay quiet there.

### What is kept

At this size a frame is about 0.65 MB after reproduction, and 1.6 MB after
the game, with the agents' decisions. Every frame of 100,000 iterations
would need more than 200 GB, more than the disk holds. So the run keeps:

- the **statistics** of every frame, as every run does (`stats.jsonl`;
  [What a run records](frames-and-stats.md)): about 400 MB for the whole
  run;
- **frames**, with the decisions, only for the last 17 iterations of every
  block. That is enough for every measure of
  [Do agents cooperate?](do-agents-cooperate.md), which looks up to 16 games
  ahead: about 8 GB;
- a **checkpoint** at the end of every block, with every brain, every token
  and the state of the random stream. A copy is kept every 10,000
  iterations, and before anything that stands out: 0.8 GB each.

Nothing is lost by keeping so little. A run is reproducible
([Is a run reproducible?](is-a-run-reproducible.md)). So any stretch of the
world can be made again exactly, frames and all, from the checkpoint before
it. A frame that was not kept is a frame that has not been made again yet.

### What is measured

- **The wander**, as a **variogram**. For each of the eight statistics of
  the thesis, it takes the median difference between the means of two
  blocks. It sets that against how far apart in time the blocks are, from
  500 iterations to 50,000. The idea comes from geology, where it asks how
  different two samples of rock are, as a function of how far apart they
  were taken. A world at rest gives a curve that levels off; a world still
  moving, one that keeps rising; a change of state, a step.
- **Cooperation**, in every window, by the five measures of
  [Do agents cooperate?](do-agents-cooperate.md):
  - how related [neighbours](the-world.md#the-network "An agent joined to another by a connection.") are;
  - the ratio of stakes on kin to stakes on others;
  - whether a node taken stays in the family;
  - how often a stake is answered with an equal one;
  - how long the same two brains face each other.
- **The brains**, at every kept checkpoint: the gain and the probes of
  [What the brains are like](what-the-brains-are-like.md). They say what a
  brain responds to, and how much.
- **Everything of [Thirty worlds](a-worlds-life.md#thirty-worlds)**, as
  block means over the whole run. They are set beside the bands of the
  baseline worlds and of
  [Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens).
- **Later, from the kept checkpoints:** a tournament across time, the
  brains of iteration 10,000 against those of iteration 90,000, in one
  world. A later population that beats an earlier one is adaptation that
  accumulates: P2, the second step on the ladder of
  [What this notebook is about](what-this-notebook-is-about.md#a-ladder-of-four-steps).
  The study of the baseline proposed this tournament
  ([How we will know](what-the-baseline-world-is.md#how-we-will-know)). It
  has since become the notebook's yardstick
  ([Eras, contests and invasions](eras-and-contests.md)).

### What it costs

On 9 October 2026, the lab's cost model gave about 27,300 agents, 25
seconds an iteration and 5.9 GB of memory. That makes 29 days for 100,000
iterations on one worker ([What a simulation costs](what-a-simulation-costs.md)).
The run keeps about 20 GB, where every frame would have needed over 200. The
thesis can be decided from iteration 52,000, about two weeks in, and the run
can be stopped after any block.

These costs may be too low. The same model gave the worlds of
[Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens)
12 seconds an iteration, and they took 17. They are also B1's costs. A run
on B2, whose brains are smaller, would have to be costed again. The model
has been fitted since to 246 recorded runs.

The run was to start once the thirty worlds of 100,000 tokens were analysed,
because the sentinel's bands of the known come from them. That analysis was
done on 11 October 2026, so the bands exist.

## What has to be built first

On 9 October 2026 the lab could not do three things this run needs. Each is
a change of its own, to be tested before the run. The first two still do
not exist, and most of the third does. Moved into `book/experiments/` as it
stands, the plan would be refused: a plan cannot yet say `frames`, `blocks`
or `keep_checkpoints`, as this one does.

1. **Frames in windows, statistics always.** Today a run that records its
   statistics as it goes must keep every frame. The reason is the count of
   families: it climbs the family tree, so it needs every iteration in order
   ([Families](genotype.md#families)). The recorder will instead read every
   frame as it is made, and the store will keep only the windows. A resumed
   run then takes its families up from the window before its checkpoint,
   which is why checkpoints fall at the ends of windows. This touches the
   engine: the run loop, the recorder and the settings. So it is made only
   while no experiment runs, and only with the proof that a run recorded
   whole is unchanged. Its test: a run recorded in windows has, row for row,
   the statistics of the same run recorded whole. And the frames it kept are
   that run's frames.
2. **Blocks and a sentinel.** The lab advances a run to the end of its next
   block, then judges it. The health rules stop it. The rules for what
   stands out flag it and keep a checkpoint. They write a line per block
   beside the run (`blocks.jsonl`) and into the Research Notebook tab.
3. **Kept checkpoints.** Copies of a run's checkpoint at chosen iterations,
   which the lab keeps and the analysis can open. They serve the brains'
   probes, making a stretch again, and the tournament. Most of this exists
   now. Since 12 October 2026, a run made with
   [`archive_every`](settings.md#archive_every) keeps an **archive**: whole
   worlds at fixed iterations, never overwritten. Each holds the network,
   every token and every brain, as a checkpoint does. The tournament they
   were to serve exists too: it is the yardstick of
   [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md)
   (`gol_tournament.py`). What an archive does not do is keep a world
   because something stands out; that needs the sentinel.

With them comes an analysis of its own, which draws the variograms, the
windows of cooperation and the gains, and decides the thesis.

## Results

None yet: the run has not been made. It waits for what has to be built
first.

## Where to go next

- [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md):
  the yardstick, and why B1 stops learning within its first hundred
  iterations.
- [Eras, contests and invasions](eras-and-contests.md): how a world's brains
  are played against their own past.
- [Every era against three](every-era-against-three.md): the longest worlds
  of the notebook, and where their evolution stopped climbing.
- [Do agents cooperate? Help, kin and partners](do-agents-cooperate.md): the
  measures this run would take in every window.

## How this was made

No run of this experiment exists, so this page has no box of runs. The plan,
`book/experiments/drafts/E11.json`, holds every setting named above, and
asks for one worker. It also records the graph statistics, which cost many times the
rest, every 25 iterations, and starts its analysis at iteration 2,000.

The numbers behind the plan were measured on 9 October 2026. They are the
wander, the trial of a simpler sentinel, the ranges of cooperation and of
the gain, and the sizes on disk. They come from the 26 worlds of
Experiment 2 that lived to the end, and from Experiment 8's worlds of
204,800 tokens. The costs come from the lab's model. The working notes,
`research/Plans.md`, list them all, to be measured again when the run is
built.

## How our view changed

- **14 October 2026.** From 9 October, this page planned the run on B1,
  then the notebook's only baseline. A note added on 14 October said that if
  the run is made, it would be planned on B2. Contests between eras had
  shown that B1's brains learn nothing after their first 50 to 100
  iterations
  ([Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md)).
  And B2 had been built, a baseline whose brains see
  ([Building B2: a world that cannot break](a-world-that-cannot-break.md)).
- **24 October 2026.** From 9 October, this page said that the longest any
  world of the notebook had lived was 3,000 iterations. Its questions called
  a hundred thousand "thirty times longer than any world". That held until
  15 October 2026. That night the first of B2's worlds played on from
  Experiment 16 reached 10,000 iterations, and two of B1 followed as pilots
  ([Every simulation](every-simulation.md#the-pilot-batches)). By 17 October
  two worlds of Experiment 18 had lived 50,000
  ([Five mixed worlds, fifty thousand iterations](mixed-at-birth-ten-long-worlds.md#five-mixed-worlds-fifty-thousand-iterations)).
  This rewrite corrected the page; the thesis keeps the words of its day.
- **24 October 2026.** From 9 October, this page said that no world had
  been seen at rest. It read the Hurst exponent of 0.8 to 0.9 as a wander
  "at every scale that 3,000 iterations can show". Its reason for "twice"
  added that differences growing as that wander does (*H* ≈ 0.85) would
  grow about seven times over a tenfold lag. Both went too far. The exponent
  was measured over blocks of 10 to 500 iterations only. Over longer
  stretches, the spectrum of the same worlds' number of agents levels off,
  from periods of 800 iterations on. A world is pulled back to its level,
  as a page of the day before had found
  ([Power laws, real and apparent](power-laws-real-and-apparent.md#the-rhythm-of-the-wander)).
  And by the measure this page uses, a random walk has *H* = 1, so 0.8 to
  0.9 is a wander around a level. Its differences level off: about 1.1 times
  over a tenfold lag, not seven. The threshold of twice still lies between a
  world at rest (1) and a random walk (3.2), and the thesis keeps its words.
  This rewrite found both slips.

<!-- turns -->
---

← [Every era against three](every-era-against-three.md) · [Does evolution keep going?](../README.md#does-evolution-keep-going) · [What would count as open-ended evolution](open-ended-test.md) →

> [!info]- Pages that link here (6)
> [What this notebook is about](what-this-notebook-is-about.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [Every era against three](every-era-against-three.md) · [Healthy and rich](healthy-and-rich.md) · [Brains mixed at birth, in long worlds](mixed-at-birth-ten-long-worlds.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
