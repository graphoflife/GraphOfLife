# One agent in a sea of average agents

> [!question] Questions of this page
> - In a world where every [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") decides as the crowd does, can one agent
>   with a [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") of its own do better?
> - Can it draw [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") from the crowd without running dry or being cut off,
>   and can its line spread, by its children or by the nodes it takes?
> - Is the brain whose line won in a world better than the crowd, or only
>   luckier?
> - What would the experiment need, and what would it cost?

## In short

This page is a plan that has not been played: a draft for the author, written
on 9 November 2026. It takes the sea of
[Brains replaced by their average](brains-replaced-by-their-average.md): a
world of round 3 played on with every agent answering every look with a brain
drawn afresh from the world's own, so that nobody is anyone in particular.
Into the sea it plants one agent, a leaf holding the sea's [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") tokens, with
a brain of its own that never changes and that its children and the nodes it
takes inherit. Four plantings are made at the same places and moments: a
marked agent of the sea, whose heirs keep deciding as the sea does, which is
chance alone; a brain of the sea itself; the brain of the world's main line;
and a brain of the world's future. Biologists ask this as whether a rare type
can invade, physicists as a small system in touch with a reservoir. The draft
thesis: no brain gains by being an individual, and none takes hold in more
than one planting in ten. It needs the sea, a few additions to the invasion
test, and at most about 20 hours on three workers for the three fastest
worlds.

## The idea

Picture a market in which nobody is anyone in particular. At every deal, a
trader acts as some trader of the market, drawn at random, would act in her
place. The market as a whole behaves as it always has; no trader has a way of
her own. Into it comes one trader who does have one, and keeps it. Her heirs
take it on, and so does every shop she takes over. Can she grow rich off the
crowd? Does she last? Does her way spread?

Biologists ask this of a rare mutant in a resident population. Its growth
rate while it is still rare, in a population that does not notice it, is its
**invasion fitness** (Metz, Nisbet and Geritz 1992). A population that no
rare mutant can invade is **evolutionarily stable** (Maynard Smith and Price
1973). The sea makes the question clean. Its crowd never changes, never
evolves an answer, and has no lines of its own for the newcomer to meet.

Physicists ask it of a small system in touch with a large **reservoir**: a
heat bath that gives and takes energy without itself changing, or, in the
grand canonical ensemble, energy and particles (Gibbs 1902). Here the
newcomer's line trades tokens and agents with the sea. If it neither dies out
nor takes over, but keeps a size and a wealth of its own while its members come
and go, it is a bound state: the kind of lasting thing in a restless sea that
[Vacuum and matter](physical-inspiration.md#vacuum-and-matter) hopes for, as
matter in a vacuum.

## What the notebook already knows

**The invasion test.** The notebook plants strategies in evolved worlds and
watches whether they spread ([The invasion test](measuring-evolution.md#the-invasion-test)).
A patch of agents gets a strategy in place of its brains. The strategy spreads
as a brain does, into children and into nodes its agents take in a [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), and
it never mutates. The same agents given frozen copies of one resident's brain
are the control. A planting **takes hold** when, 300 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") later, it is
alive, larger than it began, and larger than its control.

**A crowd, but never one.** In [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s evolved worlds, a raider written by hand
took 40% to 62% of three worlds in ten from a patch of 5% of the agents, where
the residents' own brains, planted the same way, never ended with more than
8% ([What the strategies did](strategies-written-by-hand.md#what-the-strategies-did)).
But it needed a crowd. No patch of up to 30 raiders took hold in 20
plantings, and a hundred raiders took hold 4 times in 20. Not one of 30 lone
raiders took hold, and not one of 20 lone residents was even alive 300
iterations after it was planted
([How large a crowd a raider needs](strategies-written-by-hand.md#how-large-a-crowd-a-raider-needs)).
In older worlds even the crowd failed, for want of tokens: a raider needs
three to take a [neighbour](the-world.md#the-network "An agent joined to another by a connection.") of one
([What a raid starts from](strategies-written-by-hand.md#what-a-raid-starts-from)).

**Two roads for a brain.** A brain spreads into a child at birth, and into a
node its agent takes in a game. In [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it."), conquest spreads a brain seventeen
times more than birth does
([Two roads of inheritance](questioning-the-mechanics.md#two-roads-of-inheritance)).
In round 3 the roads are more even. In the five worlds of
[Five worlds, from iteration 1,500](brains-replaced-by-their-average.md#five-worlds-from-iteration-1500),
over iterations 1,500 to 3,000, 62% to 77% of the nodes change hands in a
game, and 0.12 to 0.92 children are born per agent in an iteration. So a node
is taken from 0.8 times as often as a child is born ([seed](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 3026) to five times
as often (seeds 3018 and 3080). A line has to be followed down both roads.

**Room for better play.** In every long world of B2 the notebook has
measured, up to 50,000 iterations, an agent's best reply to the others would
have held about one node more after a game than its own play, at every era
([How much room is left](every-era-against-three.md#how-much-room-is-left)).
Better play than the crowd's exists, at least for one game. Whether any brain
a world made plays it, and whether it pays over a [lifetime](measuring-lives.md#age-lifetime-and-survival-curves "How many iterations an agent lives, from its birth to its death."), is another matter.

**Lines by chance.** In Experiment 7, lines took over worlds whose brains
never changed, by chance alone
([Lines by chance](do-the-brains-matter.md#lines-by-chance)). And the five
worlds have effective populations of 11 to 263: chance more than merit decides
which of their lines goes on
([Why it is not obvious](did-evolution-happen.md#why-it-is-not-obvious)).

## The sea

The sea is the fourth way to decide of
[A ladder of six ways to decide](brains-replaced-by-their-average.md#a-ladder-of-six-ways-to-decide),
the crowd's brain. A world of round 3 is played on from the world it kept at
iteration 1,500. From then on, every look of every agent is answered by a
brain drawn at random from a pool: copies of the brains of every agent alive
at 1,500, which never change again. An agent answers what it sees as the crowd
would, on average over its brains. But nobody has a brain of its own, nobody
inherits one, and no brain can spread, since every look draws from the same
pool. The sea's tokens, links, births and deaths go on moving; its crowd of
brains stays as it was.

The seas are that page's runs of the crowd's brain, two for each of its five
worlds, each keeping a whole world every 250 iterations. Their kept worlds of
iterations 2,000, 2,250, 2,500, 2,750 and 3,000 are five moments of each sea,
after 500 iterations to settle. A world whose sea dies has none, and that page
says first whether it does.

The seas are large beside one agent. Each world holds 30,000 tokens, and when
the sea begins between 2,154 agents (seed 3080) and 12,390 (seed 3042).

## The plantings

**Where.** A leaf, an agent with a single link, holding the sea's median
tokens, drawn at random among such leaves; if none holds the median exactly,
among those nearest it. A leaf touches the sea at one point only. It is the
humblest start there is: every further link the newcomer's line must win for
itself. And the median holding gives it neither a rich start nor a poor one.
A leaf is also fragile: if its one link carries no tokens in a game, it is
cut off. The marked agent planted on the same leaf meets that risk too, so the
comparison takes it out. After a game, 11% to 25% of the agents of the five
worlds are leaves, and a newborn starts with 1.3 to 4.2 links on average.
When the seas begin, the five worlds have from 15 (seed 3080) to 361 (seed
3042) leaves holding their median, of 2 to 9 tokens.

**Who.** Four kinds of planting are made, each in the same places at the same
moments:

1. **A marked agent of the sea.** Nothing about it changes but a mark, which
   its children and the nodes it takes inherit. Every one of them still
   answers as the sea does. Its line is chance alone: the yardstick.
2. **A brain of the sea**, drawn at random from the sea's own pool and given
   to the agent as its own. The sea plays this brain too, now and then, in
   every agent. Here one agent plays it always, and passes it on. This is the
   invasion test's control, moved to the sea.
3. **The brain of the world's main line**: the agent alive at iteration 1,500
   that most of the world's agents at 3,000 descend from, in the world as it
   was played, with the brain it had then, as
   [How each pillar is read](candidates-for-open-ended-evolution.md#how-each-pillar-is-read)
   finds it. The line that won.
4. **A brain of the world's future**, drawn at random from the world as it
   was played at iteration 3,000. It has 1,500 more iterations of evolution
   behind it than the sea's brains.

A brain of the sea and a brain of the future are drawn afresh for every
planting, so their plantings try many brains; the main line is one brain,
planted every time. A planted brain never changes, its children inherit it
whole, even in a world that mixes brains at birth, and a node its agent takes
in a game takes it on, as a strategy's does. It is never again drawn from the
pool.

**How many.** Three places at each of the five moments of each of a world's
two seas: 30 plantings of each kind in each world. A place also fixes the
random stream, so the four kinds planted there meet the same chances until
their paths part, as a strategy and its control do.

**How long.** 300 iterations, as the [lab](runs-and-records.md#the-lab-the-engine-and-pilots "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.")'s invasion tests are played, or until
the line has died out.

## What a planting measures

A planting's **line** is every agent that carries its brain, or its mark.
After every iteration we count:

- **its agents**, and how each new one came: born to an agent of the line, or
  a node the line took in a game; and how many the line lost;
- **its tokens**, and its **extraction rate**: the line's tokens per agent,
  set against the sea's mean holding. A line that holds more per agent than
  the sea does is drawing tokens from it;
- **the sea around it**: the mean holding of the sea's agents one, two and
  three steps from the line, as a share of the sea's mean holding everywhere.
  Below 1, the sea near the line runs dry. At 1, the sea acts as a reservoir,
  too large for the line to change;
- **when it died out**, if it did.

From these, a planting's success is read four ways, because they need not
agree. A hoarder can gain tokens and never take a node; a brain can spread by
its children while its line grows poorer.

- **It lives:** its line is alive after 300 iterations.
- **It gains:** its line holds more tokens than the planted agent held.
- **It spreads:** its line has more than one agent, by births, by conquests,
  or both, each counted.
- **It takes hold:** it lives and spreads, and has more agents than the marked
  agent planted in the same place at the same moment: the lab's test of an
  invasion, with the marked agent as its control.

Over all its plantings, a kind's **presence** is its line's agent-iterations:
its agents added up over the 300 iterations, counting nothing after it has
died out. A line that dies in its first iteration scores 1; one that holds a
single agent throughout scores about 300. Presence counts how long a line
lives and how large it grows together. Its mean over the plantings is a rough
measure of invasion fitness, since a line that does exactly as well as the
sea keeps, on average, the size it began with.

## What we expect

Every lone planting the notebook has made has failed, whatever it carried: 30
raiders and 20 residents. In the five worlds, more than six nodes in ten
change hands in a game. So a lone brain is copied and lost many times within a
few iterations, and chance ends its line, or spreads it, long before a small
advantage could show. The sea is the mixture of the world's own brains, which
evolved among one another: if the world sits near a balance among them, none
does much better against the mixture than the mixture does itself. And the
line that won in the world as it was played won among 11 to 263 effective
ancestors, where luck decides much.

So we expect the four kinds to do alike: lines that die young, a few that
spread for a while, and almost none that take hold. If one kind does better,
the future's is the likeliest. Its brains are what 1,500 more iterations of
evolution made, and if that evolution climbed and did not only drift, they
should draw on a sea made of their past.

> [!quote] Draft thesis (to be fixed before any run)
> **Claim.** In a sea of a world's average, no brain gains by being an
> individual with heirs. Planted alone as a leaf holding the sea's median
> tokens, a brain of the sea itself, the brain of the world's main line and a
> brain of the world's future each leave a line that lives no longer and grows
> no larger than the line of a marked agent of the sea planted in the same
> place at the same moment; and none takes hold in more than one planting in
> ten.
>
> **Why.** A lone planting has never taken hold in the notebook: not one of 30
> lone raiders, though a hundred raiders together took hold 4 times in 20, and
> not one of 20 lone residents was even alive 300 iterations later
> (Experiment 29). In these worlds 62% to 77% of the nodes change hands in a
> game, so a lone brain's line is ended or spread by chance long before a
> small advantage shows. The sea is the mixture of the world's own brains,
> which evolved among one another, so none should do much better against it
> than it does itself; and the main line won among 11 to 263 effective
> ancestors, where chance more than merit decides.
>
> **Confirm.** In all but at most one of the worlds whose sea lives, for each
> of the three brains, over its 30 plantings, each paired with the marked agent
> planted in the same place at the same moment: (1) its presence does not
> exceed the marked agent's by more than chance allows: flipping the signs of
> the 30 paired differences 10,000 times, at least one time in twenty gives a
> total as large as the one found (a one-sided [permutation test](statistics.md#permutation-tests "Whether a difference could be chance, found by shuffling which world belongs to which condition."), *p* ≥ 0.05);
> and (2) it takes hold in at most 3 of its 30 plantings.
>
> **Refute.** In two or more worlds, a brain's presence beats the marked
> agent's (*p* < 0.05), or the brain takes hold in more than 3 of 30. Then
> being an individual pays in a sea of the average, and the brain that does
> it shows what evolution, or luck, put into it.

## Further steps

These follow only if the author wants them, and most only once the sea has
answered.

- **Kin.** If no lone brain takes hold, the future's brain is planted with
  kin: the leaf and its nearest agents, 10 and 100 together, as Experiment 29
  planted raiders. A brain that takes hold only in company has an Allee
  effect, which evolution, trying one agent at a time, can miss
  ([Reading an invasion](measuring-evolution.md#reading-an-invasion)).
- **The world as it is.** The same brains planted alone in the world as it
  was played, at the same moments, with the invasion test's own control, a
  frozen copy of the planted agent's brain, in the marked agent's place. A
  brain that takes hold in the sea but not in the world meets something the
  sea lacks: lines, kin beside kin, agents that keep their ways.
- **Written by hand.** The raider can be planted only where brains are not
  mixed at birth and a [message](rules-in-detail.md#messages "Numbers an agent writes to itself and to each neighbour every phase, which their brains read.") holds at least four numbers: seeds 3080 and
  3026 of the five.
- **A sea made in the trial.** If the sea's runs do not exist, a trial can
  make its own. The invasion test can already give a whole kept world another
  strategy and let it settle before planting. That costs the settling again in
  every planting.
- **An open sea.** The first plan of this page, of 9 October 2026, also named
  a sea with an open border, which would give and take tokens as the edge of
  an endless sea would. That breaks the conservation of tokens, and is not
  proposed.

## What has to be built first

1. **The sea**, from
   [What has to be built first](brains-replaced-by-their-average.md#what-has-to-be-built-first)
   on the ladder's page: a run that starts from a kept world, and the crowd's
   brain as a way to decide, with its pool kept beside the run so that a trial
   can start from any world it keeps.
2. **A planted brain from a kept world.** The engine's strategies
   (`gol_strategies.py`) gain a kind that plants a fixed copy of a given
   brain: drawn at random from the sea's pool, drawn from another kept world
   of the same run, or the main line's. And a mark: an agent of the sea whose
   heirs carry the mark and still decide by the pool. Each never mutates,
   breeds true, and carries an id of its own so that its copies can be
   counted, as every strategy's are.
3. **The place**: a leaf holding the median tokens, or nearest it, drawn at
   random; for kin, the agents nearest it, as the invasion test's patch is
   found.
4. **What a trial records.** Today an invasion trial (`gol_trial.py`) records
   the strategy's agents and tokens after every iteration, and stops once the
   strategy has died out. It would also record the copies gained by birth and
   by conquest and the nodes lost, read from which ids are new and which
   brains changed, and the sea's holdings one, two and three steps from the
   line.
5. **The analysis**: the four readings of success, presence, the paired tests,
   and the sea's holdings around the lines, with a figure of the lines over
   time.

## What it would cost

Each planting is a trial of at most 300 iterations, and keeps no [frames](runs-and-records.md#what-a-run-records "A whole world as a run records it after one phase: every agent, its tokens and connections."). An
iteration of a sea costs about what an iteration of its world costs: the
crowd's brain runs one brain a look, as the world does. Four kinds, 30
plantings each, make 120 trials a world. If every line lived the full 300
iterations, they would take 9 hours for seed 3080, 19 for seed 3026, 30 for
seed 3057, 47 for seed 3018 and 120 for seed 3042. That is 225 hours, about
three days on the lab's three workers. The three fastest worlds take 58 of
those hours: about 20 hours on three workers.

But a planting stops when its line dies out, and lone lines die young: half of
Experiment 29's lone raiders were gone by iteration 51. So the real cost
should be a fraction of this.

The seas are the runs of the crowd's brain of the ladder's page, already
counted there
([What it would cost](brains-replaced-by-their-average.md#what-it-would-cost)).
A world played here without that page needs one sea of its own: 1,500
iterations, from 23 minutes (seed 3080) to 5 hours (seed 3042).

The further steps cost more. The world as it is doubles the trials; kin of 10
and 100 add 60 trials a world, whose lines, larger, live longer.

## Choices for the author

- **Which worlds.** Those of the ladder's page whose sea lives; or the three
  fastest first, seeds 3080, 3026 and 3057, at most about 20 hours.
- **Which plantings.** The four proposed; or only the main line's and the
  future's, each beside the marked agent.
- **Alone or with kin.** Alone first, as proposed; kin of 10 and 100 only if
  no lone brain takes hold.
- **Where.** A leaf holding the median, as proposed; or any agent drawn at
  random, so that the plantings sit where the sea's own agents sit.
- **How long.** 300 iterations, as the lab's invasion tests, as proposed; or
  longer, for the lines still alive.
- **The world as it is** beside the sea: now, or after the sea has answered.
- **The lines of the thesis.** One planting in ten, and *p* = 0.05, are
  proposals.

## Where to go next

- [Brains replaced by their average](brains-replaced-by-their-average.md): the
  ladder whose fourth way makes the sea.
- [The invasion test](measuring-evolution.md#the-invasion-test): how a strategy
  is planted, and how to read what it does.
- [How large a crowd a raider needs](strategies-written-by-hand.md#how-large-a-crowd-a-raider-needs):
  why one raider never takes hold and a hundred can.
- [Vacuum and matter](physical-inspiration.md#vacuum-and-matter): the sea as
  a vacuum, and what a lasting thing in it would mean.

## How this was made

No run or trial of this plan exists, so this page has no box of runs, and no
plan file: the plan lives on this page until the author picks it. Then it
becomes a plan in `book/experiments/`, with its thesis as written here, fixed
before any run.

The numbers come from round 3's own records, read on 9 November 2026 without
playing anything. Nodes taken in a game are 1 minus `heldHomeShare` of the
game's rows of each run's `stats.jsonl`; children per agent are `births`
over `nodes_before` of the reproduction's rows; the share of leaves is
`leaves` over `nodes` of the game's rows, and a newborn's links
`meanChildLinks` of the reproduction's rows; all are averaged over iterations
1,500 to 2,999. The agents, their median tokens and the leaves holding it are
read from each run's kept world of iteration 1,500
(`archive/world_001500.npz`), and the seconds an iteration as on the
ladder's page.

**References.** Gibbs, J. W. (1902), *Elementary Principles in Statistical
Mechanics*. Maynard Smith, J. and Price, G. R. (1973), The logic of animal
conflict. Metz, J. A. J., Nisbet, R. M. and Geritz, S. A. H. (1992), How should
we define 'fitness' for general ecological scenarios?

<!-- turns -->
---

← [What a brain can compute](what-a-brain-can-compute.md) · [Strategies written by hand](../README.md#strategies-written-by-hand)

**Also about:** [Space and dimension](../README.md#space-and-dimension).

> [!info]- Pages that link here (2)
> [Physical inspiration](physical-inspiration.md) · [Brains replaced by their average](brains-replaced-by-their-average.md)
<!-- /turns -->
