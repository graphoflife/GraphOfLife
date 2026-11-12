# Did evolution happen? Fourteen ways to find out

> [!question] Questions of this page
> - Round 3's worlds pass on what their [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") do, and keep changing it. Is
>   that evolution by selection, or could luck alone have done it?
> - Which tests would tell the two apart, and which of them need no new runs?
> - In what order should they be run?

## In short

The worlds of round 3 of the screen have two of the three things evolution
needs: children act like their parents, and lines of descent keep changing
([Candidates for open-ended evolution](candidates-for-open-ended-evolution.md)).
The third, selection, is the hard one to show. A world whose whole population
descends from about fourteen ancestors can be swept by a line that was only
lucky, and Experiment 7 saw lines take over worlds whose [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") never changed
at all. This page lists fourteen tests, grouped by six questions: did the
brains get better, was it selection or luck, does it keep going, did something
new appear, would it happen again, and is evolution what makes the world.
Eleven need only what the worlds recorded. The first to run is the era
tournament: every kept era of a world against the others, in the world itself.

## Why it is not obvious

Evolution by natural selection needs three things (Lewontin 1970). Agents must
differ. The differences must be passed on, so that children are like their
parents. And the differences must matter: some agents must leave more
descendants *because* of how they differ.

Round 3 shows the first two plainly. Two agents of one era act differently
in the same situations, a child acts more like its parent than like a
stranger (the [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") heredity index is 0.39), and the main line of descent
keeps moving away from where it was
([How each pillar is read](candidates-for-open-ended-evolution.md#how-each-pillar-is-read)).

The third is where luck can fool us. Picture a village of fourteen families.
Whichever family happens to have the most children in a few bad years can end
up as everyone's ancestor, however good its habits. The worlds of round 3 are
such villages: their effective population, the number of ancestors that
really share the future, is about fourteen in the median world. In a
population that small, chance carries a line further than a small advantage
does. And a line taking a world over proves nothing by itself: in Experiment 7,
worlds whose brains could never change were still taken over by one [founder](rules-in-detail.md#the-starting-ring "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")'s
line, by chance alone
([Lines by chance](do-the-brains-matter.md#lines-by-chance)).

So each test below either takes chance out, or asks something that chance
alone could not produce.

## What the worlds already recorded

Most tests need no new runs, because round 3 kept a great deal:

- **Twelve eras a world.** Every 250 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") each world kept its whole
  state, brains included, so the brains of iteration 250, 500, …, 3,000 can be
  played again, and so can the founders of iteration 0
  ([Eras](measuring-evolution.md#eras)).
- **Every birth.** Parent and child are recorded at every birth, so each
  world has its whole family tree.
- **Every decision.** Each phase's decisions are in the record: who staked
  what where, who had a child and how much it was given, which links were made
  and kept.
- **82 worlds that lived**, 72 of them still candidates, 21 of them now being
  played again with two new [seeds](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") each, larger and longer (Experiment 43,
  [Round 3's best worlds, played again](candidates-for-open-ended-evolution.md#round-3s-best-worlds-played-again)).

A typical shortlisted world plays an iteration in about 3 seconds, from 0.6 in
the fastest to 40 in the slowest two. That sets what the tests that replay
brains will cost.

## Did the brains get better at their world?

Evolution by selection should leave later brains better at the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") their
world plays than earlier ones. These two tests play eras against each other.

### 1 · The era tournament

- **Asks:** do later brains beat earlier brains of the same world?
- **How:** play every era against every other era inside the world itself,
  half the agents with one era's brains and half with the other's, [mutation](rules-in-detail.md#how-a-brain-changes "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.")
  off, and see whose brains hold more of the world at the end
  ([A contest, step by step](measuring-evolution.md#a-contest-step-by-step)).
  Twelve eras make 66 pairs.
- **Yes would look like:** later eras beat earlier ones, and by more the
  further apart they are: a climb. Or wins that go round in a circle, A over
  B over C over A: an arms race that keeps changing without getting better
  overall ([Plateau, climb or circle](measuring-evolution.md#plateau-climb-or-circle)).
- **No would look like:** every contest ends near a tie. The brains of
  iteration 3,000 play no better than those of iteration 250.
- **Cost:** contests of 100 iterations, each pair played once in each era's
  world: 132 contests a world for a full round robin. With 3 seconds an
  iteration that is about 11 hours a world. Played like Experiment 28, every
  era against three fixed eras, it is 72 contests; neighbouring eras only,
  22. For the 24 worlds shortlisted when this page was written, on three
  workers, that is about 8 days,
  4 days or a day and a half. The tool is built
  ([Every era against three](every-era-against-three.md)).
- **From:** the cross-time tournament of Cliff and Miller (1995), and the
  master tournament of Floreano and Nolfi (1997).
- **Planned:** as Experiment 44, every era against three in the 21 worlds of
  the shortlist ([Did the brains get better?](round-3-eras-against-each-other.md)).

### 2 · Old and new in a common garden

- **Asks:** is the progress real, or only relative to the world each era
  made for itself?
- **How:** plant an early era and a late era half and half in two fixed
  gardens: the early world's network and the late world's, mutation off. In
  the tournament each era plays where it was found. Here both play in the
  same place, as the bacteria of Lenski's long-term experiment are grown beside
  their frozen ancestors.
- **Yes would look like:** the late brains win in both gardens, so they are
  better in general. If they win only in the late garden, they are adapted to
  the world they built themselves.
- **No would look like:** the late brains win in neither garden.
- **Cost:** like the tournament, a few contests a world.
- **From:** Lenski, Rose, Simpson and Tadler (1991).

## Was it selection, or luck?

With an effective population of about fourteen, chance alone can carry a line
across a world. These three tests take chance out, or measure selection
directly.

### 3 · A shadow world of pure luck

- **Asks:** did some lines of descent spread because of what they did, or by
  chance alone?
- **How:** replay each world's family tree with every child given a parent
  drawn at random among the living, keeping the same births and deaths in
  the same iterations. That is the world with selection taken out: a shadow.
  Compare how long lines last, and how fast one takes a world over, in the
  real tree and in many shadow trees.
- **Yes would look like:** real lines last longer and sweep faster than shadow
  lines do. New lines rising above the shadow again and again would also say
  that evolution keeps going.
- **No would look like:** the real tree looks like a shadow tree.
- **Cost:** minutes a world, from the family tree alone.
- **From:** the evolutionary activity of Bedau and Packard (1992), and the
  shadow models of Bedau, Snyder and Packard (1998).

### 4 · Selection on behaviour, measured

- **Asks:** which behaviours bring more children, and does the population
  move the way that predicts?
- **How:** from the recorded decisions, give each agent a few behaviour
  numbers: how much it gives a child, how widely it spreads its stakes, how
  many links it makes or keeps. In each era, measure how strongly each number
  goes with having children (selection) and how much children resemble their
  parents in it (heredity). Together they predict how the average should move
  by the next era. Then check the prediction against what happened.
- **Yes would look like:** the predictions come true: behaviours move the way
  selection and heredity say. That is natural selection, measured behaviour by
  behaviour.
- **No would look like:** no behaviour goes with having children, or
  behaviours move where no prediction points: drift, or something not
  measured.
- **Cost:** hours of computing, no new runs. A first piece exists: fertility
  is passed on in 65 of the 72 candidates
  ([The candidates, pillar by pillar](candidates-for-open-ended-evolution.md#the-candidates-pillar-by-pillar)).
- **From:** the Price equation (Price 1970) and the selection gradients of
  Lande and Arnold (1983).

### 5 · Silent and expressed changes

- **Asks:** is behaviour being pushed somewhere, or held in place?
- **How:** most changes to a brain's numbers change nothing it does. Follow
  the line of descent from era to era and sort its changes into silent ones
  (the same decisions in a fixed set of test situations) and expressed ones
  (different decisions). Compare how many expressed changes survived with how
  many random changes of the same brains would express.
- **Yes would look like:** more expressed changes survive than chance makes,
  so new behaviour is favoured; or fewer, so behaviour is held in place
  against mutation. Either is selection at work.
- **No would look like:** expressed and silent changes survive as often as
  random changes make them: behaviour drifts.
- **Cost:** modest; the brains are in the kept worlds.
- **From:** the ratio of changes that alter a protein to changes that do not,
  in molecular evolution, and Kimura's neutral theory (1968).

## Does it keep going?

Open-ended evolution keeps producing change and new things
([What would count as open-ended evolution](measuring-evolution.md#what-would-count-as-open-ended-evolution)). These two
tests ask whether round 3 levels off, and where.

### 6 · Change, novelty, ecology, complexity

- **Asks:** how open-ended is round 3, on a scale other artificial-life
  systems are measured on?
- **How:** the MODES toolbox measures four signs, counting only kinds that
  persist: change (the surviving kinds keep changing), novelty (kinds appear
  that were never seen before), ecology (many kinds live side by side), and
  complexity (the most complex surviving kind keeps growing, for instance in
  how many of its inputs a brain really uses).
- **Yes would look like:** all four keep rising over the eras.
- **No would look like:** they level off early. In Experiment 28, the
  notebook's two longest worlds climbed for 20,000 to 25,000 iterations and
  then only wandered ([Every era against three](every-era-against-three.md)).
- **Cost:** hours; it reuses the kept worlds and the family tree.
- **From:** Dolson, Vostinar, Wiser and Ofria (2019).

### 7 · How much room is left

- **Asks:** has evolution reached a top, or is it stuck on a plateau?
- **How:** for each era, look for the best reply to the population: the
  brain that would win the most among those agents. What it would gain is the
  era's [exploitability](measuring-evolution.md#exploitability). Track it over
  the eras.
- **Yes would look like:** exploitability shrinks while the eras improve: the
  population climbs towards a top.
- **No would look like:** exploitability stays high while the eras stop
  improving: a plateau, not a top. That is what the notebook found in its
  two longest worlds, where a best reply gained about one node a game at
  every era from 2,500 to 50,000.
- **Cost:** moderate: one search an era.
- **From:** game theory's distance from an equilibrium.

## Did something new appear?

The rules of round 3 allow [messages](rules-in-detail.md#messages "Numbers an agent writes to itself and to each neighbour every phase, which their brains read."), gifts, links far away and more. These four
tests look for behaviour that uses them in ways nobody wrote in.

### 8 · Do the messages mean something?

- **Asks:** did a code evolve between senders and listeners?
- **How:** most worlds let agents send numbers to their [neighbours](the-world.md#the-network "An agent joined to another by a connection."). Test the
  brains in situations taken from their own world. Does what an agent sends
  depend on its situation? Does a listener decide differently when a message
  is changed or blanked? Measure both in every era.
- **Yes would look like:** both rise over the eras: messages carry news and
  listeners use it. That would be the evolution of communication.
- **No would look like:** listeners ignore the messages, or the messages are
  noise. In [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") the messages were only the sender's name, a label of its
  genotype ([What agents say to each other](b1-brains-lineages-and-messages.md#what-agents-say-to-each-other)).
- **Cost:** modest: tests on the kept brains.
- **From:** the signalling games of Lewis (1969) and Skyrms (2010).

### 9 · Kin help kin

- **Asks:** did helping relatives evolve?
- **How:** in worlds with gifts, the record says who gave what to whom, and the
  family tree says who is related. Do agents give to relatives more than to
  strangers? Do relatives stake on the same nodes, as a team would?
- **Yes would look like:** gifts and shared stakes go to kin far more than
  chance would place them, and more so in later eras: kin selection, as
  Hamilton's rule predicts.
- **No would look like:** gifts go to whoever is near, kin or not, as
  stakes do in B1, whose brains stake on kin as on strangers
  ([Do agents cooperate?](cooperation-help-and-gliders.md#do-agents-cooperate-help-kin-and-partners)).
- **Cost:** small: the record and the family tree.
- **From:** Hamilton (1964).

### 10 · Kinds of strategy over time

- **Asks:** which kinds of behaviour rose and fell, and did new kinds appear?
- **How:** give every agent of every era a fingerprint: its decisions in one
  fixed set of test situations. Group the fingerprints into kinds of
  strategy, and draw their shares over time, one band per kind, as the
  long-term evolution experiment draws its lineages (a Muller plot).
- **Yes would look like:** new kinds appear late, then sweep or settle beside
  the old ones: innovation, sweeps, and kinds that hold each other in check.
- **No would look like:** one kind from the start, wandering slowly.
- **Cost:** hours, from the kept brains.
- **From:** Muller plots, as in Good, McDonald, Barrick, Lenski and Desai
  (2017).

### 11 · Roles in the network

- **Asks:** do agents in different places of the network play different
  parts, and do the parts run in families?
- **How:** test brains in a [hub](measuring-networks.md#stars-and-hubs "An agent joined to far more agents than most; a star is a world held together by a single hub.")'s position and in a leaf's: do they decide
  differently? Do the children of hub-like agents play hub-like parts too?
- **Yes would look like:** distinct roles that are inherited: the start of a
  division of labour.
- **No would look like:** every brain plays the same part wherever it sits.
- **Cost:** modest, from the kept brains and the record.
- **From:** the major transitions of Maynard Smith and Szathmáry (1995).

## Would it happen again?

Whether evolution repeats tells adaptation from accident. These two tests need
worlds played again, and Experiment 43 is already playing some.

### 12 · Replay the tape

- **Asks:** is what evolved bound to happen, or an accident of history?
- **How:** start a world again from its kept state at iteration 1,500, five
  times, each with new random numbers, and play each on to 3,000. Compare the
  replays with each other and with the original: their kinds of strategy,
  their health, their era tournaments.
- **Yes would look like:** the replays arrive at the same behaviour and
  health: adaptation that repeats.
- **No would look like:** each replay goes its own way: history decides.
- **Cost:** new runs, five replays of 1,500 iterations for each world chosen.
- **From:** Gould (1989), and Blount, Borland and Lenski (2008).

### 13 · Parallel evolution in Experiment 43

- **Asks:** do two worlds with the same rules evolve the same strategies?
- **How:** Experiment 43 plays 21 of round 3's worlds again with two new seeds
  each. When it ends, compare the two seeds' late kinds of strategy, and play
  their late brains against each other in a common garden (test 2).
- **Yes would look like:** the same strategies arise with both seeds: the
  rules decide what evolves.
- **No would look like:** different strategies each time: there are many ways
  to live, and history picks one.
- **Cost:** the runs are already playing, about 11 days in all; the analysis
  takes hours.
- **From:** convergent evolution.

## Is evolution what makes the world?

### 14 · Freeze evolution halfway

- **Asks:** does evolution keep a world as it is, or would it stay so with
  frozen brains?
- **How:** continue a world from iteration 1,500 twice: once as it was, once
  with mutation off. Experiment 7 did this from the start for B1
  ([Without change: frozen or teeming](do-the-brains-matter.md#without-change-frozen-or-teeming));
  here it starts from a world that has already evolved.
- **Yes would look like:** the frozen world drifts away from the evolving one,
  in health or in behaviour: evolution is doing work.
- **No would look like:** both continuations stay alike: the world's shape no
  longer depends on evolution.
- **Cost:** new runs, 1,500 iterations twice for each world chosen.
- **From:** Experiment 7 of this notebook.

## The order to run them in

Six of the fourteen, in this order, each for a reason:

1. **The era tournament.** Its tool is built, it uses data we have, and it
   answers the plainest question: did the brains get better?
2. **The shadow world.** Minutes a world. It tells selection from luck, which
   matters most where populations are as small as round 3's.
3. **Selection on behaviour, measured.** It names the behaviours selection
   favoured, and checks that the population moved accordingly.
4. **The messages.** If a code evolved anywhere, it would be the most striking
   new thing round 3 could show.
5. **Change, novelty, ecology, complexity.** One profile of open-endedness for
   every world, comparable with what others have measured.
6. **Parallel evolution**, when Experiment 43 ends: do the same rules evolve
   the same strategies?

The author read the fourteen on 9 November 2026 and found them right; of the
ideas put before, the era contests were the one they liked best.

## What it would take to say yes

No single test settles it. A world would count as one where evolution by
selection happened if its later eras beat its earlier ones (test 1), its lines
spread faster than its shadow's (test 3), and its behaviours moved where
selection and heredity predict (test 4). It would count as a candidate for
*open-ended* evolution only if, in addition, the climb or the circle had not
come to rest by the last era (tests 1, 6 and 7), and something new appeared
along the way (tests 8 to 11).

## Where to go next

- [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md):
  the 72 worlds these tests would read, and the shortlist of 21.
- [Eras, contests and invasions](measuring-evolution.md#eras-contests-and-invasions-does-a-world-keep-getting-better): how two eras are
  played against each other, and what their scores mean.
- [Every era against three](every-era-against-three.md): the era tournament,
  played on [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.").
- [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md):
  the notebook's earlier tests, and why most numbers that rise mean nothing.

## How this was made

The fourteen tests were drawn up on 9 November 2026, after round 3's
[health](which-worlds-are-healthy.md) and its
[candidates](candidates-for-open-ended-evolution.md) had been read. They were
first put to the author as a page of their own, where each could be marked.
The costs are from round 3's own records: the seconds each iteration took in
the last thousand iterations of each shortlisted world (the field `_seconds`
of each run's `stats.jsonl`, both phases of an iteration added). Contests play
without recording [frames](runs-and-records.md#what-a-run-records "A whole world as a run records it after one phase: every agent, its tokens and connections."), so they run somewhat faster than that.

**References.** Bedau, M. A. and Packard, N. H. (1992), Measurement of
evolutionary activity, teleology, and life. Bedau, M. A., Snyder, E. and
Packard, N. H. (1998), A classification of long-term evolutionary dynamics.
Blount, Z. D., Borland, C. Z. and Lenski, R. E. (2008), Historical contingency
and the evolution of a key innovation in an experimental population of
*Escherichia coli*. Cliff, D. and Miller, G. F. (1995), Tracking the Red Queen.
Dolson, E., Vostinar, A. E., Wiser, M. J. and Ofria, C. (2019), The MODES
toolbox: measurements of open-ended dynamics in evolving systems. Floreano, D.
and Nolfi, S. (1997), God save the Red Queen! Good, B. H., McDonald, M. J.,
Barrick, J. E., Lenski, R. E. and Desai, M. M. (2017), The dynamics of molecular
evolution over 60,000 generations. Gould, S. J. (1989), *Wonderful Life*.
Hamilton, W. D. (1964), The genetical evolution of social behaviour. Kimura, M.
(1968), Evolutionary rate at the molecular level. Lande, R. and Arnold, S. J.
(1983), The measurement of selection on correlated characters. Lenski, R. E.,
Rose, M. R., Simpson, S. C. and Tadler, S. C. (1991), Long-term experimental
evolution in *Escherichia coli*. Lewis, D. (1969), *Convention*. Lewontin, R. C.
(1970), The units of selection. Maynard Smith, J. and Szathmáry, E. (1995),
*The Major Transitions in Evolution*. Price, G. R. (1970), Selection and
covariance. Skyrms, B. (2010), *Signals*.
