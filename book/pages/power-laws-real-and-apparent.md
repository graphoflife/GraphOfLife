# Power laws, real and apparent, and how to fit one

> [!question] Questions of this page
> - How is a power law fitted properly: where does its tail start, and how can
>   one tell whether it is a power law at all?
> - Are the numbers of [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") distributed as a power law? Does it matter
>   how the exponent is fitted?
> - Are the [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.")?
> - How many [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") does one [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") remove — is there a typical number, or do
>   removals come in avalanches of every size?
> - What rhythm does a world's wander have?

## In short

A power law has no typical size, and on logarithmic axes it is a straight
line. Least squares through such a plot gives a biased exponent. The right
way fits the tail by maximum likelihood, finds where it starts, and tests
whether a power law is plausible at all. In the thirty [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") worlds of 100,000
tokens, the tokens of the richest hundredth follow the most convincing power
law. The connections of the best-connected hundredth plausibly follow one.
The agents one game removes have a typical size, and a world's wander is
pulled back to its level: nothing points to a critical state.

## What a power law is, and why it is hard to see

Most quantities have a typical size. People's heights gather around an
average, and nobody is ten times as tall as anyone else. Other quantities
have no typical size at all. In the worlds of this page, the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") agent
holds 5 tokens, and a few hold thousands.

A **power law** is the purest case of no typical size. A quantity *k*
follows one if the share of values equal to *k* falls as *k*^−γ, for some
exponent γ: P(*k*) ∝ *k*^−γ. Then doubling *k* always divides the share by
the same factor, 2^γ, whether *k* is 2 or 2,000. With γ = 3, every doubling
divides it by 8.

Such a distribution is easiest to see through the share of values **at
least** *k*: the complementary cumulative distribution, or CCDF
([The degree distribution](degree.md#the-degree-distribution)). For a power
law it falls as *k*^−(γ−1). On log–log axes, where both axes are
logarithmic, that is a straight line of slope 1 − γ
([Logarithmic axes](correlation.md#logarithmic-axes-and-power-laws)).

Power laws have a special place in the study of complex systems. Networks
whose numbers of connections follow one are called **scale-free**
(Barabási and Albert 1999). Pareto (1896) found one in incomes. Sandpiles,
earthquakes and, in a famous model, extinctions come in avalanches of every
size (Bak, Tang and Wiesenfeld 1987; Bak and Sneppen 1993). And noise whose
power falls as 1/*f*, for the frequency *f*, is found in music, rivers and
heartbeats. A power law is often read as a sign of something deep. It may
mean a system poised at a critical point, or one that grows by "the rich get
richer".

Most claimed power laws do not survive a careful test (Clauset, Shalizi and
Newman 2009). And scale-free networks turn out to be rare (Broido and
Clauset 2019). The trouble is that many curves look straight over a decade
or two of log–log axes.

This page fits power laws to four things in the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") worlds. They
are the connections, the tokens, the deaths of one game, and the rhythm of a
world's wander. On the way, it shows how a real power law is told from an
apparent one. The worlds are the thirty of 100,000 tokens of
[Thirty worlds](a-worlds-life.md#thirty-worlds). Where the thirty of 10,000
tokens this page was first written from differ, both are given.

## Fitting a power law properly

Say the connections of a world follow a power law above some size. How is
its exponent γ found, and where does the law begin? This section gives the
method the literature recommends (Clauset, Shalizi and Newman 2009), with a
small example. The sections after it apply the method to the worlds.

### The obvious way, and its problem

The obvious way is to draw the CCDF on log–log axes and fit a straight line
through it. The viewer's `degreeExponent` and `tokenExponent` are exactly
that. Each is the [least-squares line](correlation.md#fitting-a-straight-line)
of ln *P*(≥ *k*) on ln *k*, one point for each distinct value, and the
exponent is 1 − slope.

The trouble is in the points:

- Neighbouring points of a cumulative distribution are not independent: each
  contains the next.
- The line counts every distinct value once. In the far tail each value
  belongs to one agent or two, so a handful of agents make many of the
  points.

So the estimate wanders, and R² looks good whatever the truth. In the median
world of this page, the line through the whole degree distribution has
R² = 0.96. Yet the distribution is not straight
([The connections of one world](#the-connections-of-one-world)).

### Maximum likelihood

Maximum likelihood asks a different question. If the values came from a
power law with exponent γ, how probable would the values we see be? The best
γ is the one that makes them most probable.

Precisely: suppose the values *x*₁ … *x*ₙ from some *x*_min upward follow the
density p(*x*) = (γ − 1) *x*_min^(γ−1) *x*^−γ. That is a power law, scaled
so that its shares above *x*_min add up to one. The γ that makes the
observed values most probable is

$$
\hat\gamma = 1 + n \Bigl[\sum_{i=1}^{n} \ln \frac{x_i}{x_{\min}}\Bigr]^{-1} .
$$

In words: take the logarithm of how many times over each value exceeds
*x*_min, add these up, divide *n* by the sum, and add one. Values far above
*x*_min make the sum large and the exponent small: that is a heavy tail. The
estimate is uncertain by about (γ̂ − 1)/√*n*, its standard error.

For whole numbers, like a degree, *x*_min is replaced by *x*_min − ½. That
corrects most of the error of treating them as continuous.

**Example.** Take a tail of four degrees, 2, 3, 4 and 8, with *k*_min = 2,
so that the formula divides by 1.5. The logarithms of 2/1.5, 3/1.5, 4/1.5
and 8/1.5 add up to 3.636. So γ̂ = 1 + 4/3.636 = 2.10. It is known to about
±1.1/√4 = ±0.55, which is to say hardly at all. A tail needs many values.

### Where does the tail start?

Real distributions are a power law, if at all, only above some *x*_min.
Below it, the body bends away. So the fit tries many starts, and keeps the
one whose fitted law sits closest to the data:

1. Take up to 24 values of *k*_min, spread evenly over the distinct values.
2. For each with at least 25 values at or above it, estimate γ̂ as above.
3. Measure the **Kolmogorov–Smirnov distance**: the largest vertical gap
   between the cumulative distribution of the tail and that of the fitted
   law.
4. Keep the *k*_min with the smallest distance.

The viewer reports five numbers from this fit:

- γ̂ (`degreeGamma`);
- the *k*_min (`degreeKMin`);
- the share of all agents in the tail (`degreeTailShare`);
- the distance (`degreeGammaKS`);
- R² of the CCDF over the tail (`degreeGammaR2`), which describes the fit
  but does not test it.

### What this does not tell you

The method finds the best power law. It does not prove there is one. Two
more checks are needed.

**Is the gap small enough?** Draw many samples from the fitted law, and fit
each the same way. Then ask how often a sample fits its own law **worse**
than the data fit theirs: how often its distance is at least the data's.
This is the test of Clauset, Shalizi and Newman, and that share is its
p-value. If it is 10% or less, the data are too far from a power law to be
one. If it is larger, a power law is a plausible description, though another
curve might be too.

The viewer does not run this test; the analysis behind this page does. Each
sample draws its values below *k*_min from the data's own, and its tail from
the fitted law. The world of [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 gets 200 samples, and every other fit 100.

**Does a different law fit as well?** A **log-normal**, a distribution whose
logarithm is normal, often fits as well over the range one has. So does a
power law with an exponential cut-off, which bends down at the top.

And a tail is a small part of a world. The tail of the connections holds 5%
of its agents at 10,000 tokens, and 1% at 100,000 tokens. A power law there
says nothing about the rest.

## The connections of one world

<!-- figure powerlaws/degree-fit -->
![Fitting the degrees of one world](../figures/powerlaws/degree-fit.svg)

**Fitting the degrees of one world.** Dots: for every number of connections k that occurs among the 14,870 agents of the world with seed 1 after its last game, the share of agents with at least k — the complementary cumulative distribution, on logarithmic axes. Dashed: the straight line least squares fits through all the dots, whose slope is 1 − γ for a power law of exponent γ (the viewer's `degreeExponent`). Red: the power law fitted by maximum likelihood to the tail from the k that fits best, k = 22 (`degreeGamma`, `degreeKMin`), which holds 0.8% of the agents ([Fitting a power law](power-laws-real-and-apparent.md#fitting-a-power-law-properly)).

> [!example]- How to make this figure
> **Runs.** `B1-100000-s001`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read the last frame after a game (`2999`) and count every agent's connections in `edges`.
> 2. Dots: for each distinct count k, the share of agents with at least k.
> 3. Dashed: least squares of ln(share) on ln(k) over the dots.
> 4. Red: `gol_series._scale_free(degrees)` — for each candidate k_min, γ = 1 + n / Σ ln(k / (k_min − ½)) over the n agents with k ≥ k_min, and the Kolmogorov–Smirnov distance; the k_min with the smallest distance wins.
>
> **To make it again:** `python3 book_figures.py powerlaws`.
<!-- /figure -->

The world of seed 1 has 14,870 agents after its last game. Least squares
through every point gives the exponent 3.15. But the points are not on a
line: below about twenty connections the distribution bends, and the dashed
line misses the body.

Maximum likelihood chooses the tail from *k*_min = 22. It holds 125 agents,
0.8% of the world, and gives γ = 3.50. With 125 values, the exponent is
known to about ±(γ − 1)/√125 = ±0.22.

Is the tail a power law? The largest gap between the data and the fitted law
is 0.068. Samples drawn from the fitted law show a gap at least as large in
55% of cases (200 samples). So a power law is **not ruled out** for the top
1% of this world. The fit says nothing about the other 99%.

## Thirty worlds

<!-- figure powerlaws/exponents -->
![Power-law exponents of the 30 worlds](../figures/powerlaws/exponents.svg)

**Power-law exponents of the 30 worlds.** One dot per world, at its mean over the measurements (every 25 iterations) from iteration 500 on; the bar is the median. Left: the exponent γ of the degree distribution by maximum likelihood on the tail (`degreeGamma`) and by least squares on the whole distribution (`degreeExponent`), and of the token distribution by least squares (`tokenExponent`). Right: the share of agents in the tail the maximum-likelihood fit chose (`degreeTailShare`) and the largest gap between that tail and the fitted law (`degreeGammaKS`).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Average `degreeGamma`, `degreeExponent`, `tokenExponent`, `degreeTailShare` and `degreeGammaKS` over the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 1,499.
> 2. One dot per run, a bar at the median.
>
> **To make it again:** `python3 book_figures.py powerlaws`.
<!-- /figure -->

Across the worlds, over their [settled life](median-and-quantiles.md#the-settled-life-of-a-world "A world from iteration 500 to its end, after its youth, where it is measured."):

- maximum likelihood finds the tail from about 21 connections, holding 1.1%
  of the agents (0.7% to 1.6%), with γ = 3.04 (2.40 to 4.11);
- least squares through everything gives 2.80 (1.95 to 3.45). It is
  **lower** in every one of the thirty worlds, because a line through the
  whole distribution is tilted by the bend of the body;
- the test, on each world's last frame, rules a power law out in 6 of the 30
  worlds and leaves it plausible in 24.

In the first two lines, each number is that of the median world, and the
brackets hold the least and the largest of the thirty.

At 10,000 tokens the tail began at about 10 connections and held 5% of the
agents, with γ = 3.34. So the top hundredth of the connections looks like a
power law with an exponent near 3, and the bulk does not. The exponent one
reports depends on the method, by a quarter to half a unit. "The network is
scale-free" would claim far more than this.

### Where the tail of the connections could come from

Where would a tail like this come from? The rules contain a known mechanism.
A child is joined to some of its parent's [candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides."): in effect, it
**copies** part of its parent's neighbourhood
([How a child is joined](how-agents-have-children.md#how-a-child-is-joined)).
An agent gains a connection whenever one of its [neighbours](the-world.md#the-network "An agent joined to another by a connection.") has a child that
is joined to it. So an agent with twice the neighbours has about twice the
chances.

The well connected then gain connections in proportion to their
connections. That is
[preferential attachment](degree.md#preferential-attachment-and-its-kernel)
(Barabási and Albert 1999), reached here by copying. The "duplication"
models of protein networks reach it the same way (Vázquez, Flammini, Maritan
and Vespignani 2003; Chung, Lu, Dewey and Galas 2003). Preferential
attachment on its own gives γ = 3.

[How the network grows](how-agents-have-children.md#how-the-network-grows)
has measured it here. An agent gains connections in proportion to those it
has, as *k*^1.02, and loses them a little faster, as *k*^1.20. The faster
losses bend the tail down at the top. Whether this is what makes the tail is
still a hypothesis. An experiment that joined children to random agents
instead would test it.

The same mechanism came back when [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") could see
([What the game adds](a-world-that-cannot-break.md#what-the-game-adds)). A
brain that chooses a newborn's links ranks its candidates by what it sees of
them. Whatever it favours, the well connected are among the candidates more
often, because they are the neighbours of more agents. In [pilots](the-lab.md#pilot-worlds "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.") in
which the brains chose a newborn's two extra links, the best-connected agent
was joined to 30% of the world. So the second baseline, [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death."), draws two of
a newborn's links at random from its parent's neighbours. And it caps at
eight the links a brain may choose.

## The tokens: a power law in the tail

<!-- figure powerlaws/tokens-fit -->
![Two candidate laws for the tokens](../figures/powerlaws/tokens-fit.svg)

**Two candidate laws for the tokens.** Dots: for every token count x that occurs among the agents alive after the last game of the 30 worlds, the share holding at least x. Red: a power law fitted by maximum likelihood to the tail, from the x that fits it best. Cyan: a log-normal distribution with the mean and standard deviation of ln x over all agents — a distribution whose logarithm is normal. Both on logarithmic axes.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Pool the `tokens` of the last frame of every run.
> 2. Dots: for each distinct x, the share of agents with at least x.
> 3. Red: `gol_series._scale_free(tokens)`, as for the degrees.
> 4. Cyan: μ and σ = the mean and standard deviation of ln x; the curve is 1 − Φ((ln x − μ)/σ), Φ the standard normal distribution.
>
> **To make it again:** `python3 book_figures.py powerlaws`.
<!-- /figure -->

The tokens of 394,056 agents, pooled over the thirty worlds after their last
game, tell a clearer story. Two laws are put to them.

The first is a **log-normal**: a distribution whose logarithm has the bell
shape of a normal distribution. The one drawn has the mean and spread of
ln *x* over all agents, μ = 1.59 and σ = 0.73. Its median is
*e*^1.59 ≈ 4.9 tokens. It fits the body up to about 20 tokens, and then
falls far below the data.

The second is a power law. From 49 tokens on, the data follow a straight
line. Maximum likelihood fits it with γ = 2.22, to 3,358 agents (0.9%), with
a largest gap of only 0.016. Samples from the fitted law fit it worse than
the data in **37%** of cases (100 samples), so a power law is plausible.

Look closer, and the line holds for one decade, to about 500 tokens. Above
that, the richest two hundred or so fall below it. At 1,000 tokens the data
hold about 70% of the share the law gives, and from 3,000 to 5,000 tokens
about half. The test does not see this bend. Its gap is decided where most
of the tail's agents are, and only about 6% of them hold 500 tokens or more.

Of the four candidates on this page, this is still the most convincing power
law. The share of agents with at least *x* tokens falls as *x*^−1.22. That
is a heavier tail than Pareto's incomes, which fell about as *x*^−1.5.

At 10,000 tokens the tail followed γ = 2.29 from 33 tokens on, and samples
fitted it worse than the data in 92% of cases. There the line held from 33
tokens to the richest agent, at about 3,300 tokens: two decades.

### Are the richest the hubs?

Are the richest rich because they are [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.")? If tokens were a fixed function of
connections, the two tails would be tied together. A token tail heavier than
the connection tail would then need the hubs to hold *more* tokens per
connection than everyone else.

They do not: agents of every class of connections hold about 1.7 tokens per
candidate on average
([Tokens and connections](how-properties-scale-together.md#tokens-and-connections)).
And among the 3,358 agents in the token tail, 40% have ten connections or
more, but **33% have only one or two**. A third of the richest agents are
rich far beyond their place.

[Gains and losses](gains-and-losses.md) showed what happens to such agents.
Those with a hundred tokens or more that survive a game lose 48% of their
tokens in it, on average. And a rich agent with one or two connections dies
in about one game in ten.

### Where the tail of the tokens could come from

That suggests a different mechanism, also a classic. An agent's tokens are
multiplied, game after game, by a random factor: sometimes a big gain, more
often a loss. Meanwhile a floor holds up the poorest. A node with nothing
dies, and the share-out tops up the poor. Random multiplication with a floor
produces a power-law tail (Champernowne 1953; Kesten 1973; Levy and Solomon
1996; Gabaix 1999 for the sizes of cities). Again this is a hypothesis. The
simplest test would be to look for that multiplication in the worlds.

## How many agents one game removes

A game removes agents in two ways. An agent **starves** when it ends the game
with no tokens. It is **cut off** when the network splits and its piece is
not the largest. The [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.") then removes that whole piece
([How a world breaks](how-a-world-breaks.md)).

<!-- figure powerlaws/avalanches -->
![How many agents one game removes](../figures/powerlaws/avalanches.svg)

**How many agents one game removes.** Of all games from iteration 500 on in the 30 worlds (30,000 games), the share that cut off at least n agents (orange) and the share in which at least n agents starved (red), for every n, on logarithmic axes. Games in which no one was removed count in the denominator, which is why the curves start below 1.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take `orphaned` and `starved` of every row with `phase` = 2 and 500 ≤ `iteration` ≤ 1,499.
> 2. For each n that occurs, the share of rows with at least n.
>
> **To make it again:** `python3 book_figures.py powerlaws`.
<!-- /figure -->

In every settled game but three of the 30,000, at least one agent is cut
off. In 94% of games a hundred or more are, and in 2% a thousand or more.
The median game cuts off 239 agents. The most ever cut off in one game was
9,460. That was half of the world it struck, and two thirds of a typical
world ([One world, six hundred iterations](how-a-world-breaks.md#one-world-six-hundred-iterations)).

On log–log axes the curve is flat to about 200 and then falls, bending, with
no straight stretch long enough to call a power law. The removals have a
typical size, a few hundred agents, about 2% of a world. Big ones are rare.

But big ones matter. Of all agents ever cut off, 16% died in the 2% of games
that cut off a thousand or more. Half died in games that cut off 353 or more
([Where the cut-off deaths happen](how-a-world-breaks.md#where-the-cut-off-deaths-happen)).

Counted against the size of a world, the worlds of 10,000 tokens did the
same. There, 2.5% of games cut off a hundred or more: a hundredth of their
tokens' worth, as a thousand is here. And 20% of the deaths by cutting fell
in them.

Death by cutting comes in **lumps**: whole branches of the network fall off
at once. Starvation comes in smaller lumps, at most 3,966 in one game.
[How a world breaks](how-a-world-breaks.md) asks where the lumps come from,
and whether they can be foreseen.

## The rhythm of the wander

How does a world's number of agents move over time? The **power spectrum**
splits the movement into rhythms, and says how much of it happens at each
([The power spectrum](autocorrelation.md#the-power-spectrum-of-a-time-series)).
A slow drift puts its power at low frequencies, a quick jitter at high ones.

<!-- figure powerlaws/spectrum -->
![The rhythm of the wander](../figures/powerlaws/spectrum.svg)

**The rhythm of the wander.** The power spectrum of the number of agents ([The power spectrum](autocorrelation.md#the-power-spectrum-of-a-time-series)): for each frequency, how much of a world's up-and-down motion happens at that rhythm — slow wanders on the left, iteration-to-iteration jitter on the right. Each world's series from iteration 500 to 1,499, with its straight-line trend removed and a Hann window applied, normalised to total power 1 and averaged over the 30 worlds; then averaged in 39 bands of equal width on the logarithmic axis. Red: the straight line least squares fits between periods of 500 and 4 iterations.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. For each run, take `nodes` of the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 1,499 (1,000 values); subtract the least-squares straight line.
> 2. Multiply by a Hann window and take |FFT|², the power at frequencies k/1,000, k = 1 … 500; divide by the total.
> 3. Average over runs; average within bands of equal width in log frequency; fit a line on log–log axes between frequencies 1/500 and 1/4.
>
> **To make it again:** `python3 book_figures.py powerlaws`.
<!-- /figure -->

Between periods of 4 and 500 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."), the power falls as *f*^−1.80. That
is close to the *f*^−2 of a **random walk**, in which every step is added to
the last (*f*^−2.11 at 10,000 tokens).

At the slowest end, periods near a thousand iterations, the spectrum
flattens a little. The world does not keep walking away: something pulls it
back. A random walk with a pull back to a level is the Ornstein–Uhlenbeck
process. Its spectrum is flat below a corner and falls as *f*^−2 above it.
Here the corner fits a pull that takes a few hundred iterations to act. The
[autocorrelation](autocorrelation.md) of 0.61 after 100 iterations, found in
[How much does the seed decide?](how-much-does-the-seed-decide.md#a-world-forgets),
fits it too.

Each world's series is a thousand iterations long here. So the slow end is
seen less well than in the worlds of 10,000 tokens, whose wander had 2,500
iterations to show it. Most of the movement happens at the slowest rhythms:
the wander is slow.

It is not 1/*f* noise, a slope of −1, which would mean memory on every time
scale. The world forgets.

## What we learned

- **One real power law, one plausible, two apparent.** The token tail
  follows a power law over one to two decades, and the test finds it
  plausible. The tail of the connections is plausibly one, for the top 1%
  (the top 5% at 10,000 tokens). The degree distribution as a whole is not,
  and neither are the lumps of death.
- **Fit the tail, then test it.** Least squares through a whole distribution
  gave a lower exponent than maximum likelihood in all thirty worlds. A good
  fit is not yet a power law, and the test cannot see a bend among the
  richest few.
- **The rules contain two textbook mechanisms for heavy tails**: copying
  neighbourhoods at birth, and multiplying tokens by chance with a floor
  below. Neither is shown here to be the cause; both could be switched off in
  an experiment.
- **No sign of a critical state.** No avalanches of every size, no 1/*f*
  noise: disturbances die out within a few hundred iterations, and the world
  is pulled back to its level. Some theories tie [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau.") to
  criticality: to systems on the edge between order and chaos, where small
  causes can have effects of every size. Whatever the truth of that, this
  world is not on such an edge. It is a world that wanders and returns
  ([What the measurements say about open-ended evolution](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution)).

## Where to go next

- [Preferential attachment and its kernel](degree.md#preferential-attachment-and-its-kernel):
  who gains new connections, and what that does to the tail.
- [Gains and losses](gains-and-losses.md): what a game does to the rich and
  to the poor.
- [How a world breaks](how-a-world-breaks.md): where the lumps of death come
  from, and whether they can be foreseen.
- [Two numbers together](correlation.md): least squares, R² and logarithmic
  axes, the tools this page says to use with care.

## How this was made

Every figure and number of this page, unless it says otherwise, reads the
thirty runs of Experiment 10, listed below. They are worlds of the baseline
B1 at 100,000 tokens, seeds 1 to 30, run for 1,500 iterations each, made
with `python3 gol_lab.py run E10`. The numbers at 10,000 tokens come from the
thirty worlds of Experiment 2, of which 26 lived to the end of their 3,000
iterations (`python3 gol_lab.py run E02`). Numbers from other pages are
quoted with a link to their page. They include B2's pilots, the attachment
kernels, the tokens per candidate, the fate of the rich and the largest cut.

The command `python3 book_figures.py powerlaws` draws every figure of this
page, and writes its numbers to `book/results/powerlaws.json`. A run's
statistics `degreeExponent` and `tokenExponent` come from
`gol_series._tail_exponent`, and `degreeGamma` and its kin from
`gol_series._scale_free`. The goodness-of-fit test is `tail_pvalue` in
`book_chapters/measures.py`. Each figure says, folded beneath it, which runs
it reads and how.

How far the token tail follows its law is read off the points of the token
figure, `book/figures/powerlaws/tokens-fit.svg`, against its red line. The
same figure drawn from the worlds of 10,000 tokens is in the notebook's
history (`git show 82062d7:book/figures/powerlaws/tokens-fit.svg`).

### The runs of Experiment 10

<!-- runs E10 -->
> [!info]- The runs behind this page
> **30 runs.** Every condition below is run once for every seed (1 to 30), for 1,500 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it.
>
> - **baseline** — changes nothing: B1 as it is; 100,000 tokens. Runs `B1-100000-s001` … `B1-100000-s030`.
>
> To make these runs again: `python3 gol_lab.py run E10`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | baseline | what it is |
> |---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 100,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | 50, 45, 40, 35, 30 | The widths of the brain's hidden layers, in order from input to output. |
> | [`hidden_activation`](settings.md#hidden_activation) | sigmoid | The curve each hidden unit applies to its sum: `sigmoid`, which passes on at most a quarter of a change, so that deep brains barely respond to what they see, or `tanh`, the same S-shape centred on zero, through which a change still passes in several layers. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 30 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | on | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | on | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`contest`](settings.md#contest) | largest | Who takes a contested node: the `largest` stake as the game weighs it, or a staker drawn by `lottery` with a chance in proportion to its stake (needs revolutions off). |
> | [`allocation`](settings.md#allocation) | full | How the game's stakes are placed: all at once (`full`), or a token at a time in rounds with the game as it stands in view, the poorest done first (`weakest_first`) or the richest starting alone (`strongest_first`), as the original engine played. |
> | [`allow_gifting`](settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in both baselines; on in a few pilots, in Experiment 24 and in half the worlds of the screen. |
> | [`random_decisions`](settings.md#random_decisions) | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none); `local` lets each estate flow through the network to the nearest living agents, split evenly at every step. |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | off | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.0 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 0 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | off | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 0 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 0 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`link_token`](settings.md#link_token) | off | Whether every link holds one token: its maker pays for it, a lapsing link gives it back to one of its ends, and a link of a dead agent to the end that lives on, so the links can never outnumber the tokens. |
> | [`link_upkeep`](settings.md#link_upkeep) | off | Whether an agent may pay a token an iteration to keep a link, each brain deciding for each neighbour: `paid_or_used` keeps a link that was paid for or used, `paid_only` only one that was paid for; `off`, only use keeps a link. |
> | [`upkeep_to`](settings.md#upkeep_to) | end | Where a link's upkeep goes: to the link's other end (`end`), or into the estate of the dead (`estate`), shared out by the world's redistribution rule. |
> | [`keep_home`](settings.md#keep_home) | 0.0 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 0 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 1 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`mate_pays`](settings.md#mate_pays) | off | Under mix_at_birth, whether the mate pays the child as many tokens as the parent gave it, as far as it can and never its last. |
> | [`self_message`](settings.md#self_message) | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 30 | one run per seed |
> | iterations | 1,500 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

## How our view changed

- **24 October 2026.** When the note on fitting a power law became part of
  this page, the figure of the tokens was read again, point by point. Since 11
  October the page had said that the tokens follow a straight line from 49
  tokens on. It said the line held for two decades, to about 5,000 tokens,
  before the few richest fall a little below it. It had also said that at
  10,000 tokens the law held over a range ten times shorter. The points show
  otherwise. At 100,000 tokens they follow the fitted law for one decade, to
  about 500 tokens. From there they fall below it: to about 70% of its share
  at 1,000 tokens, and to about half from 3,000 tokens on. At 10,000 tokens
  the law held from 33 tokens to the richest agent, at about 3,300 tokens: two
  decades, not a range ten times shorter. The test still finds a power law
  plausible at both sizes.

## References

- Bak, P. and Sneppen, K. (1993). Punctuated equilibrium and criticality in a
  simple model of evolution. *Physical Review Letters* 71, 4083–4086.
- Bak, P., Tang, C. and Wiesenfeld, K. (1987). Self-organized criticality: an
  explanation of 1/*f* noise. *Physical Review Letters* 59, 381–384.
- Barabási, A.-L. and Albert, R. (1999). Emergence of scaling in random
  networks. *Science* 286, 509–512.
- Broido, A. D. and Clauset, A. (2019). Scale-free networks are rare.
  *Nature Communications* 10, 1017.
- Champernowne, D. G. (1953). A model of income distribution. *The Economic
  Journal* 63, 318–351.
- Chung, F., Lu, L., Dewey, T. G. and Galas, D. J. (2003). Duplication models
  for biological networks. *Journal of Computational Biology* 10, 677–687.
- Clauset, A., Shalizi, C. R. and Newman, M. E. J. (2009). Power-law
  distributions in empirical data. *SIAM Review* 51, 661–703.
- Gabaix, X. (1999). Zipf's law for cities: an explanation. *Quarterly Journal
  of Economics* 114, 739–767.
- Kesten, H. (1973). Random difference equations and renewal theory for
  products of random matrices. *Acta Mathematica* 131, 207–248.
- Levy, M. and Solomon, S. (1996). Power laws are logarithmic Boltzmann laws.
  *International Journal of Modern Physics C* 7, 595–601.
- Pareto, V. (1896). *Cours d'économie politique*. Lausanne: F. Rouge.
- Vázquez, A., Flammini, A., Maritan, A. and Vespignani, A. (2003). Modeling
  of protein interaction networks. *Complexus* 1, 38–44.

<!-- turns -->
---

← [The shape of the network](what-shape-does-the-network-take.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [How properties scale together](how-properties-scale-together.md) →

**Also about:** [Methods and statistics](../README.md#methods-and-statistics).

> [!info]- Pages that link here (15)
> [Glossary](glossary.md) · [How births build the network](how-agents-have-children.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [Gains and losses: who gains, who loses, and the share-out](gains-and-losses.md) · [The largest and the smallest](the-largest-and-the-smallest.md) · [The shape of the network](what-shape-does-the-network-take.md) · [How properties scale together](how-properties-scale-together.md) · [How a world breaks](how-a-world-breaks.md) · [An agent's connections: degree, clustering, assortativity, attachment and scaling](degree.md) · [Physical inspiration](physical-inspiration.md) · [Like next to like: how alike agents are at a distance](like-next-to-like.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [What a run records](frames-and-stats.md) · [Two numbers together: correlation, a fitted line and logarithmic axes](correlation.md) · [How long a world remembers: autocorrelation and the power spectrum](autocorrelation.md)
<!-- /turns -->
