# Glossary

> [!question] Questions of this page
> - What does a word of this notebook mean, in one line?
> - Which page explains it?

Every term of the notebook, from A to Z: what it is, in one line, and the page
that explains it. On every other page, the first use of each of these words is
a link to that page, and resting the pointer on the link shows the line.

The list is made from `book/terms.json`; a word that is missing can be added
there.

<!-- glossary -->
- **[2-core](bridges.md#core-trees-and-leaves)** (also hanging tree) — What is left of a network when agents with one connection or none are peeled away again and again: the agents on loops or between them.
- **[agent](the-world.md#agents-and-tokens)** (also node) — One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.
- **[assortativity](degree.md#assortativity)** — Whether agents tend to be joined to agents like them: the well connected to the well connected.
- **[autocorrelation](autocorrelation.md)** — How much a world's value now still says about its value some iterations later.
- **[B1](b1-and-b2.md)** — The first baseline: the algorithm exactly as a new run is offered it.
- **[B2](b1-and-b2.md)** — The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.
- **[baseline](b1-and-b2.md)** — The stated rules an experiment starts from and changes one thing of: B1 or B2.
- **[bootstrap](bootstrap.md)** — How sure a number is, found by redrawing the worlds it came from many times.
- **[box dimension](ball-dimension-and-curvature.md#box-dimension)** — A network's dimension read from how many boxes of a given size it takes to cover it.
- **[brain](the-brain.md)** — The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.
- **[brain diversity](genotype.md#brain-diversity)** — How many different brains, or genotypes, a world holds.
- **[bridge](bridges.md)** — A connection that lies on no loop: cut it and the network falls into two pieces.
- **[candidate](one-iteration.md#phase-1-reproduction)** — An agent itself and each of its neighbours, the ones its brain looks at when it decides.
- **[circulation](token-flow.md#flows-that-go-round)** — The part of a flow that goes round in loops, as against the part that runs from sources to sinks.
- **[cleanup](one-iteration.md#cleanup)** — The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.
- **[clustering](degree.md#clustering)** — How often an agent's neighbours are also each other's neighbours.
- **[common ancestor](genotype.md#the-common-ancestor-of-the-living)** — The genotype every living agent descends from, and how far back it lies.
- **[connection](the-world.md#the-network)** — A link between two agents of the network: they are neighbours, and can stake on each other's nodes.
- **[correlation](correlation.md)** — Whether two quantities go up and down together, from −1 to 1.
- **[correlation length](like-next-to-like.md#how-alike-at-a-distance)** (also correlation function) — How far apart two agents can be and still be alike: how fast likeness fades with distance.
- **[cross-time tournament](eras-and-contests.md)** (also era) — Brains of a world at two moments, its eras, played against each other: do the later ones win?
- **[cut risk](bridges.md#cut-risk)** — The largest share of a world that cutting one connection would cut off.
- **[degree](degree.md)** — The number of an agent's connections: how many neighbours it has.
- **[diameter](path-length.md#radius-and-diameter)** — The longest of the shortest paths between two agents of a world.
- **[dimension](ball-dimension-and-curvature.md)** (also ball dimension) — How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.
- **[diversity score](diversity-score.md)** — How much the agents of a world differ, in one number from 0 to 1: in tokens, links, ages and genotypes.
- **[effective kinds of staking](how-different-are-the-agents.md)** — How many different ways of staking a world's brains have, when all are put in the same situations, counted by how many agents use each.
- **[effective number](effective-number.md)** — How many kinds there really are, when kinds are counted by how much of the population they hold.
- **[entropy](gini-coefficient.md#entropy-and-evenness)** (also evenness) — How evenly something is shared out among the agents, from all in one place to spread perfectly evenly.
- **[estate](estate.md)** — The tokens a dead agent leaves, and who gets them.
- **[even split](even-split.md)** — What the rules do when a brain scores no candidate above zero: the stake is spread evenly.
- **[exploitability](eras-and-contests.md#exploitability)** (also best reply) — How much better than a world's brains the best reply to them would do: zero only at the top.
- **[extinction](extinction.md)** (also extinct) — A world's end before its last iteration: it has dropped to its threshold of agents or fewer.
- **[flow module](flow-modules.md#flow-modules-and-the-map-equation)** (also map equation) — A group of agents that a token, following the flow, tends to stay inside.
- **[founder](starting-ring.md)** — One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.
- **[fractional factorial](eight-switches-at-once.md#many-switches-in-few-worlds-fractional-factorial-designs)** — A design that tests many switches together in few worlds, so that each one's effect can still be told apart.
- **[frame](frames-and-stats.md)** — A whole world as a run records it after one phase: every agent, its tokens and connections.
- **[game](one-iteration.md#phase-2-the-game)** — The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.
- **[genotype](genotype.md)** (also family tree) — The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from.
- **[Gini coefficient](gini-coefficient.md)** (also Gini, Lorenz curve) — How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all.
- **[half-cut exponent](what-makes-a-world-interesting.md#the-half-cut-exponent)** — How the links crossing the cheapest cut of a world into halves grow with its size: 2/3 for a space of three dimensions, near 0 for a bottleneck, near 1 for a folded world.
- **[health bar](health-bar.md)** — The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4.
- **[healthy world](what-makes-a-world-interesting.md)** (also interesting world) — A world that meets all six criteria of the screen of random worlds; an interesting one is alive and misses only a few (at most two, provisionally, from round 3).
- **[hegemon](revolution.md)** — The agent that made the largest single stake on a node; it wins the node unless a coalition outbids it.
- **[home stake](how-the-game-is-played.md#staking-at-home-and-keeping-ones-node)** — The tokens an agent stakes on its own node, to keep it.
- **[hub](stars-and-hubs.md)** (also star) — An agent joined to far more agents than most; a star is a world held together by a single hub.
- **[invasion test](eras-and-contests.md#the-invasion-test)** — A strategy planted as a patch in an evolved world: does it spread, or die out?
- **[iteration](one-iteration.md)** — One step of a world's time: a phase of reproduction, then a phase of the game.
- **[lab](the-lab.md)** (also engine) — The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.
- **[largest remainder](largest-remainder.md)** — How tokens are split into whole numbers: each gets the whole part of its share, and the tokens left over go to the largest fractions.
- **[least squares](correlation.md#fitting-a-straight-line)** (also R²) — The straight line that fits a cloud of points best, and how much of their spread it explains.
- **[lifetime](age-and-lifetime.md)** — How many iterations an agent lives, from its birth to its death.
- **[lightning](token-flow.md#lightning-tokens-that-go-round)** — A closed loop of the flow of a game: tokens that go round and come back where they started.
- **[lineage](genotype.md#families)** — A line of descent: the agents whose genotypes go back to one ancestor.
- **[logarithmic axis](correlation.md#logarithmic-axes-and-power-laws)** — An axis on which each step multiplies by the same factor, so that numbers of very different sizes fit on one chart.
- **[loop](bridges.md#loops)** — A path through the network that comes back to where it started without using a connection twice.
- **[median](median-and-quantiles.md)** (also quantile) — The middle value: half of the numbers lie below it and half above.
- **[message](messages.md)** — Numbers an agent writes to itself and to each neighbour every phase, which their brains read.
- **[mobility](gini-coefficient.md#mobility)** — Whether the rich stay rich, or who holds the wealth keeps changing.
- **[Muller plot](genotype.md#muller-plots)** — A chart of how lines of descent grow, shrink and replace each other over time.
- **[mutation](mutation.md)** — A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.
- **[neighbour](the-world.md#the-network)** — An agent joined to another by a connection.
- **[open-ended evolution](open-ended-test.md)** — Evolution that keeps producing new things without end, not only up to a plateau.
- **[participation ratio](effective-number.md#how-many-directions-a-cloud-of-points-uses)** — How many directions a cloud of points really uses.
- **[path length](path-length.md)** — How many connections the shortest way between two agents crosses.
- **[permutation test](bootstrap.md#permutation-tests)** — Whether a difference could be chance, found by shuffling which world belongs to which condition.
- **[pilot](the-lab.md#pilot-worlds)** — A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.
- **[power law](power-laws-real-and-apparent.md#fitting-a-power-law-properly)** — A quantity whose frequency falls as a power of its size: few huge values, many small ones.
- **[power spectrum](autocorrelation.md#the-power-spectrum-of-a-time-series)** — How much of a series' wandering comes at each speed, from slow to fast.
- **[preferential attachment](degree.md#preferential-attachment-and-its-kernel)** — The well connected gaining new connections faster than the poorly connected.
- **[random twin](random-twin.md)** — A network with the same agents and as many links each, wired at random: what a world would be if it were no space at all.
- **[revolution](revolution.md)** (also coalition) — A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.
- **[richest tenth](gini-coefficient.md#the-richest-tenths-share)** — The share of all tokens held by the richest tenth of the agents.
- **[same-position share](what-makes-a-world-interesting.md#the-same-position-share)** — Of two agents put in the same situation, how often they stake their tokens measurably differently.
- **[scaling relation](degree.md#scaling-relations-between-agents-properties)** — How one property of an agent grows with another, as a power.
- **[seed](random-numbers.md)** — The number that fixes every random choice of a run: the same settings and the same seed make the same world.
- **[setting](settings.md)** — One number or switch of a run's rules, fixed before it starts.
- **[settled life](median-and-quantiles.md#the-settled-life-of-a-world)** — A world from iteration 500 to its end, after its youth, where it is measured.
- **[share function](share-function.md)** — How a brain's two numbers become one share between 0 and 1: of tokens for a child, or of a stake marked revolutionary.
- **[share-out](gains-and-losses.md#gains-losses-and-the-share-out)** — How a cleanup deals the tokens of the dead out among the living: in the baseline, each token to a survivor drawn at random.
- **[spectral dimension](ball-dimension-and-curvature.md#the-spectral-dimension)** — A network's dimension read from how a random walker spreads over it.
- **[spectral gap](path-length.md#the-spectral-gap)** — How hard a network is to cut into two halves, and how fast a walker on it forgets where it began.
- **[survival curve](age-and-lifetime.md#survival-curves-kaplanmeier)** (also Kaplan–Meier) — The share of lives that last at least so long, counted fairly when some are cut short by the end of a run.
- **[token](the-world.md#agents-and-tokens)** — The one thing of value in a world: every agent holds a whole number of them, and their total never changes.
- **[token curvature](gains-and-losses.md#token-curvature)** — How much richer an agent's neighbourhood is than the agent itself.
- **[token flow](token-flow.md)** (also net flow) — The tokens a game moves along each connection, from the agent who stakes to the node staked on.
- **[Wilson interval](bootstrap.md#the-wilson-interval)** — The range a share probably lies in, when it was counted in only a few worlds.
- **[yes-or-no decision](binary-decision.md)** — A yes or no a brain gives from four numbers: how strongly it means yes and no, and whether the stronger always wins or chance decides in proportion.
<!-- /glossary -->

<!-- turns -->
---

← [What this notebook is about](what-this-notebook-is-about.md) · [Start here](../README.md#start-here) · [How this notebook is written](how-this-notebook-is-written.md) →

> [!info]- Pages that link here (3)
> [Welcome](../README.md) · [What this notebook is about](what-this-notebook-is-about.md) · [How this notebook is written](how-this-notebook-is-written.md)
<!-- /turns -->
