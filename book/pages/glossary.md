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
- **[2-core](measuring-networks.md#core-trees-and-leaves)** (also hanging tree) — What is left of a network when agents with one connection or none are peeled away again and again: the agents on loops or between them.
- **[addition chain](measures-atlases-and-assembly.md#the-assembly-index)** — Numbers from 1 in which each is the sum of two before it; the shortest one ending at m is the fewest joins m links can need.
- **[agent](the-world.md#agents-and-tokens)** (also node) — One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.
- **[assembly](measures-atlases-and-assembly.md#copy-number-and-assembly)** — Assembly theory's measure of a collection: each kind of object weighed by e to the power of its assembly index, times its copies beyond the first, over all the objects.
- **[assembly index](measures-atlases-and-assembly.md#the-assembly-index)** (also assembly indices) — The fewest joins that build an object from basic blocks, when anything built before can be used again.
- **[assembly theory](measures-atlases-and-assembly.md#building-with-reusable-parts)** — A way to measure how much making went into a thing: the fewest steps that build it from basic parts, and how many copies of it there are.
- **[assortativity](measuring-networks.md#assortativity)** — Whether agents tend to be joined to agents like them: the well connected to the well connected.
- **[autocorrelation](statistics.md#how-long-a-world-remembers-autocorrelation-and-the-power-spectrum)** — How much a world's value now still says about its value some iterations later.
- **[B1](b1-and-b2.md)** — The first baseline: the algorithm exactly as a new run is offered it.
- **[B2](b1-and-b2.md)** — The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.
- **[baseline](b1-and-b2.md)** — The stated rules an experiment starts from and changes one thing of: B1 or B2.
- **[bootstrap](statistics.md#how-sure-can-we-be-the-bootstrap-permutation-tests-the-wilson-interval-and-how-many-seeds)** — How sure a number is, found by redrawing the worlds it came from many times.
- **[box dimension](measuring-space.md#box-dimension)** — A network's dimension read from how many boxes of a given size it takes to cover it.
- **[brain](the-brain.md)** — The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.
- **[brain diversity](measuring-evolution.md#brain-diversity)** — How many different brains, or genotypes, a world holds.
- **[bridge](measuring-networks.md#bridges-cut-risk-loops-and-the-core-how-a-network-can-be-cut)** — A connection that lies on no loop: cut it and the network falls into two pieces.
- **[candidate](one-iteration.md#phase-1-reproduction)** — An agent itself and each of its neighbours, the ones its brain looks at when it decides.
- **[circulation](measuring-tokens.md#flows-that-go-round)** — The part of a flow that goes round in loops, as against the part that runs from sources to sinks.
- **[cleanup](one-iteration.md#cleanup)** — The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.
- **[clustering](measuring-networks.md#clustering)** — How often an agent's neighbours are also each other's neighbours.
- **[common ancestor](measuring-evolution.md#the-common-ancestor-of-the-living)** — The genotype every living agent descends from, and how far back it lies.
- **[connection](the-world.md#the-network)** — A link between two agents of the network: they are neighbours, and can stake on each other's nodes.
- **[copy number](measures-atlases-and-assembly.md#copy-number-and-assembly)** — How many identical copies of an object a collection holds.
- **[correlation](statistics.md#two-numbers-together-correlation-a-fitted-line-and-logarithmic-axes)** — Whether two quantities go up and down together, from −1 to 1.
- **[correlation length](b1-the-network-size-and-scaling.md#how-alike-at-a-distance)** (also correlation function) — How far apart two agents can be and still be alike: how fast likeness fades with distance.
- **[cross-time tournament](measuring-evolution.md#eras-contests-and-invasions-does-a-world-keep-getting-better)** (also era) — Brains of a world at two moments, its eras, played against each other: do the later ones win?
- **[cut risk](measuring-networks.md#cut-risk)** — The largest share of a world that cutting one connection would cut off.
- **[data collapse](space-at-scale.md#data-collapse)** (also finite-size scaling) — Curves of one quantity measured at many sizes that fall onto one another once its values are divided by a power of the size: then the size of the whole is its only scale.
- **[degree](measuring-networks.md#degree)** — The number of an agent's connections: how many neighbours it has.
- **[diameter](measuring-networks.md#radius-and-diameter)** — The longest of the shortest paths between two agents of a world.
- **[dimension](measuring-space.md#three-ways-to-measure-a-networks-dimension)** (also ball dimension) — How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.
- **[diversity score](measuring-health.md#the-diversity-score)** — How much the agents of a world differ, in one number from 0 to 1: in tokens, links, ages and genotypes.
- **[effective kinds of staking](b1-brains-lineages-and-messages.md#how-different-are-the-agents)** — How many different ways of staking a world's brains have, when all are put in the same situations, counted by how many agents use each.
- **[effective number](statistics.md#effective-numbers-how-many-kinds-and-how-many-directions)** — How many kinds there really are, when kinds are counted by how much of the population they hold.
- **[effective population](candidates-for-open-ended-evolution.md#how-each-pillar-is-read)** — The size of a population left to chance that would lose its lines as fast: the smaller it is, the more chance decides which line survives.
- **[ego network](measures-atlases-and-assembly.md#ego-networks-as-molecules)** — An agent, its neighbours and the links among them: the network as one agent sees it.
- **[entropy](measuring-tokens.md#entropy-and-evenness)** (also evenness) — How evenly something is shared out among the agents, from all in one place to spread perfectly evenly.
- **[estate](measuring-lives.md#the-estate-of-the-dead)** — The tokens a dead agent leaves, and who gets them.
- **[even split](measuring-tokens.md#the-even-split)** — What the rules do when a brain scores no candidate above zero: the stake is spread evenly.
- **[exploitability](measuring-evolution.md#exploitability)** (also best reply) — How much better than a world's brains the best reply to them would do: zero only at the top.
- **[extinction](rules-in-detail.md#when-a-world-ends)** (also extinct) — A world's end before its last iteration: it has dropped to its threshold of agents or fewer.
- **[false discovery rate](how-the-drawn-settings-shape-a-world.md#how-a-setting-is-tested)** — Among many tests at once, how often a result as strong as this one would be a false alarm (the rule of Benjamini and Hochberg).
- **[flow module](b1-the-game-and-its-flows.md#flow-modules-and-the-map-equation)** (also map equation) — A group of agents that a token, following the flow, tends to stay inside.
- **[founder](rules-in-detail.md#the-starting-ring)** — One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.
- **[fractional factorial](what-the-switches-did-in-pilots.md#many-switches-in-few-worlds-fractional-factorial-designs)** — A design that tests many switches together in few worlds, so that each one's effect can still be told apart.
- **[frame](runs-and-records.md#what-a-run-records)** — A whole world as a run records it after one phase: every agent, its tokens and connections.
- **[game](one-iteration.md#phase-2-the-game)** — The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.
- **[genotype](measuring-evolution.md#genotypes-families-and-the-common-ancestor)** (also family tree) — The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from.
- **[Gini coefficient](measuring-tokens.md#the-gini-coefficient-and-the-lorenz-curve)** (also Gini, Lorenz curve) — How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all.
- **[half-cut exponent](measuring-health.md#the-half-cut-exponent)** — How the links crossing the cheapest cut of a world into halves grow with its size: 2/3 for a space of three dimensions, near 0 for a bottleneck, near 1 for a folded world.
- **[health bar](measuring-health.md#the-health-bar)** — The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4.
- **[healthy world](measuring-health.md#what-makes-a-world-interesting)** (also interesting world) — A world that meets all six criteria of the screen of random worlds; an interesting one is alive and misses only a few (at most two, provisionally, from round 3).
- **[hegemon](rules-in-detail.md#how-a-coalition-takes-a-node)** — The agent that made the largest single stake on a node; it wins the node unless a coalition outbids it.
- **[heredity index](candidates-for-open-ended-evolution.md#how-each-pillar-is-read)** — How much more a child acts like its parent than like a stranger, in the same situations: 1 as its parent does, 0 as a stranger does.
- **[home stake](b1-the-game-and-its-flows.md#staking-at-home-and-keeping-ones-node)** — The tokens an agent stakes on its own node, to keep it.
- **[hub](measuring-networks.md#stars-and-hubs)** (also star) — An agent joined to far more agents than most; a star is a world held together by a single hub.
- **[invasion test](measuring-evolution.md#the-invasion-test)** — A strategy planted as a patch in an evolved world: does it spread, or die out?
- **[iteration](one-iteration.md)** — One step of a world's time: a phase of reproduction, then a phase of the game.
- **[lab](runs-and-records.md#the-lab-the-engine-and-pilots)** (also engine) — The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.
- **[largest remainder](rules-in-detail.md#splitting-tokens-into-whole-numbers)** — How tokens are split into whole numbers: each gets the whole part of its share, and the tokens left over go to the largest fractions.
- **[least squares](statistics.md#fitting-a-straight-line)** (also R²) — The straight line that fits a cloud of points best, and how much of their spread it explains.
- **[lifetime](measuring-lives.md#age-lifetime-and-survival-curves)** — How many iterations an agent lives, from its birth to its death.
- **[lightning](measuring-tokens.md#lightning-tokens-that-go-round)** — A closed loop of the flow of a game: tokens that go round and come back where they started.
- **[lineage](measuring-evolution.md#families)** — A line of descent: the agents whose genotypes go back to one ancestor.
- **[logarithmic axis](statistics.md#logarithmic-axes-and-power-laws)** — An axis on which each step multiplies by the same factor, so that numbers of very different sizes fit on one chart.
- **[loop](measuring-networks.md#loops)** — A path through the network that comes back to where it started without using a connection twice.
- **[median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread)** (also quantile) — The middle value: half of the numbers lie below it and half above.
- **[message](rules-in-detail.md#messages)** — Numbers an agent writes to itself and to each neighbour every phase, which their brains read.
- **[mobility](measuring-tokens.md#mobility)** — Whether the rich stay rich, or who holds the wealth keeps changing.
- **[Muller plot](measuring-evolution.md#muller-plots)** — A chart of how lines of descent grow, shrink and replace each other over time.
- **[mutation](rules-in-detail.md#how-a-brain-changes)** — A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.
- **[neighbour](the-world.md#the-network)** — An agent joined to another by a connection.
- **[open-ended evolution](measuring-evolution.md#what-would-count-as-open-ended-evolution)** — Evolution that keeps producing new things without end, not only up to a plateau.
- **[participation ratio](statistics.md#how-many-directions-a-cloud-of-points-uses)** — How many directions a cloud of points really uses.
- **[path length](measuring-networks.md#path-length)** — How many connections the shortest way between two agents crosses.
- **[permutation test](statistics.md#permutation-tests)** — Whether a difference could be chance, found by shuffling which world belongs to which condition.
- **[pilot](runs-and-records.md#pilot-worlds)** — A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.
- **[power law](statistics.md#fitting-a-power-law-properly)** — A quantity whose frequency falls as a power of its size: few huge values, many small ones.
- **[power spectrum](statistics.md#the-power-spectrum-of-a-time-series)** — How much of a series' wandering comes at each speed, from slow to fast.
- **[preferential attachment](measuring-networks.md#preferential-attachment-and-its-kernel)** — The well connected gaining new connections faster than the poorly connected.
- **[random twin](measuring-space.md#the-random-twin-is-a-world-a-space)** — A network with the same agents and as many links each, wired at random: what a world would be if it were no space at all.
- **[revolution](rules-in-detail.md#how-a-coalition-takes-a-node)** (also coalition) — A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.
- **[richest tenth](measuring-tokens.md#the-richest-tenths-share)** — The share of all tokens held by the richest tenth of the agents.
- **[same-position share](measuring-health.md#the-same-position-share)** — Of two agents put in the same situation, how often they stake their tokens measurably differently.
- **[scaling relation](measuring-networks.md#scaling-relations-between-agents-properties)** — How one property of an agent grows with another, as a power.
- **[seed](rules-in-detail.md#seeds-and-the-random-stream)** — The number that fixes every random choice of a run: the same settings and the same seed make the same world.
- **[setting](settings.md)** — One number or switch of a run's rules, fixed before it starts.
- **[settled life](statistics.md#the-settled-life-of-a-world)** — A world from iteration 500 to its end, after its youth, where it is measured.
- **[share function](rules-in-detail.md#the-share-function-fa-b)** — How a brain's two numbers become one share between 0 and 1: of tokens for a child, or of a stake marked revolutionary.
- **[share-out](b1-lives-and-wealth.md#gains-losses-and-the-share-out)** — How a cleanup deals the tokens of the dead out among the living: in the baseline, each token to a survivor drawn at random.
- **[spectral dimension](measuring-space.md#the-spectral-dimension)** — A network's dimension read from how a random walker spreads over it.
- **[spectral gap](measuring-networks.md#the-spectral-gap)** — How hard a network is to cut into two halves, and how fast a walker on it forgets where it began.
- **[survival curve](measuring-lives.md#survival-curves-kaplanmeier)** (also Kaplan–Meier) — The share of lives that last at least so long, counted fairly when some are cut short by the end of a run.
- **[token](the-world.md#agents-and-tokens)** — The one thing of value in a world: every agent holds a whole number of them, and their total never changes.
- **[token curvature](b1-lives-and-wealth.md#token-curvature)** — How much richer an agent's neighbourhood is than the agent itself.
- **[token flow](measuring-tokens.md#token-flow)** (also net flow) — The tokens a game moves along each connection, from the agent who stakes to the node staked on.
- **[Wilson interval](statistics.md#the-wilson-interval)** — The range a share probably lies in, when it was counted in only a few worlds.
- **[yes-or-no decision](rules-in-detail.md#a-yes-or-no-decision)** — A yes or no a brain gives from four numbers: how strongly it means yes and no, and whether the stronger always wins or chance decides in proportion.
<!-- /glossary -->

<!-- turns -->
---

[Reference: the rules and measures in detail](../README.md#reference-the-rules-and-measures-in-detail) · [The rules in detail](rules-in-detail.md) →

**Also about:** [Start here](../README.md#start-here).

> [!info]- Pages that link here (3)
> [Welcome](../README.md) · [What this notebook is about](what-this-notebook-is-about.md) · [How this notebook is written](how-this-notebook-is-written.md)
<!-- /turns -->
