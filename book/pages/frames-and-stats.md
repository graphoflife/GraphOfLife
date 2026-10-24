# What a run records

Everything the notebook shows is computed from two kinds of record that every
run writes into its folder, `GraphOfLifeRuns/<run>/`: **frames** and a
**statistics file**. This note says what is in them and how to read them, so
that every figure can be made again by hand.

## Frames

After every phase of every [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."), the whole world is written down as a
**frame**. Iteration *t* has two:

| frame | when | `phase` |
|---|---|---|
| 2*t* | after the reproduction phase of iteration *t* (and its [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.")) | 1 |
| 2*t* + 1 | after the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") of iteration *t* (and its cleanup) | 2 |

A run of 3,000 iterations has 6,000 frames. A frame holds, for every [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")
alive, in increasing order of id:

- `ids` — its id;
- `tokens` — its [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.");
- `brain_ids` and `parent_brain_ids` — its genotype and that genotype's
  parent ([Genotypes](genotype.md));
- `parent_ids` — the id of the agent it was born to (−1 for a [founder](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts."));
- `ages` — iterations since it was born (0 in the iteration of its birth);
- `delta` — how many tokens it gained (+) or lost (−) in this phase;

and, for the world: `edges`, every [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") as a pair of ids;
`nodes_before`, how many agents there were when the phase began; `cleanup`,
how many the cleanup removed and why; and `decisions`, every decision of the
phase — every birth with the tokens given, the links and the handovers, and
every stake in the game with its targets, amounts and revolutionary parts,
and who won each node.

Frames are read with `gol_store.read_frame(run, index)`.

## The statistics file

From every frame, a row of statistics is computed and appended to
`GraphOfLifeRuns/<run>/stats.jsonl`, one JSON object per line. Every row
carries `iteration` and `phase` — so "the rows with `phase` = 2" are the
world after each game — and about ninety statistics. They are the numbers
the viewer charts, and every one has a note that says how it is computed.

**On every row:**

| field | what it is | note |
|---|---|---|
| `nodes`, `edges` | agents alive, connections | |
| `nodes_before` | agents alive when the phase began | |
| `tokens` | all tokens in the world (the same on every row) | |
| `meanTokens`, `medianTokens`, `minTokens`, `maxTokens` | tokens per agent | [Tokens per agent](gini-coefficient.md#tokens-per-agent-mean-median-extremes) |
| `gini` | inequality of tokens | [Gini coefficient](gini-coefficient.md) |
| `topDecileShare` | share of all tokens held by the richest tenth | [The richest tenth](gini-coefficient.md#the-richest-tenths-share) |
| `tokenEntropy`, `tokenEvenness`, `degreeEntropy`, `degreeEvenness` | how evenly tokens and connections are spread, in bits and as a share of the most even | [Entropy and evenness](gini-coefficient.md#entropy-and-evenness) |
| `meanDegree`, `medianDegree`, `minDegree`, `maxDegree`, `density` | connections per agent; `density` is the share of all possible pairs that are joined | [Degree](degree.md) |
| `leaves` | agents with exactly one connection | [Core, trees and leaves](bridges.md#core-trees-and-leaves) |
| `distinctBrains`, `distinctParents`, `brainDiversity` | genotypes among the living, their parent genotypes, genotypes per agent | [Brain diversity](genotype.md#brain-diversity) |
| `cladesInWindow` | families: ancestors eight iterations back | [Families](genotype.md#families) |
| `births` | children born (phase 1 rows) | [Births and deaths](births-and-deaths.md) |
| `starved`, `orphaned` | agents the cleanup removed for having no tokens, or for being cut off; deaths of age ([B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s [`max_age`](settings.md#max_age)) are in no field, only in a frame's `cleanup.aged` | [Births and deaths](births-and-deaths.md) |
| `gainers`, `losers`, `maxTokenAdded`, `maxTokenLost`, `redistributed` | who gained and who lost in the phase, the extremes, and the tokens of the dead dealt out | [Gains and losses](gains-and-losses.md#gains-losses-and-the-share-out) |
| `meanInvestedShare`, `reproTokenShare`, `meanChildLinks`, `handovers` | how parents gave (phase 1) | [Reproduction statistics](reproduction-statistics.md) |
| `selfAllocationShare`, `heldHomeShare` | staking on one's own node, and keeping it (phase 2) | [Staking at home](how-the-game-is-played.md#staking-at-home-and-keeping-ones-node) |
| `spreadShare`, `revoltShare`, `totalFlow`, `meanEdgeFlow`, `maxEdgeFlow`, `prunedEdges` | the stakes on [neighbours](the-world.md#the-network "An agent joined to another by a connection."), and the connections they keep alive (phase 2) | [Token flow](token-flow.md) |
| `revolutions` | nodes won by a coalition (phase 2) | [Revolution](revolution.md) |
| `cutRiskBefore` | the largest share of the world one cut could sever, before the cleanup (in a game's row, after its unused connections are cut) | [Cut risk](bridges.md#cut-risk) |
| `gifts`, `giftTokens`, `giftShare` | gifts of tokens (phase 1): 0 in a run without gifts, `null` in the game's rows ([`allow_gifting`](settings.md#allow_gifting)) | [Reproduction statistics](reproduction-statistics.md) |

> [!note] Corrected on 23 October 2026
> This table said the gift fields are `null` in every run of this notebook, which had no gifts. In a run
> without gifts they are 0 in every reproduction row, and gifts have been switched on in [pilots](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.") since
> 15 October 2026, in Experiment 24 and in half the worlds of
> [A screen of random worlds](a-screen-of-random-worlds.md). The table also did not say where deaths of
> age are counted.

**Graph statistics** cost far more to compute: they walk the whole network.
They are filled in only on the rows of every 25th iteration (0, 25, 50, …);
on other rows they are `null`:

| field | what it is | note |
|---|---|---|
| `transitivity`, `triangles` | clustering, and the number of triangles | [Clustering](degree.md#clustering) |
| `meanPathLength` | average distance between agents | [Path length](path-length.md) |
| `radius`, `diameter` | the smallest and largest eccentricity | [Radius and diameter](path-length.md#radius-and-diameter) |
| `coreShare` | share of agents in the 2-core | [Core, trees and leaves](bridges.md#core-trees-and-leaves) |
| `bridges` | connections whose removal would split the network | [Bridges](bridges.md) |
| `cutRisk` | the largest share of the world one cut could sever | [Cut risk](bridges.md#cut-risk) |
| `cycleRank`, `loopDensity`, `components` | independent loops, loops per connection, separate pieces | [Loops](bridges.md#loops) |
| `spectralGap` | how hard the network is to cut in two; how fast a random walk forgets | [The spectral gap](path-length.md#the-spectral-gap) |
| `dimension`, `ricciCurvature` | dimension and curvature from how balls grow | [Dimension and curvature](ball-dimension-and-curvature.md) |
| `boxDimension`, `boxDimensionR2` | dimension from covering the network with boxes | [Box dimension](ball-dimension-and-curvature.md#box-dimension) |
| `lightningScore`, `cyclingShare`, `lightningLongest`, `flowImbalance`, and the `net…` versions, `netFlowShare` | tokens that go round in loops (phase 2) | [Lightning](token-flow.md#lightning-tokens-that-go-round) |
| `degreeExponent`, `tokenExponent`, `degreeGamma`, `degreeKMin`, `degreeTailShare`, `degreeGammaKS`, and their `…R2` | power-law tails of the degrees and tokens | [Fitting a power law](power-law-fit.md) |
| `tokensVsDegree`, `trianglesVsDegree`, `clusteringVsDegree`, `changeVsTokens`, and their `…R2` | how one property of agents grows with another | [Scaling relations](degree.md#scaling-relations-between-agents-properties) |
| `assortativity` | whether [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") join hubs | [Assortativity](degree.md#assortativity) |

Fields beginning with `_` (`_seconds`, `_cpu`, `_peakMB`, …) are the costs of
the run, not of the world ([What a simulation costs](what-a-simulation-costs.md)).

The file is read with `gol_record.read_stats(run)`, which returns the rows
as a list of dictionaries, in frame order.

## Recomputing a statistic

Every statistic is a function of one frame (and, for a few, the frame
before), computed by `gol_series.frame_stats(frame, previous, heavy)`. The
families need every iteration's frame in order. So a statistic can always be
recomputed from the frames, and `python3 -c "import gol_record;
gol_record.record_stored('<run>')"` rebuilds a run's whole statistics file.

<!-- turns -->
---

← [Every simulation: what was run, and what was measured](every-simulation.md) · [Methods and statistics](../README.md#methods-and-statistics) · [What the viewer can colour by](viewer-colours.md) →

**Also about:** [How the simulation works](../README.md#how-the-simulation-works).

> [!info]- Pages that link here (34)
> [Glossary](glossary.md) · [The world: tokens, agents and a network](the-world.md) · [One iteration, step by step](one-iteration.md) · [The life of a B1 world](a-worlds-life.md) · [How much does the seed decide?](how-much-does-the-seed-decide.md) · [Building B2: a world that cannot break](a-world-that-cannot-break.md) · [Births, deaths and ages](births-deaths-and-ages.md) · [How births build the network](how-agents-have-children.md) · [Births and the four ways to die](births-and-deaths.md) · [Age, lifetime and survival curves](age-and-lifetime.md) · [Reproduction statistics](reproduction-statistics.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [Gains and losses: who gains, who loses, and the share-out](gains-and-losses.md) · [The largest and the smallest](the-largest-and-the-smallest.md) · [Measuring inequality: the Gini coefficient, the richest tenth, entropy and mobility](gini-coefficient.md) · [How the game is played: what agents stake, and who wins the nodes](how-the-game-is-played.md) · [Where tokens flow](where-the-tokens-flow.md) · [Measuring the flow: token flow, lightning and circulation](token-flow.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Power laws, real and apparent](power-laws-real-and-apparent.md) · [How properties scale together](how-properties-scale-together.md) · [How a world breaks](how-a-world-breaks.md) · [Distances in a network: path length, radius and diameter, the spectral gap](path-length.md) · [Three ways to measure a network's dimension](ball-dimension-and-curvature.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [Do the brains matter?](do-the-brains-matter.md) · [Genotypes, families and the common ancestor](genotype.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [How fast should brains change?](how-fast-should-brains-change.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Every world side by side](every-world-side-by-side.md) · [The lab, the engine and pilots](the-lab.md) · [How worlds are measured](how-worlds-are-measured.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
