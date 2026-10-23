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
| `meanTokens`, `medianTokens`, `minTokens`, `maxTokens` | tokens per agent | [Tokens per agent](tokens-per-agent.md) |
| `gini` | inequality of tokens | [Gini coefficient](gini-coefficient.md) |
| `topDecileShare` | share of all tokens held by the richest tenth | [The richest tenth](richest-tenth.md) |
| `tokenEntropy`, `tokenEvenness`, `degreeEntropy`, `degreeEvenness` | how evenly tokens and connections are spread, in bits and as a share of the most even | [Entropy and evenness](entropy-and-evenness.md) |
| `meanDegree`, `medianDegree`, `minDegree`, `maxDegree`, `density` | connections per agent; `density` is the share of all possible pairs that are joined | [Degree](degree.md) |
| `leaves` | agents with exactly one connection | [Core, trees and leaves](core-trees-leaves.md) |
| `distinctBrains`, `distinctParents`, `brainDiversity` | genotypes among the living, their parent genotypes, genotypes per agent | [Brain diversity](brain-diversity.md) |
| `cladesInWindow` | families: ancestors eight iterations back | [Families](families.md) |
| `births` | children born (phase 1 rows) | [Births and deaths](births-and-deaths.md) |
| `starved`, `orphaned` | agents the cleanup removed for having no tokens, or for being cut off | [Births and deaths](births-and-deaths.md) |
| `gainers`, `losers`, `maxTokenAdded`, `maxTokenLost`, `redistributed` | who gained and who lost in the phase, the extremes, and the tokens of the dead dealt out | [Gains and losses](gains-losses-and-the-share-out.md) |
| `meanInvestedShare`, `reproTokenShare`, `meanChildLinks`, `handovers` | how parents gave (phase 1) | [Reproduction statistics](reproduction-statistics.md) |
| `selfAllocationShare`, `heldHomeShare` | staking on one's own node, and keeping it (phase 2) | [Staking at home](home-stake.md) |
| `spreadShare`, `revoltShare`, `totalFlow`, `meanEdgeFlow`, `maxEdgeFlow`, `prunedEdges` | the stakes on [neighbours](the-world.md#the-network "An agent joined to another by a connection."), and the connections they keep alive (phase 2) | [Token flow](token-flow.md) |
| `revolutions` | nodes won by a coalition (phase 2) | [Revolution](revolution.md) |
| `cutRiskBefore` | the largest share of the world one cut could sever, before the cleanup | [Cut risk](cut-risk.md) |
| `gifts`, `giftTokens`, `giftShare` | gifts of tokens — `null` in every run of this notebook, which has no gifts ([`allow_gifting`](settings.md#allow_gifting)) | |

**Graph statistics** cost far more to compute: they walk the whole network.
They are filled in only on the rows of every 25th iteration (0, 25, 50, …);
on other rows they are `null`:

| field | what it is | note |
|---|---|---|
| `transitivity`, `triangles` | clustering, and the number of triangles | [Clustering](clustering.md) |
| `meanPathLength` | average distance between agents | [Path length](path-length.md) |
| `radius`, `diameter` | the smallest and largest eccentricity | [Radius and diameter](radius-and-diameter.md) |
| `coreShare` | share of agents in the 2-core | [Core, trees and leaves](core-trees-leaves.md) |
| `bridges` | connections whose removal would split the network | [Bridges](bridges.md) |
| `cutRisk` | the largest share of the world one cut could sever | [Cut risk](cut-risk.md) |
| `cycleRank`, `loopDensity`, `components` | independent loops, loops per connection, separate pieces | [Loops](loops.md) |
| `spectralGap` | how hard the network is to cut in two; how fast a random walk forgets | [The spectral gap](spectral-gap.md) |
| `dimension`, `ricciCurvature` | dimension and curvature from how balls grow | [Dimension and curvature](ball-dimension-and-curvature.md) |
| `boxDimension`, `boxDimensionR2` | dimension from covering the network with boxes | [Box dimension](box-dimension.md) |
| `lightningScore`, `cyclingShare`, `lightningLongest`, `flowImbalance`, and the `net…` versions, `netFlowShare` | tokens that go round in loops (phase 2) | [Lightning](lightning.md) |
| `degreeExponent`, `tokenExponent`, `degreeGamma`, `degreeKMin`, `degreeTailShare`, `degreeGammaKS`, and their `…R2` | power-law tails of the degrees and tokens | [Fitting a power law](power-law-fit.md) |
| `tokensVsDegree`, `trianglesVsDegree`, `clusteringVsDegree`, `changeVsTokens`, and their `…R2` | how one property of agents grows with another | [Scaling relations](scaling-relations.md) |
| `assortativity` | whether [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") join hubs | [Assortativity](assortativity.md) |

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

> [!info]- Pages that link here (31)
> [Glossary](glossary.md) · [The world: tokens, agents and a network](the-world.md) · [One iteration, step by step](one-iteration.md) · [The life of a B1 world](a-worlds-life.md) · [How much does the seed decide?](how-much-does-the-seed-decide.md) · [Building B2: a world that cannot break](a-world-that-cannot-break.md) · [Births, deaths and ages](births-deaths-and-ages.md) · [How agents have children](how-agents-have-children.md) · [Age and lifetime](age-and-lifetime.md) · [Where do the tokens go?](where-do-the-tokens-go.md) · [How even is a world?](how-even-is-a-world.md) · [Gains and losses](gains-and-losses.md) · [The largest and the smallest](the-largest-and-the-smallest.md) · [How the game is played](how-the-game-is-played.md) · [Where the tokens flow](where-the-tokens-flow.md) · [What crosses a connection](what-crosses-a-connection.md) · [Staking at home, and keeping one's node](home-stake.md) · [What shape does the network take?](what-shape-does-the-network-take.md) · [Power laws, real and apparent](power-laws-real-and-apparent.md) · [How properties scale together](how-properties-scale-together.md) · [The geometry of a world](the-geometry-of-a-world.md) · [How a world breaks](how-a-world-breaks.md) · [Worlds of 100,000 tokens](worlds-of-100000-tokens.md) · [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md) · [Do the brains matter?](do-the-brains-matter.md) · [How fast should brains change?](how-fast-should-brains-change.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Every world side by side](every-world-side-by-side.md) · [The lab, the engine and pilots](the-lab.md) · [How worlds are measured](how-worlds-are-measured.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
