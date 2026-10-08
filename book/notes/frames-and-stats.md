# What a run records

Everything the book shows is computed from two kinds of record that every
run writes into its folder, `GraphOfLifeRuns/<run>/`: **frames** and a
**statistics file**. This note says what is in them and how to read them, so
that every figure can be made again by hand.

## Frames

After every phase of every iteration, the whole world is written down as a
**frame**. Iteration *t* has two:

| frame | when | `phase` |
|---|---|---|
| 2*t* | after the reproduction phase of iteration *t* (and its cleanup) | 1 |
| 2*t* + 1 | after the game of iteration *t* (and its cleanup) | 2 |

A run of 3,000 iterations has 6,000 frames. A frame holds, for every agent
alive, in increasing order of id:

- `ids` — its id;
- `tokens` — its tokens;
- `brain_ids` and `parent_brain_ids` — its genotype and that genotype's
  parent ([Genotypes](genotype.md));
- `parent_ids` — the id of the agent it was born to (−1 for a founder);
- `ages` — iterations since it was born (0 in the iteration of its birth);
- `delta` — how many tokens it gained (+) or lost (−) in this phase;

and, for the world: `edges`, every connection as a pair of ids;
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
| `gainers`, `losers`, `maxTokenAdded`, `maxTokenLost`, `redistributed` | who gained and who lost in the phase, the extremes, and the tokens of the dead dealt out | [Gains and losses](gains-and-losses.md) |
| `meanInvestedShare`, `reproTokenShare`, `meanChildLinks`, `handovers` | how parents gave (phase 1) | [Reproduction statistics](reproduction-statistics.md) |
| `selfAllocationShare`, `heldHomeShare` | staking on one's own node, and keeping it (phase 2) | [Staking at home](home-stake.md) |
| `spreadShare`, `revoltShare`, `totalFlow`, `meanEdgeFlow`, `maxEdgeFlow`, `prunedEdges` | the stakes on neighbours, and the connections they keep alive (phase 2) | [Token flow](token-flow.md) |
| `revolutions` | nodes won by a coalition (phase 2) | [Revolution](revolution.md) |
| `cutRiskBefore` | the largest share of the world one cut could sever, before the cleanup | [Cut risk](cut-risk.md) |
| `gifts`, `giftTokens`, `giftShare` | gifts of tokens — `null` in every run of this book, which has no gifts ([`allow_gifting`](settings.md#allow_gifting)) | |

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
| `assortativity` | whether hubs join hubs | [Assortativity](assortativity.md) |

Fields beginning with `_` (`_seconds`, `_cpu`, `_peakMB`, …) are the costs of
the run, not of the world ([Appendix A](../chapters/A-costs.md)).

The file is read with `gol_record.read_stats(run)`, which returns the rows
as a list of dictionaries, in frame order.

## Recomputing a statistic

Every statistic is a function of one frame (and, for a few, the frame
before), computed by `gol_series.frame_stats(frame, previous, heavy)`. The
families need every iteration's frame in order. So a statistic can always be
recomputed from the frames, and `python3 -c "import gol_record;
gol_record.record_stored('<run>')"` rebuilds a run's whole statistics file.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
