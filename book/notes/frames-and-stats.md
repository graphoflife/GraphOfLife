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
world after each game — and about seventy statistics. The ones this book
uses:

| field | what it is | note |
|---|---|---|
| `nodes` | agents alive | |
| `nodes_before` | agents alive when the phase began | |
| `edges` | connections | |
| `meanDegree` | connections per agent, 2·`edges`/`nodes` | [Degree](degree.md) |
| `leaves` | agents with exactly one connection | [Core, trees and leaves](core-trees-leaves.md) |
| `meanTokens`, `medianTokens`, `maxTokens` | tokens per agent: mean, median, largest | |
| `gini` | inequality of tokens | [Gini coefficient](gini-coefficient.md) |
| `topDecileShare` | share of all tokens held by the richest tenth | [The richest tenth](richest-tenth.md) |
| `distinctBrains` | genotypes among the living | [Genotypes](genotype.md) |
| `cladesInWindow` | families: ancestors eight iterations back | [Families](families.md) |
| `births` | children born (phase 1 rows) | [Births and deaths](births-and-deaths.md) |
| `starved`, `orphaned` | agents the cleanup removed for having no tokens, or for being cut off | [Births and deaths](births-and-deaths.md) |
| `meanInvestedShare` | mean share of its tokens a parent gave its child (phase 1) | |
| `reproTokenShare` | all tokens given to children, as a share of all tokens (phase 1) | |
| `selfAllocationShare` | share of all staked tokens staked on the staker's own node (phase 2) | [Staking at home](home-stake.md) |
| `heldHomeShare` | share of nodes won by their own agent (phase 2) | [Staking at home](home-stake.md) |
| `spreadShare` | share of stakers who spread rather than went all in (phase 2) | |
| `revolutions` | nodes won by a coalition (phase 2) | [Revolution](revolution.md) |

**Graph statistics** cost far more to compute, and are filled in only on the
rows of every 25th iteration (0, 25, 50, …); on other rows they are `null`:

| field | what it is | note |
|---|---|---|
| `transitivity` | clustering | [Clustering](clustering.md) |
| `meanPathLength` | average distance between agents | [Path length](path-length.md) |
| `coreShare` | share of agents in the 2-core | [Core, trees and leaves](core-trees-leaves.md) |
| `bridges` | connections whose removal would split the network | [Bridges](bridges.md) |

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
