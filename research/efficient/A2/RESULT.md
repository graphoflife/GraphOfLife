# A2. Where do tokens come from? (RESULT)

World `B1-10000-085a32-s4801`, iterations 12,000-12,499 (1,000 frames). Game phase only (stakes, deaths, estates);
the reproduction phase is not in this budget.

## What was done
Lineage = a genotype present at the start of a block plus all descendants found through `parent_brain_ids` (never a gap).
**Over the whole 500 iterations this is useless: of 1,530 start lineages exactly one is left at the end**
(the same in every 100-iteration block: the population turns over in a few iterations, median age 1) so
there are no ten winners and ten losers. I therefore cut the window in **20 blocks of 25 iterations**, started lineages afresh in each,
and pooled. Winners = the (up to) 10 lineages per block that gained most nodes (50 lineage-blocks in all; blocks end with only 1 to 10 lineages alive);
losers = the 10 that lost most (200); rest = 27,117 (almost all die out in the block).

## The table (tokens per agent per iteration; an agent = a node at the start of a game)
| group | agent-iterations | tokens per agent | own stake kept | kin stake kept (same genotype) | kin stake kept (same lineage, other genotype) | **prey** (stakes of other lineages on its nodes) | **lost** (its stakes on others' nodes) | **estate income** | tokens on its dead nodes |
|---|---|---|---|---|---|---|---|---|---|
| winners (top 10) | 1,327,769 | 2.15 | 1.251 | 0.315 | 0.480 | 0.102 | 0.101 | 0.493 | 0.391 |
| losers (bottom 10) | 407,733 | 2.03 | 1.187 | 0.330 | 0.325 | 0.181 | 0.183 | 0.396 | 0.509 |
| rest | 576,763 | 2.30 | 1.347 | 0.272 | 0.245 | 0.433 | 0.433 | 0.380 | 0.535 |
| world | 2,312,265 | 2.16 | 1.264 | 0.307 | 0.394 | 0.198 | 0.198 | 0.448 | 0.448 |

Net change of tokens per agent-iteration in the game: winners +0.102, losers -0.114, rest -0.155, world 0.
Estate / prey: winners 4.9, losers 2.2, rest 0.9, world 2.3.
(Prey equals lost in the world row by construction; estate equals "tokens on dead nodes" for the world.)
With the card's wording (a stranger = any other genotype id, not lineage) prey would be prey + kin-lineage = 0.58 for the winners
against estate 0.49. The genotype id changes by mutation in 20% of all copies, so a "stranger" by id is mostly a mutant cousin;
I count them as kin because their tokens end up in the same lineage.

## Reading
- Winners earn 4.9 times more from estates than from prey (0.49 vs 0.10 per agent per iteration), and the winners'
  prey is the smallest of the three groups. Losers and the rest are the ones that lose the most to other lineages.
  The losers' deficit is mostly deaths: 0.51 of tokens per agent-iteration sit on nodes that die.
- Estates are large for everybody: 0.45 tokens per agent-iteration, 21% of all tokens in the game phase, and the estate
  rule `parents` pays them to the parents of this iteration's newborns (about 960 heirs for 1,960 tokens). So "per head" is "per child".
- **Decision rule: estate income is larger than prey for the winners, so: the world pays per head (strictly: per child born, rule `parents`);
  `redistribution` is the first rule to study in D1/D2.** (Under the stricter genotype-id reading of "stranger" the
  winners' prey 0.58 would beat estate 0.49; the card did not say which, and I chose lineage, as the tokens stay inside the lineage.)

## Caveats
One world, one stretch (12,000-12,499), game phase only; the lineage collapse means winners/losers are lineages of one to a few dozen nodes of blocks of 25 iterations;
the blocks are not independent; winners include lineages that are simply the survivor of the block. Births, gifts and upkeep payments are not in the table.
