# Results of the efficient plan, as they come in

One entry a card. The plan is `plan.md`; each card's folder has the code and
the detail. Nothing here played a world unless it says so.

## Wave 1 (23 November 2026)

**The picture so far.** In Experiment 48's three worlds, selection sorts
brains but cannot improve one. A mutation turns a brain into something close
to an unrelated brain; the population has no common way of deciding; what it
does is reshuffled inside bounds; and the rules pay for having children, not
for playing well.

### B1. How big is one mutation, in behaviour? — too coarse

Share of 300 real situations in which a mutant decides differently from its
parent (4,000 mutants of 200 brains for each cell):

| world, iteration | two unrelated brains | engine | 1 number | 10 | 100 | 1,000 |
|---|---|---|---|---|---|---|
| s4003, 15,000 | 76% | 67% | 0.4% | 2.0% | 6.5% | 20% |
| s4003, 5,000 | 62% | 54% | 0.2% | 0.9% | 3.3% | 10% |
| s4008, 15,000 | 83% | 49% | 1.1% | 5.6% | 18% | 43% |
| s4002, 15,000 | 83% | 34% | 0.4% | 2.3% | 7.7% | 21% |

By the card's rule (more than half of the engine's mutants further than a
quarter of the distance between strangers): 69% to 98% are. The engine's
operator is too coarse in all four. In s4003 at 15,000, 47% of mutants are
as far from their parent as a stranger.

### B2. How many things can evolved brains do? — not low-dimensional

Among 1,000 brains of one kept world there are 420 to 910 genotypes. They
agree with the most common first move on a probe 39% to 66% of the time
(random founders: 30% to 35%), and 90% of the variance between them needs
110 to 200 components. No simple rule describes them, with one exception:
in s4008 at 17,500, an agent holding 2 or more tokens has a child in 98% to
99% of cases. A stand-in model (card E2) is not justified.

### A1. Does behaviour keep moving? — bounded reshuffling

Distance between the populations of two kept worlds, by the time between
them (first-token decisions; the floor is two halves of one population):

| world | floor | 2,500 apart | 10,000 apart | longest | slope |
|---|---|---|---|---|---|
| s4003 | 0.06 | 0.43 | 0.47 | 0.49 (12,500) | 0.03 |
| s4008 | 0.07 | 0.27 | 0.27 | 0.25 (15,000) | −0.06 |
| s4002 | 0.06 | 0.36 | 0.33 | 0.31 (25,000) | −0.04 |

Far above the floor at the shortest lag, and no growth with the lag. By the
card's rule: bounded wandering, faster than one kept world to the next. The
populations are also no further from their random founders than from each
other.

### D1. The rules as physics — five laws hold, five were wrong in the plan

Checked on 100 game frames of s4003 (iterations 12,000 to 12,099). Detail in
`D1/RESULT.md`.

Hold exactly: tokens are conserved; a token moves at most one link a game;
a node keeps every token staked on it; a failed attack feeds the defender;
a node with no stake starves.

Corrected:
- These worlds play the **lottery**. The largest stake loses 14.7% of
  contested nodes, and clones' stakes add up. "Many small kin stakes do not
  defend" is false here.
- **Estates go to the parents of this iteration's newborns** (`redistribution`
  = parents), one share for each child, not to everyone alike. 27% of all
  tokens move this way every iteration.
- **A core is safe only within one game.** 62% of nodes whose neighbours
  all share their genotype have a neighbour of another genotype one
  iteration later, mostly a mutant child of kin.
- A link is cut only if no stake crossed it, it was not paid for and it was
  not made this iteration.
- An agent that places its tokens one at a time also sees the board.

Numbers that matter for everything else:
- **An agent lives a median of 1 iteration**; 99% live under 9.
- About 830 agents an iteration die in culls, 390 starve, of 1,900 to 6,400.
- 43% of agents hold 1 token, 23% hold 2.
- **The surface law fails**: 79% to 92% of a territory's nodes are border
  nodes at every size. Territories are thin, not round.
- Help from more than 2 to 4 links away arrives after the agent is dead.

### A2. Where do tokens come from? — the world pays for children

Tokens per agent and iteration in the game, s4003, iterations 12,000 to
12,499:

| group | taken from others | lost to others | from estates |
|---|---|---|---|
| lineages that grew most | 0.10 | 0.10 | 0.49 |
| lineages that shrank most | 0.18 | 0.18 | 0.40 |
| the whole world | 0.20 | 0.20 | 0.45 |

Winning lineages get about five times as much from estates as from the
game, and take no more than they lose. By the card's rule: the world pays
per child. `redistribution` is the first rule to study.

### A6. The entropy cycle

About a fifth of all tokens pass through the estates of the dead every
iteration in all three worlds. s4003 drifts towards its most mixed state
(tokens per agent 4.0 to 2.9, agents with one token 23% to 33%); s4008 moves
away from it and stays there; s4002 falls slowly.

### A5. Does the largest structure grow? — no

Territories of one genotype exist well beyond a shuffled world in all three
(s4003: 340 to 420 of two or more agents against 70 to 150 shuffled; largest
70 to 170 against 10 to 90). In s4002 and s4008 they stay small (largest
about 6 and 12). None grows from window to window by the card's rule. Not
done: lifetimes, family-based territories, the family-pool runs.

### C1. The tagged-lineage assay — built

A lab trial of kind `assay` (`gol_tournament.assay`), with tests. Its first
use is Experiment 49 (card C2).

## What this changes in the plan

- **Condition 2 fails as far as behaviour goes** (B1). Experiment 49 asks
  whether it fails in fitness too.
- **Condition 4 cannot hold in these three worlds as they are**: agents live
  one iteration, territories are all border, and income comes from having
  children. A shell and a core (cards A3, D3, D4) need agents that last. The
  rule sets to look at next are those where agents live long: no cull of
  whole pieces every game (`mend_on_death`), estates not paid for births.
- All three of Experiment 48's worlds share `contest` = lottery and
  `redistribution` = parents. The evolution rank that chose them rewards
  fast turnover.
