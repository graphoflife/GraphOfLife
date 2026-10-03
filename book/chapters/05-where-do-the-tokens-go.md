# Where do the tokens go?

## In short

Into a moderately unequal world that stays that way. Within the first few
iterations the founders' perfect equality is gone, and from then on the
richest tenth of agents hold about 44% of all tokens and the Gini coefficient
of wealth sits close to 0.5 — unequal, but far from one agent owning
everything, and steady for three thousand iterations. The game is not what
the thesis expected: agents put only about 30% of their stakes on their own
node and 70% on their neighbours', and more than half of all nodes change
hands in every game. Children cost little: after its youth, a world spends
about 1% of its tokens on them per iteration.

## Thesis

```thesis E04
```

Tokens are the only thing of value in a world, and the only thing that is
conserved. Who holds them decides who can have children and who can win a
node. This chapter follows them.

## Method

```experiment E04
```

The thirty runs of Chapter 3, read for what happens to wealth:

- **the Gini coefficient** of tokens, from 0 (everyone holds the same) to 1
  (one agent holds everything); the founders start at exactly 0;
- **the share held by the richest tenth** of agents — 10% if wealth were even;
- in the game, **the share of staked tokens an agent puts on its own node**,
  rather than on a neighbour's;
- in reproduction, **the share of all tokens spent on children**.

As in Chapter 3, every figure shows the median of the thirty runs with the
middle half and nine in ten as bands, and the thesis is checked from
iteration 100 on.

## Results

### Inequality comes at once, and stays

```figure E04/gini
```

The founders start exactly equal, but the first reproduction phase ends that:
a child gets only part of its parent's tokens, and in the first five
iterations the median world's Gini coefficient is already 0.48. It rises to
0.54 in the boom of births and falls to about 0.41 when the boom fills the
world with poor agents (iterations 20 to 25). From then on it stays near
0.45 to 0.5 for the rest of the run.

Measured over each world's life from iteration 100 on, the median Gini is
0.496, and the middle half of the worlds lie between 0.48 and 0.52. Ten of
the 26 worlds that lived to the end were above 0.5, none was below 0.43. A
Gini of 0.5 means that two agents picked at random differ, on average, by as
much as the average agent holds.

```figure E04/top
```

The richest tenth hold a median 44% of all tokens (23 of the 26 worlds above
40%). The single richest agent holds about 8% — a median of 835 of the
10,000 tokens — while half of all agents hold about five tokens or fewer.

### The game is played on the neighbours

```figure E04/home
```

Of everything agents stake in a game, a median 30% goes on their own node,
from iteration 100 on. Not one of the 26 worlds put half of its stakes at
home; the most any put there was 34%. And this hardly changed in three
thousand iterations: the very first games, played by brains that had never
been selected, staked 27% at home, and the last ones 30%.

The result is a world of constant conquest. Only 45% of agents keep their
own node through a game (Chapter 8); the rest lose it to a neighbour, whose
brain the node then takes on.

### Children cost little

```figure E04/children
```

In their first iterations the founders, rich as they are, put 45% of all
tokens into children. That share falls steeply as the world fills: to 6% by
iteration 50, 2% by iteration 100, and about 1% or less for the rest of the run (a mean of
1.3% per reproduction phase from iteration 100 on).

### The dead give a little back

When agents die, their tokens are shared out among the survivors at random.
From iteration 100 on, that is a median of 117 tokens per game, about 1% of
all tokens.

## Conclusion

The thesis is refuted, though not in the direction its *refute* clause
feared most. Wealth does pool: inequality appears at once and stays. But it
pools only so far — the median Gini of 0.496 sits just below the 0.5 the
thesis asked for, while the richest tenth hold more than the 40% it asked
for. And the reason the thesis gave fails outright: agents do not stake most
of their tokens on their own node. They stake 70% on their neighbours', in
every world, from the first game to the last.

So the game is not a contest of each agent defending its own node. It is a
contest of everyone attacking everyone near them, in which more than half of
all nodes change hands every game. Something keeps the winners of that
contest from taking everything. Three things could:

- **coalitions** — a node can be taken from its biggest staker by a group of
  smaller ones, and half of all nodes are (Chapter 8);
- **turnover** — a node's tokens are staked anew in every game, so a big
  winner has to win again and again, against many neighbours at once;
- **the dead's tokens**, shared out at random — but at about 1% of the supply
  per game, they are the weakest of the three.

The first can be switched off on its own (`allow_revolutions`), and that is
the experiment this chapter asks for: does inequality rise without
coalitions?

## Details

| | |
|---|---|
| Runs | the 30 of Chapter 3: `B1-10000-s001` … `B1-10000-s030`; 26 reached iteration 3,000 |
| From iteration 100 | each world's mean over every game (reproduction phase, for children) from iteration 100 to 3,000; the median over the 26 worlds that reached the end |
| Gini | of the tokens of the living, after every game; 0 when everyone holds the same, 1 when one agent holds everything |
| Staked at home | tokens an agent stakes on its own node, as a share of all tokens staked in the game |
| Results | `book/results/E04.json`: every number above, and each world's value by seed |
| Made with | `python3 gol_lab.py analyse E04` |
