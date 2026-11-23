# D2. Exact small games (RESULT)

Question: does a team of n do better per token than loners, and does that grow with n? One phase, one team (one joint
plan) against one outsider, exact payoff tables, zero-sum solved. Code: `d2.py`; all numbers: `d2_result.json`, `table.md`.
Values below are the team's value divided by n*w (1.000 = it keeps what it had). Gap between the two sides' guaranteed
values: below 0.0065 in all but two (rule, case) pairs; the largest is 0.0108 (clique n=4 all border w=3 A, largest), then 0.0082 (clique n=3 all border w=3 A, largest). Conservation: exact in every checked case.

Setup. Outsider A = n*w tokens, B = w tokens. "one" = outsider linked to node 0 only, "all" = linked to every team node.
For n=1 both attachments and both motifs are the same game (value 1.000 everywhere); clique starts at n=2.

Rules (code): largest = `_resolve_winner` (GraphOfLife.py 2395), lottery = `_lottery` (2381), pool = `_pooled` (2349).
Revolutions: skipped (flagged tokens are a brain output; modelling them would be a guess).

### largest

| motif | attach | w | outsider | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|---|---|---|
| path | one border | 2 | A | 1.000 | 1.000 | 1.000 | 1.000 |
| path | one border | 2 | B | 1.000 | 1.000 | 1.000 | 1.000 |
| path | one border | 3 | A | 1.000 | 1.000 | 1.000 | 1.000 |
| path | one border | 3 | B | 1.000 | 1.000 | 1.000 | 1.000 |
| path | all border | 2 | A | 1.000 | 0.667 | 0.500 | 0.600 |
| path | all border | 2 | B | 1.000 | 1.000 | 1.000 | 1.000 |
| path | all border | 3 | A | 1.000 | 0.667 | 0.500 | 0.400 |
| path | all border | 3 | B | 1.000 | 1.000 | 1.000 | 1.000 |
| clique | one border | 2 | A | - | 1.000 | 1.000 | 1.000 |
| clique | one border | 2 | B | - | 1.000 | 1.000 | 1.000 |
| clique | one border | 3 | A | - | 1.000 | 1.000 | 1.000 |
| clique | one border | 3 | B | - | 1.000 | 1.000 | 1.000 |
| clique | all border | 2 | A | - | 0.667 | 0.500 | 0.600 |
| clique | all border | 2 | B | - | 1.000 | 1.000 | 1.000 |
| clique | all border | 3 | A | - | 0.667 | 0.500 | 0.400 |
| clique | all border | 3 | B | - | 1.000 | 1.000 | 1.000 |

### lottery

| motif | attach | w | outsider | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|---|---|---|
| path | one border | 2 | A | 1.000 | 1.000 | 1.000 | 1.000 |
| path | one border | 2 | B | 1.000 | 1.000 | 1.000 | 1.000 |
| path | one border | 3 | A | 1.000 | 1.000 | 1.000 | 1.000 |
| path | one border | 3 | B | 1.000 | 1.000 | 1.000 | 1.000 |
| path | all border | 2 | A | 1.000 | 1.000 | 1.000 | 1.000 |
| path | all border | 2 | B | 1.000 | 1.000 | 1.000 | 1.000 |
| path | all border | 3 | A | 1.000 | 1.000 | 1.000 | 1.000 |
| path | all border | 3 | B | 1.000 | 1.000 | 1.000 | 1.000 |
| clique | one border | 2 | A | - | 1.000 | 1.000 | 1.000 |
| clique | one border | 2 | B | - | 1.000 | 1.000 | 1.000 |
| clique | one border | 3 | A | - | 1.000 | 1.000 | 1.000 |
| clique | one border | 3 | B | - | 1.000 | 1.000 | 1.000 |
| clique | all border | 2 | A | - | 1.000 | 1.000 | 1.000 |
| clique | all border | 2 | B | - | 1.000 | 1.000 | 1.000 |
| clique | all border | 3 | A | - | 1.000 | 1.000 | 1.000 |
| clique | all border | 3 | B | - | 1.000 | 1.000 | 1.000 |

### pool

| motif | attach | w | outsider | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|---|---|---|
| path | one border | 2 | A | 1.000 | 1.000 | 1.000 | 1.000 |
| path | one border | 2 | B | 1.000 | 1.000 | 1.000 | 1.000 |
| path | one border | 3 | A | 1.000 | 1.000 | 1.000 | 1.000 |
| path | one border | 3 | B | 1.000 | 1.000 | 1.000 | 1.000 |
| path | all border | 2 | A | 1.000 | 1.000 | 1.000 | 1.000 |
| path | all border | 2 | B | 1.000 | 1.167 | 1.167 | 1.100 |
| path | all border | 3 | A | 1.000 | 1.000 | 1.000 | 1.000 |
| path | all border | 3 | B | 1.000 | 1.167 | 1.167 | 1.150 |
| clique | one border | 2 | A | - | 1.000 | 1.000 | 1.000 |
| clique | one border | 2 | B | - | 1.000 | 1.000 | 1.000 |
| clique | one border | 3 | A | - | 1.000 | 1.000 | 1.000 |
| clique | one border | 3 | B | - | 1.000 | 1.000 | 1.000 |
| clique | all border | 2 | A | - | 1.000 | 1.000 | 1.000 |
| clique | all border | 2 | B | - | 1.167 | 1.166 | 1.100 |
| clique | all border | 3 | A | - | 1.000 | 1.000 | 1.000 |
| clique | all border | 3 | B | - | 1.166 | 1.167 | 1.150 |

## Answers
1. **Size pays?** Lottery: flat at exactly 1.000 everywhere (home stakes keep your tokens in expectation, so neither side can
   gain; a zero-sum game with both sides guaranteed their own tokens). Largest, one border: flat at 1.000. Largest, all border, outsider
   A: value falls with n (path/clique w=3: 1.0, 0.667, 0.5, 0.4; w=2: 1.0, 0.667, 0.5, 0.6): size hurts. Largest, outsider B: flat 1.000.
   Pool, outsider A: flat 1.000. Pool, outsider B, all border: rises from n=1 to n=2 (1.000 to 1.167) and then falls a little
   (n=4: 1.10 to 1.15), always above 1: pooling lets two or more agents take from a poor outsider. No rule shows a steady rise with n.
2. **One border vs all border.** One border is never worse and is better where size hurts (largest A: 1.0 against 0.4-0.67).
   The inside helps by letting the border node be left empty. Under pool vs a poor outsider the opposite: all border wins (1.15 against 1.0).
3. **Best plans.** Largest, one border: the border agent sends all tokens inward and keeps nothing on the border node; the outsider
   then takes an empty node (or sits at home). The border node shows 0 tokens; no one keeps one token at home on it. Largest,
   all border, A: team puts whole stacks on one or two nodes and gives up the rest (random mixed); the outsider
   mixes big stakes (4-5 tokens) on two or three nodes. Pool: team stacks every token on one node (1 or 2 inside nodes), the
   outsider all-in on one node. Lottery: any plan that keeps tokens home or spreads evenly gives 1.000; outsider likewise.
4. **Surprises.** Lottery is trivially flat. Under largest with all border a bigger team is worse, not better.
   w=2 and w=3 differ at n=4 (0.6 vs 0.4): whole-token lumps matter.

## Limits
One phase; no children, no link cuts, no estates, no mutation; sizes 1-4 and 2-3 tokens; one outsider node (no team against team);
the outsider's own node holds nothing if it stakes away (it dies, tokens conserved). Largest all-border cases are solved to gaps
0.001-0.011, not exactly. Revolutions and family pools beyond two sides not done. Real worlds have 1-3 tokens per agent and
ties are common, so lumpiness matters.
