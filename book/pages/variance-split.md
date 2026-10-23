# Where, who and chance: splitting a decision's variance

Agents decide differently from one another. Is that because they are in
different situations (where), because their brains differ (who), because
each brain reacts differently to the situation it is in (who, there), or
because of the random inputs every brain reads (chance)? This note shows how
the variance of a decision splits into those four parts.

## The design

Take B brains and S situations, and put every brain into every situation K
times, each time with fresh random inputs — but the *same* K draws for every
brain, so that the brains meet exactly the same inputs. Write d_{bsk} for
what brain b decides in situation s on draw k (a share of tokens, a yes or
no as 1 or 0). Then the averages

- \bar d_b — brain b over all situations and draws,
- \bar d_s — situation s over all brains and draws,
- \bar d_{bs} — brain b in situation s, over its draws,
- \bar d — everything,

split the sum of squared deviations from the grand mean exactly into four
parts (a two-way analysis of variance with replicates):

$$
\begin{aligned}
\text{situation} &= B K \sum_s (\bar d_s - \bar d)^2 \\
\text{brain} &= S K \sum_b (\bar d_b - \bar d)^2 \\
\text{brain in situation} &= K \sum_{b,s} (\bar d_{bs} - \bar d_b - \bar d_s + \bar d)^2 \\
\text{chance} &= \sum_{b,s,k} (d_{bsk} - \bar d_{bs})^2
\end{aligned}
$$

Each part divided by their sum is a **share** of the variance.

## How to read the shares

- **situation** large: every brain does the same thing in the same place,
  and agents differ because their places do;
- **brain** large: some brains always do more of it than others, wherever
  they are — a trait, like a temperament;
- **brain in situation** large: brains differ in *how they respond* — one
  gives more when rich, another when poor;
- **chance** large: the random inputs decide, and a brain does not do the
  same thing twice.

"Do agents behave the same in the same circumstance?" is answered by the
last three together, against chance alone: if brain and brain-in-situation
are small, agents in the same place act alike, and their differences are the
dice.

## Units as well as shares

Shares hide sizes: a decision every agent makes the same way has a tiny
variance, and its shares can be anything. So the decision's spread is also
given in its own units: how far apart two agents in the same situation and
draw are (the standard deviation over agents), against how far one brain is
from itself over the draws.

[Chapter 70](how-different-are-the-agents.md), *How different are the agents?*, uses this split.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
