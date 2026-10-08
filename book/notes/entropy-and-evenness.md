# Entropy and evenness

How evenly is something shared out — tokens over agents, agents over
genotypes, agents over numbers of connections? **Shannon entropy** answers
with one number, measured in **bits** (Shannon 1948). This note explains it
from the beginning, then the two numbers built on it that the book uses:
**evenness** and the **effective number**.

## The idea: how many yes-or-no questions?

Suppose you pick one item at random and want to find out which category it
belongs to by asking yes-or-no questions. If there are 8 equally common
categories, 3 questions always suffice (is it in the first four? in the
first two of those? …): 8 = 2³. If one category holds almost everything,
you need almost no questions — the answer is nearly certain. Entropy is the
average number of questions an optimal questioner needs. It is large when
things are spread evenly over many categories and small when they are
concentrated.

## The definition

If the categories have shares *p*₁, …, *p*_k (each ≥ 0, adding up to 1), the
entropy is

$$
H = -\sum_{i=1}^{k} p_i \log_2 p_i \qquad (\text{with } 0 \log_2 0 = 0) .
$$

- *H* = 0 when one share is 1 and the rest are 0: no uncertainty.
- *H* = log₂ *k* when every share is 1/*k*: the most uncertainty *k*
  categories allow.

**Example.** Shares (½, ¼, ¼): *H* = −(½·log₂ ½ + ¼·log₂ ¼ + ¼·log₂ ¼) =
½·1 + ¼·2 + ¼·2 = 1.5 bits. Three equal shares would give log₂ 3 ≈ 1.585.

## Three uses in Graph of Life

| what is shared | the shares *pᵢ* | the field | its ceiling |
|---|---|---|---|
| tokens over agents | τᵢ / *T*, one per agent | `tokenEntropy` | log₂ *n* (*n* agents) |
| agents over degrees | share of agents with exactly *d* connections, one per *d* that occurs | `degreeEntropy` | log₂ (number of distinct degrees) |
| agents over genotypes | share of agents carrying genotype *g*, one per genotype | (computed for [Chapter 18](../chapters/18-how-even-is-a-world.md)) | log₂ *n* |

For tokens the question is: *if you pick one token at random, whose is it?*

## Evenness

Entropy grows with the number of categories, so the entropy of a world of
2,000 agents cannot be compared with one of 1,000. Dividing by the ceiling
removes that:

$$
J = \frac{H}{H_{\max}} ,
$$

between 0 and 1 — **Pielou's evenness** (Pielou 1966). `tokenEvenness` is
`tokenEntropy` / log₂ *n*; `degreeEvenness` is `degreeEntropy` divided by
log₂ of the number of distinct degrees.

## The effective number

$$
N_{\text{eff}} = 2^{H}
$$

is the number of **equally common** categories that would have the same
entropy. Shares (½, ¼, ¼) have *N*_eff = 2^1.5 ≈ 2.83: as diverse as 2.83
equal categories. Ecologists call this a Hill number (Hill 1973; Jost 2006);
it is the most readable form of entropy, because it is in the units of the
things counted. For tokens, *N*_eff is the number of agents who, holding equal
shares, would spread the tokens as evenly as they are spread.

## Entropy and the Gini coefficient

Both measure how unequally tokens are held, but they weigh differently. The
[Gini coefficient](gini-coefficient.md) is about differences between agents;
entropy is about the logarithms of shares, so it is more sensitive to the many
small holdings and less to the few large ones. Two worlds with the same Gini
can have different evenness ([Chapter 18](../chapters/18-how-even-is-a-world.md)).

## In the code

`_shannon(counts)` in `gol_series.py`, applied to the token counts and to the
histogram of degrees.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
