# Effective numbers: how many kinds, and how many directions

> [!question] Questions of this page
> - A world has 1,600 genotypes. Does it have 1,600 kinds of [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")?
> - When are two kinds really one?
> - The [messages](messages.md "Numbers an agent writes to itself and to each neighbour every phase, which their brains read.") of a world have 30 numbers each. Over how many directions
>   does their variety really spread?

## In short

Counting kinds can mislead: one kind may hold half the agents and the rest
one agent each. An **effective number** counts kinds by how much of the
population they hold. It is the number of equally common kinds that would be
as diverse. The same idea counts directions. The **participation ratio**
says over how many equally important directions a cloud of points really
spreads, such as a world's messages. Both run from 1, when one kind or one
direction holds everything, to the full count, when all are equal.

## Effective numbers: how many kinds there really are

A world has 1,600 genotypes. Does it have 1,600 kinds of agents? Not if one
genotype is carried by half the agents, and the rest by one agent each. The
**effective number** of kinds counts kinds by how much of the population they
hold.

### The count

Let *p*ᵢ be the share of agents of kind *i*: a genotype, a kind of behaviour,
a family. The effective number is

$$
D = \frac{1}{\sum_i p_i^2} .
$$

In words: *D* is one over the chance that two agents drawn at random are of
the same kind. That chance is the **Simpson index** (Simpson 1949). Two
properties make *D* the right count:

- if there are *n* kinds, equally common, then *D* = *n*;
- if one kind holds nearly everyone, then *D* is close to 1, however many rare
  kinds there are.

So *D* is "the number of equally common kinds that would be as diverse".

**Example.** Ten agents fall into two kinds of five: *D* = 1/(0.5² + 0.5²) =
2. Now let eight be of one kind, and one each of two others:
*D* = 1/(0.8² + 0.1² + 0.1²) = 1/0.66 ≈ 1.5. Three kinds, but worth only one
and a half.

*D* is one of the **Hill numbers**, the family of diversity counts ecologists
use (Hill 1973). This one, of order 2, weighs the common kinds most.

### Kinds that are similar

Two genotypes one [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") apart may act exactly alike. Counting them as two
kinds overstates diversity. That is why
[How different are the agents?](how-different-are-the-agents.md) groups
[brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") into kinds **by what they do**, not by their weights. Brains whose
decisions in the same situations differ no more than a brain differs from
itself, between two sets of random inputs, are one kind.

The effective number of kinds of behaviour is then set beside the effective
number of genotypes. The first can be far smaller, if most genetic
differences change nothing an agent does.

### Two effective numbers

[Brain diversity](genotype.md#brain-diversity) uses a sibling of this count:
2^*H*, with *H* the entropy of the shares
([The effective number](gini-coefficient.md#the-effective-number)). It is the
Hill number of order 1, which weighs rare kinds a little more than *D*, of
order 2, does. Both are 1 for a single kind and *n* for *n* equally common
ones. On a population with many rare kinds, 2^*H* is the larger. Where the
notebook gives an effective number, it says which.

## How many directions a cloud of points uses

A message of [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") has 30 numbers, so the messages of a world are points in a
space of 30 [dimensions](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space."). They need not use all 30. Suppose every message were
the same except for one number. Then the messages would vary along a single
direction, and the other 29 numbers would carry nothing. The **participation
ratio** counts how many directions the variety is really spread over.

### The recipe

1. **Covariance.** Take the points *x*₁, …, *x*ₙ, each a list of *d*
   numbers. Compute the *d* × *d* covariance matrix *C*. Its entry *Cᵢⱼ* is
   the covariance of number *i* with number *j* over the points.
2. **Principal directions.** The eigenvectors of *C* are the directions along
   which the cloud of points spreads, independently of each other: the
   principal components. The eigenvalue λₖ of each is the variance of the
   points along it. The eigenvalues add up to the total variance.
3. **Count them, weighted:**

$$
\text{PR} = \frac{\left(\sum_k \lambda_k\right)^2}{\sum_k \lambda_k^2} .
$$

In words: the square of the total variance, over the sum of the squared
variances of the directions. It is the effective number *D* above, with each
direction's share of the variance as its *p*.

### Reading the participation ratio

- If the variance is spread equally over *m* directions and the rest have
  none, then PR = *m*. For example, *m* eigenvalues of 1 give *m*²/*m* = *m*.
- If one direction holds almost everything, PR is close to 1.
- In between, PR is an **effective number** of directions, in the same sense
  as 2^*H* is an effective number of kinds
  ([Entropy and evenness](gini-coefficient.md#entropy-and-evenness)). It is
  how many equally important directions would spread the cloud as unevenly
  as it is spread.

So PR runs from 1 to *d*. For random points with independent numbers of equal
variance, it is close to *d*, and a little less because of sampling noise.
For 30 independent numbers measured on a few thousand points, it comes out
near 28 or 29.

### Where the participation ratio is used

[What agents say to each other](what-agents-say-to-each-other.md) uses it on
the messages agents write. There it counts how many of the 30 message numbers
a world's messages actually use. The measure comes from physics, where it
counts over how many sites a wave is spread (Bell and Dean 1970). In
neuroscience it measures the dimension of neural activity (Gao and others
2017).

## Where to go next

- [How different are the agents?](how-different-are-the-agents.md): kinds of
  behaviour, counted by what brains do, against the genotypes.
- [Measuring inequality](gini-coefficient.md#entropy-and-evenness): entropy,
  evenness and 2^*H*, the effective number of order 1.
- [What agents say to each other](what-agents-say-to-each-other.md): how many
  directions a world's messages use.
- [The diversity score](diversity-score.md): four spreads of a world in one
  number.

## References

- Bell, R. J. and Dean, P. (1970). Atomic vibrations in vitreous silica.
  *Discussions of the Faraday Society* 50, 55–61.
- Gao, P., Trautmann, E., Yu, B., Santhanam, G., Ryu, S., Shenoy, K. and
  Ganguli, S. (2017). A theory of multineuronal dimensionality, dynamics and
  measurement. *bioRxiv* 214262.
- Hill, M. O. (1973). Diversity and evenness: a unifying notation and its
  consequences. *Ecology* 54, 427–432.
- Simpson, E. H. (1949). Measurement of diversity. *Nature* 163, 688.

<!-- turns -->
---

← [Splitting variation: where, who and chance, and between worlds and within them](variance-split.md) · [Methods and statistics](../README.md#methods-and-statistics)

**Also about:** [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are).

> [!info]- Pages that link here (10)
> [Glossary](glossary.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [Measuring inequality: the Gini coefficient, the richest tenth, entropy and mobility](gini-coefficient.md) · [How alike the agents are](how-alike-the-agents-are.md) · [What agents say to each other](what-agents-say-to-each-other.md) · [How different are the agents?](how-different-are-the-agents.md) · [Genotypes, families and the common ancestor](genotype.md) · [Healthy and rich](healthy-and-rich.md) · [The diversity score](diversity-score.md) · [Every world side by side](every-world-side-by-side.md)
<!-- /turns -->
