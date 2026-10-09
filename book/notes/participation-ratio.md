# How many directions a cloud of points uses

A message has 30 numbers, so the messages of a world are points in a space
of 30 dimensions. They need not use all 30. If every message were the same
except for one number, the messages would vary along a single direction, and
the other 29 numbers would carry nothing. The **participation ratio** counts
how many directions the variety is really spread over.

## The recipe

1. **Covariance.** For the points *x*₁, …, *x*ₙ (each a list of *d* numbers),
   compute the *d* × *d* covariance matrix *C*. Its entry *Cᵢⱼ* is the
   covariance of number *i* with number *j* over the points.
2. **Principal directions.** The eigenvectors of *C* are the directions along
   which the cloud of points spreads, independently of each other (the
   principal components). The eigenvalue λₖ of each is the variance of the
   points along it. The eigenvalues add up to the total variance.
3. **Count them, weighted.**

$$
\text{PR} = \frac{\left(\sum_k \lambda_k\right)^2}{\sum_k \lambda_k^2} .
$$

## Reading it

- If the variance is spread equally over *m* directions and the rest have
  none, PR = *m*. For example, *m* eigenvalues of 1 give *m*²/*m* = *m*.
- If one direction holds almost everything, PR is close to 1.
- In between, PR is an **effective number** of directions, in the same sense
  as 2^H is an effective number of kinds ([Entropy and evenness](entropy-and-evenness.md)):
  how many equally important directions would spread the cloud as unevenly
  as it is spread.

So PR runs from 1 to *d*. For random points with independent numbers of
equal variance it is close to *d*, a little less because of sampling noise.
For 30 independent numbers measured on a few thousand points, it comes out
near 28 or 29.

## Where it is used

[Chapter 19](../chapters/19-what-agents-say-to-each-other.md) uses it on the
messages agents write, as the number of the 30 message numbers a world's
messages actually use. The measure comes from physics, where it counts over
how many sites a wave is spread (Bell and Dean 1970), and is used in
neuroscience for the dimension of neural activity (Gao et al. 2017).

## References

- Bell, R. J. and Dean, P. (1970). Atomic vibrations in vitreous silica.
  *Discussions of the Faraday Society* 50, 55–61.
- Gao, P., Trautmann, E., Yu, B., Santhanam, G., Ryu, S., Shenoy, K. and
  Ganguli, S. (2017). A theory of multineuronal dimensionality, dynamics and
  measurement. *bioRxiv* 214262.
