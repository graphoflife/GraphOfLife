# Preferential attachment and its kernel

In many growing networks the well connected gain new connections faster
than the poorly connected: popular web pages are linked to more, much-cited
papers are cited more. If the rate at which an agent gains connections is
proportional to the connections it has, the network grows by **preferential
attachment** (Barabási and Albert 1999; the same idea as Price's cumulative
advantage, Price 1976, and Simon's model of word frequencies, Simon 1955).

## The kernel

The general form is the **attachment kernel** Π(*k*): the expected number of
new connections an agent with *k* connections gains in one step. It is
measured directly. Take all agents at the start of a step, sort them by *k*,
count the connections each class gains during the step, and divide by the
number of agents in the class (Newman 2001; Jeong, Néda and Barabási 2003).
On logarithmic axes, a kernel Π(*k*) ∝ *k*^α is a straight line of slope α:

- α = 0: every agent gains alike, whatever it has. The degrees stay narrow,
  as in a random network.
- α = 1: **linear** preferential attachment. A growing network gets a
  power-law tail of degrees, with exponent 3 if nothing is ever lost
  (Barabási and Albert 1999).
- α < 1 (sublinear): the tail is cut off and bends down, close to a
  stretched exponential.
- α > 1 (superlinear): one agent ends up joined to nearly everyone (Krapivsky,
  Redner and Leyvraz 2000).

## Losses too

A network that also loses connections has a **loss kernel** as well, the
expected connections lost per agent with *k*. If both gains and losses are
proportional to *k*, every agent's connections change by a random factor
from step to step, multiplied rather than added. This is Gibrat's law of
proportionate growth, applied to connections. Whether a heavy tail forms
then depends on how gains and losses balance, and on what stops the poorest
from vanishing.

## Where it is used

[Chapter 21](../chapters/21-how-the-network-grows.md) measures both kernels
in the baseline worlds. [Chapter 29](../chapters/29-power-laws-real-and-apparent.md)
had proposed preferential attachment, reached by copying, as the source of
the heavy tail of the connections.

## References

- Barabási, A.-L. and Albert, R. (1999). Emergence of scaling in random
  networks. *Science* 286, 509–512.
- Jeong, H., Néda, Z. and Barabási, A.-L. (2003). Measuring preferential
  attachment in evolving networks. *Europhysics Letters* 61, 567–572.
- Krapivsky, P. L., Redner, S. and Leyvraz, F. (2000). Connectivity of
  growing random networks. *Physical Review Letters* 85, 4629–4632.
- Newman, M. E. J. (2001). Clustering and preferential attachment in growing
  networks. *Physical Review E* 64, 025102.
- Price, D. de S. (1976). A general theory of bibliometric and other
  cumulative advantage processes. *Journal of the American Society for
  Information Science* 27, 292–306.
- Simon, H. A. (1955). On a class of skew distribution functions.
  *Biometrika* 42, 425–440.
