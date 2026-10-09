# Scaling relations between agents' properties

How does one property of an agent grow with another — its tokens with its
connections, say? The viewer fits, on every heavy frame, four **scaling
relations** of the form

$$
y \approx a\, x^{b} ,
$$

by least squares on the logarithms of all agents of the frame:
ln *y* = ln *a* + *b* ln *x* ([Fitting a straight line](least-squares.md)). The
**exponent** *b* says how *y* grows with *x*; R² says how tightly the agents
follow the line. Agents with *x* = 0 or *y* = 0 are left out, since a
logarithm of 0 does not exist.

## The four

| field | *y* | *x* | what an exponent would mean |
|---|---|---|---|
| `tokensVsDegree` | tokens | connections | 1: tokens in proportion to connections; 0: connections buy nothing |
| `trianglesVsDegree` | triangles the agent is a corner of | connections | 2: neighbourhoods wired at random (pairs of neighbours grow as k²) |
| `clusteringVsDegree` | clustering coefficient (agents with ≥ 2 connections) | connections | −1: a hierarchical network — dense small groups inside sparser large ones (Ravasz and Barabási 2003) |
| `changeVsTokens` | the size \|Δ\| of the change in tokens over the phase | tokens after the phase | 1: everyone risks the same fraction of what they hold (Gibrat's law) |

Each is reported with its R² (`tokensVsDegreeR2`, …).

## Reading an exponent

- For **tokens and connections**: with *b* = 0.6, an agent with ten times the
  connections holds 10^0.6 ≈ 4 times the tokens — more, but less than in
  proportion.
- For **clustering**: the clustering coefficient of an agent with *k*
  neighbours is its triangles divided by *k*(*k* − 1)/2, so `clusteringVsDegree`
  ≈ `trianglesVsDegree` − 2 when the triangles follow a power law.
- For **changes**: Gibrat's law of proportionate growth (Gibrat 1931) says the
  change of a firm's size is proportional to its size; *b* < 1 means the rich
  change by a smaller fraction than the poor.

## Two cautions

- **R² is about scatter, not about power laws.** Many curved relations look
  straight over a decade or two of logarithmic axes; read R² as "how much of
  the spread does this line account for", not as proof of a law.
- **The fit is over agents, and most agents are small.** Thousands of agents
  with one or two connections dominate the least-squares sum; a handful of
  hubs barely moves it. [Chapter 30](../chapters/30-how-properties-scale-together.md)
  shows the agents behind each fit.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
