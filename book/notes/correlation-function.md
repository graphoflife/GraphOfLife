# How alike at a distance

In a magnet, a spin pointing up makes its neighbours more likely to point up,
and their neighbours a little more likely, and so on, fading with distance.
How fast it fades is one of the most telling facts about a physical system.
The tool that measures it is the **correlation function** (Chaikin and
Lubensky 1995). On a network, distance is counted in steps, and the same
tool applies.

## The definition

Give every agent *u* a value *x*(*u*), such as its tokens, its age or its
connections, standardised within its world: minus the mean, divided by the
standard deviation. For every pair of agents exactly *r* steps apart, take
the two values. The correlation function is the correlation
([Correlation](correlation.md)) over all such pairs:

$$
C(r) = \operatorname{corr}\bigl(x(u), x(v)\bigr) \quad \text{over pairs with } d(u, v) = r .
$$

- *C*(*r*) > 0: agents *r* steps apart tend to be alike;
- *C*(*r*) < 0: unlike, such as a rich agent beside poor ones;
- *C*(*r*) = 0: no relation at that distance.

*C*(1), over all connected pairs, is the assortativity of the value; for
the connections themselves, it is the assortativity of the network
([Assortativity](assortativity.md)).

Pairs at distance *r* are found by breadth-first search from agents drawn at
random. Standardising each moment's values on their own keeps moments with
different means from looking alike when they are pooled.

## How fast it fades

In most systems *C*(*r*) falls off exponentially, *C*(*r*) ∝ e^(−*r*/ξ).
The **correlation length** ξ is the distance over which likeness fades by a
factor e. A system has structure on scales up to ξ and looks uniform beyond
it.

At a **critical point**, such as a magnet exactly at its Curie temperature,
ξ grows without bound, and *C*(*r*) falls only as a power of *r*. Likeness
then reaches across the whole system. That is the sense in which critical
systems have "no scale": there is no distance beyond which they look
uniform.

## A trap: place versus closeness

On a network, a correlation at distance *r* can arise without any agent
influencing another. Suppose tokens follow connections — hubs rich, leaves
poor — and hubs sit among leaves. Then neighbours have *unlike* tokens, and
two leaves of one hub (two steps apart) have *like* tokens, purely because
of where hubs and leaves sit.

To see how much of *C*(*r*) is just place, shuffle the values among agents
that have about as many connections, and measure again. The shuffle keeps
how values follow connections and destroys everything else. What survives
it was there because of place; what disappears was closeness.

## Where it is used

[Chapter 22](../chapters/22-like-next-to-like.md) measures *C*(*r*) for
tokens, age and connections, and the share of kin at each distance.

## References

- Chaikin, P. M. and Lubensky, T. C. (1995). *Principles of Condensed Matter
  Physics*. Cambridge University Press.
