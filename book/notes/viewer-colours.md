# What the viewer can colour by

The viewer draws every agent as a dot and every connection as a line, and can
colour and size both by a measured quantity. This note says what each
quantity is, and how a number becomes a colour — so that a picture can be
read as precisely as a chart. [Chapter 26](../chapters/26-one-world-many-colours.md)
shows one world coloured six ways.

## From a number to a colour

For the frame on screen, and for the quantity chosen:

1. **Values.** Every agent (or connection) gets its value *v*.
2. **Logarithm, if chosen.** *v* becomes ln(1 + *v*) — or, for a quantity
   that can be negative, sign(*v*)·ln(1 + |*v*|), which squeezes large
   magnitudes but keeps the sign.
3. **Range.** For an amount, the range runs from the frame's smallest value
   to its largest. For a **signed** quantity (a change, a curvature), it runs
   from −*m* to +*m*, *m* the largest magnitude, so that the middle of the
   colour map always means zero.
4. **Position.** Each value is placed in the range as a number between 0
   and 1, and the **colour map** gives its colour. An unknown value (the age
   in a run recorded before ages were kept) is put in the middle.

The same colour therefore means different values in different frames: the
range follows the frame. The legend in the corner gives the range.

**Colour maps.** *viridis*, *plasma*, *magma*, *inferno*, *cividis* and
*ember* run from dark to light and suit amounts; *coolwarm* and *spectral*
diverge from a neutral middle and suit signed quantities; *turbo* is a
rainbow; *grayscale* is black to white.

## Agents

| choice | the value of agent *u* |
|---|---|
| Tokens | its tokens |
| Degree | its number of connections ([Degree](degree.md)) |
| Token change (signed) | its tokens now minus its tokens before the phase; a newborn counts from 0 |
| Token change (magnitude) | the size of that change |
| Token curvature (signed) | Σ over its neighbours of (their tokens − its tokens) ([Token curvature](token-curvature.md)) |
| Token curvature (before phase) | the same, on the network as it stood before the phase; newborns have none |
| Loops through it | fundamental cycles of a breadth-first spanning tree through it ([Loops](loops.md)) |
| Triangles | the triangles it is a corner of ([Clustering](clustering.md)) |
| Share of total tokens | its tokens divided by all tokens |
| Brain id | its genotype ([Genotypes](genotype.md)) |
| Parent brain id | its genotype's parent genotype |
| Age | iterations it has lived ([Age and lifetime](age-and-lifetime.md)) |
| Node id | its id, which is its birth order |

A word on **brain id**: a genotype is a number, and the colour map turns
numbers that are close into colours that are close. Close genotype numbers
were *born* at about the same time — they need not be related. To see kinship,
colour by family instead, as Chapter 26 does.

## Connections

| choice | the value of the connection between *u* and *v* |
|---|---|
| Average / weaker / stronger endpoint tokens | the mean, smaller, larger of their tokens |
| Endpoint token gap | \|tokens of *u* − tokens of *v*\| |
| Average / weaker / stronger endpoint degree | the mean, smaller, larger of their degrees |
| Average endpoint curvature (signed) | the mean of their token curvatures |
| Token flow | tokens staked across it in the game, both ways together ([Token flow](token-flow.md)) |
| Loops through it | fundamental cycles through it, as for agents |
| Triangles | triangles it is a side of |
| Bridge | 1 if it lies on no loop — cutting it splits the network ([Bridges](bridges.md)) — else 0 |

## The built-in looks

| look | dots coloured by | dots sized by | lines coloured by |
|---|---|---|---|
| default | tokens (log), *coolwarm* | tokens (log) | one colour |
| wealth | tokens (log), *inferno* | tokens | one colour |
| lineage | brain id, *turbo* | tokens (log) | the colour of their dots |
| structure | degree (log), *cividis* | degree | average endpoint degree (log) |
| flow | token curvature (log), *coolwarm* | size of the token change (log) | token flow (log) |
| minimal | one colour | one size | one colour |

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
