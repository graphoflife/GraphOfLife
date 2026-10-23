# What a brain sees

## In short

Every time an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") decides anything, its [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") looks at its [candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides."): the
agent itself first, then each of its [neighbours](the-world.md#the-network "An agent joined to another by a connection.") in increasing order of id.
For each candidate it gets one column of numbers, **154** in B1 and 66 in B2,
and computes one column of outputs from it
([What a brain says](brain-outputs.md)). A column holds whether the
candidate is the agent itself, how rich and how well connected the two are, a
summary of each one's neighbourhood, four messages and five random numbers.
The brain is the same for every column; only the inputs differ.

## One form per candidate

Think of each column as a form that the program fills in about one
candidate, and the brain reads. Every form has the same boxes, in the same
order. Only what is written in them changes from one candidate to the next.
What is not on the form, the brain cannot know.

This page lists the boxes: the inputs, in the order the program builds them,
for an agent *u* looking at a candidate *v*. Rows are counted from 0, as in
the code.

## The inputs

| rows | how many | what |
|---|---|---|
| 0 | 1 | 1 if *v* is *u* itself, 0 otherwise |
| 1 | 1 | ln(1 + τ(*u*)): the agent's own [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") |
| 2 | 1 | ln(1 + τ(*v*)): the candidate's tokens |
| 3 | 1 | ln(1 + deg(*u*)): the agent's own number of [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") |
| 4 | 1 | ln(1 + deg(*v*)): the candidate's |
| 5–10 | 6 | six quantiles of ln(1 + τ) over *u*'s neighbours |
| 11–16 | 6 | six quantiles of ln(1 + τ) over *v*'s neighbours |
| 17–22 | 6 | six quantiles of ln(1 + deg) over *u*'s neighbours |
| 23–28 | 6 | six quantiles of ln(1 + deg) over *v*'s neighbours |
| 29–58 | 30 | the message *u* wrote to itself |
| 59–88 | 30 | the message *u* wrote to *v* |
| 89–118 | 30 | the message *v* wrote to *u* |
| 119–148 | 30 | the message *v* wrote to itself |
| 149–153 | 5 | random numbers, each uniform between −2 and 2 |

Here τ is a number of tokens, deg a number of connections (the [degree](degree.md "The number of an agent's connections: how many neighbours it has.")), and
ln the natural logarithm. A uniform random number between −2 and 2 is one
for which every value in that range is equally likely.

The table is B1's, and it adds up to 1 + 4 + 24 + 120 + 5 = **154**. With
*m* numbers per message ([`message_amount`](settings.md#message_amount)) and
*r* random inputs ([`random_input_amount`](settings.md#random_input_amount))
it is 29 + 4*m* + *r*. B2's messages hold 8 numbers, so its brains read
29 + 32 + 5 = 66 inputs: the messages fill rows 29–60, and the random numbers
rows 61–65. Some experiments' rules add rows of their own. With gifts, for
example, a second flag after row 0 says whether the connection to the
candidate would lapse unless something crosses it. Every later row then moves
down by one ([`allow_gifting`](settings.md#allow_gifting)).

## Why logarithms?

ln(1 + *x*) makes a brain see **orders of magnitude**. The difference between
10 and 20 tokens (ln 21 − ln 11 = 0.65) is then much larger than the
difference between 1,000 and 1,010 (0.01), as it should be: doubling matters,
adding ten to a thousand does not. The "1 +" keeps 0 at 0.

## The six quantiles

A neighbourhood may have one member or thousands, but the form has a fixed
number of boxes. So the program sums a neighbourhood up in six numbers. It
sorts the values over the neighbours, *x*₀ ≤ *x*₁ ≤ … ≤ *x*ₙ₋₁, and takes six
**quantiles** at *q* = 0, 0.2, 0.4, 0.6, 0.8 and 1: the smallest value, four
values in between, and the largest. Quantile *q* is read at position
*h* = (*n* − 1)·*q* of the sorted list. When *h* falls between two positions,
the value is read off the straight line between the two values around it:

$$
Q(q) = x_{\lfloor h \rfloor} + \bigl(h - \lfloor h \rfloor\bigr)\,\bigl(x_{\lceil h \rceil} - x_{\lfloor h \rfloor}\bigr),
\qquad h = (n-1)\,q .
$$

Here ⌊*h*⌋ is *h* rounded down and ⌈*h*⌉ is *h* rounded up. An agent with no
neighbours gets six zeros. [Median and quantiles](median-and-quantiles.md)
uses the same idea to measure worlds.

**Example.** A neighbourhood holds 1, 3, 3, 10 and 40 tokens, so *n* = 5. The
logarithms ln(1 + τ), sorted, are 0.693, 1.386, 1.386, 2.398 and 3.714.

- At *q* = 0 the quantile is the smallest value, 0.693.
- At *q* = 0.2, *h* = 0.8: eight tenths of the way from the first value to
  the second, *Q* = 0.693 + 0.8·(1.386 − 0.693) = 1.248.
- At *q* = 0.4, *h* = 1.6: between two equal values, *Q* = 1.386.
- At *q* = 0.6, *h* = 2.4: *Q* = 1.386 + 0.4·(2.398 − 1.386) = 1.791.
- At *q* = 0.8, *h* = 3.2: *Q* = 2.398 + 0.2·(3.714 − 2.398) = 2.661.
- At *q* = 1 it is the largest value, 3.714.

The six inputs are (0.693, 1.248, 1.386, 1.791, 2.661, 3.714).

## When the inputs are measured

At the **start of each phase**, before anyone acts. Every agent's tokens and
degree, and every neighbourhood's quantiles, are measured once and then read
by everyone. So the order in which agents act does not change what they see.
The messages are those delivered last ([Messages](messages.md)). The random
numbers are drawn fresh for every look, column by column, from the world's
random stream ([Seeds and the random stream](random-numbers.md)).

## Why random inputs?

A brain is a fixed function: the same inputs give the same outputs. Two
neighbours who look alike to it would always be treated alike. Five random
inputs give it a way to break such ties. How strongly it listens to them is
set by its weights, so how random an agent's behaviour is can itself evolve.

In B1 they move more stakes than wealth does. In the 30 B1 worlds of 100,000
tokens,
drawing the five random numbers of every column again changes the stakes of
15% of the agents. Doubling every neighbour's tokens, as the agent sees them,
changes only 1.5% ([What the brains are like](what-the-brains-are-like.md)).

## Where it is used

- [What the brains are like](what-the-brains-are-like.md): an evolved B1
  brain hears every group of inputs equally loudly. Each input is read by a
  column of 50 first-layer weights, and that column's length is 0.67 for
  every group, the random numbers included (0.57 in a [founder](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")).
- [How a signal fades through layers](vanishing-signal.md): seeing is not
  using. A change in these inputs reaches a B1 founder's outputs at a gain of
  only 0.0006, and a B2 founder's at about 0.2
  ([B1 and B2: the two baselines](b1-and-b2.md)).
- [What a brain can compute](what-a-brain-can-compute.md): what a B2 brain
  can and cannot work out from its 66 inputs.
- [Do agents cooperate?](do-agents-cooperate.md): no input names a [genotype](genotype.md "The identity of a brain: a number that a brain keeps until it changes, and that records whose copy it was."),
  so a brain cannot tell kin from stranger, except through messages.

## Where to go next

- [What a brain says](brain-outputs.md): the outputs, row by row.
- [Messages](messages.md): who writes the four messages, and when they
  arrive.
- [Median and quantiles](median-and-quantiles.md): quantiles, as the
  notebook uses them to measure worlds.

<!-- turns -->
---

← [Splitting tokens into whole numbers](largest-remainder.md) · [How the simulation works](../README.md#how-the-simulation-works) · [What a brain says](brain-outputs.md) →

**Also about:** [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are).

> [!info]- Pages that link here (11)
> [The brain](the-brain.md) · [Every setting](settings.md) · [Seeds and the random stream](random-numbers.md) · [What a brain says](brain-outputs.md) · [Messages](messages.md) · [Physical inspiration](physical-inspiration.md) · [What the brains are like](what-the-brains-are-like.md) · [How different are the agents?](how-different-are-the-agents.md) · [What a brain can compute](what-a-brain-can-compute.md) · [Do agents cooperate?](do-agents-cooperate.md) · [Median and quantiles](median-and-quantiles.md)
<!-- /turns -->
