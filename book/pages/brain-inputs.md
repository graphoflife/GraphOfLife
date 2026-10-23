# What a brain sees

Every time an agent decides anything, its brain **looks** at a list of
**candidates**: the agent itself first, then each of its neighbours in
increasing order of id. For each candidate the brain gets one column of
**154 numbers** (in the baseline), and it computes one column of outputs
from it ([What a brain says](brain-outputs.md)). The brain is the same for
every column; only the inputs differ.

This note lists the 154 inputs in the order the program builds them, for an
agent *u* looking at a candidate *v*. Rows are counted from 0, as in the
code.

## The inputs

| rows | how many | what |
|---|---|---|
| 0 | 1 | 1 if *v* is *u* itself, 0 otherwise |
| 1 | 1 | ln(1 + τ(*u*)): the agent's own tokens |
| 2 | 1 | ln(1 + τ(*v*)): the candidate's tokens |
| 3 | 1 | ln(1 + deg(*u*)): the agent's own number of connections |
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

That is 1 + 4 + 24 + 120 + 5 = **154**. With *m* numbers per message
([`message_amount`](settings.md#message_amount)) and *r* random inputs
([`random_input_amount`](settings.md#random_input_amount)) it is
29 + 4*m* + *r*.

## Why logarithms?

ln(1 + *x*) makes a brain see **orders of magnitude**. The difference between
10 and 20 tokens (ln 21 − ln 11 = 0.65) is then much larger than the
difference between 1,000 and 1,010 (0.01), as it should be: doubling matters,
adding ten to a thousand does not. The "1 +" keeps 0 at 0.

## The six quantiles

To describe a whole neighbourhood in a fixed number of inputs, whatever its
size, the program sorts the values *x*₀ ≤ *x*₁ ≤ … ≤ *x*ₙ₋₁ over the
neighbours and takes six **quantiles** at *q* = 0, 0.2, 0.4, 0.6, 0.8 and 1:
the smallest value, four values in between, and the largest. Quantile *q* is
read at position *h* = (*n* − 1)·*q* of the sorted list, interpolating
linearly between the two values around it:

$$
Q(q) = x_{\lfloor h \rfloor} + \bigl(h - \lfloor h \rfloor\bigr)\,\bigl(x_{\lceil h \rceil} - x_{\lfloor h \rfloor}\bigr),
\qquad h = (n-1)\,q .
$$

An agent with no neighbours gets six zeros. See
[Median and quantiles](median-and-quantiles.md) for the same idea used in
measuring worlds.

**Example.** A neighbourhood with tokens 1, 3, 3, 10 and 40: the logarithms
ln(1 + τ) sorted are 0.693, 1.386, 1.386, 2.398, 3.714, and *n* = 5. At
*q* = 0.2, *h* = 0.8, so *Q* = 0.693 + 0.8·(1.386 − 0.693) = 1.248. At
*q* = 0.4, *h* = 1.6: *Q* = 1.386. At *q* = 0.6, *h* = 2.4: 1.386 + 0.4·(2.398 −
1.386) = 1.791. At *q* = 0.8, *h* = 3.2: 2.398 + 0.2·(3.714 − 2.398) = 2.661.
The six inputs are (0.693, 1.248, 1.386, 1.791, 2.661, 3.714).

## When the inputs are measured

At the **start of each phase**, before anyone acts: every agent's tokens and
degree, and every neighbourhood's quantiles, are measured once and then read
by everyone. So the order in which agents act does not change what they see.
The messages are those delivered last (see [Messages](messages.md)), and the
random numbers are drawn fresh for every look, column by column, from the
world's random stream.

## Why random inputs?

A brain is a fixed function: the same inputs give the same outputs. Two
neighbours who look alike to it would always be treated alike. Five random
inputs give it a way to break such symmetries, and since how strongly it
listens to them is set by its weights, how random an agent's behaviour is can
itself evolve.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
