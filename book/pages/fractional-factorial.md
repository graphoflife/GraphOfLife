# Many switches in few worlds: fractional factorial designs

A world has many switches. To learn what each does, the plain way is to
throw one at a time: B2, and B2 with the switch. That says what the switch
does *in B2*, but not whether it does the same once other switches are
thrown too. Throwing every combination answers that, at a price: eight
switches have 2⁸ = 256 combinations. A **fractional factorial design** plays
a sixteenth of them, chosen so that each switch's effect can still be read.

## Sixteen worlds for eight switches

Write each switch as −1 (off) or +1 (on). Take four of them — A, B, C and D —
in all sixteen combinations, and set the other four by multiplying:

$$E = BCD, \quad F = ACD, \quad G = ABC, \quad H = ABD.$$

In the world where A, B and C are on and D off, G = (+1)(+1)(+1) = +1 is on,
H = (+1)(+1)(−1) = −1 off, and so on. Every switch is then on in eight of the
sixteen worlds and off in the other eight, and every *pair* of switches is
on–on, on–off, off–on and off–off in four worlds each: the design is
balanced.

## Reading an effect

A switch's **main effect** on a measure *y* is the mean of *y* over the eight
worlds where it is on, minus the mean over the eight where it is off:

$$\text{effect}(A) = \bar y_{A=+1} - \bar y_{A=-1}.$$

Because the design is balanced, every other switch is on in half of each
eight: their effects cancel out of the difference, *if the effects add up*.

## What cannot be told apart

Sixteen worlds give fifteen independent differences, and eight switches,
twenty-eight pairs and more have effects of their own: some must share a
difference. Two switches **act together** when the effect of one depends on
whether the other is on — the effect of the pair AB is half the difference
between A's effect with B on and A's effect with B off, read as the mean of *y*
where A·B = +1 minus the mean where A·B = −1. In this design the column A·B
is the same as C·G, D·H and E·F (multiply out the definitions: C·G = C·ABC =
AB, since C·C = 1). So one difference measures the four pairs together, a
**chain** of aliases:

| chain | the pairs it adds up |
|---|---|
| AB | AB + CG + DH + EF |
| AC | AC + BG + DF + EH |
| AD | AD + BH + CF + EG |
| BC | BC + AG + DE + FH |
| BD | BD + AH + CE + FG |
| CD | CD + AF + BE + GH |
| ABCD | AE + BF + CH + DG |

Eight main effects and these seven chains are the fifteen differences. A main
effect shares its column only with triples of switches (A = BCG = …): the
design is of **resolution IV**, so no main effect is confused with a pair. If
a chain is large, some pair acts together; which one, the sixteen worlds
cannot say.

## How large is large?

Each world is played once, with one seed. How much a difference of means
between two groups of eight moves by chance alone can be read from worlds
that differ *only* in their seed: if one world's measure varies with the
seed by a standard deviation σ, a difference of two means of eight varies by

$$\sigma \sqrt{1/8 + 1/8} = \sigma / 2.$$

An effect several times σ/2 is not the seed. The chains give a second check:
if every pair added nothing, the seven chains would scatter like chance too.

[Chapter 86](eight-switches-at-once.md), *Eight switches at once*, uses this design.

<!-- turns -->
---

← [Effective numbers: how many kinds there really are](effective-number.md) · [Methods and statistics](../README.md#methods-and-statistics)
<!-- /turns -->
