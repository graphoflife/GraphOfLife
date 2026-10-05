# How does a world's size follow its tokens?

## In short

*Waiting for the runs.*

## Thesis

```thesis E08
```

Every chapter so far looked at worlds of 10,000 tokens. Whether what they
found holds at other sizes rests first on this question. If agents and
connections grow in proportion to the tokens, a bigger world is more of the
same, and the size of a world can be chosen for what an experiment can
afford. If they do not, size changes what a world is, and every result
belongs to the size it was found at.

## Method

```experiment E08
```

Worlds of the baseline B1 from 800 to 409,600 tokens, each size double the
one before — ten sizes, three seeds each, 600 iterations each. A world starts
with one founder for every hundred tokens, so the founders go from 8 to
4,096.

**What is measured.** For every run, the mean number of agents alive after
the game, and of connections, over iterations 500 to 599 — well past the
youth of Chapter 3. The same means over iterations 400 to 499 say whether a
world had settled: a world still growing or shrinking would show it as a
difference between the two.

**How it is read.** On logarithmic axes, on both sides, a power law —
agents = a × tokens^b, some number a times the tokens to the power b — is a
straight line, and its slope is the exponent b. An exponent of 1 means in proportion; below 1, bigger worlds hold
fewer agents per token; above 1, more. The line is fitted to the baseline
worlds of 3,200 tokens and more, every run a point. The interval of its slope
comes from resampling the runs of each size, and a second fit with a squared
term says whether the points bend away from a straight line. The exponent
between each size and the next shows where along the range any bend lies.
Connections per agent are read the same way: in proportion means a slope
of 0.

**The small end.** Two rules of the baseline limit it, both met in pilots
run to plan this chapter.

- *No world below 600 tokens.* The founders' starting ring needs more
  founders than neighbours, at least six, so the baseline cannot build a
  world of 500 tokens or fewer. The doubling starts at 800, not at 50.
- *A run stops at 20 agents or fewer.* Worlds of 800 and 1,600 tokens start
  with 8 and 16 founders, below that line. In the pilots, every 800-token
  world and two of three 1,600-token worlds were stopped as extinct after
  their first iteration; worlds of 3,200 tokens lived, with 282 and 481
  agents over iterations 500 to 599. The baseline worlds of 800 and 1,600
  tokens are run anyway, as the baseline does it. Beside them, a second
  condition — *stopped only when empty*, `extinction_threshold` 0 — lets
  worlds of 800 to 6,400 tokens go on as long as anyone is alive. A world
  that never falls to twenty agents is then exactly the baseline's world of
  the same seed, so where both run they must agree, and below that it shows
  how small a world can live. It has no thesis.

**The starting ring.** Each founder starts joined to its nearest founders,
four of them in worlds below 60,000 tokens and more in bigger ones — a
hundredth of the founders, so 40 at 409,600 tokens. That is the baseline,
and it is kept. The first games cut every connection that carries no
tokens, so by iteration 500 a world should have long forgotten how it
started.

**One worker.** The worlds run one after another on a single worker,
whatever else the lab is doing: the biggest needs several gigabytes of
memory. The lab estimates about two days in all, almost half of it the three
worlds of 409,600 tokens — an estimate that itself assumes what this chapter
tests, that a world's size grows in proportion to its tokens.

## Results

*Waiting for the runs.*

## Conclusion

*Waiting for the runs.*
