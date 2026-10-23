# What would count as open-ended evolution

No run can show that evolution goes on for ever. A run is finite, and the
budget is finite: a world of 10,000 tokens runs about 100,000 iterations
in a day and a half, and a world of 100,000 tokens a tenth as far in two
days. What a run can show is that, as far as it went, nothing stopped:
- later brains beat their whole past;
- they keep getting better;
- they keep doing new things;
- the world stays one in which much can happen.

This note writes the test down before any of the long runs it will judge
exists. A thesis written after its results can be bent to fit them; one
written before cannot.

## The test

A configuration of rules shows open-ended evolution, as far as a run can,
when **at least two of three new worlds** pass all four parts below.
- New means seeds that played no part in choosing the configuration: a
  configuration picked as the best of many is lucky as well as good, and
  its own seed says nothing about that.
- Every world played counts, the ones that fail included.

1. **It beats its whole past.** At every 10,000 iterations, the brains of
   that moment beat the best mixture of all earlier eras in lifetime
   contests: 100 iterations, played on both worlds, from both sides.
   - The best mixture is the Nash mixture of the eras' contest table: the
     blend of past brains that does best against everything.
   - Beating the last era is not enough. In a game like this one every
     strategy has an answer, so a chain of "beats the one before" can go
     round in a circle for ever without getting anywhere, as rock, paper
     and scissors do.
   - The margin must be larger than the spread of an era played against
     itself.
2. **It keeps rising.** The eras' ratings, read against the best mixture
   of the others (Nash averaging, Balduzzi et al., 2018), rise over the
   second half of the run. The 95% interval of the slope must lie above
   zero, bootstrapped over contests.
3. **It keeps doing new things.** New kinds of behaviour appear faster than
   they would by mutation alone, and the old ones do not simply come back.
   - Kinds of behaviour means two things: kinds of staking
     ([Chapter 70](how-different-are-the-agents.md)), and
     the signatures of the strategies written by hand. A signature is how
     a strategy answers a fixed set of situations
     ([Chapter 94](what-a-brain-can-compute.md)).
   - "By mutation alone" is the same brains changed only by the random
     mutations their lineages collected over the same time, with no
     selection.
   - A world whose kinds come and go in a cycle does not pass.
4. **It stays a world in which much can happen**
   ([Meta X](healthy-and-rich.md)):
   - it never falls to 20 agents;
   - it meets every mark of the health bar at every 10,000 iterations;
   - its agents are more diverse than those of B2's median world;
   - its tokens are no more even than B2's: a
     [Gini coefficient](gini-coefficient.md) no lower than the least of
     B2's worlds read at the same iteration.

## What it does not ask

- **Speed.** Evolution may take as long as it likes, so long as within the
  run it does not stop.
- **Typical worlds.** "Two of three new worlds" asks whether a
  configuration *can* keep evolving, not whether it usually does. A
  configuration that does so in a few worlds is a find.
- **A particular kind of novelty.** Larger groups, deeper organisation and
  new signals all count, so long as they beat the past and were not seen
  before.

## Why these four parts

Each part closes a way to be fooled:
- **Part 1** closes the circle. Later brains beat earlier ones in every
  kind of world the book has played this long, but by less and less. A
  game that only goes round would pass a test of the last step.
- **Part 2** closes the plateau that still wins by a hair.
- **Part 3** closes the arms race that only swaps two answers back and
  forth.
- **Part 4** closes worlds that evolve only because they are poor in
  everything else, such as a few hundred agents holding everything evenly.

**Related notes:** [Chapter 46 · How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md) ·
[Chapter 94 · What a brain can compute](what-a-brain-can-compute.md) ·
[Meta X · Healthy and rich](healthy-and-rich.md)

<!-- turns -->
---

← [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Does evolution keep going?](../README.md#does-evolution-keep-going) · [Exploitability](exploitability.md) →
<!-- /turns -->
