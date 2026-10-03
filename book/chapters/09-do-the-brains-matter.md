# Do the brains matter at all?

## In short

*Waiting for the runs.*

## Thesis

```thesis E07
```

Everything Chapters 3 to 8 found is a fact about the baseline, with nothing
to hold it against (Meta I). This chapter takes the brains away in two ways.
If the worlds without them look like the baseline, the rules alone make the
world, and the brains are decoration. If they do not, the difference is what
the brains — and their evolution — do.

## Method

```experiment E07
```

Three conditions of thirty seeds each, at 10,000 tokens and for 3,000
iterations, as in Chapter 3. Each changes one setting of the baseline B1.

- **baseline** — the thirty runs of Chapter 3, reused.
- **brains never change** — `mutation_sparsity` 0. A brain still "changes"
  with the same chance of one in five, at birth and after every game, and
  gets a new name when it does — but not one of its weights moves. So the only
  brains a world ever has are its founders', copied; the names still split
  exactly as often as in the baseline, which keeps the family tree of
  Chapter 8 comparable. Once one founder's descendants fill a world, every
  agent in it carries the same brain, and which name spreads from then on is
  pure chance.
- **decisions by chance** — `random_decisions`. Agents never read their
  inputs: every number a brain would have given is drawn at random instead,
  and every decision follows from that.

Two things were seen before this plan was written, in pilots run outside the
lab to size it. Every one of ten worlds deciding by chance died within three
iterations: random stakes leave connections unused, unused connections are
cut, and within two games the world falls apart into pieces too small to
live. The thirty runs of that condition record this properly; it is not the
thesis. And two worlds whose brains never change came out as different as
the thesis says — which is why it says it.

What is compared, from iteration 500 on (the settled life of a run, Meta I):

- **how alike the worlds are** — for the number of agents and the births,
  the spread across the surviving worlds of each condition (the standard
  deviation over the mean), and the share of all variation in a world's
  number of agents that lies between the worlds rather than within each one
  over time, as in Chapter 4;
- **everything Chapters 3 to 8 measured** — agents, connections, births and
  deaths, inequality, the game, the shape of the graph, genotypes and
  families — each condition against the baseline;
- **lineages**, as in Chapter 8. In a world whose brains never change, once
  one founder's descendants fill it, no lineage is better than another, so
  how often the common ancestor of all the living moves forward there is the
  rate of chance — the null that Chapter 8's sweeps were missing. It is
  reported without a thesis.

The lab's estimate for this experiment is high: it cannot know that the
worlds deciding by chance die in seconds, and counts them as full runs. The
work that is really left is the thirty runs whose brains never change, about
seven hours on four workers.

## Results

*Waiting for the runs.*

## Conclusion

*Waiting for the runs.*
