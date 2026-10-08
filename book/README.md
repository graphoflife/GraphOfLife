# Graph of Life — the research book

This book is a research programme, written down while it is carried out. It
asks what the Graph of Life algorithm actually does — and, later, whether it
can be made to keep producing new things without end, which is what
*open-ended evolution* means. [Chapter 1](chapters/01-what-this-book-is-about.md)
says why, and where to start.

It is written to be **read on its own**. Part I explains the algorithm
completely, rule by rule. Every measurement, statistic and method the book
uses has a short note of its own that defines it exactly and works an example
by hand. Every figure says what it shows and, folded under it, how to make it
again from the runs. Nothing needs the program to be understood — and
everything can be checked with it.

## How to read it

- **In order**, from [Chapter 1](chapters/01-what-this-book-is-about.md): each
  chapter ends with a link to the next.
- **By question**: every chapter starts with the questions it asks.
- **By idea**: the notes below are short and linked from wherever they are
  used; each can be read on its own.
- **In Obsidian**: open the folder `book/` as a vault. Every link, figure
  and formula works there, and the graph view shows how the chapters and
  notes hang together. The same files read in the Book tab of the Graph of
  Life app and on GitHub.

## Contents

<!-- contents -->
**Part I · How the algorithm works**

- [Chapter 1 · What this book is about](chapters/01-what-this-book-is-about.md)
- [Chapter 2 · The world: tokens, agents and a network](chapters/02-the-world.md)
- [Chapter 3 · One iteration, step by step](chapters/03-one-iteration.md)
- [Chapter 4 · The brain](chapters/04-the-brain.md)
- [Chapter 5 · How worlds are measured](chapters/05-how-worlds-are-measured.md)
- [Chapter 6 · The ideas this builds on](chapters/06-the-ideas-this-builds-on.md)
- [Chapter 7 · Is a run reproducible?](chapters/07-is-a-run-reproducible.md)

**Part II · The baseline world**

- [Chapter 8 · Thirty worlds](chapters/08-thirty-worlds.md)
- [Chapter 9 · A world's life](chapters/09-a-worlds-life.md)
- [Chapter 10 · The first hundred iterations](chapters/10-the-first-hundred-iterations.md)
- [Chapter 11 · Births, deaths and ages](chapters/11-births-deaths-and-ages.md)
- [Chapter 12 · How much does the seed decide?](chapters/12-how-much-does-the-seed-decide.md)
- [Chapter 13 · Where do the tokens go?](chapters/13-where-do-the-tokens-go.md)
- [Chapter 14 · How the game is played](chapters/14-how-the-game-is-played.md)
- [Chapter 15 · What shape does the network take?](chapters/15-what-shape-does-the-network-take.md)
- [Chapter 16 · Genotypes and lineages](chapters/16-genotypes-and-lineages.md)
- [Chapter 17 · Meta I · What the baseline world is](chapters/17-meta-1.md)

**Part III · The baseline world, measured every way**

- [Chapter 18 · How even is a world?](chapters/18-how-even-is-a-world.md)
- [Chapter 19 · Gains and losses](chapters/19-gains-and-losses.md)
- [Chapter 20 · Where the tokens flow](chapters/20-where-the-tokens-flow.md)
- [Chapter 21 · How agents have children](chapters/21-how-agents-have-children.md)
- [Chapter 22 · Power laws, real and apparent](chapters/22-power-laws-real-and-apparent.md)
- [Chapter 23 · How properties scale together](chapters/23-how-properties-scale-together.md)
- [Chapter 24 · The geometry of a world](chapters/24-the-geometry-of-a-world.md)
- [Chapter 25 · How a world breaks](chapters/25-how-a-world-breaks.md)
- [Chapter 26 · One world, many colours](chapters/26-one-world-many-colours.md)
- [Chapter 27 · Do agents cooperate?](chapters/27-do-agents-cooperate.md)
- [Chapter 28 · Questioning the mechanics](chapters/28-questioning-the-mechanics.md)
- [Chapter 29 · Meta II · What the measurements say about open-ended evolution](chapters/29-meta-2.md)

**Part IV · One change at a time**

- [Chapter 30 · Do the brains matter?](chapters/30-do-the-brains-matter.md)
- [Chapter 31 · How does a world's size follow its tokens?](chapters/31-how-does-a-worlds-size-follow-its-tokens.md)
- Chapter 32 · How fast should brains change? — *not written yet*
- Chapter 33 · What keeps wealth spread? — *not written yet*
- Chapter 34 · Meta III — *not written yet*
- Chapter 35 · How big should a brain be? — *not written yet*
- Chapter 36 · Does size change the dynamics? — *not written yet*

**Part V · Towards open-ended evolution**

- Chapter 37 · Mutation at replication, and nowhere else — *not written yet*
- Chapter 38 · Can a brain hold its place? — *not written yet*
- Chapter 39 · Kin that know each other — *not written yet*
- Chapter 40 · When cooperation produces — *not written yet*
- Chapter 41 · An open world — *not written yet*
- Chapter 42 · Meta IV — *not written yet*

**Notes · the rules**

- [Every setting](notes/settings.md)
- [The starting ring](notes/starting-ring.md)
- [Seeds and the random stream](notes/random-numbers.md)
- [The share function f(a, b)](notes/share-function.md)
- [A yes-or-no decision](notes/binary-decision.md)
- [Splitting tokens into whole numbers](notes/largest-remainder.md)
- [What a brain sees](notes/brain-inputs.md)
- [What a brain says](notes/brain-outputs.md)
- [Messages](notes/messages.md)
- [How a coalition takes a node](notes/revolution.md)
- [How a brain changes](notes/mutation.md)
- [Genotypes and the family tree](notes/genotype.md)
- [When a world ends](notes/extinction.md)

**Notes · what is measured**

- [What a run records](notes/frames-and-stats.md)
- [Births and the four ways to die](notes/births-and-deaths.md)
- [Age and lifetime](notes/age-and-lifetime.md)
- [Degree](notes/degree.md)
- [Clustering](notes/clustering.md)
- [Path length](notes/path-length.md)
- [Core, trees and leaves](notes/core-trees-leaves.md)
- [Bridges](notes/bridges.md)
- [The Gini coefficient and the Lorenz curve](notes/gini-coefficient.md)
- [The richest tenth's share](notes/richest-tenth.md)
- [Staking at home, and keeping one's node](notes/home-stake.md)
- [Families](notes/families.md)
- [The common ancestor of the living](notes/common-ancestor.md)
- [Tokens per agent: mean, median, extremes](notes/tokens-per-agent.md)
- [Entropy and evenness](notes/entropy-and-evenness.md)
- [Gains, losses and the share-out](notes/gains-and-losses.md)
- [Brain diversity](notes/brain-diversity.md)
- [Reproduction statistics](notes/reproduction-statistics.md)
- [Token flow](notes/token-flow.md)
- [Lightning: tokens that go round](notes/lightning.md)
- [Token curvature](notes/token-curvature.md)
- [Loops](notes/loops.md)
- [Cut risk](notes/cut-risk.md)
- [Radius and diameter](notes/radius-and-diameter.md)
- [The spectral gap](notes/spectral-gap.md)
- [Dimension and curvature from ball growth](notes/ball-dimension-and-curvature.md)
- [Box dimension](notes/box-dimension.md)
- [Scaling relations between agents' properties](notes/scaling-relations.md)
- [Assortativity](notes/assortativity.md)
- [What the viewer can colour by](notes/viewer-colours.md)

**Notes · statistics**

- [Median and quantiles](notes/median-and-quantiles.md)
- [Bands: many worlds in one figure](notes/bands.md)
- [The settled life of a world](notes/settled-life.md)
- [Spread between worlds](notes/spread-between-worlds.md)
- [Correlation](notes/correlation.md)
- [Autocorrelation: how long a world remembers](notes/autocorrelation.md)
- [Between worlds and within them](notes/between-and-within.md)
- [Survival curves (Kaplan–Meier)](notes/kaplan-meier.md)
- [The bootstrap](notes/bootstrap.md)
- [Permutation tests](notes/permutation-test.md)
- [How many seeds an experiment needs](notes/seeds-needed.md)
- [The Wilson interval](notes/wilson-interval.md)
- [Logarithmic axes and power laws](notes/logarithmic-axes.md)
- [Muller plots](notes/muller-plot.md)
- [Fitting a straight line, and R²](notes/least-squares.md)
- [Fitting a power law properly](notes/power-law-fit.md)
- [The power spectrum of a time series](notes/power-spectrum.md)

**Appendices**

- [Appendix A · What a simulation costs](chapters/A-costs.md)
<!-- /contents -->

## How it is made

Three steps, over and over.

1. **A plan.** An experiment's plan (`book/experiments/E…json`) fixes the
   settings, the size of the world, the seeds and how long each run goes —
   and a **thesis**: what is expected, why, and what would show it wrong. The
   thesis is written down before any of its runs exist, and the chapter
   quotes it from the plan, so it cannot be reworded once the results are
   in.
2. **The runs.** On a computer running `gol_server.py`, the chapter has a ▶
   button in the Book tab. The lab then runs the experiment's simulations,
   several at a time, each in a process of its own, and says how long is
   left. It can be paused and continued at any time.
3. **The results.** The finished runs are analysed, the figures drawn, and
   the chapter written: what was found, whether the thesis held, and what to
   try next.

Some chapters of Part II only ask, without a thesis: they look at runs that
already exist from another side.

## The rules

- **One change at a time.** Every experiment changes one setting of a stated
  baseline ([Every setting](notes/settings.md)) and keeps everything else.
- **The thesis comes first.** It is fixed in the plan before the runs start.
- **Enough seeds.** A difference seen in fewer than thirty seeds per
  condition is called *indicative*, not an effect: results in this project
  have reversed between six seeds and twenty
  ([How many seeds](notes/seeds-needed.md)).
- **No number without its null.** A measurement is reported next to what it
  would be by chance.
- **Every run can be made again.** Each simulation records its seed, the
  frozen copy of the engine it ran on and the commit it came from, and the
  versions of Python, numpy and networkx, beside it in `provenance.json`
  ([Chapter 7](chapters/07-is-a-run-reproducible.md)).
- **The dead are counted apart.** A world that dies out did not end up
  anywhere, so where the worlds of a condition end up is measured over the
  worlds that lived to the end, and the dead are reported beside them as an
  outcome of their own ([When a world ends](notes/extinction.md)).
- **A run is measured over its settled life**, from iteration 500 to its end
  ([The settled life](notes/settled-life.md)).
- **Runs are shared.** A run is named for what it is — baseline, world size,
  what differs, seed — so an experiment that needs runs which already exist
  uses them.

## Making a result again

Every chapter that reads runs says, in a box at its top, which runs they are,
every setting they were made with, and the command that makes them:
`python3 gol_lab.py run E…`. `python3 gol_lab.py verify E…` checks a few of
them against what was recorded; `python3 gol_lab.py analyse E…` computes an
experiment's results; `python3 book_figures.py` draws every figure again and
writes it, with its caption and recipe, into the chapters. On the same kind
of machine with the same libraries everything comes out identical.

## Where things are

- `book/chapters/` — the chapters;
- `book/notes/` — one note per rule, measurement or method;
- `book/figures/`, `book/diagrams/` — every figure and diagram, as SVG;
- `book/results/` — every number a chapter quotes, written by the analysis;
- `book/experiments/` — the baseline and each experiment's plan;
- `book/book.json` — the order of the chapters and notes, from which the
  contents above are made;
- `GraphOfLifeRuns/` — the runs themselves, on the machine that ran them.

The programme the book follows, and what is already known, is in
`research/Research.md` — the *Findings* page under Research.
