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
- [Chapter 7 · Physical inspiration](chapters/07-physical-inspiration.md)
- [Chapter 8 · Is a run reproducible?](chapters/08-is-a-run-reproducible.md)

**Part II · The baseline world**

- [Chapter 9 · Thirty worlds](chapters/09-thirty-worlds.md)
- [Chapter 10 · A world's life](chapters/10-a-worlds-life.md)
- [Chapter 11 · The first hundred iterations](chapters/11-the-first-hundred-iterations.md)
- [Chapter 12 · Births, deaths and ages](chapters/12-births-deaths-and-ages.md)
- [Chapter 13 · How much does the seed decide?](chapters/13-how-much-does-the-seed-decide.md)
- [Chapter 14 · Where do the tokens go?](chapters/14-where-do-the-tokens-go.md)
- [Chapter 15 · How the game is played](chapters/15-how-the-game-is-played.md)
- [Chapter 16 · What shape does the network take?](chapters/16-what-shape-does-the-network-take.md)
- [Chapter 17 · Genotypes and lineages](chapters/17-genotypes-and-lineages.md)
- [Chapter 18 · Meta I · What the baseline world is](chapters/18-meta-1.md)

**Part III · The baseline world, measured every way**

- [Chapter 19 · How even is a world?](chapters/19-how-even-is-a-world.md)
- [Chapter 20 · Gains and losses](chapters/20-gains-and-losses.md)
- [Chapter 21 · Where the tokens flow](chapters/21-where-the-tokens-flow.md)
- [Chapter 22 · How agents have children](chapters/22-how-agents-have-children.md)
- [Chapter 23 · Power laws, real and apparent](chapters/23-power-laws-real-and-apparent.md)
- [Chapter 24 · How properties scale together](chapters/24-how-properties-scale-together.md)
- [Chapter 25 · The geometry of a world](chapters/25-the-geometry-of-a-world.md)
- [Chapter 26 · How a world breaks](chapters/26-how-a-world-breaks.md)
- [Chapter 27 · One world, many colours](chapters/27-one-world-many-colours.md)
- [Chapter 28 · Do agents cooperate?](chapters/28-do-agents-cooperate.md)
- [Chapter 29 · Questioning the mechanics](chapters/29-questioning-the-mechanics.md)
- [Chapter 30 · Meta II · What the measurements say about open-ended evolution](chapters/30-meta-2.md)

**Part IV · One change at a time**

- [Chapter 31 · Do the brains matter?](chapters/31-do-the-brains-matter.md)
- [Chapter 32 · How does a world's size follow its tokens?](chapters/32-how-does-a-worlds-size-follow-its-tokens.md)
- [Chapter 33 · How fast should brains change?](chapters/33-how-fast-should-brains-change.md)
- Chapter 34 · Brains replaced by their average — *not written yet*
- Chapter 35 · What keeps wealth spread? — *not written yet*
- Chapter 36 · Meta III — *not written yet*
- Chapter 37 · How big should a brain be? — *not written yet*
- Chapter 38 · Does size change the dynamics? — *not written yet*

**Part V · Towards open-ended evolution**

- Chapter 39 · Mutation at replication, and nowhere else — *not written yet*
- Chapter 40 · Can a brain hold its place? — *not written yet*
- Chapter 41 · Kin that know each other — *not written yet*
- Chapter 42 · When cooperation produces — *not written yet*
- Chapter 43 · An open world — *not written yet*
- Chapter 44 · Meta IV — *not written yet*

**Part VI · Towards a physics: space, scale and locality**

- Chapter 45 · Worlds of 100,000 tokens — *not written yet*
- Chapter 46 · How many dimensions does a world have? — *not written yet*
- Chapter 47 · Which properties have no scale? — *not written yet*
- Chapter 48 · Local rules only — *not written yet*
- Chapter 49 · How far does a difference travel? — *not written yet*
- Chapter 50 · A source and a sink — *not written yet*
- Chapter 51 · Particles — *not written yet*
- Chapter 52 · One agent in a sea of average agents — *not written yet*
- Chapter 53 · Meta V — *not written yet*

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
  ([Chapter 8](chapters/08-is-a-run-reproducible.md)).
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
