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
- [Chapter 13 · How ages change over a world's life](chapters/13-how-ages-change.md)
- [Chapter 14 · How much does the seed decide?](chapters/14-how-much-does-the-seed-decide.md)
- [Chapter 15 · Where do the tokens go?](chapters/15-where-do-the-tokens-go.md)
- [Chapter 16 · How the game is played](chapters/16-how-the-game-is-played.md)
- [Chapter 17 · What shape does the network take?](chapters/17-what-shape-does-the-network-take.md)
- [Chapter 18 · Genotypes and lineages](chapters/18-genotypes-and-lineages.md)
- [Chapter 19 · What the brains are like](chapters/19-what-the-brains-are-like.md)
- [Chapter 20 · What agents say to each other](chapters/20-what-agents-say-to-each-other.md)
- [Chapter 21 · Do the rich stay rich?](chapters/21-do-the-rich-stay-rich.md)
- [Chapter 22 · How the network grows](chapters/22-how-the-network-grows.md)
- [Chapter 23 · Like next to like](chapters/23-like-next-to-like.md)
- [Chapter 24 · How many dimensions does a world have?](chapters/24-how-many-dimensions-does-a-world-have.md)
- [Chapter 25 · The core of a world](chapters/25-the-core-of-a-world.md)
- [Chapter 26 · Meta I · What the baseline world is](chapters/26-meta-1.md)

**Part III · The baseline world, measured every way**

- [Chapter 27 · How even is a world?](chapters/27-how-even-is-a-world.md)
- [Chapter 28 · Gains and losses](chapters/28-gains-and-losses.md)
- [Chapter 29 · Where the tokens flow](chapters/29-where-the-tokens-flow.md)
- [Chapter 30 · What crosses a connection](chapters/30-what-crosses-a-connection.md)
- [Chapter 31 · How agents have children](chapters/31-how-agents-have-children.md)
- [Chapter 32 · Power laws, real and apparent](chapters/32-power-laws-real-and-apparent.md)
- [Chapter 33 · The largest and the smallest](chapters/33-the-largest-and-the-smallest.md)
- [Chapter 34 · How properties scale together](chapters/34-how-properties-scale-together.md)
- [Chapter 35 · The geometry of a world](chapters/35-the-geometry-of-a-world.md)
- [Chapter 36 · How a world breaks](chapters/36-how-a-world-breaks.md)
- [Chapter 37 · One world, many colours](chapters/37-one-world-many-colours.md)
- [Chapter 38 · Do agents cooperate?](chapters/38-do-agents-cooperate.md)
- [Chapter 39 · Questioning the mechanics](chapters/39-questioning-the-mechanics.md)
- [Chapter 40 · Meta II · What the measurements say about open-ended evolution](chapters/40-meta-2.md)
- [Chapter 70 · How different are the agents?](chapters/70-how-different-are-the-agents.md)
- [Chapter 71 · Flow modules: the groups tokens stay in](chapters/71-flow-modules.md)
- [Chapter 72 · Lightning in every world](chapters/72-lightning-in-every-world.md)

**Part IV · One change at a time**

- [Chapter 41 · Do the brains matter?](chapters/41-do-the-brains-matter.md)
- [Chapter 42 · How does a world's size follow its tokens?](chapters/42-how-does-a-worlds-size-follow-its-tokens.md)
- Chapter 43 · Brains replaced by their average — *not written yet*
- Chapter 44 · What keeps wealth spread? — *not written yet*
- Chapter 45 · Meta III — *not written yet*

**Part V · Engineering open-ended evolution**

- [Chapter 46 · How to tell a world that keeps evolving](chapters/46-how-to-tell-a-world-that-keeps-evolving.md)
- [Chapter 47 · Brains that see, and worlds that break](chapters/47-brains-that-see-and-worlds-that-break.md)
- [Chapter 48 · A world that cannot break](chapters/48-a-world-that-cannot-break.md)
- [Chapter 49 · Who dies, and why](chapters/49-who-dies-and-why.md)
- [Chapter 50 · Why evolution stops](chapters/50-why-evolution-stops.md)
- [Chapter 51 · Strategies written by hand](chapters/51-strategies-written-by-hand.md)
- [Chapter 52 · How fast should brains change?](chapters/52-how-fast-should-brains-change.md)
- Chapter 53 · How big should a brain be? — *not written yet*
- [Chapter 54 · The switches the world was built with](chapters/54-the-switches-the-world-was-built-with.md)
- [Chapter 55 · Another game](chapters/55-another-game.md)
- [Chapter 56 · Alliances](chapters/56-alliances.md)
- [Chapter 57 · Rules without numbers](chapters/57-rules-without-numbers.md)
- [Chapter 58 · Gliders, and who helps whom](chapters/58-gliders-and-who-helps-whom.md)
- Chapter 59 · Meta IV — *not written yet*
- [Chapter 73 · Who inherits?](chapters/73-who-inherits.md)
- [Chapter 78 · Brains mixed at birth, units that cost, an eye for raids on kin](chapters/78-mixed-brains-costly-units.md)
- [Chapter 80 · Mixed at birth: ten long worlds](chapters/80-mixed-at-birth-ten-long-worlds.md)
- [Chapter 83 · Links to strangers](chapters/83-links-to-strangers.md)
- [Chapter 82 · Brains mixed at birth, with every other rule](chapters/82-mixing-with-every-other-rule.md)
- [Chapter 85 · Mixed at birth, with parents as mates](chapters/85-mixed-at-birth-with-parents-as-mates.md)
- [Chapter 86 · Eight switches at once](chapters/86-eight-switches-at-once.md)
- [Chapter 88 · Revolutions, with every other rule](chapters/88-revolutions-with-every-other-rule.md)
- [Chapter 89 · Handover, with every other rule](chapters/89-handover-with-every-other-rule.md)
- [Chapter 90 · Gifts, with every other rule](chapters/90-gifts-with-every-other-rule.md)
- [Chapter 98 · The search's best world, in new seeds](chapters/98-the-searchs-best-world-in-new-seeds.md)
- [Chapter 99 · A mate that pays its share](chapters/99-a-mate-that-pays-its-share.md)
- [Chapter 100 · Room for the steadiest world](chapters/100-room-for-the-steadiest-world.md)

**Part VI · Towards a physics: space, scale and locality**

- [Chapter 60 · Worlds of 100,000 tokens](chapters/60-worlds-of-100000-tokens.md)
- Chapter 61 · Dimensions at scale — *not written yet*
- Chapter 62 · Which properties have no scale? — *not written yet*
- Chapter 63 · Local rules only — *not written yet*
- Chapter 64 · How far does a difference travel? — *not written yet*
- Chapter 65 · A source and a sink — *not written yet*
- Chapter 66 · Particles — *not written yet*
- Chapter 67 · One agent in a sea of average agents — *not written yet*
- Chapter 68 · Meta V — *not written yet*
- [Chapter 87 · What folds a world](chapters/87-what-folds-a-world.md)

**Part VII · The long run**

- [Chapter 69 · A hundred thousand iterations](chapters/69-a-hundred-thousand-iterations.md)
- [Chapter 81 · Five mixed worlds, fifty thousand iterations](chapters/81-five-mixed-worlds-fifty-thousand-iterations.md)
- [Chapter 92 · The candidates, played on](chapters/92-the-candidates-played-on.md)
- [Chapter 93 · Five more seeds for the rules that held](chapters/93-five-more-seeds-for-the-rules-that-held.md)

**Part VIII · Across all worlds**

- [Chapter 74 · Meta VI · Every world side by side](chapters/74-meta-6-every-world.md)
- [Chapter 75 · Meta VII · Where tokens go](chapters/75-meta-7-where-tokens-go.md)
- [Chapter 76 · Meta VIII · Who helps whom](chapters/76-meta-8-who-helps-whom.md)
- [Chapter 77 · Meta IX · How alike the agents are](chapters/77-meta-9-how-alike.md)
- [Chapter 79 · Meta X · Healthy and rich](chapters/79-meta-10-healthy-and-rich.md)
- [Chapter 84 · Meta XI · What holds a world's size](chapters/84-meta-11-what-holds-a-worlds-size.md)
- [Chapter 91 · Meta XII · Rules in company](chapters/91-meta-12-rules-in-company.md)

**Part IX · A ladder of strategies**

- [Chapter 94 · What a brain can compute](chapters/94-what-a-brain-can-compute.md)
- [Chapter 95 · Every era against three](chapters/95-every-era-against-three.md)
- [Chapter 96 · How large a crowd a raider needs](chapters/96-how-large-a-crowd-a-raider-needs.md)
- [Chapter 97 · Does evolution close the door?](chapters/97-does-evolution-close-the-door.md)

**Notes · the rules**

- [Every setting](notes/settings.md)
- [The starting ring](notes/starting-ring.md)
- [Seeds and the random stream](notes/random-numbers.md)
- [The share function f(a, b)](notes/share-function.md)
- [A yes-or-no decision](notes/binary-decision.md)
- [Splitting tokens into whole numbers](notes/largest-remainder.md)
- [What a brain sees](notes/brain-inputs.md)
- [What a brain says](notes/brain-outputs.md)
- [How a signal fades through layers](notes/vanishing-signal.md)
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
- [Preferential attachment and its kernel](notes/preferential-attachment.md)
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
- [The spectral dimension](notes/spectral-dimension.md)
- [Box dimension](notes/box-dimension.md)
- [Scaling relations between agents' properties](notes/scaling-relations.md)
- [Assortativity](notes/assortativity.md)
- [How alike at a distance](notes/correlation-function.md)
- [What the viewer can colour by](notes/viewer-colours.md)
- [Pilot worlds](notes/pilot-worlds.md)
- [The invasion test](notes/invasion-test.md)
- [Exploitability](notes/exploitability.md)
- [Flows that go round](notes/circulation.md)
- [Help](notes/help.md)
- [Flow modules and the map equation](notes/map-equation.md)
- [What would count as open-ended evolution](notes/open-ended-test.md)

**Notes · statistics**

- [Median and quantiles](notes/median-and-quantiles.md)
- [Bands: many worlds in one figure](notes/bands.md)
- [The settled life of a world](notes/settled-life.md)
- [Spread between worlds](notes/spread-between-worlds.md)
- [Correlation](notes/correlation.md)
- [Autocorrelation: how long a world remembers](notes/autocorrelation.md)
- [Mobility](notes/mobility.md)
- [Between worlds and within them](notes/between-and-within.md)
- [How many directions a cloud of points uses](notes/participation-ratio.md)
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
- [Where, who and chance: splitting a decision's variance](notes/variance-split.md)
- [Effective numbers: how many kinds there really are](notes/effective-number.md)
- [Many switches in few worlds: fractional factorial designs](notes/fractional-factorial.md)

**Appendices**

- [Appendix A · What a simulation costs](chapters/A-costs.md)
- [Appendix B · What goes with what](chapters/B-what-goes-with-what.md)
- [Appendix C · Every simulation: what was run, and what was measured](chapters/C-every-simulation.md)
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
