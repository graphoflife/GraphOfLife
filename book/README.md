# Graph of Life — the Research Notebook

This notebook is a research programme, written down while it is carried out. It
asks what the Graph of Life algorithm actually does — and whether it can be
made to keep producing new things without end, which is what *open-ended
evolution* means. [What this notebook is about](pages/what-this-notebook-is-about.md)
says why.

It is written to be **read on its own**, from any page. The pages are sorted
by topic, and a word that needs explaining links to the page that explains it.
[How the simulation works](#how-the-simulation-works) explains the algorithm
completely, rule by rule. Every measurement, statistic and method the notebook
uses has a short page of its own that defines it exactly and works an example
by hand. Every figure says what it shows and, folded under it, how to make it
again from the runs. Nothing needs the program to be understood — and
everything can be checked with it.

## How to read it

- **By topic**: the contents below list every page under its topic, in the
  order that reads best; every page ends with the pages before and after it
  in its topic.
- **From anywhere**: start with any page and follow a link whenever a word
  is new.
- **By question**: every page starts with the questions it asks.
- **In Obsidian**: open the folder `book/` as a vault. Every link, figure
  and formula works there, and the graph view shows how the pages hang
  together. The same files read in the Research Notebook tab of the Graph of
  Life app and on GitHub.

## Contents

<!-- contents -->
### Start here

What this notebook is about, how to read it, and the question behind it: can evolution in a simulated world go on without end?

- [What this notebook is about](pages/what-this-notebook-is-about.md)
- [The ideas this builds on](pages/the-ideas-this-builds-on.md)

### How the simulation works

The rules of a world, step by step: tokens, agents and links, the game, births and deaths, the brains, and every setting.

- [The world: tokens, agents and a network](pages/the-world.md)
- [One iteration, step by step](pages/one-iteration.md)
- [The brain](pages/the-brain.md)
- [Questioning the mechanics](pages/questioning-the-mechanics.md)
- [Is a run reproducible?](pages/is-a-run-reproducible.md)
- [Every setting](pages/settings.md)
- [The starting ring](pages/starting-ring.md)
- [Seeds and the random stream](pages/random-numbers.md)
- [The share function f(a, b)](pages/share-function.md)
- [A yes-or-no decision](pages/binary-decision.md)
- [Splitting tokens into whole numbers](pages/largest-remainder.md)
- [What a brain sees](pages/brain-inputs.md)
- [What a brain says](pages/brain-outputs.md)
- [Messages](pages/messages.md)
- [How a coalition takes a node](pages/revolution.md)
- [How a brain changes](pages/mutation.md)
- [When a world ends](pages/extinction.md)

### The baselines B1 and B2

The two standard worlds every experiment starts from: B1, the original rules, and B2, a world that cannot break, and how it was built.

- [What the baseline world is](pages/what-the-baseline-world-is.md)
- [Thirty worlds](pages/thirty-worlds.md)
- [A world's life](pages/a-worlds-life.md)
- [The first hundred iterations](pages/the-first-hundred-iterations.md)
- [How much does the seed decide?](pages/how-much-does-the-seed-decide.md)
- [Brains that see, and worlds that break](pages/brains-that-see-and-worlds-that-break.md)
- [A world that cannot break](pages/a-world-that-cannot-break.md)
- [Who dies, and why](pages/who-dies-and-why.md)

### Agents' lives: births, deaths, ages

Who is born, who dies and when, how old agents get, and how a world's ages change.

- [Births, deaths and ages](pages/births-deaths-and-ages.md)
- [How ages change over a world's life](pages/how-ages-change.md)
- [How agents have children](pages/how-agents-have-children.md)
- [Births and the four ways to die](pages/births-and-deaths.md)
- [Age and lifetime](pages/age-and-lifetime.md)
- [Reproduction statistics](pages/reproduction-statistics.md)

### Wealth: who holds the tokens

How unequal a world is, who stays rich, and how gains and losses are shared out.

- [Where do the tokens go?](pages/where-do-the-tokens-go.md)
- [How even is a world?](pages/how-even-is-a-world.md)
- [Do the rich stay rich?](pages/do-the-rich-stay-rich.md)
- [Gains and losses](pages/gains-and-losses.md)
- [The largest and the smallest](pages/the-largest-and-the-smallest.md)
- What keeps wealth spread? — *not written yet*
- [The Gini coefficient and the Lorenz curve](pages/gini-coefficient.md)
- [The richest tenth's share](pages/richest-tenth.md)
- [Tokens per agent: mean, median, extremes](pages/tokens-per-agent.md)
- [Entropy and evenness](pages/entropy-and-evenness.md)
- [Gains, losses and the share-out](pages/gains-losses-and-the-share-out.md)

### The game and the flow of tokens

How the game is played, where tokens flow, which loops they go round, and the groups they stay in.

- [Where tokens go](pages/where-tokens-go.md)
- [How the game is played](pages/how-the-game-is-played.md)
- [Where the tokens flow](pages/where-the-tokens-flow.md)
- [What crosses a connection](pages/what-crosses-a-connection.md)
- [Flow modules: the groups tokens stay in](pages/flow-modules.md)
- [Lightning in every world](pages/lightning-in-every-world.md)
- [Staking at home, and keeping one's node](pages/home-stake.md)
- [Token flow](pages/token-flow.md)
- [Lightning: tokens that go round](pages/lightning.md)
- [Token curvature](pages/token-curvature.md)
- [Flows that go round](pages/circulation.md)
- [Flow modules and the map equation](pages/map-equation.md)

### The shape and health of the network

Degrees, cores, bridges, power laws and cuts: what shape the links take, and how a world breaks.

- [What shape does the network take?](pages/what-shape-does-the-network-take.md)
- [How the network grows](pages/how-the-network-grows.md)
- [The core of a world](pages/the-core-of-a-world.md)
- [Power laws, real and apparent](pages/power-laws-real-and-apparent.md)
- [How properties scale together](pages/how-properties-scale-together.md)
- [The geometry of a world](pages/the-geometry-of-a-world.md)
- [How a world breaks](pages/how-a-world-breaks.md)
- [Degree](pages/degree.md)
- [Preferential attachment and its kernel](pages/preferential-attachment.md)
- [Clustering](pages/clustering.md)
- [Path length](pages/path-length.md)
- [Core, trees and leaves](pages/core-trees-leaves.md)
- [Bridges](pages/bridges.md)
- [Loops](pages/loops.md)
- [Cut risk](pages/cut-risk.md)
- [Radius and diameter](pages/radius-and-diameter.md)
- [The spectral gap](pages/spectral-gap.md)
- [Assortativity](pages/assortativity.md)
- [Scaling relations between agents' properties](pages/scaling-relations.md)

### Space and dimension

Whether a world is a space, of how many dimensions, what folds it, and the physics the project hopes for.

- [Physical inspiration](pages/physical-inspiration.md)
- [Like next to like](pages/like-next-to-like.md)
- [How many dimensions does a world have?](pages/how-many-dimensions-does-a-world-have.md)
- [Worlds of 100,000 tokens](pages/worlds-of-100000-tokens.md)
- [What folds a world](pages/what-folds-a-world.md)
- Dimensions at scale — *not written yet*
- Which properties have no scale? — *not written yet*
- Local rules only — *not written yet*
- How far does a difference travel? — *not written yet*
- A source and a sink — *not written yet*
- Particles — *not written yet*
- [How alike at a distance](pages/correlation-function.md)
- [Dimension and curvature from ball growth](pages/ball-dimension-and-curvature.md)
- [The spectral dimension](pages/spectral-dimension.md)
- [Box dimension](pages/box-dimension.md)

### Size and riches of a world

What sets how many agents a world holds, and how rich each is.

- [What holds a world's size](pages/what-holds-a-worlds-size.md)
- [How does a world's size follow its tokens?](pages/how-does-a-worlds-size-follow-its-tokens.md)
- [Births as rich as the neighbours](pages/births-as-rich-as-the-neighbours.md)
- [The floor in plain B2 and in B1](pages/the-floor-in-plain-b2-and-in-b1.md)

### Brains, and how alike agents are

What the brains are and compute, the families they make, and how different agents really are.

- [How alike the agents are](pages/how-alike-the-agents-are.md)
- [Genotypes and lineages](pages/genotypes-and-lineages.md)
- [What the brains are like](pages/what-the-brains-are-like.md)
- [What agents say to each other](pages/what-agents-say-to-each-other.md)
- [Do the brains matter?](pages/do-the-brains-matter.md)
- Brains replaced by their average — *not written yet*
- [How different are the agents?](pages/how-different-are-the-agents.md)
- How big should a brain be? — *not written yet*
- [Genotypes and the family tree](pages/genotype.md)
- [How a signal fades through layers](pages/vanishing-signal.md)
- [Brain diversity](pages/brain-diversity.md)
- [Families](pages/families.md)
- [The common ancestor of the living](pages/common-ancestor.md)

### Does evolution keep going?

How to tell whether evolution goes on, why it stops, and what would count as open-ended evolution.

- [What the measurements say about open-ended evolution](pages/what-the-measurements-say-about-open-ended-evolution.md)
- [How to tell a world that keeps evolving](pages/how-to-tell-a-world-that-keeps-evolving.md)
- [Why evolution stops](pages/why-evolution-stops.md)
- [How fast should brains change?](pages/how-fast-should-brains-change.md)
- [Every era against three](pages/every-era-against-three.md)
- [A hundred thousand iterations](pages/a-hundred-thousand-iterations.md)
- [What would count as open-ended evolution](pages/open-ended-test.md)
- [Exploitability](pages/exploitability.md)

### Strategies written by hand

Raiders, turtles and crews written by hand and planted in evolved worlds: is evolution's play the best?

- [Strategies written by hand](pages/strategies-written-by-hand.md)
- [What a brain can compute](pages/what-a-brain-can-compute.md)
- [How large a crowd a raider needs](pages/how-large-a-crowd-a-raider-needs.md)
- [Does evolution close the door?](pages/does-evolution-close-the-door.md)
- [A crew of raiders](pages/a-crew-of-raiders.md)
- [Raiders that breed](pages/raiders-that-breed.md)
- [Hand-written strategies in B1](pages/hand-written-strategies-in-b1.md)
- One agent in a sea of average agents — *not written yet*
- [The invasion test](pages/invasion-test.md)

### Groups and cooperation

Help, alliances, other contests and patterns that outlast their agents: do agents work together?

- [Who helps whom](pages/who-helps-whom.md)
- [Do agents cooperate?](pages/do-agents-cooperate.md)
- [Another game](pages/another-game.md)
- [Alliances](pages/alliances.md)
- [Gliders, and who helps whom](pages/gliders-and-who-helps-whom.md)
- [Help](pages/help.md)

### Changing the rules

The switches of the engine, alone and in company: what each rule does to a world.

- [Rules in company](pages/rules-in-company.md)
- [The switches the world was built with](pages/the-switches-the-world-was-built-with.md)
- [Rules without numbers](pages/rules-without-numbers.md)
- [Who inherits?](pages/who-inherits.md)
- [Brains mixed at birth, units that cost, an eye for raids on kin](pages/mixed-brains-costly-units.md)
- [Links to strangers](pages/links-to-strangers.md)
- [Brains mixed at birth, with every other rule](pages/mixing-with-every-other-rule.md)
- [Eight switches at once](pages/eight-switches-at-once.md)
- [Revolutions, with every other rule](pages/revolutions-with-every-other-rule.md)
- [Handover, with every other rule](pages/handover-with-every-other-rule.md)
- [Gifts, with every other rule](pages/gifts-with-every-other-rule.md)

### Looking for a healthy, rich world

The search for a world that is decentralised, resilient, about three-dimensional and full of different agents, and that keeps evolving.

- [Healthy and rich](pages/healthy-and-rich.md)
- [What makes a world interesting](pages/what-makes-a-world-interesting.md)
- [Mixed at birth: ten long worlds](pages/mixed-at-birth-ten-long-worlds.md)
- [Five mixed worlds, fifty thousand iterations](pages/five-mixed-worlds-fifty-thousand-iterations.md)
- [Mixed at birth, with parents as mates](pages/mixed-at-birth-with-parents-as-mates.md)
- [A mate that pays its share](pages/a-mate-that-pays-its-share.md)
- [The candidates, played on](pages/the-candidates-played-on.md)
- [Five more seeds for the rules that held](pages/five-more-seeds-for-the-rules-that-held.md)
- [The search's best world, in new seeds](pages/the-searchs-best-world-in-new-seeds.md)
- [Room for the steadiest world](pages/room-for-the-steadiest-world.md)
- [A screen of random worlds](pages/a-screen-of-random-worlds.md)

### Methods and statistics

How runs are recorded and measured, the statistics the notebook uses, and every simulation made.

- [Every world side by side](pages/every-world-side-by-side.md)
- [How worlds are measured](pages/how-worlds-are-measured.md)
- [One world, many colours](pages/one-world-many-colours.md)
- [What a simulation costs](pages/what-a-simulation-costs.md)
- [What goes with what](pages/what-goes-with-what.md)
- [Every simulation: what was run, and what was measured](pages/every-simulation.md)
- [What a run records](pages/frames-and-stats.md)
- [What the viewer can colour by](pages/viewer-colours.md)
- [Pilot worlds](pages/pilot-worlds.md)
- [Median and quantiles](pages/median-and-quantiles.md)
- [Bands: many worlds in one figure](pages/bands.md)
- [The settled life of a world](pages/settled-life.md)
- [Spread between worlds](pages/spread-between-worlds.md)
- [Correlation](pages/correlation.md)
- [Autocorrelation: how long a world remembers](pages/autocorrelation.md)
- [Mobility](pages/mobility.md)
- [Between worlds and within them](pages/between-and-within.md)
- [How many directions a cloud of points uses](pages/participation-ratio.md)
- [Survival curves (Kaplan–Meier)](pages/kaplan-meier.md)
- [The bootstrap](pages/bootstrap.md)
- [Permutation tests](pages/permutation-test.md)
- [How many seeds an experiment needs](pages/seeds-needed.md)
- [The Wilson interval](pages/wilson-interval.md)
- [Logarithmic axes and power laws](pages/logarithmic-axes.md)
- [Muller plots](pages/muller-plot.md)
- [Fitting a straight line, and R²](pages/least-squares.md)
- [Fitting a power law properly](pages/power-law-fit.md)
- [The power spectrum of a time series](pages/power-spectrum.md)
- [Where, who and chance: splitting a decision's variance](pages/variance-split.md)
- [Effective numbers: how many kinds there really are](pages/effective-number.md)
- [Many switches in few worlds: fractional factorial designs](pages/fractional-factorial.md)
<!-- /contents -->

## How it is made

Three steps, over and over.

1. **A plan.** An experiment's plan (`book/experiments/E…json`) fixes the
   settings, the size of the world, the seeds and how long each run goes —
   and a **thesis**: what is expected, why, and what would show it wrong. The
   thesis is written down before any of its runs exist, and the page
   quotes it from the plan, so it cannot be reworded once the results are
   in.
2. **The runs.** On a computer running `gol_server.py`, the page has a ▶
   button in the Research Notebook tab. The lab then runs the experiment's simulations,
   several at a time, each in a process of its own, and says how long is
   left. It can be paused and continued at any time.
3. **The results.** The finished runs are analysed, the figures drawn, and
   the page written: what was found, whether the thesis held, and what to
   try next.

Some pages only ask, without a thesis: they look at runs that already exist
from another side.

## The rules

- **One change at a time.** Every experiment changes one setting of a stated
  baseline ([Every setting](pages/settings.md)) and keeps everything else.
- **The thesis comes first.** It is fixed in the plan before the runs start.
- **Enough seeds.** A difference seen in fewer than thirty seeds per
  condition is called *indicative*, not an effect: results in this project
  have reversed between six seeds and twenty
  ([How many seeds](pages/seeds-needed.md)).
- **No number without its null.** A measurement is reported next to what it
  would be by chance.
- **Every run can be made again.** Each simulation records its seed, the
  frozen copy of the engine it ran on and the commit it came from, and the
  versions of Python, numpy and networkx, beside it in `provenance.json`
  ([Is a run reproducible?](pages/is-a-run-reproducible.md)).
- **The dead are counted apart.** A world that dies out did not end up
  anywhere, so where the worlds of a condition end up is measured over the
  worlds that lived to the end, and the dead are reported beside them as an
  outcome of their own ([When a world ends](pages/extinction.md)).
- **A run is measured over its settled life**, from iteration 500 to its end
  ([The settled life](pages/settled-life.md)).
- **Runs are shared.** A run is named for what it is — baseline, world size,
  what differs, seed — so an experiment that needs runs which already exist
  uses them.

## Making a result again

Every page that reads runs says, in a box at its top, which runs they are,
every setting they were made with, and the command that makes them:
`python3 gol_lab.py run E…`. `python3 gol_lab.py verify E…` checks a few of
them against what was recorded; `python3 gol_lab.py analyse E…` computes an
experiment's results; `python3 book_figures.py` draws every figure again and
writes it, with its caption and recipe, into the pages. On the same kind
of machine with the same libraries everything comes out identical.

## Where things are

- `book/pages/` — every page, named by what it is about;
- `book/figures/`, `book/diagrams/` — every figure and diagram, as SVG;
- `book/results/` — every number a page quotes, written by the analysis;
- `book/experiments/` — the baseline and each experiment's plan;
- `book/book.json` — the topics, and every page with its topics, kind and
  summary, from which the contents above are made;
- `book/legacy.json` — the chapter numbers the notebook had until
  23 October 2026, and the page each became;
- `GraphOfLifeRuns/` — the runs themselves, on the machine that ran them.

The programme the notebook follows, and what is already known, is in
`research/Research.md` — the *Findings* page under Research.
