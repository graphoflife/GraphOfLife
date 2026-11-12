# Graph of Life — the Research Notebook

This notebook is a research programme, written down while it is carried out.
It asks one question: does any set of Graph of Life's rules make worlds that
pass the notebook's test of *open-ended evolution* over 100,000 [iterations](pages/one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.")?
Open-ended evolution is evolution that keeps producing new things without end,
and [What would count as open-ended evolution](pages/measuring-evolution.md#what-would-count-as-open-ended-evolution)
writes the test down.

- **What it asks now.** In every world played so far, evolution levels off.
  Why? The candidates are drift, valleys that need a crowd to cross, a fixed
  space of strategies, and a closed, zero-sum [game](pages/one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.").
- **How worlds are chosen.** By how they evolve first, against null models.
  Health only rules out worlds that are dead, a [star](pages/measuring-networks.md#stars-and-hubs "An agent joined to far more agents than most; a star is a world held together by a single hub."), uniform or frozen.
- **When it changes course.** If the experiment on why evolution levels off
  finds no arm that delays or lifts the plateau, the next step changes how a
  world is built, with an open economy or [brains](pages/the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") that can grow, and not its
  [settings](pages/settings.md "One number or switch of a run's rules, fixed before it starts.").
- **A second aim.** Whether such rules grow a space of their own, with
  particles in it, is a separate track with its own question, in its own
  part.

Start anywhere. Every page says what it asks, and every word that needs
explaining links to the page that explains it. If you want a beginning, read
[What this notebook is about](pages/what-this-notebook-is-about.md), then the
pages on [how the simulation works](#how-the-simulation-works).

- [What this notebook is about](pages/what-this-notebook-is-about.md): the
  question, why it is asked, and how to read and check a page.
- [Candidates for open-ended evolution](pages/candidates-for-open-ended-evolution.md):
  the worlds of round 3 that might keep evolving.
- [Glossary](pages/glossary.md): every term in one line.
- [How this notebook is written](pages/how-this-notebook-is-written.md): the
  rules of the research, and of its pages.
- [The lab, the engine and pilots](pages/runs-and-records.md#the-lab-the-engine-and-pilots): how an experiment is run
  and made again.

The pages fall into eight parts, each with its colour: where to start; the
research of the moment, round 3 of the screen of random worlds and its
candidates for open-ended evolution; the rules; what the worlds do; space and
physics; the search for open-ended evolution; methods and look-up, where every
rule and measure is explained in detail; and the Archive. The Archive keeps
frozen pages, what was done and found before 11 November 2026, so that every
experiment can still be read and made again; it starts shut. In the Research
Notebook tab each part opens and shuts, every page wears its part's colour, and
a mark says what kind of page it is: a study, an experiment, a plan not played
yet, a reference page or a guide. The contents can also be sorted by date, by
title or by kind, filtered, and searched. In **Obsidian**, open the folder
`book/` as a vault: every link, figure and formula works there, and the graph
view shows how the pages hang together.

## Contents

<!-- contents -->
### Start

⬜ What the notebook asks, how to read it, and the ideas it builds on.

#### Start here

- [What this notebook is about](pages/what-this-notebook-is-about.md)
- [How this notebook is written](pages/how-this-notebook-is-written.md)
- [The ideas this builds on](pages/the-ideas-this-builds-on.md)

### Now: round 3 and its candidates

🟨 The research of the moment: round 3 of the screen of random worlds, its candidates for open-ended evolution, whether they evolved, and what is played next.

#### Round 3 of the random screen

- [A screen of random worlds](pages/a-screen-of-random-worlds.md)
- [Which worlds are healthy? Three orders](pages/which-worlds-are-healthy.md)
- [How the drawn settings shape a world](pages/how-the-drawn-settings-shape-a-world.md)
- [Candidates for open-ended evolution](pages/candidates-for-open-ended-evolution.md)
- [Did evolution happen? Fourteen ways to find out](pages/did-evolution-happen.md)
- [Did the brains get better? Round 3's eras against each other](pages/round-3-eras-against-each-other.md)
- [A hundred thousand iterations](pages/a-hundred-thousand-iterations.md)

### The rules

🟦 How a world works: tokens, agents and the network, one iteration, the brain, every setting, and the two baselines.

#### How the simulation works

- [The world: tokens, agents and a network](pages/the-world.md)
- [One iteration, step by step](pages/one-iteration.md)
- [The brain](pages/the-brain.md)
- [Questioning the mechanics](pages/questioning-the-mechanics.md)
- [Is a run reproducible?](pages/is-a-run-reproducible.md)
- [Every setting](pages/settings.md)

#### The baselines B1 and B2

- [B1 and B2: the two baselines](pages/b1-and-b2.md)

### What the worlds do

🟩 What the worlds do, where it still bears on the question: what the brains are and whether they matter, what keeps wealth spread, and other contests. What B1's worlds did is in the Archive.

#### Wealth: who holds the tokens

- [What keeps wealth spread?](pages/what-keeps-wealth-spread.md)

#### Brains, and how alike agents are

- [What the brains are like](pages/what-the-brains-are-like.md)
- [Do the brains matter?](pages/do-the-brains-matter.md)
- [Brains replaced by their average](pages/brains-replaced-by-their-average.md)
- [How big should a brain be?](pages/how-big-should-a-brain-be.md)

#### Groups and cooperation

- [Other contests: a lottery, and alliances](pages/another-game.md)

### Space and physics

🟪 The second aim, a track with its own question: does a world grow a space of its own, with locality and particles in it?

#### Space and dimension

- [Physical inspiration](pages/physical-inspiration.md)
- [Is a world a space? Dimension, and what folds it](pages/how-many-dimensions-does-a-world-have.md)
- [Local rules only](pages/local-rules-only.md)
- [How far does a difference travel?](pages/how-far-does-a-difference-travel.md)
- [Particles: patterns that outlast what carries them](pages/particles.md)

### The search for open-ended evolution

🟧 How to tell whether evolution goes on and why it stops, strategies written by hand, rules without tuned numbers, and the open economy planned for when no setting lifts the plateau.

#### Does evolution keep going?

- [Does evolution keep going?](pages/how-to-tell-a-world-that-keeps-evolving.md)
- [How fast should brains change?](pages/how-fast-should-brains-change.md)
- [Every era against three](pages/every-era-against-three.md)
- [A source and a sink](pages/a-source-and-a-sink.md)

#### Changing the rules

- [Rules without numbers](pages/rules-without-numbers.md)

#### Strategies written by hand

- [Can a strategy written by hand beat evolution?](pages/strategies-written-by-hand.md)
- [What a brain can compute](pages/what-a-brain-can-compute.md)
- [One agent in a sea of average agents](pages/one-agent-in-a-sea-of-average-agents.md)

### Methods and look-up

🔳 How runs are made and measured, the statistics, every simulation, and the rules and measures in detail.

#### Methods and statistics

- [How worlds are measured](pages/how-worlds-are-measured.md)
- [Every simulation: what was run, and what was measured](pages/every-simulation.md)

#### Reference: the rules and measures in detail

- [Glossary](pages/glossary.md)
- [The rules in detail](pages/rules-in-detail.md)
- [Measuring lives: births, deaths, ages and estates](pages/measuring-lives.md)
- [Measuring tokens: inequality and flow](pages/measuring-tokens.md)
- [Measuring a network: connections, hubs, distances and cuts](pages/measuring-networks.md)
- [Measuring space: dimension and the random twin](pages/measuring-space.md)
- [Measuring evolution: genotypes, eras and the open-ended test](pages/measuring-evolution.md)
- [Measuring health: criteria, the health bar and diversity](pages/measuring-health.md)
- [How runs are made, recorded and costed](pages/runs-and-records.md)
- [Statistics: the tools of the notebook](pages/statistics.md)

### Archive

🗄️ Frozen pages, kept so that every experiment can still be read and made again: what was done and found before 11 November 2026. They are not brought up to date as the research moves on.

#### B1, as described in October 2026

- [What B1 is, and what stood in its way](pages/what-the-baseline-world-is.md)
- [B1: lives and wealth](pages/b1-lives-and-wealth.md)
- [B1: the game and its flows](pages/b1-the-game-and-its-flows.md)
- [B1: the network, size and scaling](pages/b1-the-network-size-and-scaling.md)
- [B1: brains, lineages and messages](pages/b1-brains-lineages-and-messages.md)

#### Building B2, the healthy and rich search, and the pilots

- [Building B2, and the search for a healthy, rich world](pages/building-b2-and-the-search-for-a-healthy-rich-world.md)
- [What the switches did in pilots (indicative)](pages/what-the-switches-did-in-pilots.md)

#### Side studies

- [Cooperation, help and gliders (side study)](pages/cooperation-help-and-gliders.md)
- [Space at scale (side study)](pages/space-at-scale.md)
- [Measures, atlases and assembly (side studies)](pages/measures-atlases-and-assembly.md)
<!-- /contents -->

## How our view changed

- **11 November 2026.** The front page now states the notebook's question as
  it stands: the open-ended test over 100,000 iterations, why evolution levels
  off, how worlds are chosen, when the notebook changes course, and the second
  aim. Until then it said that the notebook asks what the Graph of Life
  algorithm does, and whether a world of it can keep producing new things
  without end. Since this day the pages fall into eight parts, the last an
  Archive of frozen pages.
