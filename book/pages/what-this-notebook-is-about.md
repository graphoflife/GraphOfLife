# What this notebook is about

> [!question] Questions of this page
> - What question does this notebook ask, and why does it matter?
> - What does it ask right now, and why?
> - How are the worlds it studies chosen, and when does it change course?
> - What is a Graph of Life world, in one paragraph?
> - How do you read a page, and how do you check it?

## In short

Living things have been inventing new kinds of things for about four billion
years, and they have not stopped. No computer world has yet been shown to do
the same. This notebook follows one computer world, **Graph of Life**, and
asks one question. Does any set of its rules make worlds that pass a test of
**open-ended evolution**, written down in advance, over 100,000 iterations?
So far evolution levels off in every world played, so the notebook now asks
why. Worlds are chosen by how they evolve, against what chance alone would
do. Health only rules out worlds that are dead, a [star](measuring-networks.md#stars-and-hubs "An agent joined to far more agents than most; a star is a world held together by a single hub."), uniform or frozen. If
no change of settings lifts the plateau, the next step changes how a world is
built. A second aim, whether such rules grow a space with particles in it, is
a track of its own.

## The question

Life on Earth has produced new molecules, new cells, new bodies and new ways
of living together, one after another, for about four billion years. It has
not run out of new things, and nothing suggests it will.

People have built many artificial worlds in which simple programs copy
themselves, change and compete: Tierra, Avida, Polyworld and others
([The ideas this builds on](the-ideas-this-builds-on.md)). Each produces
surprising things for a while. Then it settles, and the novelty stops. No
artificial world has yet been shown, by an agreed test, to keep producing new
kinds of things without end.

So the notebook asks one question. **Does any set of rules of this family
make worlds that pass the test of open-ended evolution over 100,000
iterations?** The family is Graph of Life's rules, with every setting and
switch its engine has ([Every setting](settings.md)). The test is
[What would count as open-ended evolution](measuring-evolution.md#what-would-count-as-open-ended-evolution).
It was written down before any long run existed, so that it cannot be bent to
fit one. A set of rules passes when at least two of three new worlds pass its four
parts:

1. their brains beat the best mixture of their whole past;
2. their ratings keep rising;
3. they keep doing new things, faster than mutation alone would;
4. they stay worlds in which much can happen.

A hundred thousand iterations is a length the notebook can play. Three of
round 3's worlds would take about two and a half days
([A hundred thousand iterations](a-hundred-thousand-iterations.md)).

## What it asks now: why does evolution level off?

In every world played so far, evolution levels off. B1's brains stop getting
better within 50 to 100 iterations
([Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md)).
B2's go on for thousands of iterations, more and more slowly. The two longest
worlds, played to 50,000 iterations, climbed for 20,000 to 25,000 and then
only wandered ([Every era against three](every-era-against-three.md)). Round
3's eras are to be played against each other the same way. That experiment's
thesis expects the same
([Did the brains get better?](round-3-eras-against-each-other.md)).

So the question of the moment is why. Four reasons are candidates:

- **Drift.** Few ancestors share a world's future. So luck, and not how
  well a brain plays, may decide what spreads
  ([Selection, or chance?](b1-brains-lineages-and-messages.md#selection-or-chance),
  [Did evolution happen?](did-evolution-happen.md)).
- **Valleys that need a crowd to cross.** Better play exists, but it pays
  only when many agents play it at once. A raider written by hand takes hold
  only if about a hundred start together
  ([How large a crowd a raider needs](strategies-written-by-hand.md#how-large-a-crowd-a-raider-needs)).
- **A fixed space of strategies.** A brain of a fixed shape can compute only
  so much. No brain of B2, of any size, can pass on news of a third agent
  ([What a brain can compute](what-a-brain-can-compute.md)).
- **A closed, zero-sum game.** The tokens are a fixed sum: what one agent
  gains, another loses. And the game spreads them towards an even rest
  ([A source and a sink](a-source-and-a-sink.md)).

## How worlds are chosen

Worlds are chosen by evolution first. A world counts for what its brains did:
whether later brains beat earlier ones, whether selection beat luck, and
whether new things appeared. Each is measured against a null model, which
says what chance alone would do
([Did evolution happen?](did-evolution-happen.md)).

Health no longer ranks worlds. Since 11 November 2026 it is frozen at the
version of 9 November. It only rules out worlds that are dead, a star, uniform
or frozen
([Health as a filter](measuring-health.md#from-11-november-2026-health-as-a-filter)).
And a world chosen as the best of many is lucky as well as good. So a choice
is tried again in new seeds
([The search's best world, in new seeds](building-b2-and-the-search-for-a-healthy-rich-world.md#the-searchs-best-world-in-new-seeds)).

## When the notebook changes course

The experiment on why evolution levels off tries several arms: changes meant
to delay or lift the plateau. If no arm does either, the notebook stops
turning settings. The next step changes the world's structure instead. One is
an open economy, in which tokens flow in at a source and leave everywhere
([A source and a sink](a-source-and-a-sink.md)). Another is brains that can
grow ([How big should a brain be?](how-big-should-a-brain-be.md)).

## The second aim: space and particles

There is a second aim, which shaped the rules from the start. The rules are
simple and local, and they never create or destroy anything. Rules like that
might grow, by themselves, the kinds of thing physics describes: a space of a
few [dimensions](measuring-space.md#three-ways-to-measure-a-networks-dimension "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space."), and things that last in it. This is a separate track with its
own question: does a world grow a space of its own, with locality and
particles in it? It has its own part, Space and physics.
[Physical inspiration](physical-inspiration.md) explains the physical ideas
behind the rules, and [Is a world a space?](how-many-dimensions-does-a-world-have.md)
what the worlds have shown so far.

## The world in one paragraph

A Graph of Life world is a **network**: dots, called *nodes*, joined by lines,
called *connections*. On every node lives an **agent**. An agent owns two
things: a whole number of **tokens**, and a **brain**, a small neural network
that takes every decision the agent makes. The total number of tokens in the
world never changes. In every **iteration**, each agent may give some of its
tokens to a child. The child appears on a new node next to it, with a copy of
its brain, sometimes slightly changed. Then every agent stakes all of its
tokens on itself and its neighbours. Every node goes to whoever staked most on
it, or to a coalition of smaller stakers. The winner's brain is copied into the
node, and the node keeps every token staked on it. Agents left without tokens
die, and so does every part of the network cut off from the rest. Nothing is
scored and nothing is optimised: whatever keeps its tokens and its place is
what there is in the next iteration.

Three pages explain every rule exactly, to the point where you could write the
program yourself: [The world: tokens, agents and a network](the-world.md),
[One iteration, step by step](one-iteration.md) and [The brain](the-brain.md).

## What "evolution" means here

In the sense biologists use, evolution needs three things:

- **heredity**: offspring resemble their parents;
- **variation**: they do not resemble them exactly;
- **selection**: some variants leave more descendants than others, *because*
  of how they differ.

Graph of Life has the first two by construction. A child's brain is a copy of
its parent's: heredity. A copy changes with probability one in five: variation.
In the first baseline, every brain also changes with that probability after
every game ([How a brain changes](rules-in-detail.md#how-a-brain-changes)).

Whether it has the third is a question, not a given. Brains that keep their
tokens and take over neighbouring nodes spread, and the others vanish. But is
that because of how they differ, or by chance? Nothing in the rules says which
brains are good.

The answer came later. The brains matter, and selection does shape them. But at
first it shapes only their constant tendencies, because the first baseline's
brains barely see what is around them
([Do the brains matter?](do-the-brains-matter.md),
[What the brains are like](what-the-brains-are-like.md)).

## A ladder of four steps

There is no single agreed definition of open-ended evolution
([The ideas this builds on](the-ideas-this-builds-on.md) lists the main ones).
This notebook climbs a ladder of four claims, each needed before the next:

1. **P1: there is evolution.** Lineages — a brain and its descendants — last
   long enough for selection to act on them.
2. **P2: adaptation accumulates.** A late population would beat its own
   ancestors, if they were brought back and made to compete.
3. **P3: there are organisations.** Groups of agents last while their members
   come and go.
4. **P4: the space of organisations keeps growing.** New kinds keep appearing,
   without a ceiling.

The second step can be tested directly. Play the brains of a world at one
moment against the brains of the same world at an earlier one
([Eras and contests](measuring-evolution.md#eras-contests-and-invasions-does-a-world-keep-getting-better)). In the first baseline, the brains
beat their founders by iteration 50, by 361 Elo points in the [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") world.
After iteration 100 they gain nothing: those of iteration 1,000 score 0.483
against those of iteration 300, a tie
([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)).

So the notebook turned from watching the rules to building them. It made a
yardstick for evolution that keeps going, and a [health bar](measuring-health.md#the-health-bar) for
the network. Then it tried rules meant to pass both, starting from the
obstacles that
[What the measurements say about open-ended evolution](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution)
lists. That search, and the study of B1 before it,
are now in the Archive; the notebook's question is the test above.

## How the notebook is organised

The pages fall into eight parts; [the contents](../README.md#contents) list
every one of them.

- **Start.** What the notebook asks, how to read it, and the ideas it builds
  on.
- **Now: round 3 and its candidates.** Round 3's hundred worlds drawn at
  random, and which of them might keep evolving
  ([Candidates for open-ended evolution](candidates-for-open-ended-evolution.md)).
- **The rules.** How a world works, step by step, every [setting](settings.md),
  and the two baselines ([B1 and B2: the two baselines](b1-and-b2.md)).
- **What the worlds do.** What the brains are and do, where wealth goes, and
  other contests.
- **Space and physics.** The second aim, with its own question.
- **The search for open-ended evolution.** How to tell whether evolution goes
  on, and why it stops. Strategies written by hand, rules without tuned
  numbers, and the plan for an open economy.
- **Methods and look-up.** The statistics, every rule and measure in detail,
  and every simulation made.
- **Archive.** Frozen pages, shut at first. They hold B1 as it was described
  in October 2026, how B2 was built, the search for a healthy, rich world with
  its pilots, and side studies. They are kept so that every experiment can still
  be read and made again.

Many pages only **ask**: what does a world do, how often, how fast? A page that
reports an **experiment** also shows a prediction, its **thesis**. The thesis
was written down in the experiment's plan before any of its runs existed,
together with what would confirm or refute it. It is quoted as it was written,
so that it cannot be quietly adjusted to the result.

## How to read a page, and how to make it again

Every word that needs explaining is a link where a page first uses it. Rest the
pointer on it to read what it means in one line, or open the
[Glossary](glossary.md) for all of them.

Everything in this notebook can be made again on any computer with Python 3,
numpy and networkx. Each page that reads runs shows two things:

- **the runs behind it**, in a box: which runs it reads, how many, with which
  seeds and for how many iterations, what each condition changes in its
  baseline, and the command that makes them.
- under every **figure**, what it shows: what each line, band, bar or dot is.
  Folded away is **how to make this figure**: which runs, which file, which
  fields, every step of the calculation, and the command that draws it again.

Three commands do all of it. `python3 gol_lab.py run E10` makes the runs of an
experiment into the folder `GraphOfLifeRuns/` — here Experiment 10, the thirty
baseline worlds of 100,000 tokens. `python3 gol_lab.py analyse E10` computes an
experiment's results. And `python3 book_figures.py life` draws the figures of
one page from the runs, and writes them into the pages — here
[A world's life](b1-lives-and-wealth.md#the-life-of-a-b1-world). On the same kind of computer, with the same
versions of the libraries, every run, figure and number comes out identical.
[Is a run reproducible?](is-a-run-reproducible.md) shows why, and
[The lab, the engine and pilots](runs-and-records.md#the-lab-the-engine-and-pilots) says how experiments are run.

The notebook is plain Markdown and SVG pictures. It reads the same in the
Research Notebook tab of the Graph of Life app, on GitHub, and in **Obsidian**.
In Obsidian, open the folder `book/` as a vault, and every link, figure and
formula works.

## Where to go next

- [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md):
  the worlds of round 3 that might keep evolving.
- [What would count as open-ended evolution](measuring-evolution.md#what-would-count-as-open-ended-evolution):
  the test the question asks of every world.
- [The world: tokens, agents and a network](the-world.md): the rules, from the
  beginning.
- [How this notebook is written](how-this-notebook-is-written.md): the rules of
  the research, and of its pages.

## How our view changed

- **8 October 2026.** This page was first written as the notebook's opening.
  It asked what the world does, and whether it, or a version of it with
  changed rules, can keep inventing without end. It called much of the
  notebook groundwork, to understand the world before asking that question
  well.
- **11 November 2026.** Rewritten around the question as it now stands, after
  a review of the whole notebook that the owner approved. The question is the
  open-ended test over 100,000 iterations. The question of the moment is why
  evolution levels off. Worlds are chosen by evolution against null models,
  with health only as a filter. And a stop rule says when the notebook changes
  a world's structure instead of its settings. The second aim is a track of
  its own. The groundwork the page once announced, the study of B1 and the
  search for a healthy world, is now in the Archive.

<!-- turns -->
---

← [Welcome](../README.md) · [Start here](../README.md#start-here) · [How this notebook is written](how-this-notebook-is-written.md) →

> [!info]- Pages that link here (11)
> [Welcome](../README.md) · [How this notebook is written](how-this-notebook-is-written.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [The rules in detail](rules-in-detail.md) · [Physical inspiration](physical-inspiration.md) · [Particles: patterns that outlast what carries them](particles.md) · [How fast should brains change?](how-fast-should-brains-change.md) · [Measuring health: criteria, the health bar and diversity](measuring-health.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md) · [B1: brains, lineages and messages](b1-brains-lineages-and-messages.md)
<!-- /turns -->
