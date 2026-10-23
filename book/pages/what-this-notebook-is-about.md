# What this notebook is about

> [!question] Questions of this page
> - What question does this notebook ask, and why does it matter?
> - What is a Graph of Life world, in one paragraph?
> - What would count as an answer?
> - How do you read a page, and how do you check it?

## In short

Living things have been inventing new kinds of things for about four billion
years, and they have not stopped. No computer world has yet been shown to do
the same. This notebook follows one computer world, **Graph of Life**. It asks
what the world does, and whether it — or a version of it with changed rules —
can keep inventing without end. Biologists call that **[open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau.")**.
A second question runs alongside: can rules this simple grow something like
physical space? Every result comes with the worlds it was measured on, and the
command that makes it again.

## The question behind it

Life on Earth has produced new molecules, new cells, new bodies and new ways
of living together, one after another, for about four billion years. It has
not run out of new things, and nothing suggests it will.

People have built many artificial worlds in which simple programs copy
themselves, change and compete: Tierra, Avida, Polyworld and others
([The ideas this builds on](the-ideas-this-builds-on.md)). Each produces
surprising things for a while. Then it settles, and the novelty stops. No
artificial world has yet been shown, by an agreed test, to keep producing new
kinds of things without end.

This notebook follows one such world and asks what it does. The long-term aim
is to learn whether it can be open-ended. But that question cannot be asked
well until the world itself is understood. What happens in it from one
iteration to the next? What do its numbers mean? How much do they depend on
chance? Much of the notebook is that groundwork.

There is a second aim, which shaped the rules from the start. The rules are
simple and local, and they never create or destroy anything. Rules like that
might grow, by themselves, the kinds of thing physics describes: a space of a
few [dimensions](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space."), things that last in it, places where something can live.
[Physical inspiration](physical-inspiration.md) explains the physical ideas
behind the rules, and why the two aims keep meeting.

## The world in one paragraph

A Graph of Life world is a **network**: dots, called *nodes*, joined by lines,
called *connections*. On every node lives an **agent**. An agent owns two
things: a whole number of **tokens**, and a **brain**, a small neural network
that takes every decision the agent makes. The total number of tokens in the
world never changes. In every **iteration**, each agent may give some of its
tokens to a child. The child appears on a new node next to it, with a copy of
its brain, sometimes slightly changed. Then every agent stakes all of its
tokens on itself and its neighbours. Every node goes to whoever staked most on
it, or to a [coalition](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.") of smaller stakers. The winner's brain is copied into the
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
every game ([How a brain changes](mutation.md)).

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

1. **P1: there is evolution.** [Lineages](families.md "A line of descent: the agents whose genotypes go back to one ancestor.") — a brain and its descendants — last
   long enough for selection to act on them.
2. **P2: adaptation accumulates.** A late population would beat its own
   ancestors, if they were brought back and made to compete.
3. **P3: there are organisations.** Groups of agents last while their members
   come and go.
4. **P4: the space of organisations keeps growing.** New kinds keep appearing,
   without a ceiling.

The second step can be tested directly. Play the brains of a world at one
moment against the brains of the same world at an earlier one
([Eras and contests](eras-and-contests.md)). In the first baseline, the brains
beat their [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") by iteration 50, by 361 Elo points in the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world.
After iteration 100 they gain nothing: those of iteration 1,000 score 0.483
against those of iteration 300, a tie
([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)).

So the notebook turned from watching the rules to building them. It made a
yardstick for evolution that keeps going, and a [health bar](health-bar.md) for
the network. Then it tried rules meant to pass both, starting from the
obstacles that
[What the measurements say about open-ended evolution](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution)
lists.

## How the notebook is organised

The pages are sorted by topic; [the contents](../README.md#contents) list every
one of them.

- **How the simulation works.** The world, one iteration step by step, the
  brain, every [setting](settings.md "One number or switch of a run's rules, fixed before it starts."), and whether a run can be made again exactly
  ([Is a run reproducible?](is-a-run-reproducible.md)).
- **The baselines B1 and B2.** The two sets of rules every experiment starts
  from ([B1 and B2: the two baselines](b1-and-b2.md)). Thirty worlds of the
  first, about 14,000 agents each, are looked at from every side
  ([Thirty worlds](a-worlds-life.md#thirty-worlds)). The second is built so that its network
  cannot break ([A world that cannot break](a-world-that-cannot-break.md)).
- **Agents' lives, wealth, the game and the flow of tokens, the shape of the
  network, the size of a world, and brains.** The baseline measured every way,
  through every statistic the Graph of Life viewer offers and some it does
  not. Experiments change one setting at a time
  ([Do the brains matter?](do-the-brains-matter.md),
  [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)).
- **Does evolution keep going?** The yardstick
  ([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)),
  why evolution stops ([Why evolution stops](why-evolution-stops.md)), and one
  world planned for a hundred thousand iterations
  ([A hundred thousand iterations](a-hundred-thousand-iterations.md)).
- **Changing the rules, and looking for a healthy, rich world.** The rules
  built to pass the yardstick, alone and in company
  ([Rules in company](rules-in-company.md)). The search for a world that is
  healthy and whose agents differ ([Healthy and rich](healthy-and-rich.md),
  [A screen of random worlds](a-screen-of-random-worlds.md)).
- **Strategies written by hand, and groups.** Is evolution's play the best
  there is ([Strategies written by hand](strategies-written-by-hand.md))? Do
  agents work together ([Who helps whom](who-helps-whom.md))?
- **Space and dimension.** The second aim: is a world a space, of how many
  dimensions, what folds it, and does anything like a particle appear?
  [Physical inspiration](physical-inspiration.md) sets out the questions.
- **Methods and statistics.** Short pages, one per idea. They define every
  measurement and statistical method the notebook uses, with worked examples:
  [The Gini coefficient and the Lorenz curve](gini-coefficient.md),
  [Survival curves (Kaplan–Meier)](kaplan-meier.md) and many more. And what a
  simulation costs ([What a simulation costs](what-a-simulation-costs.md)).

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
  [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") and for how many iterations, and the command that makes them. Folded
  beneath it is every setting of those runs.
- under every **figure**, what it shows: what each line, band, bar or dot is.
  Folded away is **how to make this figure**: which runs, which file, which
  fields, every step of the calculation, and the command that draws it again.

Three commands do all of it. `python3 gol_lab.py run E10` makes the runs of an
experiment into the folder `GraphOfLifeRuns/` — here Experiment 10, the thirty
baseline worlds of 100,000 tokens. `python3 gol_lab.py analyse E10` computes an
experiment's results. And `python3 book_figures.py life` draws the figures of
one page from the runs, and writes them into the pages — here
[A world's life](a-worlds-life.md). On the same kind of computer, with the same
versions of the libraries, every run, figure and number comes out identical.
[Is a run reproducible?](is-a-run-reproducible.md) shows why, and
[The lab, the engine and pilots](the-lab.md) says how experiments are run.

The notebook is plain Markdown and SVG pictures. It reads the same in the
Research Notebook tab of the Graph of Life app, on GitHub, and in **Obsidian**.
In Obsidian, open the folder `book/` as a vault, and every link, figure and
formula works.

## Where to go next

- [The world: tokens, agents and a network](the-world.md): the rules, from the
  beginning.
- [B1 and B2: the two baselines](b1-and-b2.md): the two worlds every experiment
  starts from.
- [Glossary](glossary.md): every term in one line.
- [How this notebook is written](how-this-notebook-is-written.md): the rules of
  the research, and of its pages.

<!-- turns -->
---

← [Welcome](../README.md) · [Start here](../README.md#start-here) · [Glossary](glossary.md) →

> [!info]- Pages that link here (11)
> [Welcome](../README.md) · [How this notebook is written](how-this-notebook-is-written.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [How a brain changes](mutation.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Physical inspiration](physical-inspiration.md) · [How alike the agents are](how-alike-the-agents-are.md) · [Genotypes and lineages](genotypes-and-lineages.md) · [How fast should brains change?](how-fast-should-brains-change.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md)
<!-- /turns -->
