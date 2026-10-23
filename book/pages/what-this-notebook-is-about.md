# What this notebook is about

## The question behind it

Life on Earth has been producing new kinds of things for about four billion
years: new molecules, cells, bodies, ways of living together. It has not run
out of novelty, and nothing suggests it will. Biologists and computer
scientists call this property **open-ended evolution**. Many artificial
worlds have been built in which simple programs copy themselves, change and
compete — Tierra, Avida, Polyworld and others
([The ideas this builds on](the-ideas-this-builds-on.md)). They produce surprising things
for a while, and then settle: the novelty stops. No artificial world has yet
been shown, by an agreed test, to keep producing new kinds of things without
end.

This notebook follows one such world, **Graph of Life**, and asks what it does.
The long-term aim is to find out whether it — or a version of it with
changed rules — can be open-ended. But that question cannot be asked well
until the world itself is understood: what happens in it from one iteration
to the next, what its numbers mean, how much they depend on chance. Most of
this notebook is that groundwork.

There is a second aim, which shaped the rules from the start: that a world of
very simple, local, conserving rules might grow by itself the kinds of thing
physics describes — a space of a few dimensions, stable things that persist in
it, places where something can live. [Physical inspiration](physical-inspiration.md)
explains the physical ideas behind the rules, and why the two aims keep
meeting.

## The world in one paragraph

A Graph of Life world is a **network**: dots, called *nodes*, joined by
lines, called *connections*. On every node lives an **agent**. An agent owns
two things: a whole number of **tokens**, and a **brain** — a small neural
network that takes every decision the agent makes. The total number of
tokens in the world never changes. In every **iteration** each agent may give
some of its tokens to a child, which appears on a new node next to it and
inherits a copy of its brain, sometimes slightly changed. Then every agent
stakes all of its tokens on itself and its neighbours, and every node goes to
whoever staked most on it — or to a coalition of smaller stakers. The
winner's brain is copied into the node, and the node keeps every token staked
on it. Agents left without tokens die, and so does every part of the network
cut off from the rest. Nothing is scored and nothing is optimised: whatever
keeps its tokens and its place is what there is in the next iteration.

[The world: tokens, agents and a network](the-world.md), [One iteration, step by step](one-iteration.md) and
[The brain](the-brain.md) explain every rule exactly — to the point where
you could write the program yourself.

## What "evolution" means here

Evolution, in the sense biologists use, needs three things:

- **heredity** — offspring resemble their parents;
- **variation** — they do not resemble them exactly;
- **selection** — some variants leave more descendants than others, because
  of how they differ.

Graph of Life has the first two by construction. A child's brain is a copy
of its parent's (heredity); a copy changes with probability one in five, and
every brain changes with the same probability after every game
([How a brain changes](mutation.md)). Whether it has the third is a
question, not a given: brains that keep their tokens and take over
neighbouring nodes spread, and the others vanish — but is that because of
*how they differ*, or by chance? Nothing in the rules says which brains are
good. [Do the brains matter?](do-the-brains-matter.md) asks it directly.

## What would count as open-ended

There is no single agreed definition ([The ideas this builds on](the-ideas-this-builds-on.md)
lists the main ones). This project uses a ladder of four claims, each needed
before the next:

1. **P1 — there is evolution:** lineages — a brain and its descendants — last
   long enough for selection to act on them.
2. **P2 — adaptation accumulates:** a late population would beat its own
   ancestors, if they were brought back and made to compete.
3. **P3 — there are organisations:** groups of agents that persist while
   their members come and go.
4. **P4 — the space of organisations keeps growing:** new kinds keep
   appearing, without a ceiling.

The first pages answer questions that lie underneath P1: what the baseline
world does, and what one changed setting does to it. Then the notebook turns
from watching the rules to building them: a yardstick for evolution that keeps
going and a health bar for the network
([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)), then rules
meant to pass them, starting from the obstacles [What the measurements say about open-ended evolution](what-the-measurements-say-about-open-ended-evolution.md)
lists.

> [!info] Added later: selection, and the second rung
> [Do the brains matter?](do-the-brains-matter.md) found that the brains matter and
> that evolution makes worlds alike, and
> [What the brains are like](what-the-brains-are-like.md) that selection did shape the
> brains — but only their constant tendencies, since they are nearly blind
> to what they see.
> [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md) ran the test
> of P2, playing the brains of each era against earlier ones: the baseline's
> brains beat their founders by iteration 50 (+361 Elo in the median world)
> and gain nothing after iteration 100 — those of iteration 1,000 score
> 0.483 against those of iteration 300.

## How the notebook is organised

The pages are sorted by topic, and [the contents](../README.md#contents) list
every one of them.

- **How the simulation works.** The world
  ([The world: tokens, agents and a network](the-world.md)), one iteration step by
  step ([One iteration, step by step](one-iteration.md)), the brain
  ([The brain](the-brain.md)), every setting ([Every setting](settings.md)), and
  whether a run can be made again exactly
  ([Is a run reproducible?](is-a-run-reproducible.md)).
- **The baselines B1 and B2.** Thirty worlds of one fixed setting — the
  *baseline*, called B1, at 100,000 tokens, about 14,000 agents each — looked
  at from every side ([Thirty worlds](thirty-worlds.md),
  [What the baseline world is](what-the-baseline-world-is.md)); and B2, a second
  baseline built so that its network cannot break
  ([A world that cannot break](a-world-that-cannot-break.md)).
- **Agents' lives, wealth, the game and the flow of tokens, the shape of the
  network, the size of a world, and brains.** The baseline measured every way,
  through every statistic the Graph of Life viewer offers and some it does
  not, and the experiments that change one setting at a time
  ([Do the brains matter?](do-the-brains-matter.md),
  [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)).
- **Does evolution keep going?** A yardstick for a world that keeps evolving
  and a health bar for its network
  ([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)),
  why evolution stops ([Why evolution stops](why-evolution-stops.md)), and one
  world followed for a hundred thousand iterations
  ([A hundred thousand iterations](a-hundred-thousand-iterations.md)).
- **Changing the rules, and looking for a healthy, rich world.** The rules
  built to pass that yardstick, alone and in company
  ([Rules in company](rules-in-company.md)), and the search for a world that is
  healthy and whose agents differ ([Healthy and rich](healthy-and-rich.md),
  [A screen of random worlds](a-screen-of-random-worlds.md)).
- **Strategies written by hand, and groups.** Whether evolution's play is the
  best there is ([Strategies written by hand](strategies-written-by-hand.md)),
  and whether agents work together ([Who helps whom](who-helps-whom.md)).
- **Space and dimension.** The second aim, a physics: whether a world is a
  space, of how many dimensions, what folds it, what changes when every rule
  is local, and whether anything like a particle appears
  ([Physical inspiration](physical-inspiration.md) sets out the questions).
- **Methods and statistics.** Short pages, one per idea, that define every
  measurement and every statistical method the notebook uses, with worked
  examples — [The Gini coefficient](gini-coefficient.md),
  [Survival curves](kaplan-meier.md) and many more — and what a simulation
  costs ([What a simulation costs](what-a-simulation-costs.md)).

Many pages only **ask**: what does a world do, how often, how fast. The
pages that report an **experiment** also show a prediction — a
*thesis* — written down before the runs were made, in the experiment's plan,
together with what would confirm or refute it. It is quoted as it was
written, so that it cannot be quietly adjusted to the result.

## How to read a page, and how to make it again

Everything in this notebook can be made again on any computer with Python 3,
numpy and networkx. Each page that reads runs shows:

- **the runs behind it** — a box that says which runs it reads, how many,
  with which seeds and for how many iterations, and the command that makes
  them; folded beneath it, **every setting** of those runs;
- under every **figure**, what it shows — what each line, band, bar or dot
  is — and, folded away, **how to make this figure**: which runs, which file,
  which fields, every step of the calculation, and the one command that draws
  it again from the runs.

Three commands do all of it. `python3 gol_lab.py run E10` makes the runs of an
experiment — here Experiment 10, the thirty baseline worlds of 100,000 tokens
— into the folder `GraphOfLifeRuns/`.
`python3 gol_lab.py analyse E10` computes an experiment's results. And
`python3 book_figures.py life` draws the figures of one page
— here [A world's life](a-worlds-life.md) — from the runs, and writes them
into the pages. On the same kind of computer, with the same versions of
the libraries, every run, figure and number comes out identical; [Is a run reproducible?](is-a-run-reproducible.md)
shows why.

The notebook is plain Markdown and SVG pictures. It reads the same in the Book
tab of the Graph of Life app, on GitHub, and in **Obsidian**: open the folder
`book/` as a vault, and every link, figure and formula works.

## Words used throughout

- **world**, **run** — one simulation, from its founders to its end.
- **seed** — the number that fixes every random choice a run makes
  ([Seeds and the random stream](random-numbers.md)).
- **iteration** *t* — one round: a reproduction phase, then a game.
- **phase 1**, **phase 2** — the reproduction phase and the game of an
  iteration.
- **frame** — a snapshot of the whole world, recorded after every phase:
  frame 2*t* after the reproduction of iteration *t*, frame 2*t* + 1 after its
  game ([What a run records](frames-and-stats.md)).
- **agent** — the occupant of a node: its tokens and its brain.
- **candidates** — the agent itself and its neighbours: everyone an agent
  looks at and can act on.
- **genotype** — a brain's identity: copying a brain keeps it, changing a
  brain gives a new one ([Genotypes](genotype.md)).
- **B1** — the baseline: the settings every experiment starts from
  ([Every setting](settings.md)).
- **median**, **quartiles** — the middle value of a set, and the values a
  quarter and three quarters of the way up it
  ([Median and quantiles](median-and-quantiles.md)).
- **settled life** — a world's iterations from 500 to its end, over which the
  notebook measures where it settles ([The settled life](settled-life.md)).

<!-- turns -->
---

← [How to read this notebook](../README.md) · [Start here](../README.md#start-here) · [The ideas this builds on](the-ideas-this-builds-on.md) →
<!-- /turns -->
