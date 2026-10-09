# What this book is about

## The question behind it

Life on Earth has been producing new kinds of things for about four billion
years: new molecules, cells, bodies, ways of living together. It has not run
out of novelty, and nothing suggests it will. Biologists and computer
scientists call this property **open-ended evolution**. Many artificial
worlds have been built in which simple programs copy themselves, change and
compete — Tierra, Avida, Polyworld and others
([Chapter 6](06-the-ideas-this-builds-on.md)). They produce surprising things
for a while, and then settle: the novelty stops. No artificial world has yet
been shown, by an agreed test, to keep producing new kinds of things without
end.

This book follows one such world, **Graph of Life**, and asks what it does.
The long-term aim is to find out whether it — or a version of it with
changed rules — can be open-ended. But that question cannot be asked well
until the world itself is understood: what happens in it from one iteration
to the next, what its numbers mean, how much they depend on chance. Most of
this book is that groundwork.

There is a second aim, which shaped the rules from the start: that a world of
very simple, local, conserving rules might grow by itself the kinds of thing
physics describes — a space of a few dimensions, stable things that persist in
it, places where something can live. [Chapter 7](07-physical-inspiration.md)
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

[Chapter 2](02-the-world.md), [Chapter 3](03-one-iteration.md) and
[Chapter 4](04-the-brain.md) explain every rule exactly — to the point where
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
([How a brain changes](../notes/mutation.md)). Whether it has the third is a
question, not a given: brains that keep their tokens and take over
neighbouring nodes spread, and the others vanish — but is that because of
*how they differ*, or by chance? Nothing in the rules says which brains are
good. [Chapter 37](37-do-the-brains-matter.md) asks it directly.

## What would count as open-ended

There is no single agreed definition ([Chapter 6](06-the-ideas-this-builds-on.md)
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

Parts II to IV answer questions that lie underneath P1. Part V, not yet
written, will climb the ladder, by the changes to the rules that
[Meta II](36-meta-2.md) proposes.

## How the book is organised

- **Part I · How the algorithm works.** The world
  ([Chapter 2](02-the-world.md)), one iteration step by step
  ([Chapter 3](03-one-iteration.md)), the brain ([Chapter 4](04-the-brain.md)),
  how worlds are measured ([Chapter 5](05-how-worlds-are-measured.md)), the
  ideas this builds on ([Chapter 6](06-the-ideas-this-builds-on.md)), the
  physical ideas behind the rules ([Chapter 7](07-physical-inspiration.md)), and
  whether a run can be made again exactly ([Chapter 8](08-is-a-run-reproducible.md)).
- **Part II · The baseline world.** Thirty worlds of one fixed setting — the
  *baseline*, called B1 — looked at from every side: their life over 3,000
  iterations, their first hundred iterations, births and deaths, chance,
  wealth, the game, the shape of the network, lineages, what the brains are
  like and what agents say to each other, whether the rich stay rich, how
  the network grows, how far likeness reaches, and how many dimensions a
  world has ([Chapters 9 to 24](09-thirty-worlds.md)).
- **Part III · The baseline world, measured every way.** The same thirty
  worlds again, through every statistic the Graph of Life viewer offers and
  some it does not: entropy, gains and losses, the flow of tokens, how agents
  have children, power laws, scaling, the geometry of the network, how it
  breaks, pictures of one world in many colours, whether agents cooperate —
  then a chapter that questions the rules themselves, and a meta chapter on
  what it all means for open-ended evolution, with proposals for changing the
  rules ([Chapters 25 to 36](25-how-even-is-a-world.md)).
- **Part IV · One change at a time.** One setting changed against the
  baseline: worlds whose brains never change or are never asked
  ([Chapter 37](37-do-the-brains-matter.md)), worlds of other sizes
  ([Chapter 38](38-how-does-a-worlds-size-follow-its-tokens.md)), and more.
- **Part V · Towards open-ended evolution.** Changed rules, aimed at the
  ladder above, in the order [Meta II](36-meta-2.md) sets out.
- **Part VI · Towards a physics.** The second aim: worlds ten times the size
  ([Chapter 51](51-worlds-of-100000-tokens.md)), how many dimensions a world
  has at scale, which of its properties have no scale, what changes when
  every rule is local, how far a difference travels, what forms when a flow
  runs through a world, and whether anything like a particle appears
  ([Chapter 7](07-physical-inspiration.md) sets out the questions).
- **Notes.** Short notes, one per idea, that define every rule, every
  measurement and every statistical method the chapters use, with worked
  examples: [Every setting](../notes/settings.md),
  [The share function](../notes/share-function.md),
  [The Gini coefficient](../notes/gini-coefficient.md),
  [Survival curves](../notes/kaplan-meier.md) and some sixty more. The
  chapters link to them wherever they are needed;
  [the contents](../README.md) lists them all.

Many chapters only **ask**: what does a world do, how often, how fast. The
chapters that report an **experiment** also show a prediction — a
*thesis* — written down before the runs were made, in the experiment's plan,
together with what would confirm or refute it. It is quoted as it was
written, so that it cannot be quietly adjusted to the result.

## How to read a chapter, and how to make it again

Everything in this book can be made again on any computer with Python 3,
numpy and networkx. Each chapter that reads runs shows:

- **the runs behind it** — a box that says which runs it reads, how many,
  with which seeds and for how many iterations, and the command that makes
  them; folded beneath it, **every setting** of those runs;
- under every **figure**, what it shows — what each line, band, bar or dot
  is — and, folded away, **how to make this figure**: which runs, which file,
  which fields, every step of the calculation, and the one command that draws
  it again from the runs.

Three commands do all of it. `python3 gol_lab.py run E02` makes the runs of an
experiment — here Experiment 2, the thirty baseline worlds — into the folder
`GraphOfLifeRuns/`. `python3 gol_lab.py analyse E02` computes an experiment's
results. And `python3 book_figures.py life` draws the figures of one chapter
— here [A world's life](10-a-worlds-life.md) — from the runs, and writes them
into the chapters. On the same kind of computer, with the same versions of
the libraries, every run, figure and number comes out identical; [Chapter 8](08-is-a-run-reproducible.md)
shows why.

The book is plain Markdown and SVG pictures. It reads the same in the Book
tab of the Graph of Life app, on GitHub, and in **Obsidian**: open the folder
`book/` as a vault, and every link, figure and formula works.

## Words used throughout

- **world**, **run** — one simulation, from its founders to its end.
- **seed** — the number that fixes every random choice a run makes
  ([Seeds and the random stream](../notes/random-numbers.md)).
- **iteration** *t* — one round: a reproduction phase, then a game.
- **phase 1**, **phase 2** — the reproduction phase and the game of an
  iteration.
- **frame** — a snapshot of the whole world, recorded after every phase:
  frame 2*t* after the reproduction of iteration *t*, frame 2*t* + 1 after its
  game ([What a run records](../notes/frames-and-stats.md)).
- **agent** — the occupant of a node: its tokens and its brain.
- **candidates** — the agent itself and its neighbours: everyone an agent
  looks at and can act on.
- **genotype** — a brain's identity: copying a brain keeps it, changing a
  brain gives a new one ([Genotypes](../notes/genotype.md)).
- **B1** — the baseline: the settings every experiment starts from
  ([Every setting](../notes/settings.md)).
- **median**, **quartiles** — the middle value of a set, and the values a
  quarter and three quarters of the way up it
  ([Median and quantiles](../notes/median-and-quantiles.md)).
- **settled life** — a world's iterations from 500 to its end, over which the
  book measures where it settles ([The settled life](../notes/settled-life.md)).

<!-- turns -->
---

[Contents](../README.md) · [Chapter 2 · The world: tokens, agents and a network](02-the-world.md) →
<!-- /turns -->
