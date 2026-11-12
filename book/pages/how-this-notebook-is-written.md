# How this notebook is written

> [!question] Questions of this page
> - How is a page of this notebook laid out, and why?
> - What does every page promise its reader?
> - How are mistakes corrected?

## In short

This notebook is written for a student who starts on any page. So every page
says what it asks in plain words, gives the picture before the precise
statement, and links every word that needs explaining to the page that
explains it. A result is never shown without the worlds it comes from and the
way to make it again. And when we learn that something we wrote was wrong, the
page says what we know now, and keeps what we thought before, with its date.
Worlds are chosen by how they evolve, each measure beside what chance alone
would make of it. And pages whose worlds no longer bear on the question are
not deleted but frozen, in an Archive, so that every experiment can still be
made again.

## A page, from top to bottom

1. **The questions.** A page opens with the questions it asks, in a box.
2. **In short.** Three to five sentences: what the page finds, for a reader who
   reads nothing else.
3. **The picture first.** Each idea starts with the intuition — an everyday
   comparison, a small example, a drawing — and only then the precise
   statement. A formula always has words around it that say what it means.
4. **One idea per paragraph.** Short sentences, in the active voice. "We" is
   the notebook and its author; "you" is the reader.
5. **Real numbers.** Every claim comes with a number from a run, its unit, and
   where it comes from: which worlds, which experiment, which page.
6. **What we learned.** A short list near the end.
7. **Where to go next.** Two to four pages that continue the thought.
8. **How our view changed.** If anything on the page was once stated
   differently, the earlier claim and its correction, with dates, at the end.

A **reference page** gathers the details of one family of rules or measures:
the rules in detail, lives, [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), networks, space, evolution, health,
statistics, and how runs are made. Each idea has a section of its own, as a
short page had before 9 November 2026: the idea made precise, an example with
real numbers, and **Where it is used**. That last is a list of the pages that
link to the section, made by `python3 book_fill.py`, so it cannot fall behind.
A link to an idea leads to its section.

## Parts, and what each page is

The pages fall into eight **parts**, each with its colour: Start; Now, the
research of the moment (round 3 of the random screen and its candidates for
[open-ended
evolution](measuring-evolution.md#what-would-count-as-open-ended-evolution));
The rules; What the worlds do; Space and physics; The search for open-ended
evolution; Methods and look-up; and the Archive, which starts shut. A part
holds a few **topics**, and a page belongs to the part of its first topic, its
home. In the Research Notebook tab a part opens and shuts, a page wears its
part's colour as a band above its title, and a mark beside it in the contents
says what it is: a **study** of worlds already played, an **experiment** with
its own runs, a **plan** that has not been played, a **reference** page, or a
**guide** to read first (the author, 9 November 2026: fewer pages, better
sorted, in colours).

## Words and links

- **Every term is explained or linked at its first use.** The notebook's terms
  are listed in `book/terms.json`; `python3 book_fill.py` links the first use
  of each on every page to the page that explains it, and the link's title
  says in one line what it is. A new term goes into that list.
- **Pages are named by their titles**, never by a number: a link reads like the
  page it leads to. Experiments keep their numbers, "Experiment 16", because
  the lab knows them by those.
- **Every plan says where its worlds start**: from B1 or B2
  ([B1 and B2: the two baselines](b1-and-b2.md)), or, in the screen of random
  worlds, from B1's rules changed as the draw says.

## Figures

Every figure has a title that says what it shows, a caption that says what to
look at — what each line, band, bar or dot is, and how the worlds in it differ
— and, folded under it, how to make it again: which runs, which file, which
fields, every step, and the command that draws it. The notebook's code writes
captions and recipes beside the numbers they quote, so the two cannot drift
apart.

## Experiments

An experiment's page says first, in plain words, what we expected and why.
Then comes its **thesis**, word for word as it was written down before any of
its runs existed: what it claims, why it would be so, what would confirm it and
what would refute it. A thesis is never reworded once the results are in. A
thesis written while the notebook still numbered its chapters keeps those
numbers, and each links to the page it meant that day.

The runs behind a page — how many, which seeds, and what each condition
changes in its baseline — are listed in a box that folds away, in a last section, **How this was made**, so that the
page reads as an argument and the details are there for whoever wants to make
it again.

## The rules of the research

- **Evolution first.** Worlds are chosen and ranked by how they evolve:
  whether later [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") beat earlier ones, whether selection beat luck, and
  whether new things appeared
  ([Did evolution happen?](did-evolution-happen.md)). Health only rules out
  worlds that are dead, a [star](measuring-networks.md#stars-and-hubs "An agent joined to far more agents than most; a star is a world held together by a single hub."), uniform or frozen
  ([Health as a filter](measuring-health.md#from-11-november-2026-health-as-a-filter)).
- **A stated start, and what changes.** A plan says the rules its worlds start
  from, and what it changes: one [setting](settings.md "One number or switch of a run's rules, fixed before it starts.") of a baseline, as the early
  experiments did, or every setting drawn at random, as the screen does
  ([A screen of random worlds](a-screen-of-random-worlds.md)).
- **New seeds for a find.** A world chosen as the best of many is lucky as
  well as good, so it counts only once new seeds have played it again.
- **The thesis comes first.** It is fixed in the plan before the runs start.
- **Enough seeds.** A difference seen in fewer than thirty seeds per
  condition is called *indicative*, not an effect
  ([How many seeds an experiment needs](statistics.md#how-many-seeds-an-experiment-needs)).
- **No number without its null.** A measurement is reported next to what it
  would be by chance.
- **Every run can be made again**, from its plan, seed and frozen engine
  ([Is a run reproducible?](is-a-run-reproducible.md)).
- **The dead are counted apart.** A world that dies out is an outcome of its
  own, not a world that ended up somewhere
  ([When a world ends](rules-in-detail.md#when-a-world-ends)).
- **A run is measured over its settled life**: from [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 500 to its end
  in the early experiments, from iteration 100 in round 3
  ([The settled life of a world](statistics.md#the-settled-life-of-a-world)).
- **Runs are shared.** A run is named for what it is — baseline, world size,
  what differs, seed — so an experiment that needs runs which already exist
  uses them.
- **No tuned constants in new rules.** A rule added to the engine carries no
  hand-set factor beyond the tokens and the mutation's parameters: what a
  world does should emerge. ([Rules without numbers](rules-without-numbers.md)
  shows what the numbers B2 was built with do.)
- **When settings run out, change the structure.** If no change of settings
  delays or lifts the plateau of evolution, the next step changes how a world
  is built, with an open economy or brains that can grow
  ([When the notebook changes course](what-this-notebook-is-about.md#when-the-notebook-changes-course)).

## Frozen pages and the Archive

Pages whose worlds or questions no longer bear on the notebook's question are
not deleted. Pages that tell one story are merged into one, and moved to the
**Archive**, the last part, which starts shut. An archived page is frozen: it
is not brought up to date as the research moves on, but every link, figure,
thesis and box of runs on it still works, so every experiment it reports can
still be read and made again. Its In short says what still matters, and its
How our view changed records the day it was archived.

A method an archived page explained that the notebook still uses moves to a
reference page ([Statistics](statistics.md)). A draft plan that was not chosen
keeps its questions, one paragraph and its thesis, and says where in git
history its full draft is. Nothing is lost: what a page no longer shows, git
keeps.

## Where things are

- `book/pages/` — every page, named by what it is about;
- `book/figures/`, `book/diagrams/` — every figure and diagram, as SVG;
- `book/results/` — every number a page quotes, written by the analysis;
- `book/experiments/` — the baselines and each experiment's plan;
- `book/book.json` — the parts with their colours, the topics, and every
  page with its topics, kind and summary, from which the contents are made;
- `book/terms.json` — the terms, linked where a page first uses them;
- `book/legacy.json` — the chapter numbers the notebook had until
  23 October 2026, and the page each became, and every page merged into
  another, with where it went;
- `GraphOfLifeRuns/` — the runs themselves, on the machine that ran them.

The programme the notebook follows, and what is already known, is in
`research/Research.md` — the *Findings* page under Research.

## How mistakes are corrected

Findings change. When a later page shows that an earlier one was wrong or only
half right, the earlier page is changed to say what we know now, and its
section **How our view changed** keeps the old claim, the new one, the date,
and the page that changed it. Nothing is quietly deleted.

## Where to go next

- [What this notebook is about](what-this-notebook-is-about.md): the question
  behind it all.
- [Glossary](glossary.md): every term in one line.
- [The lab, the engine and pilots](runs-and-records.md#the-lab-the-engine-and-pilots): how an experiment is run.

## How our view changed

- **11 November 2026.** The rules of the research said "one change at a
  time": an experiment changes one setting "of a stated baseline". That
  described the experiments of October, not the screen of random worlds,
  which draws every setting at once, nor round 3, which has no baseline. The
  rules now say what is practised: evolution first, with health only as a
  filter; a stated start; new seeds for a find; and a change of structure when
  settings run out. A section says how pages are frozen in the Archive, and
  the parts are eight.

<!-- turns -->
---

← [What this notebook is about](what-this-notebook-is-about.md) · [Start here](../README.md#start-here) · [The ideas this builds on](the-ideas-this-builds-on.md) →

**Also about:** [Methods and statistics](../README.md#methods-and-statistics).

> [!info]- Pages that link here (3)
> [Welcome](../README.md) · [What this notebook is about](what-this-notebook-is-about.md) · [Rules without numbers](rules-without-numbers.md)
<!-- /turns -->
