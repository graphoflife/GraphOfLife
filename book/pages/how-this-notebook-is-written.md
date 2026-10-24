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

A **term page** explains one idea, so it is shorter: *In short*, the idea
made precise, an example with real numbers, **Where it is used** and **Where
to go next**. Its questions are its title.

## Words and links

- **Every term is explained or linked at its first use.** The notebook's terms
  are listed in `book/terms.json`; `python3 book_fill.py` links the first use
  of each on every page to the page that explains it, and the link's title
  says in one line what it is. A new term goes into that list.
- **Pages are named by their titles**, never by a number: a link reads like the
  page it leads to. Experiments keep their numbers, "Experiment 16", because
  the lab knows them by those.
- **The baselines** are B1 and B2 ([B1 and B2: the two baselines](b1-and-b2.md)),
  and every experiment says which it starts from.

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

The runs behind a page — how many, which [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world."), every [setting](settings.md "One number or switch of a run's rules, fixed before it starts.") — are listed in
a box that folds away, in a last section, **How this was made**, so that the
page reads as an argument and the details are there for whoever wants to make
it again.

## The rules of the research

- **One change at a time.** An experiment changes one setting of a stated
  baseline and keeps everything else.
- **The thesis comes first.** It is fixed in the plan before the runs start.
- **Enough seeds.** A difference seen in fewer than thirty seeds per
  condition is called *indicative*, not an effect
  ([How many seeds an experiment needs](bootstrap.md#how-many-seeds-an-experiment-needs)).
- **No number without its null.** A measurement is reported next to what it
  would be by chance.
- **Every run can be made again**, from its plan, seed and frozen engine
  ([Is a run reproducible?](is-a-run-reproducible.md)).
- **The dead are counted apart.** A world that dies out is an outcome of its
  own, not a world that ended up somewhere
  ([When a world ends](extinction.md)).
- **A run is measured over its settled life**, from [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 500 to its end
  ([The settled life of a world](median-and-quantiles.md#the-settled-life-of-a-world)).
- **Runs are shared.** A run is named for what it is — baseline, world size,
  what differs, seed — so an experiment that needs runs which already exist
  uses them.
- **No tuned constants in new rules.** A rule added to the engine carries no
  hand-set factor beyond the [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") and the [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.")'s parameters: what a
  world does should emerge. ([Rules without numbers](rules-without-numbers.md)
  shows what the numbers B2 was built with do.)

## Where things are

- `book/pages/` — every page, named by what it is about;
- `book/figures/`, `book/diagrams/` — every figure and diagram, as SVG;
- `book/results/` — every number a page quotes, written by the analysis;
- `book/experiments/` — the baselines and each experiment's plan;
- `book/book.json` — the topics, and every page with its topics, kind and
  summary, from which the contents are made;
- `book/terms.json` — the terms, linked where a page first uses them;
- `book/legacy.json` — the chapter numbers the notebook had until
  23 October 2026, and the page each became;
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
- [The lab, the engine and pilots](the-lab.md): how an experiment is run.

<!-- turns -->
---

← [Glossary](glossary.md) · [Start here](../README.md#start-here) · [The ideas this builds on](the-ideas-this-builds-on.md) →

**Also about:** [Methods and statistics](../README.md#methods-and-statistics).

> [!info]- Pages that link here (4)
> [Welcome](../README.md) · [What this notebook is about](what-this-notebook-is-about.md) · [When a world ends](extinction.md) · [Rules without numbers](rules-without-numbers.md)
<!-- /turns -->
