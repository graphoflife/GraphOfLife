# About this book

This book is a research programme, written down while it is carried out. It
asks what the Graph of Life algorithm actually does — and, later, whether it
can be made to keep producing new things without end, which is what
*open-ended evolution* means.

Part I is about understanding the algorithm as it is: one plain question per
chapter, answered with experiments. Part II turns to open-ended evolution.

## How it is made

Three steps, over and over.

1. **A plan.** Claude writes a chapter's *Thesis* and *Method* and the
   experiment's plan (`book/experiments/E…json`): the settings, the size of
   the world, the seeds and how long each run goes. The thesis is written
   down before any of its runs exist, and the chapter shows it from the plan,
   so it cannot be reworded once the results are in.
2. **The runs.** On a computer running `gol_server.py`, the chapter has a ▶
   button. The lab then runs the experiment's simulations, several at a time,
   each in a process of its own, and says how long is left. It can be paused
   and continued at any time, and it carries on if the page is closed. Most
   experiments take about half a day; none takes more than two days.
3. **The results.** Claude analyses the finished runs, draws the figures,
   writes *Results* and *Conclusion*, and plans what comes next.

## How a chapter is built

- **In short** — the answer, in two or three sentences.
- **Thesis** — what we expect, why, and what would show it wrong.
- **Method** — what was run: the baseline, what was changed, the seeds.
- **Results** — what happened, with figures. Every number it gives comes from
  the analysis, not from looking at a chart.
- **Conclusion** — what that means, and what to try next.
- **Details** — confidence intervals, tests, and where every run came from.

## The rules

- **One change at a time.** Every experiment changes one setting of a stated
  baseline (`book/experiments/B1.json`) and keeps everything else.
- **The thesis comes first.** It is fixed in the plan before the runs start.
- **Enough seeds.** A difference seen in fewer than about thirty seeds per
  condition is called *indicative*, not an effect: results in this project
  have reversed between six seeds and twenty.
- **No number without its null.** A measurement is reported next to what it
  would be by chance.
- **Every run can be made again.** Each simulation records its seed, the
  frozen copy of the engine it ran on and the commit it came from, and the
  versions of Python, numpy and networkx, beside it in `provenance.json` and,
  once analysed, in the chapter's results.
- **The dead are counted apart.** A world that dies out did not end up
  anywhere, so where the worlds of a condition end up is measured over the
  worlds that lived to the end, and the dead are reported beside them as an
  outcome of their own.
- **A run is measured over its settled life.** A world spends its first few
  hundred iterations in a youth unlike the rest of its life (Chapter 3) and
  then wanders (Chapter 4), so where it ends up is its average from
  iteration 500 to its end. Chapters 3 to 8 used the last fifth of a run,
  which was the rule before Meta I.
- **Runs are shared.** A run is named for what it is — baseline, world size,
  what differs, seed — so an experiment that needs runs which already exist
  uses them. Several chapters of Part I read the same thirty runs.
- **A meta chapter after every five experiments** looks for what holds across
  them.

## Making a result again

Every analysed chapter names the commit and the engine its runs were made
with. With that commit checked out, `python3 gol_lab.py run E…` makes the
runs again from their seeds, and `python3 gol_lab.py verify E…` checks a few
of them against what was recorded. On the same kind of machine with the same
libraries they come out identical, frame for frame; Chapter 2 is the test of
that.

## Where things are

| | |
|---|---|
| `book/book.json` | the order of the chapters |
| `book/chapters/` | one Markdown file per chapter |
| `book/experiments/` | the baseline and each experiment's plan |
| `book/figures/`, `book/results/` | what an analysis writes |
| `GraphOfLifeRuns/` | the runs themselves, on the machine that ran them |

The programme the book follows, and what is already known, is in
`research/Research.md` — the *Findings* page under Research.
