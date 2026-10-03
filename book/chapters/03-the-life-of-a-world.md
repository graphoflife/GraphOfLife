# The life of a world

## In short

A world has a short, violent youth and then a long middle age — but not a
steady one. In its first fifteen iterations the hundred founders become some
1,500 agents, while conquest wipes out most of their lines of descent. From
about iteration 100 on, the *typical* world holds 1,050 to 1,450 agents, and
that typical level hardly changes over the next 2,900 iterations. A *single*
world, though, keeps wandering: only eight of twenty-six ended within a fifth
of where they stood at iterations 100 to 200. Births slow to about half their
early rate, and four worlds of thirty died out. So the thesis fails: a world
has more than one phase of life, and its long middle is a slow wander, not a
steady state.

## Thesis

```thesis E02
```

This is the first look at what a world actually does over a long time, and
everything else in Part I builds on it. If a world settles, the later chapters
can describe *the* world. If it goes through phases, each of them has to say
which phase it is about.

## Method

```experiment E02
```

Thirty worlds of the baseline B1, each with 10,000 tokens — so a hundred
founders with a hundred tokens each — and each with its own seed, from 1 to
30. In its first iterations a world of this size grows to a few thousand agents. Each
runs for 3,000 iterations. Every frame is kept, and every statistic is
recorded for every frame; the statistics that describe the shape of the
graph, which take longer to compute, every 25 iterations.

These same thirty runs also answer Chapters 4, 5, 6 and 8, each asking its own
question of them. One press of ▶ runs them all.

What this chapter looks at, over the whole life of a world:

- **agents alive** after the game, the clearest measure of how big a world is;
- **connections** between them;
- **births** in each reproduction phase, and **starved** agents — those that
  lost all their tokens in the game;
- **families**: how many different ancestors from eight iterations earlier
  the living descend from. A world taken over by one lineage has one family.

Each figure shows, at every point in time, the median of the thirty runs as a
line, the middle half of them as a band, and nine in ten as a fainter band.
The median of the first hundred to two hundred iterations is compared with
the median of the last two hundred.

## Results

### The youth

```figure E02/agents
```

The figure shows the number of agents alive after every game: the dark line
is the median of the thirty worlds, the bands hold the middle half and nine
in ten of them, and the three thin lines are single worlds (seeds 1, 2 and 3).

**A boom.** Births take off at once: the median world has 900 births per
iteration around iteration 12, and its population, a hundred at the start,
is past 1,450 by then. The median peaks at about 1,850 agents around
iteration 27.

```figure E02/births
```

**A crash.** The boom turns the founders' wealth into a crowd of poor agents,
and the game starves them: starvation peaks around iteration 17, at about a
hundred agents per game in the median world. The thin lines in the figure of
agents show how ragged this is in a single world.

```figure E02/starved
```

**A bottleneck.** The world's lines of descent thin out much faster than its
agents. A conquered node takes on its conqueror's brain, so every game can
end a line, and in the first iterations most of them end: in iterations 5 to
10 the living descend from a median of 21 lines (the figure's dip). Afterwards
the count — *families*, the distinct ancestors of eight iterations before —
recovers to about 250 by iteration 100, because the world now holds many more
brains to descend from. Chapter 8 follows the founders' lines further: in a
typical world, everyone alive descends from one single founder by iteration
125.

```figure E02/families
```

One world never got past its youth: seed 23's founders went from 100 agents
to 92, 81, 35 and 7, and the run ended in its fourth iteration.

### The long middle

**The typical world holds still.** From iteration 100 on, the median of the
thirty worlds stays between about 1,050 and 1,450 agents nine tenths of the
time. Over iterations 100
to 200 the median world held 1,367 agents; over iterations 2,800 to 3,000,
1,236 — 10% fewer, inside the 20% the thesis allowed. Its connections went
from 2,037 to 2,161, 6% more.

```figure E02/edges
```

**A single world does not.** The bands stay wide all the way, and a world
moves through them: of the 26 worlds that lived to the end, only 8 ended
within a fifth of their own population at iterations 100 to 200 — 12 ended
more than a fifth higher and 6 more than a fifth lower. Connections were the
same: 7 within a fifth, 13 higher, 6 lower. Chapter 4 measures this
wandering: a world remembers its size for a few hundred iterations, and its
most crowded hundred iterations typically hold four to five times as many
agents as its emptiest.

One world showed a different kind of motion: seed 17 spent most of its first
five hundred iterations in cycles, its population doubling for two or three
iterations and then crashing by half or more, dozens of times over, before
it settled down.

**The world grows quieter.** Births and deaths do not settle the way the
population does. The median world had 68 births per iteration over iterations
100 to 200, and 40 over the last two hundred — 41% fewer; births fell by more
than a fifth in 18 of the 26 worlds. Starvation in the game fell from 10
agents per game to 5.6, and the agents removed because they were cut off from
the rest of the world from 41 to 23. With fewer births and fewer deaths
around a population that holds, the living grow older: Chapter 8 finds the
median age of the living rising from 26 iterations at iteration 100 to about
80 at the end.

### Four worlds died

Seed 23 died in its fourth iteration, and seeds 14, 24 and 20 after 313,
1,547 and 2,184 iterations. None of the three later deaths was sudden. Each
of those worlds had held about a thousand agents; each fell over 100 to 250
iterations to fewer than a hundred, lingered there for between 56 and
197 iterations, and then went below twenty agents, the size at which a run
counts as extinct and stops.

## Conclusion

The thesis holds for the typical world and fails for the single one. The
median population and the median number of connections are within a tenth of
where they were at iteration 100 to 200 — but births fell by 41%, single
worlds wandered far from where they had been, and four of thirty worlds died
out. A world has more than one phase of life:

1. **a youth** of about a hundred iterations: a boom of births, a crash, and a
   bottleneck in which most lines of descent end;
2. **a long middle** in which the typical world neither grows nor shrinks but
   every single world wanders, while births and deaths slowly become rarer
   and the living older;
3. for some worlds, **a decline** over a few hundred iterations, and death.

For the chapters after this one, that means:

- **Leave out the youth.** Every thesis in Part I is checked from iteration
  100 on.
- **The end of a run is a moment in a wander.** Where a single world stands at
  iteration 3,000 is not where it "settled". Chapter 4 asks how much of the
  difference between two worlds this wander is.
- **Dying out is an outcome of its own** — one world in seven or eight at this
  size. Where the worlds ended up is measured over the worlds that lived to
  the end, and the dead are counted apart.
- **The baseline is still changing at iteration 3,000.** Births have not
  stopped falling. Something in the world keeps changing slowly — the age of
  its agents, the shape of its network, perhaps its brains. Whether that is
  adaptation is a question for Part II.

## Details

| | |
|---|---|
| Runs | 30: `B1-10000-s001` … `B1-10000-s030`, one session each or two (paused once); 26 reached iteration 3,000 |
| Strain | `gol-1+brain_kind=float16`, with `message_amount=30` and `mutation_probability=0.2` |
| Engine | snapshot `f820369a1583af53`, commit `4bc7165` |
| Environment | Python 3.12.3, numpy 2.5.1, networkx 3.6.1, OpenBLAS (scipy-openblas 0.3.33), Intel Core Ultra 7 258V, numpy dispatch AVX2, one BLAS thread |
| Cost | 26 hours of computing, done in 7.2 hours on four workers; 7.8 GB on disk; at most 651 MB of memory for one run |
| Windows | each world's mean over iterations 100–200 and 2,800–3,000, from the statistics of every game; the median over the worlds that lived through each window (29 and 26) |
| Results | `book/results/E02.json`: every number above, each world's value by seed, and where every run came from |
| Made with | `python3 gol_lab.py analyse E02` |
