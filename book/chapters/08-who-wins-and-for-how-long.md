# Who wins, and for how long?

## In short

No genotype wins for long — but whole lineages do, again and again. A brain's
genotype typically lasts two iterations, and the most common one usually
holds under 2% of the world, as the thesis expected. Yet in 21 of 26 worlds
a single genotype held more than a tenth of the world at some moment, one of
them 62%. And lineages take over completely: in a typical world, everyone
alive descends from one single founder by iteration 125, and later the whole
world shares one ancestor that lived about 730 iterations earlier — an
ancestor that keeps moving forward as one branch of the family tree outlives
all the others, about every 350 iterations. The thesis fails: something wins
in this world, and it is the lineage, not the genotype.

## Thesis

```thesis E06
```

Evolution needs something that lasts long enough to be selected. This
chapter asks how long a brain — a *genotype* — and a family of brains last in
the baseline world, and whether any of them takes over.

## Method

```experiment E06
```

The thirty runs of Chapter 3, read for brains rather than for agents:

- **distinct genotypes per agent**: 1 means no two agents share a brain;
- **families**: how many different ancestors from eight iterations earlier
  the living descend from;
- **the share of agents that kept their own node** in the game rather than
  losing it — and with it their brain — to a neighbour;
- **revolutions**: nodes won by a coalition of smaller stakers.

How long single genotypes and single agents last, and the share of the world
the most common genotype holds, are read from the stored frames, every 25
iterations.

## Results

### Genotypes come and go

```figure E06/genotypes
```

From iteration 100 on, the median world has 0.63 distinct genotypes per
agent: a genotype is carried, on average, by about 1.6 agents. It is the same
in every world (between 0.61 and 0.66) and at every time after the first
fifty iterations.

A genotype lasts a median of two iterations from the first game it is seen
in to the last; nine in ten are gone within seven, and only 15% last more
than five. The longest-lived genotype in any world lasted 80 iterations.
These are the lives of genotypes that ended before the run did, in the 26
worlds that reached iteration 3,000.

Agents last longer than genotypes, because an agent — a node — lives on when
its brain changes or is replaced by a conqueror's:
a median of five iterations, but 20% live more than fifty, and the oldest
lived 1,903. As births become rarer (Chapter 3), the living grow older:
their median age rises from 26 iterations at iteration 100 to about 80 at the
end.

```figure E06/age
```

### But some genotypes get big

```figure E06/share
```

Most of the time the most common genotype holds a small part of the world:
a median of 1.6%. But not always. After iteration 100, a single genotype held
more than a tenth of the world at some moment in 21 of the 26 worlds, more
than a fifth in 17, more than a third in 5, and more than half in one — 62% of 1,585 agents, in seed 12. Nearly all of
those moments came in full worlds, with hundreds or thousands of agents
alive — never fewer than 139 — not at the edge of extinction. And the moments
can be long: in seed 12, for 167 iterations in a row some single genotype
held more than a tenth of the world; in seeds 27, 8 and 26, for 84, 67 and
53. Since no genotype lived more than 80 iterations, these were several
genotypes in turn — and that points to what really wins.

### Lineages take over

```figure E06/ancestor
```

Every genotype has a parent genotype — the one it was changed from — so the
genotypes of a run form one family tree. Climbing it from the living every
25 iterations gives the most recent ancestor that all of them share. The
figure shows how many iterations back that ancestor lived. While it rises
in a straight line, the living still descend from more than one founder.

**One founder.** In a typical world, everyone alive descends from a single
one of the hundred founders by iteration 125. It happened by iteration 25 in
the fastest four worlds, and by iteration 750 in the slowest.

**One ancestor, again and again.** After that, the ancestor everyone shares
does not stay put. Each time one branch of the family tree outlives all the
others, it moves forward to a later genotype. After iteration 100 it moved
forward a median of eight times per world (between 3 and 21), about once
every 350 iterations. From iteration 500 on, everyone alive shared one
ancestor that had lived a median of 730 iterations earlier; nine in ten
shared one from 590 iterations earlier, and half the world one from 300.

A closer look at seed 12 shows what such a takeover is like. Around iteration 2,740, one genotype held up to 39% of the world for
eight iterations. In the next 120
iterations its descendants — dozens of genotypes, each changed from the one
before — went from 47% of the world to all of it.

**The families of eight iterations miss this.** The measure the thesis named
— the number of distinct ancestors of eight iterations before — stays in the
hundreds (a median of 315 from iteration 100 on), because a sweeping lineage
branches into many genotypes within eight iterations. It fell to ten or fewer
in only two of the 26 worlds, both during cycles of boom and crash: to 5 in seed 4 at iteration 101, when its population fell from
3,612 to 413 in a single iteration, and to 9 in seed 17, which cycled for its
first five hundred iterations.

```figure E06/families
```

### How nodes change hands

```figure E06/home
```

In the game, only 45% of agents keep their own node (from iteration 100 on).
That share was 32% in the first games and rose to about 45% by iteration 500,
where it stayed. Every node that changes hands takes on its conqueror's brain. With some 700
nodes changing hands in a game, against 40 to 70 births in an iteration, this
is by far the main way a genotype spreads.

```figure E06/revolutions
```

A median 51% of all nodes in a game are taken by a coalition of smaller
stakers that together outweigh the biggest one.

## Conclusion

The thesis is refuted. Its first part holds: there are always hundreds of
genotypes, about one for every 1.6 agents, and each lasts a couple of
iterations. But single genotypes do take large shares of the world for a
while — a tenth or more in 21 of 26 worlds — and, more importantly, whole
lineages take it over completely. Everyone alive descends from one founder
by about iteration 125, and from then on the world is swept again and again
by one branch of its family tree.

This changes what Part II has to fix first. The ladder in Chapter 1 asks
first whether lineages last long enough for selection to act on them (P1).
Genotypes do not: two iterations is far too short. Lineages do: a branch that
takes the world over lasts hundreds of iterations. The unit to follow is the
lineage, not the genotype.

Whether those sweeps are selection or chance this chapter cannot say. In any
population that reproduces, all lines but one die out sooner or later by
chance alone; whether that happens here faster than chance would make it,
because some brains do better, needs a world in which brains make no
difference. That world is the control of Chapter 12, where every decision is
drawn at random: if its lineages sweep just as often, the sweeps here are
chance.

## Details

| | |
|---|---|
| Runs | the 30 of Chapter 3: `B1-10000-s001` … `B1-10000-s030`; 26 reached iteration 3,000 |
| Genotype | a brain's id: copying a brain keeps it, and every change to a brain gives a new one, whose parent is the id it was changed from |
| From iteration 100 | each world's mean over every game from iteration 100 to 3,000; the median over the 26 worlds that reached the end; "lowest" is a world's lowest single value in that time |
| Lives | from the frame after every game: a genotype's or an agent's life runs from the first game it is seen in to the last; lives still going at the end of a run are not counted; medians of each world's median, over the 26 worlds |
| Common ancestor | every 25 iterations, the tree of genotypes is climbed from the living, newest first, to the newest genotype that all, nine in ten and half of the living agents descend from; its age is how long before that it was first seen. A move forward is a check at which it is younger than at the one before plus the 25 iterations between |
| Read outside the analysis | from the runs' frames, for this chapter: in seed 12, the share of agents descended from genotype 721816, the most common around iteration 2,740; and in the 26 surviving worlds, the number of agents alive at each of the 2,304 games after iteration 100 in which one genotype held more than a tenth (fewer than 200 in 72 of them, never fewer than 139) |
| Results | `book/results/E06.json`: every number above, each world's lineage summary, and each world's value by seed |
| Made with | `python3 gol_lab.py analyse E06`, which reads every frame of every run (about three minutes on four processes) |
