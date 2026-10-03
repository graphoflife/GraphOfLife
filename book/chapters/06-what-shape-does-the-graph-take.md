# What shape does the graph take?

## In short

Half tree, half web. The small-world ring the founders start on is gone after
the first few games, and what replaces it is long and thin: a third of all
connections are bridges whose loss would split the world, a third of all
agents hang on by a single connection, clustering is low, and two agents are
typically about ten steps apart. But it is not a tree: about half of all
agents sit in a core of loops. The thesis expected less than half there, and
so it fails, narrowly, on its last clause.

## Thesis

```thesis E05
```

An organisation of many agents — what Part II will look for — has to live
somewhere in the graph, so the shape of the graph limits what can exist in
it. An earlier look at two long runs found the graph tree-like and thin
(`research/Research.md`, Appendix A.8); this chapter asks it of thirty.

## Method

```experiment E05
```

The thirty runs of Chapter 3, read for the shape of the graph. These
statistics take longer to compute, so they are measured every 25 iterations:

- **connections per agent**;
- **bridges**, as a share of all connections: a bridge lies on no loop, so
  cutting it splits the world in two — and the smaller piece dies;
- **clustering**: how often two neighbours of one agent are also neighbours
  of each other (0.24 in the starting ring);
- **the core**: what is left after repeatedly removing every agent with a
  single connection, as a share of all agents;
- **leaves**: agents with a single connection;
- **average distance**: how many steps it takes to get from one agent to
  another, on average.

## Results

### The ring goes at once

The founders' ring is a *small world*: each founder is joined to its four
nearest neighbours, a fifth of those connections are rewired to random
founders elsewhere, and so the ring is both clustered and short. It does not
survive the first game. Connections that carry no tokens in a game are cut at
its end, and that takes most of the long shortcuts with it: after the first
game, clustering has fallen from about 0.24 to 0.13 in the median world, the
average distance between two agents has grown from about four steps to about
ten, and a fifth of all connections are already bridges.

### Thin and long

```figure E05/degree
```

From iteration 100 on, the median world has 3.3 connections per agent — the
middle half of the worlds between 3.2 and 3.4.

```figure E05/bridges
```

**Bridges.** A median 32% of all connections are bridges, and in every one of
the 26 worlds that lived to the end it was more than a fifth. Cutting a
bridge splits the world in two, and the smaller piece dies. In the median
world, the worst single cut would take 12% of all agents with it.

```figure E05/leaves
```

**Leaves.** A median 36% of all agents have a single connection, up from 16%
in the first iterations.

```figure E05/clustering
```

**Clustering.** How often two neighbours of an agent are neighbours of each
other: a median 0.077, below 0.1 in 21 of the 26 worlds and above 0.2 in
none.

```figure E05/distance
```

**Distance.** Two agents are a median 9.7 steps apart, and the two furthest
apart about 29. A random network of the same size and the same number of
connections would be about six steps across: these worlds are stretched out.

### But a core of loops

```figure E05/core
```

Strip off every agent with a single connection, and keep doing so until none
is left: what remains is the *core*, where every agent lies on a loop or
between loops. It holds 71% of the agents after the first game, falls to about 56% between
iterations 25 and 50, and stays near there. From iteration 100 on, the
median world has 53% of its agents in the core, and only 6 of the 26 worlds
had less than half.

## Conclusion

The thesis holds in three of its four clauses: the ring is gone almost at
once, more than a fifth of all connections are bridges, and clustering stays
below 0.1. It fails in the fourth: the core holds about half of all agents,
not less, which is what the *refute* clause called "a core holding most
agents", if only just.

So the picture of a world as a tree of families is half right. A world is a
core of loops, holding about half its agents, with trees hanging off it —
the leaves alone are a third of all agents — and long enough that news takes
ten steps to cross it. Something keeps closing loops, and the rules offer one way: a child
can be joined both to its parent and to some of its parent's neighbours,
which closes a triangle the moment it is born. Whether loops last depends on whether tokens
keep flowing through them.

For the organisations Part II looks for, the core is where they can live: it
is the only part of the world where a group of agents is joined by more than
one path, and so the only part where losing one connection does not cut a
group in two.

## Details

| | |
|---|---|
| Runs | the 30 of Chapter 3: `B1-10000-s001` … `B1-10000-s030`; 26 reached iteration 3,000 |
| Measured | every 25 iterations, after the game: bridges, clustering, the core, distance and the worst single cut; the number of connections and leaves after every game |
| From iteration 100 | each world's mean over its measurements from iteration 100 to 3,000; the median over the 26 worlds that reached the end |
| Clustering | transitivity: three times the number of triangles over the number of connected triples |
| Core | the 2-core: what is left after removing every agent with fewer than two connections, again and again |
| Distance | the mean number of steps between two agents, and the diameter, the most steps between any two |
| The starting ring | 100 founders, four neighbours each, a fifth rewired: with networkx's `watts_strogatz_graph`, over seeds 1–30, a median clustering of 0.244 and 4.15 steps between two founders |
| A random network | of 1,300 agents with 3.3 connections each, about ln 1,300 ÷ ln 3.3 ≈ 6 steps across |
| Results | `book/results/E05.json`: every number above, and each world's value by seed |
| Made with | `python3 gol_lab.py analyse E05` |
