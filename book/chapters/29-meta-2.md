# Meta II · What the measurements say about open-ended evolution

Part III measured the baseline world in every way the viewer can, and a few
more. This chapter steps back and asks what all of it means for the question
the book exists for: **can open-ended evolution happen in Graph of Life, and
what would it take?** It is written the way a researcher keeps a notebook —
what we now know, what we think it means, what we propose to do about it, and
how we will know whether it worked — and, as in every chapter, so that a
student can follow each step.

> [!question] Questions of this chapter
> - What is the baseline world, now that it has been measured every way?
> - Where does it stand on the ladder of [Chapter 1](01-what-this-book-is-about.md)?
> - What stands in the way of open-ended evolution, and which changes to the
>   rules could remove it?
> - How will we measure progress — and know if we found it?

## What we are looking for

**Open-ended evolution** is evolution that does not run out of new things to
make. Life on Earth has it: four billion years in, it still produces new kinds
of organisms, new ways of living, new levels of organisation. Almost every
artificial system stops — it finds a way of doing well, fills the world with
it, and from then on only reshuffles ([Chapter 6](06-the-ideas-this-builds-on.md)).

[Chapter 1](01-what-this-book-is-about.md) set out a ladder of four claims,
each needed before the next:

1. **P1 — there is evolution:** lineages last long enough for selection to act
   on them, and selection changes the outcome.
2. **P2 — adaptation accumulates:** a late population would beat its own
   ancestors.
3. **P3 — there are organisations:** groups that persist while their members
   come and go.
4. **P4 — the space of organisations keeps growing**, without a ceiling.

The project's research plan (`research/Research.md`) adopts a definition aimed
at the top of the ladder: open-endedness as unbounded growth in the space of
**multi-agent organisations** — structures made of agents, which persist and
are reproduced. The reason is a counting argument: a brain has a fixed number
of weights, so the space of brains is a fixed box; but the number of ways to
arrange *n* agents grows faster than any exponential in *n*. If anything here
can grow without bound, it is how agents are arranged and what they do to each
other.

## The baseline world, in one paragraph

Here is what Part III found, put together. **Tokens** move almost as a random
walk would: agents spread their stakes nearly evenly over everyone they can
reach ([Chapter 20](20-where-the-tokens-flow.md)), so tokens settle locally in
proportion to connections plus one ([Chapter 23](23-how-properties-scale-together.md)),
run downhill from rich neighbourhoods to poor ([Chapter 19](19-gains-and-losses.md)),
and never circulate. Half of every agent's traffic is an equal exchange that
keeps its connections alive at their price of a token a game
([Chapter 27](27-do-agents-cooperate.md)). The **network** is a stretched web
of stars, grown locally by children joining their parents' neighbourhoods,
twice as long and thirty times easier to cut than a random network with the
same connections ([Chapter 24](24-the-geometry-of-a-world.md)); it breaks
along its poor regions, whose borders go quiet, and the global cull kills
whatever falls off ([Chapter 25](25-how-a-world-breaks.md)). **Brains** spread
by conquest eighteen times more than by birth, and change mostly in place, not
when copied ([Chapter 28](28-questioning-the-mechanics.md)); the brain on a hub
is replaced almost every game, and the same two brains rarely face each other
twice ([Chapter 20](20-where-the-tokens-flow.md), [Chapter 27](27-do-agents-cooperate.md)).
Relatives live together, but cannot recognise each other, and do not treat each
other differently ([Chapter 27](27-do-agents-cooperate.md)). Almost every
structural statistic is the same in every world: it follows from the rules,
not from what brains compute.

The short version: **a world of places, not individuals.** The stage — the
network, the landscape of wealth — is stable and rule-made. The actors — the
brains — change almost every game.

## Where the world stands on the ladder

**P1, evolution: partly.** There are lineages, and they matter: a whole world
descends from one branch of the family tree about every 350 iterations
([Chapter 16](16-genotypes-and-lineages.md)), families occupy regions
([Chapter 26](26-one-world-many-colours.md)), and worlds whose brains never
change end up in very different places, while worlds that evolve all end up
alike ([Chapter 30](30-do-the-brains-matter.md)). But heredity is weak. A brain
is changed with probability 0.2 after every game whether or not it is copied,
and a brain is overwritten whenever its node is won. The project's pilot
measurements (`research/pilot_heredity.py`, at a higher mutation rate) found a
lineage halfway to a stranger after about three iterations. Selection can only
work on what is inherited; here, little is inherited for long.

**P2, accumulation: not yet asked.** The test is a tournament across time:
bring back the brains of iteration 500 and set them against those of iteration
2,500 in the same world (Cliff and Miller 1995). Nothing in Part III could
answer it.

**P3, organisations: not found.** The candidates exist — family regions, hubs
with their stars — but nothing yet shows a group that persists while its
members change. Partnerships last one game in twenty; positions are not held.

**P4: out of reach** until P1 to P3 stand.

## What stands in the way

Part III, read together, points at six obstacles. Each is a rule, and each was
examined in [Chapter 28](28-questioning-the-mechanics.md).

1. **Variation in the living.** 97% of new genotypes are made in brains that
   are not being copied. Successful brains are eroded in place.
2. **Conquest erases.** A won node loses its brain entirely; an individual
   ends whenever its node is won, and on a hub that is almost every game.
3. **Nothing can be held.** No incumbency, no lasting partners: the "shadow of
   the future" that reciprocity needs is about 0.05.
4. **Nothing is positive-sum.** Tokens are conserved and every stake is a
   transfer; no interaction makes anything that was not there before. An
   organisation can never be worth more than its parts, so there is no reason
   for one to form (`research/Research.md`, §3.3).
5. **Nobody can tell kin from stranger.** Kin selection has its precondition —
   relatives live together — but brains cannot see relatedness.
6. **The world can neither reach across nor split.** Growth is local only, so
   the network stretches and its shortcuts are never renewed; the global cull
   kills any part that separates, so populations can never diverge in
   isolation.

## What we propose to change

Each proposal is a rule that can be switched on, run against the baseline with
thirty seeds, and measured — the method of Part IV. Those marked with a §
come from the research plan (`research/Research.md`, §4); the others from what
Part III found. They are ordered by what has to come first, not by how
interesting they are.

### First: make heredity real

- **Mutation at replication, and nowhere else** (§4.1 of the plan). A brain
  changes only when it is copied — into a child or into a won node. A brain
  that is not copied stays itself. This is planned as Chapter 37 (see
  [the contents](../README.md)); its measure is how long a lineage stays like
  its ancestor, and whether selection then separates good brains from bad.
- **A germline** (§4.2), if that is not enough: every agent carries an
  unchanged copy of the brain it was born with, and passes that on.

### Second: let individuals last

- **Incumbency.** The agent on a node gets an advantage in defending it — its
  home stake counts double, or a coalition must outweigh the home stake too.
  Measure: how long brains hold hubs; how long partners last.
- **Recombination at conquest.** A won node takes a mix of the winner's brain
  and its own, not a replacement — the closest thing to sex this world could
  have, and a way for an agent to survive in part when it loses.

### Third: let cooperation pay, and let kin find each other

- **Mutual flow yield** (§4.4). A connection across which tokens flow in both
  directions in a game yields a small number of new tokens, split between its
  ends — with a small decay of every holding, so that the world stays bounded.
  Now two agents together can be worth more than two apart, and there is a
  defection problem to solve, which is what cooperation theory is about.
- **Heritable tags.** Every brain carries a few numbers — a tag — copied with
  it and changed rarely; every agent sees its candidates' tags among its
  inputs. Tags are how cooperation can evolve without memory or reputation
  (Riolo, Cohen and Axelrod 2001), and they give relatives a way to recognise
  each other.

### Fourth: open the world

- **Proposed connections** (§4.5): two agents anywhere in each other's
  neighbourhood of neighbourhoods may form a connection if both want it, at a
  cost — and, rarely, a child is joined to an agent anywhere in the world. The
  world's shortcuts would be renewed instead of spent.
- **Let separate pieces live.** Drop the global cull: a piece that separates
  becomes its own world, with its own tokens, and may meet the main world again
  through the long-range connections above. Isolation, divergence and
  reunion — the oldest engine of new species — would become possible.
- **Several kinds of tokens** (§4.3): tokens of different colours, which agents
  value differently by inheritance. Niches, and a reason to trade.

### Later: let the possible grow

Brains that can grow (§4.6), agents that set some of their own rules (§4.7),
a second heritable structure that belongs to a relationship rather than an
agent (§4.8), and, at the very end, groups that copy themselves (§4.9) — each
only once the earlier steps hold.

Three smaller repairs from [Chapter 28](28-questioning-the-mechanics.md) can go
along with any of these: a smooth share function, a separate output for
whole-agent decisions, and the option to hold tokens back from the game.

## How we will know

A rule change is only progress if a measurement says so, against a null. The
measures, by rung of the ladder:

- **For P1.** How far a lineage's brains drift from their ancestor's in a given
  time, in weights and in behaviour, against unrelated pairs (heredity
  half-life). And selection ablation: the same world with the winner of every
  node drawn by chance among its stakers — if the real rule does no better,
  selection is not doing the work.
- **For P2.** The cross-time tournament: earlier against later brains, in the
  same world. A clean gradient — later beats earlier — is accumulation; a
  pattern of bands is cycling, as rock–paper–scissors would give.
- **For cooperation** — the measures of [Chapter 27](27-do-agents-cooperate.md):
  kin discrimination, Hamilton's accounting Σ(*rB* − *C*), reciprocity beyond
  the even split, the lifetimes of partnerships, coalitions that repeat, and
  whether regions held by one family outlast mixed ones.
- **For P3.** Groups that persist while their members turn over, found by
  methods that do not assume them — flow modules (Rosvall and Bergstrom 2008),
  or groups that carry information about their own past into their future
  better than their surroundings do (Krakauer et al. 2020) — always against a
  network rewired at random with the same connections, which Part III showed
  is very different from the real one.
- **For P4.** Bedau and Packard's evolutionary activity (1992): the cumulative
  presence of components that persist, set against a "neutral shadow" — a run
  in which the same events happen but no component has any advantage; and the
  MODES measures of change, novelty, complexity and ecology (Dolson, Vostinar,
  Wiser and Ofria 2019), each with a filter that only counts what persists.
  Open-endedness is the curves that never level off, above their shadows.

## What Part III changed in what this book teaches

Findings change the teaching, and the earlier chapters have been brought up to
date where they did:

- **Wealth follows connections because of a law**, not a mystery: the even
  split's resting state ([Chapter 13](13-where-do-the-tokens-go.md) now says
  so, and [Chapter 20](20-where-the-tokens-flow.md) derives it).
- **"Keeping one's node" is about places**: hubs are nearly always taken
  ([Chapter 14](14-how-the-game-is-played.md)).
- **The network's shape has a mechanism**: copying neighbourhoods at birth
  gives the stars, the triangles and the hierarchy
  ([Chapter 15](15-what-shape-does-the-network-take.md),
  [Chapter 23](23-how-properties-scale-together.md)).
- **"An agent" has two meanings**, the node and the brain, and the book now
  says which it means where it matters ([Chapter 28](28-questioning-the-mechanics.md)).

## Questions for the next analyses

Some questions need no new runs, only new looks at the old ones:

- **Do family regions persist as organisations?** A region held by one lineage
  for hundreds of iterations, while its nodes turn over, would be a first
  candidate for P3 — measurable now, against a null that shuffles lineages.
- **What does a brain respond to?** Feed a stored brain the same inputs with one
  changed — a neighbour's tokens, a message — and see what its stakes do. If
  brains mostly tell "me" from "others" and little else, the even split is no
  surprise.
- **Are the old at the edges?** [Chapter 26](26-one-world-many-colours.md)
  suggested it; age against distance from the hubs would say.
- **Why do worlds reproduce less as they age?** Births fall by 41% over a
  world's youth ([Chapter 11](11-births-deaths-and-ages.md)), and founders
  reproduce twenty times as often as settled agents
  ([Chapter 21](21-how-agents-have-children.md)). That is a candidate for an
  adaptation — the clearest one Part III saw.

## The road from here

Part IV continues with single settings — the rate of mutation, coalitions
switched off, the size of the brain, the size of the world — because they are
cheap and each teaches how the world works. Part V then changes rules, in the
order above: first heredity, then lasting individuals, then cooperation that
pays and kin that know each other, then an open world. The book will measure
each with the instruments of this chapter, and come back here, in Meta III, to
say what moved.

It is a long way. But Part III has made it shorter in one important sense: it
has turned a vague hope — that interesting things might emerge if we watch long
enough — into a list of specific rules that stand in the way, each of which can
be changed and tested on its own.

<!-- turns -->
---

← [Chapter 28 · Questioning the mechanics](28-questioning-the-mechanics.md) · [Contents](../README.md) · [Chapter 30 · Do the brains matter?](30-do-the-brains-matter.md) →
<!-- /turns -->
