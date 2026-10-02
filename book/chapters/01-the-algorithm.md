# The algorithm and the question

## In short

Graph of Life is a world of agents living on the nodes of a network. Each
agent holds tokens and a small neural network that decides everything it
does. Agents have children, connect, give tokens away and fight over each
other's places, and the total number of tokens never changes. Nothing is
optimised and nothing is rewarded: whatever keeps its place is what we see
next. This book first works out what such a world actually does (Part I),
and then asks whether it can be made to go on producing new things without
end (Part II).

## The world

The world is a **graph**: dots (the *nodes*) joined by lines (the *edges*).
Every node is one **agent**. An agent can only see, and only act on, itself
and the agents it is joined to — its *neighbours*.

Every agent holds a whole number of **tokens**. Tokens are the only thing of
value, and the world has a fixed supply of them: most experiments in Part I
use 10,000. Tokens move between
agents but are never created or destroyed. An agent with no tokens left
dies.

Every agent also has a **brain**: a small neural network that is never
trained. It is copied from a parent, sometimes with small random changes, and
that is the only way behaviour ever changes.

A world starts with one founder for every hundred tokens, each with an equal
share and a brain of random weights, on a *small-world* ring: each founder is
joined to its two nearest neighbours on either side, and a fifth of those
connections are rewired to a random agent elsewhere.

## One iteration

Time goes in **iterations**, and each iteration has two phases. The viewer
records a picture of the world — a *frame* — after each phase.

**1. Reproduction.** Every agent, one after another, decides what share of
its tokens to give a child. The child appears on a new node with a copy of
its parent's brain — which has one chance in five of being changed a little
on the way — and joins whichever of the parent's neighbours, and the parent
itself, the parent chooses. The parent can also hand one of its own
connections to the child. A child that ends up joined to nobody is lost at
once.

**2. The game.** Every agent spends *all* of its tokens on itself and its
neighbours: spread over several of them, or all on one. This is a game of
*Colonel Blotto*, played everywhere at once. Each node then goes to whoever
put the most on it. The winner's brain is copied into that node, and the node
holds everything that was put on it. Part of every stake can be marked as
*revolutionary*: if enough small stakers together outweigh the biggest one,
the node goes to one of them instead.

**Clean-up.** At the end of the game, every connection that carried no
tokens during it is cut — a connection a child made just before included, so
a new connection has to be used at once to last. After each phase, agents
with no tokens die, and anything no longer joined to the largest connected
piece of the world is removed. The tokens of everything
that died are shared out among the survivors at random, each survivor as
likely as any other to receive each token, so the supply stays exactly the
same.

**Change.** At the end of every game, every brain has one chance in five of
being changed a little: about a tenth of its weights are nudged at random.

## What an agent sees and decides

Before every decision, an agent's brain looks at itself and at each
neighbour in turn, and for each it sees:

- the tokens and the number of connections of itself and of that neighbour
  (as logarithms, so a hundred and a thousand differ as much as ten and a
  hundred);
- summaries of its whole neighbourhood: the spread of their tokens and
  connections;
- messages, thirty numbers each: what it and that neighbour last wrote to
  each other, and what each wrote to itself — a kind of memory;
- five random numbers, so that two agents in the same position need not do
  the same thing.

From these the brain gives, for each neighbour, a score for every decision:
how much to give a child, whether the child should join that neighbour,
whether to hand that connection over, how much to stake on that node and how
much of it as a revolution, and what message to send.
Every decision also comes with a second output that says *how* to read the
first — take the best option, or choose at random in proportion to the scores.
So how decisive or how random a lineage is, is itself something that
evolves.

## What is inherited, what changes, what is selected

This is the evolutionary reading of everything above.

- **Inherited:** a child gets its parent's brain. A conquered node gets its
  conqueror's brain.
- **Changed:** a brain has one chance in five of changing a little at birth,
  and the same chance again at the end of every game.
- **Selected:** nothing is scored. A brain survives by keeping tokens and
  keeping its place, and it spreads by having children and by winning nodes.
  The fixed supply of tokens is what makes this a competition: one agent's
  gain is always another's loss.

## The question

The long-term aim of this project is **open-ended evolution**: a world that
does not settle, but keeps producing new kinds of things for as long as it
runs, as life on Earth has. No artificial system has convincingly done this
yet, and there is no agreed test for it. `research/Research.md` (the
*Findings* page) sets out the definitions on offer and the one this project
adopts: *unbounded growth in the kinds of multi-agent organisation* —
structures made of many agents that persist, are reproduced, and could not
have been listed in advance.

It also sets out a ladder of four claims, each needed before the next:

- **P1 — there is evolution:** lineages last long enough for selection to act
  on them.
- **P2 — adaptation accumulates:** a late population beats its own ancestors.
- **P3 — there are organisations:** groups of agents that persist while their
  members change.
- **P4 — the space of organisations keeps growing.**

What is already known (from earlier pilots, with fewer seeds than this book
asks for) is that the first rung is not secure: even at the baseline's rate
of change, a genotype typically lasted a single iteration and fewer than one
in ten lasted more than five, and conquest overwrites brains as well. Part II is about the
changes to the rules that `Research.md` proposes — starting with brains that
change only when they are copied.

But first, **Part I** asks the plain questions. What happens in a world from
its first iteration onwards? How much does it depend on chance? Where do the
tokens go? What shape does the network take? Who wins, and for how long? And
what changes when the world is bigger, when brains change faster, or when
they are larger?

## The baseline

Every experiment changes one thing, and the thing it changes is measured
against a **baseline**, B1 (`book/experiments/B1.json`). It is the algorithm
as a new run is offered it, with one difference: brains store their weights
at half precision (`float16`), as every run made by hand so far has. That
keeps a brain in a quarter of the memory, which is what makes worlds of
100,000 tokens and more possible several at a time.

| Setting | B1 | What it means |
|---|---|---|
| Brain | 5 hidden layers of 50, 45, 40, 35, 30 | 15,795 weights, stored at half precision |
| Messages | 30 numbers, read in an extra pass before acting | what neighbours tell each other |
| Random inputs | 5 per neighbour | |
| Mutation | one chance in five per brain, at birth and after every game; about 10% of its weights nudged by 0.2 of their scale | how fast brains change |
| Handover, revolutions | on | |
| Gifts | off | |
| Unused connections | cut at the end of the game, if nothing crossed them in it | |
| Tokens of the dead | shared out evenly | |
| Founders | one per hundred tokens, on a ring of degree 4, a fifth rewired | |
| Extinct | at 20 agents or fewer | the run stops there |
| World size | chosen by each experiment | |

Its name in the project's strain registry is `gol-1+brain_kind=float16`:
every mechanic is as the algorithm was first defined except the brain's
precision. The message length and the rate of mutation are parameters,
cited beside the name rather than in it.

## How the experiments are read

Each experiment chapter states its thesis before the runs and then reports
what happened. The rules — enough seeds, one change at a time, every run
reproducible, and a null for every number — are on the page *About this
book*, which also explains how the runs are made and how to make any of them
again.
