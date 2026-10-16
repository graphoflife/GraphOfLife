# Flow modules and the map equation

Which agents form groups? There are many ways to ask. Agents joined to each
other, agents related to each other — or agents **between whom the tokens
actually go**. A *flow module* is a group of the third kind: a set of agents a
token, wandering the flow from agent to agent, tends to stay inside. This note
explains how such groups are found, followed through time, and checked
against chance. The Research tab's **Flow modules** view and
[Chapter 71](../chapters/71-flow-modules.md) use exactly this.

## The flow of one game

In a game every agent stakes tokens on neighbours' nodes. Add up, for each
pair of neighbours, the tokens that crossed between them in either direction:
that is the weight of the link between them in this game. Tokens an agent
kept on its own node say nothing about who it is grouped with and are left
out.

## Describing a walk in as few bits as possible

Imagine a walker that steps along these weighted links, choosing each step in
proportion to the weights, for ever. It visits each agent a in a share p_a of
its time. To write down where it goes you could give every agent a code word
and write one word per step. The **map equation** (Rosvall and Bergstrom,
2008) asks whether you can do better with groups:

- give every group a name, and every agent a code word that only has to be
  unique *within* its group (short words, reused in other groups);
- write the group's name only when the walk enters it.

If the walk lingers in groups — many steps inside, few between — the group
names are rarely needed, and the description is shorter than the plain one.
The length of the description per step is

$$
L = q \log_2 q \;-\; 2\sum_i q_i \log_2 q_i \;-\; \sum_a p_a \log_2 p_a \;+\; \sum_i (q_i + p_i)\log_2 (q_i + p_i)
$$

with p_i the share of time inside group i, q_i the share of steps that leave
it, and q the sum of the q_i. The best grouping is the one with the smallest
L. One group holding everyone is the baseline L₀ (no structure at all), and

$$
\text{compression} = \frac{L_0 - L}{L_0}
$$

says how much the groups shorten the description: 0 for flow with no group
structure, higher the more the flow keeps to groups.

## Finding the groups

Trying every grouping is impossible, so the program starts with every agent
alone and moves agents one at a time to whichever neighbouring group shortens
L most, in a shuffled but fixed order, until no move helps (the local-moving
step of the Louvain method). The result is a good grouping, not always the
best.

## Following groups through time

A group of one game and a group of the next are taken to be **the same
module** when they share enough agents. The overlap is measured as the
Jaccard index: shared agents divided by the agents in either. Two ways to
decide "enough":

- **above a floor**: the best overlaps first, each module used once, and
  none below 0.3 (the view's default, which you can move);
- **mutually best**: of the pairs above the floor, only those in which each
  module is the other's best overlap. With the floor at 0 this matches at any
  overlap at all.

A small module can be "followed" on very little: two modules of two agents
that share one agent overlap by a third. On flows made mostly of pairs — as
in B2's worlds, where an agent puts its whole stake on one neighbour — a
matching at any overlap follows chance chains, so the book matches above the
floor and checks the result against chance (below).

Because identity is carried by overlap and not by membership, a module can
keep going while its agents change. Its **turnover** is the share of its
agents in a game that were not in it the game before. The change has two
sources, and the difference matters:

- **regeneration**: newcomers that were born, and members that died — the
  module makes and loses its own agents, and the pattern outlives every one
  of them;
- **exchange**: newcomers that came from other modules, and members that
  left for one — the agents were there all along, and the border moved.

## Checking against chance

Flow on any network has some group structure: neighbours trade with
neighbours. Two nulls answer two different questions.

- **Is there group structure at all?** The same game's flow is **rewired**:
  every token that crossed a link is cut into two halves, one at each end, and
  the halves are paired again at random. Every agent keeps how much flow it
  sends and receives; who it sends to is shuffled. A real module structure
  compresses the flow better than the rewired one does.
- **Do modules last longer than chance?** The rewired flow cannot say: its
  modules are pairs and threes, which "live" long on single shared agents.
  Instead each game's own modules are kept, with their sizes, and their agents
  are **dealt among them at random**, game by game. Followed the same way,
  they show what following finds when nothing but the sizes carries over.

Two more measures need no following at all, or ask what following finds:

- **Kept together**: of two agents in one module, the share still in one
  module so many games later, against the same with the later game's agents
  dealt at random — the share chance alone keeps together.
- **Outlasting**: of the modules followed for many games, the share none of
  whose first members is still in it at the end. That is the sense in which a
  group can outlast its members.

> [!info] Added later
> An earlier version of this note said that the mutual matching needs no
> floor and that real modules live longer than those of the rewired flow. The
> first was not what the program did (it matched above the floor), and the
> second is not a fair test. [Chapter 71](../chapters/71-flow-modules.md) uses the nulls above.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
