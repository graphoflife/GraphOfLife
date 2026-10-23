# Flow modules: the groups tokens stay in

> [!question] Questions of this page
> - When agents stake tokens on one another, do the tokens stay inside
>   groups? How large are the groups, and who belongs to one?
> - Do the groups last from game to game — longer than chance would make them,
>   and longer than the network alone would?
> - Can a group outlast the agents it began with?
> - Is it the same in every world the notebook has made?

## What a flow module is

In every game each agent splits its tokens over its own node and its
neighbours' nodes ([How the game is played](how-the-game-is-played.md)). A token staked
on a neighbour's node is a token that **flows** along that link. Add up, for
each pair of neighbours, what crossed between them in either direction, and a
game becomes a weighted network: the flow.

A **flow module** is a group of agents that a token, wandering this flow from
agent to agent, tends to stay inside. The groups are found with the **map
equation** (Rosvall and Bergstrom, 2008): the grouping that lets a walk on the
flow be described in the fewest bits, by naming a group only when the walk
enters it. **Compression** says how much the groups shorten the description:
0 for flow with no group structure, more the more the flow keeps to groups.
The note [Flow modules and the map equation](map-equation.md)
explains it with the formula; the Research tab's *Flow modules* view draws it
for any run, and this page uses the same program
(`web/js/flowmodules.js`, through `tools/flowmodules_cli.js`).

A module of one game and a module of the next are **the same module** when
they share enough agents: best overlap first, each module used once, and none
sharing less than 0.3 of their agents (the Jaccard index, shared agents over
all agents of either) — the view's rule. Identity is carried by overlap, not
by membership, so a module can go on while its members change.

Every world was read for fifty consecutive games: a lab run's from its
record, from iteration 950; a pilot's played on from its checkpoint at 1,000 —
183 worlds of every kind the notebook has made, each counted once. (Eighteen pilot
worlds replay a lab world, with its settings and seed, game for game; they are
left out.)

## Groups in every world

<!-- figure flowmodules/every_world -->
![Flow modules in every world](../figures/flowmodules/every_world.svg)

**Flow modules in every world.** Each dot is a world, over fifty games. Compression: how much shorter the modules make a description of a token wandering the flow (0: no group structure); the dashed bars are the same games' flow rewired at random, each agent's amounts kept. Size: agents per module. Who is in a module: the share of the agents that send or receive any flow, which is what puts an agent in a module. How fast members change: of a module followed from one game to the next, the share of its members that are new. B1 10k: the thirty B1 worlds of 10,000 tokens (Experiment 2); B1 fixed: the same with brains that never change (Experiment 7); B1 100k: B1 at 100,000 tokens (Experiment 10); B1 sizes: B1 at 800 to 409,600 tokens (Experiment 8) and at 5,000 with and without a hidden layer (Experiments 12 and 13); B2 10k: the ten B2 worlds of Experiment 16; B2 var: B2 with one rule or rate changed (Experiments 9, 15 and 16); B2 5k: B2 at 5,000 tokens, with all its rules or with one taken away (Experiment 14) — except lapse: the condition in which any unused link lapses; B1 pil. and B2 pil.: the pilot worlds that play the engine's game, except those that replay a lab world with its settings and seed.

> [!example]- How to make this figure
> **Runs.** Fifty consecutive games of each world: a lab run's from its record, from iteration 950 when it lived that long (up to five seeds of every condition); a pilot's played on from its checkpoint at 1,000 (`python3 research/atlas/flowmodules.py lab`, `pilots`).
>
> **Data.** From `GraphOfLifeRuns/.pilots/atlas/flowmodules/*.json`.
>
> 1. For every game, the flow between each pair of neighbours: the tokens staked either way.
> 2. Group the agents by the map equation (web/js/flowmodules.js, through tools/flowmodules_cli.js).
> 3. Follow the groups from game to game as the Research view does: best overlap first, above its floor of 0.3 of the members two modules share.
> 4. Do the same on each game's flow rewired at random, every agent's amounts kept: the null.
>
> **To make it again:** `python3 book_figures.py flowmodules`.
<!-- /figure -->

The worlds fall into the same two kinds as everywhere else in the notebook.

**In B1's worlds** — and in every world where unused links lapse — the flow is
grouped, but loosely. The modules compress it by 0.36–0.48 (the medians of the
groups of worlds), where the same flow rewired at random compresses by about
0. A module holds about four agents, and every agent is in one: B1's agents
spread their stake over all their neighbours
([How the game is played](how-the-game-is-played.md)), so every agent sends and
receives. From one game to the next a module changes about an eighth of its
members (0.11–0.13).

**In B2's worlds** the flow falls apart into pairs. The modules compress it by
0.82–0.85 and hold 2.3–2.4 agents, and only about three agents in ten are in a
module at all. A B2 agent puts its whole stake on one node — its own in about
two games of three — and half of B2's agents hold a single token, which stays
at home ([How different are the agents?](how-different-are-the-agents.md)). So in a game most
B2 agents send nothing, and those that do send it to one neighbour: the two are
a module. Such flow compresses even when rewired at random (0.09–0.15), since
pairs are groups of a kind. A module changes a third of its members every game.

Some worlds stand apart within their kind:

- Where **brains never change** (Experiment 7), the modules never change
  either, in four worlds of five: these are the frozen worlds of
  [Do the brains matter?](do-the-brains-matter.md), the same agents playing the same
  game, game after game.
- Two of Experiment 15's rules leave almost nobody in a module: **children of
  any size** (1% of the agents) and **no old age** (4–7%). So do the pilot
  worlds of [Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md) in which links cost a
  token and nobody dies of age (2–8%). In those worlds hardly anyone stakes on
  a neighbour.
- Batch S's two pilot worlds with **lapsing links**, with and without gifts,
  are B1-like, as the lab's world with lapsing links is — the split
  [Lightning in every world](lightning-in-every-world.md) found for loops of flow.

## What the network makes, and what the agents make

Compression against flow rewired at random says that the flow has groups. It
does not say who made them. Rewired flow sends tokens between agents that are
not neighbours, which no token can, so any flow that keeps to the network
looks grouped beside it — the network itself has regions.

A fairer yardstick keeps the network. In every game, each agent keeps its stake
at home and the amounts it staked on neighbours, but the amounts are dealt
among its own neighbours at random. Everything is as it was except the agent's
choice of whom to stake on (`tools/flowmodules_cli.js --neighbours`). What the
real flow has beyond this is the agents' doing; the rest is the network's.

On the compression it changes nothing anywhere: 0.388 against 0.384 in a B1
world, 0.845 against 0.846 in a B2 world, and the same in the others measured.
In B1 that is no surprise: an agent that splits its stake evenly over its
neighbours gives every neighbour the same, and dealing equal amounts changes
nothing. B1's flow modules are its network's regions. Whether they last, and
whether B2's agents choose their partners, the next two sections ask.


## How long modules live

<!-- figure flowmodules/lives -->
![How long modules live](../figures/flowmodules/lives.svg)

**How long modules live.** Of all the modules fifty games hold, the share followed for at least so many games, as the Research view follows them (the median of each group's worlds; logarithmic axes, a share of 0 drawn at the bottom). The flow: as it was. Partners dealt: the same games with every agent's stakes on its neighbours dealt among the same neighbours at random — the network and every agent's amounts kept, its choice of whom to stake on undone. Members dealt: the real modules with their agents dealt among them at random, game by game — only the sizes kept. B1 fixed: two worlds whose brains never change, one frozen (every module lasts) and one teeming with births (almost none does), so their median lies between. B2 var: B2 without random inputs, at mutation rates 0.05 and 0.8, and at 5,000 tokens; lapse: B2 at 5,000 tokens where any unused link lapses.

> [!example]- How to make this figure
> **Runs.** The same fifty games of eighteen lab worlds: B1 at 10,000 tokens (three seeds), B2 at 10,000 (three), B1 at 100,000 (two), B1 whose brains never change (two), B2 without random inputs (two), B2 at mutation rates 0.05 and 0.8 (one each), B2 at 5,000 tokens (two) and with lapsing links (two) (`python3 research/atlas/flowpersist.py`).
>
> **Data.** From `GraphOfLifeRuns/.pilots/atlas/flowpersist/*.json`.
>
> 1. For every game, the flow between each pair of neighbours: the tokens staked either way.
> 2. Group the agents by the map equation (web/js/flowmodules.js, through tools/flowmodules_cli.js).
> 3. Follow the groups from game to game as the Research view does: best overlap first, above its floor of 0.3 of the members two modules share.
> 4. Deal every agent's stakes on its neighbours among the same neighbours at random and do the same (`tools/flowmodules_cli.js --neighbours 1`).
> 5. Deal each game's agents at random among modules of the sizes that game had, and follow those (`--shuffle 1`).
>
> **To make it again:** `python3 book_figures.py flowmodules`.
<!-- /figure -->

Eighteen lab worlds of every kind were read again (the same fifty games) and
followed three ways: as they were, with every agent's stakes dealt among its
neighbours, and with each game's agents dealt at random among modules of the
sizes that game had — only the sizes kept.

- **B1**: a third of the modules are followed for five games or more, and one
  in six for ten or more (0.32 and 0.17, the median of B1's worlds at 10,000
  and at 100,000 tokens alike). With the agents dealt among modules, almost
  none last (0.003 and 0): B1's modules are no accident of their sizes. But
  with every agent's partners dealt among its neighbours, nearly as many last
  (0.25–0.28 and 0.13–0.14): it is the network that holds them together. The
  world with lapsing links is the same (0.33 and 0.17, against 0.31 and 0.16).
- **B2**: pairs rarely outlive a few games — 9% are followed for five games,
  1% for ten. With partners dealt among neighbours, fewer than half as many
  last five games (4%), and an eighth as many ten (0.1%). B2's agents keep a
  partner longer than chance would: the one structure in the flow that the
  agents themselves add. They keep it longest without random inputs: in
  Experiment 16's two worlds of that kind measured, about two pairs in a
  hundred last ten games, eight to nine times as many as with partners dealt
  at random — the dice B2's brains read
  ([Why evolution stops](why-evolution-stops.md)) break up partnerships too.
- **Brains that never change**: in the frozen world every module lasts all
  fifty games — and with partners dealt at random only a quarter last ten. A
  world whose agents repeat the same moves forever keeps the same groups; the
  groups are those moves, fixed.

Some long-lived modules outlast the agents they began with. Of B1's modules
followed for ten games or more, a fifth end with none of their first members
(0.21 at 10,000 tokens, 0.18 at 100,000), and on average two thirds of the
first members are still there. But the same is true with partners dealt at
random (0.21 and 0.17): the network's regions outlast their occupants, not
groups the agents hold together. In B2, of the pairs followed for ten games —
about four times as many as with partners dealt at random — half end with
neither first member.


## Two agents of one module, games later

<!-- figure flowmodules/together -->
![Two agents of one module, games later](../figures/flowmodules/together.svg)

**Two agents of one module, games later.** Of two agents in one module, the share still in one module so many games later, over the pairs whose agents are both in a module then (the median of each group's worlds). The flow: as it was; partners dealt: every agent's stakes dealt among its own neighbours at random; members dealt: the later game's agents dealt at random among modules of its sizes. No module is followed, so no rule for following one is needed.

> [!example]- How to make this figure
> **Runs.** The same fifty games of eighteen lab worlds: B1 at 10,000 tokens (three seeds), B2 at 10,000 (three), B1 at 100,000 (two), B1 whose brains never change (two), B2 without random inputs (two), B2 at mutation rates 0.05 and 0.8 (one each), B2 at 5,000 tokens (two) and with lapsing links (two) (`python3 research/atlas/flowpersist.py`).
>
> **Data.** From `GraphOfLifeRuns/.pilots/atlas/flowpersist/*.json`.
>
> 1. For every game, the flow between each pair of neighbours: the tokens staked either way.
> 2. Group the agents by the map equation (web/js/flowmodules.js, through tools/flowmodules_cli.js).
> 3. For every two games lag apart, the pairs of agents sharing a module in the first that are both in a module in the second; the share that share one there.
> 4. The same on the flow with every agent's stakes dealt among its neighbours (`--neighbours 1`), and with the later game's agents dealt among modules of its sizes (three deals).
>
> **To make it again:** `python3 book_figures.py flowmodules`.
<!-- /figure -->

Following modules needs a rule for when a module of one game is the same as
one of the next. A plainer question needs none: of two agents in one module,
how often are they in one module again so many games later, when both are in
a module then?

- **B1**: of two agents in one module, 0.64 are together a game later and 0.46
  forty games later (the median of B1's worlds at 10,000 tokens). With
  partners dealt among their neighbours: 0.58 and 0.45 — nearly the same. At
  100,000 tokens, where modules are a little larger, more pairs stay (0.83 and
  0.70), and as many with partners dealt (0.79 and 0.67).
- **B2**: 0.53 are together a game later and 0.37 forty games later. With
  partners dealt among neighbours: 0.38 and 0.33. So B2's agents go back to
  the same partner more often than chance for a few games; over forty, hardly
  more often than their neighbourhood makes them.
- **Lapsing links**: 0.66 together a game later, but only 0.16 forty games
  later, as with partners dealt (0.63 and 0.16): where unused links lapse, the
  network itself changes fast.
- Dealt among modules of the same sizes, two agents are together again less
  than once in a hundred: being neighbours is what keeps pairs together.


## Where new members come from

<!-- figure flowmodules/renewal -->
![Where a module's new members come from](../figures/flowmodules/renewal.svg)

**Where a module's new members come from.** Of the new members of modules followed from one game to the next, the share that had been born since (the rest were alive before, in another module or none). A module that renews itself from births regenerates; one that takes in agents from around it exchanges them. Each dot is a world. B1 10k: the thirty B1 worlds of 10,000 tokens (Experiment 2); B1 fixed: the same with brains that never change (Experiment 7); B1 100k: B1 at 100,000 tokens (Experiment 10); B1 sizes: B1 at 800 to 409,600 tokens (Experiment 8) and at 5,000 with and without a hidden layer (Experiments 12 and 13); B2 10k: the ten B2 worlds of Experiment 16; B2 var: B2 with one rule or rate changed (Experiments 9, 15 and 16); B2 5k: B2 at 5,000 tokens, with all its rules or with one taken away (Experiment 14) — except lapse: the condition in which any unused link lapses; B1 pil. and B2 pil.: the pilot worlds that play the engine's game, except those that replay a lab world with its settings and seed.

> [!example]- How to make this figure
> **Runs.** Fifty consecutive games of each world: a lab run's from its record, from iteration 950 when it lived that long (up to five seeds of every condition); a pilot's played on from its checkpoint at 1,000 (`python3 research/atlas/flowmodules.py lab`, `pilots`).
>
> **Data.** From `GraphOfLifeRuns/.pilots/atlas/flowmodules/*.json`.
>
> 1. For every game, the flow between each pair of neighbours: the tokens staked either way.
> 2. Group the agents by the map equation (web/js/flowmodules.js, through tools/flowmodules_cli.js).
> 3. Follow the groups from game to game as the Research view does: best overlap first, above its floor of 0.3 of the members two modules share.
> 4. A new member is born if its id was not alive the game before.
>
> **To make it again:** `python3 book_figures.py flowmodules`.
<!-- /figure -->

A module changes members in two ways. An agent born since the last game can
join it — the module **regenerates**, making the agents it is made of — or an
agent that was already alive, in another module or none, can move in — the
module **exchanges** its members with those around it, and its border moves.

Modules renew mostly by exchange. Of a B1 module's new members, 12–15% were
born since the game before (the medians of the groups of B1 worlds); of a B2
module's, 4%. Only where agents are born and die often does a module renew
from births: in the B1 world whose modules lose 4.5% of their members to death
every game, three new members in four were born, and in the pilot worlds of B1
without revolutions, where agents come and go faster, between a third and four
fifths.

## What this says

- **Tokens keep to groups in every world, and the groups are the network's.**
  In B1 a module is about four agents of one region of the network. Dealing
  every agent's stakes among its neighbours at random gives the same modules,
  lasting as long. Agents that split their stake evenly cannot choose a group;
  their neighbourhood is their group.
- **B2's flow is pairs, and most agents take no part in it.** In a game two
  agents in three stake nothing on anyone. A pair lasts a game or two, but
  pairs that last five games are twice as common as with partners dealt at
  random among the same neighbours: the only structure in the flow that the
  agents themselves make — and they make more of it without random inputs.
- **Groups outlast their first members only as regions do.** A fifth of B1's
  long-lived modules end with none of the agents they began with, and a module
  renews mostly by agents moving in from around it rather than by births. The
  same happens when every agent's partners are dealt at random.
- **Some rules leave almost no flow at all.** Where children may be of any size,
  or nobody dies of age, almost no agent stakes on a neighbour.
- **For open-ended evolution**, a group that holds together while its members
  change is what a higher level of individual would be built from — the step
  from single cells to organisms that Maynard Smith and Szathmáry call a major
  transition. These worlds make no such group beyond the regions of their
  network. That fits what [Who helps whom](who-helps-whom.md) found for
  help: nothing in the game rewards staying with a partner. A group of the
  agents' own needs a rule that does — a support stake that counts only if it
  is returned, or a share in what a partnership wins. Alliances
  ([Alliances](alliances.md)) are such a rule, and were taken up at once;
  this measure, run on worlds with alliances, would show whether their groups
  outlast the network's.


**Related notes:** [Flow modules and the map equation](map-equation.md) ·
[Token flow](token-flow.md) · [Lightning](lightning.md) ·
[Flows that go round](circulation.md) ·
[How the game is played](how-the-game-is-played.md) ·
[Where the tokens flow](where-the-tokens-flow.md) ·
[Gliders, and who helps whom](gliders-and-who-helps-whom.md) ·
[Lightning in every world](lightning-in-every-world.md)

<!-- turns -->
---

← [What crosses a connection](what-crosses-a-connection.md) · [The game and the flow of tokens](../README.md#the-game-and-the-flow-of-tokens) · [Lightning in every world](lightning-in-every-world.md) →

**Also about:** [Groups and cooperation](../README.md#groups-and-cooperation).

> [!info]- Pages that link here (7)
> [Where tokens go](where-tokens-go.md) · [Lightning in every world](lightning-in-every-world.md) · [Flow modules and the map equation](map-equation.md) · [How alike the agents are](how-alike-the-agents-are.md) · [Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md) · [Every world side by side](every-world-side-by-side.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
