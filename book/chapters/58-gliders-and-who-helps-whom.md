# Gliders, and who helps whom

> [!question] Questions of this chapter
> - Does the game make patterns that last while the agents in them change, as
>   Conway's Game of Life makes gliders out of cells that are born and die?
> - Do tokens go round the network in rings, or run downhill?
> - What would cooperation be in a game whose only act is a stake, and how can
>   it be seen without knowing what an agent meant?
> - Is there any, in B2, in B1, and among allies?

## Why ask

The measures of Part V look at agents one by one: what they hold, what they
stake, how their brains do in a contest. [Chapter 50](50-why-evolution-stops.md)
found one blind spot in them, contests too short for what pays over a lifetime.
The author suggested another: perhaps the worlds hold structures above the
single agent that none of the measures can see, cyclic ones like the gliders of
the Game of Life, and perhaps harder to spot because they live on a network
rather than a grid.

In Conway's Game of Life the interesting objects are not cells but
**patterns**. A glider is five live cells that rebuild themselves one cell
further on, diagonally, every four generations; the cells that make it are born
and die, the glider goes on. Patterns are sorted by what they do. A *still life*
never changes. An *oscillator* repeats in place: the *blinker*, three cells in
a row that turn into three in a column and back. A *spaceship*, the glider
among them, moves. A *puffer* moves and leaves debris behind it, and a *gun*
stays where it is and sends out gliders. Much of what is known about how far
the Game of Life can go, up to machines that compute anything, is built out of
such patterns and the ways they meet.

A network has no shapes: no diagonal, no row of three. A pattern has to be
recognised by what it does, its motion measured in steps through a network
that itself changes, and every candidate compared with what chance would make,
since a world of 3,000 agents makes almost any small pattern somewhere by
accident.

Cooperation is the other thing such structure could hide.
[Chapter 56](56-alliances.md) *gave* the agents a way to cooperate. It did not
ask whether the worlds without alliances cooperate in some way of their own.

This chapter reads the record of play. B2's ten worlds of Experiment 16 and the
first ten worlds of B1's baseline set record every game with every agent's
stakes, and the games of iterations 1,000 to 2,000 are read: a settled world,
long after the founders. For alliances, pilot worlds of [Chapter 56](56-alliances.md) are played
on for 200 iterations from a checkpoint, recording their games in the same form
(`research/oee/checks/pilot_frames.py`), together with the same pilots without
alliances as their control. Every check is a script in `research/oee/checks/`.

## Chains of takeovers

The game moves two things over the network: tokens, and copies of brains. When
an agent takes a neighbour's node, the node gets a copy of its brain and every
token staked on it. Call a takeover made with at least half of the taker's
tokens a **move**: the agent's wealth and a copy of its brain go to the node
taken, together, as a packet. If, in the next game, the node taken makes a move
of its own, the packet has travelled on. A run of such moves, each made from the
node the last one took, is a **chain**.

What a chain leaves behind depends on the node it left. If the mover still
holds its own node after the game, the chain is a puffer: it leaves a copy at
every step. If the node it left is taken by someone else, or starves, the chain
is a glider: it carries the brain on and leaves nothing. A chain that goes back
to the node it came from is a blinker. A node that launches chain after chain
while staying where it is is a gun.

![Patterns of takeovers](../diagrams/patterns.svg)

Chance is the measure of all of these. If every node holding a packet moved it
on with the same chance as any agent makes a move in a game, whatever came
before, chains would be short: with that chance about 0.07 a game in both rule
sets, one chain in fifteen would make a second move and one in two hundred a
third.

<!-- figure patterns/chains -->
![How long the chains of takeovers get](../figures/patterns/chains.svg)

**How long the chains of takeovers get.** One line per world. A move is a takeover made with at least half of the taker's tokens; a chain is a run of moves, each made from the node the last one took. Dashed: the chains there would be if every node holding a packet moved it on with the same chance as any agent moves one in a game, whatever came before.

> [!example]- How to make this figure
> **Runs.** The ten B2 worlds of Experiment 16, `B2-10000-s001` … `-s010`, and the first ten worlds of the baseline set, `B1-10000-s001` … `-s010` (B1 at 10,000 tokens, Experiment 2). Both record every game with every agent's stakes; read are the games of iterations 1,000 to 2,000.
>
> **Data.** From each run's frames of the game phase, with its agents' decisions.
>
> 1. In every game, find the moves: an agent's stake on a neighbour's node that took it and was at least half of the agent's tokens (`research/oee/checks/gliders.py`).
> 2. A move made from a node that a move took in the game before continues that chain; any other starts one. Count the chains by their number of moves.
> 3. Chance: with p the share of agents that make a move in a game, averaged over the games (`structures.py`, `packet_go_on_chance`), and N the chains, N (1 − p) p^(L−1) chains would have L moves.
>
> **To make it again:** `python3 book_figures.py patterns`.
<!-- /figure -->

**In B2 almost every takeover is a move**: 99.6% of the nodes taken were taken
with at least half of the taker's tokens. Chains of three moves or more are 23
times as many as chance would make (18 to 54 in the ten worlds), and 31% of
all takeovers belong to them. The longest of a world has 18 to 32 moves. **In
B1** only 11% of takeovers are moves, since its agents spread their tokens over
every neighbour; chains of three or more are 11 times chance (6.5 to 26), and
the longest are longer, 69 to 175 moves.

<!-- figure patterns/zoo -->
![What the chains are](../figures/patterns/zoo.svg)

**What the chains are.** One dot per world, the bar the median of ten. Chains of at least three moves, over games 1,000 to 2,000. Top right: after how many of their moves the mover still held its own node, a puffer leaving copies behind (high) or a glider leaving nothing (low). Bottom left: how often a move went back to the node the packet held two moves before, divided by the chance of picking that node among the mover's neighbours. Bottom right: the distance from a chain's first node to its last, divided by a random walk's of the same length from the same node, for the 300 longest chains of each world.

> [!example]- How to make this figure
> **Runs.** The ten B2 worlds of Experiment 16, `B2-10000-s001` … `-s010`, and the first ten worlds of the baseline set, `B1-10000-s001` … `-s010` (B1 at 10,000 tokens, Experiment 2). Both record every game with every agent's stakes; read are the games of iterations 1,000 to 2,000.
>
> **Data.** From each run's frames of the game phase, with its agents' decisions.
>
> 1. Find the chains as for the figure above (`research/oee/checks/gliders.py`).
> 2. Chance of a chain of at least three moves: N p² for N chains and p as above (`structures.py`).
> 3. For every move, note whether the mover won its own node in the same game.
> 4. For every move from the third on, note whether it went to the node the packet held two moves before, and the chance 1/k of that node among the mover's k neighbours.
> 5. Distance: steps through the network at the chain's end between its first and last node; the walk's: the median of five random walks with as many steps from the first node.
>
> **To make it again:** `python3 book_figures.py patterns`.
<!-- /figure -->

**B2's chains are puffers, B1's are gliders.** In B2 the mover still holds the
node it left after 93% of its moves and never leaves it to starve: a chain is
a lineage copying itself outward, node by node, and its packet is small, three
tokens at the start and two at the end in the median of each world's 300
longest chains. In B1 the mover keeps the node it left after only 9% of its
moves; after 8% that node starves, and after the rest it is taken by another
agent. The brain moves on, and the packet grows as it goes, from 18 tokens to
82: B1's gliders are rich agents travelling the network and gathering what is
staked on the nodes they take. In both, the brain changes as it goes: in B2
each copy is made with mutations, and in B1 every brain may mutate after every
game. After 70% of the moves the chain's holder no longer carries the genotype
it started with.

Drawn on their networks, one chain of each kind, the one among each world's
longest that took the most different nodes: B2's took 28 nodes in 31 moves,
patch after patch, and B1's took 66 in 100, round and round one region of its
world, carrying a fortune that grew from 380 tokens to 1,358, a seventh of all
the tokens in that world.

<!-- figure patterns/example -->
![Two chains on their networks](../figures/patterns/example.svg)

**Two chains on their networks.** Left, in `B2-10000-s001`, the chain of its 300 longest that took the most different nodes: 31 moves over 28 nodes in games 1,120 to 1,150, its packet 4 tokens at the first move and 2 at the last. Right, the same for `B1-10000-s001`: 100 moves over 66 nodes, games 1,544 to 1,643, its packet 380 tokens, then 1358. Drawn is the network of each chain's last game around the nodes it took, with their neighbours; the chain's moves are the coloured links, drawn even where the connection a move crossed has gone since. A node the chain took that no longer exists at its last game is not drawn, the node where it began among them.

> [!example]- How to make this figure
> **Runs.** The ten B2 worlds of Experiment 16, `B2-10000-s001` … `-s010`, and the first ten worlds of the baseline set, `B1-10000-s001` … `-s010` (B1 at 10,000 tokens, Experiment 2). Both record every game with every agent's stakes; read are the games of iterations 1,000 to 2,000.
>
> **Data.** From the two runs' frames of the game phase, with their agents' decisions.
>
> 1. Track the chains as for the figures below and keep the 300 longest of each world, with the nodes each took (`research/oee/checks/gliders.py`).
> 2. Of those, take the chain that took the most different nodes.
> 3. Take the network of its last game, the nodes it took that are still there and their neighbours; add the moves as links; lay it out with `networkx.forceatlas2_layout(G, max_iter=300, seed=1)`.
>
> **To make it again:** `python3 book_figures.py patterns`.
<!-- /figure -->

**They blink a little.** A move goes back to the node the packet held two
moves before 1.37 times as often as picking among the mover's neighbours at
random would (1.31 to 1.47) in B2, and 1.27 times in B1. In B2 the likely
reason is in the rules: the node a packet came from has just staked at least
half of its tokens away, so it is now among the weakest of the new holder's
neighbours. Going round a triangle, three moves back, is rare in both and in
B1 no more common than chance.

**They wander.** A chain gets no farther from where it started than a random
walk of as many steps from the same node on the same network (0.8 of the walk's
distance in B2, 1.0 in B1). A glider in the Game of Life flies straight; these
have no direction. Nothing an agent senses says from which neighbour it came:
its inputs are its own and each neighbour's tokens and connections, messages
and noise. Messages could carry it in principle, but the brains write the name
of their genotype in them ([Chapter 20](20-what-agents-say-to-each-other.md)),
and along a puffer's trail every neighbour carries the same name.

**Guns fire without a rhythm.** The nodes that launch chains launch them very
unevenly: the variance of launches per launching node is 2.6 times their mean
in B2 and 13 times in B1, and the busiest node of a world launched 34 chains in
1,000 games in B2 and 128 in B1. But the gaps between one node's launches fall
off steadily from a single game; there is no period at which a gun fires.

**Musical chairs, at chance.** In B1, three quarters of all takeovers are made
by an agent whose own node is taken in the same game: everyone moves up a seat.
Two agents taking each other's nodes in one game, a swap, happens as often as it
would if every taker had picked its targets at random among its neighbours
(1.02 times as often, 0.99 to 1.04 in the ten worlds). Loops of three agents or
more, each taking the next one's node in the same game, happen 1,500 to 4,800
times in a world's 1,000 games, up to fourteen agents round one loop; how many
chance would make is not known. In B2 swaps are rarer than chance (0.54 to
0.85 of it) and loops of three or more almost never happen: 0 to 8 in a world's
1,000 games, among a quarter of a million takeovers.

## Channels and feuds

A stake from one agent on the same neighbour game after game is a
**channel**. In B2 a channel that carried tokens in one game carries them again
in the next 1.5 times as often as chance (0.117 against 0.077, chance being
that the agent picked as many of its neighbours as it did, at random). In B1
the agents stake on almost every neighbour every game, so channels repeat
92.6% of the time, against 93.6% by chance.

Two agents staking on each other for three games or more look, in a record of
stakes, like partners. In B2 they are not. Agents in such a pair keep their own
node less often than other agents that stake out, 0.80 against 0.89, in every
one of the ten worlds: in B2 a stake on someone's node is an attack, and two
agents attacking each other game after game are in a **feud**. In B1 the pairs
keep their nodes about as often as others (0.44 against 0.41, more often in
seven worlds and less often in three). In the pilots without alliances the
feuds are worse, 0.49 against 0.86 to 0.88.

Among allies, most lasting pairs are allied: 85% and 56% in the two worlds,
and in their game an ally's stake counts for the owner's side. Yet agents in
such pairs keep their nodes *less* often than other agents that stake out
(0.94 against 0.97, 0.91 against 0.95). Help goes where it is needed, so those
who get it are those under attack, and comparing them with the rest measures
the attack, not the help. To see help, it has to be defined by what it does
in the game itself. That comes after the tokens.

## Where the tokens go

Every stake moves tokens over a connection, to the node staked on. Over a
stretch of 50 games each connection carries a net flow, what went one way
minus what came back, and that flow splits exactly into a part that runs
downhill, from some agents to others, and a part that goes round loops (the
note [*Flows that go round*](../notes/circulation.md)). Tokens going round a
ring of agents, each giving to the next, would be the Kula ring of the
Trobriand Islands that Bronisław Malinowski described: armshells and necklaces
passed from island to island, round and round. Chance keeps each connection's
amount and throws its direction at random.

<!-- figure patterns/flows -->
![Where the tokens go](../figures/patterns/flows.svg)

**Where the tokens go.** One dot per world, the bar the median. The tokens each connection carried, net, over 50 games, split into the part that runs downhill and the part that goes round loops (the note *Flows that go round*). Top: how much more of the flow goes round than if each connection's net flow had a random direction; above zero it goes round, below it runs downhill. Gifts are stakes that did not take their target, takeovers stakes that did. Bottom: how much of each part comes back on the same connections in the next 50 games.

> [!example]- How to make this figure
> **Runs.** The ten B2 worlds of Experiment 16, `B2-10000-s001` … `-s010`, and the first ten worlds of the baseline set, `B1-10000-s001` … `-s010` (B1 at 10,000 tokens, Experiment 2). Both record every game with every agent's stakes; read are the games of iterations 1,000 to 2,000. Pilot worlds of [Chapter 56](56-alliances.md) (`GraphOfLifeRuns/.pilots`): with alliances `A10k/allies-s1` and `-s2` from iteration 500 and `AL/allies10k-s3` and `-s4` from iteration 2,000; without, `A10k/ctrl-s1` and `-s2` from iteration 500. Each goes on from its checkpoint for 200 iterations with every game recorded in the lab's form (`python3 research/oee/checks/pilot_frames.py <world> --at <iteration> --games 200`).
>
> **Data.** From the recorded games.
>
> 1. Add up, over each stretch of 50 games, the tokens staked from each node on each neighbour's node, one way minus the other, separately for gifts and takeovers (`research/oee/checks/circulation.py`).
> 2. Find the pressure that explains as much of that net flow as possible (least squares, by conjugate gradients on the network's Laplacian); the circulation is the rest. Its share is its sum of squares over the flow's.
> 3. Chance level: the same with each connection's direction thrown at random, four times, averaged.
> 4. Persistence: over the connections two consecutive stretches both used, the correlation of their downhill parts, and of their circulations.
> 5. Each world's number is the median over its stretches.
>
> **To make it again:** `python3 book_figures.py patterns`.
<!-- /figure -->

**Gifts run downhill, and keep running down the same slopes.** A gift here is
a stake that did not take its target; its tokens went to whoever held the
node. In all twenty worlds less of the gift flow goes round than chance would
make (0.09 less in the median B2 world, 0.13 in B1), so gifts flow from some
agents to others. The slope lasts: the downhill part of one stretch correlates
with the next stretch's on the same connections, 0.20 in B2 and 0.50 in B1. In
B1, whose agents spread their stakes over every neighbour, the gifts drain
towards the best-connected agents (in two worlds checked, the rank correlation
of an agent's height on the slope with its connections is −0.28 and −0.33):
the even split of [Chapter 15](15-where-do-the-tokens-go.md) again, tokens
settling in proportion to connections. In B2, whose gifts are failed raids,
they drain weakly towards the richer (−0.16 in two worlds): tribute, paid by
failed raids to whoever withstood them.

**Takeovers go round, but never in the same place twice.** More of the
takeover flow goes round than chance would make, in every world: 0.07 more in
B2, 0.24 more in B1. But the circulation of one stretch has nothing to do with
the next's: the correlation is 0.00 in both. The loops are made and gone
within 50 games, the chains above being their likeliest makers, since a route
through a network counts as circulation even when it never closes.

**With alliances, gifts go round.** The pilots with alliances are the only
worlds in which more of the gift flow goes round than chance would make: 0.07
and 0.04 more at iteration 500, where the same pilots without alliances are
0.13 and 0.08 *below*, and 0.14 and 0.07 more in the two long worlds at
iteration 2,000, when one alliance holds almost all of their agents. Allies give
to allies who give to allies. But these rings do not last either: their
circulation correlates with the next stretch's at −0.02 to 0.06. Nowhere does
the circulation of gifts come back much: 0.02 in B2, 0.09 in B1.

## Help

Cooperation needs an act that helps. In this game every act is a stake, so
what helps must be a stake, and whether a stake helped can be read from the
rules (the note [*Help*](../notes/help.md)). Take any stake that agent *u* put
on a neighbour *v*'s node, and ask how likely *v* was to keep its node, first
with every stake as it was and then without *u*'s, everyone else's unchanged.
Both chances follow from the world's own rule for deciding a node, counted
exactly. If *v* was more likely to keep its node with *u*'s stake than without
it, the stake **helped**.

Only two rules make help possible. In a revolution, a crowd of small
revolutionary stakes overthrows the largest staker, so a neighbour's stake on
*v*'s node can join *v*'s own stake in the crowd that saves it. B1 and B2 both
have revolutions. With alliances, an ally's stake counts for *v*'s side. Without
either, every stake on another's node is a bid against its holder and can only
lower its chance: the pilots without alliances have neither, and in them no
stake ever helped.

<!-- figure patterns/help -->
![Who helps whom](../figures/patterns/help.svg)

**Who helps whom.** One dot per world, the bar the median. A stake on a neighbour's node helps if, without it, the neighbour would have been less likely to keep its node (the note *Help*). Top left: the nodes so saved, counted as the sum of those effects, per 100 agents and game; the pilots without alliances, in which no stake can help, are drawn at 0.001. Top right: the pairs in which help went both ways within a stretch of 50 games, summed over the stretches, divided by the number expected from how long each pair were neighbours; worlds in which chance expects fewer than one have no dot. Bottom: whether the net help between agents goes round loops more than chance (only where stretches have at least 100 helping pairs and their help closes loops), and whether help goes to agents carrying the helper's brain more than such agents are among its neighbours (only where at least 100 stakes helped).

> [!example]- How to make this figure
> **Runs.** The ten B2 worlds of Experiment 16, `B2-10000-s001` … `-s010`, and the first ten worlds of the baseline set, `B1-10000-s001` … `-s010` (B1 at 10,000 tokens, Experiment 2). Both record every game with every agent's stakes; read are the games of iterations 1,000 to 2,000. Pilot worlds of [Chapter 56](56-alliances.md) (`GraphOfLifeRuns/.pilots`): with alliances `A10k/allies-s1` and `-s2` from iteration 500 and `AL/allies10k-s3` and `-s4` from iteration 2,000; without, `A10k/ctrl-s1` and `-s2` from iteration 500. Each goes on from its checkpoint for 200 iterations with every game recorded in the lab's form (`python3 research/oee/checks/pilot_frames.py <world> --at <iteration> --games 200`).
>
> **Data.** From the recorded games.
>
> 1. For every stake u → v on another agent's node, the chance that v keeps its node with all stakes and without u's, from the world's rule for deciding a node counted exactly: the largest stake, the owner's counting 1.5 times in B2 and the pilots, ties drawn, and the revolution's crowd (B1, B2), or the sides of allies (the pilots with alliances) (`research/oee/checks/decisive.py`). The difference is the stake's effect; above zero it is help.
> 2. Saved: the sum of the positive effects over the games, per 100 agents and game.
> 3. Returned: in each stretch of 50 games, the pairs with help from u to v that also had help from v to u; chance, for each such pair, 1 − (1 − q)^n for the n games in which v had u for a neighbour and q the help per neighbour and game of the stretch; both summed over the stretches.
> 4. Round: the net help between two agents in a stretch, split as the token flow above, minus the same for random directions.
> 5. Clones: the share of the help that went to an agent with the helper's brain id, divided by the share of such agents among the helper's neighbours, both weighted by the help.
>
> **To make it again:** `python3 book_figures.py patterns`.
<!-- /figure -->

**In B2 nobody helps.** In its ten worlds, others' stakes saved 0.005 to 0.028
nodes per 100 agents a game, against 6 to 7 nodes lost to them. Revolutions
are on in B2, but its agents stake all-in on one target, so the small stakes a
crowd is made of are rare, and the owner's one and a half counts against any
crowd. Evolution made it rarer still: in three worlds measured at three times,
help fell from 0.03–0.07 nodes per 100 agents a game at iteration 100 to
0.002–0.012 at iteration 1,900, while the nodes lost to attacks fell from 11–14
to 6.

What little help there is in B2 passes both ways. In a stretch of 50 games, two
neighbours that helped each other are rare, but over the ten worlds there were
156 such pairs where chance, from how long the pairs were neighbours, would make
28, and more than chance in every world. It goes to clones 1.2 times their share
of neighbours. Both fit a situation better than a choice. Help needs two
neighbours whose revolutionary stakes can join against the same attacker, and
that situation is the same seen from either of them, so help that passes in it
passes both ways. And nothing an agent senses says who staked on its node: no
agent could repay help if it wanted to.

**In B1 neighbours save each other's nodes**: 2.4 nodes per 100 agents a game
in the median world (1.5 to 4.4), a thirteenth of the 31 lost to attacks,
through the crowds of revolutionaries that B1's agents make by spreading a
little over every neighbour; 2.3% of all stakes on others' nodes help. But
nothing about this help is cooperation, in any of the ten worlds. It is not
returned: of the pairs in which one agent helped the other within 50 games, the
other helped back in 40%, less than the 53% expected from how long those pairs
were neighbours (0.76 times chance, summed over the stretches), and less in
every world. It does not go round: less of the net
help circulates than if each pair's had a random direction (0.08 less), in
every world. And it does not stay in the family: clones get 1.03 times their
share of neighbours. It is help by accident, the by-product of a habit, as
[Chapter 19](19-what-the-brains-are-like.md) found the habit itself to be.
Since iteration 100 it has doubled, from 0.9–1.4 nodes saved per 100 agents a
game to 2.0–3.7 at iteration 1,000, and has gone no further since, rising in
one world and falling in two by iteration 2,850.

**Allies hardly rescue each other either.** Allies put a third to almost all of
their stakes on each other's nodes (36% and 69% of all stakes at iteration 500,
95% and 99% in the long worlds at 2,000), and almost none of them decides
anything: others' stakes saved 0.016 to 0.085 nodes per 100 agents a game in the
four alliance worlds, 28 to 150 times less than in B1.
Yet agents with an ally keep their node in 99–100% of games, and agents without
one in 90–95%. An alliance protects by **peace**, not by rescue: a stake from an
ally counts for its holder, so an allied neighbour can never take the node, and
the neighbour that would have been the threat is the one that has been turned.
In the long worlds, where one alliance holds nearly everyone, nodes are lost to
attacks at 0.06 and 0.5 per 100 agents a game, against 6 to 7 in B2. The stakes
allies put on each other's nodes are gifts to holders who would have kept their
nodes anyway.

The little rescue there is among allies passes both ways too: 26 pairs rescued
each other where chance would make 3.6 in the alliance world with the most
rescue, and 10 against 0.1 and 6 against 0.5 in the long worlds. It goes to
clones 1.1 times their share, and it does not go round. Here an answer would be
possible, since the agents of these pilots see the stakes of the last game, but
it cannot be told apart from the rule: only allies can rescue each other, and
an alliance is mutual by definition, so rescue that passes between allies
passes both ways.

## What cooperation is here

Three things go by the name, and the record of play can tell them apart.

- **Rescue**, a stake without which a neighbour would have lost its node. It
  needs a rule that lets one agent's stake count for another's: revolutions,
  or alliances.
- **Peace**, neighbours that do not take each other's nodes. With alliances it
  is a rule: allies cannot.
- **Exchange**, tokens given that come back round instead of running downhill.
  It shows in the flow of gifts as circulation above chance.

None of them is in B2. B2's worlds are made of puffers, feuds and tribute:
lineages copying themselves outward, pairs attacking each other, and failed
raids feeding those who withstood them. B1 has rescue, as a by-product that
nobody returns. Alliances bring peace and exchange, and with exchange the one
ring of this chapter, gifts going round, though it does not last fifty games on
the same connections. Where rescue is rare, in B2 and among allies, it passes
both ways more often than chance; but the situations that allow it are the same
seen from either side, and in B2 no agent could know whom to repay. Returned
help is the trace reciprocity would leave, not yet reciprocity.

So the cyclic structures the author asked about exist: tokens go round loops
in every world, and chains of takeovers blink back and forth. But none of them
is a lasting structure on the network the way a glider is in the Game of
Life. Every ring found is gone in fifty games, and every chain wanders like a
drunkard. What lasts is the slope: tribute that keeps flowing down the same
connections, stretch after stretch.

## What it means for open-ended evolution

The Game of Life goes far because its patterns are reliable. A glider keeps
its shape and its direction until it meets something, and what happens when it
does is always the same, so patterns can be combined into larger ones. The
patterns of these worlds are not reliable in either way: a chain has no
direction and changes its brain at every move, and a ring dissolves within
fifty games. Help is no more reliable: who rescued whom is something no agent
can know. Two inputs would change that, and neither needs a number:

- **A sense of direction.** One more input per neighbour, *this is the node I
  came from*, would let a chain keep going away from where it was, or come
  back on purpose. Brains could then make gliders that travel instead of
  wander, and a signal could cross a world.
- **A sense of who helped.** One more input per neighbour, *its stake saved my
  node in the last game*, computed by the rules exactly as this chapter
  computes help. [Chapter 56](56-alliances.md) found support given up because
  "whoever receives it owes nothing back"; an agent that can see who rescued it
  can owe something back. Rescue could then be returned on purpose, by revolution
  in B1 or by support, without a pact, and the measures of this chapter would
  show whether it was.

The measure itself is the third result. Help computed from the rules, nodes
saved per 100 agents a game, with its three signs of cooperation against
chance, can be taken of any world that records its games. For the worlds so far
it is not a statistic that keeps rising: in B1 it doubles early and then stays
where it is, and in B2 it falls.
