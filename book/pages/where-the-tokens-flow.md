# Where tokens flow

> [!question] Questions of this page
> - Every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), every [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") stakes all its [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."). How many stay at home, how
>   many cross a [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") and are matched by a stake coming back, and how
>   many go one way only?
> - How evenly do agents spread their stakes? Can a game be described as a
>   random walk of tokens, and what would that predict about who is rich and
>   who keeps their node?
> - How many tokens cross one connection in a game, and which connections
>   carry nothing, and so are cut?
> - Do tokens go round in closed [loops](loops.md "A path through the network that comes back to where it started without using a connection twice.")? How long are the loops, and does a
>   quick count, game by game, agree with an exact one?
> - Is that the same in every world the notebook has made: both rule sets,
>   every size, every setting an experiment changed?

## In short

In every game, each agent stakes all its tokens on itself and its [neighbours](the-world.md#the-network "An agent joined to another by a connection."),
and whoever wins a node keeps what was staked on it. In a settled [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") world, 30%
of the tokens stay home, 51% are matched by a stake coming back, and only 20%
move one way. The [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") spread their tokens almost evenly, so a game is close
to a random walk of tokens, and wealth follows connections. Most connections
carry one or two tokens a game, one each way, and in B1 a connection that
carries nothing is cut. Once returned stakes cancel, almost nothing goes round
in loops, in any world. How much goes round before cancelling is set by one
rule, not by the brains: whether a link must carry tokens to survive.

## Where the tokens flow

In every game, each agent stakes all its tokens on its own node and on its
neighbours' nodes. Every token staked ends up with whoever wins the node it was
staked on. So tokens move along the connections, game after game.
[Gains and losses](gains-and-losses.md) follows the agents through a game: who
gains, who loses. This section follows the tokens.

Where does each staked token go, and how much of it comes back? Does the whole
movement obey a simple law? It does, nearly, and the law explains several
findings of other pages at once.

The worlds here are the thirty B1 worlds of 100,000 tokens of Experiment 10
([Thirty worlds](a-worlds-life.md#thirty-worlds)). This section was first
written from the thirty worlds of 10,000 tokens. Where those differ, both
numbers are given.

### The budget of a game

Start with the bookkeeping. In every game of these worlds, exactly 100,000
tokens are staked: every token, by whoever holds it. Each staked token lands in
one of three places ([Token flow](token-flow.md)):

1. on the staker's **own node**;
2. on a neighbour's node, **matched** by a stake coming back. Say A puts 5 on
   B's node, and B puts 3 on A's. Then 3 tokens go each way and cancel:
   neither balance changes because of them;
3. on a neighbour's node, **one way**: the 2 left over in the example. Only
   these move tokens between agents.

<!-- figure flow/budget -->
![Where the tokens of a game go](../figures/flow/budget.svg)

**Where the tokens of a game go.** Every token is staked in every game. Yellow: the share staked by agents on their own node. Cyan: staked on a neighbour's node, but cancelled by what that neighbour staked back — if A puts 5 on B's node and B puts 3 on A's, 3 tokens each way cancel. Red: what is left after cancelling, flow with a direction ([Token flow](token-flow.md)). Left: the game of iteration 0, pooled over the 30 worlds; right: every game from iteration 500 on of the 30 worlds, pooled. Exactly 100,000 tokens are staked in each game.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take the rows with `phase` = 2: `tokens` (all staked), `totalFlow` (staked on others) and `netFlowShare` (the share of that flow left after cancelling).
> 2. Sum over the rows: at home = Σ `tokens` − Σ `totalFlow`; cancelled = Σ `totalFlow` × (1 − `netFlowShare`); one way = Σ `totalFlow` × `netFlowShare`.
> 3. Divide each by Σ `tokens`.
>
> **To make it again:** `python3 book_figures.py flow`.
<!-- /figure -->

In the first game, 13% of the tokens stay home, 23% are matched and 64% go one
way. The [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") of these worlds have ten neighbours each. They keep less at
home than the founders of a world of 10,000 tokens, which have four. There a
quarter stays home, a quarter is matched and a half goes one way.

In settled games, from [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 500 on, 30% stays home, **51% is matched**
and only 20% goes one way. At 10,000 tokens the shares are 30%, 49% and 21%.
Most of what agents send to their neighbours is an exchange of equal amounts.
That is why half of all agents end a game with exactly what they had
([Gains and losses](gains-and-losses.md)).

### Stakes that are returned

A matched stake needs a stake in each direction. How often does a stake get one
back?

<!-- figure flow/reciprocity -->
![Stakes that are returned](../figures/flow/reciprocity.svg)

**Stakes that are returned.** Of all pairs (A, B) in a game where agent A staked at least one token on neighbour B's node, the share in which B also staked at least one token on A's — every 25 iterations, in the 30 baseline worlds. The line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, read every 25 iterations by `book_figures.sample_pass` (frame `2·t` after the reproduction phase of iteration *t*, frame `2·t + 1` after its game; `gol_store.read_frame(run, index)`).
>
> 1. In the frame after the game of every 25th iteration, read `decisions.allocations`: for every agent A and every target B ≠ A with a stake > 0, note the pair (A, B).
> 2. Count the pairs whose reverse (B, A) is also there, divided by all pairs.
> 3. In each stretch, take the median, the 25th and 75th and the 5th and 95th percentiles of those means over the worlds that still have a value there (a world that died has none after its death).
>
> **To make it again:** `python3 book_figures.py flow`.
<!-- /figure -->

When A stakes on B, B stakes back on A in 90% of cases (the thirty worlds: 88%
to 93%; 92% at 10,000 tokens). Is that reciprocity, agents answering a stake
with a stake? Not necessarily.

Look at how widely agents stake (the next figure): they stake on 81% to 96% of
everyone they can. Suppose every agent picked its targets without regard to who
staked on it. B would still stake on A with about that probability. So a
reciprocity of 90% is what breadth alone would give. Real reciprocity would
mean that B stakes *more* on A *because* A staked on B. That is a question for
[Do agents cooperate?](do-agents-cooperate.md).

### How widely agents stake

An agent with *d* connections has *d* + 1 **[candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides.")**: its own node and its
neighbours' nodes. If it split its tokens evenly, it would put 1/(*d* + 1) of
them on each, home included.

<!-- figure flow/breadth -->
![How widely agents stake](../figures/flow/breadth.svg)

**How widely agents stake.** Agents at the start of a game, sorted by their number of connections. Blue: the share of their candidates (themselves and their neighbours) on which they staked at least one token. Yellow: the share of their tokens they staked on their own node. White dots: the home share an agent would have if it split its tokens evenly over all its candidates, 1/(d + 1), averaged over the class. Red dots: the same even split dealt out in whole tokens by the rules' own rounding ([Splitting tokens into whole numbers](largest-remainder.md)), which gives the tokens left over to the largest remainders, and, when all remainders tie, to the agent's own node first. From the games of every 100th iteration from 500 on, in the 30 worlds.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, read every 25 iterations by `book_figures.sample_pass` (frame `2·t` after the reproduction phase of iteration *t*, frame `2·t + 1` after its game; `gol_store.read_frame(run, index)`).
>
> 1. For every 100th iteration t from 500 on, take each agent's degree d in frame `2·t`.
> 2. In frame `2·t + 1`, read its entry of `decisions.allocations`: the number of `targets` (its candidates), how many of them got `alloc` > 0, and `alloc[0]` / `tokens` (its home share; its first target is itself).
> 3. Average by class of d; the white dots are the mean of 1/(d + 1) in each class.
> 4. Red dots: for every agent, `GraphOfLifeSimple._apportion` of equal scores over its candidates and its tokens; the share that lands on its own node, averaged by class.
>
> **To make it again:** `python3 book_figures.py flow`.
<!-- /figure -->

Agents stake on almost all their candidates: on 96% of them for agents with
one connection, falling to 81–82% for agents with ten or more. And their
**home share** follows the [even split](even-split.md "What the rules do when a brain scores no candidate above zero: the stake is spread evenly.") closely, a little above it:

| connections | 1 | 2 | 3–4 | 5–9 | 10–49 | 50+ |
|---|---|---|---|---|---|---|
| staked at home | 58% | 40% | 28% | 19% | 10% | 2.8% |
| even split, 1/(*d*+1) | 50% | 33% | 23% | 14% | 7% | 1.2% |
| even split, in whole tokens | 60% | 44% | 34% | 23% | 13% | 3.0% |

The extra at home is not the brains' doing. Tokens are whole, and an even split
rarely comes out even: an agent with 3 tokens and 2 candidates cannot stake 1.5
on each. The rules deal the tokens left over to the candidates with the largest
remainders ([Splitting tokens into whole numbers](largest-remainder.md)). In an
even split every remainder ties. So the leftovers go down the list of
candidates in order, and an agent's own node is always first.

Dealt out in whole tokens this way, an exact even split puts 60% at home for
an agent with one connection. For one with two, it puts 44%. The brains put a
little *less* at home than that.

Often there is not even a split to round. In the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world, 42% of agents
give every candidate a score of zero or below (58% at 10,000 tokens). Then the
rules split their tokens evenly by default
([What the brains are like](what-the-brains-are-like.md)). The brains that do
give positive scores give nearly the same one to every candidate: a brain's
outputs barely move with what it sees. So the evolved brains end up doing
something close to the simplest thing possible: spreading their tokens evenly
over everyone they can reach.

That is what blind brains do.
[Brains that see, and worlds that break](a-world-that-cannot-break.md#brains-that-see-and-worlds-that-break)
gives the brains one hidden layer instead of five. With it, what an agent sees
reaches what it stakes. A founder's brain passes on about 0.2 of a change in
what it sees, instead of 0.0006. These brains stake on one neighbour or two,
and their unused connections lapse. Their worlds need the rules of
[A world that cannot break](a-world-that-cannot-break.md#a-world-that-cannot-break)
and [Who dies, and why](a-world-that-cannot-break.md#who-dies-and-why) to hold
together.

### The game as a random walk

Suppose, then, that every agent split its tokens **exactly** evenly. Follow a
single token at node *u*. In a game it goes to each of the *d*ᵤ + 1 candidates
of *u* with the same chance. Repeated game after game, that is a **random
walk** of tokens. At every step a token stays where it is or moves to a
neighbour, each with probability 1/(*d*ᵤ + 1). It is a "lazy" walk, because
staying is one of the choices.

How good is that description? For every agent at the start of a game, work out
the balance the even split would give it. That is the stakes it would receive,
minus the tokens it holds. Then set it against the balance it really ended
with.

<!-- figure flow/random-walk -->
![The game as a random walk of tokens](../figures/flow/random-walk.svg)

**The game as a random walk of tokens.** For every agent at the start of a game: x is the balance it would end the stakes with — the tokens staked on its node minus the tokens it held — if every agent split its tokens evenly over itself and its neighbours; y is the balance it really ended them with. Agents are binned by x in steps of 2; the line is the mean of y in each bin and the band its middle half. The dashed line is y = x. Over all 1,798,806 agents (the games of every 250th iteration from 500 on, 30 worlds), the straight line fitted to the points has slope 0.65, and the even split accounts for 34% of the variance (R², [Fitting a straight line](least-squares.md)); over the 23 worlds that are not held together by a single hub ([The largest and the smallest](the-largest-and-the-smallest.md)), slope 0.94 and 45%.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, read (frame `2·t` after the reproduction phase of iteration *t*, frame `2·t + 1` after its game; `gol_store.read_frame(run, index)`).
>
> 1. For every 250th iteration t from 500 on, read frame `2·t` (the start of the game): each agent's tokens τ and neighbours.
> 2. Even split: every agent u gives τ(u)/(d(u) + 1) to itself and to each neighbour; x(u) is what u receives minus τ(u).
> 3. Real: in frame `2·t + 1`, add up every `alloc` whose target is u; y(u) is that minus τ(u).
> 4. Bin by x and average y; fit y = a·x + b by least squares over all agents.
>
> **To make it again:** `python3 book_figures.py flow`.
<!-- /figure -->

Over 1,798,806 agents of the thirty worlds, the real balances follow the even
split's with a slope of only 0.65. A slope of 1 would mean that the real
balances move exactly as much as the even split's. The even split accounts for
34% of the variance, that is, of how much the real balances differ from agent
to agent.

But that is the [stars](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") speaking. Leave out the seven worlds held together by one
hub ([The largest and the smallest](the-largest-and-the-smallest.md)). Then
the slope is **0.94**, and the even split accounts for **45%**: exactly as in
the worlds of 10,000 tokens (0.94 and 46%). The median world, fitted on its
own, has 0.90.

In those worlds the brains move tokens almost exactly as much as the random
walk would, and in the same direction. The rest of the variance is what they do
differently, agent by agent.

The stars break the fit through a single agent each. A star has thousands of
leaves: agents whose one connection is the centre. The even split, worked in
fractions of a token, expects each leaf to send the centre half its tokens.
Dealt out in whole tokens, home first, a leaf with one token sends nothing, and
one with three sends one. So a star's leaves, holding about three tokens, send a
quarter of a token less each than the fractions say. Over twenty thousand
leaves, that is some five thousand tokens at one agent. A point that far out
tilts a straight line fitted to two million. The rounding of
[Splitting tokens into whole numbers](largest-remainder.md) is a small
correction for an agent, and a large one for a star.

A random walk on a network has a well-known resting state, and it takes two
lines to find. Write π(*u*) for the share of all tokens at node *u*. The walk is
at rest when, across every connection, as many tokens go one way as the other:

$$
\pi(u) \cdot \frac{1}{d_u + 1} = \pi(v) \cdot \frac{1}{d_v + 1}
\quad\text{for every connection } \{u, v\} .
$$

The left side is what *u* sends to *v* in a game: its share, split over its
*d*ᵤ + 1 candidates. The right side is what *v* sends back. The two are equal
if π(*u*) is proportional to *d*ᵤ + 1. So:

> [!summary] The resting state of the even split
> On a fixed network, tokens moved by the even split settle to
> τᵤ = *T* · (*d*ᵤ + 1) / Σᵥ (*d*ᵥ + 1): **every agent holds tokens in
> proportion to its number of connections plus one.**

Here *T* is the number of tokens in the world, and τᵤ what agent *u* holds.
This is the law behind several findings of other pages:

- agents with more connections hold more tokens
  ([How unequal is a world, and who stays rich](where-do-the-tokens-go.md#rich-agents-are-connected-agents));
- tokens run downhill, from agents richer than their neighbourhood to poorer
  ones ([Gains and losses](gains-and-losses.md));
- a rich agent with few connections drains away within a few games.

A world is never quite at rest, because the network keeps changing under the
tokens. [How properties scale together](how-properties-scale-together.md)
measures how close it comes.
[The geometry of a world](what-shape-does-the-network-take.md#the-geometry-of-a-world) measures how fast a
random walk on such a network can settle at all.

### Who keeps their node

The winner of a node, whose brain the node takes, is its largest staker. The
exception is a coalition of smaller stakers that outweighs it
([How a coalition takes a node](revolution.md)). Under the even split, the
agent living on a node of [degree](degree.md "The number of an agent's connections: how many neighbours it has.") *d* is one of *d* + 1 stakers of roughly equal
size there.

<!-- figure flow/kept -->
![Who keeps their node, by connections](../figures/flow/kept.svg)

**Who keeps their node, by connections.** Every node on which anyone staked in a game, sorted by its number of connections at the start of the game; the share that was won by the agent living on it — whether as the largest staker or through a coalition. From the games of every 25th iteration from 500 on, in the 30 worlds.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, read every 25 iterations by `book_figures.sample_pass` (frame `2·t` after the reproduction phase of iteration *t*, frame `2·t + 1` after its game; `gol_store.read_frame(run, index)`).
>
> 1. For every 25th iteration t from 500 on, count each node's connections in frame `2·t`.
> 2. In frame `2·t + 1`, read `decisions.winners`: for each node, whether `winner` = `node`.
> 3. Pool over runs and frames, and divide kept by all, per class.
>
> **To make it again:** `python3 book_figures.py flow`.
<!-- /figure -->

The more connections a node has, the less often the agent living on it keeps
it:

| connections | 1 | 2 | 3–4 | 5–9 | 10–49 | 50+ |
|---|---|---|---|---|---|---|
| node kept by its agent | 74% | 48% | 27% | 17% | 9% | **2.0%** |

At 10,000 tokens the shares are the same, within a point or two. The pattern
is 1/(*d* + 1) again, raised a little by the rounding that hands the leftover
tokens to the agent's own node first. The agent on a node is one of *d* + 1
contenders, and wins about as often as any one of them would.

So a hub does not have an owner. The agent sitting on a node with a hundred
connections is replaced by a neighbour's brain in 98 games out of 100. **Hubs
are places, not individuals.** They hold many tokens because of where they are,
and the brains on them pass through. A brain cannot "become a hub"; it can only
be, for a game or two, the brain sitting on one.

## What crosses a connection

Whatever two agents stake on each other crosses the connection between them.
The section above followed the tokens through a whole world. This section looks
at the connections one at a time. How much crosses each one? How much of that
is cancelled by what comes back? And which connections carry nothing at all?

Why look at single connections? Because a connection lives only as long as
tokens keep crossing it. In B1, every connection that no token crossed in a
game is cut at the end of that game ([Every setting](settings.md),
`prune_after` and `inactive_window`). The viewer, the app that charts a run's
statistics ([What a run records](frames-and-stats.md)), counts these cuts as
`prunedEdges`. So the flow on the connections is not only how tokens move: it
is what keeps B1's network standing. A connection that carries nothing for one
game is gone, and with it, perhaps, the only path between two parts of a world.

The second [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") changes this rule
([Building B2: a world that cannot break](a-world-that-cannot-break.md#a-world-that-cannot-break)).
In [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") an unused connection lapses only if its two ends share at least two
neighbours ([`prune_needs`](settings.md#prune_needs)). So no lapse can cut a
world in two. This section measures B1.

The flow is also where cooperation would first show. Two agents that keep
staking on each other keep their connection, and with it each other's tokens
within reach. And tokens that go round a loop and arrive back where they
started, the viewer's **lightning**, are the most organised flow there can be.
They are the subject of [Lightning in every world](#lightning-in-every-world).

### How much crosses a connection

Every 25 iterations from 500 on, take every connection at the start of the
game. Add up what its two agents staked on each other in that game, both
directions together. Separately, cancel the smaller direction against the
larger, and see what is left.

<!-- figure connections/per-connection -->
![Tokens across one connection in one game](../figures/connections/per-connection.svg)

**Tokens across one connection in one game.** Every connection at the start of the games of every 25th iteration from 500 on, in the 30 worlds, pooled: how many tokens its two agents staked on each other in that game, both directions added (blue), and what is left when the smaller direction is cancelled against the larger (cyan; [Token flow](token-flow.md)). A connection in the first class carried nothing and is cut at the end of the game.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames every 25 iterations from 500 on, read by `book_chapters.connections.edge_flow`: the connections of frame `2·t` (the start of the game of t), `decisions.allocations` and the connections of frame `2·t + 1` (after it).
>
> 1. For every connection (a, b) of frame `2·t`: the tokens a staked on b and b on a, from `decisions.allocations` of frame `2·t + 1`.
> 2. Count the connections by the sum, and by the difference, in the classes shown.
>
> **To make it again:** `python3 book_figures.py connections`.
<!-- /figure -->

Most connections carry very little. Pooled over the thirty worlds, **52%**
carried exactly two tokens, 15% exactly one, 20% three or four, and 5% five to
nine. Fewer than one connection in a hundred carried ten tokens or more. But
the classes do not stop there. A few connections carry hundreds of tokens in a
game, and about one in a million carried five thousand or more. And **7%**
carried nothing at all.

The cyan bars show how much of this is an exchange. On **70%** of all
connections the two directions cancel exactly: as many tokens went each way.
That counts the 7% that carried nothing; of the connections that carried
anything, two thirds cancel. Almost all of the connections that carried two
tokens carried one each way. Where something is left, it is mostly a single
token: 28% of the connections moved one token net, and only 2% moved more.

[Do agents cooperate?](do-agents-cooperate.md) calls this a **handshake**, and
here it is seen from the connections. Two neighbours, each splitting a few
tokens evenly over a few candidates, put one token on each other.

### Which connections carry nothing

In B1 the rule is simple. A connection across which no token was staked in a
game is cut at the end of that game. One across which anything was staked is
kept. Counted over every connection of the sampled games, the rule holds
without exception. Of those connections, 1,650,222 carried nothing, and every
one was cut; 29,186,620 carried something, and every one was kept.
(Connections with an end that died in the game are left out.) Which
connections are the empty ones?

<!-- figure connections/cut -->
![The connections a game cuts](../figures/connections/cut.svg)

**The connections a game cuts.** Left: at the start of the game of every 25th iteration from 500 on, the share of the connections across which no token was staked in that game — the ones the game cuts; the line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th). Right: the same share, pooled over the games and worlds, for connections sorted by the number of connections of their less connected end.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames every 25 iterations from 500 on, read by `book_chapters.connections.edge_flow`: the connections of frame `2·t` (the start of the game of t), `decisions.allocations` and the connections of frame `2·t + 1` (after it).
>
> 1. As above; a connection carried nothing if neither end staked a token on the other.
> 2. Sort the connections by the smaller of their two ends' degrees in frame `2·t`.
>
> **To make it again:** `python3 book_figures.py connections`.
<!-- /figure -->

In a settled game, **6%** of the connections carry nothing (the worlds: 3% to
12%). The share wanders with the world, higher in some stretches than in
others. And it is not the same for every connection. Sort the connections by
their less connected end. If that end has a single connection, the link is a
leaf's only one, and it is empty in 1.3% of games. A connection between two
agents with ten connections or more is empty in 13% to 14%.

Why? A leaf stakes on its only neighbour or keeps everything at home, and its
neighbour almost always stakes back on it. An agent with many connections and
few tokens cannot send a token across each. If its neighbour does not send one
either, the connection is gone. Every connection costs a token a game, from one
side or the other. [How a world breaks](how-a-world-breaks.md) calls this the
price of a connection; here it is paid connection by connection.

The viewer counts the connections cut after every game as `prunedEdges`: 6.7%
of all connections in the median world (5.3% at 10,000 tokens). That is a
little more than the empty share at the start of a game, because connections
also die with their agents.

### The flow, in round figures

After every game, the viewer sums up the flow on the connections in a few
numbers:

<!-- figure connections/round-figures -->
![Flow on the connections, and the loops it makes](../figures/connections/round-figures.svg)

**Flow on the connections, and the loops it makes.** After every game of the 30 worlds. Left: the viewer's `meanEdgeFlow` — the tokens staked across a connection, over the connections something crossed — and `maxEdgeFlow`, the most across one, on a logarithmic axis. Right, every 25 iterations: the longest closed loop of flow the greedy peel of [Lightning](lightning.md) found in the flow as staked (`lightningLongest`) and in what is left after cancelling (`netLightningLongest`). The line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take the rows with `phase` = 2: `meanEdgeFlow`, `maxEdgeFlow`, `lightningLongest` and `netLightningLongest`.
> 2. Cut the 1,500 iterations into 600 stretches of 2.5 and take each world's mean in each stretch.
> 3. In each stretch, take the median, the 25th and 75th and the 5th and 95th percentiles of those means over the worlds that still have a value there (a world that died has none after its death).
>
> **To make it again:** `python3 book_figures.py connections`.
<!-- /figure -->

- **The typical connection.** Over the connections something crossed, the
  mean flow is **3.0** tokens a game (3.7 at 10,000 tokens).
- **The busiest connection.** The most that crossed one connection in a game
  is about **1,400** tokens in the median world: 1.4% of all the tokens of the
  world. The busiest connection of a world of 10,000 tokens carried 640, 6.4%
  of its tokens. A larger world has busier single connections, but they carry
  a smaller part of the whole.
- **Loops of flow.** A quick, greedy search
  ([What a lightning is](#what-a-lightning-is)) finds closed loops of flow.
  The longest it finds is **41** agents long in the median world (20 at
  10,000 tokens). Once each pair's two directions cancel, the longest loop
  found is 5 or 6 agents. At 10,000 tokens it averages 2.2 over the games. A loop left after
  cancelling needs at least three agents, and in many games none was found at
  all.

What these loops are, and how much of the flow they carry, is the subject of
the next section.

## Lightning in every world

Could the flow carry tokens around loops, from A to B to C and back to A, like
money circulating in an economy? This section asks it first in the thirty B1
worlds of 100,000 tokens. Then it asks it in every world the notebook has made:
both rule sets, every size, every setting an experiment changed.

### What a lightning is

In every game each agent splits its tokens over its own node and its
neighbours' nodes. A token staked on a neighbour's node **flows** along that
connection, from the agent to the neighbour. Draw an arrow for every such
token, and a game becomes a network of arrows: the flow.

A **lightning** is what the viewer calls a closed loop in that flow: A stakes on
B, B on C, and C back on A. A loop of *L* agents carries *L* tokens, one along
each of its arrows, and scores *L*². So one long circuit counts for more than
many short ones. Ecologists measure the same thing in food webs, as a *cycling
index*. The note [Lightning: tokens that go round](lightning.md) works through
the definition with an example. Why it matters:

- **Flow that goes round is flow that can keep going.** Tokens that run one way
  end up somewhere and stop; tokens that go round can circulate for ever. In
  physics, a current that closes on itself (a vortex, a loop of current) is a
  different thing from one that runs downhill. In an economy, a cycle of trade
  is a different thing from a transfer.
  [Physical inspiration](physical-inspiration.md) hoped for both kinds of flow
  to appear in these worlds.
- **Two agents that stake on each other already form a loop of length two.** A
  token from A to B and a token from B back to A *cancel*: on balance, nothing
  has moved. So every number below comes twice: for the flow as it was staked
  (*gross*), and for what is left after cancelling (*net*).

Finding the largest set of loops in a flow is as hard as some of the hardest
problems in computing. So the [engine](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") (`gol_lightning.py`) does not try. It
peels loops off greedily. It walks along the arrows, preferring agents it has
not visited yet, until an agent repeats. It takes that loop, removes one token
from each of its arrows, and starts again. Every count is therefore a **lower
bound**: there are at least this many tokens on loops, perhaps more. A run
records eight numbers, every 25th iteration:

| statistic | what it is |
|---|---|
| `cyclingShare` | the share of the staked flow the greedy search put on loops |
| `lightningLongest` | the longest loop it found, in agents |
| `lightningScore` | the sum over its loops of the length squared (long loops count for more) |
| `netFlowShare` | the flow left after cancelling, as a share of the flow staked |
| `netCyclingShare` | the share of that net flow on loops |
| `netLightningLongest`, `netLightningScore` | the longest net loop, and the score of the net loops |
| `flowImbalance` | half the sum over agents of the gap between flow in and flow out (counted positive either way), over the total: how much of the flow *must* run one way, whatever the loops are |

### Do tokens go round?

Start with the thirty B1 worlds of 100,000 tokens. How much of their flow
goes round? Only bounds can be given ([Lightning](lightning.md)). The greedy
search finds some loops: a floor. Conservation sets a ceiling. A token that comes back to where it started
changes no balance. So an agent that ends a game with more or less than it
sent cannot have had all its flow in loops. At least the difference runs one
way.

<!-- figure flow/circulation -->
![Do tokens go round?](../figures/flow/circulation.svg)

**Do tokens go round?** Left, on the flow as staked: the share of all tokens staked on neighbours that a greedy search could place in closed loops (blue; [Lightning](lightning.md)), and the share that conservation forbids from ever going round, because some agents receive more than they send (red) — so at most 1 − red can circulate. Right, after cancelling every stake against the one coming back: the share of the flow left (cyan), and of that, the share found in loops (green). Every 25 iterations, 30 worlds; the line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take `cyclingShare`, `flowImbalance`, `netFlowShare` and `netCyclingShare` of the rows with `phase` = 2 (measured every 25 iterations).
> 2. In each stretch, take the median, the 25th and 75th and the 5th and 95th percentiles of those means over the worlds that still have a value there (a world that died has none after its death).
>
> **To make it again:** `python3 book_figures.py flow`.
<!-- /figure -->

On the flow as staked, [the budget of a game](#the-budget-of-a-game) already
pins the answer down. Two
neighbours staking on each other form a loop of length 2. A token goes from A
to B, and one comes back from B to A. So every matched stake goes round.
Matched stakes are 51% of all tokens, which is 72% of the tokens staked on
neighbours (51 of the 71). That is a floor: at least 72% of the flow goes
round.

The ceiling comes from conservation, which forbids 23% of the flow from ever
going round (red), so at most 77% can. Between 72% and 77% of the flow as
staked goes round, then. Nearly all of it goes round in loops of two:
neighbours trading equal stakes back and forth. The greedy search (blue) hunts
for long loops, and places only 13% of the flow in the loops it finds. Beside
the 72% the matched stakes guarantee, that is a weak floor.

Now cancel every stake against the one coming back (right panel). No loop of
two is left, and 28% of the flow remains. Of that, the search finds only
**0.3%** in loops. The ceiling is not much higher. Cancelling changes no
agent's balance, so the imbalance stays the same, and at least
23.4/27.8 ≈ 84% of what is left must flow one way. Between 0.3% and 16% of the
net flow can go round, and the search finds almost nothing, as at 10,000
tokens.

A theorem says what that means. Any flow on a network can be split into two
parts ([Flows that go round](circulation.md)). One part runs **downhill**,
from high to low along a "potential": a height that every agent has. The other
part **circulates**. This is the Helmholtz–Hodge decomposition (for networks,
Jiang, Lim, Yao and Ye 2011). A downhill flow can never go round: round a loop,
you would come back to the height you started from.

The net flow of a game is almost entirely of the downhill kind. Tokens move
from hills to valleys, as heat does: from agents richer than their
neighbourhood to agents poorer than it. The few loops that survive cancelling
are tokens really going round, but they carry almost nothing.

### Every world

Is it the same in every world the notebook has made? The table of every world
(`research/atlas/across.py`;
[Every world side by side](every-world-side-by-side.md)) holds these numbers
for each of them.

<!-- figure lightning/every_world -->
![Loops of flow in every world](../figures/lightning/every_world.svg)

**Loops of flow in every world.** Each dot is one world, over iterations 900–999 (400–499 for a world that did not live to 1,000), a column per kind of world; the bar is the median. Top left: how much of what agents stake on their neighbours a greedy search can put on closed loops, before a token one way and a token back cancel. Top right: what is left once they cancel. Bottom: the longest loop found, and how much of the net flow goes round. B1 10k: the thirty B1 worlds of 10,000 tokens (Experiment 2); B1 fixed: the same with brains that never change (Experiment 7); B1 100k: B1 at 100,000 tokens (Experiment 10); B1 sizes: B1 at 800 to 409,600 tokens (Experiment 8) and at 5,000 with and without a hidden layer (Experiments 12 and 13); B2 10k: the ten B2 worlds of Experiment 16; B2 var: B2 with one rule or rate changed (Experiments 9, 15 and 16); B2 5k: B2 at 5,000 tokens, with all its rules or with one taken away (Experiment 14) — except lapse: the condition in which any unused link lapses; B1 pil. and B2 pil.: the pilot worlds that play the engine's game.

> [!example]- How to make this figure
> **Runs.** every simulation made so far, as `book/results/worlds.json` holds it (`python3 research/atlas/across.py lab`, `pilots`, `write`): a lab run's statistics are the means of its record over a stretch of iterations; a pilot's, of 20 games recorded from its checkpoint and summarised the same way.
>
> **Data.** From each world's `cyclingShare`, `netFlowShare`, `lightningLongest` and `netCyclingShare` (gol_lightning.py, on the heavy rows of a run's record).
>
> 1. Take each world's mean over iterations 900–999, or 400–499 if it has no later one.
> 2. Put it in the first group of the list in book_chapters/everyworld.py it belongs to.
> 3. Draw one dot per world; the bar is each group's median.
>
> **To make it again:** `python3 book_figures.py lightning`.
<!-- /figure -->

Three things stand out.

**The two rule sets are opposites.** In the B1 worlds, the greedy search puts a
sixth of the staked flow on loops (median 0.17 at 10,000 tokens). And three
quarters of the flow cancels: the net flow is a quarter of the staked flow. In
the worlds of B2, at every setting an experiment changed, the loops hold about
a twentieth (0.045). Almost nothing cancels there: the net flow is 95% of the
flow. B1's flow is mostly agents staking on each other; B2's is mostly agents
staking on others who do not stake back. Every pair of stakes that cancels is
a loop of two, so most of B1's flow goes round, if only between neighbours.

**One rule of B2 decides it.** A single kind of B2 world flows like B1: the
condition of
[Experiment 14](a-world-that-cannot-break.md#a-world-that-cannot-break) in
which *any unused link lapses* (`prune_needs = 0`), green in the figure. In B1
too, a connection that carries no tokens is cut after the game. B2 keeps a link
alive unless cutting it is safe for the network (`prune_needs = 2`), so its
links no longer have to be used.

Where a link must carry tokens to survive, agents stake on each other, and the
flow goes round. Where it need not, the flow runs one way. Nothing in the
brains changed: one of the rules that make B2's network unbreakable also made
its economy run one way.

The [pilot](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.") worlds say the same, independently
([The switches the world was built with](the-switches-the-world-was-built-with.md)).
Of all the B2 pilots, four let unused links lapse, and they are the only ones
whose flow mostly cancels. They keep 0.11 to 0.23 of their flow after
cancelling; every other B2 pilot keeps 0.87 or more. Three of the four put
0.21–0.28 of their flow on loops, and the fourth 0.11.

**Little goes round once the cancelling is done.** In every kind of world, the
net flow on loops is a fraction of a percent (bottom right). Within one game,
nearly all of what goes round is two agents staking on each other. The longer
loops are there, but they carry little.

The B1 worlds whose brains never change (yellow) spread over the whole range:
from almost nothing on loops to more than 70% of their flow. Each of those
worlds is frozen in whatever play its founders had
([Do the brains matter?](do-the-brains-matter.md)), and the founders differ.

The pilot worlds that play the engine's game are in the figures too. Each was
played on from its saved checkpoints and recorded as a lab run is. B2's pilots flow
as B2 does (0.046 on loops, 0.95 left after cancelling), and B1's as B1 does
(0.20 and 0.26).

### The longest loop grows with the world

<!-- figure lightning/size -->
![The longest loop grows with the world](../figures/lightning/size.svg)

**The longest loop grows with the world.** The longest loop the greedy search finds in a B1 world, against the world's size, both axes logarithmic: the median grows as the tokens to the power 0.31, so a world twice the size has loops 1.24 times as long. The blue line is the same after a token one way and a token back have cancelled.

> [!example]- How to make this figure
> **Runs.** The B1 worlds of Experiment 8 (800 to 409,600 tokens, three seeds each, over iterations 400–499), of Experiments 12 and 2 (5,000 and 10,000 tokens) and of Experiment 10 (100,000 tokens), over iterations 900–999.
>
> **Data.** From each world's mean `lightningLongest` and `netLightningLongest` in book/results/worlds.json.
>
> 1. Take each world's mean of the longest loop over its stretch.
> 2. Take the median at each size; fit a straight line to the logarithms.
>
> **To make it again:** `python3 book_figures.py lightning`.
<!-- /figure -->

The longest loop grows with the size of the world, as the tokens to the power
0.31. That number is not new. The distances across these worlds grow as the
number of agents to the power 0.25–0.30
([Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md)). And the
number of agents is proportional to the tokens. The worlds have a [dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.") of
about 3.3
([How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)).
So a world's width grows as its size to the power 1/3.3 = 0.30.

**So the longest loop is as long as the world is wide.** The greedy search
finds loops that go across the world, and they lengthen exactly as the world
widens. After cancelling (blue), the longest loops stay short and grow more
slowly. They are five or six agents long at 100,000 tokens, and six or seven in
the largest worlds.

### Through a world's life

<!-- figure lightning/life -->
![Through a world's life](../figures/lightning/life.svg)

**Through a world's life.** The two statistics through the life of thirty B1 worlds (Experiment 2) and ten B2 worlds (Experiment 16): the median at each recorded iteration, with the middle half of the worlds as a band.

> [!example]- How to make this figure
> **Runs.** The thirty B1 worlds of Experiment 2 and the ten B2 worlds of Experiment 16, 10,000 tokens each.
>
> **Data.** From `stats.jsonl`, the heavy rows (every 25th iteration).
>
> 1. Read each world's statistic at every heavy row.
> 2. Draw the median of the worlds and their middle half at each iteration.
>
> **To make it again:** `python3 book_figures.py lightning`.
<!-- /figure -->

In B1 the share of the flow on loops is about the same from the first hundred
iterations to the last. In B2 it falls slowly, from 0.065 to 0.04, while the
flow left after cancelling rises towards all of it. As a B2 world ages, its
agents stake less on each other. Neither rule set builds more circulation as it
evolves.

### A greedy count and an exact one

The greedy count looks at one game at a time.
[Gliders, and who helps whom](gliders-and-who-helps-whom.md) took the net flow
of fifty games together. It split that flow exactly into a part that runs
downhill and a part that circulates. The tool is the Helmholtz–Hodge
decomposition (the note [Flows that go round](circulation.md)).

<!-- figure lightning/exact -->
![A greedy count and an exact one](../figures/lightning/exact.svg)

**A greedy count and an exact one.** Each dot is a world: across, the exact share of the flow over 50 games that circulates (the Helmholtz–Hodge decomposition of [Gliders, and who helps whom](gliders-and-who-helps-whom.md)); up, the greedy loop count of a single game, averaged over the same thousand iterations.

> [!example]- How to make this figure
> **Runs.** The ten B2 worlds of Experiment 16 and the first ten B1 worlds of Experiment 2, iterations 1,000 to 2,000.
>
> **Data.** From `GraphOfLifeRuns/.pilots/circulation_20.json` (research/oee/checks/circulation.py) and each world's `cyclingShare`.
>
> 1. Take each world's circulating share, the median over its twenty windows of 50 games.
> 2. Take its mean greedy loop share over the same iterations.
> 3. Draw one against the other.
>
> **To make it again:** `python3 book_figures.py lightning`.
<!-- /figure -->

Across the twenty worlds, the two counts agree on which worlds circulate more.
Their [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") is r = 0.88, carried by the difference between the rule
sets. Within a rule set they agree loosely in B1 (r = 0.46 over ten worlds),
and not at all in B2 (0.01).

And the exact count finds much more. Measure a flow by its energy: the sum of
the squares of the flows on the connections, as in the note. Then 80% of B1's
net flow over fifty games circulates (the median of ten worlds), and 59% of
B2's. One game's greedy loops hold only 17% and 4.5% of the tokens.
Circulation in these worlds is spread over time. A token goes from A to B in
one game and from B to C in the next; no single game closes the loop.

That is a warning about both numbers. Random flow on the same network would
circulate too. There are two ways to make it. One puts random amounts of the
same size on every connection. The other keeps each connection's own amount,
and throws its direction at random. By these two null models of
[Gliders, and who helps whom](gliders-and-who-helps-whom.md), random flow
circulates 59–60% in B1, and 50–53% in B2. The real excess over chance is
about 0.2 in B1, and less than 0.1 in B2. B1's agents
keep tokens going round more than chance would; B2's barely do.

## What we learned

- **A game is close to a random walk of tokens.** The evolved brains of B1
  spread their tokens nearly evenly over everyone they can reach, as blind
  brains do. In tokens, a world's economy is an exchange of equal stakes, plus
  a diffusion downhill.
- **Wealth follows connections, and hubs are places.** At rest, an agent holds
  tokens in proportion to its connections plus one: being rich is a matter of
  being in a well-connected place. The brains on hubs change almost every game,
  so the network's most important positions are held by no one.
- **A connection lives on a token a game.** Two connections in three carry one
  or two tokens, most of them one each way, and cancel exactly. In B1 the rule
  that cuts is exact and local: every empty connection is cut, and every
  connection that carried a token is kept. Well-connected agents with few
  tokens cannot pay for all their connections; leaves almost always can. The
  flow that keeps the network standing is a web of handshakes.
- **Big flows are rare.** A few connections carry hundreds or thousands of
  tokens in a game. Only an agent holding that many can send them, and the
  rich are the well connected
  ([How properties scale together](how-properties-scale-together.md)).
- **Whether tokens go round is decided by one rule, not by the brains.** If a
  link must carry tokens to survive, agents stake on each other and the flow
  goes round (B1, and B2 with unused links lapsing). If links survive unused,
  the flow runs one way (B2).
- **Within a game, almost all that goes round is two agents staking on each
  other.** What is not exchanged runs downhill. The longest loops are as long
  as the world is wide: they grow as its size to the power 0.31, like its
  distances. But they carry almost nothing. Over fifty games much more
  circulates, but in B2 hardly more than random flow would. No world builds
  more circulation as it evolves.
- **For [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau.") this is a warning.** A behaviour this close to
  "split evenly" leaves little for brains to differ in. And a world whose
  wealth and positions are set by the network, rather than by the brains,
  gives selection little to grip.
  [Questioning the mechanics](questioning-the-mechanics.md) asks which rules
  make it so, and
  [What the measurements say about open-ended evolution](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution)
  what could be changed.
- **The connections carry no economy of their own.** There is nothing like a
  trade route along which one part of a world could feed another. A rule that
  made flows along loops pay, or let a connection store what crosses it, could
  give them one. One rule of this kind was tried:
  [Why evolution stops](how-to-tell-a-world-that-keeps-evolving.md#why-evolution-stops) pilots a yield on stakes that
  are returned across a connection. The yield went to whoever had the most
  partners, and it built stars rather than partnerships.

## Where to go next

- [Where tokens go](where-tokens-go.md): every measure of the flow in the
  notebook, side by side.
- [Gains and losses](gains-and-losses.md): who gains and who loses in a game,
  and how a rich agent with few connections drains back.
- [Flow modules: the groups tokens stay in](flow-modules.md): who trades with
  whom, and for how long.
- [Gliders, and who helps whom](gliders-and-who-helps-whom.md): the flow of
  fifty games, split exactly into what runs downhill and what goes round.

## How this was made

The first two sections, and [Do tokens go round?](#do-tokens-go-round), read
the thirty runs of Experiment 10, listed below. They are worlds of the
baseline B1 at 100,000 tokens, [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 to 30, run for 1,500 iterations each.
The command `python3 gol_lab.py run E10` makes them. From the runs, two
commands draw the figures:

- `python3 book_figures.py flow` those of
  [Where the tokens flow](#where-the-tokens-flow) and of
  [Do tokens go round?](#do-tokens-go-round);
- `python3 book_figures.py connections` those of
  [What crosses a connection](#what-crosses-a-connection).

The rest of [Lightning in every world](#lightning-in-every-world) reads every
world the notebook has made, from the table `book/results/worlds.json`. The
commands `python3 research/atlas/across.py lab`, `pilots` and `write` make
that table ([Every world side by side](every-world-side-by-side.md)). The
exact split of fifty games comes from `research/oee/checks/circulation.py`.
The command `python3 book_figures.py lightning` draws these figures. Each
figure says, folded beneath it, which runs it reads and how.

The measures are explained in the notes
[Lightning: tokens that go round](lightning.md),
[Token flow](token-flow.md), [Flows that go round](circulation.md) and
[Logarithmic axes and power laws](logarithmic-axes.md).

### The runs of Experiment 10

<!-- runs E10 -->
> [!info]- The runs behind this page
> **30 runs.** Every condition below is run once for every seed (1 to 30), for 1,500 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it.
>
> - **baseline** — changes nothing: B1 as it is; 100,000 tokens. Runs `B1-100000-s001` … `B1-100000-s030`.
>
> To make these runs again: `python3 gol_lab.py run E10`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | baseline | what it is |
> |---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 100,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | 50, 45, 40, 35, 30 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 30 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | on | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | on | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in both baselines; on in a few pilots, in Experiment 24 and in half the worlds of the screen. |
> | [`random_decisions`](settings.md#random_decisions) | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | off | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.0 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 0 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | off | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 0 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 0 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](settings.md#keep_home) | 0.0 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 0 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 1 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`self_message`](settings.md#self_message) | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 30 | one run per seed |
> | iterations | 1,500 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

## How our view changed

- **9 October 2026.** [How widely agents stake](#how-widely-agents-stake) had
  said that the brains put "a little extra at home", above the even split.
  [Who keeps their node](#who-keeps-their-node) had said that this "home
  bias" raised how often an agent keeps its node.
  [What the brains are like](what-the-brains-are-like.md) and the note
  [Splitting tokens into whole numbers](largest-remainder.md) showed where the
  extra comes from. It is the rules' rounding of whole tokens, not the brains.
  An exact even split, dealt out in whole tokens, puts even more at home than
  the brains do.
- **11 October 2026.** [Where the tokens flow](#where-the-tokens-flow) was
  first written from the thirty worlds of 10,000 tokens. On this day it was
  rewritten from the thirty worlds of 100,000 tokens, with the small worlds'
  numbers beside the new ones where they differ.
  [What crosses a connection](#what-crosses-a-connection) was written from the
  large worlds on the same day. One finding changed with the size of a world.
  Over all 335,453 agents of the small worlds, the real balances followed the
  even split with a slope of 0.94. Over all the agents of the large worlds the
  slope is 0.65, because a few stars tilt the fit; without them it is 0.94
  again.
- **14 October 2026.** A pass through the whole notebook corrected three
  things:
  - [Where the tokens flow](#where-the-tokens-flow) and
    [What crosses a connection](#what-crosses-a-connection) had read the
    greedy search's count as the flow that goes round. That was 13% of the
    staked flow, most of it, they said, in trivial loops of two. But every
    matched stake is itself a loop of two, so at least 72% of the flow as
    staked goes round, and at most 77%. The greedy search is only a weak
    floor.
  - [What crosses a connection](#what-crosses-a-connection) had said that
    after cancelling, the longest loop at 10,000 tokens was 2 agents. A loop
    left after cancelling needs at least three agents. The 2.2 it reports is
    an average over games, in many of which no such loop was found.
  - The founders of a world of 10,000 tokens have four neighbours each, not
    five: the rules wire an odd number of neighbours as one less.
- **23 October 2026.** When the three pages became one, these statements were
  checked against the numbers the figures were drawn from, and corrected:
  - [The budget of a game](#the-budget-of-a-game) said that in settled games
    51% of the tokens are matched and 20% go one way "at both sizes". At
    10,000 tokens the shares are 49% and 21%, as this page said on
    8 October.
  - [Every world](#every-world) said that the three B2 pilots whose flow goes
    round are exactly those in which unused links lapse. In the table of every
    world, four B2 pilots let unused links lapse. Three put 0.21–0.28 of their
    flow on loops; the fourth puts 0.11 there, and keeps 0.23 after
    cancelling. What sets all four apart is how much of their flow cancels.
  - [The longest loop grows with the world](#the-longest-loop-grows-with-the-world)
    said that after cancelling, the longest loops are five or six agents in
    the largest worlds. That holds at 100,000 tokens. In the worlds of 204,800
    and 409,600 tokens the median is 6.75 and 6.5 agents.
  - The table of statistics defined `flowImbalance` as half the sum over agents
    of flow in minus flow out. That sum is always zero. The engine adds up the
    gaps without their signs.

<!-- turns -->
---

← [The even split](even-split.md) · [The game and the flow of tokens](../README.md#the-game-and-the-flow-of-tokens) · [Flow modules: the groups tokens stay in](flow-modules.md) →

> [!info]- Pages that link here (24)
> [Questioning the mechanics](questioning-the-mechanics.md) · [Splitting tokens into whole numbers](largest-remainder.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [How births build the network](how-agents-have-children.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [Gains and losses](gains-and-losses.md) · [The largest and the smallest](the-largest-and-the-smallest.md) · [Where tokens go](where-tokens-go.md) · [How the game is played](how-the-game-is-played.md) · [The even split](even-split.md) · [Flow modules: the groups tokens stay in](flow-modules.md) · [Token flow](token-flow.md) · [Token curvature](token-curvature.md) · [The shape of the network](what-shape-does-the-network-take.md) · [How properties scale together](how-properties-scale-together.md) · [Stars and hubs](stars-and-hubs.md) · [Physical inspiration](physical-inspiration.md) · [How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md) · [Who helps whom](who-helps-whom.md) · [Do agents cooperate?](do-agents-cooperate.md) · [Healthy and rich](healthy-and-rich.md) · [Every world side by side](every-world-side-by-side.md) · [One world, many colours](one-world-many-colours.md) · [What goes with what](what-goes-with-what.md)
<!-- /turns -->
