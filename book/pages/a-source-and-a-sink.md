# A source and a sink

> [!question] Questions of this page
> - What would a world be like with a flow running through it, as sunlight
>   runs through the Earth?
> - Where could [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") enter a world that grows and dies, and how could they
>   leave, with no number set by hand but the flow itself?
> - What should form in such a flow, and how would we see it?

## In short

Every world of this notebook is closed: its tokens are a fixed sum, and its
[game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") spreads them like heat towards an even rest. Life on Earth feeds on a
flow instead: energy that arrives concentrated from the Sun and leaves spread
out. This page plans a world with a flow. Tokens enter at one [node](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."), the
source, which stays with its place and passes to a [neighbour](the-world.md#the-network "An agent joined to another by a connection.") only when that
place dies; and every agent pays one token an [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."), which leaves the
world. The inflow then takes the place of the total tokens as a world's one
number: a world fed *J* tokens an iteration holds, on average, *J* agents. It
would be tried on three of round 3's candidate worlds, in the local world of
[Local rules only](local-rules-only.md), each fed as many tokens an iteration
as it held agents when closed. The draft thesis: the open world arranges
itself around its source, with tokens, births and the turnover of [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.")
falling away from it, and holds together, since a piece cut off from the
source runs down and dies.

## Why a flow

Energy is conserved, and disorder always grows. How, then, is anything as
ordered as life possible? Because the Earth is not closed
([Why life is possible](physical-inspiration.md#why-life-is-possible-a-flow-of-low-entropy)).
It takes in sunlight, a few energetic photons from a surface at about 5,800
kelvin, and sends the same energy back to space as infrared, many feeble
photons from a planet at about 255 kelvin. What leaves carries about twenty
times the entropy of what arrives, and that export pays for every ordered
thing on Earth, from a cloud to a cell (Boltzmann 1886; Schrödinger 1944).

A world of Graph of Life has the conservation and not the flow. A token is a
token, wherever it came from. In [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") the game's [even split](even-split.md "What the rules do when a brain scores no candidate above zero: the stake is spread evenly.") is a step of the
heat equation, which smooths every difference away
([Heat, not waves](physical-inspiration.md#heat-not-waves)): a closed world
runs down towards a miniature heat death.

Physics knows what can form where a flow runs through. Heat a thin layer of
oil from below, gently, and the heat simply leaks through it. Heat it more,
and the oil turns over in hexagonal cells, rising in their middles and falling
at their edges (Bénard 1900). The cells exist only while the heat flows; stop
it, and they are gone. Such patterns are called **dissipative structures**
(Nicolis and Prigogine 1977). That a flow of energy through a system tends to
organise it is an old idea of biology too (Morowitz 1968).

The same sum can be done for tokens, as an analogy only. A token that enters
at one node and leaves from any of *N* agents arrives at a known place and
leaves from one of *N*. In the language of information, it leaves carrying
ln *N* more uncertainty about where it is than it brought: for a world of
1,900 agents, about 7.5 nats, or 11 bits, a token. Tokens are not energy, and
this is not heat. But it says what an open world would export: knowledge of
where its tokens are.

## Where a source could sit

A world of Graph of Life has no fixed places. Nodes are born and die, and the
world has no coordinates. So where should tokens come in?

1. **Everywhere.** The [engine](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") can do this now:
   [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) adds
   tokens to the pool at every [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out."), dealt out like an [estate](estate.md "The tokens a dead agent leaves, and who gets them."). A flow
   without a place makes no gradient. It is kept below as a control.
2. **At the oldest living agent.** The source would jump across the world
   whenever the oldest died, and finding the oldest means looking at every
   agent at once. Not local.
3. **Where the world began.** The [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")' ring does not last: its nodes die
   like any others ([A beginning](physical-inspiration.md#a-beginning)), and
   without coordinates "near where it began" means nothing once they are gone.
4. **On a node that stays a node.** The source starts on the first founder's
   node, and stays with that node, whoever holds it: a node keeps its id when
   it is taken. Only when the node itself dies does the source pass, to the
   living neighbour that holds the most tokens. It is local, and has no number
   in it. **This is the plan's choice.**
5. **On a star that is no agent.** A node outside the game, which never dies,
   with agents linked to it. It is physics' cleanest picture, but a node that
   cannot be staked on, starve or be mended would need an exception in almost
   every rule.
6. **On several nodes.** How many would be a number to choose; it is left for
   later.

The tokens land on the source node at the start of every phase, before
anyone looks, so its holder sees them and stakes them. That holder will be by
far the richest agent of its world. In [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 3026, fed 938 tokens a phase where
the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") agent holds 9, it could take every neighbour's node in every game,
and its neighbours will fight for its own. A line that holds the source may
hold it long, as a hub's place was handed down in B1
([How a star is handed down](the-largest-and-the-smallest.md#how-a-star-is-handed-down)).

## What a sink could be

What comes in must leave somewhere, or the tokens pile up for ever. Again,
without a number set by hand:

1. **Every agent pays one token an iteration, and the token leaves the
   world.** Living costs a token, as a body radiates heat: everywhere, all the
   time. An agent that cannot pay starves. The payment falls after the game,
   so a newborn plays one game before it first pays. **This is the plan's
   choice.**
2. **Every link costs one token an iteration, and the token leaves.** Upkeep
   paid for links already exists ([`link_upkeep`](settings.md#link_upkeep));
   its token could leave the world instead of going to the link's other end
   or the estate ([`upkeep_to`](settings.md#upkeep_to)). Then the network
   itself, the world's space, would be what the flow pays for. But brains
   choose which links to pay for. Where a used link needs no payment
   (`paid_or_used`), payments may fade away and the tokens pile up; only where
   every link must be paid (`paid_only`) is the sink sure. And an agent with no
   link pays nothing, so an island of one never runs down. For a world that
   already pays upkeep, it is one setting changed.
3. **Friction.** A token is lost on every link a stake crosses: the flow
   dissipates where tokens move. It changes the game itself, since staking on
   a neighbour would cost.
4. **The estates of the dead leave**, instead of being shared out. Then the
   sink is death. In round 3's worlds only the culled leave an estate, so the
   sink would be the cull, which acts on a whole world at once.
5. **At the edge.** Tokens leave only from agents with one link, as a planet
   radiates from its surface. A sink only where the world thins out: it is
   contrived.

Why the first?

- **It is everywhere**, as the Earth radiates from all of itself.
- **It gives an open world one law.** In every phase, the tokens a world holds
  change by what comes in less what is paid. Over a long stretch the tokens
  held stop growing, so the agents, each paying a token an iteration, average
  as many as the tokens fed in an iteration. A world fed *J* tokens an
  iteration holds *J* agents on average. The inflow replaces the total tokens
  as the world's one number, and the world's size follows from it, with
  nothing tuned.
- **A piece cut off from the source runs down.** Each of its agents loses a
  token an iteration and none comes in, so the piece dies by itself. The
  islands that pieces living on would leave in a closed world
  ([What we expect](local-rules-only.md#what-we-expect)) have a local death
  here: the sink is a grave that needs no cull.
- **It needs no new output from the brain**, so it can be added to any world
  of round 3 as it was drawn.

Its price: an agent left with a single token after a game pays it and
starves, and a child born with one token must win another in its first game
or die. An open world will be one of fewer, richer agents than its closed
self, gathered where the tokens are.

## The world's one number

Today a world's one number is its total of tokens: 30,000 in round 3. In an
open world it is the inflow: *J* tokens an iteration, half of them at the
start of each phase. How should it be chosen, without tuning it?

The plan's answer: **feed each world as many tokens an iteration as it held
agents when it was closed.** Seed 3026 held a median of 1,876 agents in its
round-3 run, from iteration 500 on, so it is fed 938 tokens a phase. Then the
open world can hold about as many agents as the closed one. How many tokens it
holds at once is left to emerge.

<!-- table localrules/flow -->
| world | agents | links per agent | tokens per agent | tokens a game's cleanup shares out | inflow a phase, to feed as many agents | hours for 3,000 iterations |
|---|---|---|---|---|---|---|
| [s3026](run:B1-30000-d6955f-s3026) | 1,876 | 3.33 | 16.0 | 9.7% | 938 | 1.6 |
| [s3004](run:B1-30000-7d5208-s3004) | 1,610 | 5.79 | 15.7 | 0.09% | 805 | 0.6 |
| [s3080](run:B1-30000-625bc7-s3080) | 1,900 | 3.36 | 14.0 | 1.4% | 950 | 0.8 |
<!-- /table -->

The flow is not large for every world. In seed 3026 a game's cleanup already
shares out 9.7% of all the tokens, the estates of the agents it culls: about
2,900 tokens a game, more than the 1,876 an iteration the flow would bring. In
seed 3004, which barely culls, the cleanup shares out 0.09%, and the flow
would be about sixty times as large.

## The experiment

### Three worlds

The worlds are the three of
[How far does a difference travel?](how-far-does-a-difference-travel.md):
seeds 3026 and 3004, whose rules would keep a light cone, and seed 3080, whose
agents jump to strangers. A source's gradient may wash out through such long
links.

### Three conditions

1. **The local world, closed:** the control, played as condition 4 of
   [Local rules only](local-rules-only.md#four-conditions-one-change-each).
2. **The local world, open:** the source on a node, fed *J* tokens an
   iteration, and every agent paying a token an iteration.
3. **The same flow, spread everywhere:** the same inflow dealt out over every
   agent, as `tokens_created_per_phase` deals it now, with the same sink. It
   is not local, by design: it tells what the place adds to the flow.

Each starts as its closed control did: the same founders, seed and 30,000
tokens. Each is played for 3,000 iterations and recorded as round 3 recorded
its worlds, with every agent's decisions.

An open world holds as many tokens as it comes to hold. Should its mean
holding settle more than a fifth away from 30,000, a closed world with that
many tokens is played beside it, so that open and closed are also compared at
the same mean tokens.

### What is measured

- **Distance from the source**, for every agent in every recorded [frame](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections."), by
  walking the network outward from the source node.
- **Profiles by distance,** over iterations 500 to 3,000: how many agents
  stand at each distance; their tokens, their ages; births and deaths per
  agent; the share of nodes taken in a game, which is how fast brains turn
  over; how many kinds of brain there are; and the health of the network
  there: links per agent, and the share of agents in the
  [2-core](bridges.md#core-trees-and-leaves).
- **The outward flow:** the net tokens that cross from each distance to the
  next in a game, read from the stakes recorded with each frame. It must
  balance what the agents beyond pay.
- **What lasts in the flow.** Channels: the links that carry most of the
  outward flow, and how many of them still do fifty games later, against the
  same flow shuffled among the links at each distance. The source's dynasty:
  how long one line of descent holds the source node. Beats: peaks in the
  [power spectrum](autocorrelation.md#the-power-spectrum-of-a-time-series) of
  the tokens at each distance, against a random walk's.
- **The books and the size:** the tokens that come in and go out in every
  phase; the agents against *J*; the tokens held; the moves of the source.
- **Health and evolution**, as round 3's worlds were read, against the closed
  control: the six criteria of [Which worlds are healthy?](which-worlds-are-healthy.md)
  and the generations, heredity and late novelty of
  [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md).

## What we expect

**A gradient.** Tokens should leave the source as warmth leaves a stove in a
cold room: richest near it, poorer with every step, since every agent on the
way pays away a token an iteration. Births need tokens, so most children
should be born near the source.

**A world shaped by its sun.** Agents too far away to be fed starve, so the
world should gather round the source, as large as *J* tokens can feed.

**A wind of brains.** A node taken copies its winner's brain, and the rich
near the source take nodes in every game. Brains should turn over fastest
near the source, and lines of descent should spread outward from it.

**No islands.** A piece cut off from the source loses a token per agent an
iteration and gains none, so it dies. The open world should stay in one piece,
where the closed local world is expected to break into islands that never die.

**Against.** The source's holder may take every node near it in every game and
copy one brain over the whole neighbourhood: a monoculture at the centre,
however fast it turns over. A world whose agents jump to strangers may wash
its gradient out through long links. And whether a flow makes evolution last
longer, which is why the notebook asks, the thesis leaves open: the
measurements above will say.

> [!quote] Draft thesis (to be fixed before any run)
> **The claim.** A world fed at one node and drained everywhere arranges itself around its source, and holds together. Tokens, births and the turnover of brains are highest near the source and fall away from it, and the world stays in one piece without any cull, since a piece cut off from the source runs down and dies.
>
> **Why it would be so.** The game spreads tokens as heat spreads, from where they are many to where they are few, and every agent on the way pays away a token an iteration, so fewer reach the edge. Births need tokens, so they follow them. A node taken copies its winner's brain, and the rich near the source take nodes in every game. A piece cut off from the source loses a token per agent an iteration with none coming in, so it starves, while in the closed local world an island that holds a token can never die. Against this, the source's holder may copy one brain over its whole neighbourhood, and a world whose agents jump to strangers (seed 3080) may wash its gradient out through long links.
>
> **It holds if:** over iterations 500 to 3,000, in at least two of the three worlds, all of: (1) the median agent within two steps of the source holds at least three times the tokens of the median agent eight steps away or more; (2) births per agent within two steps are at least twice those eight steps away or more; (3) the share of nodes taken in a game is higher within two steps than at any greater distance; (4) the largest piece holds at least 90% of the agents in every game.
>
> **It fails if:** in two of the three worlds the ratio of (1) is below 1.5, so that no gradient forms; or the open world breaks into pieces, its largest holding less than 90% of the agents in a tenth of the games or more; or it dies, its agents falling to 20, or its source lost with no neighbour to pass to.

## What has to be built first

1. **The local world** of [Local rules only](local-rules-only.md), which is
   also the closed control. In a world with the cull and a [share-out](gains-and-losses.md#gains-losses-and-the-share-out "How a cleanup deals the tokens of the dead out among the living: in the baseline, each token to a survivor drawn at random.") over
   everyone, the estates of the dead would land anywhere at once, and wash any
   gradient out.
2. **The source:** a new setting, off by default. Its inflow is
   `tokens_created_per_phase`, which exists; with the source on, those tokens
   land on the source node at the start of every phase instead of joining the
   pool. The source starts on the first founder's node, stays with its node,
   and passes to the richest living neighbour when its node dies. Where it is
   goes into every frame and the statistics. A source whose node dies with no
   living neighbour is lost, and that is recorded too.
3. **The sink:** a new setting, off by default. After every game, every agent
   pays one token, which leaves the world; one left with none has starved.
   [`buy_life`](settings.md#buy_life) already takes a token an iteration from
   the old; the sink takes it from everyone, and sends it away instead of into
   the estate.
4. **The books.** Every phase's statistics gain the tokens that came in and
   went out. The lab's checks that a world's tokens add up must check an open
   world's books instead: what it held, plus what came in, less what went out.
5. **The measures** above: distances from the source in every frame, the
   outward flow from the recorded stakes, channels, the dynasty and beats.
6. **In the viewer:** the source marked, and the agents coloured by their
   distance from it.

The names need care: `token_colours`, reserved in `research/strains.md`, is
the other road to a gradient, tokens with a quality, and is not this.

## What it would cost

In round 3 the three worlds took 3.1 hours of one worker for 3,000 iterations:
1.6 for seed 3026, 0.6 for seed 3004 and 0.8 for seed 3080. An open world fed
as many tokens as its closed self held agents should hold about as many and
cost about as much: about three worker-hours a condition, an hour or so on the
lab's three workers, and twice that with the flow spread everywhere. The
closed controls are those of Local rules only; closed worlds at the open
worlds' mean tokens, if they are needed, cost about as much again. Each
condition fills about 2.3 GB of disk.

## Choices for the author

1. **The sink:** every agent pays, as proposed; every link pays; friction;
   estates that leave; or tokens that leave at the edge.
2. **The source:** one node that stays a node, as proposed; its whole
   neighbourhood; several nodes; or a star that is no agent.
3. **The inflow:** as many tokens an iteration as the closed world held
   agents, as proposed; or a round number, set as the total is set today.
4. **The world:** the local world, as proposed, or each world's own rules,
   whose cull and share-out would carry the tokens of the dead anywhere at
   once.
5. **The floor of 10,000 tokens.** An open world holds what it holds. Should
   one settle below 10,000 tokens, it could be played again with more inflow.
6. **Tokens with a quality** instead (`research/Research.md`, §4.3): a fresh
   token becomes spent when used, and only slowly fresh again, so the count is
   conserved but the quality is not.

## Where to go next

- [Local rules only](local-rules-only.md): the local world this flow runs
  through, and the islands it would leave without one.
- [How far does a difference travel?](how-far-does-a-difference-travel.md):
  the light cone of the same worlds.
- [Physical inspiration](physical-inspiration.md#why-life-is-possible-a-flow-of-low-entropy):
  why a flow of low entropy makes life possible.
- [Where tokens flow](where-the-tokens-flow.md): the game as a random walk of
  tokens, the closed world's heat.

## How this was made

No run of this experiment exists. The sizes, holdings, shares and times of
the three worlds come from their own statistics in round 3 (`stats.jsonl`,
from iteration 500 on for sizes and holdings, the whole run for the time),
and the inflow is half the median number of agents, rounded. To make the
table again: `python3 book_figures.py localrules`
(`book_chapters/localrules.py`). The working notes are in `research/Plans.md`,
under *A source and a sink*.

## References

- Bénard, H. (1900). Les tourbillons cellulaires dans une nappe liquide. *Revue générale des sciences pures et appliquées* 11, 1261–1271 and 1309–1328.
- Boltzmann, L. (1886). *Der zweite Hauptsatz der mechanischen Wärmetheorie*. Lecture to the Imperial Academy of Sciences, Vienna.
- Morowitz, H. J. (1968). *Energy Flow in Biology*. New York: Academic Press.
- Nicolis, G. and Prigogine, I. (1977). *Self-Organization in Nonequilibrium Systems*. New York: Wiley.
- Schrödinger, E. (1944). *What Is Life?* Cambridge: Cambridge University Press.

<!-- turns -->
---

← [How far does a difference travel?](how-far-does-a-difference-travel.md) · [Space and dimension](../README.md#space-and-dimension) · [Particles: patterns that outlast what carries them](particles.md) →

**Also about:** [Changing the rules](../README.md#changing-the-rules).

> [!info]- Pages that link here (3)
> [Physical inspiration](physical-inspiration.md) · [Local rules only](local-rules-only.md) · [How far does a difference travel?](how-far-does-a-difference-travel.md)
<!-- /turns -->
