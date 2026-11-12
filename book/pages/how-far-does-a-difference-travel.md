# How far does a difference travel?

> [!question] Questions of this page
> - If two worlds are the same but for one [token](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), where, and how fast, do
>   they come to differ?
> - How far can news travel through a world in one phase, by its rules, and
>   which rules let it go further?
> - Does a small difference grow until the two worlds have nothing in common,
>   or does it die away?

## In short

Take two copies of a world, and in one of them move a single token from an
[agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") to its [neighbour](the-world.md#the-network "An agent joined to another by a connection."). Where the twins come to differ, phase after phase,
shows how far news travels in a world, and whether a small change grows or
fades. By the rules, a difference can travel at most four steps a phase, three
in worlds without the look that only talks: a world's speed of light. But only
once the cull, the [share-out](measuring-lives.md#the-estate-of-the-dead "The tokens a dead agent leaves, and who gets them.") and the one random stream are local, and only 4
of the 24 worlds on round 3's shortlist when this plan was written have no
other rule that breaks the bound. The plan is thirty twin trials on each of three candidate worlds, under
three conditions, with heat spreading on the same network as the test of the
measurement. The draft thesis: under local rules the bound is never broken,
and the difference does not die away but spreads, more slowly than the bound
allows, until it holds a tenth of the world or more.

> [!quote] Draft thesis (to be fixed before any run)
> **The claim.** In a world whose rules are local, a difference of one token never outruns its cone. In round 3's candidate worlds it does not die away either: it spreads as a front well inside the cone, at a speed of its own, until it holds a large part of the world. In the sense of damage spreading, the worlds are chaotic.
>
> **Why it would be so.** The cone follows from the [engine](runs-and-records.md#the-lab-the-engine-and-pilots "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world."): a [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") reads two steps away, a [message](rules-in-detail.md#messages "Numbers an agent writes to itself and to each neighbour every phase, which their brains read.") carries what it read one step further, and an agent acts on its own node and its neighbours', four steps a phase in all, three without the look that only talks. Inside it, one token is a large change: the [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") agent of [seeds](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 3026 and 3004 held 9 and 11 tokens in round 3. An agent's decisions are made in whole steps — everything on one node or a spread stake, a child or none, the largest stake takes the node — so a small change tips some of them, and a node taken copies a different brain, which decides differently from then on. Round 3's tanh brains pass on about half of a change in what they see. Against this, the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") spreads tokens as heat spreads, and heat forgets; a token too many and a token too few can meet and cancel.
>
> **It holds if:** in the local world, in both seed 3026 and seed 3004, all three hold. (1) Not one agent in any trial differs outside the cone: four steps a phase in seed 3026, three in seed 3004. (2) In at least 20 of the 30 trials the damage is still there after 200 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") and holds at least a tenth of the agents. (3) The median over the trials of the front's speed, from 10 to 50 iterations after the move, lies between half a step and four steps an iteration, well inside the eight and the six the cones allow. The controls must behave: heat's front moves exactly one step a step, and twins with nothing moved stay identical.
>
> **It fails if:** any agent of the local world differs outside its cone: then an agent reads or acts further than this page found, or a global rule is left. Or the twins heal, becoming identical again, in more than half the trials of a world: the worlds are ordered. Or after 200 iterations the damage stays within ten steps of where the token was moved in more than half the trials: the difference stays put. A control that fails voids the trials of its world.

The full draft, with the rules or brains it would build, the experiment, what
has to be built first and what it would cost, is in the notebook's history:
`git show 9ec755d:book/pages/how-far-does-a-difference-travel.md`.

## Where to go next

- [Local rules only](local-rules-only.md): the local world these twins are
  played in, and what round 3 already says about it.
- [A source and a sink](a-source-and-a-sink.md): a world with a flow running
  through it, planned on the same local world.
- [Physical inspiration](physical-inspiration.md#a-speed-limit): the speed
  limit, and the rules that break it.
- [Is a run reproducible?](is-a-run-reproducible.md): twins with nothing moved,
  as the notebook first tested them.

## How our view changed

- **11 November 2026.** Shrunk to its questions, one paragraph and its draft
  thesis, as the owner approved in a review of the whole notebook: a plan not
  yet chosen need not carry its whole draft. The full draft is in the
  notebook's history, as the note above says, and nothing of it is lost.

## References

- Derrida, B. and Weisbuch, G. (1986). Evolution of overlaps between configurations in random Boolean networks. *Journal de Physique* 47, 1297–1303.
- Kauffman, S. A. (1969). Metabolic stability and epigenesis in randomly constructed genetic nets. *Journal of Theoretical Biology* 22, 437–467.
- Lieb, E. H. and Robinson, D. W. (1972). The finite group velocity of quantum spin systems. *Communications in Mathematical Physics* 28, 251–257.
- Lorenz, E. N. (1963). Deterministic nonperiodic flow. *Journal of the Atmospheric Sciences* 20, 130–141.
- Roberts, D. A., Stanford, D. and Susskind, L. (2015). Localized shocks. *Journal of High Energy Physics* 2015(3), 51.

<!-- turns -->
---

← [Local rules only](local-rules-only.md) · [Space and dimension](../README.md#space-and-dimension) · [Particles: patterns that outlast what carries them](particles.md) →

**Also about:** [Changing the rules](../README.md#changing-the-rules).

> [!info]- Pages that link here (3)
> [Physical inspiration](physical-inspiration.md) · [Local rules only](local-rules-only.md) · [A source and a sink](a-source-and-a-sink.md)
<!-- /turns -->
