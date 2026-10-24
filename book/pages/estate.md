# The estate of the dead

## In short

When an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") dies, the [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") it still holds are not lost: the
[cleanup](one-iteration.md#cleanup) pools them as the **estate of the dead** and hands them to the living. The setting
[`redistribution`](settings.md#redistribution) says who inherits. In both
[baselines](b1-and-b2.md), B1 and B2, every token goes to a survivor drawn at
random, each with the same chance. The notebook tried five other heirs, and found that who
inherits decides how many agents a world holds, and how poor they are.

## What happens at a death

Every death happens in the cleanup after a phase of an iteration
([Births and the four ways to die](births-and-deaths.md)). Agents with no
tokens starve, and agents no longer joined to the largest connected piece of
the network are cut off; under [`max_age`](settings.md#max_age), agents also
die when they reach that age. The tokens the dead still hold are pooled and
dealt out among the survivors, so the world's total never changes.

Who leaves an estate depends on the rules. In B1 the starved hold nothing, so
the estate is the tokens of the cut off, who make three deaths in four
([What the baseline world is](what-the-baseline-world-is.md)). In B2 no agent
can starve or be cut off, and every agent dies at age 100, so the estate is the
tokens of the old ([Who dies, and why](a-world-that-cannot-break.md#who-dies-and-why)).

**The baseline rule**, `uniform`, deals the pool in one random draw, each token
to a survivor drawn with equal chance. In the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") settled B1 world of
100,000 tokens that is about 1,300 tokens per game, 1.3% of all tokens
([Where do the tokens go?](where-do-the-tokens-go.md#the-tokens-of-the-dead)).
Rich and poor have the same chance at every token, so the estate levels the
world: in ten B1 worlds, the more of it a game scatters, the more the
[Gini coefficient](gini-coefficient.md) falls, a [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of −0.78 between
the changes ([What goes with what](what-goes-with-what.md#a-whole-world-over-time)).
In B2 worlds of 10,000 tokens the estate is small, about seventy tokens an
iteration among 3,600 agents ([Who inherits?](who-inherits.md)).

## Who may inherit

| `redistribution` | the estate goes to |
|---|---|
| `uniform` | every survivor, the same chance at every token (B1 and B2) |
| `by_tokens` | every survivor, with a chance in proportion to what it holds |
| `poorest` | the poorest survivors, raised to one common level as far as the pool reaches |
| `newborns` | the children born this iteration that are still alive, an even chance each |
| `parents` | the parents of those children, once per child |
| `children` | each dead agent's own living children, an even chance each |

`newborns` and `parents` fall back to `uniform` in an iteration without a
birth, and `children` for an estate that finds no living child.

## What each heir did

Each rule was played in [pilot worlds](pilot-worlds.md) of 10,000 tokens, two
[seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") each:

- **By wealth** (B2, 1,000 iterations): the world held 2,450 and 2,800 agents
  against 3,400, and otherwise stayed as it was; its [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") went on improving
  ([The switches the world was built with](the-switches-the-world-was-built-with.md)).
- **To the poorest** (2,000 iterations, as below;
  [Who inherits?](who-inherits.md)): B2 barely changed, at 3,600 to 3,700
  agents. B1's worlds grew larger and younger: 2,100 to 2,300 agents against
  the controls' 800 to 1,400, and a mean age of 74 to 76 iterations against
  120 to 210.
- **To the newborns or to the parents:** B2's worlds grew by half, to 5,100 to
  5,300 agents, nine in ten holding three tokens or fewer, and evolution
  stopped: later brains took only 37% to 57% of a world contested with earlier
  ones. B1 became a world of infants, with 55 to 85 agents in a hundred born
  every iteration.
- **To the dead agent's own children:** B2 changed little, and evolution went
  on at least as fast: the brains of iteration 1,000 took 80% and 74% of a
  world contested with those of 500, against 69% and 67% in the controls. B1's
  worlds grew larger and younger, less than with the poorest.

## An example: paying for births

Experiment 31 paid the estate to the newborns, or to the parents, in B2 worlds
whose brains are mixed at birth, with parents as mates
([Room for the steadiest world](room-for-the-steadiest-world.md)). Its worlds
met every mark of the [health bar](health-bar.md) and were true spaces, 2.0 to
2.65 times as wide as their [random twins](random-twin.md). But about 5,000
agents shared the 10,000 tokens, 1.9 each, and the
[diversity score](diversity-score.md "How much the agents of a world differ, in one number from 0 to 1: in tokens, links, ages and genotypes.") fell from 0.54–0.75 to 0.32–0.41. Experiment 32 found the
arithmetic ([Births as rich as the neighbours](room-for-the-steadiest-world.md#births-as-rich-as-the-neighbours)):
such a world holds about a hundred agents for every birth an iteration, and the
estates paid for 51 to 53 births an iteration, where the same world without
them had 12 to 23.

## Where it is used

- [Who inherits?](who-inherits.md): four of the heirs, in B1 and in B2.
- [What holds a world's size](what-holds-a-worlds-size.md): estates paid to
  births among the rules that set how many agents a world holds.
- [Where tokens go](where-tokens-go.md#the-estate-of-the-dead): the estate beside
  the other flows of tokens.

## Where to go next

- [Births and the four ways to die](births-and-deaths.md): who dies, and where.
- [Who inherits?](who-inherits.md): the experiment behind most of this page.
- [Births as rich as the neighbours](room-for-the-steadiest-world.md#births-as-rich-as-the-neighbours): why
  a world paid for its births fills with poor agents.

<!-- turns -->
---

← [Births and the four ways to die](births-and-deaths.md) · [Agents' lives: births, deaths, ages](../README.md#agents-lives-births-deaths-ages) · [Age and lifetime](age-and-lifetime.md) →

**Also about:** [Changing the rules](../README.md#changing-the-rules).

> [!info]- Pages that link here (7)
> [Glossary](glossary.md) · [Questioning the mechanics](questioning-the-mechanics.md) · [Every setting](settings.md) · [Births and the four ways to die](births-and-deaths.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [Gains and losses: who gains, who loses, and the share-out](gains-and-losses.md) · [Where tokens go](where-tokens-go.md)
<!-- /turns -->
