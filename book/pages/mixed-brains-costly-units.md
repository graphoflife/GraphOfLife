# Brains mixed at birth, units that cost, an eye for raids on kin

> [!question] Questions of this page
> - If a newborn's [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") is mixed from its parent's and a [neighbour](the-world.md#the-network "An agent joined to another by a connection.")'s, the way
>   sexual reproduction mixes genes, does evolution find more?
> - If every unit of a brain has to hold one of its [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")'s [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), how large do
>   brains become, and does evolution go on?
> - Does an input that says *this neighbour took a node from my own kind*
>   change who attacks whom?
> - And what becomes of a world where every link holds a token and nobody dies
>   of age?

## In short

Three changes to [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") were tried in pilot worlds, each aimed at a different
part of how evolution works. Brains mixed at birth made the worlds small and
combative, and in one of two worlds gave the largest steps of evolution yet.
Units that each hold a token shrank brains to one or two units. Yet their
evolution was stronger than with free units, from 500 to 1,000 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). An
input for raids on one's own kind changed nothing. And a world where every
link holds a token and nobody dies of age froze: little is born, little dies,
and little evolves.

## Three changes to how evolution works

[Why evolution stops](how-to-tell-a-world-that-keeps-evolving.md#why-evolution-stops)
asked why evolution slows down in these worlds. The lifetime contests of
[The switches the world was built with](the-switches-the-world-was-built-with.md)
showed that in B2's family of worlds it does not quite stop. The brains of
1,000 iterations still beat those of 500 two times out of three. This page
tries three more changes, each aimed at a different part of how evolution
works:

- **how variation is made**: brains mixed at birth;
- **what a brain costs**: every active unit holds a token;
- **what an agent can see**: an input for raids on its own kind.

## What was built

Every world is B2 at 10,000 tokens, with Experiment 16's [settings](settings.md "One number or switch of a run's rules, fixed before it starts."), played for
2,000 iterations. Each has two [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world."), as pilot worlds
([Pilot worlds](the-lab.md#pilot-worlds)):

| switch | what it does | compared with |
|---|---|---|
| `mix` | A newborn's brain is mixed from its parent's and a mate's — a neighbour of the parent, drawn at random. Each hidden unit (its incoming weights, its bias and its outgoing weights together) and each output bias comes from one or the other, with an even chance. The mixture is a new [genotype](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from."), then mutated as any copy is. | B2 |
| `took_kin` | One more input per neighbour: did it, in the last [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), take a node held by an agent of my genotype? | B2 |
| `unit_token` | Brains that can grow (up to 32 hidden units, switched on and off by mutation), where every active unit holds one of its agent's tokens. A unit switched on is paid for, one switched off pays its token back, and what an agent cannot pay for stays off. | the same growable brains with units free |
| `link_token`, no old age | Every link holds a token, paid by whoever makes it; no cap on links (valence 0) and no age limit. | — |

Each is judged by **lifetime contests**, as
[The switches the world was built with](the-switches-the-world-was-built-with.md)
judged the switches. The brains of a later moment play those of an earlier
one, half and half on the same world. They play for 100 iterations
([A contest a lifetime long](eras-and-contests.md#a-contest-a-lifetime-long)).
Each pair plays once in each moment's world.
If the later brains hold more than half of the nodes at the end, evolution
went somewhere between the two moments.

<!-- figure ideas/contests -->
![Does evolution go on?](../figures/ideas/contests.svg)

**Does evolution go on?** Lifetime contests (T = 100 iterations): the brains of a later moment against those of an earlier one, half and half on the same world, in each moment's world; the share of nodes the later brains hold, averaged over the two. Above a half the later brains are better. Each dot is a world.

> [!example]- How to make this figure
> **Runs.** Pilot worlds of 10,000 tokens with B2's settings, two seeds each, 2,000 iterations (`GraphOfLifeRuns/.pilots/Q/`, `research/oee/pilots/jobs_Q.txt`): `mix`, `took_kin`, growable brains (`brain_kind = grow`, up to 32 units) with units free and with `unit_token`, and `link_token` with `valence = 0` and `max_age = 0`; the control is batch P's B2 worlds of the same settings and seeds.
>
> **Data.** From `GraphOfLifeRuns/.pilots/batch_Q_T100.json`, `batch_P_T100.json` (`research/oee/checks/rejudge.py --T 100`).
>
> 1. Play each pair of moments for 100 iterations in each moment's world; average.
>
> **To make it again:** `python3 book_figures.py ideas`.
<!-- /figure -->
## Brains mixed at birth

Mixing changes the worlds first. They are a quarter of B2's size: 750–980
agents, against 3,200–3,650. Their conquest is twice B2's: 10–13 nodes taken
per 100 agents a game, against about 6. And more of what is taken goes round.
Over fifty games, the circulating share of the takeovers' flow is 0.63–0.64,
against 0.49–0.51 for its sign-flip null
([Gliders, and who helps whom](gliders-and-who-helps-whom.md)). B2 has
0.39–0.42, against 0.34–0.35.

And evolution goes on, in one world strongly. In the first world, the brains
of 1,000 hold nine tenths of the nodes against those of 500 (0.91). Those of
2,000 hold three quarters against those of 1,000 (0.73): the largest steps of
any pilot world so far. The second world moves like B2: 0.70, then 0.55,
where B2 gives 0.67–0.69, then 0.65–0.76.

Why would mixing help? A mixed child inherits whole units from two unrelated
brains. So a unit that does something useful can spread through the
population without carrying the rest of the brain it arose in. And two useful
units that arose in different [lineages](genotype.md#families "A line of descent: the agents whose genotypes go back to one ancestor.") can meet in one child. That is the
classical argument for sex. Here it is a hypothesis: two seeds, small and
combative worlds, and no test yet of which of these does the work.

## Units that hold a token

Where units are free, brains grow. Over 2,000 iterations the mean number of
active units rose from 4 to 5.6 in one world, and to 9.8 in the other. The
second is far above the **neutral shadow**: a unit that mutates like the
others but does nothing. The shadow shows where the count would drift if
selection did not care, and it is 3.9 there. So in a world where it costs
nothing, a larger brain is selected for.

<!-- figure ideas/units -->
![How many units brains use](../figures/ideas/units.svg)

**How many units brains use.** Growable brains of up to 32 hidden units, where switching a unit on is free (blue) and where every active unit holds one of its agent's tokens (green). The dashed lines are each world's neutral shadow: the same count for a unit that mutates but does nothing, which is where the number would sit if selection did not care.

> [!example]- How to make this figure
> **Runs.** Pilot worlds of 10,000 tokens with B2's settings, two seeds each, 2,000 iterations (`GraphOfLifeRuns/.pilots/Q/`, `research/oee/pilots/jobs_Q.txt`): `mix`, `took_kin`, growable brains (`brain_kind = grow`, up to 32 units) with units free and with `unit_token`, and `link_token` with `valence = 0` and `max_age = 0`; the control is batch P's B2 worlds of the same settings and seeds.
>
> **Data.** From each world's `series.jsonl` (`activeMean`, `shadowMean`).
>
> 1. Read the mean number of active units, and of the shadow's, every 50 iterations.
>
> **To make it again:** `python3 book_figures.py ideas`.
<!-- /figure -->
Where every unit holds a token, brains shrink: from 4 units to about 1.3
within 200 iterations. And they stay there, far below their shadow (4.1 and
5.3). A unit is worth less than the token it holds. The worlds are smaller,
with 1,150–1,440 agents. And yet from 500 to 1,000 their evolution is
stronger than that of the same brains with free units. The brains of 1,000
hold 0.72 and 0.94 of the nodes against those of 500, where free units give
0.59 and 0.65. From 1,000 to 2,000 the two are alike, and like B2: 0.65 and
0.77, against 0.60 and 0.78.

A brain of one or two units is a very small target for mutation: almost
every change touches something that matters. That may be why selection keeps
finding improvements. It is a guess to test, not a finding.

## An eye for raids on kin

The input changed nothing that was measured. The worlds are B2's size, with
3,540–3,560 agents. Evolution goes on at B2's pace: 0.65–0.74 from 500 to
1,000, and 0.60–0.61 from 1,000 to 2,000, against 0.67–0.69 and 0.65–0.76.
And help between neighbours stays at B2's level. Like the inputs of
[Gliders, and who helps whom](gliders-and-who-helps-whom.md), it tells an
agent something its rules give it no reason to act on.

## Links that hold a token, and nobody dies of age

With a token in every link and no age limit, the worlds froze. After the
first few hundred iterations, births and deaths nearly stopped. There were
some 250–300 births in the next 1,700 iterations, in worlds of 1,600–1,900
agents. At 2,000 iterations the mean age is about 1,600. Conquest almost
stopped, at 0.1–0.6 nodes taken per 100 agents a game. And the contests are
near a tie or a little above: 0.49–0.63, the mean of each pair of moments.
The only way a brain can spread is to take a node.

The network is fine: one piece, nothing cut off, 6.1–6.9 links per agent.
But little happens in it. With the age limit, the same rule keeps a [healthy
world](what-makes-a-world-interesting.md "A world that meets the six criteria of the screen of random worlds; an interesting one misses at most one.") whose evolution goes on more strongly than B2's. There the brains of
1,000 hold 0.83 and 0.93 of the nodes against those of 500
([Rules without numbers](rules-without-numbers.md)).

## What we learned

- **Three changes gave the largest steps of evolution yet.**
  - Mixing brains at birth, in one of two worlds, at every step.
  - Making units cost a token, in both worlds, from 500 to 1,000.
  - Making links hold a token, in both worlds, at both steps
    ([Rules without numbers](rules-without-numbers.md)).

  All three change how variation is made, or what structure costs. They
  change what a mutation can reach, and what a unit or a link has to earn. None changes what agents see. None is yet
  shown to keep evolution going longer than B2 does.
- **A cost turns size into a choice.** Free units grow beyond their neutral
  drift; units that cost a token shrink to almost nothing. Neither rule needs
  a number for the size of a brain.
- **Information alone changes nothing**, here as in
  [Gliders, and who helps whom](gliders-and-who-helps-whom.md). An input is
  only as useful as the rules make acting on it.
- **A world without death freezes**, even with a healthy network: turnover is
  what selection works on.
- Two seeds each are pilots. Mixing and costly units are [candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides.") for an
  experiment of ten worlds, run longer, beside B2.

## Where to go next

- [Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md):
  mixing beside 23 other rules.
- [Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md): the
  experiment that took mixing further.
- [How a brain changes](mutation.md): what a mutation does to a brain.
- [What the brains are like](what-the-brains-are-like.md): what evolved
  brains do.

## How this was made

Every world on this page is a pilot, made with `research/oee/pilots` from the
job list `jobs_Q.txt` and kept in `GraphOfLifeRuns/.pilots`. No experiment of
the lab stands behind it, so the page has no box of runs. Every figure says,
folded beneath it, which worlds it reads and how.

A first count of the contests of the mixed worlds was wrong, before this page
was written. Mixed worlds came out the other way: later brains *lost*. The
cause was in how a contest tells the two sides apart. It labels each brain by
a number: the earlier moment's below a million, the later one's above. A
brain copied without change keeps its parent's number, so children belong to
their parent's side. But a mixed brain is a new genotype and draws a fresh
number, which in these worlds is always below a million. So every child born
during a contest was counted for the *earlier* side, whoever its parent was.
The contest, `tourney.contest`, now follows a new brain's ancestry back to
the brain it was copied from. All of the mixed worlds were judged again. No other contest was touched: without mutation, mixing is the only way
a new number appears during one.

<!-- turns -->
---

← [Who inherits?](who-inherits.md) · [Changing the rules](../README.md#changing-the-rules) · [Links to strangers](links-to-strangers.md) →

**Also about:** [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are).

> [!info]- Pages that link here (9)
> [Every setting](settings.md) · [Flow modules: the groups tokens stay in](flow-modules.md) · [How alike the agents are](how-alike-the-agents-are.md) · [Rules without numbers](rules-without-numbers.md) · [Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md) · [Eight switches at once, and how sixteen worlds can test them](eight-switches-at-once.md) · [Healthy and rich](healthy-and-rich.md) · [Brains mixed at birth, in long worlds](mixed-at-birth-ten-long-worlds.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
