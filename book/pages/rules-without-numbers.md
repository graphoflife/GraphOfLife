# Rules without numbers

> [!question] Questions of this page
> - Which numbers in [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s rules were chosen by hand, and what does each do?
> - What is left of a world when all of them are taken away at once?
> - Which numbers can be replaced by something the world decides for itself,
>   and which are a matter of definition?

## In short

B2, the world that cannot break, has seven numbers in its rules that were
chosen by hand, and the notebook asks for rules without them. Taken away all
at once, the numbers show what each was for: without alliances the worlds
died out. With alliances, each lived as one peaceful alliance, but on a
network that fails the [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4."). The clearest replacement so far is a [token](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.")
in every link: then no cap on links is needed. Nothing yet replaces the age
limit.

## Why ask

Picture a rule that says: *an owner's stake on its own [node](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") counts one and a
half times.* It does its job. Defending becomes cheaper than attacking, so a
[brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") is no longer lost in almost every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."). But why one and a half, and not
1.4, or 2? Nothing in the world decides that. Someone chose it.

The author of the simulation set a rule for everything that follows. No
factor chosen by hand may sit in the rules, except the total of tokens and
the parameters of [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.")
([How this notebook is written](how-this-notebook-is-written.md#the-rules-of-the-research)).
Every other number should come out of the world. The one and a half by which
an owner's stake counts was the author's own example of what is not wanted.

B2 added seven such numbers to the original rules, those of [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it."). Each was
chosen to stop one way a world was seen to fail. Each was tested when B2 was
built
([Building B2: a world that cannot break](a-world-that-cannot-break.md)):

| number | in B2 | what it stops | shown in |
|---|---|---|---|
| [`incumbency`](settings.md#incumbency) | 1.5 | an owner's stake counts one and a half times: defending is cheaper than attacking, and a brain is not lost in almost every game | [Holding one's place](a-world-that-cannot-break.md#holding-ones-place) |
| [`birth_links`](settings.md#birth_links) | 2 | a newborn is joined to its parent and two of the parent's [neighbours](the-world.md#the-network "An agent joined to another by a connection."), drawn at random: no world grows as a tree | [A tree of blobs](a-world-that-cannot-break.md#a-tree-of-blobs) |
| [`prune_needs`](settings.md#prune_needs) | 2 | an unused link lapses only if its ends share two neighbours: pruning never cuts | [What the game adds](a-world-that-cannot-break.md#what-the-game-adds) |
| [`valence`](settings.md#valence) | 8 | a brain may choose at most 8 links: no [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") | [What the game adds](a-world-that-cannot-break.md#what-the-game-adds) |
| [`keep_home`](settings.md#keep_home) | 0.3 | 30% of an agent's tokens stay at home: no agent stakes itself to death, and fewer agents are left with a single token than when only the last token is kept | [A share kept at home](a-world-that-cannot-break.md#a-share-kept-at-home) |
| [`max_age`](settings.md#max_age) | 100 | every agent dies at 100: no agent holds its place forever, and there is room for the young | [Old age](a-world-that-cannot-break.md#old-age) |
| [`min_child`](settings.md#min_child) | 10 | a child starts with at least 10 tokens: children are not spent as missiles | [Children as missiles](a-world-that-cannot-break.md#children-as-missiles) |

Two more rules of B2 have no number in them. A death is mended by joining the
dead agent's neighbours in a ring
([Mending a death](a-world-that-cannot-break.md#mending-a-death)), and a brain
changes only when it is copied. Both stay. The brain has sizes too: one
hidden layer of 16 units, and [messages](messages.md "Numbers an agent writes to itself and to each neighbour every phase, which their brains read.") of 8 numbers. Those are left alone
here, for *How big should a brain be?*, a page still to be written.

Experiments 14 and 15 took six of B2's numbers away **one at a time**, all
but the owner's one and a half. They showed what each prevents
([Each rule taken away breaks something](a-world-that-cannot-break.md#each-rule-taken-away-breaks-something),
[What each rule of death did](a-world-that-cannot-break.md#what-each-rule-of-death-did)).
This page does the opposite. It takes all seven away at once, and asks what a
world does with none of them. Then it asks which of them can be replaced by a
rule with no number in it.

## B3: B2 without its numbers

The [pilot worlds](the-lab.md#pilot-worlds) of this page are batch N. A pilot
is a world made by hand to see whether an idea is worth an experiment, here
with two [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") of each kind. These worlds play B1's game, with each of the
seven numbers at its value in the original rules:

- an owner's stake counts once;
- a newborn has only the links its brain chooses;
- any [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") that carries no tokens lapses;
- no agent has a cap on its links;
- no agent keeps anything at home by rule;
- no agent dies of age;
- a child may have a single token.

On top of that, they keep the two rules B2 has without a number. Their
brains, of one hidden layer of 16, change only when copied, and deaths are
mended in a ring. Each brain also remembers the last game: what each
neighbour staked on it, and what it staked on each neighbour. [Revolutions](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.") are
off. We call these worlds **B3**. There are three kinds, two seeds each, at
10,000 tokens for 1,000 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."):

- **B3a**, with alliances ([Alliances](another-game.md#alliances)). In every
  game, an agent says of each neighbour whether it wants it as an ally. Two
  that both say yes are allies, and their stakes on a node count together.
- **B3b**, with alliances, and a newborn also joined to every neighbour its
  parent was allied with in the last game (`birth_allies`). That builds space
  from the alliances, with no number: how many links a newborn gets is the
  brains' doing.
- **B3c**, without alliances.

## What the six worlds did

<!-- figure numberless/agents -->
![Worlds without numbers](../figures/numberless/agents.svg)

**Worlds without numbers.** One line per world, each a mean over 10 iterations; a world that died out ends at one agent.

> [!example]- How to make this figure
> **Runs.** Pilot worlds of `research/oee/pilots/run.py`, kept in `GraphOfLifeRuns/.pilots/N`: `B3a-s1`, `-s2`, `B3b-s1`, `-s2` and `B3c-s1`, `-s2`, at 10,000 tokens for 1,000 iterations. Settings: one hidden layer of 16, messages of 8 numbers, no pre-pass, no handover, revolutions off, and every number of B2 at its original value; prototype switches `mor=all repair=true memory=true`, with `allies=true` (B3a), `allies=true birth_allies=true` (B3b), or neither (B3c).
>
> **Data.** From each world's `series.jsonl`.
>
> 1. Take the agents after each iteration's game (`nodes`) and average over consecutive stretches of 10 iterations.
>
> **To make it again:** `python3 book_figures.py numberless`.
<!-- /figure -->

**Without alliances, the worlds died.** Both B3c worlds died out: they fell
to 20 agents or fewer, the notebook's mark of a world that has ended
([When a world ends](extinction.md)). One did so at iteration 102, the other
at 432. Until then, nearly every node changed hands in every game: 88 to 91
per hundred agents. In the longer-lived world, from iteration 100 to 400, 96
agents in a hundred were born and 77 starved in every iteration. So the
average agent was one and a half iterations old.

This is the world that
[Children as missiles](a-world-that-cannot-break.md#children-as-missiles)
warns of. Children of a token or two stake everything and lose it. Nothing
holds the population up but births, until they fall short.

**With alliances, the worlds lived, but not as worlds should.** Every world
with alliances that lived became one alliance. In the last 300 iterations of
each, the largest alliance held 99.7 to 100% of the agents. And the fighting
stopped: 0.0 to 0.4 nodes changed hands per hundred agents a game. What was
left differed from world to world:

- **B3a, seed 1: a star, then a tree.** From iteration 130 to 330, one agent
  was joined to more than half of all the others. At iteration 250 it was
  joined to all but one of 4,383. Nothing stopped brains from joining their
  children to the best-connected neighbour. When the star went, the world
  grew back as a tree. Over its last 300 iterations it had 2.2 links per
  agent, and 42% of its agents had a single link. A third were in the [2-core](bridges.md#core-trees-and-leaves "What is left of a network when agents with one connection or none are peeled away again and again: the agents on loops or between them."),
  and one agent or one link could cut off 39% of the world. It was a tree of
  old agents, with a mean age of 206. There were 3.5 births per hundred
  agents an iteration, and as many agents were cut off.
- **B3a, seed 2: a world of mayflies,** the insects that live for a day. In
  every iteration, 26 agents in a hundred were born and 24 starved. The mean
  age was 21.
- **B3b: births among allies.** Seed 1 died out at iteration 74. Seed 2
  lived dense and short. It had 8 links per agent, one agent joined to a
  fifth of the world, and a [ball dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.") of 5.2. There were 53 births and
  49 starvations per hundred agents an iteration, and the mean age was 5.

<!-- figure numberless/ages -->
![How old the agents are](../figures/numberless/ages.svg)

**How old the agents are.** One line per world while it had more than 20 agents, each a mean over 10 iterations. In B2 every agent dies at 100, so the mean age settles near 50.

> [!example]- How to make this figure
> **Runs.** Pilot worlds of `research/oee/pilots/run.py`, kept in `GraphOfLifeRuns/.pilots/N`: `B3a-s1`, `-s2`, `B3b-s1`, `-s2` and `B3c-s1`, `-s2`, at 10,000 tokens for 1,000 iterations. Settings: one hidden layer of 16, messages of 8 numbers, no pre-pass, no handover, revolutions off, and every number of B2 at its original value; prototype switches `mor=all repair=true memory=true`, with `allies=true` (B3a), `allies=true birth_allies=true` (B3b), or neither (B3c).
>
> **Data.** From each world's `series.jsonl`.
>
> 1. Take the mean age of the agents after each game (`meanAge`) while the world has more than 20 agents, and average over stretches of 10 iterations.
>
> **To make it again:** `python3 book_figures.py numberless`.
<!-- /figure -->

None of the six passes the health bar
([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md#the-network-that-holds-in-numbers)).
So none has the network the notebook asks of a world in which evolution is to
work for long. The three that lived ended the fighting. With it went nearly
all selection, except that between who has children and who does not.

### Did the worlds that lived keep evolving?

A world can be played against its own past
([Eras, contests and invasions](eras-and-contests.md)). Take the brains of
one moment, an **era**, and those of a later one. Give every agent a brain of
one era or the other, at random, switch mutation off, and let the world play
on. The later era's share of the world at the end is its score: 0.5 is a
tie. Here every contest lasted 100 iterations, a [lifetime](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death.") in B2
([A contest a lifetime long](eras-and-contests.md#a-contest-a-lifetime-long)),
and was played once in each era's world. An era played against itself, in
other worlds, held 0.44 to 0.58: that much is noise.

The three worlds that lived answer in three ways:

- **B3a, seed 1: a tie.** The brains of iteration 1,000 held 0.50 of
  iteration 500's world, and 0.46 of their own.
- **B3a, seed 2: the later brains won.** They took 0.84 of their own world,
  and the whole of 500's.
- **B3b, seed 2: the earlier brains won.** The brains of 500 took the whole
  of 1,000's world, and kept a little more than half of their own. The brains
  of 1,000 held 0.00 of their own world and 0.43 of 500's.

Each of these worlds, of 1,500 to 3,400 agents, is held together by one
alliance. A contest in it tends to go to one side entirely. Whether B3 keeps
evolving, two seeds cannot say.

## What each number was doing

Every failure above was already named when B2 was built, one number at a
time. Here they come all at once:

| without | what happened |
|---|---|
| a share kept at home | agents stake themselves to death: 24 to 77 in a hundred starve in every iteration where the world lives as mayflies |
| a smallest child | the starving are children of a token or two; mean ages of 1.5 to 21 iterations |
| a cap on links | a star: one agent joined to all but one of the others |
| links at birth, and with every unused link lapsing | a tree: 2.2 links per agent, and a cut of 39% by one loss |
| an age limit | in the tree, agents live for hundreds of iterations, a mean age of 206; 3.5 in a hundred are born in an iteration, and as many are cut off |
| the home advantage | the alliances do its work, as in [Alliances](another-game.md#alliances) |

Alliances made the difference between dying and living: both worlds without
them died, and three of the four with them lived. Whether they also make the
difference between living and evolving, the contests above cannot tell.

## Numbers and definitions

Not every number in a rule is a factor chosen by hand. Some say what a rule
*means*. In "a triangle has three corners", three is not a choice: it is what
a triangle is.

The 2 of `prune_needs` is a number of that kind. The rule says that an unused
link lapses only if its two ends stay joined twice over without it. With one
shared neighbour, pruning can leave two agents hanging on a single third.
With two it cannot, and no smaller number does that. So the 2 is the
definition of the property wanted. In the same way, a unit of tokens is one
token. When the share kept at home is "never fewer than one token", that one
is the unit, not a choice.

Other numbers are **magnitudes**: 1.5, 0.3, 8, 10 and 100. Nothing in the
rules picks them; a world would work, differently, at 1.4 or at 120. The 2
of `birth_links` is a magnitude too. In the model without brains, one link at
birth or three both make a world, with thinner or thicker blobs
([Too much, and too little](a-world-that-cannot-break.md#too-much-and-too-little)).

## Replacing the magnitudes

The author's rule is aimed at the magnitudes. Each needs a replacement that
the world decides for itself. And each failure above says what that
replacement must prevent. These replacements were proposed on 15 October
2026, with this page:

- **The owner's one and a half → none.** Alliances form without it, and what
  defends a node is then whoever stands with its owner. Two worlds of B2
  with alliances show it, with an owner's stake counted once. Their largest
  alliances held 37 and 71% of the world at iteration 1,000
  ([Without the home advantage](another-game.md#without-the-home-advantage)).
- **The share kept at home and the smallest child → birth by division.** An
  agent decides only *when* to have a child, and a child always takes half of
  its parent's tokens. No child is then poorer than the parent it leaves. So
  none can be spent as a missile, and no number is needed to say how small a
  child may be.
- **The cap on links → links that cost.** In the original algorithm a link
  lives only while tokens cross it. If an agent can pay to keep the links it
  wants, how many links it holds is set by what it can pay for.
- **The links at birth → all of them.** A newborn joined to its parent and to
  every neighbour of its parent closes every triangle it can, and needs no
  number. Combined with links that cost, it is the brains that decide which
  of those links last. No world has tried it yet.
- **The age limit** has no replacement yet. It is what makes room for the
  young in B2. The tree of old agents above shows what a world without it can
  become.

## How the replacements fared

**Birth by division stops the missiles, but not the starving.** It was tried
the same day, in quick worlds of one seed each, run for at most 150
iterations. A brain that would give a child at least half its tokens divides
into two equal halves; one that would give less has no child. Alone, with
nothing kept at home, a world of 3,000 tokens starved out in 60 iterations.
With half of every agent's tokens kept at home, a world lived. But most of its
agents held a single token, and half of one token is no child. With B2's
rules of space added as well, births stopped altogether, and with no age
limit nobody died.

**Links that cost, paid as gifts, do not hold a network together.** With
gifts, an agent may hand tokens to a neighbour, and a gift keeps the link it
crosses alive
([Three switches, and two more](the-switches-the-world-was-built-with.md#three-switches-and-two-more)).
But in B2 with every unused link lapsing, the brains did not pay for the
links that held their world together. The network broke as it does without
gifts
([What the switches did to the world](the-switches-the-world-was-built-with.md#what-the-switches-did-to-the-world)).

**Links that hold a token work.** The next section shows how.

Replacing the numbers one by one leaves the problems they solved. Whatever
replaces them has to solve those problems too.

## Links that hold a token

A stronger form of links that cost was tried on 16 October 2026 (`link_token`
in `research/oee/pilots/proto.py`; [The pilots' rules](settings.md#proto)).
Every link holds one token, paid by whoever makes it: a newborn pays for the
links it is born with. When a link lapses, its token goes back to one of its
two ends, drawn at random. When an agent dies, the tokens of its links pay for
the links that mend its neighbourhood. The rest is shared out like its
[estate](estate.md "The tokens a dead agent leaves, and who gets them.").

So the tokens the agents hold and the tokens in links always add up to the
world's tokens. The links can never outnumber the tokens, and no cap is
needed.

The rule was tried in B2 at 10,000 tokens with the cap removed
(`valence` = 0), two seeds each, for 2,000 iterations. In one pair of worlds,
new links were made only as in B2: at births, and where a death is mended. In
the other, an agent could also search for a new partner (`seek`). It pays a
token for each step of a random walk through the network. If the walk ends at
a stranger and both agree, the two are joined.

- **The network stays as healthy as B2's without the cap.** It is one piece,
  with no agent or link whose loss cuts anyone off, and every agent in the
  2-core. It has 5.2 to 5.6 links per agent. The best-connected agent is
  joined to 2 to 3% of the world (B2, with its cap of 8: 1.5%). Its dimension
  of 4.1 to 4.2 is a little above the health bar's 4. So is B2's own at
  10,000 tokens: 4.4 in its world of seed 1
  ([The sixteen worlds](eight-switches-at-once.md#the-sixteen-worlds)).
- **The world is smaller.** About a third of the tokens sit in links. The
  worlds hold 1,100 to 1,500 agents, against B2's 3,200 to 3,650.
- **Evolution goes on more strongly than in B2.** In contests a lifetime
  long, the brains of 1,000 hold 0.83 and 0.93 of the world against those of
  500. With the search, they hold 0.81 and 0.87. Those of 2,000 hold 0.73
  and 0.84 against those of 1,000 (with the search, 0.71 and 0.62). In B2
  the same contests give 0.67 to 0.69, and 0.65 to 0.76.
- **The age limit is still needed.** With links that hold a token and no age
  limit, births and deaths nearly stopped. The worlds froze
  ([Links that hold a token, and nobody dies of age](mixed-brains-costly-units.md#links-that-hold-a-token-and-nobody-dies-of-age)).

So the cap on links can be replaced. A link that holds a token is a rule
without a number. Without a cap, the worlds that use it keep what the cap was
for: their links do not run away, and no hub forms. And their brains evolve
further than B2's. The rule has since been thrown into B2 together with seven
other switches ([Eight switches at once](eight-switches-at-once.md)).

## What we learned

- **Each of B2's seven numbers does a job.** With all seven gone, the worlds
  without alliances died out. Those with alliances lived as one alliance, on
  networks that fail the health bar: a star and then a tree, or worlds of
  mayflies.
- **Alliances keep such a world alive. Whether they keep it evolving, two
  seeds cannot say.** A lifetime long, the three living worlds gave a tie, a
  win for the later brains and a win for the earlier ones.
- **One of the seven numbers is a definition, six are magnitudes.** The 2 of
  `prune_needs` says what the rule means. The 1.5, 2, 8, 0.3, 100 and 10 of
  the others could each be otherwise.
- **A replacement must solve the problem its number solved.** Birth by
  division stopped the missiles but not the starving. Gifts did not make the
  brains pay for the links that held their world.
- **Links that hold a token replace the cap on links,** and evolution goes on
  more strongly with them than in B2. The age limit has no replacement yet:
  without it, worlds freeze.

## Where to go next

- [Building B2: a world that cannot break](a-world-that-cannot-break.md):
  where each of the seven numbers came from, and what each prevents on its
  own.
- [Every setting](settings.md): every number as a setting, and `link_token`
  among the pilots' rules.
- [Other contests: a lottery, and alliances](another-game.md#alliances): what
  alliances do when B2's numbers are in place.
- [Eight switches at once](eight-switches-at-once.md): links that hold a token
  thrown into B2 together with seven other switches.

## How this was made

Every world on this page is a pilot, made with `research/oee/pilots/run.py`
and kept in `GraphOfLifeRuns/.pilots`. No experiment of the lab stands behind
it, so the page has no box of runs. Each figure says, folded beneath it,
which worlds it reads and how. The command lines, and what each check found,
are in `research/oee/pilots/LOG.md`.
[Every simulation](every-simulation.md) lists every pilot condition with its
settings.

- **B3 (batch N):** `B3a-s1`, `-s2`, `B3b-s1`, `-s2`, `B3c-s1` and `-s2`, in
  `GraphOfLifeRuns/.pilots/N`, at 10,000 tokens for 1,000 iterations. The
  numbers of the worlds that lived are read over their last 300 iterations
  into `book/results/numberless.json`, by
  `python3 book_figures.py numberless`. The rest come from each world's
  `series.jsonl`. The same batch holds `N0-s1` and `-s2`, the alliances
  without the home advantage.
- **The star of B3a, seed 1:** the best-connected agent's links over the
  number of other agents (`maxDegreeShare`), in every tenth row of its
  `series.jsonl`.
- **Contests a lifetime long:** `research/oee/checks/rejudge.py`, 100
  iterations, once in each era's world, seed 17, into
  `GraphOfLifeRuns/.pilots/rejudge_T100.json`. The eras played against
  themselves are in `rejudge_null_T100.json`.
- **Birth by division:** the prototype's switch `fission`, one seed each, at
  most 150 iterations (`research/oee/pilots/LOG.md`, 15 October 2026).
- **Links that hold a token (batch P):** `linktoken-s1`, `-s2`, and with the
  search `linkseek-s1`, `-s2`, in `GraphOfLifeRuns/.pilots/P`. Each is B2 at
  10,000 tokens with `valence=0` and the prototype's
  `engine=true link_token=true` (the search adds
  `seek=true seek_max=0 toll=path`), for 2,000 iterations. B2 itself is
  `P/B2-s1` and `-s2`. Their contests are in
  `GraphOfLifeRuns/.pilots/batch_P_T100.json`. Without the age limit as well:
  `Q/linknoage-s1` and `-s2`.

## How our view changed

- **15 October 2026.** As first written that morning, this page said that
  none of the six worlds was one in which evolution could work for long. The
  three that lived had ended the fighting, and with it nearly all selection.
  So the alliances made the difference between dying and living, but not
  between living and evolving. No contest had been played in these worlds.
  That evening, contests a lifetime long answered three ways: a tie, a win
  for the later brains, and a win for the earlier ones. So whether B3 keeps
  evolving, two seeds cannot say
  ([Did the worlds that lived keep evolving?](#did-the-worlds-that-lived-keep-evolving)).
- **24 October 2026.** When this page was rewritten in plain words, five of
  its statements were checked against the records and corrected:
  - B3a's world of seed 1 was said to have one agent joined to "between 58
    and 98%" of the others from iteration 130 to 330. The table said 98% of
    the world. The world's series shows more than half throughout, and at
    iteration 250 all but one of 4,383
    (`GraphOfLifeRuns/.pilots/N/B3a-s1/series.jsonl`).
  - The table said that without an age limit, births nearly stop in the
    tree. They ran at 3.5 per hundred agents an iteration, more than three
    times B2's 1.0 ([Old age](a-world-that-cannot-break.md#old-age)), and as
    many agents were cut off.
  - In B3b's world of seed 2, the brains of 500 were said to lose a little
    more than half of their own world. They kept a little more than half: the
    brains of 1,000 held 0.43 of it
    (`GraphOfLifeRuns/.pilots/rejudge_T100.json`).
  - The table said that the share kept at home stops a world of one-token
    zombies. Experiment 15, analysed on 16 October, found that it only thins
    them out. In B2's [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world, 52% of the agents hold a single token.
    When only the last token is kept, 77% do
    ([What each rule of death did](a-world-that-cannot-break.md#what-each-rule-of-death-did)).
  - The table said that the age limit stops a hoarding queen. The queen
    arose when an owner's stake counted double. With 1.5 and a share kept at
    home, none formed
    ([Holding one's place](a-world-that-cannot-break.md#holding-ones-place)).
    The age limit's job is the one that follows: no agent holds its place
    forever, and the dead make room for births.

<!-- turns -->
---

← [Revolutions, handover and gifts: the switches the world was built with](the-switches-the-world-was-built-with.md) · [Changing the rules](../README.md#changing-the-rules) · [Who inherits?](who-inherits.md) →

**Also about:** [The baselines B1 and B2](../README.md#the-baselines-b1-and-b2).

> [!info]- Pages that link here (8)
> [How this notebook is written](how-this-notebook-is-written.md) · [Every setting](settings.md) · [Who helps whom](who-helps-whom.md) · [Other contests: a lottery, and alliances](another-game.md) · [Revolutions, handover and gifts: the switches the world was built with](the-switches-the-world-was-built-with.md) · [Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md) · [Eight switches at once, and how sixteen worlds can test them](eight-switches-at-once.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
