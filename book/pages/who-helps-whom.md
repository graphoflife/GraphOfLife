# Who helps whom

> [!question] Questions of this page
> - What has the notebook found about cooperation, in every page that looked
>   for it?
> - Where does help exist in these worlds, and why so little of it?
> - What has been tried to make helping pay, and what happened?
> - What would have to change for cooperation to evolve?

## In short

Cooperation means paying a cost so that someone else gains, and in Graph of
Life the only act is a stake. A stake **helps** a [neighbour](the-world.md#the-network "An agent joined to another by a connection.") if, without it,
the neighbour would have been more likely to lose its [node](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."). Counted that way,
help is a by-product of revolutions in [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it."), almost absent in [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death."), and rare even
among allies, whom peace protects instead. Evolution gave up support stakes,
gifts without a return, but took up alliances, which two neighbours choose
together, at once. One alliance then held nearly the whole world and the
fighting ended, but one habit spreading once is not [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau.").

## Why cooperation matters

Why would anyone pay a cost so that someone else gains? Selection favours
whatever spreads. So how can it favour helping another to spread? It is one
of the oldest questions of evolution theory. Biologists know five answers,
which Martin Nowak sorted in 2006
([Five ways cooperation can evolve](do-agents-cooperate.md#five-ways-cooperation-can-evolve)):

- **kin selection**: help your relatives, who carry copies of your genes;
- **direct reciprocity**: I help you, and you help me later;
- **indirect reciprocity**: I help whoever has a good name;
- **network reciprocity**: on a network, helpers form clusters that keep
  cheats out;
- **group selection**: groups of helpers grow and split more than other
  groups.

For this notebook, cooperation is not a side topic. The great steps in the
history of life were steps of cooperation: genes joined into genomes, cells
into organisms, organisms into colonies and societies. A world that keeps
producing new kinds of things will very likely have to produce new kinds of
togetherness.

Several pages have looked for cooperation in Graph of Life. This page sets
what they found side by side, in plain words, and leads to them.

## What helping is, here

Picture a node under attack. Its holder, V, keeps 4 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") at home, and a
neighbour, U, stakes 6 on it to take it. Another neighbour, W, adds 3 tokens
to V's node. V and W mark their stakes revolutionary. Under B1's rules such
stakes join as a crowd against the largest staker: their 7 beat U's 6, and V
keeps its node. Without W's 3 tokens, V would have lost it. So W's stake saved
V's node, and W paid 3 tokens for it, which V gained
([A worked example](do-agents-cooperate.md#a-worked-example)).

That is the whole idea. A stake on a neighbour's node is a gift and a bid at
once. Its tokens go to that node, but if the stake wins, the node takes the
staker's [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions."). The stake **helps** if, without it, the neighbour would have
been more likely to lose its node ([Help](do-agents-cooperate.md#help)). Both
chances come from the world's own rule for deciding a node, counted exactly,
with nothing sampled. A stake that turns a lost node into a tie helps by half
a node.

Only two rules let one agent's stake raise another's chance of keeping its
node:

- in a **revolution**, a neighbour's revolutionary stake joins the holder's
  own in a crowd against the largest staker
  ([How a coalition takes a node](revolution.md));
- with **alliances**, an ally's stake counts for the holder's side
  ([What an alliance is](another-game.md#what-an-alliance-is)).

Without either, a stake on someone else's node is a bid against its holder.
Then help is impossible by rule.

Why count help by the rules, and not by how the helped fare? Because help goes
where it is needed. Those who are helped are those under attack, so comparing
them with the rest measures the attack, not the help
([Channels and feuds](gliders-and-who-helps-whom.md#channels-and-feuds)).

Help alone is not yet cooperation. An agent that stakes a little on every
neighbour helps some of them by accident. Three signs, each set against what
chance would give, would show more
([From help to cooperation](do-agents-cooperate.md#from-help-to-cooperation)):

- help goes to **clones**, agents carrying the helper's own brain, more often
  than clones are among its neighbours: the mark of kin selection;
- help is **returned** within 50 [games](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") more often than chance: the mark of
  direct reciprocity;
- help **goes round**, from one agent to a second, a third and back to the
  first: the mark of indirect reciprocity.

And three different things go by the name of cooperation. The record of play
can tell them apart
([What cooperation is here](gliders-and-who-helps-whom.md#what-cooperation-is-here)):

- **rescue**: a stake without which a neighbour would have lost its node;
- **peace**: neighbours that do not take each other's nodes;
- **exchange**: gifts of tokens that come back round, instead of running
  downhill ([Flows that go round](token-flow.md#flows-that-go-round)).

## What was found

The table sets the worlds of
[Gliders, and who helps whom](gliders-and-who-helps-whom.md) against the
three kinds of cooperation. All of them hold 10,000 tokens, and that page
reads their record of play: every game, with every agent's stakes. A number
is that of the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world, or the range over the worlds.

| world | rescue | peace | exchange |
|---|---|---|---|
| B1, ten worlds of Experiment 2 | a by-product of revolutions: 2.4 nodes saved per 100 agents a game, returned less than chance | none: 31 nodes per 100 agents are lost to attacks a game | none: gifts, net of what comes back, run downhill to the best-connected agents |
| B2, the ten worlds of Experiment 16 | almost none: about 0.01 nodes saved per 100 agents a game | none: 6.5 nodes per 100 agents are lost to attacks a game | none: gifts run downhill, a tribute from failed raids to the richer |
| pilot worlds of B2's rules with alliances, four worlds | rare: 0.015 to 0.085 nodes saved per 100 agents a game | by rule: allies cannot take each other's nodes | yes: gifts go round more than chance, but no ring lasts fifty games |

### B1: kin side by side, help by accident

In the thirty B1 worlds of 100,000 tokens, what kin selection needs is
there, but not the behaviour ([Do agents cooperate?](do-agents-cooperate.md)).
A tenth of an agent's neighbours carry its own [genotype](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from.") (9.6%), 200 times as
often as two agents drawn at random. But brains cannot see genotypes, and they
stake on kin as on strangers.

Stakes come back, but not as favours. When an agent stakes on a neighbour,
the neighbour stakes back in 90% of cases. That is about what the breadth of
B1's staking gives alone, since its agents stake on nearly every neighbour
([Stakes that are returned](where-the-tokens-flow.md#stakes-that-are-returned)).
Most such pairs stake exactly the same on each other (82%). That is a
handshake, which keeps their [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") alive at the price the rules set
([Do agents answer a stake with a stake?](do-agents-cooperate.md#do-agents-answer-a-stake-with-a-stake)).

Help itself, counted in ten B1 worlds of 10,000 tokens, saves 2.4 nodes per
100 agents a game in the median world. That is a thirteenth of the 31 lost to
attacks. It comes from the crowds of revolutionaries that B1's agents make by
spreading a little over every neighbour. And it shows none of the three signs.
It is returned less than chance (0.76 times), less of it goes round than
chance would have, and clones get about their share (1.03 times). It doubled
between [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 100 and 1,000, and has gone no further since
([In B1, help by accident](gliders-and-who-helps-whom.md#in-b1-help-by-accident)).

### B2: almost no help, and feuds

In B2's ten worlds, help saves about 0.01 nodes per 100 agents a game (0.005
to 0.028), and less as the worlds evolve. B2 has revolutions, but an agent
that stakes on a neighbour puts it all on one. So the small stakes a crowd is
made of are rare, and an owner's stake counts one and a half times against
any crowd. Coalitions hardly ever win there
([How often a coalition wins](revolution.md#how-often-a-coalition-wins)).

The little help there is passes both ways more often than chance. Over the
ten worlds, 156 pairs helped each other within 50 games, where chance would
make 28. Help also goes to clones 1.2 times their share. But a situation
explains both better than a choice. Help needs two neighbours whose stakes
can join against one attacker, and that situation is the same seen from
either side. And no agent can see who staked on its node, so none could repay
help if it wanted to
([In B2, nobody helps](gliders-and-who-helps-whom.md#in-b2-nobody-helps)).

Take two B2 agents that stake on each other for three games or more. They
keep their own nodes less often than other agents that stake on others: 0.80
of the time against 0.89. In B2 a stake on someone's node is an attack, so
such a pair is in a **feud**
([Channels and feuds](gliders-and-who-helps-whom.md#channels-and-feuds)).

The flow of tokens shows the one structure B2's agents add. It falls into
pairs of partners, and they last a little longer than chance. Of these pairs,
9% last five games. When each agent's stakes are dealt among its neighbours at
random, 4% do ([How long modules live](flow-modules.md#how-long-modules-live)).
Whether these pairs are the feuds has not been checked. In B1, the groups the
flow keeps to are the network's own regions.

### Among allies: peace instead of rescue

In the pilot worlds with alliances, rescue is rare too: 28 to 150 times less
than in B1. Yet agents with an ally keep their node in 99–100% of games, and
agents without one in 90–95%. An ally's stake counts for the holder, so an
allied neighbour can never take the node: the alliance protects by peace. The
little rescue there is passes both ways more often than chance. But only
allies can rescue each other, and an alliance is mutual by definition.

Allies also give to allies who give to allies. These are the only worlds whose
gifts go round more than chance would make. But no ring lasts fifty games on
the same connections
([Among allies, peace instead of rescue](gliders-and-who-helps-whom.md#among-allies-peace-instead-of-rescue)).
In the same pilots without alliances, which have no revolutions either, no
stake ever helped.

## What was tried to make helping pay

Rules, inputs and strategies were added to give agents a reason or a way to
help. Most ran in pilot worlds, two [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") each, so they show leads, not laws
([Pilot worlds](the-lab.md#pilot-worlds)). The pilots of other contests and
of alliances ran without revolutions, since a revolution is itself a rule
about who takes a node.

| what was tried | what happened | page |
|---|---|---|
| **support**: a stake that adds to a neighbour's defence | given up: it counted once, a stake at home one and a half times; it fell from 200–300 tokens a game to about 20 by iteration 600 | [What the pilots tried](how-to-tell-a-world-that-keeps-evolving.md#what-the-pilots-tried) |
| support counted twice | given up anyway: from about 2,700 tokens in the first game to 81 and 175 by iteration 100 | [One alliance takes the world](another-game.md#one-alliance-takes-the-world) |
| a **yield on stakes returned** | it went to whoever had the most partners, and built [stars](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub."), not partnerships | [Room to go on](how-to-tell-a-world-that-keeps-evolving.md#room-to-go-on) |
| stakes **pooled by genotype** | fighting cut to about a third, and groups of clones of 8 and 11 agents on average; no better play | [What the contests did](another-game.md#what-the-contests-did) |
| a **sense of one's clones**, alone | no groups: groups of clones stayed at 3 agents, and the brains used the sense less than the [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") did | [What the contests did](another-game.md#what-the-contests-did) |
| **family tags**, sensed or pooled | sensed, the fighting stayed as it was; pooled, families of hundreds rose and fell like empires; no better play either way | [A password and a crowd, for free](another-game.md#a-password-and-a-crowd-for-free) |
| **kin pooled**: sides made by descent, within two [mutations](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") | peace, sooner, with nobody deciding anything; but small sides, 2 to 14% of the world | [One alliance takes the world](another-game.md#one-alliance-takes-the-world) |
| **alliances** by mutual consent | taken up at once: one alliance held 99% of the world by iteration 2,000 | [Alliances](another-game.md#alliances) |
| alliances in worlds without B2's hand-set numbers | they made the difference between dying and living, not between living and evolving | [Rules without numbers](rules-without-numbers.md#what-the-six-worlds-did) |
| **crews**: raiders written by hand whose revolutionary stakes join against one stranger (Experiment 33) | no crew took hold, planted at iteration 2,000, 5,000 or 10,000 | [Raiders that pool their stakes](does-evolution-close-the-door.md#raiders-that-pool-their-stakes) |
| a **sense of direction** and a **sense of who helped** | no gliders, and no more help | [Both inputs, tested](gliders-and-who-helps-whom.md#both-inputs-tested) |

**Pooled sides bring peace, not better play.** Pooling the stakes of clones,
of a family or of near kin cut the fighting, with no choice made by anyone.
Pooled by family, families grew to 150 and 490 agents on average. The
fighting fell to 1.3 to 1.6 nodes per 100 agents a game. But pooling by
genotype or by family found no better play
([Other contests: a lottery, and alliances](another-game.md#what-we-learned)).

**Alliances were taken up.** They are the first rule added for open-ended
evolution that evolution took up rather than gave up. In every game, an agent
says of each neighbour whether it wants it as an ally. Two that both say yes
are allies for that game, and their stakes count as one side. In the two long
pilot worlds, one alliance held 99% and 97% of the agents by iteration 2,000.
It spread as a habit, saying yes to every neighbour, not as a family.

And the fighting ended. Without alliances, 7.6 to 8.0 nodes in a hundred
changed hands in a game from iteration 500 to 1,000; with them, 2.2 and 3.4.
In the two long worlds it fell to about 0.3 in their last 500 iterations (0.2
and 0.4)
([One alliance takes the world](another-game.md#one-alliance-takes-the-world)).
Judged over a [lifetime](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death."), the brains inside the alliance kept improving slowly,
as brains do without alliances
([Does a world of allies keep evolving?](another-game.md#does-a-world-of-allies-keep-evolving)).
Where all of B2's hand-set numbers were taken away, alliances kept worlds
alive. Both worlds without them died, and three of the four with them lived.
Each became one alliance, and none could evolve for long
([Rules without numbers](rules-without-numbers.md#what-the-six-worlds-did)).

**Groups written by hand did no better.** Crews of raiders, whose
revolutionary stakes join against the same weak stranger, were planted in
B2's worlds at iterations 2,000, 5,000 and 10,000. None took hold, and they
died before lone raiders in 42 of 45 places. A group helps only where it can
gather. On B2's network, a stranger next to a patch touches only one or two
of its agents
([Why the raiders stopped winning](does-evolution-close-the-door.md#verdict-on-experiment-33-refuted)).

**More information made no cooperation.** The pilot worlds of batch P gave
B2's brains two new inputs per neighbour. One was a sense of direction, *this
is the node I came from*. The other was a sense of who helped, *this
neighbour's stake saved my node in the last game*. With the sense of
direction, chains of takeovers went straight back to the node they had just
left more often than in the controls. But so did one world with only the
sense of who helped, 1.66 times as often as chance. So two seeds cannot lay
the going back to the sense of direction. And chains travelled no further
than a random walk would.

Help stayed where it was: 0.006–0.012 nodes saved per 100 agents a game in
the six worlds with an input, against 0.009 in the controls. It was returned
no more than there. In 200 games, help passed both ways in 4 and 2 pairs, in
two of those worlds. In the two controls it did in 6 and 2. Gifts still ran
downhill, and evolution went on at about the control's pace
([Both inputs, tested](gliders-and-who-helps-whom.md#both-inputs-tested)).

## Why so little

Three reasons recur.

**The rules leave little room for help.** Where the largest stake on a node
takes it, a stake on someone else's node is an attack, unless a rule says
otherwise. B1's revolutions say otherwise by accident: a revolutionary stake
can keep a node for its holder. So B1 has help as a by-product. In B2,
coalitions hardly ever win, and so there is hardly any help. Wherever help or
peace appeared in these worlds, a rule made it possible: revolutions, pooled
sides, alliances.

**There is no future to reciprocate in.** Reciprocity needs the same two
agents to meet again. In B1 a connection lasts, but the brains at its ends do
not. In only 4% of connections do the same two brains face each other a game
later ([How long partners last](do-agents-cooperate.md#how-long-partners-last)).
The brain that was helped is often gone before it could return the favour.
In B2, a pair that keeps staking on each other is more often a feud than a
partnership. Nothing in the game rewards keeping a partner.

**Seeing is not enough.** Giving brains more to see did not by itself create
help. They were given a sense of their clones, of their family, of where they
came from, and of who helped them. Every [message](messages.md "Numbers an agent writes to itself and to each neighbour every phase, which their brains read.") already names its writer's
genotype, so kin could tell each other with one comparison. But B1's nearly
blind brains cannot make it
([Could kin recognise each other?](what-agents-say-to-each-other.md#could-kin-recognise-each-other)).
A brain can read what it is told; whether reading pays depends on the rules.
A sense of kin made groups only where a rule pooled the stakes of kin. An
input naming the helper changed nothing where helping itself was rare.

## What would have to change

For cooperation to evolve in these worlds, the evidence points to rules
rather than brains. Three things would open the way:

1. **A stake that can defend another's node, without being a free gift.**
   Support that counts only if it is returned, or that buys a share of what
   it defends, would be such a stake. Plain support was given up, even counted
   twice. Alliances, which make support mutual by consent, were taken up at
   once.
2. **An identity that outlives a lost node**, so that a debt can be paid
   back: an agent whose brain survives being conquered. The inputs of batch P,
   where a brain came from and who helped it, were a first step, and not
   enough.
3. **A reason to stay.** In B2 the flow of tokens runs one way, because a link
   no longer has to carry tokens to survive
   ([Lightning in every world](where-the-tokens-flow.md#lightning-in-every-world)).
   Where links must carry tokens, agents stake on each other: the raw
   material of exchange is already there.

Alliances show that a rule which makes help mutual is adopted quickly. What
they did not show is open-ended evolution. One convention, *say yes to your
neighbours*, spreads until it holds everywhere, as a single useful mutation
would. Then the world is at peace, and conquest, which copies brains from node
to node, nearly stops. The brains go on improving slowly, but nothing new
follows the alliance itself.

The open question is a rule under which cooperation is adopted *and*
competition between groups goes on. It might be a way to cheat an alliance
that pays for a while. It might be a cost to being in one, or a limit to its
reach. Then alliances would compete with each other instead of merging into
one
([Other contests: a lottery, and alliances](another-game.md#what-we-learned)).
Measured in worlds of allies, flow modules would show whether their groups
outlast the network's regions
([Flow modules: the groups tokens stay in](flow-modules.md#what-we-learned)).

## What we learned

- **Help has an exact meaning here, and it needs a rule.** A stake helps if,
  without it, the neighbour would have been more likely to lose its node.
  Only revolutions and alliances make help possible.
- **B1 helps by accident, B2 hardly at all, and allies keep the peace
  instead.** In B1 help is returned less than chance. Where it comes back more
  often, in B2 and among allies, the situations and rules that allow it
  explain why.
- **There is no future to reciprocate in.** The same two brains seldom face
  each other for long, and nothing in the game rewards keeping a partner.
- **Seeing is not enough.** Senses of kin, of family, of direction and of who
  helped made no cooperation where the rules gave nothing to gain by it.
- **A rule that makes help mutual is taken up.** Alliances brought peace, and
  gifts that go round. But one habit spreading once is not open-ended
  evolution, and what would keep cooperation contested is still open.

## Where to go next

- [Do agents cooperate? Help, kin and partners](do-agents-cooperate.md): help
  defined exactly, with a worked example, and what B1's agents do with kin
  and partners.
- [Gliders, and who helps whom](gliders-and-who-helps-whom.md): help counted
  in B1, in B2 and among allies, and the two new inputs tested.
- [Other contests: a lottery, and alliances](another-game.md): stakes pooled
  by clones and families, senses of kin, and alliances.
- [Flow modules: the groups tokens stay in](flow-modules.md): the groups the
  flow of tokens keeps to, and B2's pairs of partners.

## How our view changed

- **24 October 2026.** Since 16 October, this page said that with the two new
  inputs of batch P, help stayed rare and "none was returned". It also said
  that "the sense of direction was used to go back".
  [Gliders, and who helps whom](gliders-and-who-helps-whom.md#both-inputs-tested)
  has since set those numbers right from batch P's records. Help was returned
  in 4 and 2 pairs in two of the six worlds with an input. The two controls
  had 6 and 2. And a world with only the sense of who helped went back 1.66
  times as often as chance. So two seeds cannot lay the going back to the
  sense of direction.
- **24 October 2026.** Since 16 October, this page said of B1 that "stakes are
  returned as often as chance has it". It gave
  [Do agents cooperate?](do-agents-cooperate.md#do-agents-answer-a-stake-with-a-stake)
  as its source. That page never set the return of stakes against chance.
  [Where tokens flow](where-the-tokens-flow.md#stakes-that-are-returned) finds
  that a stake comes back in 90% of cases, about what the breadth of B1's
  staking gives alone. And help, counted by the rules, is returned less than
  chance: 0.76 times
  ([In B1, help by accident](gliders-and-who-helps-whom.md#in-b1-help-by-accident)).

<!-- turns -->
---

[Groups and cooperation](../README.md#groups-and-cooperation) · [Do agents cooperate? Help, kin and partners](do-agents-cooperate.md) →

> [!info]- Pages that link here (9)
> [What this notebook is about](what-this-notebook-is-about.md) · [Questioning the mechanics](questioning-the-mechanics.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [Flow modules: the groups tokens stay in](flow-modules.md) · [How alike the agents are](how-alike-the-agents-are.md) · [What agents say to each other](what-agents-say-to-each-other.md) · [Do agents cooperate? Help, kin and partners](do-agents-cooperate.md) · [Other contests: a lottery, and alliances](another-game.md) · [Gliders, and who helps whom](gliders-and-who-helps-whom.md)
<!-- /turns -->
