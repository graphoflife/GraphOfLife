# Every setting

> [!question] Questions of this page
> - What fixes a run of Graph of Life before it starts?
> - What does each setting do, and which values can it take?
> - What is each setting in the two baselines, B1 and B2?

## In short

A run of Graph of Life is fixed by two things: its **settings** and its
**seed** ([Seeds and the random stream](random-numbers.md)). A setting is one
number or switch of a run's rules, fixed before the run starts. The settings
say how many [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") there are, how large the brains are, and which rules are
on. This page lists the 41 settings that decide a world, and one more,
`archive_every`, that decides what a run keeps. It says what each does, and gives its
value in the two baselines, B1 and B2
([B1 and B2: the two baselines](b1-and-b2.md)). An experiment starts from one
baseline and changes a few settings, and its table of settings links each of
them to its entry here.

## How to read an entry

Each entry below is headed by the setting's name, spelt as the program and
the plans spell it. It says what the setting does, which values it can take,
and what each value means. For most of the rules added later, 0 or off is the
old rule. If a setting works only together with another one, the entry says
which. Its last line gives the setting's value in the two baselines; the plans
that define them are the files `book/experiments/B1.json` and `B2.json`.

Three letters come back often. *T* is the number of tokens, and *n* the
number of founders. *k* is the number of [neighbours](the-world.md#the-network "An agent joined to another by a connection.") each founder starts with
in the ring.

## At a glance

Every setting, with its value in each baseline. A name leads to its entry.

| setting | B1 | B2 | in one line |
|---|---|---|---|
| [`total_tokens`](#total_tokens) | 10,000 (chosen per experiment) | 10,000 (chosen per experiment) | tokens in the world |
| [`n_nodes`](#n_nodes) | 0 → 100 founders | 0 → 100 founders | how many founders |
| [`k_neighbors`](#k_neighbors) | 0 → 5, wired as 4 | 0 → 5, wired as 4 | neighbours in the starting ring |
| [`rewire_p`](#rewire_p) | 0.2 | 0.2 | shortcuts in the starting ring |
| [`hidden_layers`](#hidden_layers) | 50, 45, 40, 35, 30 | 16 | the brain's hidden layers |
| [`brain_kind`](#brain_kind) | float16 | float16 | how weights are stored |
| [`brain_bits`](#brain_bits) | 16 (unused) | 16 (unused) | only for binary brains |
| [`message_amount`](#message_amount) | 30 | 8 | numbers per message |
| [`random_input_amount`](#random_input_amount) | 5 | 5 | random inputs per look |
| [`exchange_messages`](#exchange_messages) | on | on | messages at all |
| [`message_prepass`](#message_prepass) | on | off | a look that only talks |
| [`allow_handover`](#allow_handover) | on | off | parents hand [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") to children |
| [`allow_revolutions`](#allow_revolutions) | on | on | coalitions can take [nodes](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") |
| [`allow_gifting`](#allow_gifting) | off | off | gifts of tokens |
| [`random_decisions`](#random_decisions) | off | off | the control without brains |
| [`prune_after`](#prune_after) | blotto | blotto | when unused connections are cut |
| [`inactive_window`](#inactive_window) | phase | phase | how long a connection may be unused |
| [`redistribution`](#redistribution) | uniform | uniform | who gets the tokens of the dead |
| [`tokens_created_per_phase`](#tokens_created_per_phase) | 0 | 0 | new tokens per cleanup |
| [`mutate_on_replication`](#mutate_on_replication) | off | on | brains change only when copied |
| [`incumbency`](#incumbency) | 1 | 1.5 | home advantage in the game |
| [`birth_links`](#birth_links) | 0 | 2 | links a newborn is born with |
| [`mend_on_death`](#mend_on_death) | off | on | no death splits the world |
| [`prune_needs`](#prune_needs) | 0 | 2 | only redundant connections lapse |
| [`valence`](#valence) | 0 | 8 | most connections a brain may choose |
| [`keep_home`](#keep_home) | 0 | 0.3 | a share of the tokens stays at home |
| [`max_age`](#max_age) | 0 | 100 | the longest life |
| [`buy_life`](#buy_life) | off | off | the old may buy more life |
| [`min_child`](#min_child) | 1 | 10 | the smallest child |
| [`child_floor`](#child_floor) | fixed | fixed | the least a child may start with |
| [`mix_at_birth`](#mix_at_birth) | off | off | a newborn's brain mixed from two |
| [`mix_mates`](#mix_mates) | any | any | who may be the mate |
| [`self_message`](#self_message) | own | own | what an agent writes to itself |
| [`link_search`](#link_search) | off | off | links to strangers, found by a search |
| [`link_steering`](#link_steering) | random | random | which way a search steps |
| [`link_consent`](#link_consent) | mutual | mutual | who must agree to a link found |
| [`proto`](#proto) | none | none | the pilots' rules, by name |
| [`mutation_probability`](#mutation_probability) | 0.2 | 0.2 | how often a brain changes |
| [`mutation_noise_std`](#mutation_noise_std) | 0.2 | 0.2 | how big a change is |
| [`mutation_sparsity`](#mutation_sparsity) | 0.1 | 0.1 | how many weights a change touches |
| [`extinction_threshold`](#extinction_threshold) | 20 | 20 | when a world counts as dead |
| [`archive_every`](#archive_every) | 0 | 0 | whole worlds kept for later |

## The world and its start

How many tokens a world holds, and the ring of founders it starts as
([The starting ring](starting-ring.md)).

### total_tokens

The number of tokens in the world, *T*. Tokens are never made or destroyed,
so all agents' tokens add up to *T* after every phase of every
[iteration](one-iteration.md).
Only [`tokens_created_per_phase`](#tokens_created_per_phase) above 0 could add
tokens, and no run of this notebook uses it. Every living agent holds at least
one token, so a world can never hold more than *T* agents.

The pages on how the simulation works teach with *T* = 10,000. The study of
the baseline uses *T* = 100,000.
[How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)
varies it from 800 to 409,600.

**B1 and B2:** 10,000 for a plan that names no size; every experiment chooses
its own.

### n_nodes

How many **founders** the world starts with. The value 0 means one founder
per hundred tokens:

$$
n = \left\lfloor \frac{T}{100} \right\rfloor ,
$$

where ⌊*x*⌋ is *x* rounded down. With *T* = 10,000 that is 100 founders. Each
founder gets ⌊*T*/*n*⌋ tokens. What the division leaves over is dealt out by
one [multinomial draw](random-numbers.md#how-a-draw-becomes-a-decision):
every founder has the same chance at every token
([The starting ring](starting-ring.md)).

**B1 and B2:** 0, so one founder per hundred tokens.

### k_neighbors

How many neighbours each founder starts with in the ring. The value 0 means
one neighbour for every hundred founders, but never fewer than 5:

$$
k = \max\!\left( \left\lfloor \frac{n}{100} \right\rfloor ,\ 5 \right).
$$

The ring joins every founder to its ⌊*k*/2⌋ nearest founders on either side,
so an odd *k* is wired as *k* − 1. In the baseline *k* = 5, and every founder
starts with **4** neighbours. Below 600 founders (*T* < 60,000), *k* is always
5, wired as 4. With the 1,000 founders of *T* = 100,000 it is 10: five on
either side.

**B1 and B2:** 0.

### rewire_p

In the starting ring, the probability with which each connection is moved to
a founder chosen at random. The moved connections are the shortcuts that make
the ring a "small world", as Watts and Strogatz (1998) built it
([The starting ring](starting-ring.md)).

**B1 and B2:** 0.2.

## The brain

The shape of the neural network every agent decides with, and what it reads
([The brain](the-brain.md)).

### hidden_layers

The widths of the brain's hidden layers, from input to output. B1's brain has
five: 50, 45, 40, 35 and 30 neurons. With B1's 154 inputs and 45 outputs, that
gives 15,550 weights and 245 biases: 15,795 numbers in all.

**B1:** 50, 45, 40, 35, 30. **B2:** one layer of 16.

### brain_kind

How a brain's numbers are stored:

- `float`: as 64-bit floating-point numbers;
- `float16`: as 16-bit ones, good to about three significant decimal digits;
- `binary`: as −1, 0 or +1.

Under `float16` the weights are *stored* in 16 bits, which takes a quarter of
the memory. But every calculation is done in 64 bits. Every result in this
notebook is for `float16`.

**B1 and B2:** `float16`.

### brain_bits

Used only by `binary` brains, to say how many input rows encode one number.
Float and float16 brains ignore it.

**B1 and B2:** 16, unused.

### message_amount

How many numbers one message holds, *m*. In every phase, every agent writes
one message to itself and one to each neighbour. For each
[candidate](one-iteration.md#phase-1-reproduction) it looks at, itself or a
neighbour, a brain reads four messages:

- its own message to itself;
- its own message to the candidate;
- the candidate's message to it;
- the candidate's message to itself.

So a brain has 4*m* message inputs and *m* message outputs: 120 and 30 in B1
([Messages](messages.md)).

**B1:** 30. **B2:** 8.

### random_input_amount

How many random numbers a brain reads for each candidate it looks at. Each is
drawn uniformly from −2 to 2, fresh every time. They let a brain act
differently in situations that look the same
([What a brain sees](brain-inputs.md)).

**B1 and B2:** 5.

## The rules the world was built with

The rules the [engine](the-lab.md#the-engine) had from the start, with
switches to turn them off or change them. In B1 every one of them is at the
engine's first rule.
[The switches the world was built with](the-switches-the-world-was-built-with.md)
tries several of them in B2.

### exchange_messages

Whether agents write and read messages at all. Off, every message input is 0.

**B1 and B2:** on.

### message_prepass

Whether each phase begins with an extra look in which every agent only writes
messages. On, the look that acts reads messages written in this phase, about
the world as it now is. Off, it reads those of the phase before, written
before the births, deaths and conquests since ([Messages](messages.md)). The
extra look costs a second pass of every brain in every phase. It needs
`exchange_messages`.

**B1:** on. **B2:** off.

### allow_handover

Whether a parent may move some of its own connections to its newborn child.
Off, a child is joined only to the candidates chosen for it, and the parent
keeps all its connections.

**B1:** on. **B2:** off.

### allow_revolutions

Whether a coalition of smaller stakers can take a node from its largest
staker ([How a coalition takes a node](revolution.md)). Off, every node goes
to whoever staked the most on it.

**B1 and B2:** on.

### allow_gifting

Whether agents may give tokens to their neighbours during reproduction. Off,
the brain has neither the gift inputs nor the gift outputs.
[Gifts, with every other rule](gifts-with-every-other-rule.md) turns it on,
beside each of the other rules in turn.

**B1 and B2:** off.

### random_decisions

The control without brains. On, every number a brain would produce is
replaced by an independent draw from the standard normal distribution
(mean 0, standard deviation 1). Every decision is then taken from those
numbers by exactly the same rules. The brains still exist, are still copied
and still mutate; they are just never asked.
[Do the brains matter?](do-the-brains-matter.md) uses it.

**B1 and B2:** off.

### prune_after

After which phase connections that carried no tokens are cut: `blotto` (the
[game](one-iteration.md#phase-2-the-game)), `reproduction`, or `both`. In the game, a connection carries tokens
when one of its two agents stakes at least one token on the other.

**B1 and B2:** `blotto`.

### inactive_window

How long a connection may go unused before it is cut:

- `phase`: it must carry tokens in the phase being judged, or have been made
  in it;
- `iteration`: the phase before counts too.

In B1, with `phase` and with `prune_after` = `blotto`, **every connection that
no tokens crossed in a game is cut at the end of that game**. That includes
the ones a newborn got in the reproduction phase just before.
([`prune_needs`](#prune_needs) can spare some of them.)

**B1 and B2:** `phase`.

### redistribution

Who gets the tokens of the agents a [cleanup](one-iteration.md#cleanup)
removes. There are six rules:

- `uniform`: every surviving agent has the same chance at every token. The
  pooled tokens are dealt out by one multinomial draw with equal
  probabilities.
- `by_tokens`: each survivor's chance is weighted by what it already holds.
- `poorest`: like water poured on an uneven floor, which fills the lowest
  places first. The poorest survivors are raised to one common level, as high
  as the pool reaches. The whole tokens left over go one each to agents drawn
  at that level.
- `newborns`: the pool goes to the children born this iteration that are
  still alive, each with the same chance at every token.
- `parents`: it goes to those children's parents, once per child, which pays
  for having children.
- `children`: each dead agent's estate stays in the family. It goes to the
  agent's own living children, each with the same chance at every token. An
  estate that finds no living child is dealt out as `uniform` does.

When no child born this iteration is alive, `newborns` and `parents` deal the
pool out as `uniform` does ([The estate of the dead](estate.md)).

**B1 and B2:** `uniform`.

### tokens_created_per_phase

Tokens added to the pool at every cleanup, and dealt out like the tokens of
the dead. The value 0 keeps the supply fixed. No run of this notebook uses
anything else.

**B1 and B2:** 0.

## Rules added for open-ended evolution

Nine rules were added for [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau.") on 12 October 2026. Each
answers one way a world was seen to fail once its brains could see. A tenth,
`mix_at_birth`, came later from the pilots of
[Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md).
Four more followed: `buy_life`, `mix_mates`, `self_message` and
`child_floor`.

Every one is off in B1, at the old rule, and in every run made before B2, so
those runs are unchanged by them. B2 turns on all nine of the first ones. All
are local: they read an agent, its connections and its neighbours, and nothing
further away.

### mutate_on_replication

Whether a brain changes only when it is copied: into a newborn, or into a
node its owner has just won. Off, every brain in the world may also change
after every game, wherever it sits. That wears an adaptation away within a
few dozen iterations of its making.

**B1:** off. **B2:** on.

### incumbency

How many times over an agent's stake on its own node counts when the game
decides who takes the node, *h*. With *h* = 1, the old rule, a defender needs
to outstake every attacker. With *h* = 1.5, an attacker must stake more than
one and a half times what the defender keeps at home. A defender's 10 tokens
at home then count as 15, so an attacker needs 16 to be the largest staker.
Only the choice of the winner is weighed: the node still ends with the tokens
actually staked on it.

**B1:** 1. **B2:** 1.5.

### birth_links

How many of its parent's neighbours a newborn is joined to, drawn at random,
before the links its brain chooses. Above 0, the newborn is also always
joined to its parent. With 0, the newborn has only the links its brain
chooses, and is often a leaf, joined to one agent alone. With 2, every birth
closes triangles. The links are drawn at random rather than by the brain's
liking: a liking for well-connected agents is [preferential attachment](preferential-attachment.md "The well connected gaining new connections faster than the poorly connected."), and
makes [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.").

**B1:** 0. **B2:** 2.

### mend_on_death

Whether the neighbours of an agent that dies are joined up again. Picture the
edge of a hole closing: the neighbours close in a ring round the gap.

Precisely, the neighbours fall into groups: those linked to one another,
directly or through other neighbours, form one group. Each group is joined to
the next through its least-connected member (the lowest id among equals).
With three groups or more, the last is also joined back to the first. A
neighbourhood that was one piece gains nothing.

With it on, no death can split the world. So the cleanup, which keeps only
the largest connected piece, never removes a piece cut off by a death.
Cutting unused connections can still split a world, unless
[`prune_needs`](#prune_needs) is at least 1
([A world that cannot break](a-world-that-cannot-break.md)).

**B1:** off. **B2:** on.

### prune_needs

How many neighbours the two ends of an unused connection must have in common
for it to be cut. With 0, every unused connection is cut, [bridges](bridges.md "A connection that lies on no loop: cut it and the network falls into two pieces.") as readily
as anything else. With 1, the two ends always stay joined through a third
agent, so pruning can never split the world. With 2, they stay joined twice
over, so pruning cannot leave anybody hanging on a single agent either.

**B1:** 0. **B2:** 2.

### valence

The most connections an agent's brain may choose to hold; 0 means no limit.
The links a newborn is born with (`birth_links`) and the links mending makes
are made whatever the limit. Like any other link, they count towards it. An
agent at its limit cannot have a child: it is crowded. That is what keeps a
dense region from growing denser.

**B1:** 0. **B2:** 8.

### keep_home

The share of an agent's tokens that stays on its own node whatever its brain
says. It is rounded up, and never less than one token. A parent also always
keeps at least one token. Under B2's 0.3, an agent holding 10 tokens keeps 3
at home, and stakes the other 7 as its brain says. One holding 2 tokens keeps
1.

Above 0, no agent can stake or breed itself to death: a world then loses
agents only to age ([`max_age`](#max_age)) and to being cut off. A tiny share
is a floor of one token. A larger one also keeps agents from staking nearly
everything on a single neighbour.

**B1:** 0. **B2:** 0.3.

### max_age

The age at which an agent dies, in iterations; 0 means never. Its tokens are
shared out like those of any other dead agent.

**B1:** 0. **B2:** 100.

### buy_life

Whether an agent that has reached [`max_age`](#max_age) may live on by buying
its life. It pays one token an iteration into the estate of the dead, at the
iteration's first cleanup. The token is shared out with the estate. An
agent that would have to give its last token dies as before. So a rich agent
can live long, and a poor one cannot. In a world whose agents are rich, fewer
die each iteration, so fewer births replace them. It needs `max_age`.

**B1 and B2:** off.

### min_child

The fewest tokens a child can be born with. A parent whose brain would give a
child less has no child.

**B1:** 1, so any child at all. **B2:** 10.

### child_floor

The least a child may start with:

- `fixed`, the old rule: [`min_child`](#min_child) alone;
- `neighbours`: also at least the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") holding of its parent's neighbours,
  at the moment of the birth. A parent that would give less has no child.

Where births are cheap, a world fills with children, each with a token or
two. Births are cheap, for example, when the estate of the dead is paid to the
newborns. Then almost every agent holds the same few tokens
([Room for the steadiest world](room-for-the-steadiest-world.md)). Under
`neighbours`, a child never starts with less than the median holding of the
agents it is born among. So a world can grow only as rich as it is. The rule
reads the parent's neighbourhood and nothing further, and has no number of
its own.

In the worlds of
[Births as rich as the neighbours](room-for-the-steadiest-world.md#births-as-rich-as-the-neighbours) it
changed nothing measurable. Their neighbours were as poor as the world, and
every child already started with B2's 10 tokens.
[The floor in plain B2 and in B1](room-for-the-steadiest-world.md#the-floor-in-plain-b2-and-in-b1) tries it
in plain B2, and in B1, where a child may start with a single token.

**B1 and B2:** `fixed`.

### mix_at_birth

Whether a newborn's brain is mixed from two brains before it mutates: its
parent's and a mate's, one of the parent's neighbours drawn at random. With
one hidden layer, each hidden unit comes from one brain or the other, by an
even chance. It comes whole: the weights into it, its bias and the weights
out of it. Each output's bias comes from one or the other in the same way. With more
layers, each unit of every layer comes whole, with its weights in and its
bias. A unit that works can then spread without the rest of the brain it
arose in.

The mixture is a new [genotype](genotype.md "The identity of a brain: a number that a brain keeps until it changes, and that records whose copy it was."), unless it is the parent's brain again, as it
is when the mate has the parent's genotype. A parent without a neighbour
gives its newborn a copy. Off, a newborn has a copy of its parent's brain.

The rule was first played in the pilots of
[Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md).
In those pilots and in the batches after them, worlds with it came closest to
a healthy and diverse world ([Healthy and rich](healthy-and-rich.md)).

**B1 and B2:** off.

### mix_mates

Under [`mix_at_birth`](#mix_at_birth), who may be the mate:

- `any`: any neighbour of the parent, drawn at random;
- `parents`: only a neighbour with a living child of its own. An agent counts
  as a parent from the birth of its first child that lives.

A mate passes half of its brain on, and pays nothing for the child. With
`any`, an agent that never breeds still spreads its brain through other
agents' children; with `parents` it cannot. A parent without such a neighbour
gives its newborn a copy of its own brain. It needs `mix_at_birth`.

**B1 and B2:** `any`.

### self_message

What an agent writes to itself, which each neighbour reads as one of the four
messages in its column:

- `own`, the old rule: what the agent's own column says;
- `largest`: the largest, number by number, of what it writes to all its
  candidates, itself included.

A brain scores each candidate from that candidate's column alone. So under
`own`, no message can carry news of a third agent. Whether a stranger is next
door reaches a teammate only through what the agent does
([What a brain can compute](what-a-brain-can-compute.md)). Under `largest`, a
mark one column makes ("this neighbour is a stranger") reaches every
neighbour a phase later. Over a few phases, a team can hold a field, such as
each member's nearness to its frontier. It adds no weights and no inputs. But
an agent's message to itself is then at least what it tells any neighbour.

**B1 and B2:** `own`.

## Links to strangers

Without these three settings, every link an agent makes stays within its own
neighbourhood: a newborn's links go to its parent's neighbours. With them, an
agent can reach further, mostly at a price
([Links to strangers](links-to-strangers.md)).

### link_search

Whether an agent may search beyond its neighbours for someone to link with,
and how. Every step of a jump or a walk costs the searcher one token, paid to
the agent the step enters. So a search moves tokens but never makes or
destroys any, and no agent spends its last token on one. A reach is free. How
much an agent spends is its brain's choice: a share of what reproduction left
it. The share is given by the brain's SEEK head, a pair of its outputs read
like a gift's share ([What a brain says](brain-outputs.md)).

- **off**: no search.
- **jump**: in one iteration, a walk as many steps long as the tokens spent,
  from the agent's own node. A link is proposed to the agent it ends on, if
  that is a stranger and the searcher wants it.
- **walk**: a scout goes one step further in every iteration its agent pays
  for, on from where it stood. So distance costs time as well as tokens. The
  agent proposes a link to wherever the scout stands when it wants one, and
  the scout then comes home.
- **reach**: a jump whose steps cost nothing. The walk is as many steps long
  as the share of its tokens the agent's brain chooses would pay for. So
  wealth sets how far an agent can reach, without being spent.

Whether a searcher wants a link with the stranger it found is its ACCEPT
head, read like the link decision of a birth.

**B1 and B2:** off.

### link_steering

Which way each step of a search goes:

- **random**: to a neighbour of where the search stands, drawn at random;
- **looking**: the searcher's brain looks at every neighbour of where it
  stands. It steps to the one it would most like a link with, by its ACCEPT
  head, and draws ties at random.

It needs `link_search`.

**B1 and B2:** random.

### link_consent

Who has to agree to a link a search proposes:

- **mutual**: the stranger's ACCEPT head has to say yes as well;
- **forced**: the searcher's yes is enough.

Either way, no link is made to or from an agent at its
[`valence`](#valence). It needs `link_search`.

**B1 and B2:** mutual.

## The pilots' rules

### proto

Rules that pilot worlds played before the engine had them
([Pilot worlds](pilot-worlds.md)). On 19 October 2026 they became a setting,
so that the lab can play them where they can be seen. Each is switched on by
name. The engine's own world refuses them: the lab hands a run that names one
to the pilots' own code, frozen beside the engine (`gol_pilot.py`). Such a
run keeps no [frames](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections."). Before 19 October 2026 no run of the notebook's
experiments named any; Experiments 22 and 24 were the first.

- **came_from**: one input row for each candidate. Is this the node my brain
  came from — the node whose holder took mine, or my parent if I was born
  here?
- **helped_by**: one input row for each candidate: what its stake on my node
  did in the last game. It is the chance I kept my node with every stake,
  minus the chance without its stake.
- **contest = lottery**: a node goes to a staker drawn in proportion to its
  stake as the game weighs it, instead of to the largest. This is Tullock's
  contest ([Another game](another-game.md)).
- **pool = genotype** or **family**: the stakes on a node of agents with one
  brain (or one family tag) count as one side's. The side with most takes the
  node, and the owner keeps it if its side wins. It adds an input row: is this
  candidate on my side? `family` needs a `tag_rate`.
- **kin_sense**: the same input row without the pooling. Is this candidate's
  brain mine (with a `tag_rate`, my family)?
- **tag_rate**: a family tag carried with every brain. A copy keeps its
  parent's tag, and a brain that mutates takes a new one with this
  probability. So a family is all the genotypes descended from one tag.
- **mate_pays**: under [`mix_at_birth`](#mix_at_birth), the mate pays the
  child as many tokens as the parent gave it, as far as it can and never its
  last. It pays once every birth of the phase is made.
- **grow**: a brain of one hidden layer of `float16` weights, of which only
  the first units take part. A mutation may switch the next one on or the
  last one off.
- **unit_token**: under `grow`, every unit that takes part holds a token,
  paid by the node it is in.
- **link_token**: every link holds a token. Its maker pays for it, and a link
  that lapses gives its token back to one of its ends. It needs
  [`valence`](#valence) 0: the tokens are the limit.

**B1 and B2:** none.

## How brains change

The three numbers of mutation ([How a brain changes](mutation.md)).

### mutation_probability

The probability *p* that a brain changes. It is applied twice: once to a
newborn's copy of its parent's brain, and once to every brain in the world
after every game. Under [`mutate_on_replication`](#mutate_on_replication), the
second is applied instead to every copy of a brain into a node its owner has
won.

**B1 and B2:** 0.2.

### mutation_noise_std

How large a change to one number is: a normal draw with standard deviation
σ/√(fan-in). Here σ is this setting, and the fan-in is the number of inputs
of the layer the number belongs to. So a layer that adds up many inputs gets
smaller changes to each of them. In B1's first hidden layer, with its 154
inputs, a change has a standard deviation of 0.2/√154 ≈ 0.016.

**B1 and B2:** σ = 0.2.

### mutation_sparsity

The share *s* of a brain's numbers that a change touches: each number is
changed with probability *s*. In B1 that is about 1,580 of its 15,795
numbers. The same *s* is also the probability of a rarer **reset**, which
redraws a share *s* of a weight matrix from scratch. Each weight matrix and each bias
vector is treated separately. With *s* = 0, a "change" changes nothing, but
the brain still gets a new genotype number:
[Do the brains matter?](do-the-brains-matter.md) uses exactly this.

**B1 and B2:** 0.1.

## When a run stops

### extinction_threshold

A run stops, counted as **extinct**, when an iteration ends — after its game
— with this many agents or fewer ([When a world ends](extinction.md)).

**B1 and B2:** 20.

## What a run keeps

A setting that changes what a run keeps, never what happens in it.

### archive_every

Every how many iterations a run keeps a whole world, in `archive/` beside its
frames: its network, tokens, ages and every brain. The value 0 keeps none. The
rolling checkpoint is overwritten, and serves only for resuming. These worlds
are kept, for anything that needs the brains of more than one moment. Above
all, that is the [cross-time tournament](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?") of
[How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md).
A plan asks for it in its `record` block.

**B1 and B2:** 0; a plan sets its own.

## Where to go next

- [B1 and B2: the two baselines](b1-and-b2.md): why each baseline has the
  values it has.
- [The starting ring](starting-ring.md): what the first four settings build.
- [One iteration, step by step](one-iteration.md): where in an iteration each
  rule acts.
- [The lab, the engine and pilots](the-lab.md): how an experiment's plan sets
  these for its runs.

## How our view changed

- **How many neighbours the founders of large worlds start with.** Until
  14 October 2026, the entry for `k_neighbors` said that *k* stays 5, wired
  as 4, below 10,000 founders (*T* < 1,000,000). That was a slip of
  arithmetic: *k* = max(⌊*n*/100⌋, 5) is 5 only below 600 founders. The 1,000
  founders of *T* = 100,000 start with 10 neighbours. A review of the
  whole notebook corrected it on 14 October 2026.
- **What mending prevents.** From 12 to 23 October 2026, the entry for
  `mend_on_death` said that with mending on, the cleanup never removes a piece
  cut off from the world. It never does so *because of a death*. Cutting
  unused connections can still split a world unless `prune_needs` is at least
  1, as [A world that cannot break](a-world-that-cannot-break.md) shows. The
  entry was corrected when this page was rewritten, on 23 October 2026.

<!-- turns -->
---

← [Is a run reproducible?](is-a-run-reproducible.md) · [How the simulation works](../README.md#how-the-simulation-works) · [The starting ring](starting-ring.md) →

> [!info]- Pages that link here (50)
> [Glossary](glossary.md) · [The world: tokens, agents and a network](the-world.md) · [The brain](the-brain.md) · [Questioning the mechanics](questioning-the-mechanics.md) · [Is a run reproducible?](is-a-run-reproducible.md) · [The starting ring](starting-ring.md) · [The share function f(a, b)](share-function.md) · [A yes-or-no decision](binary-decision.md) · [Splitting tokens into whole numbers](largest-remainder.md) · [What a brain sees](brain-inputs.md) · [What a brain says](brain-outputs.md) · [Messages](messages.md) · [How a coalition takes a node](revolution.md) · [How a brain changes](mutation.md) · [When a world ends](extinction.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [Building B2: a world that cannot break](a-world-that-cannot-break.md) · [The estate of the dead](estate.md) · [What crosses a connection](what-crosses-a-connection.md) · [Token flow](token-flow.md) · [How a world breaks](how-a-world-breaks.md) · [Stars and hubs](stars-and-hubs.md) · [The random twin: is a world a space?](random-twin.md) · [What folds a world](what-folds-a-world.md) · [What holds a world's size](what-holds-a-worlds-size.md) · [How alike the agents are](how-alike-the-agents-are.md) · [Do the brains matter?](do-the-brains-matter.md) · [How different are the agents?](how-different-are-the-agents.md) · [Eras and contests: does a world keep getting better?](eras-and-contests.md) · [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md) · [How fast should brains change?](how-fast-should-brains-change.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Raiders that breed](raiders-that-breed.md) · [Hand-written strategies in B1](hand-written-strategies-in-b1.md) · [Who inherits?](who-inherits.md) · [Links to strangers](links-to-strangers.md) · [Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md) · [Eight switches at once](eight-switches-at-once.md) · [Revolutions, with every other rule](revolutions-with-every-other-rule.md) · [Handover, with every other rule](handover-with-every-other-rule.md) · [Gifts, with every other rule](gifts-with-every-other-rule.md) · [Healthy and rich](healthy-and-rich.md) · [Brains mixed at birth, in long worlds](mixed-at-birth-ten-long-worlds.md) · [Who may be a mate](mixed-at-birth-with-parents-as-mates.md) · [Room and riches for the steadiest world](room-for-the-steadiest-world.md) · [The lab, the engine and pilots](the-lab.md) · [Every simulation: what was run, and what was measured](every-simulation.md) · [What a run records](frames-and-stats.md) · [Pilot worlds](pilot-worlds.md)
<!-- /turns -->
