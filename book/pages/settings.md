# Every setting

A run of Graph of Life is fixed by two things: its **settings** and its
**seed** ([Seeds and the random stream](random-numbers.md)). This note lists
every setting, says what it does, and gives its value in the **[baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")**
— the settings every experiment in this notebook starts from, and changes one at
a time. The plan that defines B1 is the file `book/experiments/B1.json`.

| setting | B1 | in one line |
|---|---|---|
| [`total_tokens`](#total_tokens) | 10,000 (chosen per experiment) | [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") in the world |
| [`n_nodes`](#n_nodes) | 0 → 100 founders | how many founders |
| [`k_neighbors`](#k_neighbors) | 0 → 5, wired as 4 | [neighbours](the-world.md#the-network "An agent joined to another by a connection.") in the starting ring |
| [`rewire_p`](#rewire_p) | 0.2 | shortcuts in the starting ring |
| [`hidden_layers`](#hidden_layers) | 50, 45, 40, 35, 30 | the brain's hidden layers |
| [`brain_kind`](#brain_kind) | float16 | how weights are stored |
| [`brain_bits`](#brain_bits) | 16 (unused) | only for binary brains |
| [`message_amount`](#message_amount) | 30 | numbers per message |
| [`random_input_amount`](#random_input_amount) | 5 | random inputs per look |
| [`exchange_messages`](#exchange_messages) | on | messages at all |
| [`message_prepass`](#message_prepass) | on | a look that only talks |
| [`allow_handover`](#allow_handover) | on | parents hand [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") to children |
| [`allow_revolutions`](#allow_revolutions) | on | coalitions can take [nodes](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") |
| [`allow_gifting`](#allow_gifting) | off | gifts of tokens |
| [`random_decisions`](#random_decisions) | off | the control without brains |
| [`prune_after`](#prune_after) | blotto | when unused connections are cut |
| [`inactive_window`](#inactive_window) | phase | how long a connection may be unused |
| [`redistribution`](#redistribution) | uniform | who gets the tokens of the dead |
| [`tokens_created_per_phase`](#tokens_created_per_phase) | 0 | new tokens per [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.") |
| [`mutate_on_replication`](#mutate_on_replication) | off | brains change only when copied |
| [`incumbency`](#incumbency) | 1 | home advantage in the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") |
| [`birth_links`](#birth_links) | 0 | links a newborn is born with |
| [`mend_on_death`](#mend_on_death) | off | a death never splits a neighbourhood |
| [`prune_needs`](#prune_needs) | 0 | only redundant connections lapse |
| [`valence`](#valence) | 0 | most connections a brain may choose |
| [`keep_home`](#keep_home) | 0 | a share of the tokens stays at home |
| [`max_age`](#max_age) | 0 | the longest life |
| [`min_child`](#min_child) | 1 | the smallest child |
| [`mix_at_birth`](#mix_at_birth) | off | a newborn's brain mixed from two |
| [`link_search`](#link_search) | off | links to strangers, found by a walk |
| [`link_steering`](#link_steering) | random | which way a search steps |
| [`link_consent`](#link_consent) | mutual | who must agree to a link found |
| [`mutation_probability`](#mutation_probability) | 0.2 | how often a brain changes |
| [`mutation_noise_std`](#mutation_noise_std) | 0.2 | how big a change is |
| [`mutation_sparsity`](#mutation_sparsity) | 0.1 | how many weights a change touches |
| [`extinction_threshold`](#extinction_threshold) | 20 | when a world counts as dead |

Below, *T* is the number of tokens, *n* the number of founders and *k* the
number of neighbours in the starting ring.

## The world

### total_tokens

The number of tokens in the world, *T*. Tokens are never made or destroyed
(unless [`tokens_created_per_phase`](#tokens_created_per_phase) is above 0,
which it never is in this notebook), so the sum of all agents' tokens is *T*
after every phase of every [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). The pages on how the simulation works teach with *T* = 10,000,
the study of the baseline uses *T* = 100,000, and
[How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)
varies it from 800 to 409,600. Since every living agent holds at least one
token, a world can never hold more than *T* agents.

### n_nodes

How many **founders** the world starts with. The value 0 means one founder
per hundred tokens:

$$
n = \left\lfloor \frac{T}{100} \right\rfloor ,
$$

where ⌊*x*⌋ is *x* rounded down. With *T* = 10,000 that is 100 founders.
Each founder gets ⌊*T*/*n*⌋ tokens; what the division leaves over is handed
out by a uniform multinomial draw ([The starting ring](starting-ring.md)).

### k_neighbors

How many neighbours each founder starts with in the ring. The value 0 means

$$
k = \max\!\left( \left\lfloor \frac{n}{100} \right\rfloor ,\ 5 \right).
$$

The ring joins every founder to its ⌊*k*/2⌋ nearest founders on either side,
so an odd *k* is wired as *k* − 1: in the baseline *k* = 5 and every founder
starts with **4** neighbours. Below 600 founders (*T* < 60,000) this is
always 5, wired as 4; with the 1,000 founders of *T* = 100,000 it is 10,
five on either side.

### rewire_p

In the starting ring, the probability with which each connection is moved
to a founder chosen at random — the "small-world" construction of Watts and
Strogatz (1998). Baseline 0.2. See [The starting ring](starting-ring.md).

## The brain

### hidden_layers

The widths of the brain's hidden layers, from input to output. The baseline
brain has five: 50, 45, 40, 35 and 30 neurons. With the 154 inputs and 45
outputs of the baseline that gives 15,550 weights and 245 biases, 15,795
numbers in all ([The brain](the-brain.md)).

### brain_kind

How a brain's numbers are stored. `float` keeps them as 64-bit floating-point
numbers, `float16` as 16-bit ones (about three significant decimal digits),
`binary` as −1, 0 or +1. The baseline uses `float16`: the weights are
*stored* in 16 bits, which takes a quarter of the memory, but every
calculation is done in 64 bits. Every result in this notebook is for `float16`.

### brain_bits

Used only by `binary` brains, to say how many input rows encode one number.
Float and float16 brains ignore it.

### message_amount

How many numbers one message holds, *m*. Baseline 30. Every agent writes one
message to itself and one to each neighbour in every phase, and reads four
messages for each neighbour it looks at, so a brain has 4*m* = 120 message
inputs and *m* = 30 message outputs. See [Messages](messages.md).

### random_input_amount

How many random numbers a brain reads for each [candidate](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides.") it looks at. Each
is drawn uniformly from −2 to 2, fresh every time. Baseline 5. They let a
brain behave differently in situations that look the same; see
[What a brain sees](brain-inputs.md).

## The mechanics

### exchange_messages

Whether agents write and read messages at all. Baseline: on. Off, every
message input is 0.

### message_prepass

Whether each phase begins with an extra look in which every agent only
writes messages. Baseline: on. Then the look that acts reads messages that
were written in this phase, about the world as it now is, rather than in the
phase before. See [Messages](messages.md).

### allow_handover

Whether a parent may move some of its own connections to its newborn
child. Baseline: on. Off, a child is only joined to the candidates chosen
for it, and the parent keeps all its connections.

### allow_revolutions

Whether a coalition of smaller stakers can take a node from its largest
staker. Baseline: on. Off, every node goes to whoever staked the most on it.
See [How a coalition takes a node](revolution.md).

### allow_gifting

Whether agents may give tokens to neighbours during reproduction. Off in
every run of this notebook; with it off, the brain has neither the gift inputs
nor the gift outputs.

### random_decisions

The control without brains. When on, every number a brain would produce is
replaced by an independent draw from the standard normal distribution
(mean 0, standard deviation 1), and every decision is taken from those
numbers by exactly the same rules. The brains still exist, are still copied,
and still mutate — they are just never asked. Used in
[Do the brains matter?](do-the-brains-matter.md).

### prune_after

After which phase connections that carried no tokens are cut: `blotto` (the
game), `reproduction`, or `both`. Baseline: `blotto`. A connection "carries
tokens" in the game when one of its two agents stakes at least one token on
the other.

### inactive_window

How long a connection may go unused before it is cut. `phase` means it must
carry tokens in the phase being judged (or have been made in it);
`iteration` also counts the phase before. Baseline: `phase`. With
`prune_after` = `blotto` this means: **every connection that no tokens
crossed in this game is cut at the end of the game** — including the ones a
newborn got in the reproduction phase just before.

### redistribution

Who gets the tokens of agents removed by the cleanup. `uniform` gives every
surviving agent the same chance at every token: the pooled tokens are dealt
out by one multinomial draw with equal probabilities. `by_tokens` weights
each survivor by what it already holds. `poorest` raises the poorest
survivors to one common level, as high as the pool reaches, and gives the
whole tokens left over one each to agents drawn at that level — water poured
on an uneven floor fills the lowest places first. `newborns` deals the pool
out evenly over the children born this iteration that are still alive;
`parents` over their parents, once per child, which pays for having
children. In an iteration without a birth these two deal it out as
`uniform` does. `children` leaves each dead agent's [estate](estate.md "The tokens a dead agent leaves, and who gets them.") to its own living
children, evenly, and deals an estate that finds no living child out as
`uniform` does: the estate stays in the family. Baseline: `uniform`.

### tokens_created_per_phase

Tokens added to the pool at every cleanup and dealt out like the tokens of
the dead. Baseline 0: the supply is fixed.

## Open-ended evolution

Nine rules added for [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau."), each the answer to one way a world was seen to
fail once its brains could see, and a tenth, `mix_at_birth`, that came later
from the pilots of [Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md).
Every one is off in the baseline B1 and in every run made before [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death."), so the
baseline's runs are unchanged by them. All are local: they read an agent,
its connections and its neighbours, and nothing further away.

### mutate_on_replication

Whether a brain changes only when it is copied — into a newborn, or into a
node its owner has just won — rather than also after every game, wherever it
sits. Baseline off: every brain in the world may change after every game,
which wears an adaptation away within a few dozen iterations of its making.

### incumbency

How many times over an agent's stake on its own node counts when the game
decides who takes the node, *h*. Baseline 1: the old rule, where a defender
needs to outstake every attacker. With *h* = 1.5 an attacker must stake more
than one and a half times what the defender keeps at home. Only who wins is
weighed; the tokens the node ends with are still what was staked on it.

### birth_links

How many of its parent's neighbours a newborn is joined to, drawn at random,
besides its parent, before the links its brain chooses. Baseline 0: the
newborn has only the links its brain chooses and is often a leaf. With 2,
every birth closes triangles. Drawn at random rather than by the brain's
liking, because a liking for well-connected agents is [preferential
attachment](preferential-attachment.md "The well connected gaining new connections faster than the poorly connected."), and makes [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.").

### mend_on_death

Whether the neighbours of an agent that dies are joined up again. They fall
into groups — neighbours linked directly to one another are one group — and
each group is joined to the next through its least-connected member, the
last back to the first when there are three or more: the neighbours close in
a ring round the gap, as the edge of a hole closes. A neighbourhood that was
one piece gains nothing. Baseline off. With it on, no
death can split the world, so the cull of the cut-off never fires.

### prune_needs

How many neighbours the two ends of an unused connection must have in common
for it to be cut. Baseline 0: every unused connection is cut, [bridges](bridges.md "A connection that lies on no loop: cut it and the network falls into two pieces.") as
readily as anything else. With 1 the ends always stay joined through a third
agent; with 2 they stay joined twice over, so pruning cannot leave anybody
hanging on a single agent either.

### valence

The most connections an agent's brain may choose to hold. Baseline 0: no
limit. The links a newborn is born with (`birth_links`) and those made by
mending do not count against it. An agent at its limit cannot have a child —
it is crowded — which is what keeps a dense region from growing denser.

### keep_home

The share of an agent's tokens that stays on its own node whatever its brain
says — rounded up, and never less than one token — while a parent always
keeps at least one token too. Baseline 0. Above 0, no agent can stake or
breed itself to death: a world then loses agents only to age (`max_age`) and
to being cut off. A tiny share is a floor of one token; a larger one also
keeps agents from staking nearly everything on a single neighbour.

### max_age

The age at which an agent dies, in iterations; 0 means never. Its tokens are
shared out like those of any other dead agent. Baseline 0.

### buy_life

Whether an agent that has reached `max_age` may live on by buying its life:
one token an iteration, paid at the first cleanup of the iteration into the
estate of the dead and shared out with it. An agent that would have to give
its last token dies as before. A rich agent can live long, a poor one cannot;
in a world whose agents are rich, fewer die each iteration, so fewer births
replace them. Needs `max_age`. Baseline off.

### min_child

The fewest tokens a child can be born with. A parent whose brain would give a
child less has no child. Baseline 1: any child at all.

### child_floor

The least a child may start with. `fixed`, the old rule: [`min_child`](#min_child)
alone. `neighbours`: also at least the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") holding of its parent's
neighbours, at the moment of the birth; a parent that would give less has no
child. Where births are cheap — the estate of the dead paid to the newborns,
say — a world fills with children, each with a token or two, until almost
every agent holds the same few
([Room for the steadiest world](room-for-the-steadiest-world.md)). Under
`neighbours` a child is never poorer than the agents it is born among, so a
world can grow only as rich as it is. It reads the parent's neighbourhood and
nothing further, and has no number of its own. Baseline `fixed`.

### mix_at_birth

Whether a newborn's brain is mixed from two before it mutates: its parent's
and a mate's, one of the parent's neighbours drawn at random. With one hidden
layer each hidden unit comes whole — the weights into it, its bias and the
weights out of it — from one brain or the other by an even chance, and so does
each output's bias; with more layers, each unit of every layer, with its
weights in and its bias. A unit that works can then spread without the rest
of the brain it arose in. The mixture is a new [genotype](genotype.md "The identity of a brain: a number that a brain keeps until it changes, and that records whose copy it was.") unless it is the
parent's brain again, as it is when the mate is of the parent's genotype; a
parent without a neighbour has a copy. Baseline off: a newborn has a copy of
its parent's brain. In the pilots of
[Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md) and the batches
after it, worlds with it came closest to a healthy and diverse world
([Healthy and rich](healthy-and-rich.md)).

### mix_mates

Under `mix_at_birth`, who may be the mate. `any`: any neighbour of the parent,
drawn at random. `parents`: only a neighbour with a living child of its own —
and a parent counts as one from the birth of its first child that lives. A
mate passes half of its brain on and pays nothing for the child; with `any`,
an agent that never breeds spreads its brain through other agents' children
all the same, and with `parents` it cannot. A parent without such a neighbour
gives its newborn a copy of its own brain. Baseline `any`.

### self_message

What an agent writes to itself, which every neighbour reads as one of the
four messages in its column. `own`: what the agent's own column says, the
old rule. `largest`: the largest, number by number, of what it writes to all
its candidates, itself included. A brain scores each candidate from that
candidate's column alone, so under `own` no message can carry news of a
third agent: whether a stranger is next door reaches a teammate only through
what the agent does
([What a brain can compute](what-a-brain-can-compute.md)). Under `largest` a
mark one column makes ("this neighbour is a stranger") reaches every
neighbour a phase later, and over a few phases a team can hold a field such
as each member's nearness to its frontier. It adds no weights and no inputs,
but an agent's message to itself is then at least what it tells any
neighbour. Baseline `own`.

## Links to strangers

Every link an agent makes otherwise is to its own neighbourhood: a newborn's,
to its parent's neighbours. These three let an agent reach further, at a
price, and are off in every run of the notebook until they are turned on.

### link_search

Whether an agent may search beyond its neighbours for someone to link with.
Every step of a jump or a walk costs the searcher one token, paid to the agent
the step enters, so a search moves tokens but never makes or destroys any, and
no agent spends its last token on one; a reach is free. How much an agent spends is its brain's
choice, a share of what reproduction left it (the SEEK head, read like a
gift's). Baseline off.

- **jump**: in one iteration, a walk as many steps long as the tokens spent,
  from the agent's own node; a link is proposed to the agent it ends on, if
  that is a stranger and the searcher wants it.
- **walk**: a scout that goes one step further in every iteration its agent
  pays for, on from where it stood, so that distance costs time as well as
  tokens; the agent proposes a link to wherever the scout stands when it wants
  one, and the scout then comes home.
- **reach**: a jump whose steps cost nothing. The walk is as many steps long
  as the share of its tokens the agent's brain chooses would pay for, so
  wealth sets how far an agent can reach, without being spent.

Whether a searcher wants a link with the stranger it found is its ACCEPT head,
read like the link decision of a birth.

### link_steering

Which way each step of a search goes. Baseline random: to a neighbour of
where the search stands, drawn at random. With **looking**, the searcher's
brain looks at every neighbour of where it stands and steps to the one it
would most like a link with (its ACCEPT head), ties drawn at random.

### link_consent

Who has to agree to a link a search proposes. Baseline mutual: the stranger's
ACCEPT head has to say yes as well. With **forced**, the searcher's yes is
enough. Either way no link is made to or from an agent at its
[`valence`](#valence).

## The pilots' rules

### proto

Rules the pilot worlds played before the [engine](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") had them
([pilot worlds](pilot-worlds.md)), carried into the engine so that the lab can
play them where they can be seen. Each is switched on by name; a run that names
none — every run of the notebook's experiments until October 2026 — plays none.

- **came_from** — one input row for each candidate: is this the node my brain
  came from, the node whose holder took mine, or my parent if I was born here?
- **helped_by** — one input row for each candidate: what its stake on my node
  did in the last game, the chance I kept my node with every stake minus the
  chance without its stake.
- **contest = lottery** — a node goes to a staker drawn in proportion to its
  stake as the game weighs it (Tullock's contest), instead of to the largest.
- **pool = genotype** or **family** — the stakes on a node of agents with one
  brain (or one family tag) count as one side's; the side with most takes the
  node, and the owner keeps it if its side wins. Adds an input row: is this
  candidate on my side?
- **kin_sense** — the same input row without the pooling: is this candidate's
  brain mine (with a tag_rate, my family)?
- **tag_rate** — a family tag carried with every brain: a copy keeps its
  parent's tag, and a brain that mutates takes a new one with this
  probability, so a family is all the genotypes descended from one tag.
- **mate_pays** — under [`mix_at_birth`](#mix_at_birth), the mate pays the
  child as many tokens as the parent gave it, as far as it can and never its
  last, once every birth of the phase is made.
- **grow** — a brain of one hidden layer of which only the first units take
  part; a mutation may switch the next one on or the last one off.
- **unit_token** — under grow, every unit that takes part holds a token, paid
  by the node it is in.
- **link_token** — every link holds a token: its maker pays for it, and a link
  that lapses gives its token back to one of its ends. Needs
  [`valence`](#valence) 0: the tokens are the limit.

## Change

### mutation_probability

The probability *p* that a brain changes. It is applied twice: once to a
newborn's copy of its parent's brain, and once to every brain in the world
after every game — or, under [`mutate_on_replication`](#mutate_on_replication),
once to every copy of a brain into a node its owner has won instead. Baseline
0.2. See [How a brain changes](mutation.md).

### mutation_noise_std

How large a change to one number is: a normal draw with standard deviation
σ/√(fan-in), where σ is this setting and the fan-in is the number of inputs
of the layer the number belongs to. Baseline σ = 0.2.

### mutation_sparsity

The share *s* of a brain's numbers that a change touches, and also the
probability of a rarer **reset**, which redraws a share *s* of a weight
matrix from scratch. Each weight matrix and each bias vector is treated
separately. Baseline 0.1. With *s* = 0 a "change" changes nothing — but the
brain is still given a new genotype number
([Do the brains matter?](do-the-brains-matter.md) uses exactly this).

### extinction_threshold

A run stops, counted as **extinct**, when an iteration ends — after its
game — with this many agents or fewer. Baseline 20. See
[When a world ends](extinction.md).

## Recording

Settings that change what a run keeps, never what happens in it.

### archive_every

Every how many iterations a run keeps a whole world — its network, tokens,
ages and every brain — in `archive/` beside its [frames](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections."); 0 keeps none. The
rolling checkpoint is overwritten and is for resuming; these are kept, for
anything that needs the brains of more than one moment, above all the
[cross-time tournament](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?") of [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md).
A plan asks for it in its `record` block.

<!-- turns -->
---

← [Is a run reproducible?](is-a-run-reproducible.md) · [How the simulation works](../README.md#how-the-simulation-works) · [The starting ring](starting-ring.md) →

> [!info]- Pages that link here (51)
> [How to read this notebook](../README.md) · [What this notebook is about](what-this-notebook-is-about.md) · [Glossary](glossary.md) · [The world: tokens, agents and a network](the-world.md) · [The brain](the-brain.md) · [Questioning the mechanics](questioning-the-mechanics.md) · [The starting ring](starting-ring.md) · [What a brain sees](brain-inputs.md) · [Messages](messages.md) · [How a coalition takes a node](revolution.md) · [How a brain changes](mutation.md) · [When a world ends](extinction.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [Brains that see, and worlds that break](brains-that-see-and-worlds-that-break.md) · [A world that cannot break](a-world-that-cannot-break.md) · [Who dies, and why](who-dies-and-why.md) · [The estate of the dead](estate.md) · [What crosses a connection](what-crosses-a-connection.md) · [Token flow](token-flow.md) · [How a world breaks](how-a-world-breaks.md) · [Stars and hubs](stars-and-hubs.md) · [The random twin: is a world a space?](random-twin.md) · [What folds a world](what-folds-a-world.md) · [What holds a world's size](what-holds-a-worlds-size.md) · [Births as rich as the neighbours](births-as-rich-as-the-neighbours.md) · [The floor in plain B2 and in B1](the-floor-in-plain-b2-and-in-b1.md) · [How alike the agents are](how-alike-the-agents-are.md) · [Do the brains matter?](do-the-brains-matter.md) · [How different are the agents?](how-different-are-the-agents.md) · [What the measurements say about open-ended evolution](what-the-measurements-say-about-open-ended-evolution.md) · [Eras and contests: does a world keep getting better?](eras-and-contests.md) · [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md) · [How fast should brains change?](how-fast-should-brains-change.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Raiders that breed](raiders-that-breed.md) · [Hand-written strategies in B1](hand-written-strategies-in-b1.md) · [Who inherits?](who-inherits.md) · [Links to strangers](links-to-strangers.md) · [Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md) · [Eight switches at once](eight-switches-at-once.md) · [Revolutions, with every other rule](revolutions-with-every-other-rule.md) · [Handover, with every other rule](handover-with-every-other-rule.md) · [Gifts, with every other rule](gifts-with-every-other-rule.md) · [Healthy and rich](healthy-and-rich.md) · [Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md) · [Five mixed worlds, fifty thousand iterations](five-mixed-worlds-fifty-thousand-iterations.md) · [Mixed at birth, with parents as mates](mixed-at-birth-with-parents-as-mates.md) · [The lab, the engine and pilots](the-lab.md) · [Every simulation: what was run, and what was measured](every-simulation.md) · [What a run records](frames-and-stats.md) · [Pilot worlds](pilot-worlds.md)
<!-- /turns -->
