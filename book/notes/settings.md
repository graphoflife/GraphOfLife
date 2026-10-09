# Every setting

A run of Graph of Life is fixed by two things: its **settings** and its
**seed** ([Seeds and the random stream](random-numbers.md)). This note lists
every setting, says what it does, and gives its value in the **baseline B1**
— the settings every experiment in this book starts from, and changes one at
a time. The plan that defines B1 is the file `book/experiments/B1.json`.

| setting | B1 | in one line |
|---|---|---|
| [`total_tokens`](#total_tokens) | 10,000 (chosen per experiment) | tokens in the world |
| [`n_nodes`](#n_nodes) | 0 → 100 founders | how many founders |
| [`k_neighbors`](#k_neighbors) | 0 → 5, wired as 4 | neighbours in the starting ring |
| [`rewire_p`](#rewire_p) | 0.2 | shortcuts in the starting ring |
| [`hidden_layers`](#hidden_layers) | 50, 45, 40, 35, 30 | the brain's hidden layers |
| [`brain_kind`](#brain_kind) | float16 | how weights are stored |
| [`brain_bits`](#brain_bits) | 16 (unused) | only for binary brains |
| [`message_amount`](#message_amount) | 30 | numbers per message |
| [`random_input_amount`](#random_input_amount) | 5 | random inputs per look |
| [`exchange_messages`](#exchange_messages) | on | messages at all |
| [`message_prepass`](#message_prepass) | on | a look that only talks |
| [`allow_handover`](#allow_handover) | on | parents hand connections to children |
| [`allow_revolutions`](#allow_revolutions) | on | coalitions can take nodes |
| [`allow_gifting`](#allow_gifting) | off | gifts of tokens |
| [`random_decisions`](#random_decisions) | off | the control without brains |
| [`prune_after`](#prune_after) | blotto | when unused connections are cut |
| [`inactive_window`](#inactive_window) | phase | how long a connection may be unused |
| [`redistribution`](#redistribution) | uniform | who gets the tokens of the dead |
| [`tokens_created_per_phase`](#tokens_created_per_phase) | 0 | new tokens per cleanup |
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
which it never is in this book), so the sum of all agents' tokens is *T*
after every phase of every iteration. Part II uses *T* = 10,000;
[Chapter 32](../chapters/32-how-does-a-worlds-size-follow-its-tokens.md)
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
starts with **4** neighbours. Below 10,000 founders (*T* < 1,000,000) this is
always 5, wired as 4.

### rewire_p

In the starting ring, the probability with which each connection is moved
to a founder chosen at random — the "small-world" construction of Watts and
Strogatz (1998). Baseline 0.2. See [The starting ring](starting-ring.md).

## The brain

### hidden_layers

The widths of the brain's hidden layers, from input to output. The baseline
brain has five: 50, 45, 40, 35 and 30 neurons. With the 154 inputs and 45
outputs of the baseline that gives 15,550 weights and 245 biases, 15,795
numbers in all ([Chapter 4](../chapters/04-the-brain.md)).

### brain_kind

How a brain's numbers are stored. `float` keeps them as 64-bit floating-point
numbers, `float16` as 16-bit ones (about three significant decimal digits),
`binary` as −1, 0 or +1. The baseline uses `float16`: the weights are
*stored* in 16 bits, which takes a quarter of the memory, but every
calculation is done in 64 bits. Every result in this book is for `float16`.

### brain_bits

Used only by `binary` brains, to say how many input rows encode one number.
Float and float16 brains ignore it.

### message_amount

How many numbers one message holds, *m*. Baseline 30. Every agent writes one
message to itself and one to each neighbour in every phase, and reads four
messages for each neighbour it looks at, so a brain has 4*m* = 120 message
inputs and *m* = 30 message outputs. See [Messages](messages.md).

### random_input_amount

How many random numbers a brain reads for each candidate it looks at. Each
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
every run of this book; with it off, the brain has neither the gift inputs
nor the gift outputs.

### random_decisions

The control without brains. When on, every number a brain would produce is
replaced by an independent draw from the standard normal distribution
(mean 0, standard deviation 1), and every decision is taken from those
numbers by exactly the same rules. The brains still exist, are still copied,
and still mutate — they are just never asked. Used in
[Chapter 31](../chapters/31-do-the-brains-matter.md).

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
out by one multinomial draw with equal probabilities. `by_tokens` would weight
each survivor by what it already holds. Baseline: `uniform`.

### tokens_created_per_phase

Tokens added to the pool at every cleanup and dealt out like the tokens of
the dead. Baseline 0: the supply is fixed.

## Change

### mutation_probability

The probability *p* that a brain changes. It is applied twice: once to a
newborn's copy of its parent's brain, and once to every brain in the world
after every game. Baseline 0.2. See [How a brain changes](mutation.md).

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
([Chapter 31](../chapters/31-do-the-brains-matter.md) uses exactly this).

### extinction_threshold

A run stops, counted as **extinct**, when an iteration ends — after its
game — with this many agents or fewer. Baseline 20. See
[When a world ends](extinction.md).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
