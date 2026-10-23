# Where the tokens flow

[Gains and losses](gains-and-losses.md) followed the agents through a game:
who gains, who loses. This page follows the **tokens**: where each staked
token goes, how much of it comes back, and whether the whole movement obeys
a simple law. It does — nearly — and the law explains several findings of
earlier pages at once.

> [!question] Questions of this page
> - Of all the tokens staked in a game, how many stay at home, how many cross
>   a connection and are matched by a stake coming back, and how many go one
>   way only?
> - How widely do agents spread their stakes, and how close is that to
>   splitting them evenly?
> - Can a game be described as a random walk of tokens? What would that
>   predict about who is rich?
> - Who keeps their node?
> - Do tokens go round in loops?

The worlds are the thirty of 100,000 tokens of [Thirty worlds](thirty-worlds.md);
where the thirty of 10,000 tokens this page was first written from differ,
both are given.

<!-- runs E10 -->
> [!info] The runs behind this page
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
> | [`allow_gifting`](settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
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

## The budget of a game

Exactly 100,000 tokens are staked in every game of these worlds — every
token, by whoever holds it. Each staked token lands in one of three places
([Token flow](token-flow.md)):

1. on the staker's **own node**;
2. on a neighbour's node, **matched** by a stake coming back: if A puts 5 on
   B's node and B puts 3 on A's, 3 tokens go each way and cancel — neither
   balance changes because of them;
3. on a neighbour's node, **one way**: the 2 left over in the example. Only
   these move tokens between agents.

<!-- figure flow/budget -->
![Where the tokens of a game go](../figures/flow/budget.svg)

**Where the tokens of a game go.** Every token is staked in every game. Yellow: the share staked by agents on their own node. Cyan: staked on a neighbour's node, but cancelled by what that neighbour staked back — if A puts 5 on B's node and B puts 3 on A's, 3 tokens each way cancel. Red: what is left after cancelling, flow with a direction ([Token flow](token-flow.md)). Left: the game of iteration 0, pooled over the 30 worlds; right: every game from iteration 500 on of the 30 worlds, pooled. Exactly 100,000 tokens are staked in each game.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](thirty-worlds.md)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take the rows with `phase` = 2: `tokens` (all staked), `totalFlow` (staked on others) and `netFlowShare` (the share of that flow left after cancelling).
> 2. Sum over the rows: at home = Σ `tokens` − Σ `totalFlow`; cancelled = Σ `totalFlow` × (1 − `netFlowShare`); one way = Σ `totalFlow` × `netFlowShare`.
> 3. Divide each by Σ `tokens`.
>
> **To make it again:** `python3 book_figures.py flow`.
<!-- /figure -->

In the first game, 13% stays home, 23% is matched and 64% goes one way — the
founders, with ten neighbours each, keep less at home than the founders of a
world of 10,000 tokens, with four (a quarter, a quarter and a half there). In
settled games, at both sizes, 30% stays home, **51% is matched** and only 20%
goes one way. Most of what agents send to their neighbours is an exchange of
equal amounts, which is why half of all agents end a game with exactly what
they had ([Gains and losses](gains-and-losses.md)).

## Stakes that are returned

<!-- figure flow/reciprocity -->
![Stakes that are returned](../figures/flow/reciprocity.svg)

**Stakes that are returned.** Of all pairs (A, B) in a game where agent A staked at least one token on neighbour B's node, the share in which B also staked at least one token on A's — every 25 iterations, in the 30 baseline worlds. The line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](thirty-worlds.md)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, read every 25 iterations by `book_figures.sample_pass` (frame `2·t` after the reproduction phase of iteration *t*, frame `2·t + 1` after its game; `gol_store.read_frame(run, index)`).
>
> 1. In the frame after the game of every 25th iteration, read `decisions.allocations`: for every agent A and every target B ≠ A with a stake > 0, note the pair (A, B).
> 2. Count the pairs whose reverse (B, A) is also there, divided by all pairs.
> 3. In each stretch, take the median, the 25th and 75th and the 5th and 95th percentiles of those means over the worlds that still have a value there (a world that died has none after its death).
>
> **To make it again:** `python3 book_figures.py flow`.
<!-- /figure -->

When A stakes on B, B stakes back on A in 90% of cases (the thirty worlds:
88% to 93%; 92% at 10,000 tokens). Is that reciprocity — agents answering a
stake with a stake? Not necessarily. Look at how widely agents stake (the
next figure): they stake on 81% to 96% of everyone they can. If every agent
picked its targets independently of who staked on it, B would still stake on
A with about that probability. A reciprocity of 90% is what breadth alone
would give. To find
real reciprocity one would have to show that B stakes *more* on A *because* A
staked on B — a question for [Do agents cooperate?](do-agents-cooperate.md).

## How widely agents stake

An agent with *d* connections has *d* + 1 **candidates**: its own node and its
neighbours' nodes. If it split its tokens evenly, it would put 1/(*d* + 1) of
them on each, home included.

<!-- figure flow/breadth -->
![How widely agents stake](../figures/flow/breadth.svg)

**How widely agents stake.** Agents at the start of a game, sorted by their number of connections. Blue: the share of their candidates (themselves and their neighbours) on which they staked at least one token. Yellow: the share of their tokens they staked on their own node. White dots: the home share an agent would have if it split its tokens evenly over all its candidates, 1/(d + 1), averaged over the class. Red dots: the same even split dealt out in whole tokens by the rules' own rounding ([Splitting tokens into whole numbers](largest-remainder.md)), which gives the tokens left over to the largest remainders, and, when all remainders tie, to the agent's own node first. From the games of every 100th iteration from 500 on, in the 30 worlds.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](thirty-worlds.md)). They are made with `python3 gol_lab.py run E10`.
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

Agents stake on almost all their candidates: 96% for agents with one
connection, falling to 81–82% for agents with ten or more. And their **home
share** follows the even split closely — a little above it:

| connections | 1 | 2 | 3–4 | 5–9 | 10–49 | 50+ |
|---|---|---|---|---|---|---|
| staked at home | 58% | 40% | 28% | 19% | 10% | 2.8% |
| even split, 1/(*d*+1) | 50% | 33% | 23% | 14% | 7% | 1.2% |
| even split, in whole tokens | 60% | 44% | 34% | 23% | 13% | 3.0% |

The extra at home is not the brains' doing. Tokens are whole, and an even
split rarely comes out even: an agent with 3 tokens and 2 candidates cannot
stake 1.5 on each. The rules deal the tokens left over to the candidates
with the largest remainders ([Splitting tokens into whole numbers](largest-remainder.md)),
and in an even split every remainder ties, so the leftovers go down the list
of candidates in order — and an agent's own node is always first. Dealt out
in whole tokens this way, an exact even split would put 60% at home for an
agent with one connection, 44% with two. The brains put a little *less* at
home than that.

Often there is not even a split to round. In the median world, 42% of agents
give every candidate a score of zero or below, and then the rules split
their tokens evenly by default ([What the brains are like](what-the-brains-are-like.md)).
And the brains that do give positive scores give nearly the same one to
every candidate, because a brain's outputs barely move with what it sees.
The evolved brains end up doing something close to the simplest thing
possible: spreading their tokens evenly over everyone they can reach.

## The game as a random walk

Suppose, then, that every agent split its tokens **exactly** evenly. Think of
a single token at node *u*: it moves to each of the *d*ᵤ + 1 candidates with
the same share. Repeated game after game, that is a **random walk** of tokens:
at every step a token stays where it is or moves to a neighbour, each with
probability 1/(*d*ᵤ + 1) — a "lazy" walk, because staying is one of the
choices.

How good is that description? For every agent at the start of a game, work
out the balance the even split would give it — stakes it would receive minus
tokens it holds — and set it against the balance it really ended with.

<!-- figure flow/random-walk -->
![The game as a random walk of tokens](../figures/flow/random-walk.svg)

**The game as a random walk of tokens.** For every agent at the start of a game: x is the balance it would end the stakes with — the tokens staked on its node minus the tokens it held — if every agent split its tokens evenly over itself and its neighbours; y is the balance it really ended them with. Agents are binned by x in steps of 2; the line is the mean of y in each bin and the band its middle half. The dashed line is y = x. Over all 1,798,806 agents (the games of every 250th iteration from 500 on, 30 worlds), the straight line fitted to the points has slope 0.65, and the even split accounts for 34% of the variance (R², [Fitting a straight line](least-squares.md)); over the 23 worlds that are not held together by a single hub ([The largest and the smallest](the-largest-and-the-smallest.md)), slope 0.94 and 45%.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](thirty-worlds.md)). They are made with `python3 gol_lab.py run E10`.
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

Over 1,798,806 agents of the thirty worlds, the real balances follow the
even split's with a slope of only 0.65, and the even split accounts for 34% of
the variance. But that is the stars speaking. Without the seven worlds held
together by one hub ([The largest and the smallest](the-largest-and-the-smallest.md)), the
slope is **0.94** and the even split accounts for **45%** — exactly as in the
worlds of 10,000 tokens (0.94 and 46%), and the median world, fitted on its
own, has 0.90. There the brains move tokens almost exactly as much as the
random walk would, in the same direction; the rest of the variance is what
they do differently, agent by agent.

The stars break the fit through a single agent each. The even split, worked
in fractions of a token, expects each of a star's thousands of leaves to send
the centre half its tokens. Dealt out in whole tokens, home first, a leaf
with one token sends nothing and one with three sends one: a star's leaves,
holding about three tokens, send a quarter of a token less each than the
fractions say. Over twenty thousand leaves that is some five thousand tokens
at one agent — a point so far out that it tilts a straight line fitted to two
million. The rounding of [Splitting tokens into whole numbers](largest-remainder.md)
is a small correction for an agent and a large one for a star.

A random walk on a network has a well-known resting state, and it can be
found in two lines. Write π(*u*) for the share of tokens at *u*. The walk is
at rest when, across every connection, as many tokens go one way as the
other:

$$
\pi(u) \cdot \frac{1}{d_u + 1} = \pi(v) \cdot \frac{1}{d_v + 1}
\quad\text{for every connection } \{u, v\} .
$$

That holds if π(*u*) is proportional to *d*ᵤ + 1. So:

> [!summary] The resting state of the even split
> On a fixed network, tokens moved by the even split settle to
> τᵤ = *T* · (*d*ᵤ + 1) / Σᵥ (*d*ᵥ + 1): **every agent holds tokens in
> proportion to its number of connections plus one.**

This is the law behind several earlier findings: that agents with more
connections hold more tokens ([Where do the tokens go?](where-do-the-tokens-go.md)),
that tokens run downhill from agents richer than their neighbourhood
([Gains and losses](gains-and-losses.md)), and that a rich agent with few
connections drains away within a few games. A world is never quite at rest,
because the network keeps changing under the tokens;
[How properties scale together](how-properties-scale-together.md) measures how close it
comes, and [The geometry of a world](the-geometry-of-a-world.md) how fast a random walk
on such a network can settle at all.

## Who keeps their node

The winner of a node — whose brain the node takes — is its largest staker,
unless a coalition of smaller stakers outweighs it
([How a coalition takes a node](revolution.md)). Under the even
split, the agent living on a node of degree *d* is one of *d* + 1 stakers of
roughly equal size there.

<!-- figure flow/kept -->
![Who keeps their node, by connections](../figures/flow/kept.svg)

**Who keeps their node, by connections.** Every node on which anyone staked in a game, sorted by its number of connections at the start of the game; the share that was won by the agent living on it — whether as the largest staker or through a coalition. From the games of every 25th iteration from 500 on, in the 30 worlds.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](thirty-worlds.md)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, read every 25 iterations by `book_figures.sample_pass` (frame `2·t` after the reproduction phase of iteration *t*, frame `2·t + 1` after its game; `gol_store.read_frame(run, index)`).
>
> 1. For every 25th iteration t from 500 on, count each node's connections in frame `2·t`.
> 2. In frame `2·t + 1`, read `decisions.winners`: for each node, whether `winner` = `node`.
> 3. Pool over runs and frames, and divide kept by all, per class.
>
> **To make it again:** `python3 book_figures.py flow`.
<!-- /figure -->

Agents with one connection keep their node in 74% of games; with two, 48%;
three or four, 27%; five to nine, 17%; ten to forty-nine, 9%; fifty or more,
**2.0%** — as at 10,000 tokens, within a point or two. The pattern is 1/(*d* + 1) again, raised a little by the rounding
that hands the leftover tokens to the agent's own node first: the agent on a
node is one of *d* + 1 contenders and wins about as often as one of them
would.

So a hub does not have an owner. The agent sitting on a node with a hundred
connections is replaced by a neighbour's brain in 98 games out of 100. **Hubs
are places, not individuals**: they hold many tokens because of where they
are, and the brains on them pass through. A brain cannot "become a hub"; it
can only be, for a game or two, the brain sitting on one.

## Do tokens go round?

Could the flow carry tokens around loops — from A to B to C and back to A —
like money circulating in an economy? [Lightning](lightning.md)
explains how this is measured, and why only bounds can be given: a greedy
search finds loops (a lower bound), and conservation sets a ceiling, since a
token that comes back to where it started changes no balance, and every agent
that ends with more or less than it sent cannot have had all its flow in
loops.

<!-- figure flow/circulation -->
![Do tokens go round?](../figures/flow/circulation.svg)

**Do tokens go round?** Left, on the flow as staked: the share of all tokens staked on neighbours that a greedy search could place in closed loops (blue; [Lightning](lightning.md)), and the share that conservation forbids from ever going round, because some agents receive more than they send (red) — so at most 1 − red can circulate. Right, after cancelling every stake against the one coming back: the share of the flow left (cyan), and of that, the share found in loops (green). Every 25 iterations, 30 worlds; the line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](thirty-worlds.md)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take `cyclingShare`, `flowImbalance`, `netFlowShare` and `netCyclingShare` of the rows with `phase` = 2 (measured every 25 iterations).
> 2. In each stretch, take the median, the 25th and 75th and the 5th and 95th percentiles of those means over the worlds that still have a value there (a world that died has none after its death).
>
> **To make it again:** `python3 book_figures.py flow`.
<!-- /figure -->

On the flow as staked, the budget at the start of this page already pins
the answer down. Two neighbours staking on each other form a loop of length
2: a token goes from A to B, and one comes back from B to A. So every matched
stake goes round, and matched stakes are 51% of all tokens — 72% of the
tokens staked on neighbours (51 of the 71). That is a floor: at least 72% of
the flow goes round. The ceiling comes from conservation, which forbids 23%
of the flow from ever going round (red), so at most 77% can. Between 72% and
77% of the flow as staked goes round, then, and nearly all of it in loops of
two: neighbours trading equal stakes back and forth. (The greedy search, the
blue line, hunts for long loops and places only 13% of the flow in the loops
it finds — a weak floor beside the 72% the matched stakes guarantee.)

After cancelling every stake against the one coming back (right panel), no
loop of two is left, and 28% of the flow remains — and of that, the search
finds only **0.3%** in loops. The ceiling is not much higher: the imbalance
does not change when stakes cancel, so at least 23.4/27.8 ≈ 84% of what is
left must flow one way. Between 0.3% and 16% of the net flow can go round;
the search finds almost nothing, as at 10,000 tokens.
[What crosses a connection](what-crosses-a-connection.md) looks at the flow connection by
connection.

There is a theorem that says what that means. Any flow on a network can be
split into two parts: one that runs **downhill**, from high to low along some
"potential" that every agent has, and one that **circulates** (the
Helmholtz–Hodge decomposition; for networks, Jiang, Lim, Yao and Ye 2011). A
downhill flow can never go round — round a loop, you would come back to the
height you started from. The net flow of a game is almost entirely of the
downhill kind: tokens move from hills to valleys, as heat does, and do not
circulate.

## What this means

- **The game is close to a random walk of tokens.** The evolved brains spread
  their tokens nearly evenly over everyone they can reach. The world's
  economy, in tokens, is an exchange of equal stakes plus a diffusion
  downhill. There is no circulation to speak of.
- **Wealth follows connections.** At rest, an agent holds tokens in proportion
  to its connections plus one. Being rich is a matter of being well connected
  — of being in a place.
- **Hubs are places.** The brains on hubs change almost every game. The
  network's most important positions are not held by anyone.
- **For open-ended evolution this is a warning.** A behaviour this close to
  "split evenly" leaves little for brains to differ in, and a world whose
  wealth and positions are set by the network rather than by the brains gives
  selection little to grip. [Questioning the mechanics](questioning-the-mechanics.md)
  asks which rules make it so, and [What the measurements say about open-ended evolution](what-the-measurements-say-about-open-ended-evolution.md) what could be
  changed.

> [!info] Added later: brains that see
> The even split is what blind brains do.
> [Brains that see, and worlds that break](brains-that-see-and-worlds-that-break.md) gives the brains
> one hidden layer instead of five, so that a change in what an agent sees
> reaches what it stakes (a founder's gain about 0.2 instead of 0.0006). Such
> brains stake on one neighbour or two, their unused connections lapse, and
> their worlds need the rules of [A world that cannot break](a-world-that-cannot-break.md)
> and [Who dies, and why](who-dies-and-why.md) to hold together.

To make every figure of this page: `python3 book_figures.py flow`.

<!-- turns -->
---

← [How the game is played](how-the-game-is-played.md) · [The game and the flow of tokens](../README.md#the-game-and-the-flow-of-tokens) · [What crosses a connection](what-crosses-a-connection.md) →
<!-- /turns -->
