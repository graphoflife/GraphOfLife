# Meta II · What the measurements say about open-ended evolution

Part III measured the baseline world in every way the viewer can, and a few
more — in this second edition on thirty worlds of 100,000 tokens, with two new
chapters on what crosses a connection ([Chapter 30](30-what-crosses-a-connection.md))
and on the largest and the smallest ([Chapter 33](33-the-largest-and-the-smallest.md)).
This chapter steps back and asks what all of it means for the question the
book exists for: **can open-ended evolution happen in Graph of Life, and
what would it take?** It is written the way a researcher keeps a notebook —
what we now know, what we think it means, what we propose to do about it, and
how we will know whether it worked — and, as in every chapter, so that a
student can follow each step.

> [!question] Questions of this chapter
> - What is the baseline world, now that it has been measured every way?
> - Where does it stand on the ladder of [Chapter 1](01-what-this-book-is-about.md)?
> - What stands in the way of open-ended evolution, and which changes to the
>   rules could remove it?
> - How will we measure progress — and know if we found it?

## What we are looking for

**Open-ended evolution** is evolution that does not run out of new things to
make. Life on Earth has it: four billion years in, it still produces new kinds
of organisms, new ways of living, new levels of organisation. Almost every
artificial system stops — it finds a way of doing well, fills the world with
it, and from then on only reshuffles ([Chapter 6](06-the-ideas-this-builds-on.md)).

[Chapter 1](01-what-this-book-is-about.md) set out a ladder of four claims,
each needed before the next:

1. **P1 — there is evolution:** lineages last long enough for selection to act
   on them, and selection changes the outcome.
2. **P2 — adaptation accumulates:** a late population would beat its own
   ancestors.
3. **P3 — there are organisations:** groups that persist while their members
   come and go.
4. **P4 — the space of organisations keeps growing**, without a ceiling.

The project's research plan (`research/Research.md`) adopts a definition aimed
at the top of the ladder: open-endedness as unbounded growth in the space of
**multi-agent organisations** — structures made of agents, which persist and
are reproduced. The reason is a counting argument: a brain has a fixed number
of weights, so the space of brains is a fixed box; but the number of ways to
arrange *n* agents grows faster than any exponential in *n*. If anything here
can grow without bound, it is how agents are arranged and what they do to each
other.

## The baseline world, in one paragraph

Here is what Part III found, put together. **Tokens** move almost as a random
walk would: agents spread their stakes nearly evenly over everyone they can
reach ([Chapter 29](29-where-the-tokens-flow.md)), so tokens settle locally in
proportion to connections plus one ([Chapter 34](34-how-properties-scale-together.md)),
run downhill from rich neighbourhoods to poor ([Chapter 28](28-gains-and-losses.md)),
and never circulate. Two connections in three carry one or two tokens a
game, most of them one each way: an equal exchange that keeps the connection
alive at its price of a token a game, and every connection that carries
nothing is cut ([Chapter 30](30-what-crosses-a-connection.md),
[Chapter 38](38-do-agents-cooperate.md)). The **network** is a stretched web
of stars, grown locally by children joining their parents' neighbourhoods,
four and a half times as long as a random network with the same
connections, with a thirty-third of its spectral gap
([Chapter 35](35-the-geometry-of-a-world.md));
it breaks along its poor regions, whose borders go quiet, and the global cull
kills whatever falls off ([Chapter 36](36-how-a-world-breaks.md)). Some worlds
condense around a single centre joined to thousands, a place handed from
parent to newborn child in **dynasties**
([Chapter 33](33-the-largest-and-the-smallest.md)). **Brains** spread by
conquest seventeen times more than by birth, and change mostly in place, not
when copied ([Chapter 39](39-questioning-the-mechanics.md)); the brain on a
hub is replaced almost every game, and the same two brains rarely face each
other twice ([Chapter 29](29-where-the-tokens-flow.md), [Chapter 38](38-do-agents-cooperate.md)).
Relatives live together, but cannot recognise each other, and do not treat each
other differently ([Chapter 38](38-do-agents-cooperate.md)). Almost every
structural statistic is the same in every world: it follows from the rules,
not from what brains compute.

The short version: **a world of places, not individuals.** The stage — the
network, the landscape of wealth — is stable and rule-made. The actors — the
brains — change almost every game.

## Where the world stands on the ladder

**P1, evolution: partly.** There are lineages, and they matter: a whole world
of 10,000 tokens descends from one branch of the family tree about every 350
iterations, a world of 100,000 about once in a run
([Chapter 18](18-genotypes-and-lineages.md)); families occupy regions that
widen as their line ages, about as a random walk spreads
([Chapter 37](37-one-world-many-colours.md), [Chapter 60](60-worlds-of-100000-tokens.md));
and worlds whose brains never change end up in very different places, while
worlds that evolve all end up alike ([Chapter 41](41-do-the-brains-matter.md)). But heredity is weak. A brain
is changed with probability 0.2 after every game whether or not it is copied,
and a brain is overwritten whenever its node is won. The project's pilot
measurements (`research/pilot_heredity.py`, at a higher mutation rate) found a
lineage 45% of the way to a stranger after one iteration and 92% after ten.
Selection can only work on what is inherited; here, little is inherited for
long.

**P2, accumulation: not yet asked.** The test is a tournament across time:
bring back the brains of iteration 500 and set them against those of iteration
2,500 in the same world (Cliff and Miller 1995). Nothing in Part III could
answer it.

> [!info] Added later: asked, and answered for the baseline
> [Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md) ran this
> tournament on ten worlds of 5,000 tokens (Experiment 12). By iteration 50
> every world's brains beat their founders — +361 Elo in the median world,
> +400 by iteration 100 — and then nothing more: the brains of iteration
> 1,000 held 0.483 of a world contested with those of iteration 300, every
> era tied with the one before it (0.47 to 0.52), and a dominance tournament
> crowned no more champions than chance. Adaptation accumulates for fifty to
> a hundred iterations and stops. [Chapter 50](50-why-evolution-stops.md)
> finds the same plateau in every pilot of Part V.

**P3, organisations: not found.** The candidates exist — family regions, hubs
with their stars — but nothing yet shows a group that persists while its
members change. The nearest thing is a star's **dynasty**: a centre joined to
thousands, passed from parent to newborn child for hundreds of iterations
while the brain on it changes every game
([Chapter 33](33-the-largest-and-the-smallest.md)). It persists while its
members come and go — but it is a place, kept by the rules of the even split,
not a group that does anything together. The same two brains face each other
one more game in only one case in twenty-five; positions are not held by any
brain.

> [!info] Added later: still not found
> None has been found since. In the pilots of Part V, connected groups of one
> genotype and of close kin shrank over time in every world.

**P4: out of reach** until P1 to P3 stand.

## What stands in the way

Part III, read together, points at six obstacles. Each is a rule, and each was
examined in [Chapter 39](39-questioning-the-mechanics.md).

1. **Variation in the living.** 97% of new genotypes are made in brains that
   are not being copied. Successful brains are eroded in place.
2. **Conquest erases.** A won node loses its brain entirely; an individual
   ends whenever its node is won, and on a hub that is almost every game.
3. **Nothing can be held.** No incumbency, no lasting partners: the "shadow of
   the future" that reciprocity needs is about 0.04 (0.05 at 10,000 tokens).
4. **Nothing is positive-sum.** Tokens are conserved and every stake is a
   transfer; no interaction makes anything that was not there before. An
   organisation can never be worth more than its parts, so there is no reason
   for one to form (`research/Research.md`, §3.3).
5. **Nobody can tell kin from stranger.** Kin selection has its precondition —
   relatives live together — but brains cannot see relatedness.
6. **The world can neither reach across nor split.** Growth is local only, so
   the network stretches and its shortcuts are never renewed; the global cull
   kills any part that separates, so populations can never diverge in
   isolation. [Chapter 42](42-how-does-a-worlds-size-follow-its-tokens.md)
   adds that a bigger world is no escape: distances grow like a power of its
   size, and the regions the cull removes grow with it.

> [!info] Added later: what Part V did about the six
> 1. Brains change only when copied in B2
>    ([Chapter 47](47-brains-that-see-and-worlds-that-break.md),
>    [Chapter 48](48-a-world-that-cannot-break.md)).
> 2. Conquest still erases.
> 3. A home stake counts 1.5 times ([Chapter 49](49-who-dies-and-why.md));
>    how long partners last has not been measured again.
> 4. A yield on stakes returned, and support stakes, were tried and gave
>    nothing that lasted ([Chapter 50](50-why-evolution-stops.md)).
> 5. Brains that see could tell kin by the names in their messages
>    ([Chapter 20](20-what-agents-say-to-each-other.md),
>    [Chapter 47](47-brains-that-see-and-worlds-that-break.md)); whether they
>    do has not been measured.
> 6. Worlds now cannot split
>    ([Chapter 48](48-a-world-that-cannot-break.md)), rather than living on
>    apart; long-range links were not adopted.

> [!info] Added later: a seventh obstacle, and the first
> [Chapter 19](19-what-the-brains-are-like.md), written after this chapter,
> opened the brains. Their **gain is 0.0015**: each of five sigmoid layers
> passes a change in the inputs on at about a quarter of its size, so a brain
> computes nearly the same thing whatever it sees. Selection can then only
> set its constants — and has, alike in every world: few children, the stake
> spread, every token revolutionary. **Brains that cannot see** come before
> the six obstacles above, because every remedy for them — incumbency worth
> defending, tags that tell kin, cooperation that pays — needs brains that
> respond to who and what is around them.
> [Chapter 20](20-what-agents-say-to-each-other.md) adds that the tags
> already exist: a message is the name of its writer's genotype, and kin
> could tell each other by it, if a brain could compare what it reads. The
> remedy is in the architecture — fewer layers, a steeper squashing, larger
> starting weights ([How a signal fades through layers](../notes/vanishing-signal.md)).
> [Chapter 47](47-brains-that-see-and-worlds-that-break.md) tried fewer
> layers, as the first step of Part V: one hidden layer of 16 gives a
> founder a gain of 0.2 — and four of ten such worlds of 5,000 tokens died
> out. Chapter 53, *How big should a brain be?*, is still to come.

## What we propose to change

Each proposal is a rule that can be switched on, run against the baseline with
thirty seeds, and measured — the method of Part IV. Those marked with a §
come from the research plan (`research/Research.md`, §4); the others from what
Part III found. They are ordered by what has to come first, not by how
interesting they are.

### First: make heredity real

- **Mutation at replication, and nowhere else** (§4.1 of the plan). A brain
  changes only when it is copied — into a child or into a won node. A brain
  that is not copied stays itself. It is tried in
  [Chapter 47](47-brains-that-see-and-worlds-that-break.md); its measure was
  to be how long a lineage stays like its ancestor, and whether selection
  then separates good brains from bad.
- **A germline** (§4.2), if that is not enough: every agent carries an
  unchanged copy of the brain it was born with, and passes that on.

> [!info] Added later: mutation at replication, tried
> [Chapter 47](47-brains-that-see-and-worlds-that-break.md) tried it with
> brains that see (Experiment 13, ten worlds of 5,000 tokens): with one
> hidden layer of 16, four worlds died out; with change only when copied as
> well, eight did. A stricter variant, which changed a brain only when it was
> copied into a newborn, froze a pilot world into one genotype with no births
> at all. B2 keeps the rule, inside rules that stop worlds breaking and
> starving ([Chapter 48](48-a-world-that-cannot-break.md),
> [Chapter 49](49-who-dies-and-why.md)). How long a lineage now stays like
> its ancestor has not been measured. A germline has not been tried.

### Second: let individuals last

- **Incumbency.** The agent on a node gets an advantage in defending it — its
  home stake counts double, or a coalition must outweigh the home stake too.
  Measure: how long brains hold hubs; how long partners last.
- **Recombination at conquest.** A won node takes a mix of the winner's brain
  and its own, not a replacement — the closest thing to sex this world could
  have, and a way for an agent to survive in part when it loses.

> [!info] Added later: incumbency, tried
> [`incumbency`](../notes/settings.md#incumbency) is now a setting. In pilots
> with brains that see, a home stake counting double cut conquest from 40–90%
> of the nodes a game to 15–25%, and births came back as the way brains
> spread — but one agent gathered 57% of all the tokens, staked them all at
> home where nobody could outbid them, and had child after child: a queen.
> B2 uses 1.5, with a share of every agent's tokens kept at home, and in the
> pilots no queen formed ([Chapter 49](49-who-dies-and-why.md)). How long
> brains hold their nodes, and how long partners last, has not been
> measured; under B2's cap on links no agent is joined to more than a few per
> cent of the others ([Chapter 48](48-a-world-that-cannot-break.md)).
> Recombination at conquest has not been tried.

### Third: let cooperation pay, and let kin find each other

- **Mutual flow yield** (§4.4). A connection across which tokens flow in both
  directions in a game yields a small number of new tokens, split between its
  ends — with a small decay of every holding, so that the world stays bounded.
  Now two agents together can be worth more than two apart, and there is a
  defection problem to solve, which is what cooperation theory is about.
- **Heritable tags.** Every brain carries a few numbers — a tag — copied with
  it and changed rarely; every agent sees its candidates' tags among its
  inputs. Tags are how cooperation can evolve without memory or reputation
  (Riolo, Cohen and Axelrod 2001), and they give relatives a way to recognise
  each other.

> [!info] Added later: a yield, tried
> A yield on stakes returned, paid for by a small tax on holdings, was tried
> in the pilots of Part V ([Chapter 50](50-why-evolution-stops.md)). It went
> to whoever had the most partners and built stars rather than partnerships,
> and evolution stopped as in every other pilot: cooperation needs a surplus
> that pairs, not hubs, can earn. Heritable tags have not been built; the
> messages already serve as tags
> ([Chapter 20](20-what-agents-say-to-each-other.md)).

### Fourth: open the world

- **Proposed connections** (§4.5): two agents anywhere in each other's
  neighbourhood of neighbourhoods may form a connection if both want it, at a
  cost — and, rarely, a child is joined to an agent anywhere in the world. The
  world's shortcuts would be renewed instead of spent.
- **Let separate pieces live.** Drop the global cull: a piece that separates
  becomes its own world, with its own tokens, and may meet the main world again
  through the long-range connections above. Isolation, divergence and
  reunion — the oldest engine of new species — would become possible.
- **Several kinds of tokens** (§4.3): tokens of different colours, which agents
  value differently by inheritance. Niches, and a reason to trade.

> [!info] Added later: the world closed instead
> In the pilots of Part V an agent could pay tokens for a short walk through
> the network and join an agent it met, if both agreed: few links were made,
> one to ten an iteration, and little changed. Part V keeps every rule local
> ([Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md),
> [Chapter 7](07-physical-inspiration.md)), so no child is joined to an agent
> anywhere; and it took the other road from letting pieces live —
> [Chapter 48](48-a-world-that-cannot-break.md) adds local rules under which
> worlds cannot split at all. A piece that lives on as a world of its own is
> planned for Chapter 63, *Local rules only*. Several kinds of tokens have
> not been tried.

### Later: let the possible grow

Brains that can grow (§4.6), agents that set some of their own rules (§4.7),
a second heritable structure that belongs to a relationship rather than an
agent (§4.8), and, at the very end, groups that copy themselves (§4.9) — each
only once the earlier steps hold.

> [!info] Added later: brains that can grow, tried
> In the pilots of Part V a brain could gain or lose hidden units by
> mutation, beside a "shadow" size that nothing reads
> ([Chapter 50](50-why-evolution-stops.md)). Without handover the real size
> followed its shadow — the left wall, not selection — and evolution stopped
> as in every other pilot. §4.7 to §4.9 have not been tried.

Three smaller repairs from [Chapter 39](39-questioning-the-mechanics.md) can go
along with any of these: a smooth share function, a separate output for
whole-agent decisions, and the option to hold tokens back from the game.

> [!info] Added later: one of the three, in part
> Only something like the last has been tried:
> [`keep_home`](../notes/settings.md#keep_home) keeps a share of every
> agent's tokens on its own node, so that no agent can stake itself to death
> ([Chapter 49](49-who-dies-and-why.md)) — held at home, not out of the game.
> The share function and the averaging of whole-agent decisions are
> unchanged.

## How we will know

A rule change is only progress if a measurement says so, against a null. The
measures, by rung of the ladder:

- **For P1.** How far a lineage's brains drift from their ancestor's in a given
  time, in weights and in behaviour, against unrelated pairs (heredity
  half-life). And selection ablation: the same world with the winner of every
  node drawn by chance among its stakers — if the real rule does no better,
  selection is not doing the work.
- **For P2.** The cross-time tournament: earlier against later brains, in the
  same world. A clean gradient — later beats earlier — is accumulation; a
  pattern of bands is cycling, as rock–paper–scissors would give.
- **For cooperation** — the measures of [Chapter 38](38-do-agents-cooperate.md):
  kin discrimination, Hamilton's accounting Σ(*rB* − *C*), reciprocity beyond
  the even split, the lifetimes of partnerships, coalitions that repeat, and
  whether regions held by one family outlast mixed ones.
- **For P3.** Groups that persist while their members turn over, found by
  methods that do not assume them — flow modules (Rosvall and Bergstrom 2008),
  or groups that carry information about their own past into their future
  better than their surroundings do (Krakauer et al. 2020) — always against a
  network rewired at random with the same connections, which Part III showed
  is very different from the real one.
- **For P4.** Bedau and Packard's evolutionary activity (1992): the cumulative
  presence of components that persist, set against a "neutral shadow" — a run
  in which the same events happen but no component has any advantage; and the
  MODES measures of change, novelty, complexity and ecology (Dolson, Vostinar,
  Wiser and Ofria 2019), each with a filter that only counts what persists.
  Open-endedness is the curves that never level off, above their shadows.

> [!info] Added later: what has been used
> The cross-time tournament is built and is Part V's yardstick, with a
> dominance tournament held against its chance null
> ([Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md)); a neutral
> shadow judged brains that can grow
> ([Chapter 50](50-why-evolution-stops.md)). The heredity half-life,
> selection ablation, the cooperation measures of [Chapter 38](38-do-agents-cooperate.md), flow modules
> and evolutionary activity have not yet been used in the book.

## What Part III changed in what this book teaches

Findings change the teaching, and the earlier chapters have been brought up to
date where they did:

- **Wealth follows connections because of a law**, not a mystery: the even
  split's resting state ([Chapter 15](15-where-do-the-tokens-go.md) now says
  so, and [Chapter 29](29-where-the-tokens-flow.md) derives it).
- **"Keeping one's node" is about places**: hubs are nearly always taken
  ([Chapter 16](16-how-the-game-is-played.md)).
- **The network's shape has a mechanism**: copying neighbourhoods at birth
  gives the stars, the triangles and the hierarchy
  ([Chapter 17](17-what-shape-does-the-network-take.md),
  [Chapter 34](34-how-properties-scale-together.md)).
- **"An agent" has two meanings**, the node and the brain, and the book now
  says which it means where it matters ([Chapter 39](39-questioning-the-mechanics.md)).

## Questions for the next analyses

Some questions need no new runs, only new looks at the old ones:

- **Do family regions persist as organisations?** A region held by one lineage
  for hundreds of iterations, while its nodes turn over, would be a first
  candidate for P3 — measurable now, against a null that shuffles lineages.
  [Chapter 23](23-like-next-to-like.md) found genotypes clustered within two
  or three steps and no territories; [Chapter 60](60-worlds-of-100000-tokens.md)
  found lines, which last longer, in patches that widen from 7 steps to 12 as
  they age. Whether a patch is ever more than the sum of its agents is the
  next question.
- **What does a brain respond to?** Feed a stored brain the same inputs with one
  changed — a neighbour's tokens, a message — and see what its stakes do. If
  brains mostly tell "me" from "others" and little else, the even split is no
  surprise. [Chapter 19](19-what-the-brains-are-like.md) did this: doubling
  every neighbour's tokens changes the stakes of 1.5% of agents, setting every
  message to 0 those of 9%. Brains respond to almost nothing.
- **Where are the old?** At 10,000 tokens a picture suggested the edges; at
  100,000 the old sit in clumps of their own ([Chapter 37](37-one-world-many-colours.md)),
  as cohorts ([Chapter 23](23-like-next-to-like.md)). Age against distance from
  the hubs would say.
- **Why do worlds reproduce less as they age?** Births per agent fall over a
  world's first five hundred iterations and then, in a large world, hold
  still ([Chapter 13](13-how-ages-change.md)); founders reproduce twenty times
  as often as settled agents ([Chapter 31](31-how-agents-have-children.md)).
  That is a candidate for an adaptation — the clearest one Part III saw — but
  one that is finished early, not one that accumulates.

> [!info] Added later: the adaptation, dated
> [Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md) dates it: the
> baseline's brains beat their founders by iteration 50 and improve no
> further. [Chapter 47](47-brains-that-see-and-worlds-that-break.md)'s births
> probe puts the change in the brains: put back into their founders' world,
> the founders' own brains have a child in about three agent-iterations in
> four, the baseline's later brains in 3 to 9 in a hundred.

## What the second edition changed

Redoing Parts II and III on worlds ten times larger left most of them as
they were, and that is itself a finding: per agent, a world of 100,000
tokens does what a world of 10,000 does — the same births and deaths within
a tenth, the same inequality, the same game, the same brains
([Chapter 60](60-worlds-of-100000-tokens.md)). What changed is how the agents
are arranged and how fast anything crosses a world:

- **The shape changes.** Half the clustering, more loops per agent, fewer
  bridges, distances twice as long, a ball dimension that rises with size
  ([Chapter 24](24-how-many-dimensions-does-a-world-have.md),
  [Chapter 25](25-the-core-of-a-world.md)).
- **Stars can take a world.** One world in four condensed around a single
  centre, and then everything about it — its core, its game, its cuts —
  follows from the star ([Chapter 33](33-the-largest-and-the-smallest.md)).
- **Lines take longer.** A takeover by one founder's line takes six times as
  long in a world ten times larger, so lines live side by side for most of a
  run, and the messages of a world speak more dialects
  ([Chapter 20](20-what-agents-say-to-each-other.md), [Chapter 60](60-worlds-of-100000-tokens.md)).
- **Demography settles.** Ages, lifetimes and births per agent hold still from
  iteration 500 on, as a stationary population's do
  ([Chapter 13](13-how-ages-change.md)).

For open-ended evolution the larger world is the better stage: more room and
more time for lines to differ, and the first lasting inherited structure, the
dynasty. It does not remove a single obstacle above — every one of them is a
rule, and the rules are the same at both sizes. The proposals above are
therefore planned at 10,000 tokens, where a run is eight times cheaper — the
first experiments of Part V, 12 to 14, ran smaller still, at 5,000 — and
those that bear on space and organisation (P3, Part VI) are to be checked at
100,000.

## The road from here

Part IV continues with single settings — the rate of mutation, coalitions
switched off, the size of the brain, the size of the world — because they are
cheap and each teaches how the world works. Part V then changes rules, in the
order above: first heredity, then lasting individuals, then cooperation that
pays and kin that know each other, then an open world. The book will measure
each with the instruments of this chapter, and come back here, in Meta III
and Meta IV, to say what moved.

> [!info] Added later: the road Part V took
> Part V did not keep this order. It built its yardstick first
> ([Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md)), then gave
> the brains sight, with change only when copied
> ([Chapter 47](47-brains-that-see-and-worlds-that-break.md)) — and the
> worlds broke. So it built rules that keep a world whole
> ([Chapter 48](48-a-world-that-cannot-break.md)) and keep agents from
> staking or breeding themselves to death, with an incumbent's advantage of
> 1.5 ([Chapter 49](49-who-dies-and-why.md)), all gathered in a second
> baseline, B2. Even then, evolution stopped in every pilot
> ([Chapter 50](50-why-evolution-stops.md)). Meta III closes Part IV; what
> Part V moved is for Meta IV, Chapter 59.

It is a long way. But Part III has made it shorter in one important sense: it
has turned a vague hope — that interesting things might emerge if we watch long
enough — into a list of specific rules that stand in the way, each of which can
be changed and tested on its own.

> [!info] Added later: the second aim
> [Chapter 7](07-physical-inspiration.md), written after this chapter, sets
> out the book's second aim — a world that grows a space and stable things in
> it — and finds it pointing at the same rules. The cull and the share-out,
> which this chapter questions for evolution's sake, also break locality, the
> speed limit physics insists on; and the organisations of P3 are what a
> physicist would call particles. Part VI follows that aim, and the rule
> changes of both parts will be tested together where they coincide.

<!-- turns -->
---

← [Chapter 39 · Questioning the mechanics](39-questioning-the-mechanics.md) · [Contents](../README.md) · [Chapter 70 · How different are the agents?](70-how-different-are-the-agents.md) →
<!-- /turns -->
