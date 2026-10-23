# The candidates, played on

> [!question] Questions of this page
> - The worlds that came nearest the author's picture at [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 1,000 —
>   do they stay there over 10,000 iterations?
> - Which of them live, which fall back to [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s poverty, and which keep
>   evolving?

<!-- experiment E20 -->

## Why ask

A world that meets every mark of the [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4.") at iteration 1,000, with a
[diversity score](diversity-score.md "How much the agents of a world differ, in one number from 0 to 1: in tokens, links, ages and genotypes.") above B2's [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world, is a **[candidate](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides.")**
([Healthy and rich](healthy-and-rich.md)). One moment says little: mixed
worlds drift in size, and their riches drift with it
([What holds a world's size](what-holds-a-worlds-size.md)). So every candidate is
played on to 10,000 iterations, and read by **[Healthy and rich](healthy-and-rich.md)'s test of a long run**:
at 10,000 it is still alive — it never fell to 20 [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") or fewer, where the
[lab](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") calls a world [extinct](extinction.md "A world's end before its last iteration: it has dropped to its threshold of agents or fewer.") — it meets every mark, it is more diverse than B2's
median world at 10,000, and it holds a smaller share of its agents at three
[tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") or fewer than the least poor of B2's ten worlds.

Twenty-five candidates were played on as [pilot](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.") worlds (batch LC), each in a
copy of its own. Nineteen reached 10,000 iterations. Then, on 18 October 2026,
the computer was restarted: the pilots had been running out of sight, in no
page of the notebook, and from then on every simulation of this notebook runs
where it can be seen — in the lab, started with ▶ beside the page that
asks for it.

Six candidates were left. Two of them use only the engine's own [settings](settings.md "One number or switch of a run's rules, fixed before it starts."):
links to strangers by a free reach, forced on the other end
([Links to strangers](links-to-strangers.md)), and [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") mixed at birth with only
the last token kept at home
([Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md)). **Experiment 20** plays
them in the lab from iteration 0. A pilot plays the engine's own [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), so a
pilot with a lab world's settings and [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") is that lab world — the two mixed
worlds of batch Q turned out to be Experiment 17's worlds of seeds 1 and 2,
agent for agent ([Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md)) — and
the lab's two worlds should be the pilots' two worlds again, game for game.
The other four candidates use rules that so far exist only in the pilots'
prototype — contests decided by lottery, brain units that cost a token, an
input that tells where a brain came from, a family tag — and follow once those
rules are part of the engine.

## The thesis

<!-- thesis E20 -->
> [!quote] The thesis of Experiment 20, written down before any of its runs existed
> **The claim.** Played by the lab from iteration 0, these two worlds are the pilot worlds of batch LC, game for game: in each, the agents alive and the links between them after every game match the pilot's, iteration by iteration, up to where the pilot stopped (iteration 4,344 for links to strangers, 6,522 for mixing with the last token kept). Played on to 10,000, each is then read by [Meta X](healthy-and-rich.md)'s test of a long run.
>
> **Why it would be so.** A pilot plays the engine's own game, so a pilot with a lab world's settings and seed is that lab world: the two mixed worlds of batch Q are Experiment 17's worlds of seeds 1 and 2, agent for agent. These two candidates of [Meta X](healthy-and-rich.md) were being played on to 10,000 iterations as pilots when the computer was restarted on 18 October 2026; pilots now run only where they can be seen, and both use engine settings only, so the lab can finish them. At iteration 1,000 both met every mark of the health bar: links to strangers by a free reach, forced on the other end (batch K), with 1,931 agents, 49% of them at three tokens or fewer and a diversity score of 0.54; mixing with only the last token kept at home (batch Y), with 1,791 agents, 79% poor and 0.62. Read where the pilots stopped, they held 2,584 agents, 59% poor, at 4,000, and 2,066, 78% poor, at 6,500 — this plan was written after those readings, so the claim is only that the lab plays the same worlds.
>
> **It holds if:** Both worlds' agents and links match their pilot's after every game up to where the pilot stopped.
>
> **It fails if:** A world's agents or links differ from its pilot's at some iteration: the lab plays another world than the pilot did, and the chapter says from where.
>
> *Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*
<!-- /thesis -->

## The runs

<!-- runs E20 -->
> [!info] The runs behind this page
> **2 runs.** Every condition below is run once for every seed (1), for 10,000 iterations — or until its world dies out. Every setting is that of the baseline **B2** (a world that cannot break) unless the condition changes it.
>
> - **links to strangers, a free reach, forced** — changes `link_search` = reach, `link_consent` = forced; 10,000 tokens. Runs `B2-10000-e10e0d-s001` … `B2-10000-e10e0d-s001`.
> - **mixed at birth, the last token kept** — changes `mix_at_birth` = on, `keep_home` = 0.01; 10,000 tokens. Runs `B2-10000-82bd1e-s001` … `B2-10000-82bd1e-s001`.
>
> To make these runs again: `python3 gol_lab.py run E20`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | links to strangers, a free reach, forced | mixed at birth, the last token kept | what it is |
> |---|---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 10,000 | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | 16 | 16 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 8 | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | off | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | off | off | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](settings.md#allow_gifting) | off | off | Whether agents may give tokens to neighbours during reproduction. Off in every run of this book. |
> | [`random_decisions`](settings.md#random_decisions) | off | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | on | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.5 | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 2 | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | on | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 2 | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 8 | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](settings.md#keep_home) | 0.3 | **0.01** | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 100 | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 10 | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | off | **on** | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`self_message`](settings.md#self_message) | own | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | **reach** | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | **forced** | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | none | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | 20 | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 | 1 | one run per seed |
> | iterations | 10,000 | 10,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B2.
<!-- /runs -->

## The twin check

**The thesis holds.** Both of the lab's worlds match their pilots: the agents
alive and the links between them after every game are the same, through all
4,344 games the pilot of links to strangers had played and all 6,522 of the
pilot of mixing with the last token kept — though both pilots had been
resumed from a checkpoint at iteration 1,000. A pilot of engine rules
is the lab's world, so the long runs read below stand as if the lab had made
them. The four candidates with pilot rules cannot be checked this way yet;
Experiment 23 is to play them.

## Where the candidates went

<!-- figure candidates/paths -->
![From 1,000 iterations to 10,000](../figures/candidates/paths.svg)

**From 1,000 iterations to 10,000.** Each line is a candidate, from where it stood at iteration 1,000 (small grey dot) to where it stood at 10,000 (large dot): green if it still met every mark of the health bar, was more diverse than B2's median world and poorer than none of B2's worlds — [Healthy and rich](healthy-and-rich.md)'s test of a long run — violet if it fell back, red if it had died out on the way (20 agents or fewer, the lab's line) and was refilled from its last agents. Yellow: B2's ten worlds played on to 10,000; blue: Experiment 17's mixed worlds that lived to 10,000. The author's picture is the top left: agents that differ, and few of them poor.

> [!example]- How to make this figure
> **Runs.** Batch LC's long runs (`GraphOfLifeRuns/.pilots/LC/`): each candidate of [Healthy and rich](healthy-and-rich.md), B2 at 10,000 tokens with the rules of the batch it came from, played on in a copy of its own from its checkpoint to 10,000 iterations (`research/oee/pilots/run.py --resume`), two of them finished by the lab from iteration 0 (Experiment 20: `B2-10000-e10e0d-s001`, `B2-10000-82bd1e-s001`); Experiment 17's ten mixed worlds (`B2-10000-4af4a3-s001` … `s010`) and B2's ten worlds played on to 10,000 (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`).
>
> **Data.** From each world's checkpoint at 1,000 and at 10,000 (`research/atlas/at.py --at 1000` and `--at 10000`, `GraphOfLifeRuns/.pilots/atlas/at1000.json` and `at10000.json`).
>
> 1. Read each world as `research/atlas/profile.py` reads every world.
> 2. Rank its four spreads among the frozen reference worlds; the diversity score is their mean.
>
> **To make it again:** `python3 book_figures.py candidates`.
<!-- /figure -->

**Five of nineteen held.** Nineteen long runs can be read at iteration
10,000: seventeen pilots, and the two of Experiment 20 in place of their
unfinished pilots. Five passed [Healthy and rich](healthy-and-rich.md)'s test there, all of them mixed at
birth: with a fast change of brains (a [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") probability of 0.8) in both
its seeds, and in one seed each without random inputs, without [revolutions](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake."),
and with an input that tells an agent where its brain came from. Eleven fell
back — seven of them into B2's poverty, two thirds of their agents or more at
three tokens or fewer, and four by a single mark of the health bar: a
[dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.") above four in two large worlds, an agent joined to more than 5% of
the others in two small ones. **Three died out on the way**: mixing with the
[estate](estate.md "The tokens a dead agent leaves, and who gets them.") of a dead agent left to its own children (at iteration 4,043), with
brains that grow (3,158) and with an input that tells who helped (4,271).
Each fell to a single agent, from which the pilot, which plays on where the
lab would stop, filled its world again.

> [!info] Added later (19 October 2026): too even, not too poor
> The author's worry was a world whose agents hold their tokens too evenly,
> not one with many poor; the share at three tokens or fewer only stood in for
> it ([Healthy and rich](healthy-and-rich.md)). Read with the
> [Gini coefficient](gini-coefficient.md) of the agents' tokens instead, the verdict turns round:
> - **Four of the five that held are more even than every one of B2's
>   worlds**, with a Gini of 0.37 to 0.44 against B2's 0.47 to 0.51. They
>   are fast brain change in both seeds, no random inputs and no
>   revolutions: rich worlds of 8 to 27 tokens per agent, spread more evenly
>   than a random exchange of tokens would leave them (0.47 to 0.49).
> - **Two that fell back did so on the share of the poor alone.** They meet
>   every other count, with tokens less even than B2's: mixing with only the
>   last token kept at home, seed 2 (80% at three or fewer, a Gini of 0.66),
>   and with an input that tells which [neighbours](the-world.md#the-network "An agent joined to another by a connection.") carry the agent's own
>   brain, seed 2 (69%, 0.56).
> - **Only the input of where a brain came from (seed 2) passes both
>   readings.**
>
> Under the clarified test — alive, every mark, more diverse than B2's
> median world, and no more even than B2 — **three of nineteen held**:
> where a brain came from, the last token kept, and the sense of kin, each in
> one seed. Across the nineteen, a world's size goes with its share of poor
> agents (a rank [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of 0.85) far more than with how evenly its
> tokens are spread (0.46 with the Gini). The small rich worlds are the even
> ones; the large poor ones are not.

**Diversity fell in sixteen of the nineteen.** What a candidate had at
iteration 1,000 — agents that differ in tokens, links, ages and [genotypes](genotype.md "The identity of a brain: a number that a brain keeps until it changes, and that records whose copy it was.") —
mostly wore off: the median diversity score fell from 0.63 to 0.56, and the
two worlds that fell furthest, mixing with the estate to the poorest and
with stakes pooled by genotype, ended less diverse than any of B2's worlds
(0.28 and 0.34).

<!-- figure candidates/sizes -->
![How large each candidate grew](../figures/candidates/sizes.svg)

**How large each candidate grew.** Each line is a candidate: its agents after each game, means over 50 iterations, on a logarithmic axis. Green: the candidates that held at 10,000 iterations; violet: those that fell back; red: those that died out on the way and were refilled from their last agents.

> [!example]- How to make this figure
> **Runs.** Batch LC's long runs (`GraphOfLifeRuns/.pilots/LC/`): each candidate of [Healthy and rich](healthy-and-rich.md), B2 at 10,000 tokens with the rules of the batch it came from, played on in a copy of its own from its checkpoint to 10,000 iterations (`research/oee/pilots/run.py --resume`), two of them finished by the lab from iteration 0 (Experiment 20: `B2-10000-e10e0d-s001`, `B2-10000-82bd1e-s001`); Experiment 17's ten mixed worlds (`B2-10000-4af4a3-s001` … `s010`) and B2's ten worlds played on to 10,000 (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`).
>
> **Data.** From each long run's `series.jsonl`.
>
> 1. Read the agents after each game.
> 2. Average over stretches of 50 iterations.
>
> **To make it again:** `python3 book_figures.py candidates`.
<!-- /figure -->

**Rich and small, or large and poor.** Sizes wandered — several worlds shrank
to a few hundred agents and grew back — and where a world ended, its riches
went with its size: across the nineteen at 10,000, the more agents, the
poorer they were (a rank correlation of 0.85). The five that held had 372 to
1,244 agents; the eight largest that lived (1,846 to 3,097) all fell back,
seven of them into poverty. It is [What holds a world's size](what-holds-a-worlds-size.md)'s finding again: a world's size is set
by how eagerly its agents breed, and a full world is a poor one
([What holds a world's size](what-holds-a-worlds-size.md)).

<!-- figure candidates/space -->
![Room, at 1,000 iterations and at 10,000](../figures/candidates/space.svg)

**Room, at 1,000 iterations and at 10,000.** Each line is a candidate: the steps within which a typical agent's ball holds half its world, over the same for its random twin ([What folds a world](what-folds-a-world.md)), at iteration 1,000 and at 10,000. Green: the candidates that held; violet: those that fell back; red: those that died out on the way. Beside the right end: B2's ten worlds played on to 10,000 (yellow) and Experiment 17's (blue).

> [!example]- How to make this figure
> **Runs.** Batch LC's long runs (`GraphOfLifeRuns/.pilots/LC/`): each candidate of [Healthy and rich](healthy-and-rich.md), B2 at 10,000 tokens with the rules of the batch it came from, played on in a copy of its own from its checkpoint to 10,000 iterations (`research/oee/pilots/run.py --resume`), two of them finished by the lab from iteration 0 (Experiment 20: `B2-10000-e10e0d-s001`, `B2-10000-82bd1e-s001`); Experiment 17's ten mixed worlds (`B2-10000-4af4a3-s001` … `s010`) and B2's ten worlds played on to 10,000 (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`).
>
> **Data.** From each world's checkpoint at 1,000 and 10,000 (`research/atlas/space.py --at 1000` and `--at 10000`).
>
> 1. Count, from a sample of agents, how many others lie within each number of steps.
> 2. Find the steps within which the typical ball holds half the world, between whole steps by the logarithm of the ball.
> 3. Do the same on a random network with every agent's links kept; divide.
>
> **To make it again:** `python3 book_figures.py candidates`.
<!-- /figure -->

**None became a space.** At iteration 1,000 a few candidates were wider than
their [random twins](random-twin.md "A network with the same agents and as many links each, wired at random: what a world would be if it were no space at all.") by half or more; by 10,000 all but one were folded, at
1.09 to 1.44 times the twin's typical distance, as B2's own worlds are. The
exception, mixing with the estate to the poorest (1.79), is one of the poor
worlds ([What folds a world](what-folds-a-world.md)).

<!-- figure candidates/evolution -->
![Do later brains beat earlier ones?](../figures/candidates/evolution.svg)

**Do later brains beat earlier ones?** Each dot is a world: the share of a contested world the later moment's brains hold after 100 iterations against the earlier moment's, the mean of two contests, one in each moment's world; the bar is the median world. 5,000/2,000: the brains of iteration 5,000 against those of 2,000. Orange: the candidates (the first step judged on the world each came from; a world that died out on the way has the steps before its death only); grey: Experiment 17's mixed worlds.

> [!example]- How to make this figure
> **Runs.** Batch LC's long runs (`GraphOfLifeRuns/.pilots/LC/`): each candidate of [Healthy and rich](healthy-and-rich.md), B2 at 10,000 tokens with the rules of the batch it came from, played on in a copy of its own from its checkpoint to 10,000 iterations (`research/oee/pilots/run.py --resume`), two of them finished by the lab from iteration 0 (Experiment 20: `B2-10000-e10e0d-s001`, `B2-10000-82bd1e-s001`); Experiment 17's ten mixed worlds (`B2-10000-4af4a3-s001` … `s010`) and B2's ten worlds played on to 10,000 (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`).
>
> **Data.** From `GraphOfLifeRuns/.pilots/lc_T100.json` (`research/oee/checks/lc_rejudge_worlds.txt`), the batches' own contests for 1,000/500, and `e17_T100.json`.
>
> 1. Seat the brains of the two moments on alternate agents of each moment's world and play 100 iterations (`research/oee/checks/rejudge.py`).
> 2. Count the agents the later moment's brains hold, as a share of all.
>
> **To make it again:** `python3 book_figures.py candidates`.
<!-- /figure -->

**Evolution goes on, slowing, as in Experiment 17.** Played for a [lifetime](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death."),
the brains of 1,000 beat those of 500 in seventeen of eighteen worlds (a
median share of a contested world of 0.72), those of 5,000 beat those of 2,000
in fifteen of sixteen (0.60), and those of 10,000 beat those of 5,000 in
eleven of sixteen (0.55); Experiment 17's mixed worlds held 0.84, 0.66 and
0.54. Ten worlds won both of the late steps; none ran away.

## What it teaches

- **A pilot of engine rules is the lab's world.** The twin check shows it
  game for game, so what the pilots found stands, and the lab can take up
  where a pilot stopped by playing it again.
- **Most candidates do not stay candidates.** Of nineteen, five held over
  10,000 iterations, eleven fell back and three died out — and the five that
  held did not hold again in new seeds
  ([Five more seeds for the rules that held](five-more-seeds-for-the-rules-that-held.md)).
- **What lasts is what [What holds a world's size](what-holds-a-worlds-size.md) predicts**: small worlds rich, large worlds
  poor, diversity wearing off, and, in the small rich worlds, no room — every
  rich candidate folded.
- **Evolution never stops, and never speeds up.** The later brains keep
  winning, by less and less, in every kind of world played this long.

**Related notes:** [Healthy and rich](healthy-and-rich.md) ·
[What holds a world's size](what-holds-a-worlds-size.md) ·
[Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md) ·
[Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md) ·
[Links to strangers](links-to-strangers.md) ·
[What folds a world](what-folds-a-world.md) ·
[Five more seeds for the rules that held](five-more-seeds-for-the-rules-that-held.md)

<!-- turns -->
---

← [A mate that pays its share](a-mate-that-pays-its-share.md) · [Looking for a healthy, rich world](../README.md#looking-for-a-healthy-rich-world) · [Five more seeds for the rules that held](five-more-seeds-for-the-rules-that-held.md) →

> [!info]- Pages that link here (8)
> [The random twin: is a world a space?](random-twin.md) · [Every era against three](every-era-against-three.md) · [Rules in company](rules-in-company.md) · [Healthy and rich](healthy-and-rich.md) · [The health bar](health-bar.md) · [The diversity score](diversity-score.md) · [Five more seeds for the rules that held](five-more-seeds-for-the-rules-that-held.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
