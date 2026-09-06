# Literature

The work this project is building on, read against one goal: **make open-ended
evolution happen in this substrate and prove it** (`Research.md`).

Every entry gives the citation, what the work actually says, and — the part
that earns it a place — **what it means here**: what we should copy, what we
should measure, or what it tells us is impossible.

Sources were checked against publication records in September 2026. Where a
claim is from a summary rather than the full paper it is marked *(abstract
only)*. Nothing here is a substitute for reading the paper before citing it in
a write-up.

---

## 1. What "open-ended" is supposed to mean

The field does not have one definition. It has a family, and they disagree
about what must grow without bound. Picking one is a design decision, not a
discovery — see `Research.md` §1.2, where we pick.

### Evolutionary activity statistics — the first operational definition
**Bedau & Packard 1992; Bedau, Snyder, Brown & Packard 1997; Bedau, Snyder &
Packard 1998.**

Give every component — a gene, a rule, a genotype — a counter that ticks for
each timestep it is present *and in use*. From the counters build three
statistics: diversity, **new** activity, and mean cumulative activity. Systems
then fall into three classes: **class 1**, no adaptive activity; **class 2**,
unbounded new activity but bounded diversity; **class 3**, both unbounded.
Class 3 is what the Phanerozoic fossil record does. By their measurement, the
artificial systems of the day did not.

The load-bearing part is not the counter, it is the **neutral shadow**: an
otherwise identical run in which adaptive benefit is randomly reassigned among
components, destroying adaptation while preserving everything else. Activity is
only meaningful as activity-above-shadow.

> **Here.** This is the template for every headline number we will produce, and
> the reason `Research.md` §6 insists each measurement carries a control. It
> also sets a trap we have already fallen into once: activity counters need
> components that *persist*, and under the measured heredity half-life of ~3
> iterations nothing persists, so activity would measure our mutation rate.

### MODES — four hallmarks, and the persistence filter
**Dolson, Vostinar, Wiser & Ofria 2019.** *Artificial Life* 25(1):50–73.

Splits open-endedness into four separately measurable potentials — **change**,
**novelty**, **complexity**, **ecology** — with C++ implementations meant to be
dropped into an existing system. The practical contribution is the
**persistence filter**: only count a component after it has survived some
number of generations, because otherwise mutational noise registers as novelty.
The paper notes the filter is used partly *because* setting up a generalisable
neutral shadow is hard.

> **Here.** The persistence filter matters more for us than for most, because
> every agent currently mutates every iteration — without a filter, our novelty
> count would essentially be our birth rate. Whatever threshold we choose is a
> free parameter that can manufacture the answer, so it gets reported.

### Three kinds of open-endedness — the one that tells us what to build
**Taylor 2019.** *Artificial Life* 25(2):207–224, "Evolutionary Innovations and
Where to Find Them".

Distinguishes three routes by their relationship to the space of phenotypic
behaviours:

- **Exploratory** — recombining existing traits into novel adaptations.
- **Expansive** — discovering and exploiting **new affordances**.
- **Transformational** — discovering **new state spaces**, often by exaptation.

The sharp claim: **standard evolutionary processes give exploratory
open-endedness only.** Expansive and transformational need extra structure —
Taylor names *multiple domains of behaviour*, *transdomain bridges*, and
*non-additive compositional systems*.

> **Here.** This is the most directly actionable definition in the literature
> for our purposes, because it names the ingredients rather than the symptoms.
> "Multiple domains" and "transdomain bridges" is an argument for
> `token_colours` (§4.3) that we arrived at independently — several kinds of
> resource *is* multiple domains, and an agent that converts between colours
> *is* a transdomain bridge. "Non-additive compositional systems" is the
> argument for organisations being more than the sum of members, which is what
> `mutual_flow_yield` (§4.4) is for.

### Variation, innovation, transition
**Banzhaf, Baumgaertner, Beslon, Doursat, Foster, McMullin, de Melo, Miconi,
Spector, Stepney & White 2016;** and the OEE workshop reports, **Taylor et al.
2016** (*Artificial Life* 22(3):408) and **Packard et al. 2019** (*Artificial
Life* 25(2):93).

Novelty comes in grades: a new point in a fixed space (variation), a new way of
using that space (innovation), or a change to the space itself (transition).
Only the third is properly open-ended.

> **Here.** This is the strongest argument against our substrate and it must be
> answered rather than dodged. Brain architecture is fixed at configuration, so
> genotype space is a fixed-dimensional box and every run is a walk inside it —
> variation, permanently. `Research.md` §1.2 answers it by changing the object
> of evolution from the genome to the organisation, whose space is not
> fixed-dimensional. If that answer fails, `growable_layers` (§4.6) is the
> fallback, and it is a direct assault on this objection rather than a way
> around it.

### Four necessary conditions — a checklist we can score ourselves against
**Soros & Stanley 2014**, "Identifying Necessary Conditions for Open-Ended
Evolution through the Artificial Life World of Chromaria"; see also **Soros,
Cheney & Stanley 2016** on how the *strictness* of the criterion matters.

Four conditions for an evolutionary process to be open-ended:

1. Individuals must meet a **minimal criterion** in order to reproduce.
2. The evolution of individuals should **create novel opportunities** for
   meeting that criterion.
3. Individuals should **decide for themselves** how to interact with the world.
4. The potential complexity of the phenotype **must not be limited by its
   representation**.

Scoring this system honestly:

| condition | this substrate | note |
|---|---|---|
| 1. minimal criterion | **yes** | reproduction costs tokens and needs at least one; below a whole token, no child |
| 2. novel opportunities | **partly** | agents change the graph, which changes who can be reached — but no rule creates a *kind* of opportunity that did not exist |
| 3. agents decide | **strongly yes** | every allocation, link, message and even the degree of randomness is a brain output |
| 4. representation does not cap complexity | **no** | fixed layer sizes. This is the failing condition |

> **Here.** Three of four, failing exactly where Banzhaf says we would. Note
> that condition 3 is one we satisfy *unusually* well — most systems hard-code
> the interaction rule and evolve only a strategy inside it.

### Unbounded evolution as a dynamical-systems property
**Adams, Zenil, Davies & Walker 2017.** *Scientific Reports* 7:997, "Formal
Definitions of Unbounded Evolution and Innovation Reveal Universal Mechanisms
for Open-Ended Evolution in Dynamical Systems".

Recasts open-endedness away from biology-flavoured criteria into dynamical
systems. **Unbounded evolution**: patterns that do not repeat within the
expected Poincaré recurrence time of the isolated system. **Innovation**:
trajectories not observed in the isolated system. They then test cellular
automata whose update rules vary with time in three different ways, and find
that **state-dependent rules** — the rule depends on the current state —
outperform the alternatives and are **the only mechanism producing open-ended
evolution in a scalable way**.

> **Here.** The most important single result for mechanism design in this list.
> It is direct empirical support for `local_rules` (§4.7), where brain outputs
> set an agent's own mutation rate, reproduction threshold or edge permissions.
> It also gives a definition we can actually implement: compare the run against
> the recurrence time of the *isolated* system. We have the isolated system —
> it is the run with `local_rules` off.

### Novelty and learnability, from the machine-learning side
**Hughes, Dennis, Parker-Holder, Behbahani, Mavalankar, Shi, Schaul &
Rocktäschel 2024.** ICML, "Open-Endedness is Essential for Artificial
Superhuman Intelligence".

Argues open-endedness is necessary for artificial superhuman intelligence, and
offers a formal definition through **novelty** and **learnability**, defined
with respect to an **observer**. *(Abstract and summaries only — I did not get
the formal statement out of the PDF, so the exact definition needs checking
before it is cited in a write-up.)*

> **Here.** Two things are useful even from the shape of it. First, the
> observer-relativity: open-endedness is not a property of the system alone but
> of the system *and* what is watching it — which is a warning that our
> detectors partly define our result. Second, **learnability** is a criterion
> we could actually use: novelty that is merely random is not open-ended;
> novelty that a model could come to predict *given the past* is. That
> distinguishes our current drift from what we want, and it is a sharper
> instrument than diversity.

### Complexity growth is not the same as size growth
**Standish 2003.** *International Journal of Computational Intelligence and
Applications*, "Open-Ended Artificial Evolution".

Notes that most people equate open-ended evolution with complexity growth
although these are a priori different, and measures complexity of Tierran
organisms using the neutrality of the genotype–phenotype map. In a size-neutral
run of Tierra, **organism size increased but complexity did not**.

> **Here.** A direct warning about the plan. If castles get *bigger* without
> getting more *organised*, that is Standish's null result, not our claim. Any
> "organisations grow" measurement needs a complexity axis that is not size —
> assembly index (§7) and causal emergence (§6) are the two candidates.

---

## 2. The game we are playing

### Colonel Blotto — no pure equilibrium, by construction
**Borel 1921**; **Roberson 2006**, "The Colonel Blotto game", *Economic Theory*
29:1–24; multiplayer generalisation by **Boix-Adserà, Edelman & Jayanti**
(2021, *Journal of Economic Theory*; earlier EC/arXiv versions).

Two commanders split a fixed force across battlefields without seeing each
other's split; each field goes to whoever committed more. **Even the simplest
Blotto games admit no pure equilibrium** — any fixed allocation is beaten by
one that concedes where it is strong and overwhelms where it is weak. Roberson
characterised the optimal marginal distributions and unique equilibrium payoffs
for the continuous, constant-sum, two-player case, using copulas to assemble
marginals into a joint distribution. The multiplayer continuous case is much
more recent and much harder.

> **Here.** This is the engine of our non-transitivity and the reason there is
> nothing to converge to. Three differences from the classical setting, all of
> which matter and none of which are covered by the theory: resources are **not
> fixed across rounds** (winning gets you tokens, so success compounds), the
> battlefields are **neighbours in an evolving graph**, and there are **many
> players**. We are in the multiplayer, dynamic, graph-structured corner, which
> is roughly where the literature stops.

### Non-transitivity has a geometry
**Czarnecki, Gidel, Tracey, Tuyls, Omidshafiei, Balduzzi & Jaderberg 2020.**
NeurIPS, "Real World Games Look Like Spinning Tops".

Real games (Go, StarCraft II, Tic-Tac-Toe) have a strategy space shaped like a
spinning top: an **upright axis of transitive strength**, and a **radial axis
of non-transitivity** — the number of cycles that exist at a given strength.
Cycles are widest at intermediate skill and narrow at both ends. The practical
consequence: you need a **population** of strategies to train against, and the
population size you need is set by the width of the top.

> **Here.** The best available frame for what our strategy space might look
> like, and it makes a testable prediction we can check with the cross-time
> tournament (`Research.md` §6.3): if we are on a spinning top, the CIAO matrix
> should show **cycles at intermediate strength and a gradient at the ends**.
> A banded CIAO plot is therefore not a failure — it is the signature. It also
> justifies keeping the whole population as the object of study rather than
> looking for a best agent, which there is not one of.

### Intransitivity plus *local* interaction preserves diversity
**Kerr, Riley, Feldman & Bohannan 2002.** *Nature* 418:171–174, "Local
dispersal promotes biodiversity in a real-life game of rock–paper–scissors".

Three *E. coli* strains — colicin-producing, sensitive, resistant — in a
rock-paper-scissors relationship. All three coexist when interaction and
dispersal are **local**; diversity collapses when the population is mixed.

> **Here.** Possibly the single most encouraging empirical result for this
> substrate, because we have both ingredients and did not put them there for
> this reason: an intransitive game *and* strictly local interaction on a
> graph. It predicts that our diversity is not incidental, and that it should
> **collapse if we make the graph well-mixed** — which is a cheap and decisive
> ablation nobody has run. Add it to the battery.

### The Red Queen
**Van Valen 1973**, "A new evolutionary law".

Extinction probability is roughly independent of how long a taxon has already
survived. The explanation offered: a species' environment is mostly *other
species*, all improving, so absolute gains buy no relative advantage. The
zero-sum relation is the **engine** of sustained change, not an obstacle to it.

> **Here.** The paper that retired this project's own zero-sum objection.
> Conservation of tokens is a Red Queen setting, not a cage.

---

## 3. Structure: evolutionary graph theory

### Spatial structure rescues cooperation
**Nowak & May 1992.** *Nature* 359:826–829.

Prisoner's dilemma on a lattice with imitate-the-best-neighbour dynamics:
cooperation survives, because cooperators form clusters whose interiors trade
only with each other and whose growth can outpace erosion at the edge. In a
well-mixed population the same game exterminates cooperation.

> **Here.** The founding reason to expect anything from a graph-structured
> version of a competitive game, and it sets the shape of what to look for: not
> a cooperative *strategy* but a cooperative **region**, with a protected
> interior and a boundary where the loss happens. That is very close to what we
> mean by an organisation.

### Graphs amplify or suppress selection
**Lieberman, Hauert & Nowak 2005.** *Nature* 433:312–316.

A Moran process on a graph: the graph's shape changes the fixation probability
of an advantageous mutant. Some graphs are **amplifiers** (a star is one),
some are **suppressors** making selection nearly neutral.

> **Here.** Usually read as "choose your graph and you have chosen your
> selection strength". Nobody chooses ours — the agents build it, so
> amplification is **endogenous and time-varying**. That is a genuinely
> unstudied setting, and it is measurable: fixation rate inside a persistent
> flow module against the rate outside it. A structure that made its own
> neighbourhood into an amplifier for its own variants would be a real result.

### b/c > k
**Ohtsuki, Hauert, Lieberman & Nowak 2006.** *Nature* 441:502–505, "A simple
rule for the evolution of cooperation on graphs and social networks".

Cooperation is favoured when the benefit-to-cost ratio of the cooperative act
exceeds the **average degree**: `b/c > k`. A good approximation across cycles,
lattices, random regular graphs, random graphs and scale-free networks.

> **Here.** *Measured*: our mean degree falls from ~3.9 to ~3.2 over 45
> iterations, and the agents partly control it. So the threshold is an
> **evolvable target rather than a parameter**, and a population could evolve
> its own graph across the cooperation threshold. That is a novel setting for a
> famous result and is publishable independently of the open-endedness claim.
> It is also the theoretical basis for `mutual_flow_yield` (§4.4), which is
> what supplies the `b` that this rule needs and that we currently lack.

### Any structure at all
**Allen, Lippner, Chen, Fotouhi, Momeni, Yau & Nowak 2017.** *Nature*
544:227–230, "Evolutionary dynamics on any population structure".

Generalises the above to **any** graph under weak selection, including graphs
where the number of neighbours varies — previously open. Gives a condition for
when selection favours one strategy over another on an arbitrary structure.

> **Here.** The tool for saying something quantitative about our *actual*
> graphs, which are irregular and changing, rather than about a regular
> idealisation of them. Weak selection is an assumption we would have to check.

### Adaptive networks — dynamics *on* and *of* the network
**Gross & Blasius 2008.** *J. R. Soc. Interface* 5(20):259–271, "Adaptive
coevolutionary networks: a review" (online 2007). See also **Perc & Szolnoki
2010**, *BioSystems* 99(2):109–125, "Coevolutionary games — a mini review".

When dynamics on the network and dynamics of the network run at comparable
speed and feed back, you get things neither gives alone: spontaneous separation
into classes of node, robust self-organised states, phase transitions with no
external parameter driving them. Perc & Szolnoki survey the game-theoretic
case, where the interaction network, reproduction capability, reputation,
mobility or age can coevolve alongside strategy.

> **Here.** We are squarely an adaptive network and did not set out to be. One
> asymmetry is ours alone and shapes everything: **edges are created only by
> reproduction and destroyed whenever they carry nothing**, so the topology
> erodes by default and can only be rebuilt through birth. `edge_proposal`
> (§4.5) exists to remove that asymmetry, and Perc & Szolnoki is the map of
> what tends to happen once you do.

---

## 4. What counts as an organisation

This is the section that matters most, because `Research.md` §1.2 makes the
organisation the unit of evolution. Three formal theories, all with algorithms.

### Chemical organisation theory — closure plus self-maintenance
**Dittrich & Speroni di Fenizio 2007.** *Bulletin of Mathematical Biology*
69(4):1199–1231, "Chemical Organisation Theory".

An **organisation** is a set of species that is **closed** (reactions among
members produce nothing outside the set) and **self-maintaining** (the set can
sustain its own members' production). The organisations of a reaction network
form an **algebraic lattice**, and they filter the phase space: attractors can
only live in regions corresponding to organisations.

> **Here.** This is the formal home for the word we have been using informally.
> It gives a definition of "castle" that is not hand-waved and not
> genealogical: a set of agents whose token flows are closed among themselves
> and which sustains its own members. It is also a *lattice*, which means
> organisations nest — which is exactly the hierarchy a major transition needs.
> **Recommended as the primary detector**, ahead of flow modules, because it is
> a definition rather than a clustering heuristic.

### RAF sets — self-sustaining collectives, found in polynomial time
**Hordijk & Steel 2004, 2011**; see **Hordijk 2023** for a concise formal
statement.

A **Reflexively Autocatalytic and Food-generated** set: every reaction in it is
catalysed by something the set itself produces or that comes from the food set,
and every reactant is reachable from the food set using only reactions in the
set. There is a **polynomial-time algorithm** to find the maximal RAF. The
result that made the theory matter: RAFs appear with high probability at modest
and chemically realistic catalysis levels — collective self-sustenance is
*easy*, and easier than individual self-replication.

> **Here.** A second, cheaper detector, and the more encouraging framing: if
> RAF-like structures arise readily in random reaction networks, structures
> that mutually sustain each other's token supply may arise readily here too.
> The mapping is direct — agents are species, staking is catalysis, the
> redistribution pool is the food set. **A polynomial-time detector for
> collective self-sustenance is exactly the tool `Research.md` §6.4 is missing.**

### Individuality as information that propagates itself
**Krakauer, Bertschinger, Olbrich, Flack & Ay 2020.** *Theory in Biosciences*
139:209–223, "The information theory of individuality".

Stop asking what an individual is made of and ask what it **propagates**. An
individual is an aggregate that preserves temporal integrity — that carries
information about its own past into its own future in excess of what the
environment supplies. Yields three distinct forms: **organismal** (mostly
self-determined), **colonial**, and **driven** (mostly environment-determined),
varying in environmental dependence and inherited information.

> **Here.** The principled version of what our flow modules approximate, and
> the three forms give us a *scale* rather than a yes/no. The honest prediction
> is that our current structures are the **driven** kind — determined by the
> environment rather than by themselves — and that the whole programme is an
> attempt to move them along the axis toward organismal. That is a much better
> framing of success than "we found castles".

### Assembly theory — complexity as history, measured by copy number
**Sharma, Czégel, Lachmann, Kempes, Walker & Cronin 2023.** *Nature*
622:321–328, "Assembly theory explains and quantifies selection and evolution".

An object's **assembly index** is the minimal number of steps to build it from
basic parts; the **assembly equation** combines that with **copy number** — how
many instances exist. The claim: high assembly index *together with* high copy
number cannot happen by chance and is therefore a signature of selection. The
molecular version is experimentally measurable by mass spectrometry.

**Contested.** Several papers argue assembly index is reducible to Shannon
entropy or standard compression and adds nothing (e.g. **Abrahão, Hernández-
Orozco, Kiani, Zenil et al. 2024**, and follow-ups on assembly theory's
relationship to computational complexity, *npj Complexity* 2025). Treat it as a
promising instrument under active dispute, not as settled.

> **Here.** Even granting the criticism, the *shape* of the idea is exactly
> what we need and I have not seen it applied this way: **an organisation that
> is both hard to build and common is evidence of selection.** Our castles have
> a natural assembly index (how many steps to construct that topology and role
> assignment) and a natural copy number (how many instances exist at once). It
> answers Standish's warning directly — it separates *big* from *hard to make*.
> Worth prototyping precisely because it is contested; if it works here it is
> an independent test of the theory.

### Autocatalysis of the possible — why the space keeps growing
**Kauffman**, the **adjacent possible**; formalised by **Tria, Loreto,
Servedio & Strogatz 2014**, *Scientific Reports* 4:5890, "The dynamics of
correlated novelties".

A generalisation of Pólya's urn in which drawing a novel item *enlarges the
urn*. It predicts **Heaps' law** for the rate at which novelties appear (number
of distinct items grows as a power of the number of draws) and **Zipf's law**
for their frequency distribution, and it fits Wikipedia edits, tagging systems,
word sequences and music listening.

> **Here.** This turns "the space of organisations grows without bound" from a
> philosophical claim into a **curve with a predicted shape**. Plot distinct
> organisations against total organisation-observations: an expanding adjacent
> possible gives Heaps' law with an exponent below 1; a closed space saturates.
> That is a cheap, decisive, and *falsifiable* test of `Research.md` P4, and it
> is the best measurement idea in this entire review.

---

## 5. Levels, and how a new one appears

### The major transitions
**Maynard Smith & Szathmáry 1995**, *The Major Transitions in Evolution*.

History as a short list of events in which entities that previously reproduced
independently could afterwards reproduce only as part of a larger whole: genes
into chromosomes, prokaryotes into eukaryotes, cells into organisms, organisms
into colonies. Each needs the same problem solved — how the parts are stopped
from defecting.

> **Here.** The template for the goal, and it rules out an easy shortcut.
> Defining an organisation as a set of relatives makes relatedness ≈ 1 by
> construction and lets kin selection explain the cooperation trivially — a
> circular argument. Eukaryogenesis is the famous transition between
> *unrelated* parties. So our organisation detectors are defined on **token
> flow only**, with genealogy kept as separate evidence.

### MLS1 and MLS2 — which selection we are claiming
**Okasha 2006**, *Evolution and the Levels of Selection*; **Price 1970** for
the underlying identity.

The Price equation splits change in a population average into a covariance
(selection) term and a transmission term, and it **nests**: apply it within and
between groups and the two levels separate. Okasha's distinction: **MLS1** is
selection on *particle*-level traits mediated by group membership; **MLS2** is
selection on the *collectives themselves* as reproducing entities. He argues
the two are better seen as a continuum, and that the continuum is what makes
sense of the major transitions.

> **Here.** Names what we are actually claiming. Castles that merely help their
> members is **MLS1** and is comparatively easy. Castles that reproduce *as
> castles* is **MLS2**, and it is what `structure_replication` (§4.9) builds in
> deliberately. The nested Price equation is a measurement we can run today —
> partition into organisations, decompose the change in a trait into
> between-group and within-group components, and a consistently larger
> between-group term is group selection. Everything it needs already exists.

### How a transition actually happened, in a lab
**Ratcliff, Denison, Borrello & Travisano 2012.** *PNAS* 109(5):1595–1600,
"Experimental evolution of multicellularity". Follow-up: **Ratcliff et al.
2015**, *Nature Communications* 6:6102, "Origins of multicellular evolvability
in snowflake yeast".

Yeast evolved into multicellular "snowflake" clusters within weeks. The
mechanism: daughter cells fail to separate after division, so clusters are
**clonal by construction**. That gives high broad-sense heritability for
*multicellular* traits and precludes genetic conflict, and the group becomes
the primary unit of selection. A single mutation created a new level of
organisation *and* potentiated higher-level evolvability.

> **Here.** The most useful engineering lesson in this review. The transition
> did not need a clever group-selection mechanism; it needed **group-level
> traits to be heritable**, and clonality delivered that almost for free. Our
> equivalent question is: *what makes an organisation's properties heritable?*
> Currently nothing does — which is why `contracts` (§4.8) and
> `structure_replication` (§4.9) exist, and why they may matter more than any
> amount of ecology.

---

## 6. Is the higher level real?

If we claim the organisation is the unit, we should be able to show the
organisation is where the causation is.

### Causal emergence — macro can beat micro
**Hoel, Albantakis & Tononi 2013.** *PNAS* 110(49):19790–19795, "Quantifying
causal emergence shows that macro can beat micro". Extended in **Hoel et al.
2016**, *Neuroscience of Consciousness*.

Build causal models at micro and macro scales and measure **effective
information**. Although the macro is fully specified by the micro, EI can
**peak at a macro scale**. Causal emergence is then the supersedence of a macro
model over a micro one.

> **Here.** A quantitative test of our central claim. If effective information
> peaks when the system is described as *organisations interacting* rather than
> *agents interacting*, then the organisation is not our bookkeeping
> convenience, it is where the causal structure is. That is as close to proof
> of "the organisation is the unit" as this kind of argument gets.

### The practical version
**Rosas, Mediano, Jensen, Seth, Barrett, Carhart-Harris & Bor 2020.** *PLOS
Computational Biology* 16(12):e1008289, "Reconciling emergences".

An information-decomposition treatment giving a quantitative definition of
**downward causation** and a second mode, **causal decoupling**, with criteria
that can be **computed efficiently on large systems**. Demonstrated on Conway's
Game of Life, Reynolds flocking, and ECoG data.

> **Here.** Hoel's measure does not scale; this one is designed to. Since it
> has already been applied to the Game of Life and to flocking — both
> "structures made of many simple parts" — it should transfer to a population
> of agents and their organisations with little adaptation. **This is the
> practical route to §6 above and should be preferred over implementing EI
> directly.**

### Structure defined negatively
**Crutchfield & Hanson 1993**, computational mechanics on cellular automata;
**Beer** on gliders in the Game of Life as autopoietic systems; **Maturana &
Varela** for autopoiesis itself.

Find structure by first learning the **regular background** — what the system
does when nothing is happening — and declaring structure to be whatever the
background fails to predict. Gliders and domain walls are the residue.

> **Here.** The methodological idea this project keeps rediscovering the hard
> way: structure is what a background model does not predict, so the background
> model has to be built *first*. Counting conquest cycles gave thousands and
> looked like a finding until a random-neighbour null showed the rate was
> **below chance**. The flow-module compression has not yet had the same
> treatment.

### The map equation
**Rosvall & Bergstrom 2008.** *PNAS* 105(4):1118–1123.

Find communities by compression: describe a random walk with a two-level code —
one codebook naming modules, one naming positions inside each — so names can be
reused across modules. Minimise the description length. Because it describes
**flow** rather than counting edges, it finds where things *go*.

> **Here.** Implemented, and running as the *Flow modules* view. It reads token
> allocations, so it sees choices rather than topology. But it is a clustering
> heuristic, not a definition — chemical organisation theory and RAF give
> definitions, and should be preferred where they apply. Keep the map equation
> as the fast approximate detector.

---

## 7. The systems we will be compared with

| system | what it did | the mechanism that made it work | what we have instead |
|---|---|---|---|
| **Tierra** (Ray 1991) | self-replicating machine code in shared memory; parasites, hyper-parasites and cheats within hours | genome **length can grow**; open-ended template matching | fixed genome — but an evolving interaction topology, which Tierra has no analogue of |
| **Avida** (Ofria & Wilke 2004) | the instrument of digital evolution; showed complex functions need rewarded intermediates | rewarded task hierarchy, growable genome | no reward function at all — better for open-endedness, harder to measure |
| **Geb** (Channon 2001) | argued to pass Bedau class 3 by activity statistics | unbounded genome plus coevolution | the bar for our §6.6 |
| **Polyworld** (Yaeger 1994) | embodied agents with evolved neural controllers | embodiment and ecology | ecology is what `token_colours` and `mutual_flow_yield` are for |
| **Echo** (Holland) | the system Bedau's classes were calibrated on | — | the neutral-shadow methodology came from here |
| **Chromaria** (Soros & Stanley 2014) | first world where the minimal criterion's strictness could be manipulated directly | tunable minimal criterion | our criterion is fixed at "hold a token"; making it tunable is cheap and untried |
| **Lenia + Quality-Diversity** (Chan 2019; Faldor & Cully 2024) | automatic discovery of diverse self-organising patterns | QD search over a continuous CA | closest modern analogue to "find the structures"; QD is a search method we could borrow for detection |

The pattern, and it is uncomfortable: **every system that convincingly showed
open-endedness let its entities grow.** Our bet is that letting the
*organisation* grow substitutes for letting the *genome* grow. Nobody has
demonstrated that substitution works. If it does, that is the contribution.

---

## 8. Method

### Neutral models
**Kimura 1968** and the neutral theory generally.

The lasting contribution is less the claim that most substitutions are neutral
than the discipline it forced: a selective explanation must beat an explicit
drift model before it is accepted.

> **Here.** Learned twice already. Conquest cycles ran at 0.87–0.91× a
> random-neighbour null — *below* chance, so the finding was the opposite of
> the one first written down. Flow-module compression still has no null and
> should not be quoted as evidence until it does.

### Cross-time tournaments
**Cliff & Miller 1995**, CIAO — Current Individual versus Ancestral Opponents.

Plot the current population against ancestors from every earlier generation as
a matrix. Genuine progress gives a clean gradient; a cycle gives bands.

> **Here.** The decisive measurement, and the one that cannot be fooled by a
> system that forgets — a population with no heredity cannot beat its own past.
> Checkpoints already store what it needs. Read together with Czarnecki's
> spinning top, **bands are a positive result**, not a negative one.

### Timescale
**Lenski**, the long-term evolution experiment, from 1988.

Twelve *E. coli* populations, tens of thousands of generations, frozen samples
so any ancestor can be revived and competed. Aerobic citrate use appeared after
~31,000 generations and only because earlier, unremarkable mutations had made
it reachable.

> **Here.** The corrective. Our runs are hundreds of iterations. The frozen
> fossil record is our checkpoint file and the revive-and-compete design is
> CIAO, so we have the method — what we lack is the timescale, and "we did not
> run long enough" must be distinguishable from "it levelled off". That
> distinction is exactly what a Heaps' law fit (§4) provides, since a
> saturating curve and a slow power law look different long before either
> finishes.

---

## 9. What this reading changes

Six things I did not have before, in order of how much they change the plan.

1. **Heaps' law is the test for P4.** Tria et al. turn "the space grows without
   bound" into a curve with a predicted exponent. Cheap, falsifiable, and it
   distinguishes "not long enough" from "levelled off" — the failure mode that
   would otherwise sink the whole programme. **Add to `Research.md` §6.6.**
2. **Chemical organisation theory and RAF give us definitions, not
   heuristics.** Closed-and-self-maintaining, with a polynomial-time algorithm
   for the RAF version. This should displace flow modules as the primary
   organisation detector. **Rewrite §6.4.**
3. **Taylor's transdomain bridges independently justify `token_colours`.**
   Multiple domains plus bridges between them is his recipe for *expansive*
   open-endedness, and several token types with agents that convert between
   them is exactly that. This raises my confidence in §4.3 and lowers it for
   spending effort elsewhere first.
4. **The well-mixed ablation is missing and is cheap.** Kerr et al. predict our
   diversity depends on locality. Randomising the graph each iteration should
   collapse it. If it does not, our spatial structure is not doing what we
   think. **Add to the battery.**
5. **Rosas et al. gives a scalable emergence test.** Already demonstrated on
   the Game of Life and flocking. This is the practical way to ask whether the
   organisation is where the causation is, and it is stronger evidence than any
   count of structures.
6. **Ratcliff: the transition needed heritable group traits, not clever group
   selection.** Snowflake yeast worked because clusters were clonal by
   construction. Our analogue is missing entirely — nothing makes an
   organisation's properties heritable. This raises the priority of §4.8/§4.9
   above the ecology work.

And one that cuts the other way, kept because it should:

7. **Standish measured size growing while complexity did not.** If our
   organisations get bigger without getting harder to build, that is his null
   result. Assembly index and causal emergence are the two ways to tell the
   difference, and at least one of them has to be in place *before* we report
   growth.

---

## Sources

- [Open-Ended Evolution: Perspectives from the OEE Workshop in York](https://direct.mit.edu/artl/article/22/3/408/2841/Open-Ended-Evolution-Perspectives-from-the-OEE) — Taylor et al. 2016
- [An Overview of Open-Ended Evolution (OEE II editorial)](https://arxiv.org/abs/1909.04430) — Packard et al. 2019
- [The MODES Toolbox](https://direct.mit.edu/artl/article/25/1/50/2915/The-MODES-Toolbox-Measurements-of-Open-Ended) — Dolson, Vostinar, Wiser & Ofria 2019
- [Evolutionary Innovations and Where to Find Them](https://arxiv.org/abs/1806.01883) — Taylor 2019
- [Formal Definitions of Unbounded Evolution and Innovation](https://www.nature.com/articles/s41598-017-00810-8) — Adams, Zenil, Davies & Walker 2017
- [Open-Endedness is Essential for Artificial Superhuman Intelligence](https://arxiv.org/abs/2406.04268) — Hughes et al. 2024
- [Open-Ended Artificial Evolution](https://arxiv.org/pdf/nlin/0210027) — Standish 2003
- [Identifying Necessary Conditions for Open-Ended Evolution (Chromaria)](https://www.semanticscholar.org/paper/4671423a1b65f3e35dce603f8746e72ae31193dc) — Soros & Stanley 2014
- [How the Strictness of the Minimal Criterion Impacts Open-Ended Evolution](https://www.uvm.edu/neurobotics/pubs/pdf/2016_SorosCheneyStanley_HowTheStrictnessOfTheMinimalCriterionImpactsOpenEndedEvolution_ALIFE.pdf) — Soros, Cheney & Stanley 2016
- [Toward Artificial Open-Ended Evolution within Lenia using Quality-Diversity](https://arxiv.org/abs/2406.04235) — Faldor & Cully 2024
- [The Multiplayer Colonel Blotto Game](https://www.sciencedirect.com/science/article/abs/pii/S0899825621000592) — Boix-Adserà, Edelman & Jayanti 2021
- [Colonel Blotto Game: An Analysis and Extension to Networks](https://arxiv.org/html/2407.16707)
- [Real World Games Look Like Spinning Tops](https://arxiv.org/pdf/2004.09468) — Czarnecki et al. 2020
- [Local dispersal promotes biodiversity in a real-life game of rock–paper–scissors](https://www.eurekalert.org/news-releases/469007) — Kerr, Riley, Feldman & Bohannan 2002
- [A simple rule for the evolution of cooperation on graphs and social networks](https://www.aidenlab.org/papers/Nature.Cooperation.Graphs.pdf) — Ohtsuki, Hauert, Lieberman & Nowak 2006
- [Evolutionary dynamics on any population structure](https://www.nature.com/articles/nature21723) — Allen et al. 2017
- [Adaptive coevolutionary networks: a review](https://royalsocietypublishing.org/doi/10.1098/rsif.2007.1229) — Gross & Blasius 2008
- [Coevolutionary games — a mini review](https://www.sciencedirect.com/science/article/abs/pii/S0303264709001646) — Perc & Szolnoki 2010
- [Chemical Organisation Theory](https://link.springer.com/article/10.1007/s11538-006-9130-8) — Dittrich & Speroni di Fenizio 2007
- [A Concise and Formal Definition of RAF Sets and the RAF Algorithm](https://arxiv.org/pdf/2303.01809) — Hordijk 2023
- [Autocatalytic sets and chemical organizations](https://iopscience.iop.org/article/10.1088/1367-2630/aa9fcd)
- [The information theory of individuality](https://pubmed.ncbi.nlm.nih.gov/32212028/) — Krakauer, Bertschinger, Olbrich, Flack & Ay 2020
- [Assembly theory explains and quantifies selection and evolution](https://pubmed.ncbi.nlm.nih.gov/37794189/) — Sharma, Walker, Cronin et al. 2023
- [Assembly Theory Reduced to Shannon Entropy…](https://arxiv.org/pdf/2408.15108) — the critique
- [Assembly theory and its relationship with computational complexity](https://www.nature.com/articles/s44260-025-00049-9)
- [The dynamics of correlated novelties](https://www.nature.com/articles/srep05890) — Tria, Loreto, Servedio & Strogatz 2014
- [Quantifying causal emergence shows that macro can beat micro](https://www.pnas.org/doi/10.1073/pnas.1314922110) — Hoel, Albantakis & Tononi 2013
- [Reconciling emergences](https://arxiv.org/abs/2004.08220) — Rosas et al. 2020
- [Experimental evolution of multicellularity](https://www.pnas.org/doi/10.1073/pnas.1115323109) — Ratcliff et al. 2012
- [Origins of multicellular evolvability in snowflake yeast](https://www.nature.com/articles/ncomms7102) — Ratcliff et al. 2015
- [Evolution and the Levels of Selection](https://ndpr.nd.edu/reviews/evolution-and-the-levels-of-selection/) — Okasha 2006
- [Passing the ALife Test: Activity Statistics Classify Evolution in Geb as Unbounded](https://link.springer.com/chapter/10.1007/3-540-44811-X_45) — Channon 2001
