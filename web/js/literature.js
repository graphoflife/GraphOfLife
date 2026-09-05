/*
 * The reading list: the work this project is built on and measured against.
 *
 * Prose kept as prose, the same way Explain.STEPS keeps the walk-through and
 * StatDetail.EXPLANATIONS keeps the statistic descriptions. Twenty entries
 * share one skeleton — a title, who said it, the idea, and what it means here
 * — so the skeleton is written once below and the words are data. Written out
 * by hand instead, it was 470 lines of index.html in which the only thing that
 * varied was the sentences.
 *
 * It also has to be kept in step with research/Research.md §9, which covers
 * the same ground for a different reader. Prose is easy to diff against prose
 * and hard to diff against <article><h4>.
 *
 * The `here` note is the part that earns an entry its place, and the part to
 * argue with: the citations are summarised from working knowledge, which the
 * page says of itself.
 */
const Literature = {
  LEAD: `The ideas this simulation is built on, and the ones it has to be
        measured against. The claim under test is a single sentence &mdash;
        <b>this algorithm shows open-ended evolution</b> &mdash; and most of
        what follows exists either to say what that would mean or to say why
        it probably is not happening.`,

  CAUTION: `These summaries are written from working knowledge, not from the
           papers open on the desk. Treat every year and every attribution as
           needing a check before it goes anywhere that matters. The <em>what
           it means here</em> notes are the part worth arguing with.`,

  GROUPS: [
    {
      heading: 'Open-ended evolution',
      entries: [
        {
          title: 'Evolutionary activity statistics',
          cite: 'Bedau &amp; Packard 1992; Bedau, Snyder &amp; Packard 1998',
          what: `The first serious attempt to make &ldquo;is this system still
                evolving?&rdquo; a number rather than an impression. Give
                every component &mdash; a gene, a rule, a genotype &mdash; an
                activity counter that ticks up for every timestep it is
                present and in use. Sum those counters and you get the
                cumulative activity of the whole population; watch how new
                components accumulate activity over time and systems fall into
                three classes. <b>Class 1</b> has bounded diversity and no new
                activity: it settles. <b>Class 2</b> has unbounded new
                activity but bounded diversity: things keep changing, nothing
                accumulates. <b>Class 3</b> has both unbounded &mdash; new
                things keep appearing <em>and</em> the stock of things keeps
                growing. Class 3 is what the biosphere does, and by Bedau's
                own measurement none of the artificial life systems of the day
                reached it.`,
          here: `The counters need components that persist long enough to
                accumulate anything. A brain id here names a genotype &mdash;
                a copy keeps its source's id, only a mutation makes a new one
                &mdash; which is exactly the identity an activity counter
                needs, and the reason that change was made. What has not been
                done is the measurement itself: nothing in this project has
                yet plotted activity against a neutral shadow, which is the
                only form in which the number means anything.`
        },
        {
          title: 'MODES, and the persistence filter',
          cite: 'Dolson, Vostinar, Wiser &amp; Ofria 2019',
          what: `A toolkit that splits open-endedness into four separately
                measurable things: <b>change</b> (the population is not what
                it was), <b>novelty</b> (something never seen before is here),
                <b>complexity</b> (the most complex thing present is more
                complex than before), and <b>ecology</b> (how many distinct
                niches are occupied). The move that makes it work is the
                <em>persistence filter</em>: only count a component after it
                has survived some number of generations, so that mutational
                noise &mdash; variants that appear and die immediately &mdash;
                does not register as novelty.`,
          here: `This filter matters more here than in most systems, because
                every reproduction mutates. Without a filter the novelty count
                would essentially be the birth rate, which measures nothing.
                Any honest activity or novelty statistic for this simulation
                has to pass through a persistence threshold first, and the
                threshold has to be reported, because it is a free parameter
                that can manufacture the answer.`
        },
        {
          title: 'Novelty that expands the space',
          cite: 'Banzhaf et al. 2016; Taylor et al. 2016; Packard et al. 2019',
          what: `The sharpest distinction in the field, and the strongest
                argument against the claim this project is testing. Novelty
                comes in grades. <b>Variation</b> is a new point in a space
                that was always there. <b>Innovation</b> is a new way of using
                that space. A <b>transition</b> changes the space itself
                &mdash; new state variables, new degrees of freedom, rules
                that were not previously expressible. Banzhaf and colleagues
                put it as the difference between novelty <em>within</em> a
                fixed possibility space and novelty that <em>enlarges</em> it,
                and argue that only the second deserves the name open-ended.`,
          here: `Take this seriously and the claim is in trouble. The brain
                architecture here is fixed at configuration time: a set number
                of layers of a set width, evolved by copy and mutation with no
                operator that adds a neuron, a layer, or an input. The space
                of possible genotypes is a fixed-dimensional box, and every
                run is a walk inside it. Under the strict reading this is
                variation, permanently, and no amount of running changes that.
                The counter-argument this project is actually making is that
                the space that matters is not the genotype space but the space
                of <em>multi-agent structures</em> &mdash; castles &mdash;
                whose size is not bounded by the architecture. That argument
                is only as good as the evidence that such structures exist and
                grow, which is what the rest of the Research tab is for.`
        },
        {
          title: 'Ablation as method',
          cite: 'Soros &amp; Stanley 2014',
          what: `Rather than define open-endedness, propose necessary
                conditions for it &mdash; in their case, things like a minimal
                criterion for reproduction, and freedom to accumulate. Then
                test them the way one tests any causal claim: remove one,
                rerun, and show the open-endedness goes away. The
                methodological point outlives the particular conditions. A
                property that survives every ablation was never caused by the
                mechanism you claimed.`,
          here: `This is the single most useful method available to this
                project, because every mechanism it needs is already a toggle
                in the simulation settings: messages on or off, message
                pre-pass on or off, revolutions on or off, mutation rate,
                brain kind. If module persistence or lineage depth or activity
                is really produced by non-transitivity, turning revolutions
                off should visibly cost it. No such ablation has been run yet.
                Every claim in this project is correlational until one is.`
        },
        {
          title: 'The comparison class',
          cite: 'Ray 1991 (Tierra); Ofria &amp; Wilke 2004 (Avida); Channon 2001 (Geb); Yaeger 1994 (Polyworld)',
          what: `The systems any new one gets compared to. Tierra put
                self-replicating machine-code programs in a shared memory and
                got parasites, hyper-parasites and social cheats within hours
                &mdash; none of them designed in. Avida turned that into an
                instrument, with a rewarded task hierarchy, and produced the
                well-known result that a complex function can only evolve if
                the useless-looking intermediate steps are also rewarded. Geb
                and Polyworld pushed at ecology and embodiment instead.`,
          here: `The instructive difference is that all of them let the genome
                grow. A Tierra creature's program can get longer; an Avida
                organism's genome can gain instructions. That is precisely the
                space-expanding novelty above, and it is what this system does
                not have. Where this system is unusual by comparison is that
                the <em>interaction graph is itself evolved</em> &mdash; in
                Avida the spatial arrangement is a fixed grid, whereas here
                who can reach whom is a product of what the agents did. That
                is the axis along which this simulation could plausibly say
                something the older systems could not.`
        }
      ]
    },
    {
      heading: 'Evolutionary game theory on graphs',
      entries: [
        {
          title: 'Spatial structure rescues cooperation',
          cite: 'Nowak &amp; May 1992',
          what: `Play the prisoner's dilemma on a lattice, where each cell
                copies its most successful neighbour, and cooperation survives
                &mdash; not because anyone is nice, but because cooperators
                form clusters whose interiors trade only with each other, and
                a cluster can grow faster than defectors can eat its edge. In
                a well-mixed population the same game exterminates cooperation
                completely. Structure, on its own, changes the outcome of the
                game.`,
          here: `This is the founding reason to expect anything interesting
                from a graph-structured version of a zero-sum game. It also
                sets the shape of what to look for: not a cooperative
                <em>strategy</em>, but a cooperative <em>region</em>, whose
                boundary is where the loss happens and whose interior is
                protected. That is very close to what this project has been
                calling a castle.`
        },
        {
          title: 'Amplifiers and suppressors of selection',
          cite: 'Lieberman, Hauert &amp; Nowak 2005',
          what: `Evolutionary graph theory proper. Put a Moran process on a
                graph and the graph's shape changes the fixation probability
                of an advantageous mutant. Some graphs are
                <em>amplifiers</em>, raising fixation probability above the
                well-mixed value &mdash; a star is a strong one, and certain
                constructions amplify arbitrarily. Others <em>suppress</em>,
                making selection nearly neutral. The same mutant, the same
                advantage, and the outcome is set by the wiring.`,
          here: `The result is usually read as: choose your graph and you have
                chosen your selection strength. Here nobody chooses it. The
                agents build the graph &mdash; reproduction wires a newborn
                into its parent's neighbourhood, and links that carry no
                tokens are pruned &mdash; so amplification is endogenous and
                time-varying. A structure that made its own neighbourhood into
                an amplifier for its own variants would be a real finding, and
                it is measurable: fixation rate inside a persistent flow
                module against the rate outside it.`
        },
        {
          title: 'b/c &gt; k',
          cite: 'Ohtsuki, Hauert, Lieberman &amp; Nowak 2006',
          what: `A rule of startling simplicity: on a graph where each node has
                about <em>k</em> neighbours, cooperation is favoured when the
                benefit-to-cost ratio of the cooperative act exceeds the
                average degree. Sparse graphs help cooperators, dense ones
                hurt them, and the threshold is just the degree. The intuition
                is that a cooperator's help is spread over its neighbours
                while its cost is its own, so fewer neighbours means a larger
                share of the benefit returns to relatives and clustered
                partners.`,
          here: `Mean degree in these runs falls from about 3.9 to about 3.2
                over a few hundred iterations. If the b/c rule transfers at
                all, that is the graph moving in the direction that favours
                cooperation, and it is doing so on its own. The caveat is
                real: the rule is derived for a fixed regular graph and a
                donation game, and this is neither. It is a hypothesis about
                why the degree falls, not an explanation, and the honest test
                is whether cooperative structure appears <em>after</em> the
                fall rather than before.`
        },
        {
          title: 'Five rules for the evolution of cooperation',
          cite: 'Nowak 2006',
          what: `The catalogue: <b>kin selection</b> (help relatives),
                <b>direct reciprocity</b> (help those who helped you),
                <b>indirect reciprocity</b> (help those with a good
                reputation), <b>network reciprocity</b> (help those near you,
                in a structured population), and <b>group selection</b>
                (groups of cooperators outcompete groups of defectors). Each
                comes with its own threshold condition.`,
          here: `Two of the five are simply unavailable here, and that is a
                design fact rather than an oversight. Agents have no
                persistent identity visible to one another &mdash; there is no
                way to recognise the neighbour who helped you last iteration
                &mdash; so direct and indirect reciprocity have nothing to
                attach to. What is available is network reciprocity, which is
                the graph structure itself, and kin selection, since a newborn
                is placed beside its parent and so starts life among
                relatives. Any cooperation found here has to be explained by
                those two, or by something the catalogue does not list.`
        },
        {
          title: 'Adaptive networks',
          cite: 'Gross &amp; Blasius 2008',
          what: `A review of systems where the dynamics <em>on</em> the network
                and the dynamics <em>of</em> the network run at comparable
                speed and feed back into each other. The recurring finding is
                that this coupling produces things neither dynamic gives
                alone: spontaneous separation into distinct classes of node,
                robust self-organised states, and phase transitions with no
                external parameter driving them.`,
          here: `This simulation is squarely an adaptive network &mdash; tokens
                move over links, and the links themselves are made and cut by
                how tokens moved. One asymmetry is worth naming, because it
                shapes everything: edges are only <em>created</em> by
                reproduction, and only <em>destroyed</em> by carrying nothing.
                There is no operator that lets two existing agents form a new
                link. The topology therefore erodes by default and can only be
                rebuilt through birth, which puts a hard constraint on what
                structures are reachable and is a prime candidate for the next
                mechanism to add.`
        }
      ]
    },
    {
      heading: 'The game being played',
      entries: [
        {
          title: 'Colonel Blotto',
          cite: 'Borel 1921; Roberson 2006',
          what: `Two commanders split a fixed force across several battlefields
                without seeing each other's split; each field goes to whoever
                committed more there, and the winner is whoever takes the most
                fields. The game has no equilibrium in pure strategies &mdash;
                any fixed allocation is beaten by one that concedes where it
                is strong and overwhelms where it is weak &mdash; and Roberson
                gave the equilibrium in mixed strategies for the continuous
                case, where the marginal on each field is uniform up to a
                bound.`,
          here: `This is the game phase, with two differences that matter.
                Resources are not fixed across rounds &mdash; winning gets you
                tokens, which are the resource &mdash; so success compounds.
                And the battlefields are the neighbours, so the board itself
                changes as agents are born and die. The no-pure-equilibrium
                property is the engine of non-transitivity: because there is
                no allocation that beats everything, there is always in
                principle a strategy that beats the current best. Whether the
                population <em>finds</em> such cycles is a separate, empirical
                question &mdash; and the measurement so far says conquest
                2-cycles run at 0.87&ndash;0.91&times; the rate a
                random-neighbour null produces, which is to say <em>below</em>
                chance.`
        },
        {
          title: 'The Red Queen',
          cite: 'Van Valen 1973',
          what: `From the observation that extinction probability appears
                roughly independent of how long a taxon has already survived:
                species do not get safer with age. Van Valen's explanation was
                that the environment of any species is mostly <em>other
                species</em>, each of which is also improving, so an absolute
                gain in fitness buys no relative gain. The zero-sum law it
                rests on &mdash; that adaptive gains in one lineage come at
                others' expense &mdash; is the mechanism, not an obstacle to
                it.`,
          here: `This paper retired an argument this project was making. The
                claim had been that conserved tokens make the system zero-sum
                and therefore closed to open-ended evolution
                (<code>research/Research.md</code> &sect;3.1, now marked
                retracted). Van Valen's whole point is that sustained
                evolutionary change is <em>driven</em> by a zero-sum relation,
                not prevented by it. The conservation law here bounds the
                total token count and nothing else; it says nothing about the
                complexity of the arrangements those tokens can be in.`
        },
        {
          title: 'Current Individual Ancestral Opponents',
          cite: 'Cliff &amp; Miller 1995',
          what: `How to see progress in a coevolving population, where the
                usual fitness plot is useless because the yardstick moves.
                Play the current population against ancestors from every
                earlier generation and plot the results as a matrix. Genuine
                progress gives a clean gradient &mdash; later beats earlier,
                everywhere. A cycle gives a banded pattern: a strategy that
                loses to its immediate ancestor but beats one from further
                back.`,
          here: `Directly runnable here, and not yet run. Checkpoints already
                store whole populations, so a late population can be replayed
                against a stored earlier one under the same graph. It is the
                cleanest available test for whether the non-transitivity that
                the Blotto game makes <em>possible</em> is actually being
                <em>used</em> &mdash; which matters, since the conquest-cycle
                count came out below chance.`
        }
      ]
    },
    {
      heading: 'What counts as an individual',
      entries: [
        {
          title: 'The major transitions',
          cite: 'Maynard Smith &amp; Szathm&aacute;ry 1995',
          what: `The history of life read as a short list of events in which
                entities that could previously reproduce independently could
                afterwards only reproduce as part of a larger whole: genes
                into chromosomes, prokaryotes into eukaryotic cells, cells
                into multicellular organisms, organisms into colonies. Each
                transition is a new level of selection, and each needs the
                same problem solved &mdash; how the parts are stopped from
                defecting against the whole.`,
          here: `This is the template for what a castle would have to be, and
                it also rules out an easy definition. Defining a castle as a
                set of agents sharing a lineage would make relatedness near 1
                by construction, and kin selection would then explain the
                cooperation trivially &mdash; the argument would be circular.
                Symbiosis is a transition too, and eukaryogenesis is the
                famous case of a transition between <em>unrelated</em>
                parties. A castle here is therefore defined on token flow and
                nothing else, with genealogy kept strictly as a separate
                measurement so it can be used as evidence rather than assumed.`
        },
        {
          title: 'The Price equation',
          cite: 'Price 1970',
          what: `An exact identity splitting the change in any population
                average into a selection term &mdash; the covariance between
                fitness and the trait &mdash; and a transmission term, the
                fitness-weighted average of how much offspring differ from
                parents. It assumes nothing about the system, and it nests:
                apply it within groups and between groups and the two levels
                of selection separate cleanly.`,
          here: `The nested form is the tool for asking whether a castle is a
                unit of selection at all. Partition into flow modules,
                decompose the change in some trait into a between-module and a
                within-module component, and a between-module term that is
                consistently the larger one is what group-level selection
                would look like. Everything needed for it exists &mdash; a
                partition, a per-agent trait, a fitness proxy in token gain
                &mdash; and it has not been computed.`
        },
        {
          title: 'Information-theoretic individuality',
          cite: 'Krakauer, Bertschinger, Olbrich, Flack &amp; Ay 2020',
          what: `Stop asking what an individual is made of and ask what it
                <em>propagates</em>. An individual is a subset of the world
                that carries information about its own past into its own
                future in excess of what its environment supplies &mdash; a
                region of high self-predictive information. The definition
                comes in degrees rather than a yes or no, and it explicitly
                allows an individual whose material components are entirely
                replaced.`,
          here: `This is the principled version of what the flow modules
                approximate. The map equation finds groups a token-walk tends
                to stay inside, which is a proxy for the real quantity: a set
                whose own state predicts its own next state better than the
                surroundings do. The proxy is used because it is cheap and
                already implemented. The measured module structure &mdash;
                41.5% compression, longest-lived module 48 frames, 23%
                membership turnover per frame &mdash; is exactly the shape
                this definition anticipates, a persistent pattern whose matter
                is exchanged. It is also, as things stand, uncalibrated: no
                null model has been run against it, and a number without a
                null is not yet evidence.`
        },
        {
          title: 'Structure defined negatively',
          cite: 'Crutchfield &amp; Hanson 1993; Beer 2004 (on gliders in the Game of Life)',
          what: `Computational mechanics finds structure in a spatially
                extended system by first learning the regular background
                &mdash; the pattern the system falls into when nothing is
                happening &mdash; and then declaring structure to be
                everything that background fails to predict. Gliders and
                domain walls are what is left over after the regularity is
                subtracted. Beer's reading of the glider makes the related
                point that a persistent pattern is individuated by its
                behaviour, not by its cells.`,
          here: `The most transferable methodological idea in this list, and
                the one this project keeps rediscovering the hard way.
                Structure is what a background model does not predict &mdash;
                which means the background model has to be built first.
                Counting cycles gave thousands of them and looked like a
                finding, until a random-neighbour null showed the rate was
                <em>below</em> chance. The flow-module compression has not yet
                been through the same treatment.`
        },
        {
          title: 'The map equation',
          cite: 'Rosvall &amp; Bergstrom 2008',
          what: `Find community structure by compression. Describe a random
                walk on the network with a two-level code: one codebook naming
                modules, and a separate codebook naming positions inside each
                module, so that names can be reused across modules the way
                street names are reused across cities. The description length
                is <code>L = plogp(q) &minus; 2&sum;plogp(q&#7522;) &minus;
                &sum;plogp(p&#8336;) + &sum;plogp(q&#7522;+p&#7522;)</code>,
                and the best partition is whichever minimises it. Because it
                describes flow rather than counting edges, it finds where
                things <em>go</em> rather than where links happen to be.`,
          here: `Implemented and running &mdash; this is the <em>Flow
                modules</em> view in this tab. Built directly on the token
                allocations, so it reads the choices agents made rather than
                the graph they sit in, which is what makes it a candidate
                detector for castles defined by flow. Compression of 0% means
                the flow has no group structure at all; the method is
                constructed so that structureless flow returns one module
                rather than inventing several, which
                <code>tests/test_flowmodules.js</code> checks against a ring.`
        }
      ]
    },
    {
      heading: 'Method',
      entries: [
        {
          title: 'Neutral models',
          cite: 'Kimura 1968; Harvey &amp; Thompson 2005 (on drift in artificial evolution); the standard practice throughout ecology',
          what: `Most of what looks like structure is what randomness produces
                anyway. The neutral theory's lasting contribution is less the
                claim that most substitutions are neutral than the discipline
                it forced: a selective explanation now has to beat an explicit
                drift model before it is accepted at all.`,
          here: `The lesson this project has had to learn twice. Conquest
                cycles looked abundant in the thousands and turned out to run
                at 0.87&ndash;0.91&times; a random-neighbour null &mdash;
                below chance, so the finding was the opposite of the one first
                written down. The flow-module numbers on the previous page
                have <em>no</em> null model behind them yet and should be read
                as measurements of the method rather than facts about the
                system. Written into <code>research/Research.md</code> as a
                standing rule, alongside the other one earned the same way:
                nothing measured on fewer than about thirty seeds gets
                reported as an effect.`
        },
        {
          title: 'The long-term evolution experiment',
          cite: 'Lenski, from 1988 onward',
          what: `Twelve populations of <em>E. coli</em>, propagated daily for
                tens of thousands of generations, with samples frozen along
                the way so any ancestor can be revived and competed against
                its own descendants. One population evolved the ability to use
                citrate aerobically &mdash; after about thirty-one thousand
                generations, and only because earlier, apparently unremarkable
                mutations had made it reachable.`,
          here: `A corrective on timescale, and a rebuke to impatience. The
                single most consequential innovation in the best-observed
                evolution experiment ever run took thirty-one thousand
                generations and depended on potentiating changes that were
                invisible when they happened. Runs here are hundreds of
                iterations. The frozen fossil record has a direct analogue in
                the checkpoint files, and the replay-the-ancestor design is
                the same one CIAO calls for &mdash; which is a second reason
                to build it.`
        }
      ]
    }
  ],

  /**
   * Paint the whole page once. It never changes, takes no run, and has no
   * state, so there is nothing to redraw and no reason to do it lazily.
   */
  render() {
    const host = document.getElementById('research-literature');
    if (!host) return;

    const entry = e => `
        <article>
          <h4>${e.title}</h4>
          <p class="lit-cite">${e.cite}</p>
          <p>${e.what}</p>
          <p class="lit-here">${e.here}</p>
        </article>`;

    const group = g => `
      <section class="lit-group">
        <h3>${g.heading}</h3>${g.entries.map(entry).join('')}
      </section>`;

    host.innerHTML = `
    <div class="lit-page">
      <div class="explain-intro">
        <h2>Literature</h2>
        <p class="lit-lead">${this.LEAD}</p>
        <p class="lit-caution">${this.CAUTION}</p>
      </div>${this.GROUPS.map(group).join('')}
    </div>`;
  }
};
