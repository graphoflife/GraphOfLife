# The ideas this builds on

> [!question] Questions of this page
> - What has been tried before, and what would [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau.") mean?
> - Which ideas from game theory, network science, population genetics and
>   statistics does this world use?

## In short

Graph of Life puts together ideas from several fields: artificial life, game
theory, network science, population genetics and statistics. This page
introduces each one far enough to read the rest of the notebook, and says
where it shows up in this world. There is no agreed definition of open-ended
evolution, but there are good tests. The most useful is old: bring a
population's ancestors back, and see whether the descendants beat them. The
full references are at the end. The project's longer reading notes are in
`research/Literature.md`.

## Open-ended evolution: what it would mean

There is no single definition. These are the main attempts, and this project
will have to face each of them:

- **Evolutionary activity** (Bedau and Packard 1992; Bedau, Snyder, Brown and
  Packard 1997; Bedau, Snyder and Packard 1998). Give every component — a
  gene, a [genotype](genotype.md "The identity of a brain: a number that a brain keeps until it changes, and that records whose copy it was.") — a counter that grows while it is present and in use. Then
  watch three numbers: how many components there are, how much *new* activity
  appears, and how much activity accumulates. Systems fall into three classes:
  no adaptive activity; unbounded new activity but bounded diversity; and both
  unbounded, which is what the fossil record shows. The lasting lesson is the
  **neutral shadow**: activity counts only above what an identical system
  without adaptation produces. [Do the brains matter?](do-the-brains-matter.md)
  builds a shadow of this kind for [lineages](families.md "A line of descent: the agents whose genotypes go back to one ancestor."), and finds it hard.
- **MODES** (Dolson, Vostinar, Wiser and Ofria 2019) splits open-endedness into
  four kinds of growth that can be measured: change, novelty, complexity and
  ecology. It adds a **persistence filter**: a component counts only once it
  has survived for a while, or random [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") would look like novelty. In
  this world a genotype typically lasts two iterations
  ([Genotypes and lineages](genotypes-and-lineages.md)), so that filter decides
  almost everything.
- **Three kinds of open-endedness** (Taylor 2019). *Exploratory*: new
  combinations within a fixed space. *Expansive*: new opportunities discovered.
  *Transformational*: new spaces altogether. Taylor argues that ordinary
  evolution gives only the first. The others need several domains of behaviour,
  and bridges between them.
- **Four necessary conditions** (Soros and Stanley 2014):
  1. reproduction must require meeting a minimal criterion;
  2. evolution must create new opportunities to meet it;
  3. individuals must decide for themselves how they interact;
  4. the representation must not limit how complex an individual can become.

  Graph of Life meets the first: a child costs at least one token. It meets
  the third more fully than most systems, since every action is a [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.")'s
  output. It meets the second only partly. And it does not meet the fourth at
  all, since a brain's size is fixed.
- **Unbounded evolution as a property of dynamical systems** (Adams, Zenil,
  Davies and Walker 2017). A system evolves without bound when its patterns do
  not recur within the time a closed system would take to repeat itself. In
  their models of evolving rules, only rules that depend on the system's own
  state produced open-ended behaviour in a way that scaled.
- **Size is not complexity** (Standish 2003). In a run of the Tierra system,
  organisms grew larger without becoming more complex. Growth in anything this
  notebook measures will have to be checked against that.

## Artificial worlds that came before

| system | what it is | what made it interesting |
|---|---|---|
| **Tierra** (Ray 1991) | self-copying machine code in a shared memory | parasites, hyper-parasites and cheats evolved within hours; genomes could grow |
| **Avida** (Ofria and Wilke 2004) | digital organisms that earn computing time by performing logic functions | showed how complex functions evolve through rewarded intermediate steps |
| **Polyworld** (Yaeger 1994) | creatures with evolved neural networks in a simulated ecology | evolved behaviours such as foraging and fleeing |
| **Echo** (Holland 1995) | agents trading resources by tags | the model on which evolutionary activity was first calibrated |
| **Geb** (Channon 2001) | neural-network agents that evolve and interact | argued, by activity statistics, to reach Bedau's unbounded class |
| **Chromaria** (Soros and Stanley 2014) | a world built to test the four conditions | the strictness of its minimal criterion could be set by hand |

Graph of Life differs from all of them in one respect: the agents rebuild the
**network** they live on, through every birth, handover and stake. It has no
reward of any kind. And unlike Tierra or Geb, its genomes cannot grow.

## The game: Colonel Blotto

In **Colonel Blotto** (Borel 1921), two commanders divide a fixed force
between several battlefields, without seeing each other's division. Each
battlefield goes to whoever sent more. Every fixed division can be beaten: give
up a little where the other is strong, and send more where it is weak. So there
is no single best division. Only a mixture of divisions, played at random, can
be optimal; Roberson (2006) worked out that mixture for two players.

The game of [One iteration, step by step](one-iteration.md#phase-2-the-game) is
a Blotto game with three differences that the theory does not cover:

1. there are many players;
2. each can reach only the battlefields of its neighbourhood, and they change;
3. what is won — tokens — is the next game's force, so success compounds.

Two more ideas belong here. Games without a best strategy can keep a
population moving. Rock–paper–scissors among three strains of *E. coli* keeps
all three alive when they meet **locally**, and only then (Kerr, Riley, Feldman
and Bohannan 2002). And the **Red Queen** (Van Valen 1973): when an organism's
surroundings are mostly other organisms, all of them changing, no improvement
buys a lasting advantage. A fixed supply of tokens fought over by neighbours is
such a setting.

In this world the game has not kept the population moving. Played against the
brains of earlier eras, the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.")'s brains stop improving within their first
hundred iterations. Later brains neither beat the ones before them nor go round
in a circle: they tie
([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)).
Experiment 16 found the same in [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death."), a world built so that it cannot break
([Why evolution stops](how-to-tell-a-world-that-keeps-evolving.md#why-evolution-stops)). Why is still open. It is not
that the game has been solved: a strategy written by hand took half of an
evolved world ([Strategies written by hand](strategies-written-by-hand.md)).
Evolution stops short of better play, and the better play found so far needs
several agents acting together.

## Networks

- **Small worlds** (Watts and Strogatz 1998). A ring of nodes, each joined to
  its nearest few, with a few random shortcuts, is both clustered and short.
  It is the [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")' ring of [The world: tokens, agents and a network](the-world.md).
- **Measures of a network** — degree, clustering, distances, bridges, the
  2-core — are standard (Newman 2003). [Degree](degree.md),
  [Clustering](clustering.md), [Path length](path-length.md),
  [Bridges](bridges.md) and [Core, trees and leaves](core-trees-leaves.md)
  define each. The bridges are found by depth-first search (Tarjan 1974), the
  core by peeling (Seidman 1983).
- **Evolution on graphs.** Where individuals sit in a network changes what
  selection does. On a lattice, cooperators survive in clusters, where in a
  mixed population they die out (Nowak and May 1992). Some graphs amplify
  selection, and others suppress it (Lieberman, Hauert and Nowak 2005).
  Cooperation spreads when its benefit, divided by its cost, exceeds the
  average number of neighbours (Ohtsuki, Hauert, Lieberman and Nowak 2006). In
  these results the graph is given. In Graph of Life the agents build it.

## Genealogy and chance

- **Neutral evolution** (Kimura 1968): many changes spread not because they
  are better, but by chance. The discipline it taught: a story of selection
  has to beat a model of chance alone.
- **The Moran model** (Moran 1958): in a population of fixed size, one
  individual at a time is copied and one replaced, at random. Even with no
  differences between them, one lineage eventually takes over.
- **The coalescent** (Kingman 1982): followed backwards, the lineages of the
  living merge, two by two, until all meet in one **[common ancestor](common-ancestor.md "The genotype every living agent descends from, and how far back it lies.")**. In a
  population of fixed size *N*, replaced by chance alone, that ancestor lived a
  number of generations back proportional to *N*.
  [Genotypes and lineages](genotypes-and-lineages.md) finds the common ancestor
  of each world's living agents. [Do the brains matter?](do-the-brains-matter.md)
  compares its movements with those in a world where no brain is better than
  another.
- **Muller plots** (after Muller 1932) draw, stacked over time, the share of a
  population that descends from each of several ancestors. A lineage taking
  over appears as a band that widens until it fills the picture
  ([The first hundred iterations](a-worlds-life.md#the-first-hundred-iterations),
  [Genotypes and lineages](genotypes-and-lineages.md); [Muller plots](muller-plot.md)).

## Measuring

- **Inequality**: the **Lorenz curve** (Lorenz 1905) and the **Gini
  coefficient** (Gini 1912): [Where do the tokens go?](where-do-the-tokens-go.md)
  and [The Gini coefficient and the Lorenz curve](gini-coefficient.md).
- **[Lifetimes](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death.") with censoring**: the Kaplan–Meier estimate (Kaplan and Meier
  1958): [Survival curves](kaplan-meier.md).
- **Intervals by resampling**: the [bootstrap](bootstrap.md "How sure a number is, found by redrawing the worlds it came from many times.") (Efron 1979). **Intervals for
  proportions**: Wilson (1927).
- **[Correlations](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of a series with itself**, and the bias of measuring them
  against the series' own mean: Bartlett (1946), Marriott and Pope (1954),
  Kendall (1954). See [Autocorrelation](autocorrelation.md).
- **[Power laws](power-law-fit.md "A quantity whose frequency falls as a power of its size: few huge values, many small ones.")**, and why a straight stretch on log–log axes is weak evidence
  of one: Clauset, Shalizi and Newman (2009). See
  [Logarithmic axes](logarithmic-axes.md).
- **Testing by shuffling**: the permutation test goes back to Fisher (1935).
  See [Permutation tests](permutation-test.md).
- **Pictures of networks**: the ForceAtlas2 layout (Jacomy, Venturini, Heymann
  and Bastian 2014) places joined nodes near each other.
- **Progress against the past**: compare a population with its own ancestors,
  brought back to compete. That shows whether it is getting better, or only
  going round in circles (Cliff and Miller 1995). The standing example is the
  Lenski experiment: twelve *E. coli* populations followed for tens of
  thousands of generations, with frozen samples of their ancestors (Lenski,
  Rose, Simpson and Tadler 1991; Blount, Borland and Lenski 2008). It is the
  test for the second rung of the ladder in
  [What this notebook is about](what-this-notebook-is-about.md).
  [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)
  runs it on the baseline.

## Where to go next

- [What this notebook is about](what-this-notebook-is-about.md): the ladder of
  four claims these ideas are tested against.
- [Eras and contests: does a world keep getting better?](eras-and-contests.md):
  the test of bringing ancestors back, as this notebook plays it.
- [Why evolution stops](how-to-tell-a-world-that-keeps-evolving.md#why-evolution-stops): what the game and the rules
  do to evolution here.

## How our view changed

- **14 October 2026.** A note added here said that, like Blotto, the game has
  an answer: play unpredictably. Evolution finds it quickly, the note said,
  using the brains' random inputs as dice, and that is why the baseline stops
  improving ([Why evolution stops](how-to-tell-a-world-that-keeps-evolving.md#why-evolution-stops)).
- **15 October 2026.** The reason did not survive the tests that followed.
  Random founder brains already roll those dice, and a strategy written by hand
  took half of an evolved world
  ([Strategies written by hand](strategies-written-by-hand.md)). The game is
  not solved. The text above now says what is known.

## References

- Adams, A., Zenil, H., Davies, P. C. W. and Walker, S. I. (2017). Formal definitions of unbounded evolution and innovation reveal universal mechanisms for open-ended evolution in dynamical systems. *Scientific Reports* 7, 997.
- Bartlett, M. S. (1946). On the theoretical specification and sampling properties of autocorrelated time-series. *Supplement to the Journal of the Royal Statistical Society* 8(1), 27–41.
- Bedau, M. A. and Packard, N. H. (1992). Measurement of evolutionary activity, teleology, and life. In *Artificial Life II*, 431–461. Addison-Wesley.
- Bedau, M. A., Snyder, E., Brown, C. T. and Packard, N. H. (1997). A comparison of evolutionary activity in artificial evolving systems and in the biosphere. In *Proceedings of the Fourth European Conference on Artificial Life*, 125–134. MIT Press.
- Bedau, M. A., Snyder, E. and Packard, N. H. (1998). A classification of long-term evolutionary dynamics. In *Artificial Life VI*, 228–237. MIT Press.
- Blount, Z. D., Borland, C. Z. and Lenski, R. E. (2008). Historical contingency and the evolution of a key innovation in an experimental population of *Escherichia coli*. *PNAS* 105(23), 7899–7906.
- Borel, É. (1921). La théorie du jeu et les équations intégrales à noyau symétrique. *Comptes Rendus de l'Académie des Sciences* 173, 1304–1308.
- Clauset, A., Shalizi, C. R. and Newman, M. E. J. (2009). Power-law distributions in empirical data. *SIAM Review* 51(4), 661–703.
- Channon, A. (2001). Passing the ALife test: activity statistics classify evolution in Geb as unbounded. In *Advances in Artificial Life (ECAL 2001)*, LNCS 2159, 417–426. Springer.
- Cliff, D. and Miller, G. F. (1995). Tracking the Red Queen: measurements of adaptive progress in co-evolutionary simulations. In *Advances in Artificial Life (ECAL 1995)*, LNCS 929, 200–218. Springer.
- Dolson, E. L., Vostinar, A. E., Wiser, M. J. and Ofria, C. (2019). The MODES toolbox: measurements of open-ended dynamics in evolving systems. *Artificial Life* 25(1), 50–73.
- Efron, B. (1979). Bootstrap methods: another look at the jackknife. *The Annals of Statistics* 7(1), 1–26.
- Fisher, R. A. (1935). *The Design of Experiments*. Edinburgh: Oliver and Boyd.
- Gini, C. (1912). *Variabilità e mutabilità*. Bologna: Cuppini.
- Holland, J. H. (1995). *Hidden Order: How Adaptation Builds Complexity*. Addison-Wesley.
- Jacomy, M., Venturini, T., Heymann, S. and Bastian, M. (2014). ForceAtlas2, a continuous graph layout algorithm for handy network visualization designed for the Gephi software. *PLoS ONE* 9(6), e98679.
- Kaplan, E. L. and Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association* 53(282), 457–481.
- Kendall, M. G. (1954). Note on bias in the estimation of autocorrelation. *Biometrika* 41(3–4), 403–404.
- Kerr, B., Riley, M. A., Feldman, M. W. and Bohannan, B. J. M. (2002). Local dispersal promotes biodiversity in a real-life game of rock–paper–scissors. *Nature* 418, 171–174.
- Kimura, M. (1968). Evolutionary rate at the molecular level. *Nature* 217, 624–626.
- Kingman, J. F. C. (1982). The coalescent. *Stochastic Processes and their Applications* 13(3), 235–248.
- Lenski, R. E., Rose, M. R., Simpson, S. C. and Tadler, S. C. (1991). Long-term experimental evolution in *Escherichia coli*. I. Adaptation and divergence during 2,000 generations. *The American Naturalist* 138(6), 1315–1341.
- Lieberman, E., Hauert, C. and Nowak, M. A. (2005). Evolutionary dynamics on graphs. *Nature* 433, 312–316.
- Lorenz, M. O. (1905). Methods of measuring the concentration of wealth. *Publications of the American Statistical Association* 9(70), 209–219.
- Marriott, F. H. C. and Pope, J. A. (1954). Bias in the estimation of autocorrelations. *Biometrika* 41(3–4), 390–402.
- Moran, P. A. P. (1958). Random processes in genetics. *Mathematical Proceedings of the Cambridge Philosophical Society* 54(1), 60–71.
- Muller, H. J. (1932). Some genetic aspects of sex. *The American Naturalist* 66(703), 118–138.
- Newman, M. E. J. (2003). The structure and function of complex networks. *SIAM Review* 45(2), 167–256.
- Nowak, M. A. and May, R. M. (1992). Evolutionary games and spatial chaos. *Nature* 359, 826–829.
- Ofria, C. and Wilke, C. O. (2004). Avida: a software platform for research in computational evolutionary biology. *Artificial Life* 10(2), 191–229.
- Ohtsuki, H., Hauert, C., Lieberman, E. and Nowak, M. A. (2006). A simple rule for the evolution of cooperation on graphs and social networks. *Nature* 441, 502–505.
- Ray, T. S. (1991). An approach to the synthesis of life. In *Artificial Life II*, 371–408. Addison-Wesley.
- Roberson, B. (2006). The Colonel Blotto game. *Economic Theory* 29(1), 1–24.
- Seidman, S. B. (1983). Network structure and minimum degree. *Social Networks* 5(3), 269–287.
- Soros, L. B. and Stanley, K. O. (2014). Identifying necessary conditions for open-ended evolution through the artificial life world of Chromaria. In *Artificial Life 14*, 793–800. MIT Press.
- Standish, R. K. (2003). Open-ended artificial evolution. *International Journal of Computational Intelligence and Applications* 3(2), 167–175.
- Tarjan, R. E. (1974). A note on finding the bridges of a graph. *Information Processing Letters* 2(6), 160–161.
- Taylor, T. (2019). Evolutionary innovations and where to find them: routes to open-ended evolution in natural and artificial systems. *Artificial Life* 25(2), 207–224.
- Van Valen, L. (1973). A new evolutionary law. *Evolutionary Theory* 1, 1–30.
- Watts, D. J. and Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature* 393, 440–442.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association* 22(158), 209–212.
- Yaeger, L. (1994). Computational genetics, physiology, metabolism, neural systems, learning, vision, and behavior or PolyWorld: life in a new context. In *Artificial Life III*, 263–298. Addison-Wesley.

<!-- turns -->
---

← [How this notebook is written](how-this-notebook-is-written.md) · [Start here](../README.md#start-here)

**Also about:** [Does evolution keep going?](../README.md#does-evolution-keep-going).

> [!info]- Pages that link here (6)
> [What this notebook is about](what-this-notebook-is-about.md) · [One iteration, step by step](one-iteration.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [Genotypes and lineages](genotypes-and-lineages.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md)
<!-- /turns -->
