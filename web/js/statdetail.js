/*
 * Clicking a statistic opens this: what the number means, and how it moved
 * across the whole run.
 *
 * The history comes from /api/runs/<id>/series, which reduces every frame to a
 * handful of scalars on the server. Pulling thousands of full frames into the
 * browser just to plot one line would be far slower.
 *
 * The plotted points respect the active phase filter, so looking at the game
 * phases alone gives a curve of game phases alone rather than a sawtooth
 * alternating between two different kinds of moment.
 */
/**
 * Round numbers to rule a grid at.
 *
 * A grid drawn at whatever the data happens to span gives labels like 0.0473,
 * which are noise to read. This steps up to the nearest 1, 2, 2.5 or 5 times a
 * power of ten, so the lines land on values worth putting a number against.
 */
function niceTicks(lo, hi, target = 5) {
  if (!(hi > lo)) return { ticks: [lo], step: 1 };

  const rough = (hi - lo) / Math.max(1, target);
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const scaled = rough / magnitude;
  const step = magnitude *
    (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 2.5 ? 2.5 : scaled <= 5 ? 5 : 10);

  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) {
    // Repeated addition drifts; snap a near-zero tick to exactly zero.
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return { ticks, step };
}

const StatDetail = {
  seriesCache: new Map(),   // runId -> series payload
  currentKey: null,

  EXPLANATIONS: {
    nodes: 'How many agents are alive right now. Agents appear by reproduction and vanish by starving, or by being cut off from the largest connected group and culled with it.',
    edges: 'How many connections exist. An edge is created when a newborn is wired into its parent\u2019s neighbourhood, and pruned at the end of a game phase if it carried no tokens through it.',
    tokens: 'Total tokens in the world. Tokens are conserved by default, so a flat line is the expected behaviour and any drift would mean a bug. It moves only if the run was configured to mint tokens each phase.',
    meanDegree: 'The average number of connections an agent has: twice the number of connections divided by the number of agents, since every connection is counted at both of its ends.',
    maxDegree: 'The number of connections held by the single most connected agent \u2014 the size of the largest hub in the population.',
    medianTokens: 'How many tokens the middle agent holds, when all agents are lined up from poorest to richest. Half the population holds less than this and half holds more. It sits far below the average whenever a few very rich agents are pulling that average up, so the gap between the two is itself a reading of how lopsided wealth has become.',
    maxTokens: 'How many tokens the single richest agent holds. Read against the size of the whole pile it says how much of the world one agent has managed to gather, and a line that climbs while the population is flat is wealth moving rather than wealth appearing, since tokens are conserved by default.',
    gini: 'How concentrated wealth is, on a scale from 0 to 1. Zero means every agent holds exactly the same number of tokens. One means a single agent holds everything and the rest hold nothing. A rising line means tokens are pooling into fewer hands.',
    distinctBrains: 'How many different genomes are present in the population. A brain id names a genotype: a copy keeps the id of the brain it came from and only a mutation makes a new one, so two agents carrying identical weights count once between them.',
    brainDiversity: 'The number of different genomes present, divided by the number of agents. One means no two agents share a genome. A low value means a single lineage has been copied across much of the population and taken it over.',
    births: 'How many agents reproduced this phase, each spending part of its own tokens to do so.',
    revolutions: 'How many nodes were taken by a coalition rather than by the single largest allocator. A revolution succeeds when the accumulated weaker allocators outweigh everyone above them plus the current holder. Only recorded on runs created with revolutions enabled; an absent value means the rule was off and nodes simply went to whoever allocated the most, which is not the same as the rule being on and never firing.',
    starved: 'How many agents were removed for holding zero tokens at the end of the phase.',
    orphaned: 'How many agents were removed for sitting outside the largest connected group. They may have been perfectly wealthy \u2014 they were simply cut off from everyone else, and cleanup keeps only the largest group.',
    loops: 'How many of the graph\u2019s independent loops pass through this element. Counting every loop through a node is intractable, so what is counted is a basis: take a spanning tree of the graph, and each connection left over closes exactly one loop, with those loops generating all the others. The tree is built breadth-first so the loops come out short and local. A different tree would give a different basis, so read this as a fair sample of the loop structure rather than a canonical count.',
    cycleRank: 'How many independent loops the graph contains. Counting loops one at a time is hopeless, since their number grows exponentially, but the independent count is exact and cheap: take a spanning tree, and every connection left over closes exactly one new loop \u2014 connections minus agents plus the number of separate pieces. Zero means the graph is a tree with no loops anywhere in it.',
    loopDensity: 'What share of connections are redundant, in the sense that removing one would not split the graph. Computed as the number of independent loops divided by the number of connections, where the independent loop count is connections minus agents plus the number of separate pieces. A low value means a spindly, tree-like graph where almost every connection is load-bearing; a high one means a densely interwoven graph with slack in it.',
    bridges: 'How many connections lie on no loop at all. Removing a bridge splits the graph into two pieces, so these are the connections holding otherwise separate regions together. That makes them the dangerous ones here: a connection that carries no tokens through a game phase is pruned, and if it was a bridge, everything on the smaller side is cut off and culled.',
    cutRisk: 'The largest share of the population that a single connection could cut off. Some connections lie on no loop, so removing one splits the graph into a smaller side and a larger one; this is the biggest such smaller side there is, as a fraction of everyone alive. It matters because cleanup keeps only the largest piece: a lone connection with a tenth of the world behind it is a tenth-of-the-world extinction waiting for the zero-flow prune to reach that one edge. Simply counting such connections cannot tell a dumbbell of two halves from a blob with three leaves stuck on it, though only one of those is fragile. Zero means no single connection costs anything; the ceiling is 50%, a graph split down the middle by one connection.',
    cutRiskBefore: 'The largest share of the population a single connection could cut off, measured on the graph as it stood before this phase removed anybody. Some connections lie on no loop, so removing one splits the graph in two; this is the biggest smaller side any of them has, as a fraction of everyone. Measuring it beforehand is the whole point. A recorded frame shows the graph after the cull, so a cut that severs a whole side changes both the graph and the death toll in the same row, and cause cannot be told from consequence afterwards. Taken before the cull, it is a state the cull has not touched and can be read as a predictor of one. Runs made before the engine started recording this simply do not have it.',
    spectralGap: 'How hard the population is to cut in two, as a single number: the second-smallest eigenvalue of the normalised Laplacian, computed on the largest connected group. Near zero means there is a cheap cut somewhere \u2014 two halves joined by very little. Away from zero means every way of splitting the graph is expensive, which is what an expander is, and is close to the most a graph this sparse can manage. Cheeger\u2019s inequality ties this to the sparsest cut from both sides, so a small value promises that a good cut exists and a large one proves that none does. That makes it the check on any claim that the population has divided into communities: methods that look for communities always return something, and this is what distinguishes a real division from a line drawn through an indivisible graph. Found by Lanczos iteration, which is what makes it trustworthy on a crowded spectrum \u2014 plain power iteration on these graphs returned a value that halved every time the iteration count doubled, and so was measuring the iteration count.',
    lightningScore: 'During a game phase agents send tokens along their links. A lightning is a closed loop of that flow: tokens leaving an agent, crossing links, and arriving back where they started, like current round a circuit. A loop of L hops carries L tokens, one per edge, and scores L squared, so one token round a triangle is a small thing and ten tokens round a ten-edge loop is a large one. Every token belongs to at most one lightning. The loops are found greedily \u2014 walk the flow until an agent turns up twice, which closes the circle, take that loop out and go again, preferring steps to agents not yet in the walk so that long loops are found rather than the first triangle stumbled into. This entry is the total score, summed over every loop found in this phase, counting the flow exactly as it was sent. It is a lower bound and cannot be anything else: maximising the total rewards packing flow into few long loops, which contains the problem of finding the longest cycle, which is the Hamiltonian cycle problem in disguise and has no efficient exact answer.',
    cyclingShare: 'During a game phase agents send tokens along their links. A lightning is a closed loop of that flow: tokens leaving an agent, crossing links, and arriving back where they started, like current round a circuit. A loop of L hops carries L tokens, one per edge, and scores L squared, so one token round a triangle is a small thing and ten tokens round a ten-edge loop is a large one. Every token belongs to at most one lightning. The loops are found greedily \u2014 walk the flow until an agent turns up twice, which closes the circle, take that loop out and go again, preferring steps to agents not yet in the walk so that long loops are found rather than the first triangle stumbled into. This entry is the share of all moved tokens that the search managed to place in some closed loop, counting the flow exactly as it was sent. It travels better between runs than the raw score does, because it does not grow with the size of the world. Being greedy, it is a lower bound on how much circulation is really there.',
    lightningLongest: 'During a game phase agents send tokens along their links. A lightning is a closed loop of that flow: tokens leaving an agent, crossing links, and arriving back where they started, like current round a circuit. A loop of L hops carries L tokens, one per edge, and scores L squared, so one token round a triangle is a small thing and ten tokens round a ten-edge loop is a large one. Every token belongs to at most one lightning. The loops are found greedily \u2014 walk the flow until an agent turns up twice, which closes the circle, take that loop out and go again, preferring steps to agents not yet in the walk so that long loops are found rather than the first triangle stumbled into. This entry is the number of hops in the longest single loop found this phase, counting the flow exactly as it was sent. A long lightning means tokens travelling a long way round the graph and returning, which says far more about structure than many short ones do: a triangle can happen by accident, a forty-hop circuit cannot.',
    flowImbalance: 'The share of this phase\u2019s token flow that conservation forbids from ever circulating. Every token that goes round a closed loop leaves its agent and comes back to it, so a loop moves nobody\u2019s net balance. Whatever net imbalance is left over \u2014 agents that received more than they sent, or sent more than they received \u2014 is one-way transport that no decomposition could bend into a circle. This is exact, with no heuristic in it, and it is a ceiling: the share of flow that circulates can never exceed one minus this. Because the searches that look for circulation are greedy and so report floors, this ceiling is what says whether such a floor is close to the truth or a weak reading of it.',
    netLightningScore: 'During a game phase agents send tokens along their links. A lightning is a closed loop of that flow: tokens leaving an agent, crossing links, and arriving back where they started, like current round a circuit. A loop of L hops carries L tokens, one per edge, and scores L squared, so one token round a triangle is a small thing and ten tokens round a ten-edge loop is a large one. Every token belongs to at most one lightning. The loops are found greedily \u2014 walk the flow until an agent turns up twice, which closes the circle, take that loop out and go again, preferring steps to agents not yet in the walk so that long loops are found rather than the first triangle stumbled into. Everything here is measured on the net flow, where reciprocal amounts are cancelled off first: if one agent sends five tokens to a neighbour and the neighbour sends three back, that is two tokens one way and nothing back. Every link then carries flow in at most one direction, so a two-hop loop cannot exist at all and every lightning left is tokens genuinely going round rather than sloshing between neighbours. That is the stricter claim: mutual exchange between neighbours is easy and everywhere, a circuit that survives cancellation is not. This entry is the total score over the net flow. On a test case of a ten-hop ring buried in reciprocal noise, the score before cancelling is 500 and this is 100, which is the ring and nothing else.',
    netCyclingShare: 'During a game phase agents send tokens along their links. A lightning is a closed loop of that flow: tokens leaving an agent, crossing links, and arriving back where they started, like current round a circuit. A loop of L hops carries L tokens, one per edge, and scores L squared, so one token round a triangle is a small thing and ten tokens round a ten-edge loop is a large one. Every token belongs to at most one lightning. The loops are found greedily \u2014 walk the flow until an agent turns up twice, which closes the circle, take that loop out and go again, preferring steps to agents not yet in the walk so that long loops are found rather than the first triangle stumbled into. Everything here is measured on the net flow, where reciprocal amounts are cancelled off first: if one agent sends five tokens to a neighbour and the neighbour sends three back, that is two tokens one way and nothing back. Every link then carries flow in at most one direction, so a two-hop loop cannot exist at all and every lightning left is tokens genuinely going round rather than sloshing between neighbours. That is the stricter claim: mutual exchange between neighbours is easy and everywhere, a circuit that survives cancellation is not. This entry is the share of net-flow tokens that ended up in a closed loop, measured against the net total rather than the gross one \u2014 so it answers "of the tokens that actually went somewhere, how many went round" instead of being diluted by the reciprocity that cancelling just removed.',
    netLightningLongest: 'During a game phase agents send tokens along their links. A lightning is a closed loop of that flow: tokens leaving an agent, crossing links, and arriving back where they started, like current round a circuit. A loop of L hops carries L tokens, one per edge, and scores L squared, so one token round a triangle is a small thing and ten tokens round a ten-edge loop is a large one. Every token belongs to at most one lightning. The loops are found greedily \u2014 walk the flow until an agent turns up twice, which closes the circle, take that loop out and go again, preferring steps to agents not yet in the walk so that long loops are found rather than the first triangle stumbled into. Everything here is measured on the net flow, where reciprocal amounts are cancelled off first: if one agent sends five tokens to a neighbour and the neighbour sends three back, that is two tokens one way and nothing back. Every link then carries flow in at most one direction, so a two-hop loop cannot exist at all and every lightning left is tokens genuinely going round rather than sloshing between neighbours. That is the stricter claim: mutual exchange between neighbours is easy and everywhere, a circuit that survives cancellation is not. This entry is the number of hops in the longest single loop of net flow, and it is the strongest single statement this analysis makes. A long circuit that survives cancelling every reciprocal pair is tokens travelling a long way round the graph and coming back, which cannot happen by accident.',
    netFlowShare: 'How much of this phase\u2019s token flow survives cancelling reciprocal amounts against each other. If one agent sends five tokens to a neighbour and the neighbour sends three back, only two of those eight are really going anywhere, and the other six cancel. A low value means neighbours are mostly trading back and forth; a high one means most flow has a direction. Cancelling this way cannot change any agent\u2019s net balance, since it removes equal and opposite amounts, so anything that bounds circulation by way of net balances bounds it identically before and after.',
    coreShare: 'What fraction of agents survive repeatedly deleting everyone who has only a single connection, until nobody has one. What is left is the 2-core: the part of the graph where every agent sits on some loop. What gets peeled away is the fringe of tree-shaped whiskers hanging off it. A high share means a solidly interwoven population; a low one means most agents dangle off a comparatively small centre and are cheap to lose.',
    triangles: 'How many closed triples of mutually connected agents there are \u2014 the shortest loop it is possible to have. Unlike loops in general these can be counted exactly and quickly, because the triangles sitting on a connection are just the neighbours its two ends have in common.',
    transitivity: 'The chance that two agents connected to the same third agent are also connected to each other, from 0 to 1. Computed over the whole graph as three times the number of triangles divided by the number of connected triples. A high value means tight local cliques where a neighbourhood closes back on itself; a low one means a sparse, tree-like structure with little local redundancy.',
    dimension: 'How many dimensions the graph behaves as though it has, in the spirit of the Wolfram Physics Project. Measured by walking outward from an agent and counting how many agents sit exactly r steps away, averaged over many starting points. In flat space of d dimensions that shell grows as r to the power d minus one, so the growth curve carries the geometry. Radii whose ball has already swallowed half the graph are discarded, since past that the growth is measuring the boundary rather than the shape. The slope of log shell against log radius, plus one, estimates the dimension. Checked against lattices of known dimension it returns 1.00 for a chain, 1.92 for a square grid and 2.56 for a cubic one: exact in one dimension and increasingly conservative above it, because a few steps is not enough room for the growth to reach its limit. Read it as a rough index rather than a precise figure. A small-world graph has no honest dimension at all, since its neighbourhoods grow exponentially rather than as a power, and that shows up here as a large or unsteady number \u2014 a real finding about the graph rather than a broken measurement.',
    ricciCurvature: 'The Ricci scalar of the graph: whether neighbourhoods hold more or less than flat space allows. Measured by walking outward from an agent and counting how many agents sit exactly r steps away, averaged over many starting points. In flat space of d dimensions that shell grows as r to the power d minus one, so the growth curve carries the geometry. Radii whose ball has already swallowed half the graph are discarded, since past that the growth is measuring the boundary rather than the shape. In flat space that growth is exactly a power of r; in curved space the curve bends, and the bend is the curvature rather than noise. A ball in d dimensions with Ricci scalar R has a shell going as r to the power d minus one, times one minus R r squared over 6d, so fitting log shell against both log radius and radius squared recovers R from the second coefficient. Negative means neighbourhoods hold more than flat space allows, which is what a graph that keeps branching gives \u2014 a tree is strongly negative. Positive means they close back on themselves. It also reads as how expander-like the graph is, since sparse expanders are negatively curved as a theorem. Needs at least four usable radii: three points would fit the three parameters exactly and return a number with no evidence in it, so below that it is left blank.',
    diameter: 'The longest shortest path in the graph: take the two agents furthest apart, and this is how many steps separate them. The worst case for anything travelling across the population. An exact answer needs the shortest path between every pair, far too slow to do per frame, so a spread of starting points is swept and then swept again from the furthest agent any of them reached \u2014 the double-sweep trick, which is exact on a tree. It can only ever under-report, since a pair further apart may not have been sampled. Checked against known graphs it is exact on paths, cycles, stars, grids and trees, and about one step short on small-world graphs of the kind this simulation makes.',
    radius: 'How central the most central agent manages to be: the fewest steps from which the whole population can be reached. Formally the smallest eccentricity in the graph, where an agent\u2019s eccentricity is its distance to the agent furthest from it. Estimated from sampled sweeps rather than computed exactly, which would need the shortest path between every pair. It can only ever over-report, since some agent that was never tried may be more central than any that was.',
    meanPathLength: 'The average number of steps between two agents, over the pairs actually sampled. Computing it over every pair would be far too slow per frame, so a spread of starting points is swept and averaged. Because it averages over many pairs rather than reporting a single extreme, it is steady from frame to frame, which makes it the more reliable reading of how tightly connected the population is.',
    degreeEntropy: 'How much variety there is in how connected agents are, measured in bits as the Shannon entropy of the distribution of connection counts. Zero means every agent has exactly the same number of connections. Higher values mean a broader mix of heavily connected hubs and sparsely connected leaves.',
    degreeEvenness: 'How evenly the different connection counts are represented, as a share of the most even they could be. The underlying measure is the Shannon entropy of how many connections agents have, and the ceiling is the value it would reach if every connection count that occurs at all occurred equally often. One means exactly that: no particular degree is more common than another.',
    tokenEntropy: 'How evenly wealth is spread, measured in bits, by treating each agent\u2019s share of the tokens as a probability and taking the Shannon entropy. It reaches its highest value, the base-2 logarithm of the population, when everybody holds the same amount, and falls as wealth concentrates. Because that ceiling depends on how many agents there are, the raw number moves when the population changes even if the spread of wealth does not.',
    tokenEvenness: 'How evenly wealth is spread, as a percentage of the most even it could possibly be. The underlying measure is the Shannon entropy of wealth \u2014 each agent\u2019s share of the tokens treated as a probability \u2014 which peaks when everyone holds the same amount. Dividing by that peak removes the dependence on population size, so 100% means perfectly equal wealth whether there are ten agents or ten thousand, and this can be compared between frames as the population grows.',
    components: 'How many separate pieces the graph is in, where a piece is a set of agents that can reach each other by following connections. Cleanup keeps only the largest piece and culls the rest, so a recorded frame should always read 1; anything else would mean the culling step had missed something.',
    degreeGamma: 'The scale-free estimate asks whether the number of connections agents have follows a power law above some threshold, and it is done the way the literature does it rather than by fitting a line through a histogram. The exponent is estimated by maximum likelihood, because regression on log-log axes is biased and the bias is worst out in the tail where the hubs are. The threshold is found rather than assumed: every candidate is tried and the one whose fitted curve sits closest to the data wins, closest meaning the smallest Kolmogorov-Smirnov distance, the largest gap between the observed curve and the fitted one. The estimate reports several numbers together \u2014 the exponent, where the tail was judged to start, what fraction of agents fall in that tail, how much of their scatter the fitted curve accounts for, and how far it still misses. This entry is the exponent. Grown networks usually land between 2 and 3. Below 2 the average number of connections stops settling down as the graph grows, which is a claim about the network rather than a measurement artefact.',
    degreeGammaR2: 'The scale-free estimate asks whether the number of connections agents have follows a power law above some threshold, and it is done the way the literature does it rather than by fitting a line through a histogram. The exponent is estimated by maximum likelihood, because regression on log-log axes is biased and the bias is worst out in the tail where the hubs are. The threshold is found rather than assumed: every candidate is tried and the one whose fitted curve sits closest to the data wins, closest meaning the smallest Kolmogorov-Smirnov distance, the largest gap between the observed curve and the fitted one. The estimate reports several numbers together \u2014 the exponent, where the tail was judged to start, what fraction of agents fall in that tail, how much of their scatter the fitted curve accounts for, and how far it still misses. This entry is the R\u00b2 of that fit, the share of the scatter the fitted curve accounts for, measured over the fitted tail alone rather than the whole distribution. That restriction is why it needs care: 99% over the top half a percent of agents is a tight fit to very few of them.',
    degreeKMin: 'The scale-free estimate asks whether the number of connections agents have follows a power law above some threshold, and it is done the way the literature does it rather than by fitting a line through a histogram. The exponent is estimated by maximum likelihood, because regression on log-log axes is biased and the bias is worst out in the tail where the hubs are. The threshold is found rather than assumed: every candidate is tried and the one whose fitted curve sits closest to the data wins, closest meaning the smallest Kolmogorov-Smirnov distance, the largest gap between the observed curve and the fitted one. The estimate reports several numbers together \u2014 the exponent, where the tail was judged to start, what fraction of agents fall in that tail, how much of their scatter the fitted curve accounts for, and how far it still misses. This entry is where the tail was judged to begin \u2014 the smallest number of connections above which the power law is claimed to hold. A threshold that climbs over a run means the distribution is only becoming straight further and further out.',
    degreeTailShare: 'The scale-free estimate asks whether the number of connections agents have follows a power law above some threshold, and it is done the way the literature does it rather than by fitting a line through a histogram. The exponent is estimated by maximum likelihood, because regression on log-log axes is biased and the bias is worst out in the tail where the hubs are. The threshold is found rather than assumed: every candidate is tried and the one whose fitted curve sits closest to the data wins, closest meaning the smallest Kolmogorov-Smirnov distance, the largest gap between the observed curve and the fitted one. The estimate reports several numbers together \u2014 the exponent, where the tail was judged to start, what fraction of agents fall in that tail, how much of their scatter the fitted curve accounts for, and how far it still misses. This entry is the fraction of agents lying in the fitted tail, and it is the honesty check on the whole estimate: a scale-free fit covering half a percent of the population is describing the handful of hubs, not the network. A large share means the power law reaches down into the ordinary agents.',
    degreeGammaKS: 'The scale-free estimate asks whether the number of connections agents have follows a power law above some threshold, and it is done the way the literature does it rather than by fitting a line through a histogram. The exponent is estimated by maximum likelihood, because regression on log-log axes is biased and the bias is worst out in the tail where the hubs are. The threshold is found rather than assumed: every candidate is tried and the one whose fitted curve sits closest to the data wins, closest meaning the smallest Kolmogorov-Smirnov distance, the largest gap between the observed curve and the fitted one. The estimate reports several numbers together \u2014 the exponent, where the tail was judged to start, what fraction of agents fall in that tail, how much of their scatter the fitted curve accounts for, and how far it still misses. This entry is the Kolmogorov-Smirnov distance that remains after the best threshold was chosen \u2014 the largest gap between the observed tail and the fitted curve. Smaller is better. Being a distance rather than a share, it can be compared between frames whose tails hold different numbers of agents.',
    boxDimension: 'The fractal dimension of the graph, from box covering, and roughly speaking how many directions the network has. The graph is covered with boxes of size l, where no two agents in a box are more than l steps apart, and the number of boxes needed is counted. A self-similar network needs a count that falls as a power of the box size, and that power is the dimension. Near 1 the network is chain-like, near 2 sheet-like. Covering a graph exactly is NP-hard, so boxes are grown greedily around the most connected agents, which makes this an upper bound on the box count and so a lower bound on how efficiently the graph can be covered. This entry is the exponent itself; the fit also reports how well the counts really follow a power of box size, which has to be checked before the exponent means anything.',
    boxDimensionR2: 'Whether the graph really is fractal, or whether its box counts merely fall. The graph is covered with boxes of size l, where no two agents in a box are more than l steps apart; a self-similar network needs the number of boxes to fall as a power of l, and the exponent of that power is its fractal dimension. This entry is the R\u00b2 of that fit \u2014 how much of the scatter the straight line on log-log axes accounts for. It matters more here than for most fits, because small-world networks shrink exponentially rather than fractally, and an exponential curve still looks respectable on log-log axes over a short range. A low value means the fractal dimension is not measuring anything real.',
    degreeExponent: 'The exponent of the distribution of how many connections agents have. Fitted from the complementary CDF \u2014 the fraction of agents with at least k connections \u2014 because a binned histogram is worthless out in the tail where each bin holds one or two agents. If the probability of having k connections falls as k to the minus gamma, that fraction falls as k to the minus gamma minus one, so the exponent comes straight off the slope. Grown networks typically land between 2 and 3, and the classic preferential-attachment model sits at exactly 3; below 2 the average number of connections stops settling down as the graph grows. Fitted as a straight line on log-log axes, which turns a power law into a slope. Such a fit returns two numbers: the exponent, which is the slope itself, and R\u00b2, the share of the scatter the line accounts for. A high R\u00b2 is not proof of a power law \u2014 log-normal and stretched-exponential curves look just as straight over two decades \u2014 so read it as whether this exponent means anything, not as whether the relationship is truly scale free. This entry is the exponent.',
    degreeExponentR2: 'How well the distribution of connection counts really follows a power law. The fit is taken on the complementary CDF \u2014 the fraction of agents with at least k connections \u2014 because a binned histogram is worthless out in the tail where each bin holds one or two agents. Fitted as a straight line on log-log axes, which turns a power law into a slope. Such a fit returns two numbers: the exponent, which is the slope itself, and R\u00b2, the share of the scatter the line accounts for. A high R\u00b2 is not proof of a power law \u2014 log-normal and stretched-exponential curves look just as straight over two decades \u2014 so read it as whether this exponent means anything, not as whether the relationship is truly scale free. This entry is the R\u00b2. Treat it as answering "is this exponent meaningful", not "is this network scale free".',
    tokenExponent: 'The exponent of the wealth distribution: how quickly the number of agents holding at least so many tokens falls away as that threshold rises. A Pareto-like tail means most tokens sit with few agents, and the smaller the exponent the more extreme that is. Because tokens are conserved by default, this measures how one fixed pile gets shared out rather than how much of it there is. Fitted as a straight line on log-log axes, which turns a power law into a slope. Such a fit returns two numbers: the exponent, which is the slope itself, and R\u00b2, the share of the scatter the line accounts for. A high R\u00b2 is not proof of a power law \u2014 log-normal and stretched-exponential curves look just as straight over two decades \u2014 so read it as whether this exponent means anything, not as whether the relationship is truly scale free. This entry is the exponent.',
    tokenExponentR2: 'How well the wealth distribution really follows a power law \u2014 whether the number of agents holding at least so many tokens falls away as a clean power of that threshold, or merely falls. Fitted as a straight line on log-log axes, which turns a power law into a slope. Such a fit returns two numbers: the exponent, which is the slope itself, and R\u00b2, the share of the scatter the line accounts for. A high R\u00b2 is not proof of a power law \u2014 log-normal and stretched-exponential curves look just as straight over two decades \u2014 so read it as whether this exponent means anything, not as whether the relationship is truly scale free. This entry is the R\u00b2.',
    tokensVsDegree: 'How an agent\u2019s wealth scales with how connected it is. This is the relationship between the two things an agent can accumulate, and whether they are the same game or two different ones: a slope above 1 means wealth grows faster than connections, so the well-connected are disproportionately rich, while a slope near 0 means position buys nothing. Fitted as a straight line on log-log axes, which turns a power law into a slope. Such a fit returns two numbers: the exponent, which is the slope itself, and R\u00b2, the share of the scatter the line accounts for. A high R\u00b2 is not proof of a power law \u2014 log-normal and stretched-exponential curves look just as straight over two decades \u2014 so read it as whether this exponent means anything, not as whether the relationship is truly scale free. This entry is the exponent.',
    tokensVsDegreeR2: 'How tightly an agent\u2019s wealth follows how connected it is. Fitted as a straight line on log-log axes, which turns a power law into a slope. Such a fit returns two numbers: the exponent, which is the slope itself, and R\u00b2, the share of the scatter the line accounts for. A high R\u00b2 is not proof of a power law \u2014 log-normal and stretched-exponential curves look just as straight over two decades \u2014 so read it as whether this exponent means anything, not as whether the relationship is truly scale free. This entry is the R\u00b2. A low value alongside a steep slope means the trend is real but the scatter is enormous \u2014 being well connected helps on average while saying very little about any particular agent.',
    trianglesVsDegree: 'How the number of triangles an agent sits in grows with its number of connections, where a triangle is a closed triple of mutually connected agents. If neighbourhoods were wired at random the count would grow with the square of the connection count, so a slope well below 2 means hubs are joining parts of the graph that do not join each other. Fitted as a straight line on log-log axes, which turns a power law into a slope. Such a fit returns two numbers: the exponent, which is the slope itself, and R\u00b2, the share of the scatter the line accounts for. A high R\u00b2 is not proof of a power law \u2014 log-normal and stretched-exponential curves look just as straight over two decades \u2014 so read it as whether this exponent means anything, not as whether the relationship is truly scale free. This entry is the exponent.',
    trianglesVsDegreeR2: 'How tightly the number of triangles an agent sits in follows its number of connections, a triangle being a closed triple of mutually connected agents. Fitted as a straight line on log-log axes, which turns a power law into a slope. Such a fit returns two numbers: the exponent, which is the slope itself, and R\u00b2, the share of the scatter the line accounts for. A high R\u00b2 is not proof of a power law \u2014 log-normal and stretched-exponential curves look just as straight over two decades \u2014 so read it as whether this exponent means anything, not as whether the relationship is truly scale free. This entry is the R\u00b2.',
    clusteringVsDegree: 'How an agent\u2019s clustering coefficient \u2014 the chance that two of its neighbours are neighbours of each other \u2014 changes with how many connections it has. This is the clearest self-similarity signature available here. A slope near minus 1 is the hallmark of a hierarchical network: small dense neighbourhoods grouped into larger, sparser ones, the same arrangement repeating at every scale. A flat slope means the neighbourhood of a hub looks like the neighbourhood of a leaf and there is no hierarchy to speak of. Only agents with at least two neighbours have a coefficient at all, so the rest are left out. Fitted as a straight line on log-log axes, which turns a power law into a slope. Such a fit returns two numbers: the exponent, which is the slope itself, and R\u00b2, the share of the scatter the line accounts for. A high R\u00b2 is not proof of a power law \u2014 log-normal and stretched-exponential curves look just as straight over two decades \u2014 so read it as whether this exponent means anything, not as whether the relationship is truly scale free. This entry is the exponent.',
    clusteringVsDegreeR2: 'How tightly an agent\u2019s clustering coefficient \u2014 the chance that two of its neighbours are neighbours of each other \u2014 follows its number of connections. Fitted as a straight line on log-log axes, which turns a power law into a slope. Such a fit returns two numbers: the exponent, which is the slope itself, and R\u00b2, the share of the scatter the line accounts for. A high R\u00b2 is not proof of a power law \u2014 log-normal and stretched-exponential curves look just as straight over two decades \u2014 so read it as whether this exponent means anything, not as whether the relationship is truly scale free. This entry is the R\u00b2, and it is worth reading carefully, because the claim that the network is hierarchical rests entirely on that trend being real rather than on a handful of hubs.',
    changeVsTokens: 'How the size of a phase\u2019s change in an agent\u2019s tokens scales with how many tokens it holds. A slope of 1 means changes are proportional \u2014 everyone risks the same fraction of what they have, which is what multiplicative growth looks like and what produces heavy-tailed wealth all on its own. Below 1 means the rich move a smaller share than the poor, which is stabilising; above 1 means they move more, which is not. Fitted as a straight line on log-log axes, which turns a power law into a slope. Such a fit returns two numbers: the exponent, which is the slope itself, and R\u00b2, the share of the scatter the line accounts for. A high R\u00b2 is not proof of a power law \u2014 log-normal and stretched-exponential curves look just as straight over two decades \u2014 so read it as whether this exponent means anything, not as whether the relationship is truly scale free. This entry is the exponent.',
    changeVsTokensR2: 'How tightly the size of a phase\u2019s change in an agent\u2019s tokens follows how many tokens it holds. Fitted as a straight line on log-log axes, which turns a power law into a slope. Such a fit returns two numbers: the exponent, which is the slope itself, and R\u00b2, the share of the scatter the line accounts for. A high R\u00b2 is not proof of a power law \u2014 log-normal and stretched-exponential curves look just as straight over two decades \u2014 so read it as whether this exponent means anything, not as whether the relationship is truly scale free. This entry is the R\u00b2.',
    assortativity: 'Whether well-connected agents attach to other well-connected agents, as a correlation between the number of connections at each end of a link, running from -1 to 1. Positive means like joins like, hubs wiring to hubs. Negative means hubs sit among sparsely connected agents, which is what most grown networks do. Zero means how connected an agent is says nothing about how connected its neighbours are. Newman\u2019s formula, over the connections of this frame alone.',
    density: 'Connections present as a share of every connection that could exist between the agents alive. Falls quickly as the population grows, because the number of possible connections grows with the square of the number of agents while the real count grows far more slowly.',
    medianDegree: 'The number of connections held by the middle agent, when all agents are lined up from least to most connected. Half the population has fewer than this and half has more. It sits well under the average whenever a few heavily connected agents are pulling that average up.',
    minDegree: 'The number of connections held by the least connected surviving agent. Zero would mean an agent with no connections at all, which cleanup removes, so on a recorded frame this is normally one.',
    leaves: 'How many agents have exactly one connection. Such an agent depends entirely on a single neighbour: if that one connection is pruned for carrying no tokens, the agent is cut off from everyone and culled.',
    meanTokens: 'The total number of tokens divided by the number of agents. Because tokens are conserved by default, this moves only when the population changes \u2014 it rises when agents die and falls when they are born.',
    minTokens: 'How many tokens the poorest surviving agent holds. Anything that reached zero was already removed by cleanup, so this is the poorest agent that survived rather than the poorest that existed.',
    topDecileShare: 'What share of all the tokens in the world is held by the richest tenth of agents. If wealth were perfectly even this would read 10%; the further above that it climbs, the more the top of the population owns.',
    cladesInWindow: 'How many separate families the population divides into, counting a family as everything descended from one agent that was alive a few iterations ago. Absent unless every iteration of the run was recorded, because ancestry is a chain and a chain cannot be reconstructed from sampled frames.',
    distinctParents: 'How many different genomes the agents alive right now mutated away from \u2014 one step back up the ancestry, not a count of families. Two agents whose parents carried the same genome count once. This is deliberately a shallow measure: after seventeen iterations of one run there were 105 distinct parent genomes and only a single surviving family, so a large number here says nothing about how many separate lines are really competing.',
    reproTokenShare: 'What share of every token in the world was committed to newborns this phase, adding up what all the parents handed over. This is how much of the entire economy the population spent on reproducing, which is a different question from how much any individual parent gave away: a world where a handful of poor agents each hand over everything they have spends very little of the whole while costing each of those parents everything.',
    meanInvestedShare: 'The average share of its own tokens that a reproducing agent handed to its child. Reproduction is paid for out of the parent\u2019s own pile, so this is how much of itself the typical parent gave away \u2014 a per-parent figure, which says nothing on its own about how many parents there were or how large a bite this took out of the world.',
    handovers: 'How many connections a parent gave to its newborn this phase. The connection moves from parent to child rather than being copied, so the parent ends up with one fewer. Only recorded on runs created with handover enabled; an absent value means the mechanic was switched off, which is not the same as it being on and never used. This can never add a connection to the graph: normally only one end changes and the total holds, but where the newborn was already wired to that neighbour the moved connection merges into the one a simple graph can hold, and the total falls by one.',
    meanChildLinks: 'The average number of connections a newborn was wired into. One means children hang off a single neighbour and depend entirely on it; higher values mean they are born well embedded in the graph.',
    totalFlow: 'How many tokens crossed a connection this phase, ignoring whatever agents kept staked on themselves. This is the volume of real traffic in the network.',
    meanEdgeFlow: 'The average number of tokens carried by each connection that carried anything at all. Connections that carried nothing are excluded, which is the same set that gets pruned at the end of the phase, so this is the average load on the connections that survived.',
    maxEdgeFlow: 'The number of tokens carried by the single busiest connection \u2014 the largest amount sent along any one link this phase.',
    selfAllocationShare: 'What share of all allocated tokens agents kept staked on their own node rather than spending on neighbours. A high value means a defensive population holding what it has; a low one means an aggressive population pushing outward.',
    revoltShare: 'What share of allocated tokens were flagged as revolutionary. Tokens flagged this way count toward a coalition against the largest allocator instead of backing the strongest bid, so this is how much of the phase\u2019s spending went into ganging up rather than into winning outright.',
    spreadShare: 'What share of agents spread their tokens proportionally across several targets instead of putting everything on a single one. Each agent makes this choice with its own mode head, so the line tracks a strategy that evolves rather than a setting that was fixed in advance.',
    heldHomeShare: 'What share of nodes were still held by their occupant at the end of the phase. The rest were taken over, and the winner\u2019s genome copied into them.',
    prunedEdges: 'How many connections were removed for carrying no tokens this phase. Connections have to be used to survive.',
    maxTokenAdded: 'The largest gain any single agent made this phase. In a reproduction phase that is usually a newborn receiving its endowment; in a game phase it is a node that collected a heavy bid, often one conquered by a wealthy neighbour. It will not cancel against the largest loss, for two reasons. Both are maxima over individual agents, and the biggest winner and the biggest loser are different agents with nothing tying one\u2019s fortune to the other\u2019s. Summed across everyone, gains exceed losses by exactly what the agents who did not survive the phase were holding when it began. Those agents are gone from the recorded frame, so their losses are never counted, while the tokens they let go of turn up as gains for the survivors \u2014 some spent on neighbours during the phase, the rest scattered by cleanup. Tokens are still conserved overall; it is the bookkeeping that is one-sided, because a frame can only describe agents still alive to be described.',
    maxTokenLost: 'The largest loss any single agent took this phase, given as a positive number. In a reproduction phase that is a parent paying for a child; in a game phase it is an agent that spent its pile on neighbours and got little back. Summed across everyone, gains exceed losses by exactly what the agents who did not survive the phase were holding when it began. Those agents are gone from the recorded frame, so their losses are never counted, while the tokens they let go of turn up as gains for the survivors \u2014 some spent on neighbours during the phase, the rest scattered by cleanup. Tokens are still conserved overall; it is the bookkeeping that is one-sided, because a frame can only describe agents still alive to be described.',
    gainers: 'How many agents ended the phase holding more tokens than they started it with.',
    losers: 'How many agents ended the phase holding fewer tokens than they started it with.',
    redistributed: 'How many tokens were scattered evenly over the survivors at the end of the phase, which is what keeps the global count conserved. Mostly this is what the agents who died this phase were holding, plus anything the run was configured to mint each phase; that setting is zero by default, and while it is, this is the estate of the dead and nothing else. It reads zero when there were no survivors to scatter it over \u2014 the pool is then dropped, and a single fresh agent is given the whole world instead.'
  },

  // Statistics that count nodes, and so are also meaningful as a percentage
  // of the population that entered the phase.
  SHARE_KEYS: new Set(['births', 'revolutions', 'starved', 'orphaned', 'leaves',
                       'gainers', 'losers']),

  init() {
    this.el = document.getElementById('statDetail');
    this.titleEl = document.getElementById('statDetailTitle');
    this.textEl = document.getElementById('statDetailText');
    this.footEl = document.getElementById('statDetailFoot');
    this.canvas = document.getElementById('statDetailChart');
    this.progressEl = document.getElementById('statDetailProgress');
    this.progressFill = document.getElementById('statDetailProgressFill');
    this.progressLabel = document.getElementById('statDetailProgressLabel');

    document.getElementById('statDetailClose').addEventListener('click', () => this.close());
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && !this.el.classList.contains('hidden')) this.close();
    });
    window.addEventListener('resize', () => {
      if (!this.el.classList.contains('hidden')) this.redraw();
    });
  },

  close() {
    this.el.classList.add('hidden');
    this.currentKey = null;
  },

  /** Drop a run's cached series, e.g. after it has advanced. */
  invalidate(runId) {
    this.seriesCache.delete(runId);
  },

  async open(key, label) {
    if (!Viewer.runId) return;

    this.currentKey = key;
    this.titleEl.textContent = label || key;
    this.textEl.textContent = this.EXPLANATIONS[key] || 'No description for this value.';
    this.el.classList.remove('hidden');

    const runId = Viewer.runId;
    const cached = this.seriesCache.get(runId);
    if (cached) {
      this.series = cached;
      this.hideProgress();
      this.redraw();
      // A cached payload can still be a coarse one, so keep climbing from
      // where the last visit left off rather than settling for it.
      if (cached.complete) return;
    } else {
      this.showProgress();
    }

    try {
      await this.load(runId);
    } catch (err) {
      this.hideProgress();
      this.footEl.textContent = `Could not load history: ${err.message}`;
      return;
    }
    this.hideProgress();
    this.redraw();
  },

  /**
   * Climb to full resolution, drawing at every step.
   *
   * A long run is minutes of work to summarise, and waiting for all of it
   * before drawing anything shows an empty chart for that whole time, which
   * reads as broken rather than busy. Each pass covers the whole run and is
   * finer than the last, so the chart gains resolution as it loads.
   */
  async load(runId) {
    const token = (this.loadToken = (this.loadToken || 0) + 1);
    await SeriesLoad.climb(runId, {
      cancelled: () => this.loadToken !== token || Viewer.runId !== runId,
      onStep: (payload) => {
        this.seriesCache.set(runId, payload);
        this.series = payload;
        const at = SeriesLoad.fraction(payload);
        if (payload.complete) this.hideProgress();
        else if (at !== null) {
          this.setProgress(at, payload.points || 0, payload.totalPoints || 0);
        }
        this.redraw();
      }
    });
  },

  // ---- progress ------------------------------------------------------

  showProgress() {
    this.footEl.textContent = '';
    this.progressEl.classList.remove('hidden');
    this.setProgress(null, 0, 0);
  },

  hideProgress() {
    this.progressEl.classList.add('hidden');
  },

  /**
   * Move the bar.
   *
   * A null fraction means the server has not said how much there is to do yet,
   * so the bar sweeps rather than claiming a position it cannot know.
   */
  setProgress(fraction, done, total) {
    if (fraction === null) {
      this.progressFill.classList.add('indeterminate');
      this.progressFill.style.width = '';
      this.progressLabel.textContent = 'Analysing frames…';
      return;
    }
    this.progressFill.classList.remove('indeterminate');
    this.progressFill.style.width = `${Math.round(fraction * 100)}%`;
    this.progressLabel.textContent =
      `Refining… ${formatNumber(done)} of ${formatNumber(total)} points (${Math.round(fraction * 100)}%)`;
  },

  /**
   * Points for the current statistic under the active phase filter.
   *
   * Node-count statistics are converted to a share of the population that
   * entered the phase, because "40 births" means something quite different in
   * a world of 100 agents than in one of 4,000.
   */
  points() {
    const key = this.currentKey;
    if (!this.series || !key) return { xs: [], ys: [], asShare: false };

    const s = this.series.series;
    const values = s[key] || [];
    const phases = s.phase || [];
    const iterations = s.iteration || [];
    const before = s.nodes_before || [];
    const nodes = s.nodes || [];

    const asShare = this.SHARE_KEYS.has(key);
    const xs = [], ys = [];

    for (let i = 0; i < values.length; i++) {
      if (!Viewer.framePassesFilter(phases[i])) continue;
      const v = values[i];
      if (v === null || v === undefined) continue;

      if (asShare) {
        // Older runs predate nodes_before; the post-phase count is the closest
        // honest stand-in, so the curve stays usable rather than empty.
        const denominator = before[i] || nodes[i] || 0;
        if (!denominator) continue;
        ys.push((v / denominator) * 100);
      } else {
        ys.push(v);
      }
      xs.push(iterations[i]);
    }
    return { xs, ys, asShare };
  },

  redraw() {
    const { xs, ys, asShare } = this.points();
    const canvas = this.canvas;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);

    if (!ys.length) {
      ctx.fillStyle = '#5b6b7c';
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText('No data for this statistic under the current phase filter.', 10, h / 2);
      this.footEl.textContent = '';
      return;
    }

    const padL = 62, padR = 12, padT = 10, padB = 30;
    const plotW = w - padL - padR, plotH = h - padT - padB;

    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }

    // Widen to the round numbers, so the top and bottom lines are labelled
    // values rather than wherever the data happened to stop.
    const yGrid = niceTicks(lo, hi, 5);
    if (yGrid.ticks.length > 1) {
      lo = Math.min(lo, yGrid.ticks[0]);
      hi = Math.max(hi, yGrid.ticks[yGrid.ticks.length - 1]);
    }

    const xLo = xs[0], xHi = xs[xs.length - 1];
    const xGrid = niceTicks(xLo, xHi, 5);

    const xAt = v => padL + (xHi === xLo ? plotW / 2 : ((v - xLo) / (xHi - xLo)) * plotW);
    const yAt = v => padT + plotH - ((v - lo) / (hi - lo)) * plotH;

    const fmtY = v => asShare
      ? `${+v.toFixed(2)}%`
      : (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US')
                             : String(+v.toFixed(Math.abs(v) < 1 ? 3 : 2)));

    ctx.font = '10px system-ui, sans-serif';
    ctx.lineWidth = 1;

    // Horizontal grid
    for (const v of yGrid.ticks) {
      const y = Math.round(yAt(v)) + 0.5;
      if (y < padT - 1 || y > padT + plotH + 1) continue;
      ctx.strokeStyle = '#1e2733';
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();

      ctx.fillStyle = '#8fa3b5';
      const label = fmtY(v);
      ctx.fillText(label, padL - 6 - ctx.measureText(label).width, y + 3);
    }

    // Vertical grid
    for (const v of xGrid.ticks) {
      const x = Math.round(xAt(v)) + 0.5;
      if (x < padL - 1 || x > padL + plotW + 1) continue;
      ctx.strokeStyle = '#1e2733';
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();

      ctx.fillStyle = '#8fa3b5';
      const label = Math.round(v).toLocaleString('en-US');
      ctx.fillText(label, x - ctx.measureText(label).width / 2, h - 12);
    }

    // Axes, a shade brighter than the grid
    ctx.strokeStyle = '#33404f';
    ctx.beginPath();
    ctx.moveTo(padL + 0.5, padT);
    ctx.lineTo(padL + 0.5, padT + plotH + 0.5);
    ctx.lineTo(padL + plotW, padT + plotH + 0.5);
    ctx.stroke();

    // Where the frame on screen sits, so the number in the strip has a home
    const currentIteration = Viewer.frame ? Viewer.frame.iteration : null;
    if (currentIteration !== null && currentIteration >= xLo && currentIteration <= xHi) {
      ctx.strokeStyle = '#4fb3ff';
      ctx.globalAlpha = 0.45;
      ctx.beginPath();
      ctx.moveTo(xAt(currentIteration), padT);
      ctx.lineTo(xAt(currentIteration), padT + plotH);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // The curve
    ctx.strokeStyle = '#4fb3ff';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < ys.length; i++) {
      const x = xAt(xs[i]), y = yAt(ys[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = '#5b6b7c';
    ctx.font = '9px system-ui, sans-serif';
    const axisLabel = 'iteration';
    ctx.fillText(axisLabel, padL + plotW - ctx.measureText(axisLabel).width, h - 1);

    const sampled = this.series && this.series.sampled;
    this.footEl.textContent =
      `${formatNumber(ys.length)} point${ys.length === 1 ? '' : 's'} · ${Viewer.phaseFilterLabel()}` +
      (asShare ? ' · shown as a share of the nodes that entered the phase' : '') +
      (sampled ? ` · sampled every ${formatNumber(this.series.stride)} iterations` : '');
  }
};
