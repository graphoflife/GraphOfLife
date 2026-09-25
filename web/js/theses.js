/*
 * The claims this project makes about its graphs, each as a chart.
 *
 * These were a view of their own — a list, a canvas, a legend, a loader — that
 * drew exactly what the Time series tab draws: a few statistics of one run
 * over time, with a line at a value the claim is stated in. So they are that
 * tab, pre-filled. A thesis is its prose and the statistics that decide it;
 * choosing one fills in the lines and the threshold, and its prose appears
 * under the chart for as long as the chart plots its statistics.
 *
 * Kept apart from diagrams.js because it is writing rather than machinery. It
 * sat in the middle of the chart code for a while, sixty lines of prose in a
 * quoting style of its own, and the chart code is long enough without it.
 */
const THESES = [
  {
    id: 'shortcut',
    title: 'The shortcut budget is spent',
    claim: 'The graph starts as a small world and cannot stay one. Its '
      + 'long-range edges are a one-time endowment that can only be lost.',
    why: 'Every edge the engine creates joins nodes at most two hops apart: a '
      + 'newborn links into its parent’s neighbourhood, and a handover moves '
      + 'an edge within it. Nothing anywhere creates a link between distant '
      + 'parts of the graph. Meanwhile edges are destroyed freely — every '
      + 'edge carrying no tokens is cut at the end of each Blotto phase. The '
      + 'Watts–Strogatz rewiring the world is built with is therefore a '
      + 'budget that is spent and never refilled.',
    confirm: 'Bridges rise as a share of edges, loop density falls, and the '
      + 'graph sits far from where a Watts–Strogatz graph of the same size '
      + 'and degree would be.',
    refute: 'The measures hold near their starting values, or return toward '
      + 'them after an early excursion — which would mean something is '
      + 'replacing long-range structure that this reading of the code says '
      + 'cannot be replaced.',
    stats: ['loopDensity', 'transitivity', 'bridgeShare'],
    guides: []
  },
  {
    id: 'fragility',
    title: 'Fragility leads the cull',
    claim: 'How lopsided the graph is predicts how many agents the next '
      + 'cleanup removes.',
    why: 'Cleanup keeps only the largest connected component and kills '
      + 'everything else. So a bridge with a tenth of the population behind '
      + 'it is not a curiosity — it is a tenth-of-the-population extinction, '
      + 'waiting for the zero-flow prune to happen to cut that one edge. If '
      + 'that is what is going on, the worst cut available *before* a phase '
      + 'should say something about the cull that phase produces.',
    confirm: 'Worst cut, taken before the cull, moves ahead of the culled '
      + 'share — and keeps doing so against a control that shifts one series '
      + 'in time and destroys only the alignment between them.',
    refute: 'No relationship beyond the shifted control, which would mean the '
      + 'culls are removing stragglers rather than severed regions, and the '
      + 'bridge structure is not what kills anyone.',
    note: 'The pre-cull reading is what makes this answerable at all. The '
      + 'ordinary bridge count is taken after cleanup, in the same frame as '
      + 'the cull, so a cut that severs a whole side moves both numbers at '
      + 'once and cause cannot be told from effect. Runs recorded before the '
      + 'engine started taking the pre-cull reading cannot test this.',
    stats: ['cutRiskBefore', 'culledShare'],
    guides: []
  },
  {
    id: 'lopsided',
    title: 'No cut costs more than a tenth',
    claim: 'Squash every redundant blob to a point and what is left is a tree '
      + 'whose edges are exactly the bridges. The claim is about how lopsided '
      + 'that tree is: no single bridge has more than a tenth of the '
      + 'population on one side of it.',
    why: 'That is the shape of a decentralised network — plenty of small '
      + 'tree-like fringes hanging off local points, but no chokepoint whose '
      + 'loss splits the world in half. It is also what real social and '
      + 'information networks look like: a well-connected core that no cut '
      + 'divides cheaply, with small pieces attached by single edges. If this '
      + 'population is organising rather than merely growing, this is the '
      + 'shape it would organise into.',
    confirm: 'Worst cut stays under the line and does not trend toward it. '
      + 'The population is a core with fringes, and no one edge holds it '
      + 'together.',
    refute: 'Worst cut climbs toward 50%, meaning the graph is a dumbbell '
      + 'held by one edge and every measurement of “organisation” is really a '
      + 'measurement of which half survived.',
    stats: ['cutRisk'],
    guides: [{ axis: 'y', at: 0.1, label: 'the tenth' }]
  },
  {
    id: 'curvature',
    title: 'The graph is negatively curved',
    claim: 'Neighbourhoods hold more than flat space allows, and keep doing '
      + 'so.',
    why: 'The dimension estimate walks outward from a node and fits how fast '
      + 'the frontier grows. In curved space that line bends, and the bend is '
      + 'the Ricci scalar rather than noise — the same fit gives both, and '
      + 'until now the second half was being discarded as residual. The sign '
      + 'is the interesting part: a branching, tree-like graph is strongly '
      + 'negative, and bounded-degree expanders are negatively curved as a '
      + 'theorem, so this doubles as a reading on how expander-like the graph '
      + 'is.',
    confirm: 'Curvature is negative and stable, or drifts steadily — either '
      + 'is a geometry rather than an artefact.',
    refute: 'It swings wildly frame to frame, in which case there are too few '
      + 'usable radii for the fit and the number is measuring the sampling, '
      + 'not the graph.',
    stats: ['ricciCurvature', 'dimension'],
    guides: []
  },
  {
    id: 'core',
    title: 'A core with whiskers',
    claim: 'The population splits into a well-connected core and a '
      + 'tree-shaped fringe hanging off it, rather than being uniform.',
    why: 'Peel away everyone with a single connection, and keep peeling until '
      + 'nobody has one. What survives is the part where every agent sits on '
      + 'some loop; what is peeled is fringe. Real networks are '
      + 'overwhelmingly this shape, and no standard random-graph model '
      + 'reproduces it, so finding it here would not be a property inherited '
      + 'from the starting graph.',
    confirm: 'Core share settles well below 1 and stays there — a stable '
      + 'division of labour between core and fringe rather than a transient.',
    refute: 'Core share sits near 1 (no fringe, the graph is uniformly '
      + 'interwoven) or falls toward 0 (no core, the graph is becoming a '
      + 'tree).',
    stats: ['coreShare', 'leafShare'],
    guides: []
  }
];
