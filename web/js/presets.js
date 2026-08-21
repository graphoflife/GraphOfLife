/*
 * Visualization presets.
 *
 * A preset is the whole look: colours, sizes, background, and how the graph
 * arranges itself — forces, dimensions and whether positions carry between
 * frames. Applying one therefore changes the arrangement as well as the
 * palette.
 *
 * The four built-ins are starting points; anything you save is your own and
 * lives in localStorage, so it survives reloads without needing the server to
 * know about per-machine display preferences.
 */
const Presets = {
  STORAGE_KEY: 'gol.presets.v1',

  /** Layout every built-in starts from, unless it says otherwise. */
  BASE_LAYOUT: {
    forceCharge: 20, forceLink: 0.12, forceCenter: 0.012,
    forceAngular: 0.15, forceDamping: 0.86, forceTheta: 1.2,
    dimensions: 3, layoutCarry: true, autoFit: true
  },

  BUILT_IN: {
    // What the viewer opens with. Everything below is a place to go from here;
    // this is the starting point, so it is the one preset that has to be a
    // sensible view of a graph nobody has looked at yet rather than a way of
    // answering a particular question.
    //
    // Applied on load by Viewer.init, which merges it into the settings before
    // anything is pushed to the layout or the controls.
    default: {
      // Tuned on a large graph, where the default settings get slow and
      // cluttered: small nodes, thin translucent edges and a stronger, looser,
      // less exactly approximated repulsion, which is what keeps a few
      // thousand agents legible and moving.
      //
      // Applied on load by Viewer.init, which merges it into the settings
      // before anything is pushed to the layout or the controls.
      nodeColorBy: 'tokens', nodeColorLog: true, nodeColormap: 'coolwarm',
      nodeColorReverse: false,
      nodeSizeBy: 'tokens', nodeSizeLog: true, nodeSizeMin: 0.5, nodeSizeMax: 5,
      nodeAlpha: 1,
      nodeOutline: false, nodeOutlineColor: '#ffffff',
      nodeOutlineAlpha: 0.55, nodeOutlineWidth: 0.6,
      nodeGlow: false, nodeGlowColorBy: 'node',
      nodeGlowSize: 2.6, nodeGlowStrength: 0.35,

      edgeShow: true, edgeColorBy: 'constant', edgeColorLog: false,
      edgeColormap: 'cividis', edgeColorReverse: false,
      edgeFlatColor: '#ffffff', edgeWidthBy: 'constant', edgeWidthLog: false,
      edgeWidthMin: 0.3, edgeWidthMax: 0.3, edgeAlpha: 0.7,

      bgStyle: 'solid', bgColorA: '#20252c', bgColorB: '#20252c',
      showLegend: true, showEdgeLegend: true, layoutCarry: true,

      // Strong repulsion against weak links spreads the graph out; the high
      // theta trades exactness in the far field for speed, which is where the
      // cost lives on a large graph.
      forceCharge: 82, forceLink: 0.07, forceCenter: 0.012,
      forceAngular: 0.15, forceDamping: 0.78, forceTheta: 1.7,
      dimensions: 3,

      // On, so the view keeps the whole graph framed as the population grows
      // rather than letting it drift off the edge. Panning or zooming hands
      // control back; orbiting does not, since turning a graph changes what
      // "framed" means rather than saying you want a different view.
      autoFit: true,

      focusRadius: 2,

      distMetric: 'node:tokens', histDistX: 'log', histDistY: 'log',
      heatX: 'node:tokens', heatY: 'node:degree',
      histHeatX: 'log', histHeatY: 'log', histHeatCount: 'log',
      trajX: 'nodes', trajY: 'edges', histTrajX: 'log', histTrajY: 'log'
    },

    wealth: {
      nodeColorBy: 'tokens', nodeColorLog: true, nodeColormap: 'inferno',
      nodeSizeBy: 'tokens', nodeSizeLog: false,
      edgeColorBy: 'constant', edgeWidthBy: 'constant', edgeAlpha: 0.18,
      bgStyle: 'radial', nodeAlpha: 0.95,
      distMetric: 'node:tokens', histDistX: 'log', histDistY: 'linear',
      heatX: 'node:degree', heatY: 'node:tokens',
      histHeatX: 'linear', histHeatY: 'log', histHeatCount: 'log',
      forceCharge: 24, forceLink: 0.12, forceCenter: 0.012,
      forceAngular: 0.15, forceDamping: 0.86, dimensions: 3
    },
    lineage: {
      nodeColorBy: 'brain_id', nodeColorLog: false, nodeColormap: 'turbo',
      nodeSizeBy: 'tokens', nodeSizeLog: true,
      edgeColorBy: 'source', edgeAlpha: 0.3, bgStyle: 'solid', nodeAlpha: 0.9,
      distMetric: 'node:brain_id', histDistX: 'linear', histDistY: 'log',
      heatX: 'node:age', heatY: 'node:tokens',
      histHeatX: 'linear', histHeatY: 'log', histHeatCount: 'log',
      // Looser and more open, so separate lineages drift apart visibly.
      forceCharge: 45, forceLink: 0.09, forceCenter: 0.009,
      forceAngular: 0.2, forceDamping: 0.88, dimensions: 3
    },
    structure: {
      nodeColorBy: 'degree', nodeColorLog: true, nodeColormap: 'cividis',
      nodeSizeBy: 'degree', nodeSizeLog: false,
      edgeColorBy: 'avg_degree', edgeColorLog: true,
      edgeWidthBy: 'avg_degree', edgeWidthLog: true, edgeAlpha: 0.35,
      bgStyle: 'linear', nodeAlpha: 0.85,
      distMetric: 'node:degree', histDistX: 'linear', histDistY: 'log',
      heatX: 'node:degree', heatY: 'node:loops',
      histHeatX: 'linear', histHeatY: 'log', histHeatCount: 'log',
      // Strong angular spread and tight links, which is what makes the
      // branching shape of the graph legible.
      forceCharge: 12, forceLink: 0.2, forceCenter: 0.014,
      forceAngular: 0.5, forceDamping: 0.86, dimensions: 3
    },
    flow: {
      // Where wealth is running uphill: curvature says which nodes sit in a
      // valley their neighbours are draining into, and which are the peaks.
      nodeColorBy: 'token_curvature', nodeColorLog: true, nodeColormap: 'coolwarm',
      nodeSizeBy: 'abs_token_delta', nodeSizeLog: true,
      edgeColorBy: 'flow', edgeColorLog: true, edgeWidthBy: 'flow',
      edgeWidthLog: true, edgeAlpha: 0.4,
      bgStyle: 'solid', nodeAlpha: 0.9,
      distMetric: 'node:token_curvature', histDistX: 'log', histDistY: 'log',
      heatX: 'node:tokens', heatY: 'node:token_curvature',
      histHeatX: 'log', histHeatY: 'log', histHeatCount: 'log',
      forceCharge: 20, forceLink: 0.12, forceCenter: 0.012,
      forceAngular: 0.15, forceDamping: 0.86, dimensions: 3
    },
    minimal: {
      nodeColorBy: 'constant', nodeColormap: 'grayscale', nodeSizeBy: 'constant',
      edgeColorBy: 'constant', edgeWidthBy: 'constant', edgeAlpha: 0.12,
      bgStyle: 'solid', nodeAlpha: 0.7,
      distMetric: 'node:degree', histDistX: 'linear', histDistY: 'linear',
      heatX: 'node:degree', heatY: 'node:tokens',
      histHeatX: 'linear', histHeatY: 'log', histHeatCount: 'log',
      forceCharge: 18, forceLink: 0.12, forceCenter: 0.012,
      forceAngular: 0.08, forceDamping: 0.86, dimensions: 3
    }
  },

  /** A built-in, with any layout key it omits filled in from BASE_LAYOUT. */
  builtIn(name) {
    const preset = this.BUILT_IN[name];
    return preset ? { ...this.BASE_LAYOUT, ...preset } : null;
  },

  load() {
    try {
      return JSON.parse(localStorage.getItem(this.STORAGE_KEY)) || {};
    } catch (err) {
      return {};
    }
  },

  save(all) {
    try {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(all));
      return true;
    } catch (err) {
      alert(`Could not save preset: ${err.message}`);
      return false;
    }
  },

  names() {
    return Object.keys(this.load()).sort();
  },

  get(name) {
    return this.load()[name] || null;
  },

  put(name, settings) {
    const clean = (name || '').trim();
    if (!clean) return false;
    const all = this.load();
    all[clean] = { ...settings };
    return this.save(all);
  },

  remove(name) {
    const all = this.load();
    if (!(name in all)) return false;
    delete all[name];
    return this.save(all);
  }
};
