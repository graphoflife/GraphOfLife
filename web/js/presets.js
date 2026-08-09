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
    wealth: {
      nodeColorBy: 'log_tokens', nodeColormap: 'inferno', nodeSizeBy: 'tokens',
      edgeColorBy: 'constant', edgeWidthBy: 'constant', edgeAlpha: 0.18,
      bgStyle: 'radial', nodeAlpha: 0.95,
      forceCharge: 24, forceLink: 0.12, forceCenter: 0.012,
      forceAngular: 0.15, forceDamping: 0.86, dimensions: 3
    },
    lineage: {
      nodeColorBy: 'brain_id', nodeColormap: 'turbo', nodeSizeBy: 'log_tokens',
      edgeColorBy: 'source', edgeAlpha: 0.3, bgStyle: 'solid', nodeAlpha: 0.9,
      // Looser and more open, so separate lineages drift apart visibly.
      forceCharge: 45, forceLink: 0.09, forceCenter: 0.009,
      forceAngular: 0.2, forceDamping: 0.88, dimensions: 3
    },
    structure: {
      nodeColorBy: 'log_degree', nodeColormap: 'cividis', nodeSizeBy: 'degree',
      edgeColorBy: 'log_avg_degree', edgeWidthBy: 'log_avg_degree', edgeAlpha: 0.35,
      bgStyle: 'linear', nodeAlpha: 0.85,
      // Strong angular spread and tight links, which is what makes the
      // branching shape of the graph legible.
      forceCharge: 12, forceLink: 0.2, forceCenter: 0.014,
      forceAngular: 0.5, forceDamping: 0.86, dimensions: 3
    },
    minimal: {
      nodeColorBy: 'constant', nodeColormap: 'grayscale', nodeSizeBy: 'constant',
      edgeColorBy: 'constant', edgeWidthBy: 'constant', edgeAlpha: 0.12,
      bgStyle: 'solid', nodeAlpha: 0.7,
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
