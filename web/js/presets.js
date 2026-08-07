/*
 * Visualization presets.
 *
 * The four built-ins are starting points; anything you save is your own and
 * lives in localStorage, so it survives reloads without needing the server to
 * know about per-machine display preferences.
 */
const Presets = {
  STORAGE_KEY: 'gol.presets.v1',

  BUILT_IN: {
    wealth: {
      nodeColorBy: 'log_tokens', nodeColormap: 'inferno', nodeSizeBy: 'tokens',
      edgeColorBy: 'constant', edgeWidthBy: 'constant', edgeAlpha: 0.18,
      bgStyle: 'radial', nodeAlpha: 0.95
    },
    lineage: {
      nodeColorBy: 'brain_id', nodeColormap: 'turbo', nodeSizeBy: 'log_tokens',
      edgeColorBy: 'source', edgeAlpha: 0.3, bgStyle: 'solid', nodeAlpha: 0.9
    },
    structure: {
      nodeColorBy: 'log_degree', nodeColormap: 'cividis', nodeSizeBy: 'degree',
      edgeColorBy: 'log_avg_degree', edgeWidthBy: 'log_avg_degree', edgeAlpha: 0.35,
      bgStyle: 'linear', nodeAlpha: 0.85
    },
    minimal: {
      nodeColorBy: 'constant', nodeColormap: 'grayscale', nodeSizeBy: 'constant',
      edgeColorBy: 'constant', edgeWidthBy: 'constant', edgeAlpha: 0.12,
      bgStyle: 'solid', nodeAlpha: 0.7
    }
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
