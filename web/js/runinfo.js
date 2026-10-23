/**
 * What a run is, in words a reader can take in: a short title, and every setting
 * beside what it means. Shared by the Simulations tab and the viewer. No DOM here,
 * so the tests load it on its own (tests/test_view.js), as they do runstats.js.
 */
const RunInfo = {
  /**
   * `B1 · 20,000 tokens · seed 1001`: the run as the lab named it, without the list
   * of every setting it changed, which a world of the screen of random settings can
   * make 400 characters long. A run named by hand keeps its own name.
   */
  shortTitle(run) {
    const lab = run && run.lab;
    const name = (run && run.name) || '';
    if (!lab || !lab.baseline || !name.startsWith(`${lab.baseline} · `)) return name;
    const tokens = (lab.world || {}).total_tokens;
    return [lab.baseline,
            ...(tokens != null ? [`${Number(tokens).toLocaleString('en-US')} tokens`] : []),
            `seed ${lab.seed}`,
            ...(lab.replicate ? [lab.replicate] : [])].join(' · ');
  },

  /** What the lab changed from the run's baseline, setting by setting, the world's size aside. */
  changed(run) {
    const lab = (run && run.lab) || {};
    const world = Object.entries(lab.world || {}).filter(([key]) => key !== 'total_tokens');
    return Object.fromEntries([...world, ...Object.entries(lab.set || {})]);
  },

  /**
   * Every setting of a run, as rows a dialog can show. `groups` is the form's
   * grouping, [[heading, [[key, label], ...]], ...]; `glossary` is book/settings.json;
   * `text` writes a value out. Each row says what the setting means and whether the
   * run changed it from its baseline.
   */
  rows(run, groups, glossary, text) {
    const cfg = (run && run.config) || {};
    const changed = this.changed(run);
    const meanings = (glossary && glossary.settings) || {};
    return groups.map(([heading, keys]) => [heading, keys.map(([key, label]) => ({
      key,
      label,
      value: text(cfg[key]),
      meaning: (meanings[key] || {}).meaning || '',
      changed: Object.prototype.hasOwnProperty.call(changed, key),
    }))]);
  },
};
