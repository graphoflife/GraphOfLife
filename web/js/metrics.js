/*
 * The vocabulary of things a frame can be measured by.
 *
 * One registry drives all five places a quantity can be chosen: node colour,
 * node size, edge colour, edge width, and the two charts under the canvas.
 * Adding a metric here makes it available everywhere at once.
 *
 * Log is a toggle rather than a separate entry. The old settings spelled it
 * into the key — `log_tokens` beside `tokens` — which doubled the length of
 * every menu and meant "tokens on a log scale" and "tokens" were unrelated
 * strings, so nothing could ask what a chart was actually showing.
 *
 * Node and edge metrics share several names (`loops`, `triangles`), so
 * anywhere both domains are on offer the key is qualified: `node:loops`
 * against `edge:loops`. The visual settings do not need that, since a
 * nodeColorBy is always a node metric.
 */
const Metrics = {
  // `signed` marks quantities that read as up-or-down rather than more-or-less.
  // They get a range centred on zero, so the middle of a colour map means "no
  // change" and a log scale keeps the sign instead of discarding it.
  NODE: [
    { key: 'tokens',           label: 'Tokens' },
    { key: 'degree',           label: 'Degree' },
    { key: 'token_delta',      label: 'Token change', signed: true },
    { key: 'abs_token_delta',  label: 'Token change (magnitude)' },
    { key: 'token_curvature',  label: 'Token curvature', signed: true },
    // Measured on the graph as it stood before this phase ran, so it can be
    // read against the change the phase then produced.
    { key: 'token_curvature_pre', label: 'Token curvature (before phase)',
      signed: true, needsPrevious: true },
    { key: 'loops',            label: 'Loops through it' },
    { key: 'triangles',        label: 'Triangles' },
    { key: 'token_share',      label: 'Share of total tokens', format: 'share' },
    { key: 'brain_id',         label: 'Brain id' },
    { key: 'parent_brain_id',  label: 'Parent brain id' },
    // Iterations lived. The node id used to stand in for this, and an ordinal
    // is not a duration — it says who is older, never by how much, and nothing
    // at all across two runs.
    { key: 'age',              label: 'Age (iterations lived)' },
    { key: 'node_id',          label: 'Node id (birth order)' }
  ],

  EDGE: [
    { key: 'avg_tokens',    label: 'Average endpoint tokens' },
    { key: 'min_tokens',    label: 'Weaker endpoint tokens' },
    { key: 'max_tokens',    label: 'Stronger endpoint tokens' },
    { key: 'token_gap',     label: 'Endpoint token gap' },
    { key: 'avg_degree',    label: 'Average endpoint degree' },
    { key: 'min_degree',    label: 'Weaker endpoint degree' },
    { key: 'max_degree',    label: 'Stronger endpoint degree' },
    { key: 'avg_curvature', label: 'Average endpoint curvature', signed: true },
    { key: 'flow',          label: 'Token flow (game phase)' },
    { key: 'loops',         label: 'Loops through it' },
    { key: 'triangles',     label: 'Triangles' },
    { key: 'bridge',        label: 'Bridge (on no loop)' }
  ],

  // Choices that are not measurements of anything, and so have no distribution
  // to plot. They belong in the visual menus but not in the chart menus.
  CONSTANT: { key: 'constant', label: 'Constant' },
  INHERIT: { key: 'source', label: 'Inherit from node colour' },

  list(domain) {
    return domain === 'edge' ? this.EDGE : this.NODE;
  },

  get(domain, key) {
    return this.list(domain).find(m => m.key === key) || null;
  },

  label(domain, key) {
    if (key === 'constant') return this.CONSTANT.label;
    if (key === 'source') return this.INHERIT.label;
    const m = this.get(domain, key);
    return m ? m.label : key;
  },

  /**
   * Metrics that describe the state before the phase rather than after it.
   *
   * These need the preceding frame, and they have no value at all for a node
   * that did not exist yet — which is the point: a newborn has no "before" to
   * have changed from. Those come back as NaN and are dropped by the charts.
   */
  NEEDS_PREVIOUS: new Set(['token_curvature_pre']),

  needsPrevious(key) {
    return this.NEEDS_PREVIOUS.has(key);
  },

  /** Whether this quantity is centred on zero rather than running upward. */
  isSigned(domain, key) {
    const m = this.get(domain, key);
    return Boolean(m && m.signed);
  },

  /**
   * A log scale that survives negative values.
   *
   * Signed quantities keep their direction and compress only the magnitude,
   * so a large loss stays on the far side of zero from a large gain. Anything
   * else is clamped at zero, since log of a negative is not a number the
   * colour map can use.
   */
  applyLog(value, signed) {
    return signed
      ? Math.sign(value) * Math.log1p(Math.abs(value))
      : Math.log1p(Math.max(0, value));
  },

  /** Undo applyLog, for printing an axis in the units the reader chose. */
  undoLog(value, signed) {
    return signed
      ? Math.sign(value) * Math.expm1(Math.abs(value))
      : Math.expm1(Math.max(0, value));
  },

  format(domain, key, value, log = false) {
    const m = this.get(domain, key);
    const raw = log ? this.undoLog(value, Boolean(m && m.signed)) : value;

    if (m && m.format === 'share') return `${(raw * 100).toFixed(2)}%`;

    const n = Math.round(raw);
    if (m && m.signed) return (n > 0 ? '+' : '') + n.toLocaleString('en-US');
    return n.toLocaleString('en-US');
  },

  // ---- qualified keys, for menus that offer both domains ----------------

  qualify(domain, key) { return `${domain}:${key}`; },

  parse(qualified) {
    const at = String(qualified || '').indexOf(':');
    if (at < 0) return { domain: 'node', key: String(qualified || '') };
    return { domain: qualified.slice(0, at), key: qualified.slice(at + 1) };
  },

  // ---- building the menus ----------------------------------------------

  /**
   * Fill a <select> with one domain's metrics.
   * `extras` appends the non-measurement choices a visual setting allows.
   */
  fillSelect(select, domain, { extras = [], selected = null } = {}) {
    if (!select) return;
    const option = m => `<option value="${m.key}">${m.label}</option>`;
    select.innerHTML = extras.map(option).join('') + this.list(domain).map(option).join('');
    if (selected !== null) select.value = selected;
  },

  /** Fill a <select> with both domains, grouped and domain-qualified. */
  fillDomainSelect(select, selected = null) {
    if (!select) return;
    const group = (domain, label) =>
      `<optgroup label="${label}">` +
      this.list(domain).map(m =>
        `<option value="${this.qualify(domain, m.key)}">${m.label}</option>`).join('') +
      '</optgroup>';

    select.innerHTML = group('node', 'Nodes') + group('edge', 'Edges');
    if (selected !== null) select.value = selected;
  },

  // ---- migrating the old settings --------------------------------------

  /**
   * Rewrite a settings object written before log became a toggle.
   *
   * Saved presets and the built-ins both stored `log_tokens` in the same field
   * that now holds `tokens`, so without this an old preset would silently
   * select nothing and the view would fall back to a flat colour.
   */
  LEGACY_PAIRS: [
    ['nodeColorBy', 'nodeColorLog'],
    ['nodeSizeBy', 'nodeSizeLog'],
    ['edgeColorBy', 'edgeColorLog'],
    ['edgeWidthBy', 'edgeWidthLog']
  ],

  migrateSettings(settings) {
    if (!settings) return settings;

    for (const [byKey, logKey] of this.LEGACY_PAIRS) {
      const value = settings[byKey];
      if (typeof value !== 'string' || !value.startsWith('log_')) continue;
      settings[byKey] = value.slice(4);
      // An explicit flag in the same object wins: it was written by the new
      // code and describes intent, where the prefix is only leftover spelling.
      if (settings[logKey] === undefined) settings[logKey] = true;
    }

    // Flow used to be read on a log scale unconditionally, with no linear
    // option offered. Keep those presets looking as they did.
    for (const [byKey, logKey] of this.LEGACY_PAIRS) {
      if (settings[byKey] === 'flow' && settings[logKey] === undefined) {
        settings[logKey] = true;
      }
    }

    return settings;
  }
};
