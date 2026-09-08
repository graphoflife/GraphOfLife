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
   * React to a menu the moment its value moves, not when it is committed.
   *
   * A `<select>` on its own only reports a `change`, and arrowing through the
   * options is a change per step in some browsers and nothing at all until
   * Enter in others. Chart menus are for browsing — you move down the list to
   * see what each one looks like — so all three signals are taken: `input` and
   * `change` for the browsers that send them, and the arrow keys directly,
   * read on the next frame once the element has settled on its new value.
   *
   * Duplicate events for one move are harmless: the handler is given the
   * value, and redrawing a chart with the value it already has is a no-op the
   * reader cannot see.
   *
   * The deferral is a timer rather than an animation frame. A frame callback
   * does not run at all while the tab is not compositing — backgrounded, or
   * simply not on screen — so the menu would move and the chart would not
   * follow until something else woke the page up.
   *
   * The one case no page can reach is a *popped-open* native menu in Chrome,
   * where the list is drawn by the operating system and the keystrokes never
   * arrive. Arrowing through a focused, closed menu works everywhere.
   */
  onPick(select, handler) {
    if (!select) return;
    let last = select.value;
    const fire = () => {
      if (select.value === last) return;
      last = select.value;
      handler(select.value);
    };
    select.addEventListener('change', fire);
    select.addEventListener('input', fire);
    select.addEventListener('keydown', event => {
      if (!['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End'].includes(event.key)) {
        return;
      }
      setTimeout(fire, 0);
    });
  },

  /**
   * Put a back and a forward button either side of a menu.
   *
   * Because the keyboard cannot finish the job. A native menu that has been
   * popped open belongs to the operating system, and in Chrome the arrow keys
   * never reach the page at all — so moving through the list to see what each
   * option looks like is impossible with the mouse and unreliable with the
   * keyboard. Two buttons make it a click either way.
   *
   * They dispatch `input` and `change` rather than calling any handler
   * directly, so a menu gains this without knowing about it and whatever was
   * already listening keeps working — including code written before this
   * existed.
   *
   * Inserted as siblings rather than wrapped in a new element: several of
   * these menus sit inside flex rows whose styling reaches them through the
   * parent, and a wrapper would quietly break that.
   */
  addSteppers(select) {
    if (!select || select.dataset.stepped) return;
    // The buttons go either side of the menu, and a node with no parent has no
    // sides — `before` and `after` on a detached element do nothing at all,
    // silently. Callers that build their controls before attaching them ask
    // again once the controls are in the page, so the flag is only set once
    // the buttons are really there.
    if (!select.parentNode) return;
    select.dataset.stepped = '1';

    const step = (by) => {
      const options = [...select.options].filter(o => !o.disabled);
      if (options.length < 2) return;
      const at = options.indexOf(select.selectedOptions[0]);
      const next = options[Math.min(options.length - 1, Math.max(0, at + by))];
      if (!next || next === select.selectedOptions[0]) return;
      select.value = next.value;
      select.dispatchEvent(new Event('input', { bubbles: true }));
      select.dispatchEvent(new Event('change', { bubbles: true }));
      sync();
    };

    const make = (by, glyph, title) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'step-btn';
      button.textContent = glyph;
      button.title = title;
      button.tabIndex = -1;            // the menu itself is the tab stop
      button.addEventListener('click', () => step(by));
      return button;
    };
    const back = make(-1, '\u2039', 'Previous');
    const forward = make(1, '\u203a', 'Next');

    // Greyed at the ends, so the list has a visible beginning and end rather
    // than a button that silently does nothing.
    const sync = () => {
      const options = [...select.options].filter(o => !o.disabled);
      const at = options.indexOf(select.selectedOptions[0]);
      back.disabled = at <= 0;
      forward.disabled = at < 0 || at >= options.length - 1;
    };
    select.addEventListener('change', sync);
    select.addEventListener('input', sync);
    // Most of these menus are filled after they are built, so whoever fills
    // one asks the buttons to look again — otherwise they would be greyed on
    // the strength of an empty list.
    select._syncSteppers = sync;
    sync();

    select.before(back);
    select.after(forward);
  },

  /**
   * Fill a <select> with one domain's metrics.
   * `extras` appends the non-measurement choices a visual setting allows.
   */
  fillSelect(select, domain, { extras = [], selected = null } = {}) {
    if (!select) return;
    const option = m => `<option value="${m.key}">${m.label}</option>`;
    select.innerHTML = extras.map(option).join('') + this.list(domain).map(option).join('');
    if (selected !== null) select.value = selected;
    select._syncSteppers?.();
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
    select._syncSteppers?.();
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

  // Every settings field that holds a metric key. The four visual ones take a
  // bare key, since the domain is fixed by which setting it is; the three
  // chart ones take a domain-qualified one, since they offer both. Not trajX
  // or trajY: those name a series statistic, which is a different vocabulary.
  METRIC_FIELDS: [
    'nodeColorBy', 'nodeSizeBy', 'edgeColorBy', 'edgeWidthBy',
    'distMetric', 'heatX', 'heatY'
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

    // `age` used to be the node id wearing the wrong name, and now it is the
    // duration it always claimed to be. A preset saved under the old meaning
    // asked for birth order and would silently get something else, so it is
    // pointed at the key that kept that meaning.
    for (const field of this.METRIC_FIELDS) {
      const { domain, key } = this.parse(settings[field]);
      if (key !== 'age') continue;
      settings[field] = String(settings[field]).includes(':')
        ? this.qualify(domain, 'node_id')
        : 'node_id';
    }

    return settings;
  }
};
