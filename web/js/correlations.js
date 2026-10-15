/*
 * What goes with what: every pair of quantities in a window of a run, or over
 * every simulation the book has measured, correlated and drawn.
 *
 * Four kinds of point (the level):
 *
 *   agents   every agent in each game of the window: its tokens, its
 *            neighbourhood, its place in the network, what it staked, and
 *            what then happened to it;
 *   stats    every game: the world's statistics after it;
 *   changes  the same, as the change from one game to the next. Two
 *            statistics that both drift over a run correlate whatever they
 *            have to do with each other; their changes do not share the drift;
 *   worlds   every simulation in book/results/worlds.json.
 *
 * The matrix on the left is every pair, coloured by r; a cell opens the pair
 * on the right: its points, and the mean of y in bins of x with the middle
 * half of y as a band, on linear or logarithmic axes. All the arithmetic is
 * gol_atlas, answered by the server or by the same Python in the browser's
 * worker — nothing is computed here, only drawn.
 */
const Correlations = {
  ...WindowView,
  runs: [],
  runId: null,
  level: 'agents',
  colourBy: 'best',
  summary: null,
  detail: null,
  pair: null,
  hovered: null,
  // What the window is made from, by level, as starting values: a game of a
  // large world is thousands of agents, so a few games are plenty for the
  // agents; a correlation over games needs a couple of hundred.
  SPANS: { agents: 20, stats: 200, changes: 200 },
  STEP: 2,
  // The outcomes of an agent look ten iterations ahead (gol_atlas.HORIZON), so
  // a window opened by default ends that far before the run does.
  AHEAD: 11,
  // Kept apart from the view, so asking for a pair does not cancel the table
  // it is a pair of, and a new table does cancel the pair.
  pairJobs: { name: 'a pair of the correlations' },

  init() {
    const el = id => document.getElementById(id);
    this.canvas = el('atlasMatrix');
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.scatter = el('atlasScatter');
    this.noteEl = el('atlasNote');
    this.readoutEl = el('atlasReadout');
    this.listEl = el('atlasList');
    this.spanEl = el('atlasSpan');
    this.stepEl = el('atlasStep');
    this.windowEl = el('atlasWindow');
    this.periodEl = el('atlasPeriod');
    this.hideEl = el('atlasHide');
    this.xlogEl = el('atlasXLog');
    this.ylogEl = el('atlasYLog');

    for (const button of document.querySelectorAll('#atlasLevel .seg-btn')) {
      button.addEventListener('click', () => this.setLevel(button.dataset.level));
    }
    for (const button of document.querySelectorAll('#atlasColour .seg-btn')) {
      button.addEventListener('click', () => {
        this.colourBy = button.dataset.colour;
        for (const other of document.querySelectorAll('#atlasColour .seg-btn')) {
          other.classList.toggle('active', other === button);
        }
        this.drawMatrix();
      });
    }
    this.spanEl.value = String(this.SPANS[this.level]);
    this.stepEl.value = String(this.STEP);
    this.spanEl.addEventListener('change', () => this.load(this.runId, this.windowStart));
    this.stepEl.addEventListener('change', () => this.load(this.runId, this.windowStart));
    this.windowEl.addEventListener('change', () => this.load(this.runId, Number(this.windowEl.value)));
    this.periodEl.addEventListener('change', () => this.loadWorlds());
    this.hideEl.addEventListener('change', () => { this.drawMatrix(); this.list(); });
    this.xlogEl.addEventListener('change', () => this.loadPair());
    this.ylogEl.addEventListener('change', () => this.loadPair());
    el('atlasSwap').addEventListener('click', () => {
      if (!this.pair) return;
      this.pair = [this.pair[1], this.pair[0]];
      [this.xlogEl.checked, this.ylogEl.checked] = [this.ylogEl.checked, this.xlogEl.checked];
      this.loadPair();
    });

    this.canvas.addEventListener('mousemove', e => {
      const box = this.canvas.getBoundingClientRect();
      const at = this.cells ? this.cells.at(e.clientX - box.left, e.clientY - box.top) : null;
      if (String(at) === String(this.hovered)) return;
      this.hovered = at;
      this.drawMatrix();
      this.readoutEl.textContent = at ? this.describePair(at[0], at[1]) : '';
    });
    this.canvas.addEventListener('mouseleave', () => {
      this.hovered = null;
      this.drawMatrix();
      this.readoutEl.textContent = '';
    });
    this.canvas.addEventListener('click', e => {
      const box = this.canvas.getBoundingClientRect();
      const at = this.cells ? this.cells.at(e.clientX - box.left, e.clientY - box.top) : null;
      if (at) this.choose(at[0], at[1]);
    });
    this.scatter.addEventListener('mousemove', e => this.hoverPoint(e));
    this.listEl.addEventListener('click', e => {
      const item = e.target.closest('[data-pair]');
      if (item) this.choose(...item.dataset.pair.split(',').map(Number));
    });
    if (window.ResizeObserver) {
      new ResizeObserver(() => this.draw()).observe(this.canvas.parentElement.parentElement);
    }
  },

  // ---- reading -----------------------------------------------------------

  setLevel(level) {
    if (level === this.level) return;
    this.level = level;
    for (const button of document.querySelectorAll('#atlasLevel .seg-btn')) {
      button.classList.toggle('active', button.dataset.level === level);
    }
    const worlds = level === 'worlds';
    for (const el of document.querySelectorAll('#research-correlations .atlas-run')) el.hidden = worlds;
    for (const el of document.querySelectorAll('#research-correlations .atlas-agents')) el.hidden = level !== 'agents';
    for (const el of document.querySelectorAll('#research-correlations .atlas-worlds')) el.hidden = !worlds;
    if (!worlds) this.spanEl.value = String(this.SPANS[level]);
    this.windowStart = null;
    if (worlds) this.loadWorlds();
    else this.load(this.runId);
  },

  async load(runId, from = null) {
    if (this.level === 'worlds') return this.loadWorlds();
    const sameRun = runId === this.runId;
    this.runId = runId;
    const run = this.runs.find(r => r.id === runId);
    if (!run) return;
    const span = Math.max(2, Number(this.spanEl.value) || this.SPANS[this.level]);
    const ahead = this.level === 'agents' ? this.AHEAD : 0;
    const at = from ?? (sameRun && this.windowStart != null ? this.windowStart
      : run.frame_count - FrameWindow.PHASES * (span + ahead));
    const plan = FrameWindow.plan(run.frame_count, at, span);
    this.windowStart = plan.start;
    FrameWindow.bindScrubber(this.windowEl, plan);
    const step = this.level === 'agents' ? Math.max(1, Number(this.stepEl.value) || this.STEP) : 1;
    return this.read(`Correlating ${this.level === 'agents' ? 'the agents' : 'the statistics'} of a window`,
      signal => API.getAtlas(runId, this.level, plan.start, plan.indices.length, step,
                             null, null, false, false, { signal }));
  },

  async loadWorlds() {
    return this.read('Correlating every world',
      signal => API.getWorldsAtlas(this.periodEl.value || '', null, null, false, false, { signal }));
  },

  /** Ask for a table's summary, then open its strongest pair. */
  read(label, ask) {
    Jobs.cancel(this.pairJobs);
    this.summary = null;
    this.detail = null;
    this.pair = null;
    this.draw();
    return Jobs.run(this, label, async job => {
      this.say('Reading and correlating every pair…');
      this.summary = await ask(job.signal);
      if (this.level === 'worlds') this.offerPeriods();
      this.describe();
      this.list();
      this.drawMatrix();
      // The strongest pair to start with, but one that is not a yes or a no:
      // a cloud squeezed onto two lines is a poor first picture.
      const kinds = this.summary.variables.map(v => v.kind);
      const ranked = this.strongest();
      const first = ranked.find(([i, j]) => kinds[i] !== 'bool' && kinds[j] !== 'bool') || ranked[0];
      if (first) this.choose(first[0], first[1]);
      else this.drawScatter();
    }, err => this.say(`Could not correlate: ${err.message}`));
  },

  /** One pair, on the axes ticked. */
  loadPair() {
    if (!this.summary || !this.pair) return null;
    const [i, j] = this.pair;
    const a = this.names()[i], b = this.names()[j];
    const xlog = this.xlogEl.checked, ylog = this.ylogEl.checked;
    return Jobs.run(this.pairJobs, `Reading ${this.label(a)} against ${this.label(b)}`, async job => {
      this.detail = this.level === 'worlds'
        ? await API.getWorldsAtlas(this.periodEl.value || '', a, b, xlog, ylog, { signal: job.signal })
        : await API.getAtlas(this.runId, this.level, this.windowStart,
                             this.spanCount(), this.stepCount(), a, b, xlog, ylog, { signal: job.signal });
      this.marked = -1;
      this.drawScatter();
      this.readoutEl.textContent = this.describeDetail();
      this.explainPair();
    }, err => this.say(`Could not read the pair: ${err.message}`));
  },

  spanCount() {
    const run = this.runs.find(r => r.id === this.runId);
    const span = Math.max(2, Number(this.spanEl.value) || this.SPANS[this.level]);
    return FrameWindow.plan(run ? run.frame_count : 0, this.windowStart, span).indices.length;
  },

  stepCount() {
    return this.level === 'agents' ? Math.max(1, Number(this.stepEl.value) || this.STEP) : 1;
  },

  /** Open a pair, on the axes on which it is strongest. */
  choose(i, j) {
    const p = this.find(i, j);
    if (!p) return;
    // The summary names the axes as "x-y" for the pair in its own order.
    const best = p[3] || 'lin-lin';
    const [xa, ya] = (p[0] === i ? best : best.split('-').reverse().join('-')).split('-');
    this.pair = [i, j];
    this.xlogEl.checked = xa === 'log';
    this.ylogEl.checked = ya === 'log';
    this.xlogEl.disabled = !this.loggable(i);
    this.ylogEl.disabled = !this.loggable(j);
    this.drawMatrix();
    this.loadPair();
  },

  offerPeriods() {
    const periods = this.summary.periods || [];
    if (!periods.length || this.periodEl.options.length) return;
    for (const period of periods) {
      const option = document.createElement('option');
      option.value = period;
      option.textContent = period;
      this.periodEl.appendChild(option);
    }
  },

  // ---- what the summary says --------------------------------------------

  names() {
    return this.summary ? this.summary.variables.map(v => v.name) : [];
  },

  /** A world statistic by the name the rest of the site gives it; an agent's quantity by its own. */
  label(name) {
    if (this.level === 'agents' || typeof RunStats === 'undefined') return name;
    const label = RunStats.label(name);
    return label && label !== name ? label : name;
  },

  about(name) {
    const variable = this.summary && this.summary.variables.find(v => v.name === name);
    if (variable && variable.about) return variable.about;
    return (typeof RunStats !== 'undefined' && RunStats.EXPLANATIONS[name]) || '';
  },

  loggable(i) {
    return ['pos', 'count', 'signed'].includes(this.summary.variables[i].kind);
  },

  /** The summary's entry for a pair, whichever way round it is asked for. */
  find(i, j) {
    if (!this.summary) return null;
    if (!this._index) this._index = new Map();
    if (this._indexed !== this.summary) {
      this._index.clear();
      for (const p of this.summary.pairs) this._index.set(`${p[0]},${p[1]}`, p);
      this._indexed = this.summary;
    }
    return this._index.get(i < j ? `${i},${j}` : `${j},${i}`) || null;
  },

  /** The value a cell is coloured by: r on the best axes, on linear ones, or by rank. */
  cellValue(i, j) {
    const p = this.find(i, j);
    if (!p) return null;
    return this.colourBy === 'linear' ? p[2] : this.colourBy === 'rank' ? p[5] : p[4];
  },

  strongest() {
    if (!this.summary) return [];
    const hide = this.hideEl.checked;
    return this.summary.pairs
      .filter(p => p[4] !== null && (!hide || !p[7]))
      .sort((p, q) => Math.abs(q[4]) - Math.abs(p[4]))
      .slice(0, 24)
      .map(p => [p[0], p[1]]);
  },

  describePair(i, j) {
    const p = this.find(i, j);
    const names = this.names();
    if (!p) return `${this.label(names[i])} and ${this.label(names[j])}: too few rows to say`;
    const r = v => (v === null || v === undefined ? '–' : v.toFixed(2));
    return `${this.label(names[i])} and ${this.label(names[j])}: r = ${r(p[2])} on linear axes, `
      + `${r(p[4])} on ${this.axesText(p[3])}, rank ${r(p[5])}, over ${formatNumber(p[6])} rows`
      + (p[7] ? ' — related by how they are made' : '');
  },

  axesText(axes) {
    const [x, y] = (axes || 'lin-lin').split('-');
    if (x === 'lin' && y === 'lin') return 'linear axes';
    if (x === 'log' && y === 'log') return 'both axes logarithmic';
    return x === 'log' ? 'a logarithmic x' : 'a logarithmic y';
  },

  describeDetail() {
    const d = this.detail;
    if (!d) return '';
    const r = v => (v === null || v === undefined ? '–' : v.toFixed(3));
    let text = `${this.label(d.a)} against ${this.label(d.b)}: r = ${r(d.r)} on these axes, `
      + `${r(d.lin)} on linear ones, rank ${r(d.rank)}, ${formatNumber(d.n)} points`;
    if (d.definitional) text += '. The two are related by how they are made, so their correlation says little about the world';
    return text + '.';
  },

  /** What the two quantities of the open pair are, under the picture. */
  explainPair() {
    const el = document.getElementById('atlasAbout');
    if (!el || !this.detail) return;
    el.innerHTML = '';
    for (const name of [this.detail.a, this.detail.b]) {
      const line = document.createElement('p');
      const term = document.createElement('b');
      term.textContent = this.label(name);
      line.append(term, ` — ${this.about(name) || name}`);
      el.appendChild(line);
    }
  },

  describe() {
    const s = this.summary;
    if (!s) return;
    let note = `${formatNumber(s.rows)} ${this.level === 'agents' ? 'agent-games' : this.level === 'worlds' ? 'worlds' : 'rows'}`
      + ` (${s.note}), ${s.variables.length} quantities, ${formatNumber(s.pairs.length)} pairs.`;
    const missing = s.variables.filter(v => !v.finite).map(v => v.name);
    if (missing.length) note += ` No values for ${missing.join(', ')}.`;
    note += ' Click a cell, or a line below, to see the pair.';
    this.say(note);
  },

  list() {
    this.listEl.innerHTML = '';
    if (!this.summary) return;
    const names = this.names();
    for (const [i, j] of this.strongest()) {
      const p = this.find(i, j);
      const item = document.createElement('li');
      item.dataset.pair = `${i},${j}`;
      item.textContent = `${this.label(names[i])} · ${this.label(names[j])}   r = ${p[4].toFixed(2)}`
        + ` (${this.axesText(p[3])}), rank ${p[5] === null ? '–' : p[5].toFixed(2)}`;
      this.listEl.appendChild(item);
    }
  },

  // ---- drawing -----------------------------------------------------------

  resize() {},

  draw() {
    this.drawMatrix();
    this.drawScatter();
  },

  drawMatrix() {
    if (!this.canvas) return;
    const hide = this.hideEl && this.hideEl.checked;
    this.cells = drawMatrix(this.canvas, {
      names: this.names(),
      value: (i, j) => this.cellValue(i, j),
      muted: (i, j) => { const p = this.find(i, j); return !!(p && p[7] && hide); },
      selected: this.pair,
      hovered: this.hovered,
      message: this.summary ? 'nothing to correlate' : 'no window read yet'
    });
  },

  drawScatter() {
    if (!this.scatter) return;
    const d = this.detail;
    if (!d) {
      this.points = drawScatter(this.scatter, { message: this.pair ? 'reading…' : 'pick a pair' });
      return;
    }
    let colours = null;
    if (d.points.group) {
      const order = [...new Set(d.points.group)];
      this.groups = order;
      colours = d.points.group.map(g => Ink.line(order.indexOf(g)));
    }
    this.points = drawScatter(this.scatter, {
      points: d.points, bins: d.bins, ticks: d.ticks || {}, colours, marked: this.marked ?? -1,
      chrome: {
        ticks: true,
        xLabel: this.label(d.a) + (d.xlog ? ' (log)' : ''),
        yLabel: this.label(d.b) + (d.ylog ? ' (log)' : ''),
        legend: colours ? this.groups.slice(0, 8).map((g, k) => ({ colour: Ink.line(k), label: g })) : null
      }
    });
  },

  /** On every world, which world a point is. */
  hoverPoint(e) {
    const d = this.detail;
    if (!d || !this.points || !d.points.label) return;
    const box = this.scatter.getBoundingClientRect();
    const i = this.points.nearest(e.clientX - box.left, e.clientY - box.top);
    if (i === this.marked) return;
    this.marked = i;
    this.drawScatter();
    this.readoutEl.textContent = i >= 0
      ? `${d.points.label[i]}${d.points.group ? ` (${d.points.group[i]})` : ''}`
      : this.describeDetail();
  }
};
