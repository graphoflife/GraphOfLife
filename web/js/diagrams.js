/*
 * The viewer's four charts, off the viewer.
 *
 * In the Viewer a chart is tied to the frame currently on screen and to the one
 * run being watched. Here they are instruments in their own right: pick the
 * iterations, pick the statistics, keep the settings, save the picture — and,
 * for the time series, put lines from *different* simulations on one pair of
 * axes.
 *
 * The drawing is not reimplemented. `drawHistogram`, `drawHeatmap` and
 * `drawTrajectory` live in stats.js and are the same functions the Viewer
 * calls, so a chart here and a chart there cannot disagree about what a
 * histogram of the same numbers looks like. Only the multi-line time chart is
 * new, because nothing in the Viewer draws several runs at once.
 *
 * Two kinds of tab, and the difference decides everything about how they load:
 *
 *   frame   a distribution *within* one moment — the histogram and the
 *           heatmap. Needs whole frames, so it reads a few and pools them.
 *   series  a quantity *across* time — the correlation and the time series.
 *           Needs the run summary, which arrives progressively.
 */
const Diagrams = {

  TABS: [
    { id: 'histogram', label: 'Histogram',   kind: 'frame'  },
    { id: 'heatmap',   label: 'Heatmap',     kind: 'frame'  },
    { id: 'correlate', label: 'Correlation', kind: 'series' },
    { id: 'timeline',  label: 'Time series', kind: 'series' }
  ],

  /** Where saved settings live, so they survive a reload. */
  STORE: 'gol.diagrams.presets',

  // Colours a line gets when several share one chart, in order. Taken from the
  // page's own palette rather than a colormap, because a colormap is a
  // gradient and these have to be told apart, not ordered.
  LINE_INK: ['#5ac8fa', '#ffd166', '#ff6b6b', '#7ee787', '#c792ea',
             '#f78c6c', '#89ddff', '#e5e5e5'],

  active: 'histogram',
  runs: [],
  runId: null,

  // Per tab, because the whole point is that each is configured on its own.
  settings: {
    histogram: { metric: 'node:tokens', logX: false, logY: false,
                 iteration: null, span: 1, colormap: 'viridis', reverse: false },
    heatmap:   { x: 'node:tokens', y: 'node:degree', logX: false, logY: false,
                 logCount: true, iteration: null, span: 1,
                 colormap: 'viridis', reverse: false },
    correlate: { x: 'nodes', y: 'tokens', phase: 'all',
                 logX: false, logY: false, colormap: 'viridis', reverse: false },
    timeline:  { lines: [], align: 'absolute', colormap: 'viridis', reverse: false }
  },

  // One cached series per run, so adding a second line from a run already on
  // the chart costs nothing.
  series: new Map(),
  frames: null,

  // ---- setup ------------------------------------------------------------

  init() {
    this.host = document.getElementById('research-diagrams');
    if (!this.host) return;

    this.tabBar = document.getElementById('diagramTabs');
    this.controlsEl = document.getElementById('diagramControls');
    this.presetsEl = document.getElementById('diagramPresets');
    this.noteEl = document.getElementById('diagramNote');
    this.canvas = document.getElementById('diagramCanvas');

    this.tabBar.replaceChildren(...this.TABS.map(tab => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = tab.label;
      button.dataset.tab = tab.id;
      button.addEventListener('click', () => this.show(tab.id));
      return button;
    }));

    if (window.ResizeObserver) {
      new ResizeObserver(() => this.draw()).observe(this.canvas.parentElement);
    }
    this.show(this.active);
  },

  setRuns(runs) {
    this.runs = runs;
    if (this.controlsEl && this.controlsEl.childElementCount) this.controls();
  },

  say(text) { if (this.noteEl) this.noteEl.textContent = text; },

  get tab() { return this.TABS.find(t => t.id === this.active); },
  get now() { return this.settings[this.active]; },

  show(id) {
    this.active = id;
    for (const button of this.tabBar.querySelectorAll('button')) {
      button.classList.toggle('active', button.dataset.tab === id);
    }
    this.controls();
    this.listPresets();
    this.refresh();
  },

  // ---- controls ---------------------------------------------------------

  /**
   * Build this tab's controls.
   *
   * Rebuilt on every tab change rather than kept as four hidden panels: the
   * tabs share almost no controls, and four sets of stale inputs is four sets
   * of state to keep in step with the settings they are meant to reflect.
   */
  controls() {
    const bar = this.controlsEl;
    bar.replaceChildren();
    const s = this.now;

    const field = (label, node) => {
      const wrap = document.createElement('label');
      wrap.className = 'lineage-filter';
      if (label) wrap.append(document.createTextNode(label));
      wrap.append(node);
      bar.append(wrap);
      return node;
    };
    const select = (options, value, onChange) => {
      const el = document.createElement('select');
      for (const [v, text] of options) {
        const option = document.createElement('option');
        option.value = v; option.textContent = text;
        el.append(option);
      }
      el.value = value;
      el.addEventListener('change', () => { onChange(el.value); });
      return el;
    };
    const number = (value, min, onChange) => {
      const el = document.createElement('input');
      el.type = 'number'; el.min = String(min); el.value = value === null ? '' : String(value);
      el.addEventListener('change', () => onChange(el.value === '' ? null : Number(el.value)));
      return el;
    };
    const toggle = (label, on, onChange) => {
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'axis-btn' + (on ? ' active' : '');
      el.textContent = label;
      el.addEventListener('click', () => onChange(!on));
      return el;
    };

    if (this.tab.kind === 'frame') {
      const metricSelect = () => {
        const el = document.createElement('select');
        Metrics.fillDomainSelect(el, null);
        return el;
      };
      if (this.active === 'histogram') {
        const el = field('Metric', metricSelect());
        el.value = s.metric;
        el.addEventListener('change', () => { s.metric = el.value; this.draw(); });
      } else {
        const x = field('x', metricSelect());
        x.value = s.x;
        x.addEventListener('change', () => { s.x = x.value; this.draw(); });
        const y = field('y', metricSelect());
        y.value = s.y;
        y.addEventListener('change', () => { s.y = y.value; this.draw(); });
      }

      // Which moment. Blank means the last recorded one, which is the useful
      // default — you almost always want to see where a run got to.
      field('At iteration', number(s.iteration, 0, v => {
        s.iteration = v; this.refresh();
      })).placeholder = 'last';
      field('summed over', number(s.span, 1, v => {
        s.span = Math.max(1, v || 1); this.refresh();
      })).title = 'How many consecutive iterations to pool into one distribution';
      bar.lastChild.append(document.createTextNode(' iterations'));

      const axes = document.createElement('span');
      axes.className = 'axis-toggles';
      axes.append(toggle('log x', s.logX, v => { s.logX = v; this.controls(); this.draw(); }));
      axes.append(toggle('log y', s.logY, v => { s.logY = v; this.controls(); this.draw(); }));
      if (this.active === 'heatmap') {
        axes.append(toggle('log n', s.logCount,
                           v => { s.logCount = v; this.controls(); this.draw(); }));
      }
      bar.append(axes);
    }

    if (this.active === 'correlate') {
      const keys = this.statOptions();
      const x = field('x', select(keys, s.x, v => { s.x = v; this.draw(); }));
      const y = field('y', select(keys, s.y, v => { s.y = v; this.draw(); }));
      x.title = 'Horizontal axis'; y.title = 'Vertical axis';
      field('Phase', select([['all', 'Both phases'], ['1', 'Reproduction'], ['2', 'Game']],
                            s.phase, v => { s.phase = v; this.draw(); }));
      const axes = document.createElement('span');
      axes.className = 'axis-toggles';
      axes.append(toggle('log x', s.logX, v => { s.logX = v; this.controls(); this.draw(); }));
      axes.append(toggle('log y', s.logY, v => { s.logY = v; this.controls(); this.draw(); }));
      bar.append(axes);
    }

    if (this.active === 'timeline') {
      const runOptions = this.runs.map(r => [r.id, r.name]);
      const pick = { run: this.runId || (this.runs[0] && this.runs[0].id) || '',
                     stat: 'nodes', phase: 'all' };
      const runEl = field('Add', select(runOptions, pick.run, v => { pick.run = v; }));
      runEl.title = 'Which simulation the line comes from';
      const statEl = field('', select(this.statOptions(), pick.stat, v => { pick.stat = v; }));
      const phaseEl = field('', select([['all', 'Both phases'], ['1', 'Reproduction'],
                                        ['2', 'Game']], pick.phase, v => { pick.phase = v; }));
      const add = document.createElement('button');
      add.type = 'button';
      add.className = 'ghost small';
      add.textContent = 'Add line';
      add.addEventListener('click', () => {
        if (!runEl.value) return;
        s.lines.push({ run: runEl.value, stat: statEl.value, phase: phaseEl.value });
        // Rebuild the controls, not just the chart: the list of lines lives in
        // the control bar, so a line added without this is drawn but cannot be
        // seen in the legend or taken off again.
        this.controls();
        this.refresh();
      });
      bar.append(add);

      field('Align', select([['absolute', 'Absolute — a shorter run stops early'],
                             ['stretch', 'Stretched — every run fills the width']],
                            s.align, v => { s.align = v; this.draw(); }));
    }

    // Shared by every tab: how it is coloured, and getting a picture out.
    const style = document.createElement('span');
    style.className = 'axis-toggles';
    const maps = Object.keys(COLORMAPS).map(k => [k, k]);
    const mapEl = select(maps, s.colormap, v => { s.colormap = v; this.draw(); });
    mapEl.title = 'Colour style';
    style.append(mapEl);
    style.append(toggle('flip', s.reverse, v => { s.reverse = v; this.controls(); this.draw(); }));

    const save = document.createElement('button');
    save.type = 'button';
    save.className = 'ghost small';
    save.textContent = 'Save image';
    save.addEventListener('click', () => this.saveImage());
    style.append(save);

    const keep = document.createElement('button');
    keep.type = 'button';
    keep.className = 'ghost small';
    keep.textContent = 'Save settings';
    keep.addEventListener('click', () => this.savePreset());
    style.append(keep);
    bar.append(style);

    if (this.active === 'timeline') this.listLines();
  },

  /** The series statistics on offer, labelled the way the Viewer labels them. */
  statOptions() {
    const payload = this.series.get(this.runId);
    const keys = payload && payload.keys && payload.keys.length
      ? payload.keys.filter(k => k !== '_frame')
      : Object.keys(Viewer.STAT_LABELS || {});
    return keys.map(k => [k, (Viewer.STAT_LABELS || {})[k] || k]);
  },

  /** The lines on the time chart, with a way to take one off again. */
  listLines() {
    const s = this.now;
    const list = document.createElement('div');
    list.className = 'diagram-lines';
    s.lines.forEach((line, i) => {
      const run = this.runs.find(r => r.id === line.run);
      const chip = document.createElement('span');
      chip.className = 'diagram-line';
      const swatch = document.createElement('i');
      swatch.style.background = this.LINE_INK[i % this.LINE_INK.length];
      const phase = line.phase === 'all' ? '' :
                    line.phase === '1' ? ' · reproduction' : ' · game';
      chip.append(swatch, document.createTextNode(
        `${(Viewer.STAT_LABELS || {})[line.stat] || line.stat} — `
        + `${run ? run.name : line.run}${phase}`));
      const drop = document.createElement('button');
      drop.type = 'button';
      drop.textContent = '×';
      drop.title = 'Remove this line';
      drop.addEventListener('click', () => {
        s.lines.splice(i, 1);
        this.controls();
        this.refresh();
      });
      chip.append(drop);
      list.append(chip);
    });
    this.controlsEl.append(list);
  },

  // ---- saved settings ---------------------------------------------------

  allPresets() {
    try { return JSON.parse(localStorage.getItem(this.STORE)) || {}; }
    catch (err) { return {}; }
  },

  savePreset() {
    const name = prompt('Name for these settings');
    if (!name) return;
    const all = this.allPresets();
    (all[this.active] = all[this.active] || []).push({
      name, at: Date.now(), runId: this.runId,
      settings: JSON.parse(JSON.stringify(this.now))
    });
    try { localStorage.setItem(this.STORE, JSON.stringify(all)); }
    catch (err) { this.say(`Could not save: ${err.message}`); return; }
    this.listPresets();
  },

  listPresets() {
    const saved = (this.allPresets()[this.active]) || [];
    this.presetsEl.replaceChildren();
    if (!saved.length) return;

    const title = document.createElement('span');
    title.className = 'diagram-preset-title';
    title.textContent = 'Saved:';
    this.presetsEl.append(title);

    saved.forEach((preset, i) => {
      const chip = document.createElement('span');
      chip.className = 'diagram-preset';
      const open = document.createElement('button');
      open.type = 'button';
      open.textContent = preset.name;
      open.title = 'Show this again';
      open.addEventListener('click', () => {
        this.settings[this.active] = JSON.parse(JSON.stringify(preset.settings));
        // A saved chart is of a particular simulation, so it takes its run
        // with it — restoring the settings and leaving the run behind would
        // draw something the reader never saved.
        if (preset.runId && this.runs.some(r => r.id === preset.runId)) {
          Research.picker.value = preset.runId;
          Research.userPicked = true;
          Research.open(preset.runId);
        }
        this.controls();
        this.refresh();
      });
      const drop = document.createElement('button');
      drop.type = 'button';
      drop.textContent = '×';
      drop.title = 'Forget these settings';
      drop.addEventListener('click', () => {
        const all = this.allPresets();
        all[this.active].splice(i, 1);
        localStorage.setItem(this.STORE, JSON.stringify(all));
        this.listPresets();
      });
      chip.append(open, drop);
      this.presetsEl.append(chip);
    });
  },

  // ---- getting a picture out --------------------------------------------

  /**
   * The chart as a PNG, on an opaque background.
   *
   * The canvas is transparent where nothing was drawn, and a transparent PNG
   * dropped into a document or a slide comes out as black-on-black or
   * white-on-white depending on where it lands. So the page's own background
   * is painted underneath first.
   */
  saveImage() {
    const source = this.canvas;
    const out = document.createElement('canvas');
    out.width = source.width;
    out.height = source.height;
    const ctx = out.getContext('2d');
    ctx.fillStyle = getComputedStyle(document.body).backgroundColor || '#0d1117';
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(source, 0, 0);

    const link = document.createElement('a');
    const run = this.runs.find(r => r.id === this.runId);
    link.download = `${(run ? run.name : 'graphoflife').replace(/\W+/g, '-')}`
      + `-${this.active}.png`;
    link.href = out.toDataURL('image/png');
    link.click();
  },

  // ---- loading ----------------------------------------------------------

  async load(runId) {
    this.runId = runId;
    this.frames = null;
    if (this.controlsEl && this.controlsEl.childElementCount) this.controls();
    await this.refresh();
  },

  /** Fetch whatever this tab needs, then draw. */
  async refresh() {
    if (!this.canvas) return;
    const token = (this.token = (this.token || 0) + 1);

    if (this.tab.kind === 'frame') {
      if (!this.runId) { this.frames = null; this.say('Choose a simulation above.'); this.draw(); return; }
      const run = this.runs.find(r => r.id === this.runId);
      if (!run) return;
      const s = this.now;

      // Frames come in pairs, one per phase. A blank iteration means the last
      // one recorded.
      const iterations = Math.max(1, Math.floor(run.frame_count / 2));
      const last = iterations - 1;
      const at = s.iteration === null ? last : Math.max(0, Math.min(last, s.iteration));
      const span = Math.max(1, s.span || 1);
      const from = Math.max(0, (at - span + 1)) * 2;
      const count = Math.min(span * 2, run.frame_count - from);

      this.say('Reading frames…');
      try {
        const reply = await API.getFrames(this.runId, from, count, null);
        if (this.token !== token) return;
        this.frames = reply.frames || [];
      } catch (err) {
        if (this.token !== token) return;
        this.say(`Could not read the frames: ${err.message}`);
        this.frames = null;
      }
      this.draw();
      return;
    }

    // A series tab. Cached per run, and climbed so a long run draws early.
    if (!this.needsSeries().length) { this.draw(); return; }
    for (const id of this.needsSeries()) {
      if (this.series.has(id)) continue;
      this.say('Reading the run…');
      try {
        await SeriesLoad.climb(id, {
          cancelled: () => this.token !== token,
          onStep: payload => {
            this.series.set(id, payload);
            this.draw();
          }
        });
      } catch (err) {
        if (this.token !== token) return;
        this.say(`Could not read ${id}: ${err.message}`);
      }
    }
    if (this.token !== token) return;
    this.draw();
  },

  /** Which runs this tab needs summarised — several, on the time chart. */
  needsSeries() {
    if (this.active === 'timeline') {
      return [...new Set(this.now.lines.map(l => l.run))];
    }
    return this.runId ? [this.runId] : [];
  },

  // ---- drawing ----------------------------------------------------------

  draw() {
    if (!this.canvas) return;
    const s = this.now;
    const ink = { colormap: s.colormap, reverse: s.reverse };

    if (this.active === 'histogram' || this.active === 'heatmap') {
      if (!this.frames || !this.frames.length) {
        drawHistogram(this.canvas, null, ink);
        return;
      }
      // Every frame in the window contributes its nodes, so a span of several
      // iterations is one distribution over all of them rather than several
      // charts side by side.
      const pool = key => {
        const parsed = Metrics.parse(key);
        const out = [];
        for (const frame of this.frames) {
          const metrics = new FrameMetrics(frame, Viewer.settings || {});
          const values = parsed.domain === 'edge'
            ? metrics.edgeValues(parsed.key) : metrics.nodeValues(parsed.key);
          for (const v of values) out.push(v);
        }
        return out;
      };

      if (this.active === 'histogram') {
        const parsed = Metrics.parse(s.metric);
        drawHistogram(this.canvas, pool(s.metric), {
          ...ink, bins: 30, logScale: s.logX, logCount: s.logY,
          signed: Metrics.isSigned(parsed.domain, parsed.key),
          format: v => Metrics.format(parsed.domain, parsed.key, v)
        });
        this.say(this.frameNote());
        return;
      }

      const x = Metrics.parse(s.x), y = Metrics.parse(s.y);
      if (x.domain !== y.domain) {
        drawHeatmap(this.canvas, null, null, {
          message: 'Pick two node metrics or two edge metrics — mixing them has no pairing.'
        });
        return;
      }
      drawHeatmap(this.canvas, pool(s.x), pool(s.y), {
        ...ink, logX: s.logX, logY: s.logY, logCount: s.logCount,
        signedX: Metrics.isSigned(x.domain, x.key),
        signedY: Metrics.isSigned(y.domain, y.key),
        formatX: v => Metrics.format(x.domain, x.key, v),
        formatY: v => Metrics.format(y.domain, y.key, v)
      });
      this.say(this.frameNote());
      return;
    }

    if (this.active === 'correlate') {
      const payload = this.series.get(this.runId);
      if (!payload) { drawTrajectory(this.canvas, null, { ...ink, message: 'Choose a simulation above.' }); return; }
      const series = payload.series || {};
      const xs = series[s.x], ys = series[s.y];
      if (!xs || !ys) {
        drawTrajectory(this.canvas, null, { ...ink, message: 'This run has no history for one of these.' });
        return;
      }
      const phases = series.phase || [], iterations = series.iteration || [];
      const usable = v => v !== null && v !== undefined && Number.isFinite(v);
      const points = [];
      for (let i = 0; i < xs.length; i++) {
        if (s.phase !== 'all' && String(phases[i]) !== s.phase) continue;
        if (!usable(xs[i]) || !usable(ys[i])) continue;
        points.push({ x: xs[i], y: ys[i], t: iterations[i] });
      }
      drawTrajectory(this.canvas, points, {
        ...ink, logX: s.logX, logY: s.logY,
        xLabel: (Viewer.STAT_LABELS || {})[s.x] || s.x,
        yLabel: (Viewer.STAT_LABELS || {})[s.y] || s.y,
        footer: `${formatNumber(points.length)} frames · colour is time`
      });
      this.say(`${formatNumber(points.length)} points`
        + (payload.complete ? '' : ' — still refining'));
      return;
    }

    this.drawTimeline(ink);
  },

  frameNote() {
    const s = this.now;
    const seen = this.frames.reduce((n, f) => n + (f.ids || []).length, 0);
    const at = this.frames.length
      ? `${formatNumber(this.frames[0].iteration)}–`
        + `${formatNumber(this.frames[this.frames.length - 1].iteration)}`
      : '—';
    return `iterations ${at}, ${formatNumber(this.frames.length)} frames, `
      + `${formatNumber(seen)} values pooled`;
  },

  /**
   * Several statistics, possibly from several runs, on one pair of axes.
   *
   * Each line keeps its own vertical scale, because these are different
   * quantities in different units and forcing them onto one axis flattens
   * whichever has the smaller spread. The legend carries each range, so the
   * shapes are comparable even though the scales are not.
   *
   * Horizontally there is a real choice, and it is the reader's: **absolute**
   * puts every line on the same iteration axis, so a run half as long stops
   * halfway across; **stretched** gives each line the full width, so two runs
   * of different lengths can be compared by shape.
   */
  drawTimeline(ink) {
    const s = this.now;
    const canvas = this.canvas;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const box = canvas.getBoundingClientRect();
    if (box.width < 1 || box.height < 1) return;
    canvas.width = Math.round(box.width * dpr);
    canvas.height = Math.round(box.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, box.width, box.height);

    if (!s.lines.length) {
      ctx.fillStyle = 'rgba(255,255,255,0.45)';
      ctx.font = '13px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Add a line: pick a simulation, a statistic, and a phase.',
                   box.width / 2, box.height / 2);
      ctx.textAlign = 'left';
      this.say('');
      return;
    }

    const tracks = [];
    for (const [i, line] of s.lines.entries()) {
      const payload = this.series.get(line.run);
      if (!payload) continue;
      const series = payload.series || {};
      const values = series[line.stat];
      if (!values) continue;
      const phases = series.phase || [], iterations = series.iteration || [];
      const points = [];
      for (let k = 0; k < values.length; k++) {
        if (line.phase !== 'all' && String(phases[k]) !== line.phase) continue;
        const v = values[k];
        if (v === null || v === undefined || !Number.isFinite(v)) continue;
        points.push({ t: iterations[k], v });
      }
      if (points.length < 2) continue;
      const vs = points.map(p => p.v);
      tracks.push({
        line, points, colour: this.LINE_INK[i % this.LINE_INK.length],
        lo: Math.min(...vs), hi: Math.max(...vs),
        lastT: points[points.length - 1].t
      });
    }

    if (!tracks.length) {
      ctx.fillStyle = 'rgba(255,255,255,0.45)';
      ctx.font = '13px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Reading…', box.width / 2, box.height / 2);
      ctx.textAlign = 'left';
      return;
    }

    const pad = { left: 44, right: 14, top: 12, bottom: 26 };
    const plotW = Math.max(1, box.width - pad.left - pad.right);
    const plotH = Math.max(1, box.height - pad.top - pad.bottom);
    const longest = Math.max(...tracks.map(t => t.lastT), 1);

    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.left + 0.5, pad.top + 0.5, plotW, plotH);

    for (const track of tracks) {
      if (track.hi === track.lo) track.hi = track.lo + 1;
      const span = s.align === 'stretch' ? track.lastT : longest;
      const xAt = t => pad.left + (span > 0 ? t / span : 0.5) * plotW;
      const yAt = v => pad.top + (1 - (v - track.lo) / (track.hi - track.lo)) * plotH;

      ctx.strokeStyle = track.colour;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      track.points.forEach((p, k) => {
        const x = xAt(p.t), y = yAt(p.v);
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText('0', pad.left, box.height - 9);
    const right = s.align === 'stretch'
      ? 'each run, whole' : `${formatNumber(longest)} iterations`;
    ctx.textAlign = 'right';
    ctx.fillText(right, pad.left + plotW, box.height - 9);
    ctx.textAlign = 'left';

    const round = v => (Math.abs(v) >= 100 ? v.toFixed(0)
                      : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(4));
    this.say(tracks.map(t => {
      const run = this.runs.find(r => r.id === t.line.run);
      return `${(Viewer.STAT_LABELS || {})[t.line.stat] || t.line.stat}`
        + ` (${run ? run.name : t.line.run}) ${round(t.lo)}–${round(t.hi)}`;
    }).join('   ·   '));
  },

  // The Research tab asks every view for these; a chart has no layout of its
  // own to measure, so drawing is all either of them needs to do.
  resize() { },
};
