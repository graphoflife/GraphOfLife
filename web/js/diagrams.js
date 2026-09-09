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
                 iteration: null, span: 1, colormap: 'viridis', reverse: false,
                 title: '', grid: true, guides: [] },
    heatmap:   { x: 'node:tokens', y: 'node:degree', logX: false, logY: false,
                 logCount: true, iteration: null, span: 1,
                 colormap: 'viridis', reverse: false,
                 title: '', grid: true, guides: [] },
    correlate: { x: 'nodes', y: 'tokens', phase: 'all',
                 logX: false, logY: false, colormap: 'viridis', reverse: false,
                 title: '', grid: true, guides: [] },
    // A line is a whole row of choices, so it carries them itself rather than
    // borrowing chart-wide ones: which run, which statistic, which phase, how
    // much of it, and whether it is stretched to the full width.
    timeline:  { lines: [], colormap: 'viridis', reverse: false,
                 title: '', grid: true, guides: [] }
  },

  // One cached series per run, so adding a second line from a run already on
  // the chart costs nothing.
  series: new Map(),
  frames: null,

  // The fields a per-node or per-edge metric is computed from. A frame also
  // carries the cleanup report and the winners and pruned edges beside the
  // allocations, and none of those is plotted here.
  FRAME_FIELDS: ['iteration', 'phase', 'ids', 'tokens', 'edges', 'delta',
                 'ages', 'brain_ids', 'parent_brain_ids', 'decisions.allocations'],

  // Stop pooling once this many values have been gathered. A span measured in
  // iterations is a very different amount of data at sixty agents and at forty
  // thousand — twenty iterations of the larger is tens of megabytes fetched to
  // draw a histogram that thirty bins could not tell apart from two.
  MAX_POOLED: 400000,

  // Frames per request. The backends cap a batch, so a long span is several
  // requests rather than one silently truncated one.
  BATCH: 64,

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

  /** A statistic's name, however it is spelled in that tab's menus. */
  nameOf(key, domain = false) {
    if (domain) {
      const parsed = Metrics.parse(key);
      return Metrics.label(parsed.domain, parsed.key);
    }
    return (Viewer.STAT_LABELS || {})[key] || key;
  },

  /**
   * What this chart would be called if nobody named it.
   *
   * Used for the title, for the suggested filename and for the suggested name
   * of a saved setting — one description, three places, so they agree.
   */
  defaultTitle() {
    const s = this.now;
    const run = this.runs.find(r => r.id === this.runId);
    const of = run ? ` — ${run.name}` : '';
    if (this.active === 'histogram') return `${this.nameOf(s.metric, true)}${of}`;
    if (this.active === 'heatmap') {
      return `${this.nameOf(s.y, true)} vs ${this.nameOf(s.x, true)}${of}`;
    }
    if (this.active === 'correlate') {
      return `${this.nameOf(s.y)} vs ${this.nameOf(s.x)} over time${of}`;
    }
    if (!s.lines.length) return 'Time series';
    const names = [...new Set(s.lines.map(l => this.nameOf(l.stat)))];
    return `${names.join(', ')} over time`;
  },

  get title() { return this.now.title || this.defaultTitle(); },

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
      Metrics.onPick(el, onChange);
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
        Metrics.onPick(el, v => { s.metric = v; this.draw(); });
        Metrics.addSteppers(el);
      } else {
        const x = field('x', metricSelect());
        x.value = s.x;
        Metrics.onPick(x, v => { s.x = v; this.draw(); });
        Metrics.addSteppers(x);
        const y = field('y', metricSelect());
        y.value = s.y;
        Metrics.onPick(y, v => { s.y = v; this.draw(); });
        Metrics.addSteppers(y);
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
      const add = document.createElement('button');
      add.type = 'button';
      add.className = 'ghost small';
      add.textContent = 'Add line';
      add.addEventListener('click', () => {
        const previous = s.lines[s.lines.length - 1];
        s.lines.push(previous
          ? { ...previous }
          : { run: this.runId || (this.runs[0] && this.runs[0].id) || '',
              stat: 'nodes', phase: 'all', stretch: false, maxFrames: null });
        this.controls();
        this.refresh();
      });
      bar.append(add);
    }

    // Shared by every tab: what it is called, what it is compared against,
    // how it is coloured, and getting a picture out.
    const name = document.createElement('input');
    name.type = 'text';
    name.className = 'diagram-title';
    name.value = s.title;
    name.placeholder = this.defaultTitle();
    name.title = 'Chart title. Left blank it describes itself.';
    name.addEventListener('change', () => { s.title = name.value.trim(); this.draw(); });
    field('Title', name);

    const guideAxis = document.createElement('select');
    for (const [v, text] of [['x', 'x'], ['y', 'y']]) {
      const option = document.createElement('option');
      option.value = v; option.textContent = text;
      guideAxis.append(option);
    }
    const guideAt = document.createElement('input');
    guideAt.type = 'number';
    guideAt.step = 'any';
    // Its own class, not the per-line cap's: they are different fields that
              // happen to be the same size, and sharing a name makes each one
              // findable only by counting.
    guideAt.className = 'diagram-guide';
    guideAt.placeholder = 'value';
    const guideAdd = document.createElement('button');
    guideAdd.type = 'button';
    guideAdd.className = 'ghost small';
    guideAdd.textContent = 'Add line at';
    guideAdd.title = 'A constant line to compare against — a threshold, a target';
    guideAdd.addEventListener('click', () => {
      const at = Number(guideAt.value);
      if (!Number.isFinite(at) || guideAt.value === '') return;
      s.guides.push({ axis: guideAxis.value, at });
      guideAt.value = '';
      this.controls();
      this.draw();
    });
    bar.append(guideAdd);
    field('', guideAxis);
    field('=', guideAt);

    const gridToggle = toggle('grid', s.grid, v => {
      s.grid = v; this.controls(); this.draw();
    });

    const style = document.createElement('span');
    style.append(gridToggle);
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

    // The constant lines already added, each with a way to take it off.
    if (s.guides.length) {
      const guides = document.createElement('div');
      guides.className = 'diagram-lines';
      s.guides.forEach((guide, i) => {
        const chip = document.createElement('span');
        chip.className = 'diagram-preset';
        const text = document.createElement('span');
        text.textContent = `${guide.axis} = ${guide.at}`;
        const drop = document.createElement('button');
        drop.type = 'button';
        drop.textContent = '×';
        drop.title = 'Remove this line';
        drop.addEventListener('click', () => {
          s.guides.splice(i, 1); this.controls(); this.draw();
        });
        chip.append(text, drop);
        guides.append(chip);
      });
      this.controlsEl.append(guides);
    }

    if (this.active === 'timeline') this.listLines();

    // Every menu in this bar gets a pair of buttons to step through it with,
    // done here rather than as each is built: they are built detached, and
    // buttons cannot be put either side of an element that has no sides yet.
    for (const menu of bar.querySelectorAll('select')) Metrics.addSteppers(menu);
  },

  /** The series statistics on offer, labelled the way the Viewer labels them. */
  statOptions(runId = this.runId) {
    const payload = this.series.get(runId) || this.series.get(this.runId);
    const keys = payload && payload.keys && payload.keys.length
      ? payload.keys.filter(k => k !== '_frame')
      : Object.keys(Viewer.STAT_LABELS || {});
    return keys.map(k => [k, (Viewer.STAT_LABELS || {})[k] || k]);
  },

  /**
   * One editable row per line.
   *
   * A line is not a label with a delete button — it is the whole set of
   * choices that produced it, still changeable. Which simulation, which
   * statistic, which phase, how many frames of it, and whether it is stretched
   * to the full width so a short run can be compared with a long one by shape.
   */
  listLines() {
    const s = this.now;
    const list = document.createElement('div');
    list.className = 'diagram-lines';

    s.lines.forEach((line, i) => {
      const row = document.createElement('div');
      row.className = 'diagram-line-row';

      const swatch = document.createElement('i');
      swatch.style.background = this.LINE_INK[i % this.LINE_INK.length];
      row.append(swatch);

      const pick = (options, value, onChange, title, live = true) => {
        const el = document.createElement('select');
        for (const [v, text] of options) {
          const option = document.createElement('option');
          option.value = v; option.textContent = text;
          el.append(option);
        }
        el.value = value;
        if (title) el.title = title;
        // Redrawing as the reader moves through a menu is right for a
        // statistic, which is already in hand, and wrong for a simulation,
        // which is a fresh summary each time — arrowing past four runs would
        // start four of them.
        if (live) Metrics.onPick(el, onChange);
        else el.addEventListener('change', () => onChange(el.value));
        row.append(el);
        return el;
      };

      pick(this.runs.map(r => [r.id, r.name]), line.run,
           v => { line.run = v; this.refresh(); }, 'Which simulation', false);
      pick(this.statOptions(line.run), line.stat,
           v => { line.stat = v; this.draw(); }, 'Which statistic');
      pick([['all', 'Both phases'], ['1', 'Reproduction'], ['2', 'Game']], line.phase,
           v => { line.phase = v; this.draw(); }, 'Which phase');

      // Blank is every frame there is, which is what the number shows when it
      // is left alone rather than an empty box that gives no idea of the scale.
      const cap = document.createElement('input');
      cap.type = 'number';
      cap.min = '2';
      cap.className = 'diagram-cap';
      cap.value = line.maxFrames === null ? '' : String(line.maxFrames);
      cap.placeholder = String(this.availableFrames(line));
      cap.title = 'How many frames of this line to show. Blank is all of them.';
      cap.addEventListener('change', () => {
        line.maxFrames = cap.value === '' ? null : Math.max(2, Number(cap.value));
        this.draw();
      });
      row.append(cap);

      const stretch = document.createElement('button');
      stretch.type = 'button';
      stretch.className = 'axis-btn' + (line.stretch ? ' active' : '');
      stretch.textContent = 'stretch';
      stretch.title = 'Give this line the full width, whatever its length, so runs '
        + 'of different lengths can be compared by shape';
      stretch.addEventListener('click', () => {
        line.stretch = !line.stretch;
        this.controls();
        this.draw();
      });
      row.append(stretch);

      const drop = document.createElement('button');
      drop.type = 'button';
      drop.className = 'diagram-drop';
      drop.textContent = '×';
      drop.title = 'Remove this line';
      drop.addEventListener('click', () => {
        s.lines.splice(i, 1);
        this.controls();
        this.refresh();
      });
      row.append(drop);
      list.append(row);
    });
    this.controlsEl.append(list);
  },

  /** How many frames a line could show, which is what its cap defaults to. */
  availableFrames(line) {
    const payload = this.series.get(line.run);
    if (!payload) return 0;
    const phases = (payload.series || {}).phase || [];
    if (line.phase === 'all') return phases.length;
    return phases.filter(p => String(p) === line.phase).length;
  },

  // ---- saved settings ---------------------------------------------------

  allPresets() {
    try { return JSON.parse(localStorage.getItem(this.STORE)) || {}; }
    catch (err) { return {}; }
  },

  savePreset() {
    // Prefilled with what the chart already calls itself, so keeping a setting
    // is one keystroke rather than an invitation to invent a name.
    const name = prompt('Name for these settings', this.title);
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

    const suggested = prompt('Save the picture as', `${this.title}.png`);
    if (!suggested) return;
    const link = document.createElement('a');
    link.download = suggested.endsWith('.png') ? suggested : `${suggested}.png`;
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
    // What the tab was showing when this started. Checked between requests, so
    // walking away stops the work rather than leaving it running unseen.
    const epoch = Research.epoch;

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

      // One frame further back than the window needs, and each frame is given
      // the one before it. "Before phase" metrics — token curvature above all
      // — are computed against the previous graph, so without this they are
      // every node NaN and the chart reports no values at all.
      const lead = from > 0 ? 1 : 0;

      this.say('Reading frames…');
      try {
        // A batch is capped server-side, so a long span is several requests
        // rather than one truncated one. Reading them in order also means each
        // frame can be handed the one before it, which is what the
        // "before phase" metrics are computed against.
        const wanted = count + lead;
        const read = [];
        let pooled = 0;
        for (let at = 0; at < wanted; at += this.BATCH) {
          const size = Math.min(this.BATCH, wanted - at);
          const reply = await API.getFrames(this.runId, from - lead + at, size,
                                            this.FRAME_FIELDS, this.MAX_POOLED - pooled);
          if (this.token !== token) return;
          const batch = reply.frames || [];
          read.push(...batch);
          pooled += batch.reduce((n, f) => n + (f.ids || []).length, 0);
          this.say(`Reading frames… ${formatNumber(read.length)} of `
                   + `${formatNumber(wanted)}`);
          // Either the run ran out or the value budget did; both mean stop.
          if (batch.length < size || pooled >= this.MAX_POOLED) break;
        }
        for (let i = 1; i < read.length; i++) read[i].previous = read[i - 1];
        this.frames = read.slice(lead);
        this.asked = count;
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
          cancelled: () => this.token !== token || Research.stale(epoch),
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

  /** Title, axis names, grid, constant lines and legend, in one place. */
  chromeFor(xLabel, yLabel, legend = []) {
    const s = this.now;
    return { title: this.title, xLabel, yLabel, legend,
             ticks: true, grid: s.grid !== false, guides: s.guides || [] };
  },

  draw() {
    if (!this.canvas) return;
    const s = this.now;
    const ink = { colormap: s.colormap, reverse: s.reverse };

    // The suggested title describes what is being drawn, so it has to follow
    // the drawing. Set when the controls are built, it went stale the moment a
    // metric changed and offered the name of the chart before this one.
    const name = this.controlsEl && this.controlsEl.querySelector('.diagram-title');
    if (name) name.placeholder = this.defaultTitle();

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
          format: v => Metrics.format(parsed.domain, parsed.key, v),
          chrome: this.chromeFor(this.nameOf(s.metric, true), 'how many')
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
        formatY: v => Metrics.format(y.domain, y.key, v),
        chrome: this.chromeFor(this.nameOf(s.x, true), this.nameOf(s.y, true))
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
      let points = [];
      for (let i = 0; i < xs.length; i++) {
        if (s.phase !== 'all' && String(phases[i]) !== s.phase) continue;
        if (!usable(xs[i]) || !usable(ys[i])) continue;
        points.push({ x: xs[i], y: ys[i], t: iterations[i] });
      }

      // Some pairs are never recorded on the same frame — births belong to the
      // reproduction phase and revolutions to the game — so on a frame either
      // one or the other is missing and pairing them frame by frame gives
      // nothing at all. Falling back to one point per *iteration* pairs the two
      // halves of it, which is the only pairing those two have.
      //
      // The phase filter is deliberately ignored in that branch: keeping it
      // would leave one of the two with nothing, and being here at all means
      // they live on opposite halves of an iteration.
      let pairing = 'one point per frame';
      if (points.length < 2) {
        const byIteration = new Map();
        for (let i = 0; i < xs.length; i++) {
          const at = iterations[i];
          let slot = byIteration.get(at);
          if (!slot) { slot = { x: null, y: null, t: at }; byIteration.set(at, slot); }
          if (slot.x === null && usable(xs[i])) slot.x = xs[i];
          if (slot.y === null && usable(ys[i])) slot.y = ys[i];
        }
        const paired = [...byIteration.values()]
          .filter(p => p.x !== null && p.y !== null)
          .sort((a, b) => a.t - b.t);
        if (paired.length >= 2) {
          points = paired;
          pairing = 'one point per iteration, across its phases';
        }
      }

      drawTrajectory(this.canvas, points, {
        ...ink, logX: s.logX, logY: s.logY,
        xLabel: this.nameOf(s.x), yLabel: this.nameOf(s.y),
        footer: `${formatNumber(points.length)} points · colour is time`,
        chrome: this.chromeFor(this.nameOf(s.x), this.nameOf(s.y))
      });
      this.say(`${formatNumber(points.length)} points, ${pairing}`
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
    // Say when the span was cut short, rather than quietly drawing less than
    // was asked for and letting the reader believe otherwise. Two things can
    // cut it: the value budget, or simply running out of run.
    const short = this.asked && this.frames.length < this.asked
      ? (seen >= this.MAX_POOLED * 0.95
         ? ` — stopped at ${formatNumber(this.MAX_POOLED)} values`
         : ` of ${formatNumber(this.asked)} asked for`)
      : '';
    return `iterations ${at}, ${formatNumber(this.frames.length)} frames, `
      + `${formatNumber(seen)} values pooled${short}`;
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

    const tracks = [];
    for (const [i, line] of s.lines.entries()) {
      const payload = this.series.get(line.run);
      if (!payload) continue;
      const series = payload.series || {};
      const values = series[line.stat];
      if (!values) continue;
      const phases = series.phase || [], iterations = series.iteration || [];
      let points = [];
      for (let k = 0; k < values.length; k++) {
        if (line.phase !== 'all' && String(phases[k]) !== line.phase) continue;
        const v = values[k];
        if (v === null || v === undefined || !Number.isFinite(v)) continue;
        points.push({ t: iterations[k], v });
      }
      // A cap takes the *first* frames rather than thinning them, so the line
      // is the run's opening at full detail rather than the whole run at
      // lower. Thinning is what the sampled series already did once.
      if (line.maxFrames && points.length > line.maxFrames) {
        points = points.slice(0, line.maxFrames);
      }
      if (points.length < 2) continue;
      const vs = points.map(p => p.v);
      const run = this.runs.find(r => r.id === line.run);
      tracks.push({
        line, points, colour: this.LINE_INK[i % this.LINE_INK.length],
        lo: Math.min(...vs), hi: Math.max(...vs),
        firstT: points[0].t, lastT: points[points.length - 1].t,
        label: `${this.nameOf(line.stat)} — ${run ? run.name : line.run}`
          + (line.phase === 'all' ? '' :
             line.phase === '1' ? ' · reproduction' : ' · game')
      });
    }

    const chrome = this.chromeFor('iterations', 'per line, see legend',
                                  tracks.map(t => ({ label: t.label, colour: t.colour })));
    const pad = _chromePad(chrome);
    const { ctx, w, h, outer } = _prepareCanvas(canvas, pad);

    if (!s.lines.length || !tracks.length) {
      ctx.fillStyle = 'rgba(255,255,255,0.45)';
      ctx.font = '13px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(s.lines.length ? 'Reading…'
                   : 'Add a line: pick a simulation, a statistic, and a phase.',
                   w / 2, h / 2);
      ctx.textAlign = 'left';
      if (!s.lines.length) this.say('');
      return;
    }

    // Absolute lines share one iteration axis, so a run half as long stops
    // halfway across. A stretched line gets the whole width whatever its
    // length, which is what lets two runs be compared by shape.
    const longest = Math.max(...tracks.map(t => t.lastT), 1);
    _axes(ctx, w, h, {
      x: { lo: 0, hi: longest, format: v => formatNumber(Math.round(v)) },
      grid: chrome.grid, guides: (s.guides || []).filter(g => g.axis === 'x')
    });

    for (const track of tracks) {
      if (track.hi === track.lo) track.hi = track.lo + 1;
      const span = track.line.stretch ? (track.lastT - track.firstT) || 1 : longest;
      const base = track.line.stretch ? track.firstT : 0;
      const xAt = t => ((t - base) / span) * w;
      const yAt = v => (1 - (v - track.lo) / (track.hi - track.lo)) * h;

      // Each line keeps its own vertical scale: these are different quantities
      // in different units, and one shared axis flattens whichever has the
      // smaller spread. The legend carries every range.
      for (const guide of (s.guides || []).filter(g => g.axis === 'y')) {
        if (guide.at < track.lo || guide.at > track.hi) continue;
        ctx.save();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = track.colour;
        ctx.globalAlpha = 0.5;
        const at = Math.round(yAt(guide.at)) + 0.5;
        ctx.beginPath(); ctx.moveTo(0, at); ctx.lineTo(w, at); ctx.stroke();
        ctx.restore();
      }

      ctx.strokeStyle = track.colour;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      track.points.forEach((p, k) => {
        const x = xAt(p.t), y = yAt(p.v);
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    const round = v => (Math.abs(v) >= 100 ? v.toFixed(0)
                      : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(4));
    chrome.legend = tracks.map(t => ({
      colour: t.colour,
      label: `${t.label}   ${round(t.lo)}–${round(t.hi)}`
        + (t.line.stretch ? '   (stretched)' : '')
        + (t.line.maxFrames ? `   first ${formatNumber(t.points.length)}` : '')
    }));
    _chrome(ctx, pad, outer, chrome);
    this.say(`${tracks.length} line${tracks.length === 1 ? '' : 's'}`);
  },

  // The Research tab asks every view for these; a chart has no layout of its
  // own to measure, so drawing is all either of them needs to do.
  resize() { },
};
