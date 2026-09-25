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
    // `cutoff` drops the opening iterations from every line. A run's first
    // moments are the seeded graph shaking itself out — a swing far larger than
    // anything that follows — and on a shared axis that opening transient sets
    // the scale for the whole chart, flattening the part worth looking at.
    timeline:  { lines: [], cutoff: 50, logX: false, logY: false,
                 colormap: 'viridis', reverse: false,
                 title: '', grid: true, guides: [] }
  },

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

  // ---- setup ------------------------------------------------------------

  init() {
    this.host = document.getElementById('research-diagrams');
    if (!this.host) return;

    this.tabBar = document.getElementById('diagramTabs');
    this.controlsEl = document.getElementById('diagramControls');
    this.presetsEl = document.getElementById('diagramPresets');
    this.noteEl = document.getElementById('diagramNote');
    this.meaningsEl = document.getElementById('diagramMeanings');
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
    for (const run of runs) SeriesLoad.noteSize(run.id, run.frame_count);
    if (this.controlsEl && this.controlsEl.childElementCount) this.controls();
  },

  say(text) { if (this.noteEl) this.noteEl.textContent = text; },

  /**
   * Fill the time series in from a thesis: its statistics, of one run, and
   * the value its claim is stated in.
   */
  applyThesis(id) {
    const thesis = THESES.find(t => t.id === id);
    if (!thesis) return;
    const run = this.runId || (this.runs[0] && this.runs[0].id) || '';
    const s = this.settings.timeline;
    s.lines = thesis.stats.map(stat => ({ run, stat, phase: 'all',
                                          stretch: false, maxFrames: null }));
    s.guides = thesis.guides.map(g => ({ ...g }));
    s.title = thesis.title;
    if (this.active !== 'timeline') this.show('timeline');
    else { this.controls(); this.refresh(); }
  },

  /**
   * The thesis the chart is making, if any: the one whose statistics are
   * exactly the ones plotted.
   *
   * Worked out from the lines rather than remembered from the last button
   * pressed, which it used to be — a stored choice checked against the lines
   * every time. The run is free to change, which is how one claim is compared
   * across runs, but add or swap a statistic and the chart is no longer the
   * argument the prose is making, so the prose goes. A chart put together by
   * hand that plots a thesis's statistics is making its argument too.
   */
  activeThesis() {
    const plotted = new Set(this.settings.timeline.lines.map(l => l.stat));
    return THESES.find(t => t.stats.length === plotted.size
                            && t.stats.every(stat => plotted.has(stat))) || null;
  },

  /**
   * What each statistic on the chart measures, printed under it.
   *
   * A chart of four lines named "Bridges" and "Cheeger" is unreadable to
   * anyone who does not already know what those are, and the explanation lives
   * one dialog away in the Viewer — far enough that nobody goes. It is the
   * same text from the same place, so there is one description of a statistic
   * and not two that can disagree.
   *
   * Each entry is keyed to its line by colour, and a statistic drawn twice —
   * two runs, two phases — is described once.
   */
  explain(tracks) {
    if (!this.meaningsEl) return;
    this.meaningsEl.replaceChildren();
    if (!tracks || !tracks.length) return;

    // A thesis argues something, and the statistics below are its evidence,
    // so the argument comes first.
    const thesis = this.active === 'timeline' ? this.activeThesis() : null;
    if (thesis) {
      const box = document.createElement('div');
      box.className = 'diagram-thesis';
      const head = document.createElement('h4');
      head.textContent = thesis.title;
      box.append(head);
      for (const [key, label] of [['claim', 'The claim'], ['why', 'Why it would be so'],
                                  ['confirm', 'It holds if'], ['refute', 'It fails if'],
                                  ['note', 'Note']]) {
        if (!thesis[key]) continue;
        const para = document.createElement('p');
        const tag = document.createElement('b');
        tag.textContent = label;
        para.append(tag, document.createTextNode(' ' + thesis[key]));
        box.append(para);
      }
      this.meaningsEl.append(box);
    }

    const seen = new Set();
    for (const track of tracks) {
      const key = track.line.stat;
      if (seen.has(key)) continue;
      seen.add(key);
      const text = (StatDetail.EXPLANATIONS || {})[key];
      if (!text) continue;

      const entry = document.createElement('p');
      const swatch = document.createElement('i');
      swatch.style.background = track.colour;
      const name = document.createElement('b');
      name.textContent = this.nameOf(key);
      entry.append(swatch, name, document.createTextNode(' ' + text));
      this.meaningsEl.append(entry);
    }
  },

  /** A statistic's name, however it is spelled in that tab's menus. */
  nameOf(key, domain = false) {
    if (domain) {
      const parsed = Metrics.parse(key);
      return Metrics.label(parsed.domain, parsed.key);
    }
    return (Viewer.STAT_LABELS || {})[key]
      || (SeriesLoad.DERIVED[key] && SeriesLoad.DERIVED[key].label) || key;
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
          Research.pick(preset.runId);
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

  /**
   * Bring the chart up to date: load whatever it lacks, then draw.
   *
   * Every control ends here, so none of them has to know whether what it
   * changed needs data. They used to choose between this and draw() one by
   * one, and once the depth a run is loaded to depended on which statistics
   * are plotted, the two that change a statistic were still choosing draw():
   * switching a line to bridges left it empty, the chart saying "Reading…"
   * with nothing reading.
   *
   * Nothing missing draws at once and starts no job, so the bar does not
   * flash on every keystroke in the title — and whatever is still loading is
   * for a chart no longer on screen, so it stops. A load already under way for
   * exactly what is missing is left to finish. Anything else is one job for
   * the whole tab, and starting it cancels the one before.
   */
  async refresh() {
    if (!this.canvas) return;
    const wants = this.wants();
    if (!wants) {
      Jobs.cancel(this);
      this.draw();
      return;
    }
    if (wants === this.wanted && Jobs.busy(this)) {
      this.draw();
      return;
    }
    this.wanted = wants;
    return Jobs.run(this, 'Reading', (job) => this._refresh(job), err => {
      const frames = this.tab.kind === 'frame';
      if (frames) this.frames = null;
      this.say(`Could not read the ${frames ? 'frames' : 'run'}: ${err.message}`);
      this.draw();
    });
  },

  /**
   * What the chart needs and does not have yet, named so that two asks for
   * the same thing compare equal — or null when it has everything.
   */
  wants() {
    if (this.tab.kind === 'frame') {
      const span = this.frameSpan();
      return span && !(this.frames && this.framesFor === span.key) ? span.key : null;
    }
    const needs = [...this.needsSeries()];
    if (needs.every(([id, stats]) => SeriesLoad.ready(id, stats, this.frameCount(id)))) return null;
    return needs.map(([id, stats]) => `${id}:${[...new Set(stats)].sort().join('+')}`)
      .sort().join(' ');
  },

  /**
   * The frames a frame tab reads — where they start, how many, and a key
   * naming exactly that read. Null without a run to read them from.
   */
  frameSpan() {
    const run = this.runs.find(r => r.id === this.runId);
    if (!run) return null;
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
    return { from, count, lead, key: `${run.id}:${from}:${count}` };
  },

  async _refresh(job) {
    if (this.tab.kind === 'frame') {
      const { from, count, lead, key } = this.frameSpan();
      this.say('Reading frames…');
      // In order, so each frame can be handed the one before it, which is
      // what the "before phase" metrics are computed against; and only until
      // MAX_POOLED values have been gathered.
      const indices = Array.from({ length: count + lead }, (_, k) => from - lead + k);
      const read = [];
      for await (const batch of FrameWindow.read(this.runId, indices, this.FRAME_FIELDS,
                                                 this.MAX_POOLED, { signal: job.signal })) {
        read.push(...batch);
        job.report(read.length, indices.length, 'Reading frames');
      }
      for (let i = 1; i < read.length; i++) read[i].previous = read[i - 1];
      this.frames = read.slice(lead);
      this.framesFor = key;
      this.asked = count;
      this.draw();
      return;
    }

    // A series tab. Climbed so a long run draws early, and only as deep as the
    // statistics plotted from it need. A run whose history is already enough
    // costs nothing, which is what makes adding a second line from it free; a
    // run cut short last time carries on from where it stopped.
    const wanted = this.needsSeries();
    if (!wanted.size) { this.draw(); return; }
    for (const [id, stats] of wanted) {
      if (SeriesLoad.ready(id, stats, this.frameCount(id))) continue;
      if (!SeriesLoad.cache.has(id)) this.say('Reading the run…');
      try {
        await SeriesLoad.climb(id, stats, {
          job,
          text: wanted.size > 1 ? `Summarising ${id}` : 'Summarising the run',
          onStep: () => this.draw()
        });
      } catch (err) {
        // One run that cannot be read does not keep the others off the
        // chart. Being stopped does.
        if (err.name === 'AbortError') throw err;
        this.say(`Could not read ${id}: ${err.message}`);
      }
    }
    this.draw();
  },

  /**
   * How many frames a run has, as Research last listed it — which it does on
   * every entry to the tab, so a run that has grown since is asked for again.
   * A run missing from the list has been deleted or is too young to chart,
   * and has nothing newer to ask for.
   */
  frameCount(runId) {
    const run = this.runs.find(r => r.id === runId);
    return run ? run.frame_count : 0;
  },

  /**
   * Which runs this tab needs summarised, and which statistics of each —
   * several runs on the time chart, one on the correlation.
   */
  needsSeries() {
    const wanted = new Map();
    const want = (run, stat) => {
      if (!run) return;
      if (!wanted.has(run)) wanted.set(run, []);
      wanted.get(run).push(stat);
    };
    if (this.active === 'timeline') {
      for (const line of this.now.lines) want(line.run, line.stat);
    } else {
      want(this.runId, this.now.x);
      want(this.runId, this.now.y);
    }
    return wanted;
  },

  // ---- drawing ----------------------------------------------------------

  //: frame -> its FrameMetrics. The pooled values depend on the frame alone,
  //: and the chart is drawn again on every resize and every keystroke in its
  //: title, twice over for a heatmap; it used to build every frame's metrics
  //: afresh each time.
  _metrics: new WeakMap(),

  metricsOf(frame) {
    let metrics = this._metrics.get(frame);
    if (!metrics) {
      metrics = new FrameMetrics(frame);
      this._metrics.set(frame, metrics);
    }
    return metrics;
  },

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

    // Only the time series names several statistics at once and so is the only
    // chart that needs them spelled out under it; the others say what they are
    // on their own axes.
    if (this.active !== 'timeline') this.explain(null);

    if (this.active === 'histogram' || this.active === 'heatmap') {
      if (!this.frames || !this.frames.length) {
        drawHistogram(this.canvas, null, ink);
        if (!this.runId) this.say('Choose a simulation above.');
        return;
      }
      // Every frame in the window contributes its nodes, so a span of several
      // iterations is one distribution over all of them rather than several
      // charts side by side.
      const pool = key => {
        const parsed = Metrics.parse(key);
        const out = [];
        for (const frame of this.frames) {
          const metrics = this.metricsOf(frame);
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
      const payload = SeriesLoad.cache.get(this.runId);
      if (!payload) { drawTrajectory(this.canvas, null, { ...ink, message: 'Choose a simulation above.' }); return; }
      const found = SeriesLoad.pairs(payload.series || {}, s.x, s.y,
                                     phase => s.phase === 'all' || String(phase) === s.phase);
      if (!found) {
        drawTrajectory(this.canvas, null, { ...ink, message: 'This run has no history for one of these.' });
        return;
      }
      const points = found.points;
      const pairing = found.pairing || 'one point per frame';

      drawTrajectory(this.canvas, points, {
        ...ink, logX: s.logX, logY: s.logY,
        xLabel: this.nameOf(s.x), yLabel: this.nameOf(s.y),
        footer: `${formatNumber(points.length)} points · colour is time`,
        chrome: this.chromeFor(this.nameOf(s.x), this.nameOf(s.y))
      });
      this.say(`${formatNumber(points.length)} points, ${pairing}`
        + (SeriesLoad.ready(this.runId, [s.x, s.y], this.frameCount(this.runId))
           ? '' : ' — still refining'));
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
    // Whether any line had data that the cutoff then removed. Without this a
    // cutoff past the end of every run is indistinguishable from a chart still
    // loading, and the reader is told to wait for something that will never
    // arrive.
    let cutAway = false;
    for (const [i, line] of s.lines.entries()) {
      const payload = SeriesLoad.cache.get(line.run);
      if (!payload) continue;
      const series = payload.series || {};
      const values = SeriesLoad.column(series, line.stat);
      if (!values) continue;
      const phases = series.phase || [], iterations = series.iteration || [];
      // The opening iterations, dropped before anything else looks at the
      // numbers, so they set neither the vertical scale nor the frame count.
      const from = Math.max(0, Math.round(s.cutoff || 0));
      let points = [];
      let held = 0;
      for (let k = 0; k < values.length; k++) {
        if (line.phase !== 'all' && String(phases[k]) !== line.phase) continue;
        const v = values[k];
        if (v === null || v === undefined || !Number.isFinite(v)) continue;
        held++;
        if (iterations[k] < from) continue;
        points.push({ t: iterations[k], v });
      }
      if (held >= 2 && points.length < 2) cutAway = true;
      // A cap takes the *first* frames rather than thinning them, so the line
      // is the run's opening at full detail rather than the whole run at
      // lower. Thinning is what the sampled series already did once.
      if (line.maxFrames && points.length > line.maxFrames) {
        points = points.slice(0, line.maxFrames);
      }
      if (points.length < 2) continue;
      // A log scale is applied to the values themselves, so the vertical range
      // and everything drawn against it are in the same units. Signed, because
      // several of these statistics go negative and log of a negative number is
      // not a reason to drop the line.
      const vs = points.map(p => (s.logY ? Metrics.applyLog(p.v, p.v < 0) : p.v));
      const run = this.runs.find(r => r.id === line.run);
      tracks.push({
        line, points, colour: this.LINE_INK[i % this.LINE_INK.length],
        mapped: vs,
        lo: Math.min(...vs), hi: Math.max(...vs),
        rawLo: Math.min(...points.map(p => p.v)),
        rawHi: Math.max(...points.map(p => p.v)),
        firstT: points[0].t, lastT: points[points.length - 1].t,
        label: `${this.nameOf(line.stat)} — ${run ? run.name : line.run}`
          + (line.phase === 'all' ? '' :
             line.phase === '1' ? ' · reproduction' : ' · game')
      });
    }

    // One line can name its own vertical axis; several cannot share one, so the
    // axis says what it is actually showing instead of naming a unit no line is
    // in. Both are truthful about the same drawing.
    const yName = tracks.length === 1
      ? this.nameOf(tracks[0].line.stat) + (s.logY ? ' (log)' : '')
      : 'share of each line’s own range';
    const chrome = this.chromeFor('iterations' + (s.logX ? ' (log)' : ''), yName,
                                  tracks.map(t => ({ label: t.label, colour: t.colour })));
    const pad = _chromePad(chrome);
    const { ctx, w, h, outer } = _prepareCanvas(canvas, pad);

    if (!s.lines.length || !tracks.length) {
      ctx.fillStyle = Ink.of('dim');
      ctx.font = '13px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(cutAway
                   ? `Nothing left after iteration ${formatNumber(s.cutoff)} — lower the cutoff.`
                   : s.lines.length ? 'Reading…'
                   : 'Add a line: pick a simulation, a statistic, and a phase.',
                   w / 2, h / 2);
      ctx.textAlign = 'left';
      if (!s.lines.length) this.say('');
      else if (cutAway) this.say(`the cutoff removed every frame of `
                                 + `${s.lines.length === 1 ? 'the line' : 'every line'}`);
      this.explain(null);
      return;
    }

    // Absolute lines share one iteration axis, so a run half as long stops
    // halfway across. A stretched line gets the whole width whatever its
    // length, which is what lets two runs be compared by shape. The axis starts
    // at the cutoff rather than at zero: the dropped iterations are not on the
    // chart, and leaving room for them would waste the width they used to fill.
    const earliest = Math.min(...tracks.map(t => t.firstT));
    const longest = Math.max(...tracks.map(t => t.lastT), 1);
    const mapT = t => (s.logX ? Metrics.applyLog(t, false) : t);
    const loT = mapT(earliest), hiT = Math.max(mapT(longest), loT + 1e-9);

    // With one line the vertical axis can carry that line's real units. With
    // several it cannot — they are different quantities on different scales,
    // each normalised to its own range — so it carries that normalisation
    // honestly, as a percentage, and the legend gives every line its range.
    const single = tracks.length === 1 ? tracks[0] : null;

    // A lone line's scale takes in its constant lines, with a little room
    // beyond. A threshold is most worth seeing when the data stays clear of
    // it: "no cut costs more than a tenth" is confirmed exactly when the line
    // never reaches 0.1, and a range fitted to the data alone left the
    // threshold off the chart in precisely that case.
    const yGuides = (s.guides || []).filter(g => g.axis === 'y');
    const mapY = v => (s.logY ? Metrics.applyLog(v, v < 0) : v);
    if (single) {
      for (const guide of yGuides) {
        const v = mapY(guide.at);
        if (v > single.hi) single.hi = v + 0.05 * (v - single.lo);
        if (v < single.lo) single.lo = v - 0.05 * (single.hi - v);
      }
    }
    if (single && single.hi === single.lo) single.hi = single.lo + 1;
    _axes(ctx, w, h, {
      x: { lo: loT, hi: hiT,
           format: v => formatNumber(Math.round(s.logX ? Metrics.undoLog(v, false) : v)) },
      y: single
        ? { lo: single.lo, hi: single.hi,
            format: v => formatNumber(s.logY ? Metrics.undoLog(v, v < 0) : v) }
        : { lo: 0, hi: 100, format: v => `${Math.round(v)}%` },
      grid: chrome.grid, guides: (s.guides || []).filter(g => g.axis === 'x'),
      pad
    });

    for (const track of tracks) {
      if (track.hi === track.lo) track.hi = track.lo + 1;
      const span = track.line.stretch ? (mapT(track.lastT) - mapT(track.firstT)) || 1
                                      : (hiT - loT);
      const base = track.line.stretch ? mapT(track.firstT) : loT;
      const xAt = t => ((mapT(t) - base) / span) * w;
      const yAt = v => (1 - (v - track.lo) / (track.hi - track.lo)) * h;

      // Each line keeps its own vertical scale: these are different quantities
      // in different units, and one shared axis flattens whichever has the
      // smaller spread. The legend carries every range. A constant line is
      // given in real units, so it is mapped the same way the values were
      // before it is compared against them.
      for (const guide of yGuides) {
        const value = mapY(guide.at);
        if (value < track.lo || value > track.hi) continue;
        ctx.save();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = track.colour;
        ctx.globalAlpha = 0.5;
        const at = Math.round(yAt(value)) + 0.5;
        ctx.beginPath(); ctx.moveTo(0, at); ctx.lineTo(w, at); ctx.stroke();
        // A thesis names its threshold; a line added by hand has no name.
        if (guide.label) {
          ctx.globalAlpha = 0.85;
          ctx.fillStyle = track.colour;
          ctx.font = '11px system-ui, sans-serif';
          ctx.fillText(guide.label, 6, at - 5);
        }
        ctx.restore();
      }

      ctx.strokeStyle = track.colour;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      track.points.forEach((p, k) => {
        const x = xAt(p.t), y = yAt(track.mapped[k]);
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    const round = v => (Math.abs(v) >= 100 ? v.toFixed(0)
                      : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(4));
    // Ranges are quoted in the statistic's own units whatever the axis is
    // doing, because that is the number the reader recognises.
    chrome.legend = tracks.map(t => ({
      colour: t.colour,
      label: `${t.label}   ${round(t.rawLo)}–${round(t.rawHi)}`
        + (t.line.stretch ? '   (stretched)' : '')
        + (t.line.maxFrames ? `   first ${formatNumber(t.points.length)}` : '')
    }));
    _chrome(ctx, pad, outer, chrome);
    this.say(`${tracks.length} line${tracks.length === 1 ? '' : 's'}`
             + (s.cutoff ? ` · from iteration ${formatNumber(s.cutoff)}` : ''));
    this.explain(tracks);
  },

  // The Research tab asks every view for these; a chart has no layout of its
  // own to measure, so drawing is all either of them needs to do.
  resize() { },
};
