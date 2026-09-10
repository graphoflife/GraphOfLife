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
    if (this.controlsEl && this.controlsEl.childElementCount) this.controls();
  },

  say(text) { if (this.noteEl) this.noteEl.textContent = text; },

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

    // Only the time series names several statistics at once and so is the only
    // chart that needs them spelled out under it; the others say what they are
    // on their own axes.
    if (this.active !== 'timeline') this.explain(null);

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
    // Whether any line had data that the cutoff then removed. Without this a
    // cutoff past the end of every run is indistinguishable from a chart still
    // loading, and the reader is told to wait for something that will never
    // arrive.
    let cutAway = false;
    for (const [i, line] of s.lines.entries()) {
      const payload = this.series.get(line.run);
      if (!payload) continue;
      const series = payload.series || {};
      const values = series[line.stat];
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
      ctx.fillStyle = 'rgba(255,255,255,0.45)';
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
      for (const guide of (s.guides || []).filter(g => g.axis === 'y')) {
        if (guide.at < track.rawLo || guide.at > track.rawHi) continue;
        ctx.save();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = track.colour;
        ctx.globalAlpha = 0.5;
        const value = s.logY ? Metrics.applyLog(guide.at, guide.at < 0) : guide.at;
        const at = Math.round(yAt(value)) + 0.5;
        ctx.beginPath(); ctx.moveTo(0, at); ctx.lineTo(w, at); ctx.stroke();
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
