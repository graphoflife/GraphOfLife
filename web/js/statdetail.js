/*
 * Clicking a statistic opens this: what the number means, and how it moved
 * across the whole run.
 *
 * The history comes from /api/runs/<id>/series, which reduces every frame to a
 * handful of scalars on the server. Pulling thousands of full frames into the
 * browser just to plot one line would be far slower.
 *
 * The plotted points respect the active phase filter, so looking at the game
 * phases alone gives a curve of game phases alone rather than a sawtooth
 * alternating between two different kinds of moment.
 */
const StatDetail = {
  currentKey: null,

  init() {
    this.el = document.getElementById('statDetail');
    this.titleEl = document.getElementById('statDetailTitle');
    this.textEl = document.getElementById('statDetailText');
    this.footEl = document.getElementById('statDetailFoot');
    this.canvas = document.getElementById('statDetailChart');

    document.getElementById('statDetailClose').addEventListener('click', () => this.close());
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && !this.el.classList.contains('hidden')) this.close();
    });
    window.addEventListener('resize', () => {
      if (!this.el.classList.contains('hidden')) this.redraw();
    });
  },

  close() {
    this.el.classList.add('hidden');
    this.currentKey = null;
  },

  async open(key, label) {
    if (!Viewer.runId) return;

    this.currentKey = key;
    this.titleEl.textContent = label || key;
    this.textEl.textContent = RunStats.explain(key) || 'No description for this value.';
    this.el.classList.remove('hidden');

    // Whatever is known of the run draws at once, and the load adds only what
    // this statistic is missing. It is started first so that an empty chart
    // can say it is loading rather than that there is nothing to show.
    const loading = Viewer.loadHistory([key]);
    this.redraw();
    await loading;
  },

  /**
   * Pick up the open statistic's history if leaving the Viewer or hiding the
   * tab cut it short. A trajectory cut short needs nothing here: its Load
   * history button comes back.
   */
  resume() {
    if (this.currentKey) Viewer.loadHistory([this.currentKey]);
  },

  /** Redraw, if the popup is open. */
  refresh() {
    if (!this.el.classList.contains('hidden')) this.redraw();
  },

  /** Say under the chart that its history could not be read. */
  failed(err) {
    this.footEl.textContent = `Could not load history: ${err.message}`;
  },

  /**
   * Points for the current statistic under the active phase filter.
   *
   * Node-count statistics are converted to a share of the population that
   * entered the phase, because "40 births" means something quite different in
   * a world of 100 agents than in one of 4,000.
   */
  points() {
    const key = this.currentKey;
    const payload = SeriesLoad.cache.get(Viewer.runId);
    if (!payload || !key) return { xs: [], ys: [], asShare: false };

    const s = payload.series;
    const values = s[key] || [];
    const phases = s.phase || [];
    const iterations = s.iteration || [];
    const before = s.nodes_before || [];
    const nodes = s.nodes || [];

    const asShare = RunStats.POPULATION_COUNTS.has(key);
    const xs = [], ys = [];

    for (let i = 0; i < values.length; i++) {
      if (!Viewer.framePassesFilter(phases[i])) continue;
      const v = values[i];
      if (v === null || v === undefined) continue;

      if (asShare) {
        // Older runs predate nodes_before; the post-phase count is the closest
        // honest stand-in, so the curve stays usable rather than empty.
        const denominator = before[i] || nodes[i] || 0;
        if (!denominator) continue;
        ys.push((v / denominator) * 100);
      } else {
        ys.push(v);
      }
      xs.push(iterations[i]);
    }
    return { xs, ys, asShare };
  },

  redraw() {
    const { xs, ys, asShare } = this.points();

    if (!ys.length) {
      const { ctx, h } = _prepareCanvas(this.canvas);
      ctx.fillStyle = Ink.of('dim');
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText(Jobs.busy(Viewer)
        ? 'Summarising the run\u2026'
        : 'No data for this statistic under the current phase filter.', 10, h / 2);
      this.footEl.textContent = '';
      return;
    }

    // Drawn on the axes every other chart uses. This popup drew its own grid,
    // labels and axes beside them, and so looked like none of the others.
    const pad = { left: 62, right: 12, top: 10, bottom: 30 };
    const { ctx, w, h } = _prepareCanvas(this.canvas, pad);

    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }

    // Widen to the round numbers the axis labels, so the top and bottom lines
    // are labelled values rather than wherever the data happened to stop.
    const yTicks = _axisTicks(lo, hi, _tickCounts(w, h).down);
    if (yTicks.length > 1) {
      lo = Math.min(lo, yTicks[0]);
      hi = Math.max(hi, yTicks[yTicks.length - 1]);
    }
    const xLo = xs[0], xHi = xs[xs.length - 1];

    // Where the frame on screen sits, so the number in the strip has a home.
    const current = Viewer.frame ? Viewer.frame.iteration : null;
    _axes(ctx, w, h, {
      x: { lo: xLo, hi: xHi, format: v => Math.round(v).toLocaleString('en-US') },
      y: {
        lo, hi,
        format: v => asShare
          ? `${+v.toFixed(2)}%`
          : (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US')
                                 : String(+v.toFixed(Math.abs(v) < 1 ? 3 : 2)))
      },
      guides: current !== null && current >= xLo && current <= xHi
        ? [{ axis: 'x', at: current, colour: Ink.of('accent') }] : [],
      pad
    });

    // The curve
    const xAt = v => (xHi === xLo ? w / 2 : ((v - xLo) / (xHi - xLo)) * w);
    const yAt = v => h - ((v - lo) / (hi - lo)) * h;
    ctx.strokeStyle = Ink.of('accent');
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < ys.length; i++) {
      const x = xAt(xs[i]), y = yAt(ys[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = Ink.of('dim');
    ctx.font = '9px system-ui, sans-serif';
    const axisLabel = 'iteration';
    ctx.fillText(axisLabel, w - ctx.measureText(axisLabel).width, h + pad.bottom - 1);

    const payload = SeriesLoad.cache.get(Viewer.runId);
    const sampled = payload && payload.sampled;
    this.footEl.textContent =
      `${formatNumber(ys.length)} point${ys.length === 1 ? '' : 's'} · ${Viewer.phaseFilterLabel()}` +
      (asShare ? ' · shown as a share of the nodes that entered the phase' : '') +
      (sampled ? ` · sampled every ${formatNumber(payload.stride)} iterations` : '');
  }
};
