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
    const { ctx, w, h } = _prepareCanvas(this.canvas);

    if (!ys.length) {
      ctx.fillStyle = Ink.of('dim');
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText(Jobs.busy(Viewer)
        ? 'Summarising the run\u2026'
        : 'No data for this statistic under the current phase filter.', 10, h / 2);
      this.footEl.textContent = '';
      return;
    }

    const padL = 62, padR = 12, padT = 10, padB = 30;
    const plotW = w - padL - padR, plotH = h - padT - padB;

    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }

    // Widen to the round numbers, so the top and bottom lines are labelled
    // values rather than wherever the data happened to stop.
    const yTicks = _axisTicks(lo, hi, 5);
    if (yTicks.length > 1) {
      lo = Math.min(lo, yTicks[0]);
      hi = Math.max(hi, yTicks[yTicks.length - 1]);
    }

    const xLo = xs[0], xHi = xs[xs.length - 1];
    const xTicks = _axisTicks(xLo, xHi, 5);

    const xAt = v => padL + (xHi === xLo ? plotW / 2 : ((v - xLo) / (xHi - xLo)) * plotW);
    const yAt = v => padT + plotH - ((v - lo) / (hi - lo)) * plotH;

    const fmtY = v => asShare
      ? `${+v.toFixed(2)}%`
      : (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US')
                             : String(+v.toFixed(Math.abs(v) < 1 ? 3 : 2)));

    ctx.font = '10px system-ui, sans-serif';
    ctx.lineWidth = 1;

    // Horizontal grid
    for (const v of yTicks) {
      const y = Math.round(yAt(v)) + 0.5;
      if (y < padT - 1 || y > padT + plotH + 1) continue;
      ctx.strokeStyle = Ink.of('grid');
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();

      ctx.fillStyle = Ink.of('label');
      const label = fmtY(v);
      ctx.fillText(label, padL - 6 - ctx.measureText(label).width, y + 3);
    }

    // Vertical grid
    for (const v of xTicks) {
      const x = Math.round(xAt(v)) + 0.5;
      if (x < padL - 1 || x > padL + plotW + 1) continue;
      ctx.strokeStyle = Ink.of('grid');
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();

      ctx.fillStyle = Ink.of('label');
      const label = Math.round(v).toLocaleString('en-US');
      ctx.fillText(label, x - ctx.measureText(label).width / 2, h - 12);
    }

    // Axes, a shade brighter than the grid
    ctx.strokeStyle = Ink.of('axis');
    ctx.beginPath();
    ctx.moveTo(padL + 0.5, padT);
    ctx.lineTo(padL + 0.5, padT + plotH + 0.5);
    ctx.lineTo(padL + plotW, padT + plotH + 0.5);
    ctx.stroke();

    // Where the frame on screen sits, so the number in the strip has a home
    const currentIteration = Viewer.frame ? Viewer.frame.iteration : null;
    if (currentIteration !== null && currentIteration >= xLo && currentIteration <= xHi) {
      ctx.strokeStyle = Ink.of('accent');
      ctx.globalAlpha = 0.45;
      ctx.beginPath();
      ctx.moveTo(xAt(currentIteration), padT);
      ctx.lineTo(xAt(currentIteration), padT + plotH);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // The curve
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
    ctx.fillText(axisLabel, padL + plotW - ctx.measureText(axisLabel).width, h - 1);

    const payload = SeriesLoad.cache.get(Viewer.runId);
    const sampled = payload && payload.sampled;
    this.footEl.textContent =
      `${formatNumber(ys.length)} point${ys.length === 1 ? '' : 's'} · ${Viewer.phaseFilterLabel()}` +
      (asShare ? ' · shown as a share of the nodes that entered the phase' : '') +
      (sampled ? ` · sampled every ${formatNumber(payload.stride)} iterations` : '');
  }
};
