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

    // Drawn as the Diagrams tab draws a time series (drawTimeline), so the two
    // look alike: this popup once drew its own grid, labels and axes, and so
    // looked like none of the other charts.
    const current = Viewer.frame ? Viewer.frame.iteration : null;
    const track = ys.length ? {
      points: xs.map((t, i) => ({ t, v: ys[i] })), mapped: ys,
      lo: Math.min(...ys), hi: Math.max(...ys), colour: Ink.of('accent')
    } : null;
    drawTimeline(this.canvas, track ? [track] : [], {
      chrome: { xLabel: 'iteration', ticks: true, grid: true, legend: [] },
      // Where the frame on screen sits, so the number in the strip has a home.
      guides: track && current !== null && current >= xs[0] && current <= xs[xs.length - 1]
        ? [{ axis: 'x', at: current, colour: Ink.of('accent') }] : [],
      yFormat: v => asShare
        ? `${+v.toFixed(2)}%`
        : (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US')
                               : String(+v.toFixed(Math.abs(v) < 1 ? 3 : 2))),
      message: track ? null
        : Jobs.busy(Viewer) ? 'Summarising the run\u2026'
        : 'No data for this statistic under the current phase filter.'
    });
    if (!track) {
      this.footEl.textContent = '';
      return;
    }

    const payload = SeriesLoad.cache.get(Viewer.runId);
    const sampled = payload && payload.sampled;
    this.footEl.textContent =
      `${formatNumber(ys.length)} point${ys.length === 1 ? '' : 's'} · ${Viewer.phaseFilterLabel()}` +
      (asShare ? ' · shown as a share of the nodes that entered the phase' : '') +
      (sampled ? ` · sampled every ${formatNumber(payload.stride)} iterations` : '');
  }
};
