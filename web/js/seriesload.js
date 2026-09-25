/*
 * Load a run's statistics at increasing resolution, drawing all the way.
 *
 * Summarising one frame of a large run costs well over a second — loops,
 * triangles, bridges, distance sweeps, two dimension estimates. A few hundred
 * of them is minutes, and a caller that waits for the whole thing shows an
 * empty box for all of it, which reads as broken rather than as busy.
 *
 * So the run is fetched in bisection order: both ends, then the middle, then
 * the middles of the halves. Every step is a chart of the *whole* run, and
 * each one is finer than the last — the picture gains resolution rather than
 * growing sideways.
 *
 * The request names the statistics the chart plots, and the backend decides
 * from them how deep to go: five sixths of what a frame costs goes on the
 * statistics that walk the graph, and most charts plot none of them.
 *
 * Each reply is everything the backend knows of the run. gol_series.History
 * keeps that — in series.json on a server, in the worker's memory in a
 * browser — so this keeps the latest reply and nothing else. It used to merge
 * replies itself, because each held only the samples asked for, and had to
 * guard against a chart falling back to two points when a second load began
 * and a cheap row blanking a bridge count. Those are the history's business
 * now, and cannot happen here.
 *
 * The server caps a run at 300 samples however long it ran, which is more
 * points than these charts have pixels for.
 */
const SeriesLoad = {

  /**
   * Roughly doubling, which is what bisection adds at each depth. The last
   * step is the whole grid.
   *
   * The first step is two points, so a caller has axes and a range almost
   * immediately even on a run that will take minutes to finish — and on a run
   * already summarised, the first step is also the last.
   */
  STEPS: [2, 3, 5, 9, 17, 33, 65, 129, 257, null],

  //: How often a step in flight asks the server how far it has got.
  POLL_MS: 700,

  //: runId -> the latest reply: everything known of that run's history. One
  //: for the whole page, so a history summarised for the Viewer is already
  //: there for Diagrams, and the other way round.
  cache: new Map(),

  /** The stored statistics charts of `keys` read: a derived one by what it is made of. */
  columns(keys) {
    return [...new Set(keys.flatMap(key =>
      (Metrics.DERIVED[key] ? Metrics.DERIVED[key].needs : [key])))];
  },

  /**
   * Whether the history in hand answers a chart of `keys` on a run of
   * `frames` frames: every sample, with the graph statistics if the chart
   * plots any, and of the run as big as it is now. A history finished at one
   * size used to count as finished until something called forget(), and only
   * the Viewer ever did, so Diagrams kept drawing a growing run as it once was.
   */
  ready(runId, keys, frames) {
    const have = this.cache.get(runId);
    if (!have || !(have.frames >= frames)) return false;
    const deep = this.columns(keys).some(key => have.heavyKeys.includes(key));
    return deep ? have.heavy : have.complete;
  },

  /**
   * Hear a run's size from whoever has just read it.
   *
   * A history longer than the run describes frames that are gone: the run was
   * resumed from an earlier checkpoint. It is dropped, which is the one case
   * ready() cannot tell from a size alone. One that is shorter is only behind,
   * and ready() already says so.
   */
  noteSize(runId, frames) {
    const have = this.cache.get(runId);
    if (have && have.frames > frames) this.cache.delete(runId);
  },

  /**
   * Fetch as much of the run as charts of `keys` need, calling
   * `onStep(reply)` as each step arrives.
   *
   * `job` carries the abort signal and the progress report. The bar counts
   * samples finished at the depth the chart needs, and only ever moves
   * forward. While a step is in flight the server says how many of that
   * step's frames it has summarised (/series/progress), and the bar moves
   * through the step in proportion. Frames and samples are different units —
   * a sample is two frames, one per phase — and reporting both straight
   * through made the bar jump back and forth between them.
   */
  async climb(runId, keys, { onStep, job, text = 'Summarising the run' } = {}) {
    const columns = this.columns(keys);
    const opts = job ? { signal: job.signal } : undefined;
    const cancelled = () => Boolean(job && job.cancelled);

    let shown = 0;
    let total = (this.cache.get(runId) || {}).totalPoints || 0;
    const advance = (n) => {
      const at = Math.min(total, Math.floor(n));
      if (!job || !total || at <= shown) return;
      shown = at;
      job.report(at, total, text);
    };
    if (job && total) job.report(0, total, text);

    // One step, with the server's own count moving the bar while it runs.
    const step = async (points) => {
      const before = shown;
      const target = points === null ? total : Math.max(before, Math.min(points, total || points));
      const poll = job && setInterval(async () => {
        try {
          const p = await API.getSeriesProgress(runId, opts);
          if (p && p.building && p.total) advance(before + (target - before) * (p.done / p.total));
        } catch (err) { /* the bar simply does not move */ }
      }, this.POLL_MS);
      try {
        return await API.getSeries(runId, points, columns, opts);
      } finally {
        if (poll) clearInterval(poll);
      }
    };

    for (const points of this.STEPS) {
      if (cancelled()) break;
      const reply = await step(points);
      if (cancelled()) break;
      this.cache.set(runId, reply);
      total = reply.totalPoints || 0;
      advance(reply.done);
      onStep?.(reply);
      if (reply.done >= reply.totalPoints) break;
    }
    return this.cache.get(runId);
  }
};
