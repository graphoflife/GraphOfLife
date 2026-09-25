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
 * growing sideways. Nothing is computed twice, because each request only pays
 * for the samples the previous one did not take.
 *
 * It is fetched at the depth the chart on screen needs. Five sixths of what a
 * frame costs goes on the statistics that walk the graph, and most charts plot
 * none of them, so the caller says whether it does (Metrics.needsHeavy). Both
 * depths used to be fetched every time, cheap first: a chart of bridge counts
 * then sat empty through the whole cheap pass — half a minute on a long run —
 * while its bar read 100%, because the cheap pass had filled every sample and
 * the bar counted samples.
 *
 * **What a caller is handed is accumulated, never replaced.** A caller that
 * already holds part of a run passes it as `from` and the climb adds to it,
 * merging rows by frame so that an expensive value is never blanked by a cheap
 * row for the same frame. Starting from nothing made a chart that had reached
 * three hundred points drop back to two the moment another load began: it
 * looked like the load had restarted, and the finished picture was replaced by
 * a sketch of itself.
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
   * immediately even on a run that will take minutes to finish.
   */
  STEPS: [2, 3, 5, 9, 17, 33, 65, 129, 257, null],

  //: How often a step in flight asks the server how far it has got.
  POLL_MS: 700,

  //: runId -> everything known of that run's history. One copy for the whole
  //: page: the Viewer and Diagrams each used to keep their own, so a history
  //: summarised for one was summarised again for the other, and the Diagrams
  //: copy was never forgotten when its run grew.
  cache: new Map(),

  /**
   * Whether the history in hand already answers a chart of these statistics:
   * every sample, and the expensive values too if it plots any of them.
   */
  ready(runId, keys) {
    const have = this.cache.get(runId);
    return Boolean(have && have.complete && (have.heavy || !Metrics.needsHeavy(keys)));
  },

  /** Drop a run's history, because the run has grown since. */
  forget(runId) {
    this.cache.delete(runId);
  },

  /**
   * Fetch as much of the run as charts of `keys` need, calling
   * `onStep(payload)` with everything known so far.
   *
   * Picks up from whatever the cache already holds, and keeps it up to date at
   * every step. `job` carries the abort signal and the progress report.
   *
   * The bar counts samples and only ever moves forward. While a step is in
   * flight the server says how many of that step's frames it has summarised
   * (/series/progress, which nothing asked for until now), and the bar moves
   * through the step in proportion. Frames and samples are different units —
   * a sample is two frames, one per phase — and reporting both straight
   * through made the bar jump back and forth between them.
   */
  async climb(runId, keys, { onStep, job, text = 'Summarising the run' } = {}) {
    const heavy = Metrics.needsHeavy(keys);
    const from = this.cache.get(runId) || null;
    const opts = job ? { signal: job.signal } : undefined;
    const cancelled = () => Boolean(job && job.cancelled);

    const gathered = new Map();      // "iteration:phase" -> row
    const columns = new Set();       // every key any reply carried, in order
    let shape = from;                // the latest reply's metadata
    let covered = false;             // the latest reply held every sample
    let deep = false;                // ...and carried the expensive values

    const result = () => this._asPayload(gathered, columns, shape, {
      complete: Boolean(from && from.complete) || covered,
      heavy: Boolean(from && from.heavy) || (covered && deep)
    });

    if (from) this._absorb(gathered, columns, from);

    let shown = 0;
    const total = () => (shape && shape.totalPoints) || 0;
    const advance = (n) => {
      const of = total();
      const at = Math.min(of, Math.floor(n));
      if (!job || !of || at <= shown) return;
      shown = at;
      job.report(at, of, text);
    };
    if (job && total()) job.report(0, total(), text);

    // One step, with the server's own count moving the bar while it runs.
    const step = async (points) => {
      const before = shown;
      const target = points === null ? total() : Math.min(points, total() || points);
      const poll = job && setInterval(async () => {
        try {
          const p = await API.getSeriesProgress(runId, opts);
          if (p && p.building && p.total) advance(before + (target - before) * (p.done / p.total));
        } catch (err) { /* the bar simply does not move */ }
      }, this.POLL_MS);
      try {
        return await API.getSeries(runId, points, heavy, opts);
      } finally {
        if (poll) clearInterval(poll);
      }
    };

    for (const points of this.STEPS) {
      if (cancelled()) break;
      const reply = await step(points);
      if (cancelled()) break;
      shape = reply;
      covered = Boolean(reply.complete);
      deep = Boolean(reply.heavy);
      this._absorb(gathered, columns, reply);
      advance(reply.points || total());
      const payload = result();
      this.cache.set(runId, payload);
      onStep?.(payload);
      if (covered) break;
    }
    return result();
  },

  /** Merge a reply's rows into what is gathered, by frame. */
  _absorb(gathered, keys, payload) {
    const columns = payload.series || {};
    for (const key of payload.keys || []) keys.add(key);
    const length = (columns.iteration || []).length;
    for (let i = 0; i < length; i++) {
      const row = {};
      for (const key of payload.keys || []) row[key] = columns[key][i];
      // Keyed by iteration and phase rather than by position: two loads sample
      // the same frames, and a row must land on the one it describes.
      const at = `${row.iteration}:${row.phase}`;
      const before = gathered.get(at);
      // A cheap row must never overwrite an expensive one for the same frame;
      // it would blank the structural values an earlier load already found.
      gathered.set(at, before && payload.heavy === false
        ? { ...row, ...this._known(before) }
        : { ...before, ...row });
    }
  },

  /** The values already known, so an incoming cheap row cannot blank them. */
  _known(row) {
    const out = {};
    for (const key of Object.keys(row)) {
      if (row[key] !== null && row[key] !== undefined) out[key] = row[key];
    }
    return out;
  },

  /**
   * Everything gathered so far, back in the shape a reply has.
   *
   * The keys are the union of every reply's rather than the latest one's. A
   * coarse request leaves off what only a whole run can give — the family
   * count needs an unbroken chain of ancestry — so taking the latest reply's
   * keys dropped that statistic from the chart the moment a second load began.
   */
  _asPayload(gathered, keys, shape, { complete, heavy }) {
    const rows = [...gathered.values()].sort(
      (a, b) => (a.iteration - b.iteration) || (a.phase - b.phase));
    const series = {};
    for (const key of keys) series[key] = rows.map(row => row[key] ?? null);
    return {
      ...(shape || {}),
      count: rows.length,
      keys: [...keys],
      series,
      points: new Set(rows.map(row => row.iteration)).size,
      heavy,
      complete
    };
  }
};
