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
 * It is fetched twice over: once for the cheap statistics, once for the ones
 * that walk the graph. Five sixths of what a frame costs goes on the latter and
 * most charts plot none of them, so the whole run arrives at full resolution in
 * a sixth of the time and the structural quantities fill in behind it.
 *
 * **What a caller is handed is accumulated, never replaced.** The two passes
 * each climb from two points, so handing the reply straight through made a
 * chart that had reached three hundred points drop back to two the moment the
 * second pass began — it looked like the load had restarted, and the finished
 * picture was replaced by a sketch of itself. Rows are merged by frame instead,
 * heavy values overwriting light ones for the same frame, so the chart only
 * ever gains.
 *
 * The server caps a run at 300 samples however long it ran, which is more
 * points than these charts have pixels for.
 */
const SeriesLoad = {

  /**
   * Roughly doubling, which is what bisection adds at each depth.
   *
   * The first step is two points, so a caller has axes and a range almost
   * immediately even on a run that will take minutes to finish.
   */
  STEPS: [2, 3, 5, 9, 17, 33, 65, 129, 257],

  /**
   * Fetch, calling `onStep(payload)` with everything gathered so far.
   *
   * `cancelled()` is polled before and after every request, so switching run,
   * switching view or leaving the tab abandons the climb rather than carrying
   * on computing into a view nobody is looking at.
   */
  async climb(runId, { onStep, cancelled = () => false } = {}) {
    const gathered = new Map();     // frame index -> row
    let shape = null;               // the last reply, for its metadata
    let heavyDone = false;

    const absorb = (payload) => {
      shape = payload;
      const columns = payload.series || {};
      const keys = payload.keys || [];
      const length = (columns.iteration || []).length;
      for (let i = 0; i < length; i++) {
        const row = {};
        for (const key of keys) row[key] = columns[key][i];
        // Keyed by iteration and phase rather than by position: the two passes
        // sample the same frames, and a row must land on the one it describes.
        const at = `${row.iteration}:${row.phase}`;
        const before = gathered.get(at);
        // A light row must never overwrite a heavy one for the same frame; it
        // would blank the structural values the heavy pass already found.
        gathered.set(at, before && payload.heavy === false
          ? { ...row, ...this._known(before) }
          : { ...before, ...row });
      }
      if (payload.heavy) heavyDone = payload.complete;
      return this._asPayload(gathered, shape, heavyDone);
    };

    for (const heavy of [false, true]) {
      for (const points of this.STEPS) {
        if (cancelled()) return this._asPayload(gathered, shape, heavyDone);
        const payload = await API.getSeries(runId, points, heavy);
        if (cancelled()) return this._asPayload(gathered, shape, heavyDone);
        onStep?.(absorb(payload));
        if (payload.complete) break;
        if (payload.totalPoints && points >= payload.totalPoints) break;
      }
      if (cancelled()) return this._asPayload(gathered, shape, heavyDone);
      const whole = await API.getSeries(runId, null, heavy);
      if (cancelled()) return this._asPayload(gathered, shape, heavyDone);
      onStep?.(absorb(whole));
    }
    return this._asPayload(gathered, shape, heavyDone);
  },

  /** The values already known, so an incoming light row cannot blank them. */
  _known(row) {
    const out = {};
    for (const key of Object.keys(row)) {
      if (row[key] !== null && row[key] !== undefined) out[key] = row[key];
    }
    return out;
  },

  /** Everything gathered so far, back in the shape a reply has. */
  _asPayload(gathered, shape, heavyDone) {
    const rows = [...gathered.values()].sort(
      (a, b) => (a.iteration - b.iteration) || (a.phase - b.phase));
    const keys = shape ? shape.keys : [];
    const series = {};
    for (const key of keys) series[key] = rows.map(row => row[key]);
    return {
      ...(shape || {}),
      count: rows.length,
      keys,
      series,
      points: rows.length,
      heavy: heavyDone,
      complete: Boolean(shape && shape.complete)
    };
  },

  /**
   * How far along a payload is, as a fraction, for a progress readout.
   *
   * How many of the run's samples are in hand, which reaches one while the
   * heavy pass is still filling those samples in behind it. `payload.heavy`
   * is what says whether that second pass has finished.
   */
  fraction(payload) {
    if (!payload || !payload.totalPoints) return null;
    return Math.min(1, (payload.points || 0) / payload.totalPoints);
  }
};
