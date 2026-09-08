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
   * Fetch, calling `onStep(payload)` after each pass.
   *
   * Returns the finest payload. `cancelled()` is polled between passes so
   * switching run or leaving the tab abandons the climb rather than carrying
   * on drawing into a view nobody is looking at.
   */
  async climb(runId, { onStep, cancelled = () => false } = {}) {
    let last = null;

    for (const points of this.STEPS) {
      if (cancelled()) return last;
      const payload = await API.getSeries(runId, points);
      if (cancelled()) return last;
      last = payload;
      onStep?.(payload, false);
      // Nothing coarser was left out, so there is nothing finer to ask for.
      if (payload.complete) return payload;
      // A run with fewer samples than this step wanted is already whole.
      if (payload.totalPoints && points >= payload.totalPoints) break;
    }

    if (cancelled()) return last;
    const whole = await API.getSeries(runId);
    if (cancelled()) return last;
    onStep?.(whole, true);
    return whole;
  },

  /** How far along a payload is, as a fraction, for a progress readout. */
  fraction(payload) {
    if (!payload || !payload.totalPoints) return null;
    return Math.min(1, (payload.points || 0) / payload.totalPoints);
  }
};
