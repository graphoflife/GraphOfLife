/*
 * A contiguous window of one run's frames, and the control that moves it.
 *
 * Both Research views read a run the same way and for the same reason. A run
 * is thousands of frames, the picture is a few hundred pixels wide, and
 * neither of the things being drawn survives sampling: ancestry is a chain,
 * and a module's identity is its overlap with the frame before it. Reading
 * every Nth frame does not thin either picture, it destroys it — a lineage
 * read at stride 32 came back 97% rootless, because the parents had lived and
 * died between two samples. So the window is contiguous, and you move it
 * rather than widening it.
 *
 * It is measured in *recorded iterations*, which is the unit a reader counts
 * in. One is two frames, one per phase — the same relation
 * `gol_store.frames_recorded_before` is built on and `getFramesStrided`
 * already assumes. A run recorded with `export_every` above 1 has its recorded
 * iterations further apart than one apiece, which is why `describe` reports
 * the iteration numbers written in the frames rather than arithmetic on the
 * frame index: that stays true whatever the run was recorded at.
 */
const FrameWindow = {
  // A recorded iteration is a reproduction frame and a game frame.
  PHASES: 2,
  // Frames per request. One request per frame meant two hundred round trips
  // for a window, each carrying the whole topology of the world so that two
  // columns could be read out of it; the backends take a range and a field
  // list, so this is now the size of a batch rather than a concurrency limit.
  BATCH: 32,

  /**
   * Which frames to read, given where the reader wants the window to start.
   *
   * `from` is a frame index, since that is what the slider holds. It is
   * clamped so the window never hangs off the end of the run.
   */
  plan(total, from, maxIterations) {
    const span = Math.min(total, Math.max(1, maxIterations) * this.PHASES);
    const start = Math.max(0, Math.min(Math.round(from) || 0, total - span));
    const indices = [];
    for (let i = start; i < start + span; i++) indices.push(i);
    return { start, indices, total };
  },

  /**
   * Point a range input at a plan, and hide it when there is nothing to move.
   *
   * Stepping by a whole iteration rather than by one frame keeps the window
   * from opening on a game frame, which would pair each iteration's second
   * half with the next one's first and leave every module matched across a
   * boundary that is not there.
   */
  bindScrubber(el, plan) {
    if (!el) return;
    el.min = '0';
    el.max = String(Math.max(0, plan.total - plan.indices.length));
    el.step = String(this.PHASES);
    el.value = String(plan.start);
    el.closest('label').hidden = plan.indices.length >= plan.total;

    // Say which stretch is on screen. A slider with no numbers on it tells you
    // that there is more run than window and nothing else — not where you are
    // in it, and not how much of it you are looking at.
    const readout = el.parentElement.querySelector('span');
    if (readout) {
      const first = Math.floor(plan.start / this.PHASES);
      const last = first + Math.ceil(plan.indices.length / this.PHASES) - 1;
      readout.textContent =
        `iterations ${formatNumber(first)}–${formatNumber(last)} `
        + `of ${formatNumber(Math.ceil(plan.total / this.PHASES))}`;
    }
  },

  /**
   * The planned frames, a batch at a time.
   *
   * Yielded rather than returned whole so a caller can draw what it has and
   * stop early — both views draw every batch, and the lineage also shortens
   * the window once the first frame has said how big this world is.
   *
   * `fields` is the columns the caller actually reads. A frame of a large run
   * is mostly its edge list, and neither view looks at it: the lineage wants
   * two arrays of brain ids, the flow view wants ids and allocations. Asking
   * for the whole thing was tens of megabytes to parse per window.
   */
  async *read(runId, indices, fields = null) {
    for (let at = 0; at < indices.length; at += this.BATCH) {
      const slice = indices.slice(at, at + this.BATCH);
      const reply = await API.getFrames(runId, slice[0], slice.length, fields);
      yield reply.frames || [];
    }
  },

  /**
   * Where the window sits, in the run's own iteration numbers, or '' when it
   * covers the whole run and there is nothing to say.
   */
  describe(frames, total) {
    if (!frames.length || frames.length >= total) return '';
    return `iterations ${formatNumber(frames[0].iteration)}–`
         + `${formatNumber(frames[frames.length - 1].iteration)} of the `
         + `${formatNumber(Math.ceil(total / this.PHASES))} recorded`;
  }
};
