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
 * `SimConfig.frames_before` is built on and `RunStore.getIterations`
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
   * stop early.
   *
   * `fields` is the columns the caller actually reads. A frame of a large run
   * is mostly its edge list, and Flow modules never looks at it. Asking for
   * the whole thing was tens of megabytes to parse per window.
   *
   * `sightings`, when given, caps how many agents are read over the whole
   * window: one measured in iterations is very different work in a world of
   * sixty and in one of forty thousand. Each batch is asked for what is left
   * of it, and the read ends once it is spent. Every batch used to be handed
   * all of it, so each caller kept a running total of its own to know when to
   * stop, and read up to nearly twice the cap before it did.
   */
  async *read(runId, indices, fields = null, sightings = 0, opts = {}) {
    let seen = 0;
    for (let at = 0; at < indices.length; at += this.BATCH) {
      // Checked between batches as well as passed down, so a long window stops
      // at the next boundary even if the request in flight has already been
      // answered. Thrown rather than returned: a window cut short must not
      // look to its reader like a window that ended.
      opts.signal?.throwIfAborted();
      const slice = indices.slice(at, at + this.BATCH);
      const reply = await API.getFrames(runId, slice[0], slice.length, fields,
                                        sightings ? sightings - seen : 0, opts);
      const got = reply.frames || [];
      yield got;
      // Counted as the backends count it.
      seen += got.reduce((n, f) => n + (f.ids || f.brain_ids || []).length, 0);
      // Short of what was asked for means the run ran out, or the backend
      // stopped on what was left of the budget: nothing more is coming.
      if (got.length < slice.length || (sightings && seen >= sightings)) return;
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
  },

  /**
   * The window bar and the canvas of a view over a window of frames, wired to
   * `view`, whose elements are named from `prefix`: the scrubber and the span
   * that move and size the window (both read it again), the phase buttons
   * (then `onPhase()`), the shortest-lifetime slider (a redraw), hovering,
   * and resizing. Lineage and Flow modules each wired all of it, the same
   * way, element by element. False when the page has no such view.
   */
  wire(view, prefix, onPhase) {
    const el = name => document.getElementById(prefix + name);
    view.canvas = el('Canvas');
    if (!view.canvas) return false;
    view.ctx = view.canvas.getContext('2d');
    view.noteEl = el('Note');
    view.readoutEl = el('Readout');
    view.minLifeEl = el('MinLife');
    view.spanEl = el('Span');

    // Moving the window reads it again, so it acts on release rather than on
    // every pixel of the drag, and a new span on commit rather than on every
    // keystroke.
    const scrub = el('Window');
    scrub.addEventListener('change', () => view.load(view.runId, Number(scrub.value)));
    view.spanEl.value = String(view.MAX_ITERATIONS);
    view.spanEl.addEventListener('change', () => view.load(view.runId, view.windowStart));

    view.minLifeEl.addEventListener('input', () => {
      el('MinLifeValue').textContent = view.minLifeEl.value;
      view.draw();
    });

    const buttons = document.querySelectorAll(`#${prefix}Phase .seg-btn`);
    for (const button of buttons) {
      button.addEventListener('click', () => {
        view.phase = button.dataset.phase;
        for (const other of buttons) other.classList.toggle('active', other === button);
        onPhase();
      });
    }

    view.canvas.addEventListener('mousemove', e => view.hover(e));
    view.canvas.addEventListener('mouseleave', () => {
      view.hovered = null;
      view.readoutEl.textContent = '';
      view.draw();
    });
    if (window.ResizeObserver) {
      new ResizeObserver(() => { view.resize(); view.draw(); }).observe(view.canvas.parentElement);
    }
    return true;
  }
};

/**
 * What a view over a window of frames is besides its picture, for Lineage and
 * Flow modules to take on, each of which had its own identical copy.
 */
const WindowView = {
  /** The tab owns the list; a view only needs to know what is in it. */
  setRuns(runs) {
    this.runs = runs;
  },

  say(text) {
    this.noteEl.textContent = text;
  },

  /** Size the canvas to its box, at the screen's pixel density. */
  resize() {
    const box = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (!box.width || !box.height) return;
    this.canvas.width = Math.round(box.width * dpr);
    this.canvas.height = Math.round(box.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = box.width;
    this.h = box.height;
  },

  /** Why there is nothing to draw, in the middle of the canvas. */
  drawNothing(loaded) {
    const ctx = this.ctx;
    ctx.fillStyle = Ink.of('dim');
    ctx.font = '12px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(loaded ? 'Nothing lasted that long.' : 'No run loaded.', this.w / 2, this.h / 2);
    ctx.textAlign = 'left';
  }
};
