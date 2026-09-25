/*
 * One loading job at a time, owned by whatever is on screen.
 *
 * Every view used to bring its own idea of "stop that" — Research kept an
 * epoch, Lineage a token, Flow modules compared a run id, the stat detail a
 * second token, the Viewer a boolean — and none of them could stop anything.
 * They stopped *asking for more*; the request already in flight ran to the end
 * and painted whatever it found. Lineage is a single long request, so it could
 * not be interrupted at all: leaving the view and coming back to another one
 * still meant waiting for it, which is what "sometimes nothing is working"
 * was.
 *
 * So there is one mechanism, and it hangs on two words:
 *
 *   owner   what the work is for — a view, a panel. Starting a job for an
 *           owner cancels that owner's previous job, because asking a second
 *           question of the same thing means you no longer want the first
 *           answer.
 *   signal  an AbortSignal that reaches the actual fetch. Cancelling ends the
 *           request rather than merely ignoring it.
 *
 * `only(owner)` cancels everything else, which is how switching view stops the
 * view you left, and how hiding the tab stops all of it.
 */
const Jobs = {
  //: owner -> the job currently running for it.
  _live: new Map(),

  //: Owners whose most recent job ran to the end. Cancelling on the way out
  //: is only half of being in control — the other half is picking the work up
  //: again on the way back in, and a view cannot do that unless something
  //: remembers that its last load was cut short rather than finished.
  _done: new Set(),

  //: Set by Jobs.attach once the page exists; until then reporting is a no-op
  //: so a job started during boot does not have to care.
  _bar: null,

  /**
   * Start work for an owner, cancelling whatever that owner was doing.
   *
   * `fn` is handed a job with `signal` for the request layer and `report` for
   * the progress bar. Returning normally finishes the job; throwing an abort
   * is not an error and is swallowed, because the caller asked for it.
   */
  run(owner, label, fn) {
    this.cancel(owner);
    this._done.delete(owner);

    const controller = new AbortController();
    const job = {
      owner,
      label,
      signal: controller.signal,
      get cancelled() { return controller.signal.aborted; },
      report: (done, total, text) => {
        if (controller.signal.aborted) return;
        this._show(owner, text || label, done, total);
      }
    };
    this._live.set(owner, { job, controller });
    this._show(owner, label, 0, 0);

    return (async () => {
      try {
        const result = await fn(job);
        if (!controller.signal.aborted) this._done.add(owner);
        return result;
      } catch (err) {
        // An abort is the caller getting what it asked for, not a failure.
        if (err && (err.name === 'AbortError' || controller.signal.aborted)) return null;
        throw err;
      } finally {
        if (this._live.get(owner) && this._live.get(owner).job === job) {
          this._live.delete(owner);
          this._clear(owner);
        }
      }
    })();
  },

  /** Stop this owner's work, if it has any. */
  cancel(owner) {
    const live = this._live.get(owner);
    if (!live) return;
    this._live.delete(owner);
    live.controller.abort();
    this._clear(owner);
  },

  /** Stop everything that is not this owner. The whole point of the file. */
  only(owner) {
    for (const other of [...this._live.keys()]) {
      if (other !== owner) this.cancel(other);
    }
  },

  /** Stop everything, for leaving the page or hiding the tab. */
  cancelAll() {
    for (const owner of [...this._live.keys()]) this.cancel(owner);
  },

  /**
   * Whether this owner's last job ran to the end.
   *
   * False both for work that was interrupted and for work never started, and
   * the caller wants the same thing in either case: to load.
   */
  finished(owner) {
    return this._done.has(owner);
  },

  /**
   * Whether this owner has work that was begun and did not finish — cut short
   * by leaving the view or hiding the tab — and is not running now.
   */
  interrupted(owner) {
    return !this._live.has(owner) && !this._done.has(owner);
  },

  /** Whether anything is running, for a caller that wants to avoid piling on. */
  busy(owner) {
    return owner === undefined ? this._live.size > 0 : this._live.has(owner);
  },

  // ---- the bar ----------------------------------------------------------

  /**
   * Attach to the element that reports progress.
   *
   * One bar, in the top bar, so what is loading is visible from whichever tab
   * you are on rather than being written into whichever panel happened to
   * start it. It carries a Stop, because a thing you cannot stop is a thing
   * you do not control.
   */
  attach(el) {
    this._bar = el;
    if (!el) return;
    el.innerHTML =
      '<span class="job-text"></span>'
      + '<span class="job-track"><i class="job-fill"></i></span>'
      + '<button type="button" class="job-stop" title="Stop loading">Stop</button>';
    el.querySelector('.job-stop').addEventListener('click', () => this.cancelAll());
    this._clear();
  },

  _show(owner, text, done, total) {
    const el = this._bar;
    if (!el) return;
    el.hidden = false;
    el.querySelector('.job-text').textContent =
      total > 0 ? `${text} ${Math.min(done, total)} / ${total}` : text;
    const fill = el.querySelector('.job-fill');
    // Without a total there is nothing honest to fill, so the bar says it is
    // working rather than inventing a fraction.
    const known = total > 0;
    el.classList.toggle('is-indeterminate', !known);
    fill.style.width = known ? `${Math.round(100 * Math.min(1, done / total))}%` : '';
  },

  _clear() {
    const el = this._bar;
    if (!el) return;
    if (this._live.size) return;          // something else is still going
    el.hidden = true;
    el.classList.remove('is-indeterminate');
  }
};
