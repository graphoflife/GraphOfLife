/*
 * The Research tab: pick a run and look at it, or read the literature the
 * whole project is measured against.
 *
 * The run list and the picker live here rather than in any one view, so there
 * is one list, one fetch, and switching between views reloads nothing.
 *
 * The modes are a table rather than a chain of comparisons. A mode is one of
 * two kinds, and the table says which by naming the module that implements it:
 *
 *   view  a way of looking at one run. Lineage and FlowView already share the
 *         same six methods — init, setRuns, say, resize, draw, load — and this
 *         is where that becomes an interface instead of a coincidence.
 *   page  words on a screen. No run, so no picker, no run list, nothing to
 *         resize; it is painted once, the first time it is asked for.
 *
 * Everything below iterates the table rather than naming a mode, so a fourth
 * is an entry here and a section in index.html.
 */
const Research = {
  MODES: {
    lineage:    { view: Lineage },
    flow:       { view: FlowView },
    notes:      { page: Notes },
    literature: { page: Literature },
    graphs:     { page: Graphs },
    theses:     { view: Theses }
  },

  runs: [],
  runId: null,
  mode: 'lineage',

  /** The modes that are about a run, which is every mode that has a view. */
  get runModes() {
    return Object.values(this.MODES).filter(m => m.view);
  },

  get view() { return this.MODES[this.mode].view; },

  init() {
    this.picker = document.getElementById('researchRun');
    if (!this.picker) return;
    for (const { view } of this.runModes) view.init();

    this.picker.addEventListener('change', () => this.open(this.picker.value));
    document.getElementById('researchRefresh')
      .addEventListener('click', () => this.listRuns());

    for (const button of document.querySelectorAll('#researchModes button')) {
      button.addEventListener('click', () => this.show(button.dataset.mode));
    }
  },

  /**
   * Re-read the list of runs on the way into the tab, every time.
   *
   * It used to list once per page load. A simulation created afterwards — or
   * one that had not yet recorded the two frames these views need, which is
   * every simulation for its first moments — never appeared in the picker, and
   * the only way to see it was to notice the Refresh button. Listing costs one
   * request and does not start the engine.
   */
  async setActive(active) {
    if (!active) return;
    await this.listRuns();
  },

  show(mode) {
    this.mode = mode;
    for (const button of document.querySelectorAll('#researchModes button')) {
      button.classList.toggle('active', button.dataset.mode === mode);
    }
    for (const name of Object.keys(this.MODES)) {
      document.getElementById(`research-${name}`).hidden = name !== mode;
    }

    const { view, page } = this.MODES[mode];
    // Only a run has a run to pick.
    document.getElementById('researchRunField').hidden = !view;

    // Painted on the way in rather than at startup: it is the same words every
    // time, and a visitor who never opens this tab should not be building them.
    page?.render();
    if (!view) return;

    // A canvas sized while it was hidden has no size, so a view measures
    // itself again on the way in.
    view.resize();
    view.draw();
    if (this.runId) view.load(this.runId);
  },

  async listRuns() {
    // Each view owns its own note element and already has the method for
    // writing to it, so this asks rather than reaching past it into the DOM.
    const say = (text) => {
      for (const { view } of this.runModes) view.say(text);
    };
    // Only said while there is nothing on screen yet. Re-entering the tab
    // relists, and overwriting a drawn view's note with "Looking for
    // simulations…" every time reads as though it were about to reload.
    if (!this.runId) say('Looking for simulations…');
    try {
      await API.choose();
      const data = await API.listRuns();
      // Both views need at least two frames to compare anything, so a
      // simulation that has just started is not offered until it has some.
      this.runs = (data.runs || []).filter(r => r.frame_count > 1);
      for (const { view } of this.runModes) view.setRuns(this.runs);

      this.picker.replaceChildren(...this.runs.map(run => {
        const option = document.createElement('option');
        option.value = run.id;
        option.textContent = `${run.name} — ${formatNumber(run.frame_count)} frames`;
        return option;
      }));
      if (!this.runs.length) {
        say('No simulation has recorded enough frames yet. '
          + 'Run one from the Simulations tab and come back.');
        return;
      }
      // Keep whatever was being looked at, if it is still there.
      const had = this.runId;
      const wanted = this.runs.some(r => r.id === had) ? had : this.runs[0].id;
      this.picker.value = wanted;
      // Only load when the choice actually changed. Re-entering the tab should
      // not refetch a window that is already drawn — the lineage keeps no
      // frames, so that would be two hundred requests for the same picture.
      if (wanted !== had) await this.open(wanted);
    } catch (err) {
      say(`Could not reach the simulations: ${err.message}`);
    }
  },

  async open(runId) {
    this.runId = runId;
    await this.view?.load(runId);
  }
};
