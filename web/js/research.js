/*
 * The Research tab: pick a run and look at it, or read the literature the
 * whole project is measured against.
 *
 * The run list and the picker live here rather than in any one view, so there
 * is one list, one fetch, and switching between views reloads nothing.
 *
 * The modes are a table rather than a chain of comparisons. Lineage and
 * FlowView already implement the same five methods — init, setRuns, resize,
 * draw, load — and MODES is where that becomes an interface instead of a
 * coincidence: everything below iterates the table rather than naming a mode.
 * A mode with no `view` is a page rather than a way of looking at a run, which
 * is the one fact that decides whether it gets the picker, the run list and a
 * line to report trouble on. Adding a fourth mode is an entry here and a
 * section in index.html; it is not a new branch anywhere.
 */
const Research = {
  MODES: {
    lineage:    { view: Lineage,  note: 'lineageNote' },
    flow:       { view: FlowView, note: 'flowNote' },
    literature: { }
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
    // Static, stateless and cheap. Nothing about it changes later, so there is
    // no redraw to arrange and no reason to defer it.
    Literature.render();

    this.picker.addEventListener('change', () => this.open(this.picker.value));
    document.getElementById('researchRefresh')
      .addEventListener('click', () => this.listRuns());

    for (const button of document.querySelectorAll('#researchModes button')) {
      button.addEventListener('click', () => this.show(button.dataset.mode));
    }
  },

  async setActive(active) {
    if (!active || this._loaded) return;
    this._loaded = true;
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

    // A mode with no view has no run, so the picker has nothing to pick for.
    const view = this.view;
    document.getElementById('researchRunField').hidden = !view;
    if (!view) return;

    // A canvas sized while it was hidden has no size, so a view measures
    // itself again on the way in.
    view.resize();
    view.draw();
    if (this.runId) view.load(this.runId);
  },

  async listRuns() {
    const say = (text) => {
      for (const { note } of this.runModes) {
        document.getElementById(note).textContent = text;
      }
    };
    say('Looking for simulations…');
    try {
      await API.choose();
      const data = await API.listRuns();
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
      const wanted = this.runs.some(r => r.id === this.runId) ? this.runId : this.runs[0].id;
      this.picker.value = wanted;
      await this.open(wanted);
    } catch (err) {
      say(`Could not reach the simulations: ${err.message}`);
    }
  },

  async open(runId) {
    this.runId = runId;
    await this.view?.load(runId);
  }
};
