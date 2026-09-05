/*
 * The Research tab: pick a run, then look at it one of two ways — or read the
 * literature the whole project is measured against, which needs no run at all.
 *
 * The run list and the picker live here rather than in either view, so there
 * is one list, one fetch, and switching between the views does not reload
 * anything. Each view is handed the run and gets on with it.
 */
const Analysis = {
  runs: [],
  runId: null,
  mode: 'lineage',

  init() {
    this.picker = document.getElementById('analysisRun');
    if (!this.picker) return;
    Lineage.init();
    FlowView.init();

    this.picker.addEventListener('change', () => this.open(this.picker.value));
    document.getElementById('analysisRefresh')
      .addEventListener('click', () => this.listRuns());

    for (const button of document.querySelectorAll('#analysisModes button')) {
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
    for (const button of document.querySelectorAll('#analysisModes button')) {
      button.classList.toggle('active', button.dataset.mode === mode);
    }
    for (const name of ['lineage', 'flow', 'literature']) {
      document.getElementById(`analysis-${name}`).hidden = mode !== name;
    }
    // The run picker means nothing to the reading list.
    document.getElementById('analysisRunField').hidden = mode === 'literature';
    if (mode === 'literature') return;

    // A canvas sized while it was hidden has no size, so both views measure
    // themselves again on the way in.
    const view = mode === 'flow' ? FlowView : Lineage;
    view.resize();
    view.draw();
    if (this.runId) view.load(this.runId);
  },

  async listRuns() {
    const say = (text) => {
      document.getElementById('lineageNote').textContent = text;
      document.getElementById('flowNote').textContent = text;
    };
    say('Looking for simulations…');
    try {
      await API.choose();
      const data = await API.listRuns();
      this.runs = (data.runs || []).filter(r => r.frame_count > 1);
      Lineage.setRuns(this.runs);
      FlowView.setRuns(this.runs);

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
    await (this.mode === 'flow' ? FlowView : Lineage).load(runId);
  }
};
