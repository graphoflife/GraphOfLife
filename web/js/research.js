/*
 * The Viewer's four instruments: Lineage, Flow modules, Diagrams and
 * Correlations, each a way of looking at one run other than its network.
 *
 * They were a tab of their own, Research, with a run picker of their own; one
 * run chosen twice, in two places, was two pickers to keep in step and a
 * question every visitor had to answer again. They live in the Viewer now as
 * its modes, and look at whatever run the Viewer has — the picker at the top
 * of the Viewer is the only one. (The tab's other half, its documents, are
 * pages of the Research Notebook: see book.js.)
 *
 * The modes are a table rather than a chain of comparisons. Each names the
 * module that implements it, and Lineage, FlowView, Diagrams and Correlations
 * share the same six methods — init, setRuns, say, resize, draw, load — which
 * is where that becomes an interface instead of a coincidence. Everything
 * below iterates the table rather than naming a mode, so another one is an
 * entry here, a section in index.html and a button in the Viewer's head.
 */
const Research = {
  MODES: {
    lineage:    { group: 'analysis', label: 'Lineage',      view: Lineage },
    flow:       { group: 'analysis', label: 'Flow modules', view: FlowView },
    // Both of these summarise or correlate a whole window of the run, which is
    // minutes of work on a large world: they only start when this mode is
    // opened on a run, never in the background. The theses live in Diagrams,
    // as ready-made time series.
    diagrams:   { group: 'analysis', label: 'Diagrams',     view: Diagrams },
    correlations: { group: 'analysis', label: 'Correlations', view: Correlations }
  },

  runs: [],
  runId: null,
  mode: null,
  active: false,

  /** The modes, each with its view. */
  get runModes() {
    return Object.values(this.MODES);
  },

  get view() { return this.mode ? this.MODES[this.mode].view : null; },

  init() {
    for (const { view } of this.runModes) view.init();
    // A theme change repaints whichever instrument is open; the others draw
    // in the new inks when they are next shown.
    window.addEventListener('themechange', () => { if (this.active) this.view?.draw(); });
  },

  /**
   * One of the instruments was chosen in the Viewer's head. The list of runs
   * is read on the way in, every time, because it goes stale the moment a
   * simulation is created or records its first frames — and listing costs
   * one request and does not start the engine.
   */
  async enter(mode) {
    this.mode = mode;
    this.active = true;
    for (const name of Object.keys(this.MODES)) {
      document.getElementById(`research-${name}`).hidden = name !== mode;
    }
    const { view } = this.MODES[mode];
    // Whatever the mode you just left was reading, it is not what you are
    // looking at now. A view owns its own loads, so the view is what to keep.
    Jobs.only(view);
    // A canvas sized while it was hidden has no size, so a view measures
    // itself again on the way in.
    view.resize();
    view.draw();

    await this.listRuns();
    if (!this.active || this.mode !== mode) return;
    this.open(Viewer.runId);
  },

  /** Back to the network, or away from the Viewer: whatever was being read stops. */
  leave() {
    this.active = false;
    if (this.view) Jobs.cancel(this.view);
  },

  /** Back from a hidden window: pick up a load that hiding cut short. */
  resume() {
    if (this.active && this.view && Jobs.due(this.view)) this.open(this.runId);
  },

  /**
   * The run to read, as the Viewer has it. Coming back to a view that has
   * read this run, or is still reading it, is coming back to a picture it has
   * or is about to have; another run, or a read cut short, is read.
   */
  open(runId) {
    this.runId = runId || null;
    const view = this.view;
    if (!view) return;
    if (!this.runId) {
      view.say('Choose a simulation in the picker above.');
      return;
    }
    Jobs.only(view);
    if (view.runId !== this.runId || Jobs.due(view)) view.load(this.runId);
  },

  /** Choose a run for the reader, as a saved diagram does: the Viewer takes it, and so do the views. */
  pick(runId) {
    return Viewer.load(runId);
  },

  async listRuns() {
    // Each view owns its own note element and already has the method for
    // writing to it, so this asks rather than reaching past it into the DOM.
    const say = (text) => {
      for (const { view } of this.runModes) view.say(text);
    };
    // Only said while there is nothing on screen yet: overwriting a drawn
    // view's note every time reads as though it were about to reload.
    if (!this.runId) say('Looking for simulations…');
    try {
      await API.choose();
      const data = await API.listRuns();
      // Every instrument needs at least two frames to compare anything, so a
      // simulation that has just started is not offered until it has some.
      this.runs = (data.runs || []).filter(r => r.frame_count > 1);
      for (const { view } of this.runModes) view.setRuns(this.runs);
      if (!this.runs.length) {
        say('No simulation has recorded enough frames yet. '
          + 'Run one from the Simulations tab and come back.');
      }
    } catch (err) {
      say(`Could not reach the simulations: ${err.message}`);
    }
  }
};
