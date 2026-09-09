/*
 * The Research tab: point an instrument at a run, or read a document.
 *
 * The run list and the picker live here rather than in any one view, so there
 * is one list, one fetch, and switching between views reloads nothing.
 *
 * The modes are a table rather than a chain of comparisons. A mode is one of
 * two kinds, and the table says which by naming the module that implements it:
 *
 *   view  a way of looking at one run. Lineage, FlowView and Theses share the
 *         same six methods — init, setRuns, say, resize, draw, load — and this
 *         is where that becomes an interface instead of a coincidence.
 *   page  words on a screen. No run, so no picker, no run list, nothing to
 *         resize; it is painted once, the first time it is asked for.
 *
 * They are also grouped, and the grouping is the same distinction one level
 * up: **analysis** is the instruments, **reading** is the documents. Six
 * buttons in one row said those were the same kind of thing, which made the
 * tab read as a list of features rather than as two places.
 *
 * Everything below iterates the table rather than naming a mode, so another
 * one is an entry here and a section in index.html — including its buttons,
 * which are built from this rather than written out twice.
 */
const Research = {
  GROUPS: [
    { id: 'analysis', label: 'Analysis' },
    { id: 'reading',  label: 'Reading' }
  ],

  MODES: {
    lineage:    { group: 'analysis', label: 'Lineage',      view: Lineage },
    flow:       { group: 'analysis', label: 'Flow modules', view: FlowView },
    // `choose` means no run is opened until one is picked. The others show
    // something the moment they have a run; this one starts a minutes-long
    // summary of whatever happened to be first in the list, which is a slow
    // answer to a question nobody asked.
    theses:     { group: 'analysis', label: 'Theses',       view: Theses, choose: true },
    diagrams:   { group: 'analysis', label: 'Diagrams',     view: Diagrams, choose: true },
    notes:      { group: 'reading',  label: 'Findings',     page: Notes },
    literature: { group: 'reading',  label: 'Literature',   page: Literature },
    graphs:     { group: 'reading',  label: 'Graphs',       page: Graphs }
  },

  runs: [],
  runId: null,
  mode: 'lineage',

  // Bumped whenever the visible mode or the chosen run changes. A view checks
  // it between requests and stops asking for more the moment it is no longer
  // the thing on screen — otherwise leaving a tab left a summary of a large run
  // grinding away behind it, and the page felt stuck because it was.
  epoch: 0,

  /** True once whatever a caller started is no longer what is being looked at. */
  stale(at) { return at !== this.epoch; },

  /** The modes that are about a run, which is every mode that has a view. */
  get runModes() {
    return Object.values(this.MODES).filter(m => m.view);
  },

  get view() { return this.MODES[this.mode].view; },
  get group() { return this.MODES[this.mode].group; },

  init() {
    this.picker = document.getElementById('researchRun');
    if (!this.picker) return;
    for (const { view } of this.runModes) view.init();

    this.groupBar = document.getElementById('researchGroups');
    this.modeBar = document.getElementById('researchModes');

    this.groupBar.replaceChildren(...this.GROUPS.map(group => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = group.label;
      button.dataset.group = group.id;
      // Picking a group lands on its first mode, so a group is never selected
      // with nothing shown under it.
      button.addEventListener('click', () => this.show(this.firstIn(group.id)));
      return button;
    }));

    this.modeBar.replaceChildren(...Object.entries(this.MODES).map(([name, mode]) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = mode.label;
      button.dataset.mode = name;
      button.dataset.group = mode.group;
      button.addEventListener('click', () => this.show(name));
      return button;
    }));

    // An explicit choice sticks. `choose` means "do not pick one for them",
    // not "throw away the one they picked" — clearing it on every visit back
    // to the tab would be its own kind of rude.
    this.picker.addEventListener('change', () => {
      this.userPicked = !!this.picker.value;
      this.open(this.picker.value);
    });
    document.getElementById('researchRefresh')
      .addEventListener('click', () => this.listRuns());

    this.show(this.mode);
  },

  firstIn(group) {
    return Object.keys(this.MODES).find(name => this.MODES[name].group === group);
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
    this.epoch++;
    const group = this.group;

    for (const button of this.groupBar.querySelectorAll('button')) {
      button.classList.toggle('active', button.dataset.group === group);
    }
    for (const button of this.modeBar.querySelectorAll('button')) {
      button.classList.toggle('active', button.dataset.mode === mode);
      button.hidden = button.dataset.group !== group;
    }
    for (const name of Object.keys(this.MODES)) {
      document.getElementById(`research-${name}`).hidden = name !== mode;
    }

    const { view, page, choose } = this.MODES[mode];
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

    // The picker shows nothing chosen for a mode that waits to be asked, so
    // entering it does not silently begin summarising a run that takes minutes
    // and that nobody asked about.
    const waiting = choose && !this.userPicked;
    this.picker.value = waiting ? '' : (this.runId || '');
    view.load(waiting ? null : (this.runId || null));
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

      const blank = document.createElement('option');
      blank.value = '';
      blank.textContent = 'Choose a simulation…';
      this.picker.replaceChildren(blank, ...this.runs.map(run => {
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

      const waiting = this.MODES[this.mode].choose && !this.userPicked;
      // Keep whatever was being looked at, if it is still there.
      const had = this.runId;
      const wanted = this.runs.some(r => r.id === had) ? had
                   : (waiting ? '' : this.runs[0].id);
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
    this.runId = runId || null;
    this.epoch++;
    await this.view?.load(this.runId);
  }
};
