/*
 * The Simulations tab.
 *
 * One thing on screen: the simulations, as a grid of cards. Everything that
 * acts on a simulation is on its card; everything that makes one is behind the
 * first cell, which opens the settings in a dialog. The settings used to sit
 * permanently beside the list, which meant most of the tab was a form nobody
 * was filling in.
 *
 * The form itself is unchanged and still generated from whatever the backend
 * reports as its defaults, so adding a field to SimConfig only needs a
 * matching input in index.html.
 */
const RunsView = {
  defaults: null,
  runs: [],
  pollTimer: null,
  failedPolls: 0,

  // Runs we have asked to stop but which have not stopped yet. A worker only
  // notices the request between iterations, so on a slow world the button can
  // sit there for many seconds looking as though the click missed. Holding the
  // request here lets the card say so straight away.
  stopping: new Set(),

  // Which config keys a card shows. Kept short on purpose: a card is for
  // telling runs apart at a glance, and the whole dataclass on every card is
  // not a glance. The rest is behind "Settings" in the menu.
  CARD_SETTINGS: [
    ['total_tokens', 'tokens'],
    ['brain_kind', 'brain'],
    ['hidden_layers', 'hidden'],
    ['seed', 'seed']
  ],

  // Everything, grouped the way the form groups it, for reading one run off
  // against another. Short labels and no explanations — the explanations are
  // in the form that sets them.
  ALL_SETTINGS: [
    ['Simulation', [
      ['total_tokens', 'Total tokens'],
      ['tokens_created_per_phase', 'New tokens each phase'],
      ['exchange_messages', 'Exchange messages'],
      ['message_amount', 'Message size'],
      ['message_prepass', 'Message pre-pass'],
      ['random_input_amount', 'Noise inputs'],
      ['allow_handover', 'Handover'],
      ['allow_revolutions', 'Revolutions']
    ]],
    ['Seed graph', [
      ['n_nodes', 'Agents'],
      ['k_neighbors', 'Neighbours k'],
      ['rewire_p', 'Shortcuts'],
      ['seed', 'Seed']
    ]],
    ['Brain', [
      ['brain_kind', 'Kind'],
      ['brain_bits', 'Bits per input'],
      ['hidden_layers', 'Hidden layers']
    ]],
    ['Mutation', [
      ['mutation_probability', 'Probability'],
      ['mutation_noise_std', 'Noise std'],
      ['mutation_sparsity', 'Sparsity']
    ]],
    ['Run control', [
      ['extinction_threshold', 'Extinct below'],
      ['checkpoint_every', 'Checkpoint every'],
      ['export_every', 'Record every'],
      ['export_decisions', 'Record decisions']
    ]]
  ],

  init() {
    this.grid = document.getElementById('simGrid');
    if (!this.grid) return;
    this.form = document.getElementById('newRunForm');
    this.errorEl = document.getElementById('createError');
    this.dialog = document.getElementById('simDialog');

    this.form.addEventListener('submit', e => this.onCreate(e));
    document.getElementById('resetDefaults').addEventListener('click', () => this.applyDefaults());
    document.getElementById('refreshRuns').addEventListener('click', () => this.refresh());
    document.getElementById('simDialogClose').addEventListener('click', () => this.closeDialog());
    document.getElementById('simDialogCancel').addEventListener('click', () => this.closeDialog());
    document.getElementById('settingsDialogClose')
      .addEventListener('click', () => document.getElementById('settingsDialog').close());

    for (const input of this.form.querySelectorAll('[data-cfg]')) {
      input.addEventListener('input', () => {
        this.syncDependentFields();
        this.updateDerived();
      });
    }
    this.syncDependentFields();

    // Choosing a brain also chooses what that brain needs. Bound to `change`
    // and not `input`, so it fires when a kind is actually picked rather than
    // while the field is being touched.
    const kind = document.getElementById('cfg_brain_kind');
    if (kind) kind.addEventListener('change', () => this.applyBrainPreset(kind.value));

    // One menu open at a time, and a click anywhere else closes it.
    document.addEventListener('click', () => this.closeMenus());
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') this.closeMenus();
    });
  },

  /**
   * Say where the simulation is running.
   *
   * With a server this is one quiet line. Without one the page has to fetch a
   * Python runtime, numpy and networkx before anything can happen, which takes
   * several seconds on a first visit and looks like a hung page if nothing
   * says otherwise.
   */
  showBackend(progress) {
    const el = document.getElementById('backendNotice');
    if (!el) return;
    el.classList.remove('hidden');

    if (!API.runsInBrowser) {
      el.innerHTML = 'Running against your local <b>gol_server.py</b>. '
        + 'Runs are written to disk and survive a reload.';
      return;
    }

    const stage = (progress && progress.stage) || 'idle';
    if (stage === 'ready') {
      el.innerHTML = 'Running <b>in this browser</b>. Runs are stored locally and never leave this machine.';
      this.showStorage(el);
      return;
    }
    const detail = (progress && progress.detail) || 'starting';
    const line = document.createElement('span');
    line.textContent = ` — ${detail}. `;
    el.replaceChildren(
      Object.assign(document.createElement('span'), { className: 'spin' }),
      Object.assign(document.createElement('span'), {
        textContent: 'Setting up the engine in your browser'
      }),
      line,
      Object.assign(document.createElement('span'), {
        textContent: 'This happens once, and the download is cached afterwards.'
      })
    );
  },

  /**
   * How much has been stored, once the browser will say.
   *
   * Written into its own element rather than appended, because the notice is
   * redrawn more than once while a backend settles and every answer that came
   * back late added another copy of the same sentence.
   */
  async showStorage(el) {
    try {
      const info = await API.storage();
      if (!info || !info.usage) return;
      let span = el.querySelector('.backend-storage');
      if (!span) {
        span = document.createElement('span');
        span.className = 'backend-storage';
        el.appendChild(span);
      }
      span.textContent = ` — ${formatBytes(info.usage)} used so far`
        + (info.persisted ? ', kept until you delete it.' : '.');
    } catch (err) {
      /* the browser may simply refuse to say */
    }
  },

  /**
   * Choose a backend and ask it for its defaults — the first time this tab is
   * opened, and not before.
   *
   * This used to run at page load. On a static host there is no server to
   * answer, so it fell through to the in-browser engine and started fetching a
   * Python runtime — several megabytes and a WebAssembly compile — while the
   * front page was trying to lay out five thousand agents and paint them sixty
   * times a second. A visitor who only reads the front page or the explanation
   * now downloads no Python at all.
   */
  async activate() {
    if (this._activated) return;
    this._activated = true;

    await API.choose();
    API.onProgress = (progress) => this.showBackend(progress);
    this.showBackend(API.progress);

    try {
      this.defaults = await API.defaults();
      this.applyDefaults();
      this.showBackend({ stage: 'ready' });
    } catch (err) {
      this.errorEl.textContent = API.runsInBrowser
        ? `The in-browser engine could not start: ${err.message}`
        : `Could not reach the server: ${err.message}`;
    }
    await this.refresh();
  },

  // ---- the settings dialog ---------------------------------------------

  /**
   * Open the settings.
   *
   * `config` pre-fills the form, which is what "new simulation with the same
   * settings" is: the same dialog, opened on somebody else's numbers.
   */
  openDialog({ config = null, title = 'New simulation', name = '' } = {}) {
    if (!this.defaults) return;
    document.getElementById('simDialogTitle').textContent = title;
    this.fillForm(config || this.defaults.config);
    document.getElementById('runName').value = name;
    this.errorEl.textContent = '';
    this.errorEl.classList.remove('ok');
    this.dialog.showModal();
  },

  closeDialog() {
    if (this.dialog.open) this.dialog.close();
  },

  /**
   * Every setting a run was made with, as a list.
   *
   * A card shows four; this shows all of them, in the order the form asks for
   * them, so two runs can be read off against each other. The seed is always a
   * number here — one is drawn and written down when a run is created, so
   * "blank means random" no longer means "nobody will ever know which".
   */
  showSettings(run) {
    const cfg = run.config || {};
    const body = document.getElementById('settingsList');
    document.getElementById('settingsDialogTitle').textContent = `Settings — ${run.name}`;

    const groups = this.ALL_SETTINGS.map(([heading, rows]) => {
      const group = document.createElement('section');
      group.append(Object.assign(document.createElement('h3'), { textContent: heading }));
      const list = document.createElement('dl');
      for (const [key, label] of rows) {
        let value = cfg[key];
        if (value === null || value === undefined || value === '') value = 'auto';
        else if (Array.isArray(value)) value = value.join(' / ');
        else if (typeof value === 'boolean') value = value ? 'yes' : 'no';
        else if (typeof value === 'number') value = formatNumber(value);
        list.append(Object.assign(document.createElement('dt'), { textContent: label }),
                    Object.assign(document.createElement('dd'), { textContent: String(value) }));
      }
      group.append(list);
      return group;
    });
    body.replaceChildren(...groups);
    document.getElementById('settingsDialog').showModal();
  },

  fillForm(cfg) {
    for (const input of this.form.querySelectorAll('[data-cfg]')) {
      const value = cfg[input.dataset.cfg];
      if (input.type === 'checkbox') input.checked = Boolean(value);
      else if (Array.isArray(value)) input.value = value.join(', ');
      else input.value = (value === null || value === undefined) ? '' : value;
    }
    this.syncDependentFields();
    this.updateDerived();
  },

  applyDefaults() {
    if (!this.defaults) return;
    this.fillForm(this.defaults.config);
    this.errorEl.textContent = '';
  },

  /**
   * Fill in what a brain kind wants, when that kind is chosen.
   *
   * A binary brain is not a float brain with cheaper weights: its units carry
   * a bit each rather than many, so it needs wider layers, and its smallest
   * possible mutation is a whole step, so it needs a gentler rate. Leaving
   * those to be known meant picking "binary" quietly produced a run that
   * mostly died out. The numbers come from the engine, not from here, so the
   * form and the defaults cannot drift apart.
   */
  applyBrainPreset(kind) {
    const preset = (this.defaults && this.defaults.brain_presets || {})[kind];
    if (!preset) return;
    const layers = document.getElementById('cfg_hidden_layers');
    const sparsity = document.getElementById('cfg_mutation_sparsity');
    if (layers && preset.hidden_layers) layers.value = preset.hidden_layers.join(', ');
    if (sparsity && preset.mutation_sparsity !== undefined) {
      sparsity.value = preset.mutation_sparsity;
    }
    this.updateDerived();
  },

  /**
   * Settings that only mean something alongside another one.
   *
   * The pre-pass is a pass that exists to send messages; with messages off it
   * has nothing to do, and the engine rejects the combination outright. Better
   * to make it unaskable than to let it be asked and refused — so the box is
   * disabled and cleared, and the reason is on it.
   */
  syncDependentFields() {
    const messages = document.getElementById('cfg_exchange_messages');
    const prepass = document.getElementById('cfg_message_prepass');
    if (!messages || !prepass) return;
    prepass.disabled = !messages.checked;
    if (prepass.disabled) prepass.checked = false;
    prepass.closest('.field').classList.toggle('disabled', prepass.disabled);
    prepass.title = prepass.disabled
      ? 'Needs "Exchange messages": a pass that only sends messages has nothing to do without them.'
      : '';
  },

  readConfig() {
    const config = {};
    for (const input of this.form.querySelectorAll('[data-cfg]')) {
      const key = input.dataset.cfg;

      if (input.type === 'checkbox') {
        config[key] = input.checked;
      } else if (key === 'hidden_layers') {
        config[key] = input.value.split(',')
          .map(s => parseInt(s.trim(), 10))
          .filter(n => Number.isFinite(n) && n > 0);
      } else if (input.value === '') {
        // Blank means "use the default"; a blank seed means "random".
        if (key === 'seed') config[key] = null;
      } else {
        const num = Number(input.value);
        config[key] = Number.isFinite(num) ? num : input.value;
      }
    }
    return config;
  },

  async onCreate(event) {
    event.preventDefault();
    this.errorEl.textContent = '';
    try {
      const name = document.getElementById('runName').value;
      await API.createRun(name, this.readConfig());
      this.closeDialog();
      await this.refresh();
    } catch (err) {
      this.errorEl.textContent = err.message;
    }
  },

  updateDerived() {
    const el = document.getElementById('derivedInfo');
    if (!el) return;
    const cfg = this.readConfig();

    const messages = cfg.message_amount ?? 5;
    const noise = cfg.random_input_amount ?? 5;
    // 1 is-self flag + 28 magnitudes, then the messages and the noise.
    const magnitudes = 28;
    const inputs = 1 + magnitudes + 4 * messages + noise;
    // Revolutions add a fraction pair; handover adds a yes/no pair plus its
    // mode pair. Both are absent from the brain when switched off.
    const outputs = 9 + (cfg.allow_revolutions ? 2 : 0)
                      + (cfg.allow_handover ? 4 : 0)
                      + messages;

    // A binary brain spreads its magnitudes across a ladder of bits, so its
    // first layer is wider even though each weight costs a fraction as much.
    // The flag, the messages and the noise are already bits and stay one row.
    const binary = cfg.brain_kind === 'binary';
    const bits = binary ? (cfg.brain_bits || 16) : 1;
    const firstLayer = binary
      ? magnitudes * bits + 1 + 4 * messages + noise
      : inputs;
    const bytesPerWeight = binary ? 1 : (cfg.brain_kind === 'float16' ? 2 : 8);

    const n = cfg.n_nodes > 0 ? cfg.n_nodes : Math.floor((cfg.total_tokens || 0) / 100);
    const k = cfg.k_neighbors > 0 ? cfg.k_neighbors : Math.max(Math.floor(n / 100), 5);

    const layers = (cfg.hidden_layers && cfg.hidden_layers.length) ? cfg.hidden_layers : [50, 45, 40, 35, 30];
    const sizes = [firstLayer, ...layers, outputs];
    let params = 0;
    for (let i = 0; i < sizes.length - 1; i++) params += sizes[i] * sizes[i + 1] + sizes[i + 1];

    el.textContent =
      `${formatNumber(inputs)} inputs, ${formatNumber(firstLayer)} first layer, `
      + `${formatNumber(outputs)} outputs, ${formatNumber(params)} weights per brain `
      + `(≈ ${formatBytes(params * bytesPerWeight)}). `
      + `Seed graph ${formatNumber(n)} agents, ${formatNumber(k)} neighbours each. `
      + `The pre-pass, if on, doubles the forward passes per iteration.`;
  },

  // ---- the grid --------------------------------------------------------

  async refresh() {
    try {
      const data = await API.listRuns();
      this.runs = data.runs || [];
      this.failedPolls = 0;
      this.render();
    } catch (err) {
      // A poll that fails is not a poll that should stop. The server going
      // away for a moment — a restart, a slow frame write — used to end the
      // polling for good, so a run carried on in the background while its card
      // sat frozen until the page was reloaded by hand.
      this.failedPolls += 1;
      const p = document.createElement('p');
      p.className = 'sim-empty';
      p.textContent = `Could not load simulations: ${err.message}`;
      this.grid.replaceChildren(this.newCard(), p);
    }
    this.schedulePoll();
  },

  /**
   * Poll while something is running, and keep trying after a failure.
   *
   * Backed off as failures pile up, so a server that is gone for good is not
   * asked once a second forever, but never given up on entirely.
   */
  schedulePoll() {
    clearTimeout(this.pollTimer);
    const failed = this.failedPolls;
    if (!failed && !this.runs.some(r => r.running)) return;
    const delay = failed ? Math.min(15000, 1500 * 2 ** (failed - 1)) : 1500;
    this.pollTimer = setTimeout(() => this.refresh(), delay);
  },

  render() {
    const open = this.openMenuFor;          // survive a redraw mid-poll
    this.grid.replaceChildren(this.newCard(), ...this.runs.map(run => this.card(run)));
    if (open) this.toggleMenu(open, true);
  },

  /** The first cell: the only way to make a simulation. */
  newCard() {
    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'sim-card sim-card-new';
    el.append(
      icon('<path d="M12 5v14M5 12h14"/>', 'sim-new-mark'),
      Object.assign(document.createElement('span'), { textContent: 'Create new Simulation' })
    );
    el.addEventListener('click', () => this.openDialog());
    return el;
  },

  card(run) {
    const el = document.createElement('article');
    el.className = 'sim-card';
    el.dataset.id = run.id;

    const running = Boolean(run.running);
    // The run has stopped, so forget the request; anything still in the set is
    // a request the worker has yet to act on.
    if (!running) this.stopping.delete(run.id);
    const stopping = running && this.stopping.has(run.id);

    // `running` is the live fact and the status is a stored one, so where they
    // disagree the live one wins. A backend that has restarted still has
    // "running" written down for whatever was going when it went away, and the
    // card used to believe it — a green pulsing dot over a button offering to
    // resume the thing it claimed was already going.
    let state = run.status || 'idle';
    if (stopping) state = 'stopping';
    else if (running) state = 'running';
    else if (state === 'running') state = 'interrupted';

    // ---- name, id, and the menu ----
    const head = document.createElement('header');
    head.className = 'sim-head';

    const title = document.createElement('div');
    title.className = 'sim-title';
    const h3 = document.createElement('h3');
    h3.textContent = run.name;
    h3.title = run.name;
    title.append(h3);
    if (run.name !== run.id) {
      const id = document.createElement('span');
      id.className = 'sim-id';
      id.textContent = run.id;
      title.append(id);
    }

    const menuButton = document.createElement('button');
    menuButton.type = 'button';
    menuButton.className = 'icon-button sim-menu-button';
    menuButton.title = 'More';
    menuButton.setAttribute('aria-label', `More actions for ${run.name}`);
    menuButton.append(icon('<circle cx="12" cy="5" r="1.6" FILL/>'
                         + '<circle cx="12" cy="12" r="1.6" FILL/>'
                         + '<circle cx="12" cy="19" r="1.6" FILL/>'));
    menuButton.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleMenu(run.id);
    });

    head.append(title, menuButton);

    // ---- what it is doing ----
    const status = document.createElement('div');
    status.className = `sim-state sim-state-${state}`;
    const dot = document.createElement('span');
    dot.className = 'sim-dot';
    const word = document.createElement('span');
    word.textContent = state;
    status.append(dot, word);

    // ---- what it has done ----
    const facts = document.createElement('div');
    facts.className = 'sim-facts';
    for (const [label, value] of [
      ['iterations', formatNumber(run.iteration)],
      ['frames', formatNumber(run.frame_count)],
      ['size', formatBytes(run.size_bytes)]
    ]) {
      const cell = document.createElement('div');
      cell.append(
        Object.assign(document.createElement('b'), { textContent: value }),
        Object.assign(document.createElement('i'), { textContent: label })
      );
      facts.append(cell);
    }

    const settings = document.createElement('div');
    settings.className = 'sim-settings';
    for (const [key, label] of this.CARD_SETTINGS) {
      let value = (run.config || {})[key];
      if (value === null || value === undefined || value === '') value = 'auto';
      else if (Array.isArray(value)) value = value.join('/');
      else if (typeof value === 'boolean') value = value ? 'yes' : 'no';
      else if (typeof value === 'number') value = formatNumber(value);
      const span = document.createElement('span');
      span.append(
        Object.assign(document.createElement('i'), { textContent: label }),
        document.createTextNode(' ' + value)
      );
      settings.append(span);
    }

    el.append(head, status, facts, settings);

    if (run.error) {
      const pre = document.createElement('pre');
      pre.className = 'sim-error';
      pre.textContent = run.error;
      el.append(pre);
    }

    // ---- the two things you actually do with it ----
    const actions = document.createElement('footer');
    actions.className = 'sim-actions';

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = `sim-run${running ? ' is-running' : ''}`;
    toggle.disabled = stopping;
    if (stopping) {
      toggle.append(Object.assign(document.createElement('span'), { className: 'spin' }),
                    document.createTextNode('Stopping'));
      toggle.title = 'Waiting for the worker to notice, which it does between iterations';
    } else if (running) {
      // Two bars: the same symbol every player uses for "hold it there".
      toggle.append(icon('<path d="M9 5v14M15 5v14"/>'),
                    document.createTextNode('Pause'));
      toggle.title = 'Pause after the current iteration';
    } else {
      // An arrow. Resume when there is a checkpoint to resume from.
      toggle.append(icon('<path d="M8 5.5 19 12 8 18.5Z" FILL/>'),
                    document.createTextNode(run.has_checkpoint ? 'Resume' : 'Run'));
      toggle.title = run.has_checkpoint
        ? `Continue from iteration ${formatNumber(run.checkpoint_iteration || 0)}`
        : 'Start this simulation';
    }
    toggle.addEventListener('click', () => this.toggleRun(run, running));

    const inspect = document.createElement('button');
    inspect.type = 'button';
    inspect.className = 'sim-inspect';
    inspect.title = 'Open in the Viewer';
    inspect.append(eyeIcon(), document.createTextNode('Inspect'));
    inspect.disabled = !run.frame_count;
    if (!run.frame_count) inspect.title = 'Nothing recorded yet';
    inspect.addEventListener('click', () => {
      App.showView('viewer');
      Viewer.load(run.id);
    });

    actions.append(toggle, inspect);
    el.append(actions, this.menu(run));
    return el;
  },

  /** The three-dot menu. Built with the card, shown only when asked for. */
  menu(run) {
    const menu = document.createElement('div');
    menu.className = 'sim-menu';
    menu.hidden = this.openMenuFor !== run.id;
    menu.addEventListener('click', e => e.stopPropagation());

    const item = (label, hint, handler, className = '') => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = className;
      button.append(Object.assign(document.createElement('b'), { textContent: label }),
                    Object.assign(document.createElement('i'), { textContent: hint }));
      button.addEventListener('click', async () => {
        this.closeMenus();
        try { await handler(); } catch (err) { alert(err.message); }
      });
      menu.append(button);
      return button;
    };

    item('Settings', 'Everything this run was made with',
         () => this.showSettings(run));

    item('Duplicate', 'A copy with all of its recorded data, ready to go its own way',
         async () => { await API.copyRun(run.id); await this.refresh(); });

    item('New from these settings', 'Opens the settings filled in from this one',
         () => this.openDialog({
           config: run.config,
           title: 'New simulation',
           name: ''
         }));

    item('Delete', 'This and everything it recorded', async () => {
      if (!confirm(`Delete "${run.name}" and all of its recorded data? This cannot be undone.`)) return;
      await API.deleteRun(run.id);
      await this.refresh();
    }, 'danger');

    return menu;
  },

  toggleMenu(runId, force = false) {
    const wasOpen = this.openMenuFor === runId;
    this.closeMenus();
    if (wasOpen && !force) return;
    this.openMenuFor = runId;
    const card = [...this.grid.querySelectorAll('.sim-card')]
      .find(c => c.dataset.id === runId);
    const menu = card && card.querySelector('.sim-menu');
    if (menu) menu.hidden = false;
    else this.openMenuFor = null;
  },

  closeMenus() {
    this.openMenuFor = null;
    for (const menu of this.grid ? this.grid.querySelectorAll('.sim-menu') : []) {
      menu.hidden = true;
    }
  },

  async toggleRun(run, running) {
    try {
      if (running) {
        // Marked before the request goes out, and redrawn immediately: the
        // point of the cue is to cover the wait, so it cannot wait itself.
        this.stopping.add(run.id);
        this.render();
        await API.stopRun(run.id);
      } else {
        await API.startRun(run.id);
      }
      await this.refresh();
    } catch (err) {
      this.stopping.delete(run.id);
      this.render();
      alert(err.message);
    }
  }
};

/** A small inline icon, from a path drawn on a 24-unit square. */
function icon(body, className = 'sim-icon') {
  const wrap = document.createElement('span');
  wrap.className = className;
  wrap.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" `
    + `stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">`
    + body.replace(/FILL/g, 'fill="currentColor" stroke="none"')
    + `</svg>`;
  return wrap;
}

/**
 * The eye that means "look at this".
 *
 * The same one the Explanation puts behind its two observation steps, from the
 * same definition — an agent looking at its neighbourhood and a person looking
 * at a run are the same verb, and they should not be two different drawings.
 */
function eyeIcon() {
  const wrap = document.createElement('span');
  wrap.className = 'sim-icon';
  wrap.innerHTML = Emblems.svg('eye', { ink: 'currentColor', width: 1.7 });
  return wrap;
}
