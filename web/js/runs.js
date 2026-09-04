/*
 * The Runs tab: create simulations, drive them, and pick one to inspect.
 *
 * The form is generated from whatever the server reports as its defaults, so
 * adding a field to SimConfig only needs a matching input in index.html.
 */
const RunsView = {
  defaults: null,
  runs: [],
  pollTimer: null,

  // Runs we have asked to stop but which have not stopped yet. A worker only
  // notices the request between iterations, so on a slow world the button can
  // sit there for many seconds looking as though the click missed. Holding the
  // request here lets the card say so straight away.
  stopping: new Set(),

  // Which config keys to show on a run card, and how to label them. Kept
  // explicit so cards stay readable rather than dumping the whole dataclass.
  SETTING_LABELS: [
    ['total_tokens', 'tokens'],
    ['n_nodes', 'n'],
    ['k_neighbors', 'k'],
    ['rewire_p', 'shortcuts'],
    ['brain_kind', 'brain'],
    ['hidden_layers', 'hidden'],
    ['message_amount', 'msg'],
    ['random_input_amount', 'noise'],
    ['exchange_messages', 'exchange'],
    ['allow_handover', 'handover'],
    ['allow_revolutions', 'revolutions'],
    ['mutation_probability', 'mut p'],
    ['mutation_noise_std', 'mut std'],
    ['mutation_sparsity', 'mut sparse'],
    ['tokens_created_per_phase', 'created/phase'],
    ['extinction_threshold', 'extinct <'],
    ['checkpoint_every', 'ckpt every'],
    ['export_every', 'record every'],
    ['export_decisions', 'decisions'],
    ['seed', 'seed']
  ],

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

    const stage = (progress && progress.stage) || 'loading';
    if (stage === 'ready') {
      el.innerHTML = '<b>Running in your browser.</b> The same Python engine, '
        + 'through Pyodide \u2014 nothing is sent anywhere and no account is needed. '
        + 'Runs are saved in this browser and survive a reload'
        + '<span id="storageUsage"></span>. '
        + 'Expect a few thousand agents before things get slow; '
        + 'for larger work clone the repository and run it locally.';
      this.showStorage();
      return;
    }
    const detail = (progress && progress.detail) || 'starting';
    el.innerHTML = `<span class="spin"></span>Setting up the engine in your browser \u2014 ${detail}. `
      + 'This happens once, and the download is cached afterwards.';
  },

  /** How much has been stored, once the browser will say. */
  async showStorage() {
    if (!API.runsInBrowser) return;
    try {
      const info = await API.storage();
      const el = document.getElementById('storageUsage');
      if (!el || !info || !info.usage) return;
      el.textContent = ` \u2014 ${formatBytes(info.usage)} used so far`
        + (info.persisted ? '' : ', though the browser may clear it if space runs short');
    } catch (err) {
      /* an unavailable estimate is not worth saying anything about */
    }
  },

  init() {
    this.form = document.getElementById('newRunForm');
    this.listEl = document.getElementById('runList');
    this.errorEl = document.getElementById('createError');

    this.form.addEventListener('submit', e => this.onCreate(e));
    document.getElementById('resetDefaults').addEventListener('click', () => this.applyDefaults());
    document.getElementById('refreshRuns').addEventListener('click', () => this.refresh());

    for (const input of this.form.querySelectorAll('[data-cfg]')) {
      input.addEventListener('input', () => {
        this.syncDependentFields();
        this.updateDerived();
      });
    }
    this.syncDependentFields();
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

  /**
   * Choose a backend and ask it for its defaults — the first time this tab is
   * opened, and not before.
   *
   * This used to run at page load. On a static host there is no server to
   * answer, so it fell through to the in-browser engine and started fetching a
   * Python runtime — several megabytes and a WebAssembly compile — while the
   * front page was trying to lay out five thousand agents and paint them sixty
   * times a second. On a machine running gol_server.py none of that happened,
   * which is why the front page was smooth locally and not when published.
   *
   * A visitor who only reads the front page or the explanation now downloads
   * no Python at all.
   */
  async activate() {
    if (this._activated) return;
    this._activated = true;

    // Settle on a backend before anything is asked of it, so the notice is
    // truthful from the first paint rather than after the first failure.
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

  // ---- form ------------------------------------------------------------

  applyDefaults() {
    if (!this.defaults) return;
    const cfg = this.defaults.config;

    for (const input of this.form.querySelectorAll('[data-cfg]')) {
      const value = cfg[input.dataset.cfg];
      if (input.type === 'checkbox') input.checked = Boolean(value);
      else if (Array.isArray(value)) input.value = value.join(', ');
      else input.value = (value === null || value === undefined) ? '' : value;
    }
    this.errorEl.textContent = '';
    this.syncDependentFields();
    this.updateDerived();
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

  updateDerived() {
    const el = document.getElementById('derivedInfo');
    if (!el) return;
    const cfg = this.readConfig();

    const messages = cfg.message_amount ?? 5;
    const noise = cfg.random_input_amount ?? 5;
    const inputs = 29 + 4 * messages + noise;
    // Revolutions add a fraction pair; handover adds a yes/no pair plus its
    // mode pair. Both are absent from the brain when switched off.
    const outputs = 9 + (cfg.allow_revolutions ? 2 : 0)
                      + (cfg.allow_handover ? 4 : 0)
                      + messages;

    // A binary brain spreads every input across a ladder of bits, so its first
    // layer is far wider even though each weight costs a fraction as much.
    const binary = cfg.brain_kind === 'binary';
    const bits = binary ? (cfg.brain_bits || 16) : 1;
    const bytesPerWeight = binary ? 1 : (cfg.brain_kind === 'float16' ? 2 : 8);

    const n = cfg.n_nodes > 0 ? cfg.n_nodes : Math.floor((cfg.total_tokens || 0) / 100);
    const k = cfg.k_neighbors > 0 ? cfg.k_neighbors : Math.max(Math.floor(n / 100), 5);

    const layers = (cfg.hidden_layers && cfg.hidden_layers.length) ? cfg.hidden_layers : [50, 45, 40, 35, 30];
    const sizes = [inputs * bits, ...layers, outputs];
    let params = 0;
    for (let i = 0; i < sizes.length - 1; i++) params += sizes[i] * sizes[i + 1] + sizes[i + 1];

    el.textContent =
      `Brain ${sizes.join('→')} · ${formatNumber(params)} params · ` +
      `seed graph n=${formatNumber(n)}, k=${k} · ` +
      `checkpoint ≈ ${formatBytes(params * bytesPerWeight * Math.max(1, n))}`;
  },

  async onCreate(event) {
    event.preventDefault();
    this.errorEl.textContent = '';
    try {
      const name = document.getElementById('runName').value;
      const meta = await API.createRun(name, this.readConfig());
      document.getElementById('runName').value = '';
      await this.refresh();
      this.errorEl.textContent = `Created "${meta.name}".`;
      this.errorEl.classList.add('ok');
      setTimeout(() => this.errorEl.classList.remove('ok'), 2500);
    } catch (err) {
      this.errorEl.textContent = err.message;
    }
  },

  // ---- list ------------------------------------------------------------

  async refresh() {
    try {
      const data = await API.listRuns();
      this.runs = data.runs || [];
      this.failedPolls = 0;
      this.render();
      Viewer.syncRunPicker(this.runs);
    } catch (err) {
      // A poll that fails is not a poll that should stop. The server going
      // away for a moment — a restart, a slow frame write — used to end the
      // polling for good, so a run carried on in the background while its card
      // sat frozen until the page was reloaded by hand.
      this.failedPolls = (this.failedPolls || 0) + 1;
      const p = document.createElement('p');
      p.className = 'empty';
      p.textContent = `Could not load runs: ${err.message}`;
      this.listEl.replaceChildren(p);
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
    const failed = this.failedPolls || 0;
    if (!failed && !this.runs.some(r => r.running)) return;
    const delay = failed ? Math.min(15000, 1500 * 2 ** (failed - 1)) : 1500;
    this.pollTimer = setTimeout(() => this.refresh(), delay);
  },

  render() {
    if (!this.runs.length) {
      this.listEl.innerHTML = '<p class="empty">No simulations yet. Create one on the left.</p>';
      return;
    }
    this.listEl.innerHTML = '';
    for (const run of this.runs) this.listEl.appendChild(this.card(run));
  },

  settingsHtml(cfg) {
    const parts = [];
    for (const [key, label] of this.SETTING_LABELS) {
      let value = cfg[key];
      if (value === null || value === undefined || value === '') value = 'auto';
      else if (Array.isArray(value)) value = value.join('/');
      else if (typeof value === 'boolean') value = value ? 'yes' : 'no';
      else if (typeof value === 'number') value = formatNumber(value);
      parts.push(`<span><i>${label}</i> ${escapeHtml(String(value))}</span>`);
    }
    return `<div class="run-settings">${parts.join('')}</div>`;
  },

  card(run) {
    const el = document.createElement('div');
    el.className = 'run-card';
    const cfg = run.config || {};

    // One button that flips between starting and stopping, rather than two
    // where only ever one of them applies.
    const running = Boolean(run.running);
    // The run has stopped, so forget the request; anything still in the set is
    // a request the worker has yet to act on.
    if (!running) this.stopping.delete(run.id);
    const stopping = running && this.stopping.has(run.id);
    const actionLabel = stopping ? 'Stopping' : running ? 'Stop'
                                 : (run.has_checkpoint ? 'Resume' : 'Start');

    el.innerHTML = `
      <div class="run-head">
        <div>
          <h3>${escapeHtml(run.name)}</h3>
          ${run.name !== run.id ? `<span class="run-id">${escapeHtml(run.id)}</span>` : ''}
        </div>
        <span class="status status-${stopping ? 'stopping' : run.status}">${
          stopping ? '<span class="spin"></span>stopping' : escapeHtml(run.status)}</span>
      </div>
      <div class="run-meta">
        <span><b>${formatNumber(run.iteration)}</b> iter</span>
        <span><b>${formatNumber(run.frame_count)}</b> frames</span>
        <span>${formatBytes(run.size_bytes)}</span>
        ${(run.checkpoint_iteration !== null && run.checkpoint_iteration !== undefined)
          ? `<span>resume @ ${formatNumber(run.checkpoint_iteration)}</span>` : ''}
        <span>${formatTime(run.created_at)}</span>
      </div>
      ${this.settingsHtml(cfg)}
      ${run.error ? `<pre class="run-error">${escapeHtml(run.error)}</pre>` : ''}
      <div class="run-actions">
        <button class="${running ? 'warn' : 'primary'}" data-act="toggle"${
          stopping ? ' disabled' : ''}>${stopping
            ? '<span class="spin"></span>Stopping\u2026' : actionLabel}</button>
        <button data-act="open">Inspect</button>
        <button class="danger" data-act="delete">Delete</button>
      </div>
    `;

    el.querySelector('[data-act="toggle"]').addEventListener('click', async () => {
      try {
        if (running) {
          // Mark it before the request goes out, and redraw immediately: the
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
    });

    el.querySelector('[data-act="open"]').addEventListener('click', () => {
      App.showView('viewer');
      Viewer.load(run.id);
    });

    el.querySelector('[data-act="delete"]').addEventListener('click', async () => {
      if (!confirm(`Delete "${run.name}" and all of its recorded data? This cannot be undone.`)) return;
      try { await API.deleteRun(run.id); await this.refresh(); }
      catch (err) { alert(err.message); }
    });

    return el;
  }
};

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text ?? '';
  return div.innerHTML;
}
