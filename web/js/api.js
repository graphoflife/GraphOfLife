/*
 * Where the simulation actually runs.
 *
 * Two backends answer the same questions. The server one talks to
 * gol_server.py on your own machine, which is what you get when you clone the
 * project and start it. The browser one runs the identical Python engine
 * inside the page through Pyodide, which is what you get on a static host
 * where there is no server to talk to.
 *
 * Which one is in use is decided once, by asking: if /api/defaults answers,
 * there is a server. That check is the whole switch, and it is why the same
 * interface serves both without knowing which it is driving.
 */

/** Talks to gol_server.py. */
const ServerBackend = {

  async _request(method, path, body, signal) {
    const options = { method, headers: {} };
    if (body !== undefined) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }
    // The whole point of the signal: this ends the request rather than
    // leaving it to finish and be ignored. A long read of a large run is the
    // case that matters — it used to run to the end whatever you did next.
    if (signal) options.signal = signal;

    const response = await fetch(path, options);
    const text = await response.text();

    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (err) {
        throw new Error(`Bad response from server: ${text.slice(0, 200)}`);
      }
    }

    if (!response.ok) {
      throw new Error((payload && payload.error) || `HTTP ${response.status}`);
    }
    return payload;
  },

  defaults()            { return this._request('GET', '/api/defaults'); },
  describe(config)      { return this._request('POST', '/api/describe', { config }); },
  listRuns()            { return this._request('GET', '/api/runs'); },
  getRun(id)            { return this._request('GET', `/api/runs/${encodeURIComponent(id)}`); },
  createRun(name, config) { return this._request('POST', '/api/runs', { name, config }); },
  deleteRun(id)         { return this._request('DELETE', `/api/runs/${encodeURIComponent(id)}`); },
  startRun(id)          { return this._request('POST', `/api/runs/${encodeURIComponent(id)}/start`, {}); },
  stopRun(id)           { return this._request('POST', `/api/runs/${encodeURIComponent(id)}/stop`); },
  copyRun(id, name)     { return this._request('POST', `/api/runs/${encodeURIComponent(id)}/copy`, { name }); },
  renameRun(id, name)   { return this._request('POST', `/api/runs/${encodeURIComponent(id)}/rename`, { name }); },
  getFrame(id, index, opts)   { return this._request('GET', `/api/runs/${encodeURIComponent(id)}/frames/${index}`, undefined, opts && opts.signal); },
  // A contiguous run of frames, cut down to the fields the caller reads.
  // `fields` is the difference between two arrays and the whole topology of
  // a forty-thousand-node world.
  getFrames(id, from, count, fields, sightings, opts) {
    const query = `?from=${from}&count=${count}`
      + (fields && fields.length ? `&fields=${fields.join(',')}` : '')
      + (sightings ? `&sightings=${sightings}` : '');
    return this._request('GET', `/api/runs/${encodeURIComponent(id)}/frames${query}`,
                         undefined, opts && opts.signal);
  },
  // The genotype forest of a window, already reduced to what can be drawn.
  getLineage(id, from, count, phase, opts) {
    return this._request('GET', `/api/runs/${encodeURIComponent(id)}/lineage`
      + `?from=${from}&count=${count}&phase=${phase || 'all'}`, undefined, opts && opts.signal);
  },
  // `points` asks for a coarse pass over the whole run rather than every
  // sample of it, so a chart can be drawn before the full build finishes.
  // `keys` names the statistics the chart plots; the server decides from them
  // how deep to summarise.
  getSeries(id, points, keys, opts) {
    const query = [];
    if (points) query.push(`points=${points}`);
    if (keys) query.push(`keys=${keys.map(encodeURIComponent).join(',')}`);
    return this._request('GET', `/api/runs/${encodeURIComponent(id)}/series`
      + (query.length ? `?${query.join('&')}` : ''), undefined, opts && opts.signal);
  },
  getSeriesProgress(id, opts) { return this._request('GET', `/api/runs/${encodeURIComponent(id)}/series/progress`, undefined, opts && opts.signal); }
};

/** Runs the same Python in a worker, through Pyodide. */
const BrowserBackend = {
  _worker: null,
  _pending: new Map(),
  _nextId: 1,
  progress: { stage: 'idle', detail: '' },
  onProgress: null,

  _ensure() {
    if (this._worker) return this._worker;
    this._worker = new Worker('js/sim-worker.js');

    this._worker.onmessage = (event) => {
      const msg = event.data || {};

      if (msg.type === 'progress') {
        this.progress = msg.progress;
        if (this.onProgress) this.onProgress(msg.progress);
        return;
      }
      const waiting = this._pending.get(msg.id);
      if (!waiting) return;
      this._pending.delete(msg.id);
      msg.ok ? waiting.resolve(msg.result) : waiting.reject(new Error(msg.error));
    };

    this._worker.onerror = (err) => {
      const message = err.message || 'the simulation worker failed to start';
      for (const [, waiting] of this._pending) waiting.reject(new Error(message));
      this._pending.clear();
    };

    return this._worker;
  },

  _send(type, payload = {}, signal = null) {
    const worker = this._ensure();
    const id = this._nextId++;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      // Pyodide runs to completion — there is no interrupting it mid-frame.
      // What can be done is refuse the answer, so a superseded read never
      // paints over what is on screen now. The work is wasted either way;
      // the picture is not. A read already stopped never gets this far:
      // API._call does not send it.
      signal?.addEventListener('abort', () => {
        if (!this._pending.delete(id)) return;
        const stop = new Error('aborted');
        stop.name = 'AbortError';
        reject(stop);
      }, { once: true });
      worker.postMessage({ id, type, ...payload });
    });
  },

  defaults()            { return this._send('defaults'); },
  describe(config)      { return this._send('describe', { config }); },
  listRuns()            { return this._send('list'); },
  getRun(id)            { return this._send('get', { runId: id }); },
  createRun(name, config) { return this._send('create', { name, config }); },
  deleteRun(id)         { return this._send('remove', { runId: id }); },
  startRun(id)          { return this._send('start', { runId: id }); },
  stopRun(id)           { return this._send('stop', { runId: id }); },
  copyRun(id, name)     { return this._send('copy', { runId: id, name }); },
  renameRun(id, name)   { return this._send('rename', { runId: id, name }); },
  // Handed over as text and parsed here, as a server's reply is.
  getFrame(id, index, opts)   {
    return this._send('frame', { runId: id, index }, opts && opts.signal).then(text => JSON.parse(text));
  },
  getFrames(id, from, count, fields, sightings, opts) {
    return this._send('frames', { runId: id, from, count, fields, sightings },
                      opts && opts.signal);
  },
  getLineage(id, from, count, phase, opts) {
    return this._send('lineage', { runId: id, from, count, phase }, opts && opts.signal);
  },
  getSeries(id, points, keys, opts) {
    return this._send('series', { runId: id, points, keys }, opts && opts.signal);
  },
  getSeriesProgress(id, opts) { return this._send('seriesProgress', { runId: id }, opts && opts.signal); },
  storage()             { return this._send('storage'); }
};

const API = {
  backend: null,

  /**
   * Settle on a backend, once.
   *
   * A missing server is the ordinary case on a static host, not a failure, so
   * the probe is quiet about it. Anything else the server says — including an
   * error — still means a server is there.
   */
  async choose() {
    if (this.backend) return this.backend;
    try {
      const response = await fetch('/api/defaults', { method: 'GET' });
      this.backend = response.ok ? ServerBackend : BrowserBackend;
    } catch (err) {
      this.backend = BrowserBackend;
    }
    return this.backend;
  },

  get runsInBrowser() { return this.backend === BrowserBackend; },
  get progress() { return BrowserBackend.progress; },
  set onProgress(fn) { BrowserBackend.onProgress = fn; },

  async _call(method, ...args) {
    const backend = await this.choose();
    // A read already stopped is not sent, and an answer that lands after its
    // read was stopped answers nothing: both end as the abort they are, so no
    // caller has to check for itself after every await. The in-browser
    // backend only hears of an abort that happens while it is waiting.
    const signal = args[args.length - 1]?.signal;
    signal?.throwIfAborted();
    const answer = await backend[method](...args);
    signal?.throwIfAborted();
    return answer;
  },

  // Only the in-browser backend stores anything locally; with a server the
  // question has no meaning and the notice does not ask it.
  storage()               { return BrowserBackend.storage(); }
};

// Every question the backends answer is passed straight through by its name.
// Written out one by one, a new question was an edit in three places, one of
// them fifteen near-identical lines here. The reads take a trailing
// { signal }; the writes do not, since cancelling a create or a rename
// halfway is not a thing anyone wants.
for (const name of Object.keys(ServerBackend)) {
  if (name.startsWith('_') || typeof ServerBackend[name] !== 'function') continue;
  API[name] = function (...args) { return this._call(name, ...args); };
}

/** Human-readable byte size. */
function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatNumber(n) {
  return (n ?? 0).toLocaleString('en-US');
}
