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
  name: 'server',

  async _request(method, path, body) {
    const options = { method, headers: {} };
    if (body !== undefined) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }

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
  listRuns()            { return this._request('GET', '/api/runs'); },
  getRun(id)            { return this._request('GET', `/api/runs/${encodeURIComponent(id)}`); },
  createRun(name, config) { return this._request('POST', '/api/runs', { name, config }); },
  deleteRun(id)         { return this._request('DELETE', `/api/runs/${encodeURIComponent(id)}`); },
  startRun(id)          { return this._request('POST', `/api/runs/${encodeURIComponent(id)}/start`, {}); },
  stopRun(id)           { return this._request('POST', `/api/runs/${encodeURIComponent(id)}/stop`); },
  getFrame(id, index)   { return this._request('GET', `/api/runs/${encodeURIComponent(id)}/frames/${index}`); },
  getSeries(id)         { return this._request('GET', `/api/runs/${encodeURIComponent(id)}/series`); },
  getSeriesProgress(id) { return this._request('GET', `/api/runs/${encodeURIComponent(id)}/series/progress`); }
};

/** Runs the same Python in a worker, through Pyodide. */
const BrowserBackend = {
  name: 'browser',
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
      // A run that ended on its own, or died. Nothing is waiting on these;
      // the interface finds out when it next asks for the list.
      if (msg.type === 'runStopped' || msg.type === 'runError') return;

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

  _send(type, payload = {}) {
    const worker = this._ensure();
    const id = this._nextId++;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      worker.postMessage({ id, type, ...payload });
    });
  },

  defaults()            { return this._send('defaults'); },
  listRuns()            { return this._send('list'); },
  getRun(id)            { return this._send('get', { runId: id }); },
  createRun(name, config) { return this._send('create', { name, config }); },
  deleteRun(id)         { return this._send('remove', { runId: id }); },
  startRun(id)          { return this._send('start', { runId: id }); },
  stopRun(id)           { return this._send('stop', { runId: id }); },
  getFrame(id, index)   { return this._send('frame', { runId: id, index }); },
  getSeries(id)         { return this._send('series', { runId: id }); },
  getSeriesProgress(id) { return this._send('seriesProgress', { runId: id }); }
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
    return backend[method](...args);
  },

  defaults()              { return this._call('defaults'); },
  listRuns()              { return this._call('listRuns'); },
  getRun(id)              { return this._call('getRun', id); },
  createRun(name, config) { return this._call('createRun', name, config); },
  deleteRun(id)           { return this._call('deleteRun', id); },
  startRun(id)            { return this._call('startRun', id); },
  stopRun(id)             { return this._call('stopRun', id); },
  getFrame(id, index)     { return this._call('getFrame', id, index); },
  getSeries(id)           { return this._call('getSeries', id); },
  getSeriesProgress(id)   { return this._call('getSeriesProgress', id); }
};

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

function formatTime(seconds) {
  if (!seconds) return '—';
  return new Date(seconds * 1000).toLocaleString();
}
