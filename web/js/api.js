/* Thin wrapper over the local JSON API. Every call talks to your own machine. */
const API = {
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
