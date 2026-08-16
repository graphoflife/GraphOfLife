/*
 * The simulation, running in the page.
 *
 * This is what stands in for gol_server.py when there is no server: a worker
 * that loads Pyodide, imports the very same engine the desktop version runs,
 * and answers the same questions the HTTP API answers. The engine is not
 * reimplemented and not adapted — GraphOfLifeSimple.py is fetched and imported
 * unchanged. A JavaScript translation would have been a second implementation
 * to keep honest, and a simulation this sensitive to arithmetic would not have
 * stayed honest for long.
 *
 * It lives in a worker because a single iteration takes hundreds of
 * milliseconds. On the page's own thread the interface would lock solid for
 * the length of a run.
 *
 * Advancing happens in slices rather than in one loop, so a stop arrives
 * between slices instead of after the run has finished. That is the whole
 * reason gol_browser.step takes a count and returns.
 */

const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/';

// Resolved against this worker's own location, which is js/, so the step up
// is deliberate. Relative to the page would be wrong here and relative to the
// origin would break the moment the site is served from a subpath, which is
// exactly how GitHub Pages serves a project site.
const PY_DIR = new URL('../py/', self.location.href).href;

// Order matters only in that the engine must arrive before anything that
// imports it.
const PY_FILES = [
  'gol_config.py',
  'GraphOfLifeSimple.py',
  'gol_store.py',
  'gol_series.py',
  'gol_browser.py'
];

let pyodide = null;
let ready = null;
let progress = { stage: 'idle', detail: '', done: 0, total: 0 };

/** How many iterations to run before looking at the message queue again. */
const SLICE = 1;

const running = new Set();

function report(stage, detail, done = 0, total = 0) {
  progress = { stage, detail, done, total };
  self.postMessage({ type: 'progress', progress });
}

async function boot() {
  report('loading', 'fetching the Python runtime');
  importScripts(PYODIDE + 'pyodide.js');
  pyodide = await self.loadPyodide({ indexURL: PYODIDE });

  report('loading', 'loading numpy');
  await pyodide.loadPackage(['numpy', 'micropip']);

  // networkx only, without the plotting extras micropip would otherwise pull
  // in behind it — matplotlib and its dependencies are five megabytes this
  // page never draws with.
  report('loading', 'loading networkx');
  await pyodide.runPythonAsync(
    `import micropip\nawait micropip.install('networkx', deps=False)`
  );

  report('loading', 'loading the engine');
  for (const name of PY_FILES) {
    const response = await fetch(PY_DIR + name);
    if (!response.ok) throw new Error(`could not read ${PY_DIR}${name} (${response.status})`);
    pyodide.FS.writeFile(`/home/pyodide/${name}`, await response.text());
  }

  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, '/home/pyodide')
import gol_browser
`);
  report('ready', 'ready');
}

function ensureReady() {
  if (!ready) ready = boot();
  return ready;
}

/**
 * Call into gol_browser and bring the answer back as plain data.
 *
 * Arguments are handed over as a JSON string and parsed on the Python side
 * rather than pasted into the expression. Pasting looked simpler and was
 * wrong: JSON writes true, false and null, none of which Python knows, so a
 * configuration with a switch in it failed as soon as it crossed over. It also
 * means nothing a caller supplies is ever evaluated as code.
 *
 * The answer comes back as JSON too. Converting a frame member by member
 * through Pyodide's own bridge costs several times what encoding and parsing
 * it does, and a frame is a deep tree of lists.
 */
function call(target, args = []) {
  pyodide.globals.set('_call_args', JSON.stringify(args));
  const json = pyodide.runPython(`
import json as _json, math as _math

def _finite(value):
    """
    Statistics are occasionally not numbers — a ratio with nothing underneath
    it, a range that never got a value. JSON has no way to write those, so
    they travel as null and the interface treats them as missing, which is
    what they are.
    """
    if isinstance(value, float) and not _math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value

_json.dumps(_finite(${target}(*_json.loads(_call_args))),
            separators=(',', ':'), allow_nan=False)
`);
  return JSON.parse(json);
}

/** Advance one run by a slice, then hand control back. */
async function pump(runId) {
  if (!running.has(runId)) return;

  let meta;
  try {
    meta = call('gol_browser.RUNS.step', [runId, SLICE]);
  } catch (err) {
    running.delete(runId);
    self.postMessage({ type: 'runError', runId, message: String(err).slice(0, 400) });
    return;
  }

  if (meta.status === 'extinct') {
    running.delete(runId);
    self.postMessage({ type: 'runStopped', runId, meta });
    return;
  }

  // Yielding to the queue between slices is what makes stopping possible.
  setTimeout(() => pump(runId), 0);
}

const handlers = {
  async defaults() {
    return call('gol_browser.RUNS.defaults');
  },
  async list() {
    return { runs: call('gol_browser.RUNS.list') };
  },
  async get({ runId }) {
    return call('gol_browser.RUNS.get', [runId]);
  },
  async create({ name, config }) {
    return call('gol_browser.RUNS.create', [name || '', config || {}]);
  },
  async remove({ runId }) {
    running.delete(runId);
    call('gol_browser.RUNS.delete', [runId]);
    return { ok: true };
  },
  async start({ runId }) {
    call('gol_browser.RUNS.set_status', [runId, 'running']);
    running.add(runId);
    pump(runId);
    return { ok: true };
  },
  async stop({ runId }) {
    running.delete(runId);
    call('gol_browser.RUNS.set_status', [runId, 'stopped']);
    return { ok: true };
  },
  async frame({ runId, index }) {
    return call('gol_browser.RUNS.frame', [runId, Number(index)]);
  },
  async series({ runId }) {
    report('series', 'summarising frames');
    const payload = call('gol_browser.RUNS.series', [runId]);
    report('ready', 'ready');
    return payload;
  },
  async seriesProgress() {
    return { building: progress.stage === 'series', done: progress.done, total: progress.total };
  }
};

self.onmessage = async (event) => {
  const { id, type, ...rest } = event.data || {};
  try {
    await ensureReady();
    const handler = handlers[type];
    if (!handler) throw new Error(`the worker has no handler for "${type}"`);
    self.postMessage({ id, ok: true, result: await handler(rest) });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message ? err.message : err).slice(0, 500) });
  }
};
