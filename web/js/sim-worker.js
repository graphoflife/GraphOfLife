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

importScripts('runstore.js');

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

/*
 * Runs are metadata here and frames are in IndexedDB; Python only holds the
 * live worlds. That split is what lets a run survive a reload: reopening the
 * page reads the run list and its frames straight back out of storage, with
 * the interpreter uninvolved until something needs to be advanced again.
 */

const running = new Set();

/** Metadata as the interface expects it. */
function meta(run) {
  return {
    id: run.id,
    name: run.name,
    created_at: run.created_at,
    status: run.status,
    iteration: run.iteration,
    frame_count: run.frame_count,
    checkpoint_iteration: run.checkpoint_iteration,
    has_checkpoint: run.checkpoint_iteration !== null,
    running: running.has(run.id),
    error: run.error || null,
    config: run.config,
    size_bytes: run.size_bytes || 0
  };
}

async function loadRun(runId) {
  const run = await RunStore.getRun(runId);
  if (!run) throw new Error(`no run called ${runId}`);
  return run;
}

/**
 * Make sure Python has a world for this run.
 *
 * After a reload there is metadata and there are frames, but no world — the
 * interpreter started empty. A checkpoint is what bridges that, and without
 * one the run can still be read, just not continued.
 */
async function ensureWorld(run) {
  if (call('gol_browser.WORLDS.has', [run.id])) return;

  const bytes = await RunStore.getCheckpoint(run.id);
  if (!bytes) {
    throw new Error(
      'this run has no resume point, so it can be inspected but not continued'
    );
  }
  const path = `/home/pyodide/${run.id}.npz`;
  pyodide.FS.writeFile(path, new Uint8Array(bytes));
  call('gol_browser.WORLDS.restore', [run.id, run.config, path]);
  try { pyodide.FS.unlink(path); } catch (err) { /* already gone */ }
}

async function saveCheckpoint(run) {
  const path = `/home/pyodide/${run.id}.npz`;
  call('gol_browser.WORLDS.checkpoint', [run.id, path]);
  const bytes = pyodide.FS.readFile(path);
  try { pyodide.FS.unlink(path); } catch (err) { /* already gone */ }
  await RunStore.putCheckpoint(run.id, bytes.buffer);
  run.checkpoint_iteration = run.iteration;
}

/** Advance one run by a slice, store what it produced, then hand back. */
async function pump(runId) {
  if (!running.has(runId)) return;

  let run;
  try {
    run = await loadRun(runId);
    const slice = call('gol_browser.WORLDS.step', [runId, SLICE]);

    if (slice.frames.length) {
      await RunStore.putFrames(runId, run.frame_count, slice.frames);
      run.frame_count += slice.frames.length;
      for (const frame of slice.frames) {
        // Rough, and deliberately so: it is for telling the reader how much of
        // their disk a run is using, not for accounting.
        run.size_bytes += 120 * frame.ids.length + 40 * frame.edges.length;
      }
    }
    run.iteration = slice.iteration;

    const every = run.config.checkpoint_every || 0;
    if (every && run.iteration % every === 0) await saveCheckpoint(run);

    if (slice.extinct) {
      running.delete(runId);
      run.status = 'extinct';
      if (every) await saveCheckpoint(run);
    }
    await RunStore.putRun(run);
  } catch (err) {
    running.delete(runId);
    if (run) {
      run.status = 'error';
      run.error = String(err && err.message ? err.message : err).slice(0, 400);
      await RunStore.putRun(run).catch(() => {});
    }
    self.postMessage({ type: 'runError', runId, message: String(err).slice(0, 400) });
    return;
  }

  if (!running.has(runId)) return;
  // Yielding to the queue between slices is what makes stopping possible.
  setTimeout(() => pump(runId), 0);
}

let counter = 0;

const handlers = {
  async defaults() {
    return call('gol_browser.WORLDS.defaults');
  },

  async list() {
    const runs = await RunStore.listRuns();
    return { runs: runs.map(meta) };
  },

  async get({ runId }) {
    return meta(await loadRun(runId));
  },

  async create({ name, config }) {
    const stamp = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const day = `${String(stamp.getFullYear()).slice(2)}_${pad(stamp.getMonth() + 1)}_${pad(stamp.getDate())}`;

    // Distinct within the day even across reloads, since the counter starts
    // over but the stored runs do not.
    const existing = await RunStore.listRuns();
    const taken = new Set(existing.map(r => r.id));
    let id;
    do { id = `GOL_${day}_n${String(++counter).padStart(3, '0')}`; } while (taken.has(id));

    const prepared = call('gol_browser.WORLDS.create', [id, config || {}]);
    const run = {
      id,
      name: (name || '').trim() || id,
      created_at: Date.now() / 1000,
      status: 'idle',
      iteration: 0,
      frame_count: 0,
      checkpoint_iteration: null,
      error: null,
      config: prepared.config,
      size_bytes: 0
    };
    await RunStore.putRun(run);
    return meta(run);
  },

  async remove({ runId }) {
    running.delete(runId);
    call('gol_browser.WORLDS.drop', [runId]);
    await RunStore.deleteRun(runId);
    return { ok: true };
  },

  async start({ runId }) {
    const run = await loadRun(runId);
    await ensureWorld(run);
    run.status = 'running';
    run.error = null;
    await RunStore.putRun(run);
    running.add(runId);
    pump(runId);
    return { ok: true };
  },

  async stop({ runId }) {
    running.delete(runId);
    const run = await loadRun(runId);
    run.status = 'stopped';
    // A stop is the likeliest moment for someone to close the tab, so this is
    // where it is worth paying for a resume point.
    if (run.config.checkpoint_every) {
      try { await saveCheckpoint(run); } catch (err) { /* nothing to save yet */ }
    }
    await RunStore.putRun(run);
    return { ok: true };
  },

  async frame({ runId, index }) {
    return RunStore.getFrame(runId, Number(index));
  },

  async series({ runId }) {
    const run = await loadRun(runId);
    const totalIterations = Math.max(0, Math.floor(run.frame_count / 2));
    const stride = call('gol_browser.WORLDS.sample_stride', [totalIterations]);

    report('series', 'reading frames', 0, 0);
    const frames = await RunStore.getFramesStrided(runId, stride);

    report('series', 'summarising', 0, frames.length);
    const rows = frames.length ? call('gol_browser.WORLDS.stats', [frames]) : [];
    report('ready', 'ready');

    if (!rows.length) {
      return { count: 0, keys: [], series: {}, stride, sampled: false,
               nodeCountKeys: call('gol_browser.WORLDS.node_count_keys') };
    }
    const keys = Object.keys(rows[0]);
    const series = {};
    for (const key of keys) series[key] = rows.map(row => row[key]);
    return {
      count: rows.length, keys, series, stride, sampled: stride > 1,
      totalIterations,
      nodeCountKeys: call('gol_browser.WORLDS.node_count_keys')
    };
  },

  async seriesProgress() {
    return { building: progress.stage === 'series', done: progress.done, total: progress.total };
  },

  async storage() {
    const usage = await RunStore.usage();
    return { persisted: await RunStore.requestPersistence(), ...(usage || {}) };
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
