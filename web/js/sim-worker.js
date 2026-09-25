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

// All written into the interpreter's files before anything is imported, so
// the order does not matter.
const PY_FILES = [
  'gol_config.py',
  'GraphOfLifeSimple.py',
  'gol_store.py',
  'gol_spectral.py',
  'gol_lightning.py',
  'gol_series.py',
  'gol_lineage.py',
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
  // The engine's files, asked for all at once and while the runtime and its
  // packages download, rather than one after another once they are in.
  const sources = Promise.all(PY_FILES.map(async name => {
    const response = await fetch(PY_DIR + name);
    if (!response.ok) throw new Error(`could not read ${PY_DIR}${name} (${response.status})`);
    return response.text();
  }));
  // Awaited below; until then a failure must not be reported as unhandled.
  sources.catch(() => {});

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
  (await sources).forEach((text, i) => pyodide.FS.writeFile(`/home/pyodide/${PY_FILES[i]}`, text));

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
 * Keep only the named fields of a frame, `a.b` reaching one level in.
 *
 * gol_server._project, for the same callers. Taking each name literally made
 * `decisions.allocations` come back as a key of that name holding nothing, so
 * on the static site Flow modules said the run had recorded no decisions and
 * the edge flow metric read zero everywhere.
 */
function project(frame, fields) {
  const out = {};
  for (const name of fields) {
    const dot = name.indexOf('.');
    if (dot < 0) { out[name] = frame[name]; continue; }
    const head = name.slice(0, dot);
    const branch = frame[head];
    if (branch && typeof branch === 'object' && !Array.isArray(branch)) {
      (out[head] ||= {})[name.slice(dot + 1)] = branch[name.slice(dot + 1)];
    }
  }
  return out;
}

let lineageFieldsRead = null;
/** What gol_lineage.forest reads of a frame, asked of Python once. */
function lineageFields() {
  return (lineageFieldsRead ||= call('gol_browser.WORLDS.lineage_fields'));
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
 * The answer comes back as JSON too; gol_browser.answer says why.
 */
function call(target, args = []) {
  return callWritten(target, JSON.stringify(args));
}

/** The same, with the arguments already written as JSON. */
function callWritten(target, argsJson) {
  pyodide.globals.set('_call_args', argsJson);
  return JSON.parse(pyodide.runPython(`gol_browser.answer(${target}, _call_args)`));
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
  // Whether it is going is a live fact, and the stored status only what was
  // last written. A run that was advancing when the tab closed is still marked
  // as running, and nothing is advancing it any more: the worker it belonged
  // to died with the page. Where the two disagree the live one wins, both
  // ways, the way the server settles it. Only the answer is corrected — what
  // is stored is the run's own history and stays as it was written.
  const live = running.has(run.id);
  const status = live ? 'running' : (run.status === 'running' ? 'interrupted' : run.status);

  return {
    id: run.id,
    name: run.name,
    created_at: run.created_at,
    status,
    iteration: run.iteration,
    frame_count: run.frame_count,
    checkpoint_iteration: run.checkpoint_iteration,
    has_checkpoint: run.checkpoint_iteration !== null,
    running: live,
    error: run.error || null,
    config: run.config,
    strain: run.strain || null,
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
  const restored = call('gol_browser.WORLDS.restore', [run.id, run.config, path]);
  try { pyodide.FS.unlink(path); } catch (err) { /* already gone */ }

  // Resuming drops the future the checkpoint never lived through, as the
  // server does. A tab closed after the last checkpoint left frames past it,
  // and new ones written after those put every later frame at the wrong
  // iteration: a frame's index is what says which iteration it is.
  if (run.frame_count > restored.frames) await RunStore.deleteFramesFrom(run.id, restored.frames);
  run.frame_count = restored.frames;
  run.iteration = restored.iteration;
  await RunStore.putRun(run);
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
      // Each frame arrives as the JSON the engine wrote, and is stored from it.
      // What they cost is reported by the store that wrote them; it used to be
      // guessed from the agent and edge counts.
      run.size_bytes += await RunStore.putFrames(runId, run.frame_count, slice.frames);
      run.frame_count += slice.frames.length;
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
    return;
  }

  if (!running.has(runId)) return;
  // Yielding to the queue between slices is what makes stopping possible.
  setTimeout(() => pump(runId), 0);
}

let counter = 0;

/**
 * The next free run id.
 *
 * Distinct within the day even across reloads, since the counter starts over
 * but the stored runs do not.
 */
async function nextRunId() {
  const stamp = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  const day = `${String(stamp.getFullYear()).slice(2)}_${pad(stamp.getMonth() + 1)}_${pad(stamp.getDate())}`;
  const taken = new Set((await RunStore.listRuns()).map(r => r.id));
  let id;
  do { id = `GOL_${day}_n${String(++counter).padStart(3, '0')}`; } while (taken.has(id));
  return id;
}

const handlers = {
  async defaults() {
    return call('gol_browser.WORLDS.defaults');
  },

  async describe({ config }) {
    return call('gol_browser.WORLDS.describe', [config]);
  },

  async list() {
    const runs = await RunStore.listRuns();
    return { runs: runs.map(meta) };
  },

  async get({ runId }) {
    return meta(await loadRun(runId));
  },

  async create({ name, config }) {
    const id = await nextRunId();
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
      // Which algorithm this is, handed back by gol_browser.create. Stored on
      // the run rather than derived on demand, so a result can never be found
      // without it. See research/strains.md.
      strain: prepared.strain || null,
      size_bytes: 0
    };
    await RunStore.putRun(run);
    return meta(run);
  },

  /**
   * Duplicate a run whole: its metadata, every frame, and its checkpoint.
   *
   * A fork rather than a backup — the copy resumes from exactly where the
   * original is and goes its own way. It is never marked as running, because
   * nothing is advancing it.
   */
  async rename({ runId, name }) {
    const run = await loadRun(runId);
    run.name = String(name || '').trim() || run.name;
    await RunStore.putRun(run);
    return meta(run);
  },

  async copy({ runId, name }) {
    const source = await loadRun(runId);
    const id = await nextRunId();

    const copied = {
      ...source,
      id,
      name: (name || '').trim() || `${source.name || runId} (copy)`,
      created_at: Date.now() / 1000,
      status: 'idle',
      error: null
    };
    await RunStore.putRun(copied);
    await RunStore.copyFrames(runId, id);

    const checkpoint = await RunStore.getCheckpoint(runId);
    if (checkpoint) await RunStore.putCheckpoint(id, checkpoint);

    return meta(copied);
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
    return RunStore.getFrameText(runId, Number(index));
  },

  /**
   * The genotype forest of a window, aggregated in here.
   *
   * Same reason as the server's: a real window holds more than a million
   * genotypes and the page can draw a couple of thousand. Doing it here also
   * keeps the work off the main thread, which is what was freezing the tab.
   * Each frame is cut to what the forest reads before the window crosses into
   * Python, which it does as one JSON string.
   */
  async lineage({ runId, from, count, phase }) {
    const fields = lineageFields();
    const read = [];
    for (let at = from; at < from + count; at += 16) {
      const batch = await RunStore.getFrameRange(runId, at, Math.min(16, from + count - at));
      if (!batch.length) break;
      for (const frame of batch) read.push(project(frame, fields));
    }
    return call('gol_browser.WORLDS.lineage', [read, phase || 'all']);
  },

  /**
   * A contiguous run of frames, cut down to the fields the caller reads.
   *
   * The projection matters here too, even with no network in the way: every
   * frame is copied across the worker boundary, and a window of two hundred
   * whole frames of a large world is tens of megabytes of structured clone
   * so that two arrays can be read out of each.
   */
  async frames({ runId, from, count, fields, sightings }) {
    const frames = await RunStore.getFrameRange(runId, Number(from), Number(count));
    // Stopping once enough agents have been seen, as the server does: a
    // window measured in iterations is very different work in a world of sixty
    // and one of forty thousand.
    const kept = [];
    let seen = 0;
    for (const frame of frames) {
      kept.push(fields && fields.length ? project(frame, fields) : frame);
      seen += (frame.ids || frame.brain_ids || []).length;
      if (sightings && seen >= sightings) break;
    }
    return { frames: kept };
  },

  async series({ runId, points, keys }) {
    const run = await loadRun(runId);

    // Python keeps the run's history and says what this request still needs;
    // the frames are in IndexedDB, out of its reach, so they are read here and
    // handed over. The same gol_series.History the server keeps, so a run
    // summarised in the browser is summarised by the same rules, and each step
    // of a climb summarises only what the one before did not.
    const plan = call('gol_browser.WORLDS.series_plan',
                      [runId, run.frame_count, points ?? null, keys ?? null]);
    report('series', 'reading frames', 0, 0);
    const read = await RunStore.getIterations(runId, plan.iterations);

    // Written around each frame's own stored text rather than parsed here
    // only to be written out again on the way over.
    report('series', 'summarising', 0, read.length);
    const frames = read.map(({ index, text }) => `{"index":${index},"frame":${text}}`).join(',');
    const reply = callWritten('gol_browser.WORLDS.series_absorb',
                              `[${JSON.stringify(runId)},[${frames}],${plan.heavy}]`);
    report('ready', 'ready');
    return reply;
  },

  async seriesProgress() {
    return { building: progress.stage === 'series', done: progress.done, total: progress.total };
  },

  async storage() {
    const usage = await RunStore.usage();
    return { persisted: await RunStore.requestPersistence(), ...(usage || {}) };
  }
};

// What can be answered out of IndexedDB alone, without the Python runtime.
//
// Everything used to wait for boot, which meant that reading a list of runs —
// pure storage, no engine — downloaded and started Pyodide first. On a static
// host that is tens of megabytes to fill a dropdown, and it is why the Viewer
// could not afford to populate its run picker on the way in.
//
// An allow-list rather than a deny-list: a handler that needs the engine and
// is left off this by mistake still works, where one that does not need it and
// is wrongly added would fail only in the browser, only on a cold start.
const NO_ENGINE = new Set(['list', 'get', 'frame', 'frames', 'rename',
                           'seriesProgress', 'storage']);

self.onmessage = async (event) => {
  const { id, type, ...rest } = event.data || {};
  try {
    if (!NO_ENGINE.has(type)) await ensureReady();
    const handler = handlers[type];
    if (!handler) throw new Error(`the worker has no handler for "${type}"`);
    self.postMessage({ id, ok: true, result: await handler(rest) });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message ? err.message : err).slice(0, 500) });
  }
};
