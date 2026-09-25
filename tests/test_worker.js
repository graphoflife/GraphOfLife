/*
 * The in-browser backend's bookkeeping, against a pretend engine and store.
 *
 *     node tests/test_worker.js
 *
 * sim-worker.js runs the engine in Pyodide and keeps runs in IndexedDB,
 * neither of which exists here. Both are stood in for: a world that only
 * counts its iterations, and a store that only keeps records, frames and
 * checkpoints, copying what it is given as IndexedDB does. That is enough to
 * test what the worker itself decides — when a slice's frames are stored,
 * when a stop takes effect, and what a checkpoint says it holds.
 *
 * The store can be asked to do something in the middle of storing a slice's
 * frames, which is how the interleavings below are made to happen every time
 * rather than when the timing happens to allow. Stopping a run at exactly
 * that moment used to leave its checkpoint ahead of its record, the run
 * written back as running, and, if the tab closed then, a hole in its frames.
 *
 * Same harness as the other tests: collect every test_ function, run them
 * all, print a dot or an F, and report at the end.
 */
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const SOURCE = fs.readFileSync(path.join(root, 'web', 'js', 'sim-worker.js'), 'utf8');

const tick = (ms = 1) => new Promise(resolve => setTimeout(resolve, ms));

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

// ---------------------------------------------------------------------------

/**
 * Records, frames and checkpoints in memory. Each call takes a tick, as a
 * transaction does, and what is stored is a copy, so the worker cannot change
 * a record by holding on to it. `during`, when set, runs once in the middle
 * of the next frame write, after the slice has stepped and before its frames
 * are stored.
 */
function pretendStore() {
  const runs = new Map(), frames = new Map(), checkpoints = new Map();
  const copy = value => JSON.parse(JSON.stringify(value));
  const store = {
    runs, frames, checkpoints, during: null,
    async getRun(id) { await tick(); return runs.has(id) ? copy(runs.get(id)) : null; },
    async putRun(run) { await tick(); runs.set(run.id, copy(run)); },
    async listRuns() { await tick(); return [...runs.values()].map(copy); },
    async deleteRun(id) { await tick(); runs.delete(id); frames.delete(id); checkpoints.delete(id); },
    async putFrames(id, start, texts) {
      await tick();
      if (store.during) {
        const now = store.during;
        store.during = null;
        await now();
      }
      const list = frames.get(id) || [];
      texts.forEach((text, k) => { list[start + k] = text; });
      frames.set(id, list);
      return texts.length;
    },
    async deleteFramesFrom(id, from) { await tick(); (frames.get(id) || []).length = from; },
    async getCheckpoint(id) { await tick(); return checkpoints.get(id) || null; },
    async putCheckpoint(id, bytes) { await tick(); checkpoints.set(id, bytes.slice(0)); },
    async copyFrames(from, to) { await tick(); frames.set(to, [...(frames.get(from) || [])]); }
  };
  return store;
}

/** The iteration a stored checkpoint holds. */
const heldBy = bytes => JSON.parse(Buffer.from(bytes).toString()).iteration;

/**
 * gol_browser as the worker sees it through Pyodide: worlds that count their
 * iterations, write two frames each, and checkpoint the count.
 */
function pretendPython() {
  const worlds = new Map();
  const files = new Map();
  let args = '[]';
  const WORLDS = {
    defaults: () => ({ config: {} }),
    has: id => worlds.has(id),
    create: (id, config) => { worlds.set(id, { iteration: 0 }); return { config, strain: 'test' }; },
    restore: (id, config, file) => {
      const { iteration } = JSON.parse(files.get(file));
      worlds.set(id, { iteration });
      return { iteration, frames: 2 * iteration };
    },
    step: (id, count) => {
      const world = worlds.get(id);
      if (!world) throw new Error(`no world loaded for ${id}`);
      const frames = [];
      for (let k = 0; k < count; k++) {
        frames.push(JSON.stringify({ iteration: world.iteration, phase: 1 }),
                    JSON.stringify({ iteration: world.iteration, phase: 2 }));
        world.iteration += 1;
      }
      return { iteration: world.iteration, extinct: false, frames };
    },
    checkpoint: (id, file) => {
      const world = worlds.get(id);
      if (!world) throw new Error(`no world loaded for ${id}`);
      files.set(file, JSON.stringify({ iteration: world.iteration }));
      return { iteration: world.iteration, bytes: 1 };
    },
    drop: id => { worlds.delete(id); }
  };
  return {
    worlds,
    globals: { set: (name, value) => { args = value; } },
    runPython(code) {
      const [, method] = code.match(/WORLDS\.(\w+), _call_args/);
      return JSON.stringify(WORLDS[method](...JSON.parse(args)) ?? null);
    },
    async runPythonAsync() {},
    async loadPackage() {},
    FS: {
      writeFile: (file, data) => files.set(file, typeof data === 'string' ? data : Buffer.from(data).toString()),
      readFile: file => new Uint8Array(Buffer.from(files.get(file))),
      unlink: file => files.delete(file)
    }
  };
}

/**
 * sim-worker.js itself, over `store`, with a pretend Python of its own: a new
 * one is a page that was reloaded, which holds no worlds. `send` posts a
 * message and resolves with the worker's answer.
 */
function worker(store) {
  const python = pretendPython();
  const waiting = new Map();
  const self = {
    location: { href: 'http://localhost/js/sim-worker.js' },
    loadPyodide: async () => python,
    postMessage(message) {
      const reply = waiting.get(message.id);
      if (!reply) return;
      waiting.delete(message.id);
      reply(message);
    }
  };
  const fetch = async () => ({ ok: true, text: async () => '' });
  new Function('importScripts', 'self', 'RunStore', 'fetch', SOURCE)(() => {}, self, store, fetch);

  let next = 0;
  const send = (type, payload = {}) => new Promise((resolve, reject) => {
    const id = ++next;
    waiting.set(id, message => (message.ok ? resolve(message.result) : reject(new Error(message.error))));
    self.onmessage({ data: { id, type, ...payload } });
  });
  return { send, python };
}

/** Start a run, and wait until it has stored at least `frames` frames. */
async function runUntil(store, send, runId, frames) {
  await send('start', { runId });
  while ((store.frames.get(runId) || []).length < frames) await tick(2);
}

/**
 * Do `act` in the middle of the next slice, after the world has stepped and
 * before the slice has stored its frames, then wait until whatever that slice
 * had left to do is done too. The interleaving happens every time, rather
 * than when the timing allows.
 */
async function midSlice(store, act) {
  let acted = null;
  store.during = async () => { acted = act(); await tick(20); };
  while (!acted) await tick(1);
  const answer = await acted;
  await tick(60);
  return answer;
}

/**
 * Everything a stored run should agree on once it has stopped: every frame
 * where its index says, one per phase of each iteration up to the one on
 * record, and a checkpoint of that same iteration.
 */
function assertWhole(store, runId, label) {
  const record = store.runs.get(runId);
  const frames = store.frames.get(runId) || [];
  assert(record.status === 'stopped', `${label}: the run was left ${record.status}, not stopped`);
  assert(record.frame_count === 2 * record.iteration && frames.length === record.frame_count,
    `${label}: ${frames.length} frames stored and ${record.frame_count} on record `
    + `for ${record.iteration} iterations`);
  // By index, not forEach, which would pass over a missing frame in silence.
  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i] && JSON.parse(frames[i]);
    assert(frame && frame.iteration === Math.floor(i / 2) && frame.phase === 1 + (i % 2),
      `${label}: frame ${i} is ${frames[i]}`);
  }
  const held = heldBy(store.checkpoints.get(runId));
  assert(record.checkpoint_iteration === record.iteration && held === record.iteration,
    `${label}: the checkpoint holds iteration ${held} and the record says `
    + `${record.checkpoint_iteration}, at iteration ${record.iteration}`);
}

// ---------------------------------------------------------------------------

async function test_a_stop_in_the_middle_of_a_slice_waits_for_it() {
  // A stop's message waits out the step and is taken at the first await
  // after it, which is the slice storing its frames: the world has moved on
  // and the record has not. It used to checkpoint that world under the
  // record's iteration, and the slice then wrote the run back as running.
  const store = pretendStore();
  const { send, python } = worker(store);
  const run = await send('create', { name: 'r', config: { checkpoint_every: 5 } });
  await runUntil(store, send, run.id, 6);
  await midSlice(store, () => send('stop', { runId: run.id }));
  assertWhole(store, run.id, 'stopped mid-slice');
  assert(python.worlds.get(run.id).iteration === store.runs.get(run.id).iteration,
    'the world went on past the iteration the stop recorded');
}

async function test_a_run_stopped_as_its_tab_closes_resumes_without_a_hole() {
  // The stop is taken mid-slice and the tab closes before that slice stores
  // its frames. A checkpoint of the world the slice had stepped, under the
  // record's older iteration, resumed from an iteration whose frames were
  // never written: a hole where they should have been.
  const store = pretendStore();
  let { send } = worker(store);
  const run = await send('create', { name: 'r', config: { checkpoint_every: 2 } });
  await runUntil(store, send, run.id, 8);
  store.during = () => {
    send('stop', { runId: run.id });
    return new Promise(() => {});           // the tab closes: this slice never ends
  };
  while (store.during) await tick(1);
  await tick(60);                           // whatever the stop writes, it has written

  ({ send } = worker(store));               // the page, opened again
  const at = store.runs.get(run.id).frame_count;
  await runUntil(store, send, run.id, at + 6);
  await send('stop', { runId: run.id });
  assertWhole(store, run.id, 'resumed after the tab closed');
}

async function test_a_stop_and_a_start_at_once_leave_one_run() {
  // Between two slices a pump only asked whether its run was running, which
  // after a stop and an immediate start it was again: two pumps went on side
  // by side.
  const store = pretendStore();
  const { send } = worker(store);
  const run = await send('create', { name: 'r', config: { checkpoint_every: 5 } });
  await runUntil(store, send, run.id, 4);
  await Promise.all([send('stop', { runId: run.id }), send('start', { runId: run.id })]);
  const at = store.frames.get(run.id).length;
  while (store.frames.get(run.id).length < at + 6) await tick(2);
  await send('stop', { runId: run.id });
  assertWhole(store, run.id, 'stopped and started at once');
}

async function test_a_rename_during_a_slice_is_kept() {
  const store = pretendStore();
  const { send } = worker(store);
  const run = await send('create', { name: 'before', config: {} });
  await runUntil(store, send, run.id, 4);
  await midSlice(store, () => send('rename', { runId: run.id, name: 'after' }));
  await send('stop', { runId: run.id });
  assert(store.runs.get(run.id).name === 'after',
    `a slice in flight wrote the old name back: "${store.runs.get(run.id).name}"`);
}

async function test_a_run_deleted_during_a_slice_stays_deleted() {
  const store = pretendStore();
  const { send } = worker(store);
  const run = await send('create', { name: 'r', config: {} });
  await runUntil(store, send, run.id, 4);
  await midSlice(store, () => send('remove', { runId: run.id }));
  assert(!store.runs.has(run.id), 'a slice in flight wrote a deleted run back into existence');
  assert(!store.frames.has(run.id), 'a slice in flight stored frames for a deleted run');
}

// ---------------------------------------------------------------------------

const tests = Object.entries({
  test_a_stop_in_the_middle_of_a_slice_waits_for_it,
  test_a_run_stopped_as_its_tab_closes_resumes_without_a_hole,
  test_a_stop_and_a_start_at_once_leave_one_run,
  test_a_rename_during_a_slice_is_kept,
  test_a_run_deleted_during_a_slice_stays_deleted
}).sort(([a], [b]) => a.localeCompare(b));

// A test that is written and never listed here is worse than no test: it reads
// as coverage and runs never.
const written = (fs.readFileSync(__filename, 'utf8')
  .match(/^(async )?function test_/gm) || []).length;
if (written !== tests.length) {
  console.error(`${written} tests are written and ${tests.length} are listed to run`);
  process.exit(1);
}

(async () => {
  const failures = [];
  const started = Date.now();
  for (const [name, fn] of tests) {
    try {
      await fn();
      process.stdout.write('.');
    } catch (err) {
      failures.push([name, err]);
      process.stdout.write('F');
    }
  }
  const elapsed = ((Date.now() - started) / 1000).toFixed(1);
  console.log(`\n\n${tests.length - failures.length} passed, ${failures.length} failed `
            + `in ${elapsed}s`);
  for (const [name, err] of failures) console.log(`\n--- ${name} ---\n${err.stack}`);
  process.exit(failures.length ? 1 : 0);
})();
