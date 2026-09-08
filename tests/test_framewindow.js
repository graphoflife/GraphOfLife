/*
 * The frame window, checked against runs whose answer is known by hand.
 *
 *     node tests/test_framewindow.js
 *
 * Same shape as tests/test_flowmodules.js: collect every test_ function, run
 * them all, print a dot or an F, and report at the end.
 */
const fs = require('fs');
const path = require('path');

const source = fs.readFileSync(
  path.join(__dirname, '..', 'web', 'js', 'framewindow.js'), 'utf8');
// formatNumber is the page's, and describe() is the only thing that wants it.
const FrameWindow = new Function(
  `const formatNumber = n => String(n); ${source}; return FrameWindow;`)();

// ---------------------------------------------------------------------------

/** A stand-in for a range input and the label that hides with it. */
function scrubber() {
  const label = { hidden: false };
  const readout = { textContent: '' };
  return {
    min: '', max: '', step: '', value: '',
    closest: () => label,
    parentElement: { querySelector: () => readout },
    label, readout
  };
}

/** Frames as a run records them: two per iteration, in order. */
function run(iterations, exportEvery = 1) {
  const frames = [];
  for (let i = 0; i < iterations; i++) {
    for (const phase of [1, 2]) frames.push({ iteration: i * exportEvery, phase });
  }
  return frames;
}

// ---------------------------------------------------------------------------

function test_a_short_run_is_shown_whole() {
  // Nothing to scroll: the window has to cover it and the slider go away,
  // rather than a run of twelve frames being shown twelve frames at a time
  // with a control that does nothing.
  const plan = FrameWindow.plan(12, 0, 100);
  if (plan.indices.length !== 12) {
    throw new Error(`took ${plan.indices.length} of a 12-frame run`);
  }
  if (plan.start !== 0) throw new Error('a short run should start at the beginning');

  const el = scrubber();
  FrameWindow.bindScrubber(el, plan);
  if (!el.label.hidden) throw new Error('the slider was offered with nothing to move');
}

function test_the_scrubber_says_which_stretch_is_on_screen() {
  // A slider with no numbers on it says there is more run than window and
  // nothing else: not where you are in it, and not how much you are seeing.
  const el = scrubber();
  FrameWindow.bindScrubber(el, FrameWindow.plan(2000, 400, 50));
  // formatNumber is stubbed to String() up top, so no thousands separators.
  const said = el.readout.textContent;
  if (said !== 'iterations 200–249 of 1000') {
    throw new Error(`the readout said "${said}", wanted iterations 200–249 of 1000`);
  }
}

function test_a_long_run_is_capped_at_the_asked_for_iterations() {
  // The whole point: a run of thousands of frames must not be read whole.
  const plan = FrameWindow.plan(10000, 0, 100);
  if (plan.indices.length !== 200) {
    throw new Error(`100 iterations should be 200 frames, got ${plan.indices.length}`);
  }
  if (plan.indices[0] !== 0 || plan.indices[199] !== 199) {
    throw new Error('the window should be the first 200 frames, contiguously');
  }

  const el = scrubber();
  FrameWindow.bindScrubber(el, plan);
  if (el.label.hidden) throw new Error('a run longer than the window needs the slider');
  if (el.max !== '9800') throw new Error(`the slider stops at ${el.max}, wanted 9800`);
}

function test_the_window_is_contiguous() {
  // Ancestry is a chain and a module's identity is its overlap with the frame
  // before it. Either one read at a stride is destroyed rather than thinned.
  const { indices } = FrameWindow.plan(5000, 1000, 100);
  for (let i = 1; i < indices.length; i++) {
    if (indices[i] !== indices[i - 1] + 1) {
      throw new Error(`a gap at ${i}: ${indices[i - 1]} then ${indices[i]}`);
    }
  }
}

function test_the_window_cannot_hang_off_the_end() {
  // Dragging the slider to the far right, or asking for a start past the end,
  // must give the last full window rather than a short one or an empty one.
  for (const from of [4900, 5000, 999999]) {
    const plan = FrameWindow.plan(5000, from, 100);
    if (plan.indices.length !== 200) {
      throw new Error(`from ${from} gave ${plan.indices.length} frames, wanted 200`);
    }
    if (plan.indices[plan.indices.length - 1] !== 4999) {
      throw new Error(`from ${from} ended at ${plan.indices[plan.indices.length - 1]}`);
    }
  }
  const back = FrameWindow.plan(5000, -50, 100);
  if (back.start !== 0) throw new Error('a negative start should clamp to the beginning');
}

function test_the_slider_steps_by_whole_iterations() {
  // A window opening on a game frame would pair each iteration's second half
  // with the next one's first, and every module would be matched across a
  // boundary that is not there.
  const el = scrubber();
  FrameWindow.bindScrubber(el, FrameWindow.plan(5000, 0, 100));
  if (Number(el.step) !== FrameWindow.PHASES) {
    throw new Error(`the slider steps by ${el.step} frames, not by an iteration`);
  }
  if (Number(el.max) % FrameWindow.PHASES !== 0) {
    throw new Error(`the last position, ${el.max}, is mid-iteration`);
  }
}

function test_frames_are_read_in_batches_and_all_of_them_arrive() {
  const asked = [];
  const API = {
    getFrames: (_run, from, count) => {
      for (let i = from; i < from + count; i++) asked.push(i);
      return Promise.resolve({ frames: Array.from({ length: count }, (_, k) => ({ i: from + k })) });
    }
  };
  const scoped = new Function('API', `const formatNumber = n => String(n); ${source}; return FrameWindow;`)(API);

  const { indices } = scoped.plan(50, 0, 100);
  return (async () => {
    const got = [];
    let batches = 0;
    for await (const batch of scoped.read('r', indices)) { batches++; got.push(...batch); }
    if (got.length !== 50) throw new Error(`${got.length} frames came back of 50`);
    if (asked.join() !== indices.join()) throw new Error('the wrong frames were asked for');
    if (batches !== Math.ceil(50 / scoped.BATCH)) {
      throw new Error(`${batches} batches for 50 frames at ${scoped.BATCH} apiece`);
    }
  })();
}

function test_only_the_fields_a_caller_reads_are_asked_for() {
  // A frame of a large run is mostly its edge list, and neither view looks at
  // it. Asking for whole frames was tens of megabytes parsed per window so
  // that two columns could be read out of each one.
  let sawFields = null;
  const API = {
    getFrames: (_run, from, count, fields) => {
      sawFields = fields;
      return Promise.resolve({ frames: [] });
    }
  };
  const scoped = new Function('API', `const formatNumber = n => String(n); ${source}; return FrameWindow;`)(API);
  const wanted = ['iteration', 'phase', 'brain_ids'];
  return (async () => {
    for await (const _b of scoped.read('r', scoped.plan(10, 0, 100).indices, wanted)) break;
    if (String(sawFields) !== String(wanted)) {
      throw new Error(`the projection was ${sawFields}, wanted ${wanted}`);
    }
  })();
}

function test_reading_can_stop_early() {
  // The lineage stops once the first frame says how big the world is. Frames
  // past that must never be fetched at all, which is the point of yielding.
  const asked = [];
  const API = {
    getFrames: (_run, from, count) => {
      for (let i = from; i < from + count; i++) asked.push(i);
      return Promise.resolve({ frames: Array.from({ length: count }, (_, k) => ({ i: from + k })) });
    }
  };
  const scoped = new Function('API', `const formatNumber = n => String(n); ${source}; return FrameWindow;`)(API);

  return (async () => {
    for await (const _batch of scoped.read('r', scoped.plan(1000, 0, 100).indices)) break;
    if (asked.length !== scoped.BATCH) {
      throw new Error(`${asked.length} frames were fetched after one batch, `
                    + `wanted ${scoped.BATCH}`);
    }
  })();
}

function test_where_the_window_sits_is_read_off_the_frames() {
  // Not computed from the frame index: a run recorded with export_every above
  // one has its iterations further apart than one apiece, and arithmetic on
  // the index would quietly report the wrong numbers.
  const sparse = run(100, 10).slice(0, 200);       // iterations 0, 10, 20, …
  const said = FrameWindow.describe(sparse, 2000);
  if (!said.includes('0') || !said.includes('990')) {
    throw new Error(`described a window of iterations 0–990 as "${said}"`);
  }

  // Covering the whole run, there is nothing to say about where it sits.
  if (FrameWindow.describe(run(10), 20) !== '') {
    throw new Error('a window covering the whole run still claimed to be partial');
  }
  if (FrameWindow.describe([], 20) !== '') throw new Error('no frames should say nothing');
}

function test_one_iteration_is_the_smallest_window() {
  // A guard against a zero or negative setting silently asking for no frames
  // and drawing an empty picture with no error.
  for (const asked of [0, -5]) {
    const plan = FrameWindow.plan(500, 0, asked);
    if (plan.indices.length !== FrameWindow.PHASES) {
      throw new Error(`a window of ${asked} iterations gave ${plan.indices.length} frames`);
    }
  }
}

// ---------------------------------------------------------------------------

const tests = Object.entries({
  test_a_short_run_is_shown_whole,
  test_the_scrubber_says_which_stretch_is_on_screen,
  test_a_long_run_is_capped_at_the_asked_for_iterations,
  test_the_window_is_contiguous,
  test_the_window_cannot_hang_off_the_end,
  test_the_slider_steps_by_whole_iterations,
  test_frames_are_read_in_batches_and_all_of_them_arrive,
  test_only_the_fields_a_caller_reads_are_asked_for,
  test_reading_can_stop_early,
  test_where_the_window_sits_is_read_off_the_frames,
  test_one_iteration_is_the_smallest_window
}).sort(([a], [b]) => a.localeCompare(b));

// A test that is written and never listed here is worse than no test: it reads
// as coverage and runs never.
const written = (fs.readFileSync(__filename, 'utf8').match(/^function test_/gm) || []).length;
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
