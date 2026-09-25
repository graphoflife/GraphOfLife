/*
 * The Explanation's timing and accounting, against the run it draws.
 *
 *     node tests/test_view.js
 *
 * Roughly seven thousand lines of browser code had no test under it, and every
 * defect found in it was found by a person looking at the screen: a supply
 * ball that jittered before it set off, piles that filled before anything had
 * reached them, conquest colour leaking into the staking phase. Those are the
 * ones worth catching here, because being wrong about them is *invisible* — a
 * pile that fills too early still looks like a pile filling.
 *
 * stepview.js is a plain top-level object that reaches for no global while it
 * is being defined, and its timing functions touch no canvas, so all of this
 * runs with nothing stubbed but a three-method layout for show(). The fixture
 * is the recording the Explanation itself draws.
 *
 * Same harness as tests/test_flowmodules.js: collect every test_ function, run
 * them all, print a dot or an F, and report at the end.
 */
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const StepView = new Function(
  `${fs.readFileSync(path.join(root, 'web', 'js', 'stepview.js'), 'utf8')}; return StepView;`)();

// jobs.js needs nothing from the page until it is attached to one.
const Jobs = new Function(
  `${fs.readFileSync(path.join(root, 'web', 'js', 'jobs.js'), 'utf8')}; return Jobs;`)();

// seriesload.js reaches for API only when a climb runs, so each test hands it
// a pretend backend. Metrics is for the Diagrams tests further down.
const Metrics = new Function('window', ['colormaps.js', 'metrics.js']
  .map(name => fs.readFileSync(path.join(root, 'web', 'js', name), 'utf8')).join('\n')
  + '; return Metrics;')({ devicePixelRatio: 1 });
// What the run statistics are called, what they mean, and the ratios among
// them: data only, so it loads bare.
const RunStats = new Function(
  `${fs.readFileSync(path.join(root, 'web', 'js', 'runstats.js'), 'utf8')}; return RunStats;`)();
const SERIES_SOURCE = fs.readFileSync(path.join(root, 'web', 'js', 'seriesload.js'), 'utf8');
/** A fresh loader, with a cache of its own, talking to `api`. */
const loaderFor = api => new Function('API', 'RunStats',
  `${SERIES_SOURCE}; return SeriesLoad;`)(api, RunStats);

// presets.js needs nothing from the page, so it loads the same bare way.
const Presets = new Function(
  `${fs.readFileSync(path.join(root, 'web', 'js', 'presets.js'), 'utf8')}; return Presets;`)();
const RUN = JSON.parse(
  fs.readFileSync(path.join(root, 'web', 'data', 'explain-run.json'), 'utf8'));

// ---------------------------------------------------------------------------

/** Enough of a view for the timing functions: no canvas, no layout. */
function viewOf(stage, effect, since = 0) {
  return {
    stage,
    effects: new Set(String(effect || '').split(/\s+/).filter(Boolean)),
    since,
    _plan: null,
    tokenRef: RUN.tokens
  };
}

// The only stub anything here needs: show() drives the force layout, and none
// of what is being tested depends on where it puts anything.
const fakeLayout = () => ({ setFrame() {}, reheat() {}, tick: () => false });

const stageWhere = pred => RUN.stages.find(pred);
const total = stage => stage.tokens.reduce((a, b) => a + b, 0);
const heldMap = stage => new Map(stage.ids.map((id, i) => [id, stage.tokens[i]]));

/** Everything the world holds at one moment of a moving schedule. */
function worldAt(stage, move) {
  let sum = 0;
  stage.ids.forEach((id, i) => { sum += move.held(id, stage.tokens[i]); });
  return sum;
}

// ---------------------------------------------------------------------------



async function test_a_second_job_for_an_owner_cancels_the_first() {
  // Asking a second question of the same thing means the first answer is no
  // longer wanted. Before this, each view kept its own token, two of the four
  // forgot to check theirs, and Lineage could not be stopped at all.
  Jobs.cancelAll();
  let firstSignal = null;
  const first = Jobs.run('lineage', 'first', job => {
    firstSignal = job.signal;
    return new Promise(resolve => setTimeout(resolve, 50));
  });
  const second = Jobs.run('lineage', 'second', async () => 'done');

  if (!firstSignal.aborted) {
    throw new Error('starting a second job for the same owner did not abort the first');
  }
  if (await second !== 'done') throw new Error('the second job did not run');
  await first;
  if (Jobs.busy('lineage')) throw new Error('the owner is still busy after both settled');
}

async function test_only_cancels_everything_else() {
  // Switching view is exactly this: whatever is not what you are looking at
  // stops, and what you are looking at carries on.
  Jobs.cancelAll();
  const signals = {};
  const hold = () => new Promise(resolve => setTimeout(resolve, 30));
  for (const owner of ['lineage', 'flow', 'diagrams']) {
    Jobs.run(owner, owner, job => { signals[owner] = job.signal; return hold(); });
  }
  Jobs.only('diagrams');

  if (!signals.lineage.aborted || !signals.flow.aborted) {
    throw new Error('Jobs.only left another owner running');
  }
  if (signals.diagrams.aborted) {
    throw new Error('Jobs.only cancelled the owner it was asked to keep');
  }
  Jobs.cancelAll();
}

async function test_a_cancelled_job_neither_reports_nor_counts_as_finished() {
  // Two halves of being in control. A job that was stopped must not keep
  // moving the progress bar, and must not be mistaken for one that finished —
  // that mistake is what left a view blank when you came back to it, because
  // it believed it had already loaded.
  Jobs.cancelAll();
  const reports = [];
  const realShow = Jobs._show;
  Jobs._show = (text, done, total) => reports.push([text, done, total]);
  try {

  let job = null;
  const running = Jobs.run('flow', 'reading', j => {
    job = j;
    return new Promise(resolve => setTimeout(resolve, 20));
  });
  job.report(1, 10);
  Jobs.cancel('flow');
  const before = reports.length;
  job.report(5, 10);
  await running;

  if (reports.length !== before) {
    throw new Error('a cancelled job went on reporting progress');
  }
  if (Jobs.finished('flow')) {
    throw new Error('a cancelled job was recorded as finished');
  }
  if (!Jobs.due('flow')) {
    throw new Error('a cancelled job was not left due a load, so nothing would resume it');
  }

  await Jobs.run('flow', 'reading', async () => 'ok');
  if (!Jobs.finished('flow') || Jobs.due('flow')) {
    throw new Error('a job that ran to the end was not recorded as finished');
  }
  } finally {
    Jobs._show = realShow;
  }
}

async function test_an_abort_is_not_an_error() {
  // The caller asked for it. A view that treated its own cancellation as a
  // failure would print "Could not read the window" every time you switched
  // away — the opposite of the calm this is for.
  Jobs.cancelAll();
  const running = Jobs.run('lineage', 'x', job => new Promise((resolve, reject) => {
    job.signal.addEventListener('abort', () => {
      const stop = new Error('aborted'); stop.name = 'AbortError'; reject(stop);
    });
  }));
  Jobs.cancel('lineage');
  const result = await running;           // must resolve, not throw
  if (result !== null) throw new Error('an aborted job did not resolve to null');
}

async function test_a_failed_job_says_so_and_is_tried_again() {
  // Each view used to catch its own errors inside the job, so a load that
  // failed was recorded as finished, and coming back to the view never tried
  // it again. Jobs decides now: the owner hears about the failure, and is
  // left due another go.
  Jobs.cancelAll();
  let heard = null;
  const result = await Jobs.run('lineage', 'x', async () => { throw new Error('gone'); },
                                err => { heard = err.message; });
  assert(result === null, 'a failed job with someone to tell still rejected');
  assert(heard === 'gone', 'the owner was not told why its load failed');
  assert(!Jobs.finished('lineage') && Jobs.due('lineage'),
    'a failed load was recorded as finished, so nothing would try it again');

  // With nobody to tell, the error is passed on rather than swallowed.
  const passed = await Jobs.run('lineage', 'x', async () => { throw new Error('gone'); })
    .then(() => 'swallowed', err => err.message);
  assert(passed === 'gone', 'a failure with nobody to hear it was swallowed');
}

async function test_an_answer_after_a_stop_is_a_stop() {
  // Every view used to ask, after each await, whether it had been stopped in
  // the meantime; one that forgot would paint an answer to a question nobody
  // was asking any more. The facade asks now, once, for all of them.
  const API = new Function(
    `${fs.readFileSync(path.join(root, 'web', 'js', 'api.js'), 'utf8')}; return API;`)();
  const controller = new AbortController();
  let sent = 0;
  API.backend = {
    getLineage: async () => { sent++; controller.abort(); return { nodes: [] }; }
  };
  const ending = promise => promise.then(() => 'answered', err => err.name);

  const late = await ending(API.getLineage('run', 0, 10, 'all', { signal: controller.signal }));
  assert(late === 'AbortError', `an answer that landed after the stop came back ${late}`);
  const after = await ending(API.getLineage('run', 0, 10, 'all', { signal: controller.signal }));
  assert(after === 'AbortError' && sent === 1, 'a read already stopped was sent anyway');
}

function test_a_saved_preset_is_read_in_todays_vocabulary() {
  // Translating an old preset used to happen wherever one was applied, so
  // every place that applied one had to remember to. It happens where saved
  // presets are read now, once.
  const stored = { old: { nodeColorBy: 'log_tokens', edgeColorBy: 'flow', distMetric: 'node:age' } };
  const storage = { getItem: () => JSON.stringify(stored), setItem() {} };
  const Saved = new Function('Metrics', 'localStorage',
    `${fs.readFileSync(path.join(root, 'web', 'js', 'presets.js'), 'utf8')}; return Presets;`)(
    Metrics, storage);

  const preset = Saved.get('old');
  assert(preset.nodeColorBy === 'tokens' && preset.nodeColorLog === true,
    `a log_ spelling came back as ${preset.nodeColorBy}, log ${preset.nodeColorLog}`);
  assert(preset.edgeColorLog === true, 'flow lost the log scale it was always drawn on');
  assert(preset.distMetric === Metrics.qualify('node', 'node_id'),
    `age kept its old meaning: ${preset.distMetric}`);
}

function test_the_default_look_is_stated_once() {
  // The default preset is the only statement of how the Viewer starts. It
  // used to be stated three times: viewer.js and index.html each carried a
  // copy, both overwritten during init before anything read them, and both
  // rotted — viewer.js disagreed with the preset in eighteen places. Six
  // layout keys were guarded against coming back; every key is now.
  const opening = Presets.builtIn('default');

  // The other built-in looks inherit the layout they do not override, so the
  // base has to carry all of it as well.
  for (const key of ['forceCharge', 'forceLink', 'forceCenter',
                     'forceAngular', 'forceDamping', 'forceTheta']) {
    if (typeof opening[key] !== 'number' || typeof Presets.BASE_LAYOUT[key] !== 'number') {
      throw new Error(`${key} is missing from the default preset or BASE_LAYOUT, and `
                    + `nothing else declares it — a look would start with it undefined`);
    }
  }

  const viewer = fs.readFileSync(path.join(root, 'web', 'js', 'viewer.js'), 'utf8');
  const start = viewer.indexOf('  settings: {');
  const literal = viewer.slice(start, viewer.indexOf('\n  },\n', start));
  const inViewer = Object.keys(opening).filter(key => new RegExp(`\\b${key}\\s*:`).test(literal));
  if (inViewer.length) {
    throw new Error(`viewer.js states ${inViewer.join(', ')} again; init() replaces them `
                  + `with the preset before anything reads them`);
  }

  const markup = fs.readFileSync(path.join(root, 'web', 'index.html'), 'utf8');
  const inMarkup = Object.keys(opening).filter(key => {
    const tag = markup.match(new RegExp(`<[^>]*\\bid="${key}"[^>]*>`));
    return tag && /\bvalue="|\bchecked\b/.test(tag[0]);
  });
  if (inMarkup.length) {
    throw new Error(`index.html gives ${inMarkup.join(', ')} a value again; `
                  + `syncControlsFromSettings overwrites it during init, so it `
                  + `can only ever be right by coincidence`);
  }
}

function test_the_world_holds_its_whole_supply_at_both_ends_of_the_game() {
  // The invariant that makes the animation honest: tokens are conserved, so
  // any moment where the world is not holding its full supply is a moment when
  // the difference is drawn on the links. It has to leave and it has to arrive.
  const stage = stageWhere(s => s.step === 'game.stake' && s.marks.staked.length);
  const supply = total(stage);
  const g = StepView.GAME;

  const atRest = worldAt(stage, StepView._staking(viewOf(stage, 'stakes', 0)));
  const landed = worldAt(stage, StepView._staking(viewOf(stage, 'stakes', g.hold + g.travel + 1)));
  if (Math.abs(atRest - supply) > 1e-6) {
    throw new Error(`before the tokens set off the world holds ${atRest}, not ${supply}`);
  }
  if (Math.abs(landed - supply) > 1e-6) {
    throw new Error(`after they land the world holds ${landed}, not ${supply}`);
  }

  // And in flight, strictly less — the missing tokens are the ones on screen.
  let lowest = Infinity;
  for (let t = g.hold; t < g.hold + g.travel; t += 0.05) {
    lowest = Math.min(lowest, worldAt(stage, StepView._staking(viewOf(stage, 'stakes', t))));
  }
  if (!(lowest < supply)) {
    throw new Error(`the world never dipped below ${supply} mid-flight, so nothing `
                  + `was ever drawn as travelling`);
  }
}

function test_a_pile_never_fills_before_anything_reaches_it() {
  // The defect this is named for: reading the balance as one smooth slide from
  // old to new had piles filling up before a single token had crossed. A pile
  // must not exceed its final holding until the arrival window opens.
  const stage = stageWhere(s => s.step === 'game.stake' && s.marks.staked.length);
  const g = StepView.GAME;
  const had = heldMap(stage);

  const end = StepView._staking(viewOf(stage, 'stakes', g.hold + g.travel + 1));
  const finalOf = new Map(stage.ids.map(id => [id, end.held(id, had.get(id))]));

  // Two thirds of the way across, nothing has arrived yet (arrivals ramp over
  // the last `spread` of the crossing), so no pile may be above where it ends
  // unless it is on its way down.
  const early = StepView._staking(viewOf(stage, 'stakes', g.hold + g.travel * 0.5));
  for (const id of stage.ids) {
    const now = early.held(id, had.get(id));
    const start = had.get(id), finish = finalOf.get(id);
    const ceiling = Math.max(start, finish) + 1e-9;
    if (now > ceiling) {
      throw new Error(`agent ${id} held ${now} halfway across, above both its `
                    + `start (${start}) and its finish (${finish})`);
    }
  }
}

function test_a_stake_lands_where_the_engine_says_it_lands() {
  // The accounting the eye cannot check. Once the crossing is done, an agent
  // holds exactly what was staked on it — its own stake never left, and every
  // other stake has arrived.
  const stage = stageWhere(s => s.step === 'game.stake' && s.marks.staked.length);
  const g = StepView.GAME;
  const move = StepView._staking(viewOf(stage, 'stakes', g.hold + g.travel + 1));
  const had = heldMap(stage);

  const onto = new Map(stage.ids.map(id => [id, 0]));
  for (const [to, , amount] of stage.marks.staked) {
    if (onto.has(to)) onto.set(to, onto.get(to) + amount);
  }

  let checked = 0;
  for (const id of stage.ids) {
    const drawn = move.held(id, had.get(id));
    const staked = onto.get(id);
    if (Math.abs(drawn - staked) > 1e-6) {
      throw new Error(`agent ${id} is drawn holding ${drawn} at the end of the `
                    + `crossing; the stakes on it come to ${staked}`);
    }
    checked++;
  }
  if (checked < 20) throw new Error(`only ${checked} agents checked; the fixture looks wrong`);
}

function test_a_parent_pays_for_its_child_as_the_tokens_cross() {
  // A child appears holding what it was given, but the snapshot is taken after
  // the spending — so unless the parent is handed it back until the tokens
  // leave, they appear from nowhere and nobody is seen paying.
  const stage = stageWhere(s => s.step === 'repro.born' && s.marks.parents.length);
  const b = StepView.BIRTH;
  const had = heldMap(stage);
  const [parent, child] = stage.marks.parents[0];
  const paid = had.get(child);
  if (!(paid > 0)) throw new Error('the first recorded child was given nothing');

  const start = StepView._inheritance(viewOf(stage, 'inherit', b.hold));
  const end = StepView._inheritance(viewOf(stage, 'inherit', b.hold + b.travel + 1));

  const parentBefore = start.held(parent, had.get(parent));
  const parentAfter = end.held(parent, had.get(parent));
  const childBefore = start.held(child, had.get(child));
  const childAfter = end.held(child, had.get(child));

  if (Math.abs(parentBefore - (had.get(parent) + paid)) > 1e-6) {
    throw new Error(`the parent starts holding ${parentBefore}; it should still `
                  + `have the ${paid} it is about to hand over`);
  }
  if (Math.abs(parentAfter - had.get(parent)) > 1e-6) {
    throw new Error(`the parent ends holding ${parentAfter}, not ${had.get(parent)}`);
  }
  if (Math.abs(childBefore) > 1e-6) {
    throw new Error(`the child starts holding ${childBefore}, before anything reached it`);
  }
  if (Math.abs(childAfter - paid) > 1e-6) {
    throw new Error(`the child ends holding ${childAfter}, not the ${paid} it was given`);
  }
}

function test_a_disc_is_sized_by_what_is_held_now_not_at_the_end_of_the_step() {
  // The ring of dots has always animated: a pile loses what it sends when the
  // dots set off and gains what it is sent when they land. The disc under it
  // was sized from the end-of-step snapshot instead, so an agent's size
  // snapped to its new value while its tokens were still visibly crossing the
  // link — the same fact, drawn two different ways at once.
  const stage = stageWhere(s => s.step === 'repro.born' && s.marks.parents.length);
  const b = StepView.BIRTH;
  const [parent, child] = stage.marks.parents[0];
  const had = heldMap(stage);
  const paid = had.get(child) || 0;
  if (paid <= 0) throw new Error('picked a birth with no endowment to follow');

  const at = t => StepView._holding(viewOf(stage, 'inherit', t)).now;

  const opening = at(b.hold);
  const closing = at(b.hold + b.travel + 1);

  if (Math.abs(opening.get(parent) - (had.get(parent) + paid)) > 1e-6) {
    throw new Error(`the parent's disc opens sized for ${opening.get(parent)}, `
                  + `wanted ${had.get(parent) + paid} — its own ${had.get(parent)} `
                  + `plus the ${paid} it has not handed over yet`);
  }
  if (Math.abs(opening.get(child)) > 1e-6) {
    throw new Error(`the child's disc opens sized for ${opening.get(child)}, before `
                  + `anything has reached it`);
  }
  if (Math.abs(closing.get(parent) - had.get(parent)) > 1e-6 ||
      Math.abs(closing.get(child) - paid) > 1e-6) {
    throw new Error('the discs do not settle on the end-of-step amounts');
  }
}

function test_the_handover_never_creates_or_strands_a_token() {
  // At every moment the world holds its supply less exactly what is in the
  // air, so it may dip by as much as the endowments being carried and not one
  // token more, and it may never rise above the supply at all.
  const stage = stageWhere(s => s.step === 'repro.born' && s.marks.parents.length);
  const b = StepView.BIRTH;
  const supply = total(stage);
  const had = heldMap(stage);
  const handed = stage.marks.parents.reduce((n, [, child]) => n + (had.get(child) || 0), 0);
  if (!(handed > 0)) throw new Error('nothing is handed over in the fixture stage');

  let deepest = supply;
  for (let t = 0; t <= b.hold + b.travel + 0.5; t += 0.02) {
    const held = worldAt(stage, StepView._inheritance(viewOf(stage, 'inherit', t)));
    deepest = Math.min(deepest, held);
    if (held > supply + 1e-6) {
      throw new Error(`at ${t.toFixed(2)}s the world held ${held}, above the ${supply} `
                    + `it started with — a handover cannot mint tokens`);
    }
    if (held < supply - handed - 1e-6) {
      throw new Error(`at ${t.toFixed(2)}s the world held ${held}; only ${handed} is `
                    + `being carried, so it cannot fall below ${supply - handed}`);
    }
  }
  if (!(deepest < supply)) {
    throw new Error('the world never dipped, so nothing was ever drawn crossing a link');
  }
}

function test_one_token_is_one_dot_and_the_reference_is_fifteen() {
  const ref = RUN.tokens;
  if (StepView.dots(1, ref) !== 1) {
    throw new Error(`one token drew ${StepView.dots(1, ref)} dots`);
  }
  if (StepView.dots(ref, ref) !== StepView.TOKEN_DOTS) {
    throw new Error(`the reference pile drew ${StepView.dots(ref, ref)} dots, `
                  + `not ${StepView.TOKEN_DOTS}`);
  }
}

function test_no_pile_draws_more_dots_than_it_holds_tokens() {
  // Small piles have to stay countable: three dots on an agent holding two
  // tokens is a lie the reader can check by eye.
  const ref = RUN.tokens;
  for (const stage of RUN.stages) {
    for (const held of stage.tokens) {
      if (held <= 0) continue;
      const drawn = StepView.dots(held, ref);
      if (drawn > held) {
        throw new Error(`${drawn} dots drawn for ${held} tokens`);
      }
      if (drawn > StepView.TOKEN_DOTS) {
        throw new Error(`${drawn} dots drawn, above the cap of ${StepView.TOKEN_DOTS}`);
      }
    }
  }
}

function test_the_starving_die_first_and_the_stranded_second() {
  // Two removals in order, for different reasons, and the recording keeps one
  // list. The split is read back from the tokens: starvation runs first, so
  // anyone removed holding nothing starved and anyone removed still holding
  // something was cut adrift.
  const stage = stageWhere(s => (s.marks.removed || []).length > 1);
  const view = { shown: new Map(), layout: fakeLayout() };
  StepView.show(view, stage, { effect: 'none' });

  const [starved, stranded] = view.waves;
  const had = heldMap(stage);
  const removed = new Set(stage.marks.removed);

  for (const id of starved) {
    if (had.get(id) !== 0) throw new Error(`agent ${id} is in the starving wave holding ${had.get(id)}`);
  }
  for (const id of stranded) {
    if (!(had.get(id) > 0)) throw new Error(`agent ${id} is in the stranded wave holding nothing`);
  }
  const together = new Set([...starved, ...stranded]);
  if (together.size !== removed.size || [...removed].some(id => !together.has(id))) {
    throw new Error(`the two waves cover ${together.size} agents and the recording `
                  + `removed ${removed.size}`);
  }
  if (!starved.length || !stranded.length) {
    throw new Error('the fixture stage should have one of each kind to be worth testing');
  }
}

function test_eyes_do_not_all_look_away_at_once() {
  // Switching every gaze on one clock made the whole graph flick together,
  // which reads as a cut rather than as forty agents each reading their own
  // neighbourhood. The phases have to be spread through the cycle.
  const stage = stageWhere(s => s.step === 'game.observe' && s.ids.length > 20);
  const view = viewOf(stage, 'eyes');

  StepView.gaze(view, 0);
  const phases = new Set([...view._gaze.values()].map(g => Math.round(g.through * 20)));
  if (phases.size < 8) {
    throw new Error(`${view._gaze.size} eyes share only ${phases.size} distinct `
                  + `phases; they are looking on one clock`);
  }

  // And over a tenth of a second only a slice of them changes target.
  StepView.gaze(view, 0);
  const before = new Map([...view._gaze].map(([id, g]) => [id, g.at]));
  StepView.gaze(view, 0.1);
  let moved = 0;
  for (const [id, g] of view._gaze) if (before.get(id) !== g.at) moved++;
  const share = moved / view._gaze.size;
  if (share > 0.5) {
    throw new Error(`${moved} of ${view._gaze.size} eyes changed target in a tenth `
                  + `of a second (${(share * 100).toFixed(0)}%) — that is a cut, not a scan`);
  }
}

function test_every_eye_looks_at_a_neighbour_or_itself() {
  const stage = stageWhere(s => s.step === 'game.observe' && s.ids.length > 20);
  const view = viewOf(stage, 'eyes');
  const near = new Map(stage.ids.map(id => [id, new Set([id])]));
  for (const [a, b] of stage.edges) {
    if (near.has(a)) near.get(a).add(b);
    if (near.has(b)) near.get(b).add(a);
  }
  for (let t = 0; t < 3; t += 0.37) {
    StepView.gaze(view, t);
    for (const [id, g] of view._gaze) {
      if (!near.get(id).has(g.at)) {
        throw new Error(`agent ${id} is looking at ${g.at}, which is neither a `
                      + `neighbour nor itself`);
      }
    }
  }
}

function test_the_supply_schedule_is_fixed_once_the_ball_sets_off() {
  // The jitter that shipped. The firing order is worked out from where the
  // agents are, and the layout is still settling — so recomputing it each
  // frame let two agents swap places, which across the wrap threw the whole
  // schedule and the ball answered by jumping backwards.
  const stage = stageWhere(s => s.ids.length > 20);
  const view = viewOf(stage, 'arrive', 0.4);
  const spot = new Map(stage.ids.map((id, i) => {
    const a = (i / stage.ids.length) * Math.PI * 2;
    return [id, { x: Math.cos(a) * 100, y: Math.sin(a) * 100 }];
  }));

  const place = id => spot.get(id) || null;
  const count = held => StepView.dots(held, RUN.tokens);
  StepView._supply(view, place, count, 600, 400);
  const first = new Map(view._plan);
  if (!first.size) throw new Error('no shots were scheduled');

  // The layout moves under it, exactly as a settling force layout would.
  for (const [id, p] of spot) spot.set(id, { x: p.y * 1.3, y: -p.x * 0.7 });
  StepView._supply(view, place, count, 600, 400);

  if (view._plan.size !== first.size) {
    throw new Error(`the schedule changed size, ${first.size} to ${view._plan.size}`);
  }
  for (const [key, when] of first) {
    if (view._plan.get(key) !== when) {
      throw new Error(`shot ${key} moved from ${when} to ${view._plan.get(key)} `
                    + `when the agents did — the schedule is being recomputed`);
    }
  }
}

function test_tokens_are_fired_in_the_order_the_ball_comes_round() {
  // The ball fires at whatever it is pointing at, so at any moment the tokens
  // that have left are exactly the front of the schedule — never one from
  // further round the ring. That is the property the sort exists for, and the
  // one a reordering would break.
  const stage = stageWhere(s => s.ids.length > 20);
  const spot = new Map(stage.ids.map((id, i) => {
    const a = (i / stage.ids.length) * Math.PI * 2;
    return [id, { x: Math.cos(a) * 100, y: Math.sin(a) * 100 }];
  }));
  const place = id => spot.get(id) || null;
  const count = held => StepView.dots(held, RUN.tokens);

  let previously = 0;
  for (const t of [0, 0.4, 0.7, 1.0, 1.4, 1.8, 2.2]) {
    const view = viewOf(stage, 'arrive', t);
    const supply = StepView._supply(view, place, count, 600, 400);
    const run = Math.max(0, t - StepView.SUPPLY.lead) / StepView.SUPPLY.sweep;

    let fired = 0;
    for (const [key, when] of view._plan) {
      const [id, k] = key.split(':');
      const gone = Boolean(supply.shot(Number(id), Number(k)));
      if (gone !== (run >= when)) {
        throw new Error(`at ${t}s shot ${key} is ${gone ? '' : 'not '}in the air, but `
                      + `the ball is at ${run.toFixed(3)} and it is due at ${when.toFixed(3)}`);
      }
      if (gone) fired++;
    }
    if (fired < previously) {
      throw new Error(`${fired} tokens had left at ${t}s, fewer than the ${previously} `
                    + `that had left earlier — the ball went backwards`);
    }
    previously = fired;
  }
  if (previously !== stage.ids.reduce((n, id, i) => n + count(stage.tokens[i]), 0)) {
    throw new Error('the sweep finished without firing every token');
  }
}

function test_a_brain_stays_marked_once_its_turn_has_passed() {
  // What should build up is that every brain was touched, not just the one
  // being touched — so the set only ever grows across a step.
  const stage = stageWhere(s => s.step === 'game.mutate');
  const view = viewOf(stage, 'mutate', 0);
  let seen = new Set();
  for (let t = 0; t <= 2; t += 0.05) {
    view.since = t;
    StepView.mutating(view);
    const now = new Set(view._mutated ? view._mutated.keys() : []);
    for (const id of seen) {
      if (!now.has(id)) throw new Error(`agent ${id} was marked mutated and then unmarked`);
    }
    seen = now;
  }
  if (seen.size !== stage.ids.length) {
    throw new Error(`${seen.size} of ${stage.ids.length} brains were marked by the end`);
  }
}

function test_a_conquest_does_not_show_its_answer_before_the_brain_arrives() {
  // Showing the result from the first frame gave away the outcome before the
  // journey that decides it. Progress must start at zero and reach one.
  const stage = stageWhere(s => (s.marks.taken || []).length);
  if (!stage) throw new Error('the fixture has no stage where a node was taken');
  const role = { pairs: stage.marks.taken };

  const opening = viewOf(stage, 'conquer', 0);
  StepView._conquest(opening, role);
  for (const [node, at] of opening._conquest) {
    if (at.t !== 0) throw new Error(`node ${node} was ${at.t} of the way conquered at the start`);
  }

  const later = viewOf(stage, 'conquer',
                       StepView.CONQUEST.travel + StepView.CONQUEST.stagger + 1);
  StepView._conquest(later, role);
  for (const [node, at] of later._conquest) {
    if (at.t !== 1) throw new Error(`node ${node} never finished: ${at.t}`);
  }
}

function test_the_conquest_waits_for_the_stakes_on_the_step_that_shows_both() {
  // Conquest colour leaking into the staking phase: on a step drawing both,
  // nothing about the outcome exists until the tokens have landed, because
  // that is what decides it.
  const stage = stageWhere(s => (s.marks.taken || []).length);
  const role = { pairs: stage.marks.taken };
  const view = viewOf(stage, 'stakes conquer', StepView.GAME.hold);
  StepView._conquest(view, role);
  if (view._conquest !== null) {
    throw new Error('a conquest was under way while the stakes were still crossing');
  }
}

// ---------------------------------------------------------------------------

// ---- the run's history ----------------------------------------------------

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const assert = (ok, message) => { if (!ok) throw new Error(message); };

/**
 * A backend for a run of `total` samples that keeps its history the way
 * gol_series.History does, in miniature: a request summarises the samples of
 * its bisection prefix it has not got at that depth, and the reply is
 * everything known. `nodes` is cheap; `bridges` walks the graph and is null
 * on a sample only summarised cheaply. `calls` records what was asked for.
 */
function pretendRun(total, calls = [], { delay = 0 } = {}) {
  const cheap = new Set(), deep = new Set();
  const state = { building: false, done: 0, total: 0 };
  return {
    state,
    async getSeries(runId, points, keys) {
      calls.push({ points, keys });
      const heavy = (keys || []).includes('bridges');
      const take = points === null ? total : Math.min(total, Math.max(2, points));
      const missing = [];
      for (let it = 0; it < take; it++) if (!(heavy ? deep : cheap).has(it)) missing.push(it);
      Object.assign(state, { building: missing.length > 0, done: 0, total: 2 * missing.length });
      for (const it of missing) {
        if (delay) { await sleep(delay); state.done += 1; await sleep(delay); state.done += 1; }
        cheap.add(it);
        if (heavy) deep.add(it);
      }
      state.building = false;
      const known = [...cheap].sort((x, y) => x - y);
      const rows = known.flatMap(it => [1, 2].map(phase =>
        ({ iteration: it, phase, nodes: 100 + it, bridges: deep.has(it) ? it : null })));
      const series = {};
      for (const key of ['iteration', 'phase', 'nodes', 'bridges']) series[key] = rows.map(r => r[key]);
      return {
        keys: ['iteration', 'phase', 'nodes', 'bridges'], series, count: rows.length,
        frames: 2 * total, totalPoints: total, heavyKeys: ['bridges'],
        done: (heavy ? deep : cheap).size, complete: cheap.size === total, heavy: deep.size === total
      };
    },
    async getSeriesProgress() { return { ...state }; }
  };
}

async function test_a_history_is_asked_for_by_what_is_plotted() {
  const calls = [];
  const loader = loaderFor(pretendRun(20, calls));

  await loader.climb('run', ['nodes']);
  assert(calls.length && calls.every(c => c.keys.join() === 'nodes'),
    `a population chart asked for ${JSON.stringify(calls[0] && calls[0].keys)}`);

  // A derived statistic is asked for by what it is made of.
  calls.length = 0;
  await loader.climb('run', ['bridgeShare']);
  assert(calls.length && calls.every(c => c.keys.join() === 'bridges,edges'),
    `bridges / edges asked for ${JSON.stringify(calls[0] && calls[0].keys)}`);
}

async function test_a_summarised_run_is_one_request() {
  // Every reply is the whole history, so on a run already summarised the
  // first step of a climb is also its last. It used to be ten requests.
  const calls = [];
  const loader = loaderFor(pretendRun(20, calls));
  await loader.climb('run', ['nodes']);

  calls.length = 0;
  const steps = [];
  await loader.climb('run', ['nodes'], { onStep: reply => steps.push(reply) });
  assert(calls.length === 1, `a summarised run took ${calls.length} requests`);
  assert(steps.length === 1 && steps[0].count === 40, 'the one reply was not the whole history');
}

function test_a_history_of_a_smaller_run_is_not_ready() {
  // A history finished at one size used to count as finished until something
  // said to forget it, and only the Viewer ever did.
  const loader = loaderFor({});
  loader.cache.set('run', { frames: 40, complete: true, heavy: false, heavyKeys: ['bridges'] });
  assert(loader.ready('run', ['nodes'], 40), 'a finished history of the run as it is was not ready');
  assert(!loader.ready('run', ['nodes'], 42), 'a history of the run before it grew counts as finished');
  assert(!loader.ready('run', ['bridges'], 40), 'a cheap history passes for one with the graph statistics');
  assert(!loader.ready('other', ['nodes'], 0), 'a run never asked about counts as ready');
}

function test_a_history_longer_than_its_run_is_dropped() {
  // Resuming a run from an earlier checkpoint cuts it back, and its history
  // then describes frames that are gone. A history merely behind stays: that
  // is ready()'s to judge, and it is still worth drawing while the rest loads.
  const loader = loaderFor({});
  loader.cache.set('cut', { frames: 40 });
  loader.cache.set('behind', { frames: 40 });
  loader.noteSize('cut', 30);
  loader.noteSize('behind', 50);
  assert(!loader.cache.has('cut'), 'the history of frames that are gone was kept');
  assert(loader.cache.has('behind'), 'a history only behind the run was thrown away');
}

function test_a_derived_statistic_reads_like_a_stored_one() {
  const loader = loaderFor({});
  const series = { bridges: [2, null, 3], edges: [10, 10, 0], nodes: [5, 6, 7] };
  const share = loader.column(series, 'bridgeShare');
  assert(share[0] === 0.2, `bridges / edges read ${share[0]} where 2 of 10 is 0.2`);
  assert(share[1] === null && share[2] === null,
    'a missing count, or no edges to divide by, read as a number rather than as nothing measured');
  assert(loader.column(series, 'nodes') === series.nodes, 'a stored column did not come back as itself');
  assert(loader.column(series, 'leafShare') === null,
    'a ratio of columns this run does not have came back as something');
}

/**
 * The Viewer's history load, reading through the real API facade from
 * `backend`, with the trajectory and the stat popup it draws into counted
 * rather than drawn.
 */
function viewerHistoryWith(backend) {
  const API = new Function(
    `${fs.readFileSync(path.join(root, 'web', 'js', 'api.js'), 'utf8')}; return API;`)();
  API.backend = backend;
  const loader = loaderFor(API);
  const detail = { said: '', refresh() {}, failed(err) { this.said = `Could not load history: ${err.message}`; } };
  const viewer = { runId: 'run', frameCount: 40 };
  new Function('Viewer', 'Jobs', 'SeriesLoad', 'StatDetail', 'Metrics', 'RunStats',
    fs.readFileSync(path.join(root, 'web', 'js', 'viewer-panels.js'), 'utf8'))(
    viewer, Jobs, loader, detail, Metrics, RunStats);
  viewer.updateTrajectory = () => {};
  return { viewer, detail, loader };
}

async function test_a_history_that_fails_says_so_and_never_rejects() {
  // What resume() does on the way back to the Viewer: start a load and keep
  // nothing of it. When the server was gone, the rejection went to the
  // console as uncaught and the popup went on saying "24 points".
  Jobs.cancelAll();
  const { viewer, detail } = viewerHistoryWith({
    async getSeries() { throw new Error('Failed to fetch'); },
    async getSeriesProgress() { return {}; }
  });
  const settled = await viewer.loadHistory(['nodes']).then(() => 'resolved', () => 'rejected');
  assert(settled === 'resolved', 'a failed history load rejected, with nothing waiting on it');
  assert(/Could not load history: Failed to fetch/.test(detail.said),
    `the popup said "${detail.said}" about a load that failed`);
}

async function test_the_trajectory_and_the_popup_share_one_history_load() {
  // They used to share the popup's job, so whichever asked second cancelled
  // the other: opening a statistic partway through the trajectory's bridge
  // counts stopped them, and the Load history button came back.
  Jobs.cancelAll();
  const calls = [];
  const { viewer, loader } = viewerHistoryWith(pretendRun(20, calls, { delay: 1 }));

  // A load of the graph statistics brings the cheap ones too.
  const trajectory = viewer.loadHistory(['bridges']);
  while (!loader.cache.has('run')) await sleep(1);
  assert(viewer.loadHistory(['nodes']) === trajectory,
    'asking for a cheap statistic during a load of the graph statistics started another');
  await trajectory;
  assert(loader.ready('run', ['bridges', 'nodes'], 40), 'the load did not finish');

  // A cheap load does not bring the graph statistics: asking for one widens it.
  Jobs.cancelAll();
  const asked = [];
  const fresh = viewerHistoryWith(pretendRun(20, asked, { delay: 1 }));
  const cheap = fresh.viewer.loadHistory(['nodes']);
  while (!fresh.loader.cache.has('run')) await sleep(1);
  const wider = fresh.viewer.loadHistory(['bridges']);
  assert(wider !== cheap, 'a statistic the load in flight does not bring was left waiting');
  await Promise.all([cheap, wider]);
  assert(fresh.loader.ready('run', ['nodes', 'bridges'], 40) && !Jobs.busy(fresh.viewer),
    'widening the load did not bring both');
  const last = asked[asked.length - 1].keys;
  assert(last.includes('nodes') && last.includes('bridges'),
    `the widened load asked for ${last}`);
}

function test_two_statistics_pair_by_frame_or_else_by_iteration() {
  const loader = loaderFor({});
  const every = () => true;

  // Recorded together: one point per frame, under the phase filter.
  const together = { iteration: [0, 0, 1, 1], phase: [1, 2, 1, 2],
                     nodes: [5, 6, 7, 8], edges: [9, 10, 11, 12] };
  const framed = loader.pairs(together, 'nodes', 'edges', phase => phase === 2);
  assert(framed.pairing === 'one point per frame' && framed.points.map(p => p.x).join() === '6,8',
    `two statistics recorded together paired as ${JSON.stringify(framed)}`);

  // Never on the same frame, births on reproduction and revolutions on the
  // game, so each iteration's two halves make its point.
  const apart = { iteration: [0, 0, 1, 1], phase: [1, 2, 1, 2],
                  births: [3, null, 4, null], revolutions: [null, 1, null, 2] };
  const joined = loader.pairs(apart, 'births', 'revolutions', every);
  assert(joined.pairing === 'one point per iteration, across its phases'
         && JSON.stringify(joined.points.map(p => [p.x, p.y])) === '[[3,1],[4,2]]',
    `births against revolutions paired as ${JSON.stringify(joined)}`);

  assert(loader.pairs(apart, 'births', 'missing', every) === null,
    'a statistic the history lacks was paired anyway');
  const lone = { iteration: [0], phase: [1], births: [3], revolutions: [null] };
  assert(loader.pairs(lone, 'births', 'revolutions', every).pairing === null,
    'one point was offered as a path');
}

async function test_the_bar_counts_samples_and_only_moves_forward() {
  // The server counts frames, two to a sample, and counts only the frames of
  // the step in flight. Passing that straight to the bar made it run 4%, 6%,
  // 8%, 3%, 34%, 66%, 15% — two different measures taking turns.
  const total = 20;
  const loader = loaderFor(pretendRun(total, [], { delay: 1 }));
  loader.POLL_MS = 3;

  const reports = [];
  const job = { signal: null, report: (done, of) => reports.push([done, of]) };
  await loader.climb('run', ['bridges'], { job });

  assert(reports.length, 'the bar was never told anything');
  assert(reports.every(([, of]) => of === total),
    `the bar was given totals other than the ${total} samples: ${JSON.stringify(reports)}`);
  for (let i = 1; i < reports.length; i++) {
    assert(reports[i][0] > reports[i - 1][0],
      `the bar went from ${reports[i - 1][0]} to ${reports[i][0]}`);
  }
  assert(reports[reports.length - 1][0] === total, 'the bar did not reach the end');
  const boundaries = new Set([2, 3, 5, 9, 17, total]);
  assert(reports.some(([done]) => !boundaries.has(done)),
    'the bar only moved between steps; the server\'s own count was never used');
}

// ---- Diagrams: a control that changes what is plotted loads it -------------

const DIAGRAMS_SOURCE = ['theses.js', 'diagrams.js', 'diagram-controls.js']
  .map(name => fs.readFileSync(path.join(root, 'web', 'js', name), 'utf8')).join('\n');

/**
 * The Diagrams tab with nothing on screen: a canvas that is never painted and
 * a count of how often it would have been, reading run histories through
 * `loader` and frames through `api`.
 */
const COLORMAPS = new Function(
  `${fs.readFileSync(path.join(root, 'web', 'js', 'colormaps.js'), 'utf8')}; return COLORMAPS;`)();

function diagramsWith(loader, api = {}) {
  const FrameWindow = new Function('API', `const formatNumber = n => String(n); ${
    fs.readFileSync(path.join(root, 'web', 'js', 'framewindow.js'), 'utf8')}; return FrameWindow;`)(api);
  const diagrams = new Function('SeriesLoad', 'Metrics', 'Jobs', 'API', 'FrameWindow', 'RunStats',
    'COLORMAPS', `${DIAGRAMS_SOURCE}; return Diagrams;`)(
    loader, Metrics, Jobs, api, FrameWindow, RunStats, COLORMAPS);
  diagrams.canvas = {};
  diagrams.drawn = 0;
  diagrams.draw = function () { this.drawn += 1; };
  diagrams.say = () => {};
  return diagrams;
}

const lineOf = stat => ({ run: 'run', stat, phase: 'all', stretch: false, maxFrames: null });

async function test_switching_a_line_to_a_graph_statistic_loads_it() {
  const calls = [];
  const diagrams = diagramsWith(loaderFor(pretendRun(20, calls)));
  diagrams.active = 'timeline';
  diagrams.settings.timeline.lines = [lineOf('nodes')];
  await diagrams.refresh();
  assert(calls.length && calls.every(c => !c.keys.includes('bridges')),
    'a population line asked for graph statistics');

  // What the statistic menu does: change the line, then refresh.
  calls.length = 0;
  diagrams.settings.timeline.lines[0].stat = 'bridges';
  await diagrams.refresh();
  assert(calls.some(c => c.keys.includes('bridges')),
    'switching the line to bridges never asked for them, and the chart would say '
    + '"Reading…" with nothing reading');
}

async function test_a_change_that_needs_nothing_new_draws_without_loading() {
  const calls = [];
  const diagrams = diagramsWith(loaderFor(pretendRun(20, calls)));
  diagrams.active = 'timeline';
  diagrams.settings.timeline.lines = [lineOf('nodes')];
  await diagrams.refresh();

  calls.length = 0;
  const drawn = diagrams.drawn;
  diagrams.settings.timeline.logY = true;          // what the log y toggle does
  const pending = diagrams.refresh();
  assert(!Jobs.busy(diagrams),
    'a redraw started a job, so the bar would flash on every keystroke in the title');
  await pending;
  assert(!calls.length, 'a change of scale asked the server for the run again');
  assert(diagrams.drawn > drawn, 'the change was never drawn');
}

async function test_a_redraw_during_a_load_leaves_the_load_running() {
  const calls = [];
  const diagrams = diagramsWith(loaderFor(pretendRun(40, calls, { delay: 1 })));
  diagrams.active = 'timeline';
  diagrams.settings.timeline.lines = [lineOf('bridges')];

  const first = diagrams.refresh();
  await sleep(12);
  diagrams.settings.timeline.logY = true;
  await diagrams.refresh();
  await first;
  assert(calls.filter(c => c.points === 2).length === 1,
    'a redraw threw the load away and started it again from its first step');
}

async function test_a_frame_chart_reads_its_frames_once() {
  let reads = 0;
  const api = {
    async getFrames(id, from, count) {
      reads += 1;
      return { frames: Array.from({ length: count }, (_, i) => ({ iteration: from + i, ids: [1, 2] })) };
    }
  };
  const diagrams = diagramsWith(loaderFor(pretendRun(20)), api);
  diagrams.runs = [{ id: 'run', frame_count: 40 }];
  diagrams.runId = 'run';
  diagrams.active = 'histogram';
  await diagrams.refresh();
  const first = reads;
  assert(first > 0, 'the histogram never read its frames');

  diagrams.settings.histogram.colormap = 'magma';   // what the colour map menu does
  await diagrams.refresh();
  assert(reads === first, 'changing the colour map read the same frames again');

  diagrams.settings.histogram.iteration = 3;        // a different window
  await diagrams.refresh();
  assert(reads > first, 'moving to another iteration did not read it');
}

/**
 * Just enough of a document to build the Diagrams bar in: elements that hold
 * children, attributes and listeners, menus whose options can be written as
 * markup, text, and the before/after the menus' step buttons are put with.
 */
function fakeDocument() {
  class Element {
    constructor(tag) {
      this.tagName = tag.toUpperCase();
      this.children = [];
      this.parentNode = null;
      this.dataset = {};
      this.style = {};
      this.listeners = {};
      this.textContent = '';
      this.value = '';
      // A view of className, as a document's is, so either way of marking an
      // element reads the same.
      this.className = '';
      const names = () => this.className.split(/\s+/).filter(Boolean);
      this.classList = {
        contains: name => names().includes(name),
        add: name => { if (!names().includes(name)) this.className = [...names(), name].join(' '); },
        remove: name => { this.className = names().filter(n => n !== name).join(' '); },
        toggle: (name, on = !names().includes(name)) =>
          (on ? this.classList.add(name) : this.classList.remove(name))
      };
    }
    append(...nodes) {
      for (const node of nodes) { node.parentNode = this; this.children.push(node); }
    }
    replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
    get lastChild() { return this.children[this.children.length - 1] || null; }
    before(node) { this._beside(node, 0); }
    after(node) { this._beside(node, 1); }
    _beside(node, offset) {
      const siblings = this.parentNode.children;
      node.parentNode = this.parentNode;
      siblings.splice(siblings.indexOf(this) + offset, 0, node);
    }
    addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
    dispatchEvent(event) { for (const listener of this.listeners[event.type] || []) listener(event); return true; }
    click() { this.dispatchEvent(new Event('click')); }
    querySelectorAll(tag) {
      const found = [];
      const walk = node => node.children.forEach(child => {
        if (child.tagName === tag.toUpperCase()) found.push(child);
        walk(child);
      });
      walk(this);
      return found;
    }
    get options() { return this.querySelectorAll('option'); }
    get selectedOptions() { return this.options.filter(option => option.value === this.value); }
    set innerHTML(markup) {
      this.children = [];
      for (const [, value, text] of markup.matchAll(/<option value="([^"]*)">([^<]*)<\/option>/g)) {
        const option = new Element('option');
        option.value = value;
        option.textContent = text;
        this.append(option);
      }
    }
  }
  return {
    createElement: tag => new Element(tag),
    createTextNode: text => Object.assign(new Element('#text'), { textContent: text })
  };
}

async function test_every_diagram_control_builds_and_asks_for_what_it_changes() {
  // The bar is built by code that nothing ran outside a browser: renaming a
  // helper it called once left the whole Research tab dead at start-up, and
  // no test noticed. Here each tab's bar is built, and every control on it is
  // used, in a document just big enough to hold them.
  const saved = globalThis.document;
  globalThis.document = fakeDocument();
  try {
    const diagrams = diagramsWith(loaderFor(pretendRun(20)));
    diagrams.runs = [{ id: 'run', name: 'A run', frame_count: 40 }];
    diagrams.runId = 'run';
    diagrams.controlsEl = document.createElement('div');
    diagrams.settings.timeline.lines = [lineOf('nodes')];
    let asked = 0;
    diagrams.refresh = () => { asked += 1; };

    // Not a way of changing the chart: a picture out, the settings kept, and
    // the two fields that only say where the next constant line goes.
    const aside = el => ['Save image', 'Save settings'].includes(el.textContent)
      || el.className === 'diagram-guide'
      || (el.tagName === 'SELECT' && !el.title && el.options.length === 2
          && el.options[0].value === 'x');

    for (const tab of Object.keys(diagrams.settings)) {
      diagrams.active = tab;
      diagrams.controls();
      // Taken as built: some controls build the bar again, and each keeps to
      // the elements it was built with.
      const built = ['input', 'select', 'button']
        .flatMap(tag => diagrams.controlsEl.querySelectorAll(tag));
      // Choosing the phase a line already has is choosing nothing.
      const controls = built.filter(el => !aside(el) && el.className !== 'step-btn'
        && !(el.className.includes('seg-btn') && el.classList.contains('active')));
      assert(controls.length >= 8, `the ${tab} bar was built with ${controls.length} controls`);

      for (const el of controls) {
        const before = asked;
        if (el.tagName === 'BUTTON') {
          // A constant line needs a value first.
          if (el.textContent === 'Add line at') {
            built.find(i => i.className === 'diagram-guide').value = '3';
          }
          el.click();
        } else if (el.tagName === 'SELECT') {
          const other = el.options.find(o => o.value !== el.value);
          if (!other) continue;
          el.value = other.value;
          el.dispatchEvent(new Event('change'));
        } else {
          el.value = el.type === 'number' ? '4' : 'a title';
          el.dispatchEvent(new Event('change'));
        }
        assert(asked > before,
          `on the ${tab} bar, "${el.textContent || el.title || el.placeholder || el.tagName}" `
          + 'changed nothing the chart was asked to show');
      }
    }
  } finally {
    globalThis.document = saved;
  }
}

function test_the_chart_says_which_thesis_it_is_making() {
  // Worked out from what is plotted, not remembered from a button: a chart
  // put together by hand makes the argument as much as one filled in by it,
  // and a chart with another statistic added no longer makes it.
  const diagrams = diagramsWith(loaderFor(pretendRun(20)));
  diagrams.settings.timeline.lines = ['ricciCurvature', 'dimension'].map(lineOf);
  const found = diagrams.activeThesis();
  assert(found && found.id === 'curvature',
    `a chart of curvature and dimension is making ${found ? found.id : 'no argument'}`);
  diagrams.settings.timeline.lines.push(lineOf('nodes'));
  assert(diagrams.activeThesis() === null, 'a chart with another statistic added still claims a thesis');
}

function test_no_diagram_control_draws_without_asking_what_it_needs() {
  // The rule the four tests above hold refresh() to is only worth anything if
  // the controls go through it. Two of them once called draw() directly, which
  // was harmless until what a run loads came to depend on what is plotted.
  const source = fs.readFileSync(path.join(root, 'web', 'js', 'diagram-controls.js'), 'utf8');
  const direct = (source.match(/this\.draw\(\)/g) || []).length;
  assert(!direct, `${direct} control${direct === 1 ? '' : 's'} call draw() directly, `
    + 'and so never load what they change');
}

// ---- the canvas palette ------------------------------------------------------

function test_every_colour_a_chart_asks_for_is_in_the_stylesheet() {
  const Ink = new Function(
    `${fs.readFileSync(path.join(root, 'web', 'js', 'ink.js'), 'utf8')}; return Ink;`)();
  const css = fs.readFileSync(path.join(root, 'web', 'css', 'style.css'), 'utf8');
  const start = css.indexOf(':root');
  const palette = css.slice(start, css.indexOf('}', start));
  const missing = Object.values(Ink.ROLES)
    .filter(token => !new RegExp(`${token}\\s*:`).test(palette));
  assert(!missing.length,
    `ink.js asks the stylesheet for ${missing.join(', ')}, which :root does not define`);
}

function test_no_canvas_writes_a_colour_out() {
  // A colour written into a canvas call is one the stylesheet cannot reach.
  // Twenty-eight of them were, and when the panels were lightened they stayed
  // tuned to the old ones: the stat popup's grid went darker than its panel.
  const dir = path.join(root, 'web', 'js');
  const found = [];
  for (const name of fs.readdirSync(dir).filter(n => n.endsWith('.js'))) {
    fs.readFileSync(path.join(dir, name), 'utf8').split('\n').forEach((line, i) => {
      if (/(fillStyle|strokeStyle)\s*=\s*['"]#[0-9a-fA-F]{3,8}['"]/.test(line)) found.push(`${name}:${i + 1}`);
    });
  }
  assert(!found.length, `colours written out on a canvas, where a retheme cannot reach them: ${found.join(', ')}`);
}

// ---- the browser worker ------------------------------------------------------

function test_the_worker_cuts_frames_the_way_the_server_does() {
  // gol_server._project: a dotted name reaches one level in. The worker took
  // every name literally, so on the static site `decisions.allocations` came
  // back as a key of that name holding nothing, and Flow modules and the edge
  // flow metric found no decisions at all.
  const source = fs.readFileSync(path.join(root, 'web', 'js', 'sim-worker.js'), 'utf8');
  const body = source.match(/function project\(frame, fields\) \{[\s\S]*?\n\}\n/);
  assert(body, 'sim-worker.js no longer has project()');
  const project = new Function(`${body[0]}; return project;`)();

  const frame = { iteration: 3, ids: [1, 2], tokens: [5, 6],
                  decisions: { allocations: [{ agent: 1 }], winners: [9] } };
  const cut = project(frame, ['iteration', 'decisions.allocations', 'missing.inner']);
  assert(cut.iteration === 3, 'a plain field did not come through');
  assert(cut.decisions && cut.decisions.allocations === frame.decisions.allocations,
    'decisions.allocations did not arrive where the page reads it');
  assert(!('winners' in cut.decisions), 'the rest of decisions came along too');
  assert(!('tokens' in cut) && !('missing' in cut), 'fields nobody asked for came along');
}

const tests = Object.entries({
  test_the_world_holds_its_whole_supply_at_both_ends_of_the_game,
  test_a_pile_never_fills_before_anything_reaches_it,
  test_a_stake_lands_where_the_engine_says_it_lands,
  test_a_parent_pays_for_its_child_as_the_tokens_cross,
  test_a_disc_is_sized_by_what_is_held_now_not_at_the_end_of_the_step,
  test_the_handover_never_creates_or_strands_a_token,
  test_one_token_is_one_dot_and_the_reference_is_fifteen,
  test_no_pile_draws_more_dots_than_it_holds_tokens,
  test_the_starving_die_first_and_the_stranded_second,
  test_eyes_do_not_all_look_away_at_once,
  test_every_eye_looks_at_a_neighbour_or_itself,
  test_the_supply_schedule_is_fixed_once_the_ball_sets_off,
  test_tokens_are_fired_in_the_order_the_ball_comes_round,
  test_a_brain_stays_marked_once_its_turn_has_passed,
  test_a_conquest_does_not_show_its_answer_before_the_brain_arrives,
  test_the_conquest_waits_for_the_stakes_on_the_step_that_shows_both,
  test_the_default_look_is_stated_once,
  test_a_saved_preset_is_read_in_todays_vocabulary,
  test_a_second_job_for_an_owner_cancels_the_first,
  test_only_cancels_everything_else,
  test_a_cancelled_job_neither_reports_nor_counts_as_finished,
  test_an_abort_is_not_an_error,
  test_a_failed_job_says_so_and_is_tried_again,
  test_an_answer_after_a_stop_is_a_stop,
  test_a_history_is_asked_for_by_what_is_plotted,
  test_a_summarised_run_is_one_request,
  test_a_history_of_a_smaller_run_is_not_ready,
  test_a_history_longer_than_its_run_is_dropped,
  test_a_derived_statistic_reads_like_a_stored_one,
  test_two_statistics_pair_by_frame_or_else_by_iteration,
  test_a_history_that_fails_says_so_and_never_rejects,
  test_the_trajectory_and_the_popup_share_one_history_load,
  test_the_bar_counts_samples_and_only_moves_forward,
  test_switching_a_line_to_a_graph_statistic_loads_it,
  test_a_change_that_needs_nothing_new_draws_without_loading,
  test_a_redraw_during_a_load_leaves_the_load_running,
  test_a_frame_chart_reads_its_frames_once,
  test_no_diagram_control_draws_without_asking_what_it_needs,
  test_the_chart_says_which_thesis_it_is_making,
  test_every_diagram_control_builds_and_asks_for_what_it_changes,
  test_every_colour_a_chart_asks_for_is_in_the_stylesheet,
  test_no_canvas_writes_a_colour_out,
  test_the_worker_cuts_frames_the_way_the_server_does
}).sort(([a], [b]) => a.localeCompare(b));

// A test that is written and never listed here is worse than no test: it reads
// as coverage and runs never.
// Async tests count too: a guard that only saw `function test_` would let an
// `async function test_` be written, listed nowhere, and never run.
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
      // Awaited, so an async test that rejects is a failure rather than a
      // pass with an unhandled rejection printed somewhere after the summary.
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
