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

const tests = Object.entries({
  test_the_world_holds_its_whole_supply_at_both_ends_of_the_game,
  test_a_pile_never_fills_before_anything_reaches_it,
  test_a_stake_lands_where_the_engine_says_it_lands,
  test_a_parent_pays_for_its_child_as_the_tokens_cross,
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
  test_the_conquest_waits_for_the_stakes_on_the_step_that_shows_both
}).sort(([a], [b]) => a.localeCompare(b));

// A test that is written and never listed here is worse than no test: it reads
// as coverage and runs never.
const written = (fs.readFileSync(__filename, 'utf8').match(/^function test_/gm) || []).length;
if (written !== tests.length) {
  console.error(`${written} tests are written and ${tests.length} are listed to run`);
  process.exit(1);
}

const failures = [];
const started = Date.now();
for (const [name, fn] of tests) {
  try {
    fn();
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
