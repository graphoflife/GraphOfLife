/*
 * The flow modules, checked against graphs whose answer is known by hand.
 *
 *     node tests/test_flowmodules.js
 *
 * Same shape as tests/test_engine.py: collect every test_ function, run them
 * all, print a dot or an F, and report at the end.
 */
const fs = require('fs');
const path = require('path');

const source = fs.readFileSync(
  path.join(__dirname, '..', 'web', 'js', 'flowmodules.js'), 'utf8');
const FlowModules = new Function(`${source}; return FlowModules;`)();

// ---------------------------------------------------------------------------

function clique(offset, size, weight = 10) {
  const edges = [];
  for (let a = 0; a < size; a++) {
    for (let b = a + 1; b < size; b++) edges.push([offset + a, offset + b, weight]);
  }
  return edges;
}

function labelsOf(module, from, count) {
  return new Set(Array.from({ length: count }, (_, i) => module[from + i]));
}

// ---------------------------------------------------------------------------

function test_plogp_is_zero_at_zero() {
  // The map equation is a sum of these, and modules with no exit flow are
  // ordinary. If this returned NaN every code length downstream would be NaN.
  if (FlowModules.plogp(0) !== 0) throw new Error('plogp(0) should be 0');
  if (Math.abs(FlowModules.plogp(0.5) + 0.5) > 1e-12) {
    throw new Error('plogp(0.5) should be -0.5');
  }
}

function test_two_cliques_joined_by_a_thread_are_two_modules() {
  // The case the method exists for: flow that stays inside two groups, with
  // barely any crossing between them.
  const edges = [...clique(0, 6), ...clique(6, 6), [0, 6, 1]];
  const found = FlowModules.partition(12, edges);

  const left = labelsOf(found.module, 0, 6);
  const right = labelsOf(found.module, 6, 6);
  if (left.size !== 1) throw new Error(`the first clique split into ${left.size}`);
  if (right.size !== 1) throw new Error(`the second clique split into ${right.size}`);
  if ([...left][0] === [...right][0]) {
    throw new Error('the two cliques were put in one module');
  }
  if (!(found.codeLength < found.baseline)) {
    throw new Error(`splitting cost ${found.codeLength} bits against ${found.baseline} `
                  + `for no split; a real split has to be shorter`);
  }
}

function test_one_clique_is_one_module() {
  // Nothing to find. The method must not invent structure in flow that has
  // none, which is the failure mode that makes a module count meaningless.
  const found = FlowModules.partition(8, clique(0, 8));
  if (found.modules !== 1) {
    throw new Error(`a single clique was split into ${found.modules} modules`);
  }
}

function test_structureless_flow_barely_compresses() {
  // A ring, where a walk has no reason to linger anywhere. Whatever partition
  // comes out, it should not claim to have found much.
  const edges = [];
  const n = 40;
  for (let a = 0; a < n; a++) edges.push([a, (a + 1) % n, 1]);
  const ring = FlowModules.partition(n, edges);

  const two = FlowModules.partition(12, [...clique(0, 6), ...clique(6, 6), [0, 6, 1]]);
  const ringSaving = (ring.baseline - ring.codeLength) / ring.baseline;
  const cliqueSaving = (two.baseline - two.codeLength) / two.baseline;
  if (!(cliqueSaving > ringSaving)) {
    throw new Error(`a ring compressed by ${ringSaving.toFixed(3)} and two cliques `
                  + `by ${cliqueSaving.toFixed(3)}; the cliques must do better`);
  }
}

function test_an_agent_nobody_trades_with_is_in_no_module() {
  // Isolated agents would otherwise each count as a module and inflate every
  // module count in the interface.
  const found = FlowModules.partition(7, clique(0, 6));
  if (found.module[6] !== -1) throw new Error('an agent with no flow got a module');
  if (found.isolated !== 1) throw new Error(`counted ${found.isolated} isolated, wanted 1`);
}

function test_flow_comes_from_the_allocations_and_ignores_self_stakes() {
  const frame = {
    ids: [10, 11, 12],
    decisions: {
      allocations: [
        { agent: 10, targets: [10, 11], alloc: [7, 5] },   // 7 on itself
        { agent: 11, targets: [11, 12], alloc: [1, 4] },
        { agent: 12, targets: [12], alloc: [9] }           // nothing but itself
      ]
    }
  };
  const { edges, hasFlow } = FlowModules.flowOf(frame);
  if (!hasFlow) throw new Error('a frame with allocations should report flow');
  const weight = new Map(edges.map(([a, b, w]) => [`${a},${b}`, w]));
  if (weight.get('0,1') !== 5) throw new Error('10 -> 11 should carry 5');
  if (weight.get('1,2') !== 4) throw new Error('11 -> 12 should carry 4');
  if (edges.length !== 2) throw new Error(`self-stakes leaked in: ${edges.length} edges`);

  const bare = FlowModules.flowOf({ ids: [1, 2], edges: [[1, 2]] });
  if (bare.hasFlow) throw new Error('a frame without decisions has no flow to read');
}

function test_a_module_keeps_its_name_while_its_members_are_replaced() {
  // The point of L1. A group that swaps out a member at a time is one group
  // walking, not a new group each iteration — which is the only way a pattern
  // whose matter turns over can be followed at all.
  let previous = new Map([[0, new Set([1, 2, 3, 4])]]);
  const walked = [
    new Set([2, 3, 4, 5]),
    new Set([3, 4, 5, 6]),
    new Set([4, 5, 6, 7])
  ];
  for (const members of walked) {
    const matched = FlowModules.track(previous, new Map([['x', members]]));
    const carried = matched.get('x');
    if (!carried || carried.id !== 0) {
      throw new Error('a module that shifted by one member was treated as new');
    }
    previous = new Map([[carried.id, members]]);
  }

  // A group sharing nothing is a different group, however similar in size.
  const stranger = FlowModules.track(previous, new Map([['y', new Set([90, 91, 92, 93])]]));
  if (stranger.get('y')) throw new Error('a module sharing no members was matched anyway');
}

function test_turnover_is_reported_for_a_module_that_persists() {
  // Long life plus high turnover is the signature worth hunting: a pattern
  // being carried by different agents each time.
  const frames = [];
  for (let t = 0; t < 6; t++) {
    // Two groups of five, each shifting one member along per iteration.
    const left = [0, 1, 2, 3, 4].map(i => 100 + i + t);
    const right = [0, 1, 2, 3, 4].map(i => 200 + i + t);
    const ids = [...left, ...right];
    const allocations = [];
    const join = (group) => {
      for (const a of group) {
        for (const b of group) if (a !== b) {
          allocations.push({ agent: a, targets: [b], alloc: [10] });
        }
      }
    };
    join(left);
    join(right);
    allocations.push({ agent: left[0], targets: [right[0]], alloc: [1] });
    frames.push({ iteration: t, phase: 2, ids, decisions: { allocations } });
  }

  const { history } = FlowModules.follow(frames);
  const facts = FlowModules.summarise(history);
  if (!facts) throw new Error('nothing came back');
  if (facts.longestLife < 5) {
    throw new Error(`the longest module lived ${facts.longestLife} frames of 6`);
  }
  if (!(facts.meanTurnover > 0.05)) {
    throw new Error(`turnover came out ${facts.meanTurnover.toFixed(3)}; a group `
                  + `shifting a member each time should show some`);
  }
  if (!(facts.compression > 0)) {
    throw new Error('two well-separated groups should compress');
  }
}

function test_frames_without_decisions_are_counted_not_guessed() {
  const { history, withoutFlow } = FlowModules.follow([
    { iteration: 0, phase: 2, ids: [1, 2], edges: [[1, 2]] },
    { iteration: 1, phase: 2, ids: [1, 2], edges: [[1, 2]] }
  ]);
  if (history.length) throw new Error('modules were invented without flow to read');
  if (withoutFlow !== 2) throw new Error(`counted ${withoutFlow} frames without flow`);
}

// ---------------------------------------------------------------------------

const tests = Object.entries({
  test_plogp_is_zero_at_zero,
  test_two_cliques_joined_by_a_thread_are_two_modules,
  test_one_clique_is_one_module,
  test_structureless_flow_barely_compresses,
  test_an_agent_nobody_trades_with_is_in_no_module,
  test_flow_comes_from_the_allocations_and_ignores_self_stakes,
  test_a_module_keeps_its_name_while_its_members_are_replaced,
  test_turnover_is_reported_for_a_module_that_persists,
  test_frames_without_decisions_are_counted_not_guessed
}).sort(([a], [b]) => a.localeCompare(b));

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
