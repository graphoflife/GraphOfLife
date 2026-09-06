/*
 * FrameMetrics, on frames whose answers are known by hand.
 *
 *     node tests/test_stats.js
 *
 * stats.js is the largest file in the browser and the least covered. What
 * cover it had is tests/test_stats_parity.py, which runs `summary()` against
 * gol_series.py — forty-odd scalars, and a real guard, but only the half that
 * Python also computes.
 *
 * The other half is everything the viewer draws with: reading a metric off a
 * frame, scaling it, turning it into a position in a colour map, and looking
 * up an edge. None of it is shared with Python, so none of it was compared
 * with anything. It is also the half where being wrong is quiet — a metric
 * that falls through its switch returns a flat 0.5 for every node, and a flat
 * colour is what a legitimately uniform quantity looks like too.
 *
 * Frames here are small and hand-made rather than recorded, so the expected
 * answer can be worked out on paper and written down beside the assertion.
 *
 * Same harness as tests/test_flowmodules.js: collect every test_ function, run
 * them all, print a dot or an F, and report at the end.
 */
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const source = ['colormaps.js', 'metrics.js', 'graphstats.js', 'stats.js']
  .map(name => fs.readFileSync(path.join(root, 'web', 'js', name), 'utf8'))
  .join('\n');
// The same bridge tests/test_stats_parity.py uses: one `window` for the pixel
// ratio, and nothing else.
const load = what => new Function('window', `${source}; return ${what};`)({ devicePixelRatio: 1 });
const FrameMetrics = load('FrameMetrics');
const Metrics = load('Metrics');

// ---------------------------------------------------------------------------

const SETTINGS = {
  nodeColorBy: 'tokens', nodeSizeBy: 'tokens',
  nodeColorLog: false, nodeSizeLog: false,
  nodeColormap: 'viridis', nodeColorReverse: false,
  edgeColorBy: 'constant', edgeWidthBy: 'constant',
  edgeColorLog: false, edgeWidthLog: false
};

/**
 * Five agents in a line with a triangle at one end.
 *
 *   1 — 2 — 3 — 4 — 5
 *       \_______/          (2 — 4, closing a triangle 2-3-4)
 *
 * Tokens 10, 20, 30, 40, 50: distinct, so every ordering is unambiguous, and
 * summing to 150.
 */
function frame(over = {}) {
  const ids = [1, 2, 3, 4, 5];
  return {
    iteration: 7, phase: 2, nodes_before: 5,
    ids,
    tokens: [10, 20, 30, 40, 50],
    edges: [[1, 2], [2, 3], [3, 4], [4, 5], [2, 4]],
    brain_ids: [100, 100, 101, 102, 102],
    parent_brain_ids: [-1, -1, 100, 101, 101],
    parent_ids: [-1, 1, 2, 3, 4],
    ages: [7, 5, 3, 1, 0],
    delta: [1, -2, 3, -4, 5],
    cleanup: { starved: 0, orphaned: 0, redistributed: 0, resurrected: false },
    ...over
  };
}

const metricsOf = (f = frame(), s = SETTINGS) => new FrameMetrics(f, { ...SETTINGS, ...s });
const at = (m, id) => m.frame.ids.indexOf(id);

// ---------------------------------------------------------------------------

function test_every_registered_node_metric_is_actually_readable() {
  // The quiet failure this file exists for. A metric listed in the registry
  // but missing from the switch falls to `default: 0.5` — every node the same
  // mid-scale value, which on screen is a flat colour, which is exactly what a
  // uniform quantity looks like. Nothing errors and nothing looks broken.
  const body = FrameMetrics.prototype.nodeValues.toString();
  const missing = Metrics.list('node')
    .filter(m => !body.includes(`'${m.key}'`))
    .map(m => m.key);
  if (missing.length) {
    throw new Error(`node metrics offered in the menus with no case in `
                  + `nodeValues, so they read as a flat 0.5: ${missing.join(', ')}`);
  }
}

function test_every_registered_edge_metric_is_actually_readable() {
  // Same, one layer down: edgeValues delegates to _edgeRaw, and _edgeRaw has
  // the switch and the `default: 0`.
  const body = FrameMetrics.prototype._edgeRaw.toString();
  const missing = Metrics.list('edge')
    .filter(m => !body.includes(`'${m.key}'`))
    .map(m => m.key);
  if (missing.length) {
    throw new Error(`edge metrics offered in the menus with no case in `
                  + `_edgeRaw, so they read as a flat 0: ${missing.join(', ')}`);
  }
}

function test_every_node_metric_gives_one_number_per_node() {
  const m = metricsOf();
  for (const metric of Metrics.list('node')) {
    const values = m.nodeValues(metric.key);
    if (values.length !== m.frame.ids.length) {
      throw new Error(`${metric.key} gave ${values.length} values for `
                    + `${m.frame.ids.length} agents`);
    }
    for (const v of values) {
      if (typeof v !== 'number') throw new Error(`${metric.key} gave a ${typeof v}`);
      // NaN is allowed and meaningful — it is how "not recorded" is said — but
      // an infinity is arithmetic that went wrong.
      if (!Number.isNaN(v) && !Number.isFinite(v)) {
        throw new Error(`${metric.key} gave ${v}`);
      }
    }
  }
}

function test_every_edge_metric_gives_one_number_per_edge() {
  const m = metricsOf();
  for (const metric of Metrics.list('edge')) {
    const values = m.edgeValues(metric.key);
    if (values.length !== m.frame.edges.length) {
      throw new Error(`${metric.key} gave ${values.length} values for `
                    + `${m.frame.edges.length} links`);
    }
    for (const v of values) {
      if (!Number.isFinite(v)) throw new Error(`${metric.key} gave ${v}`);
    }
  }
}

function test_the_metrics_read_off_the_frame_say_what_the_frame_says() {
  // Worked out from the fixture at the top: tokens as given, degree from the
  // drawn graph, ages as given, ids in order.
  const m = metricsOf();
  const expect = {
    tokens: [10, 20, 30, 40, 50],
    degree: [1, 3, 2, 3, 1],            // 1—2, 2—{1,3,4}, 3—{2,4}, 4—{3,5,2}, 5—4
    token_delta: [1, -2, 3, -4, 5],
    abs_token_delta: [1, 2, 3, 4, 5],
    age: [7, 5, 3, 1, 0],
    node_id: [1, 2, 3, 4, 5],
    brain_id: [100, 100, 101, 102, 102],
    triangles: [0, 1, 1, 1, 0]          // only 2-3-4 closes
  };
  for (const [key, want] of Object.entries(expect)) {
    const got = Array.from(m.nodeValues(key));
    if (JSON.stringify(got) !== JSON.stringify(want)) {
      throw new Error(`${key} read ${JSON.stringify(got)}, wanted ${JSON.stringify(want)}`);
    }
  }
  const share = Array.from(m.nodeValues('token_share'));
  if (Math.abs(share.reduce((a, b) => a + b, 0) - 1) > 1e-9) {
    throw new Error(`shares of the total came to ${share.reduce((a, b) => a + b, 0)}`);
  }
}

function test_the_edge_metrics_say_what_the_two_endpoints_say() {
  // Symmetry alone is not enough to pin these down — min and max are both
  // symmetric, so swapping one for the other passes a symmetry check and
  // draws the wrong width on every link. Worked out from the fixture:
  //
  //   links   1—2   2—3   3—4   4—5   2—4
  //   tokens 10,20 20,30 30,40 40,50 20,40
  //   degree  1,3   3,2   2,3   3,1   3,3
  //
  // and the only triangle is 2-3-4, so its three links are on a loop and the
  // two leaf links are not.
  const m = metricsOf();
  const expect = {
    avg_tokens: [15, 25, 35, 45, 30],
    min_tokens: [10, 20, 30, 40, 20],
    max_tokens: [20, 30, 40, 50, 40],
    token_gap:  [10, 10, 10, 10, 20],
    avg_degree: [2, 2.5, 2.5, 2, 3],
    min_degree: [1, 2, 2, 1, 3],
    max_degree: [3, 3, 3, 3, 3],
    triangles:  [0, 1, 1, 0, 1],
    bridge:     [1, 0, 0, 1, 0]
  };
  for (const [key, want] of Object.entries(expect)) {
    const got = Array.from(m.edgeValues(key));
    if (JSON.stringify(got) !== JSON.stringify(want)) {
      throw new Error(`${key} read ${JSON.stringify(got)}, wanted ${JSON.stringify(want)}`);
    }
  }
}

function test_an_endpoint_metric_does_not_depend_on_which_end_you_name() {
  // Every edge metric here is a function of the two endpoints and nothing
  // else, so naming them the other way round has to give the same answer. An
  // asymmetry would mean the drawn value depended on the order the engine
  // happened to record the link in.
  const m = metricsOf();
  for (const metric of Metrics.list('edge')) {
    for (const [a, b] of m.frame.edges) {
      const forward = m._edgeRaw(metric.key, a, b);
      const backward = m._edgeRaw(metric.key, b, a);
      if (forward !== backward) {
        throw new Error(`${metric.key} on ${a}—${b} is ${forward} one way and `
                      + `${backward} the other`);
      }
    }
  }
}

function test_an_edge_is_found_from_either_end_and_a_missing_one_is_not_invented() {
  const m = metricsOf();
  m.frame.edges.forEach(([a, b], e) => {
    if (m.edgeSlot(a, b) !== e) throw new Error(`${a}—${b} looked up as ${m.edgeSlot(a, b)}, wanted ${e}`);
    if (m.edgeSlot(b, a) !== e) throw new Error(`${b}—${a} did not find the same link`);
  });
  if (m.edgeSlot(1, 5) >= 0) throw new Error('a link that is not in the frame was found anyway');
  if (m.edgeSlot(1, 999) >= 0) throw new Error('a link to an agent that does not exist was found');
}

function test_a_metric_with_nothing_behind_it_says_so_rather_than_showing_a_number() {
  // A run recorded before a field existed gives NaN for every node. The range
  // must not be dragged anywhere by that, and the key has to say the metric is
  // absent instead of printing a confident 0 to 1 over nodes all painted at
  // mid-scale.
  const older = frame();
  delete older.ages;
  const m = metricsOf(older, { nodeColorBy: 'age' });

  if (m.hasValues('age')) throw new Error('a frame with no ages claimed to have some');
  if (!m.hasValues('tokens')) throw new Error('tokens went missing');
  if (!Array.from(m.nodeValues('age')).every(Number.isNaN)) {
    throw new Error('a missing metric produced numbers');
  }
  if (m.colorRangeText[0] !== 'not recorded') {
    throw new Error(`the key reads "${m.colorRangeText.join(' … ')}" for a metric `
                  + `with no values behind it`);
  }

  // And one gap among real values is dropped rather than dragging the range.
  const mixed = metricsOf(frame({ ages: [-1, 5, 3, 1, 0] }), { nodeColorBy: 'age' });
  const [lo, hi] = mixed.nodeRange('age', false);
  if (lo !== 0 || hi !== 5) {
    throw new Error(`one unknown age pulled the range to ${lo}–${hi}, wanted 0–5`);
  }
}

function test_a_position_in_the_colour_map_stays_between_its_ends() {
  const norm = FrameMetrics._norm;
  if (norm(10, [10, 50]) !== 0) throw new Error('the bottom of the range is not 0');
  if (norm(50, [10, 50]) !== 1) throw new Error('the top of the range is not 1');
  if (norm(30, [10, 50]) !== 0.5) throw new Error('the middle of the range is not 0.5');
  if (norm(-999, [10, 50]) !== 0) throw new Error('below the range did not clamp');
  if (norm(999, [10, 50]) !== 1) throw new Error('above the range did not clamp');
  // A gap sits in the middle rather than at an end, so it cannot be mistaken
  // for an extreme.
  if (norm(NaN, [10, 50]) !== 0.5) throw new Error('a gap was not put at mid-scale');
}

function test_a_range_over_nothing_or_over_one_value_is_still_usable() {
  const m = metricsOf();
  const flat = m._rangeLinear([4, 4, 4]);
  if (!(flat[1] > flat[0])) {
    throw new Error(`a constant metric gave the empty range ${flat}, which divides by zero`);
  }
  const none = m._rangeLinear([NaN, NaN]);
  if (none[0] !== 0 || none[1] !== 1) throw new Error(`all gaps gave ${none}, wanted 0–1`);
  const some = m._rangeLinear([NaN, 3, NaN, 9]);
  if (some[0] !== 3 || some[1] !== 9) throw new Error(`gaps were counted: ${some}`);
}

function test_a_log_scale_can_be_undone() {
  for (const v of [0, 1, 5, 100, 9999]) {
    const back = Metrics.undoLog(Metrics.applyLog(v, false), false);
    if (Math.abs(back - v) > 1e-6) throw new Error(`${v} came back as ${back}`);
  }
  // A signed quantity keeps its direction: compressing a loss must not turn it
  // into a gain, which is what makes the middle of a diverging map mean zero.
  for (const v of [-500, -1, 0, 1, 500]) {
    const there = Metrics.applyLog(v, true);
    if (Math.sign(there) !== Math.sign(v)) throw new Error(`${v} changed sign to ${there}`);
    const back = Metrics.undoLog(there, true);
    if (Math.abs(back - v) > 1e-6) throw new Error(`${v} came back as ${back}`);
  }
}

function test_a_log_scale_keeps_the_order_it_was_given() {
  // The only thing a colour map needs from a scale: if one agent held more
  // than another, it must still read as more afterwards.
  const m = metricsOf();
  const raw = Array.from(m.nodeValues('tokens'));
  const scaled = Array.from(m.scaledNodeValues('tokens', true));
  for (let i = 1; i < raw.length; i++) {
    if ((raw[i] > raw[i - 1]) !== (scaled[i] > scaled[i - 1])) {
      throw new Error(`agents ${i - 1} and ${i} swapped order under the log scale`);
    }
  }
}

function test_wealth_rank_counts_from_the_richest() {
  const m = metricsOf();
  if (m.wealthRank(at(m, 5)) !== 1) throw new Error('the richest agent is not first');
  if (m.wealthRank(at(m, 1)) !== 5) throw new Error('the poorest agent is not last');
  const ranks = m.frame.ids.map((_, i) => m.wealthRank(i)).sort((a, b) => a - b);
  if (JSON.stringify(ranks) !== JSON.stringify([1, 2, 3, 4, 5])) {
    throw new Error(`ranks were ${ranks}, which is not one of each`);
  }
}

function test_the_hover_card_describes_the_agent_it_was_asked_about() {
  const m = metricsOf();
  const i = at(m, 4);
  const d = m.nodeDetail(i);
  if (d.id !== 4) throw new Error(`asked about agent 4 and got ${d.id}`);
  if (d.tokens !== 40) throw new Error(`agent 4 holds 40, the card says ${d.tokens}`);
  if (d.degree !== 3) throw new Error(`agent 4 has 3 links, the card says ${d.degree}`);
  if (d.delta !== -4) throw new Error(`agent 4 changed by -4, the card says ${d.delta}`);
  if (d.spawnedBy !== 3) throw new Error(`agent 4's parent is 3, the card says ${d.spawnedBy}`);
  if (!d.hasDelta) throw new Error('a frame with deltas said it had none');

  // A founder has no parent, and -1 is a sentinel rather than an agent.
  if (m.nodeDetail(at(m, 1)).spawnedBy !== null) {
    throw new Error('a founder was given agent -1 as its parent');
  }
  const older = frame();
  delete older.delta;
  if (metricsOf(older).nodeDetail(0).hasDelta) {
    throw new Error('a frame recorded before deltas existed claimed to have them');
  }
}

function test_what_crossed_each_link_is_read_from_the_decisions() {
  // The bug the parity test was written for went the other way — an
  // optimisation began skipping agents culled during cleanup, so the totals
  // disagreed. Here: a stake between two agents is flow on the link between
  // them, a stake on yourself is not, and a stake at somebody no longer in the
  // frame is not lost silently but simply not drawn.
  const f = frame({
    decisions: {
      allocations: [
        { agent: 1, targets: [1, 2], alloc: [3, 7] },     // 3 stays home
        { agent: 2, targets: [3], alloc: [5] },
        { agent: 3, targets: [2], alloc: [4] },           // back the other way
        { agent: 5, targets: [99], alloc: [6] }           // a agent that is gone
      ]
    }
  });
  const m = metricsOf(f);
  if (!m.hasFlow) throw new Error('a frame with allocations reported no flow');

  if (m._edgeRaw('flow', 1, 2) !== 7) {
    throw new Error(`1—2 carried ${m._edgeRaw('flow', 1, 2)}, wanted the 7 that was staked`);
  }
  // 2—3 carries both directions: five one way and four the other.
  if (m._edgeRaw('flow', 2, 3) !== 9) {
    throw new Error(`2—3 carried ${m._edgeRaw('flow', 2, 3)}, wanted 5 + 4`);
  }
  if (m._edgeRaw('flow', 3, 4) !== 0) {
    throw new Error('a link nobody staked across carried something');
  }
  const total = Array.from(m.edgeValues('flow')).reduce((a, b) => a + b, 0);
  if (total !== 16) throw new Error(`${total} crossed the links; 7 + 5 + 4 is what was `
                                  + `staked between agents that are both here`);
}

function test_a_frame_without_decisions_has_no_flow_rather_than_zero_flow() {
  const m = metricsOf();
  if (m.hasFlow) throw new Error('a frame with no decisions claimed to know what crossed');
  if (Array.from(m.edgeValues('flow')).some(v => v !== 0)) {
    throw new Error('flow was invented for a frame that never recorded any');
  }
}

function test_a_colour_comes_out_of_the_map_for_every_agent() {
  const m = metricsOf();
  m.frame.ids.forEach((id, i) => {
    const css = m.nodeColorCssByIndex(i, 1);
    if (!/^rgba?\(/.test(css)) throw new Error(`agent ${id} was given "${css}"`);
    const byId = m.nodeColorCss(id, 1);
    if (byId !== css) throw new Error(`agent ${id} is one colour by index and another by id`);
  });
}

function test_the_key_reads_in_the_units_that_were_chosen() {
  const linear = metricsOf(frame(), { nodeColorBy: 'tokens', nodeColorLog: false });
  if (linear.colorRangeText.join('–') !== '10–50') {
    throw new Error(`the key reads ${linear.colorRangeText.join('–')}, wanted 10–50`);
  }
  // On a log scale the key still has to say tokens, not logs of tokens.
  const log = metricsOf(frame(), { nodeColorBy: 'tokens', nodeColorLog: true });
  if (log.colorRangeText.join('–') !== '10–50') {
    throw new Error(`under a log scale the key reads ${log.colorRangeText.join('–')}, `
                  + `which is not in tokens`);
  }
  if (!log.colorLabel.includes('log')) throw new Error('a log scale is not labelled as one');
}

// ---------------------------------------------------------------------------

const tests = Object.entries({
  test_every_registered_node_metric_is_actually_readable,
  test_every_registered_edge_metric_is_actually_readable,
  test_every_node_metric_gives_one_number_per_node,
  test_every_edge_metric_gives_one_number_per_edge,
  test_the_metrics_read_off_the_frame_say_what_the_frame_says,
  test_the_edge_metrics_say_what_the_two_endpoints_say,
  test_an_endpoint_metric_does_not_depend_on_which_end_you_name,
  test_an_edge_is_found_from_either_end_and_a_missing_one_is_not_invented,
  test_a_metric_with_nothing_behind_it_says_so_rather_than_showing_a_number,
  test_a_position_in_the_colour_map_stays_between_its_ends,
  test_a_range_over_nothing_or_over_one_value_is_still_usable,
  test_a_log_scale_can_be_undone,
  test_a_log_scale_keeps_the_order_it_was_given,
  test_wealth_rank_counts_from_the_richest,
  test_the_hover_card_describes_the_agent_it_was_asked_about,
  test_what_crossed_each_link_is_read_from_the_decisions,
  test_a_frame_without_decisions_has_no_flow_rather_than_zero_flow,
  test_a_colour_comes_out_of_the_map_for_every_agent,
  test_the_key_reads_in_the_units_that_were_chosen
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
