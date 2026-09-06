/*
 * The Explanation's anchors, against the script they point into.
 *
 *     node tests/test_explain_anchors.js
 *
 * The walk-through finds the lines it lights by searching explain_minimal.py
 * for text held in explain.js, so that editing the script cannot silently
 * point a step at the wrong lines. Nothing checked that the anchors still
 * found anything, and one of them had already drifted: step 8's tail matched
 * an early `return hegemon` eight columns in, and the region stopped short of
 * the mob-walking loop the step's own words describe. It had been wrong since
 * the anchor was written, on a page shipped to the public, and no test, no
 * error and nothing on screen said so.
 *
 * This calls the real Explain.regions rather than a copy of it, so the thing
 * under test is the thing that ships.
 *
 * Same harness as tests/test_flowmodules.js: collect every test_ function, run
 * them all, print a dot or an F, and report at the end.
 */
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
// explain.js is a plain top-level `const Explain = {...}` and reaches for no
// global while it is being defined, so it loads with nothing stubbed.
const Explain = new Function(
  `${fs.readFileSync(path.join(root, 'web', 'js', 'explain.js'), 'utf8')}; return Explain;`)();
const lines = fs.readFileSync(path.join(root, 'explain_minimal.py'), 'utf8').split('\n');

/** Every anchor pair in the walk-through, flattened, with where it came from. */
function pairs() {
  const out = [];
  Explain.STEPS.forEach((step, i) => {
    const coded = Array.isArray(step.code[0]) ? step.code : [step.code];
    coded.forEach(([head, tail], k) => {
      out.push({ where: `step ${i + 1} (${step.title})${coded.length > 1 ? ` region ${k + 1}` : ''}`,
                 head, tail });
    });
  });
  return out;
}

/** Resolve a step the way the page does, on the real script. */
function resolve(step) {
  // `lines` last: the page fills it after fetching the script, and the
  // module's own empty default would otherwise win and every step resolve
  // to nothing.
  return Explain.regions.call({ ...Explain, lines }, step);
}

/** Line numbers, 1-based, where an anchor matches under the shipped rule. */
function hits(anchor) {
  const at = [];
  lines.forEach((line, i) => {
    // Mirrors Explain.regions' matcher. Kept in step with it by
    // test_the_matcher_here_is_the_one_that_ships below.
    const match = /^\s/.test(anchor) ? line.startsWith(anchor) : line.includes(anchor);
    if (match) at.push(i + 1);
  });
  return at;
}

// A region longer than this is not pointing at anything a reader can hold.
// The 146-line region that prompted two-region support was three times it.
const LONGEST_SENSIBLE_REGION = 60;

// ---------------------------------------------------------------------------

function test_every_anchor_still_finds_its_line() {
  // The whole reason the anchors are text: the script can be edited freely,
  // and this is what notices when an edit moved the ground under a step.
  const missing = [];
  for (const { where, head, tail } of pairs()) {
    if (!hits(head).length) missing.push(`${where}: head ${JSON.stringify(head)}`);
    if (!hits(tail).length) missing.push(`${where}: tail ${JSON.stringify(tail)}`);
  }
  if (missing.length) {
    throw new Error(`anchors that match nothing in explain_minimal.py:\n  `
                  + missing.join('\n  '));
  }
}

function test_every_step_resolves_to_as_many_regions_as_it_names() {
  // A step naming two regions and resolving one is the silent half-failure:
  // the page lights the first and says nothing about the second.
  const short = [];
  for (const step of Explain.STEPS) {
    const wanted = Array.isArray(step.code[0]) ? step.code.length : 1;
    const got = resolve(step).length;
    if (got !== wanted) short.push(`${step.title}: named ${wanted}, resolved ${got}`);
  }
  if (short.length) throw new Error(`steps that did not resolve:\n  ${short.join('\n  ')}`);
}

function test_a_head_picks_out_one_line_in_the_whole_script() {
  // A head starts the search, so a second match anywhere means the region's
  // position depends on which one happens to come first in the file.
  const many = [];
  for (const { where, head } of pairs()) {
    const at = hits(head);
    if (at.length > 1) many.push(`${where}: head ${JSON.stringify(head)} at ${at}`);
  }
  if (many.length) throw new Error(`heads matching more than one line:\n  ${many.join('\n  ')}`);
}

function test_a_tail_picks_out_one_line_after_its_head() {
  // This is the check that catches the drift. A tail need not be unique in the
  // whole file — only after its own head, which is the window the resolver
  // actually searches. Step 8's tail matched twice in that window and the
  // region ended at the wrong one.
  const many = [];
  for (const { where, head, tail } of pairs()) {
    const from = hits(head)[0];
    if (from === undefined) continue;              // reported by another test
    const after = hits(tail).filter(n => n > from);
    if (after.length > 1) {
      many.push(`${where}: tail ${JSON.stringify(tail)} at ${after} — `
              + `the region ends at ${after[0]} and stops short of ${after[after.length - 1]}`);
    }
  }
  if (many.length) {
    throw new Error(`tails matching more than one line after their head:\n  ${many.join('\n  ')}`);
  }
}

function test_no_region_is_too_long_to_read() {
  const huge = [];
  for (const step of Explain.STEPS) {
    for (const { from, to } of resolve(step)) {
      const size = to - from + 1;
      if (size > LONGEST_SENSIBLE_REGION) huge.push(`${step.title}: ${size} lines`);
    }
  }
  if (huge.length) {
    throw new Error(`regions longer than ${LONGEST_SENSIBLE_REGION} lines:\n  `
                  + huge.join('\n  '));
  }
}

function test_regions_run_forwards_and_land_inside_the_script() {
  for (const step of Explain.STEPS) {
    for (const { from, to } of resolve(step)) {
      if (!(to >= from)) throw new Error(`${step.title}: region ends before it starts`);
      if (from < 0 || to >= lines.length) {
        throw new Error(`${step.title}: region ${from}–${to} is outside a `
                      + `${lines.length}-line script`);
      }
    }
  }
}

function test_the_blotto_step_reaches_the_revolution() {
  // The specific thing that was broken, named so that a regression reads as
  // itself rather than as an anonymous count. `resolve` decides who takes a
  // node, and the mob walk is the half that makes revolutions possible; the
  // step's text describes it, so the lit region has to contain it.
  const step = Explain.STEPS.find(s => s.title === 'Colonel Blotto Game');
  if (!step) throw new Error('the Colonel Blotto step has been renamed');
  const regions = resolve(step);
  if (regions.length !== 2) throw new Error(`the step resolved ${regions.length} regions, wanted 2`);

  const lit = lines.slice(regions[1].from, regions[1].to + 1).join('\n');
  for (const needed of ['def resolve(', 'while i < len(mob)', 'return random.choice(rung)']) {
    if (!lit.includes(needed)) {
      throw new Error(`the region for resolve() stops before ${JSON.stringify(needed)}; `
                    + `it covers lines ${regions[1].from + 1}–${regions[1].to + 1}`);
    }
  }
}

function test_the_matcher_here_is_the_one_that_ships() {
  // hits() above reimplements the rule so it can count matches, which the page
  // never needs to do. If the two drift, every uniqueness check above is
  // measuring something the page does not do. So: the region this file
  // computes has to be the region Explain.regions computes.
  for (const step of Explain.STEPS) {
    const coded = Array.isArray(step.code[0]) ? step.code : [step.code];
    const mine = coded.map(([head, tail]) => {
      const from = hits(head)[0];
      if (from === undefined) return null;
      const to = hits(tail).find(n => n > from);
      return to === undefined ? null : { from: from - 1, to: to - 1 };
    }).filter(Boolean);

    const theirs = resolve(step);
    if (JSON.stringify(mine) !== JSON.stringify(theirs)) {
      throw new Error(`${step.title}: this test computes ${JSON.stringify(mine)} and `
                    + `Explain.regions computes ${JSON.stringify(theirs)} — the rule in `
                    + `hits() no longer matches the one that ships`);
    }
  }
}

// ---------------------------------------------------------------------------

const tests = Object.entries({
  test_every_anchor_still_finds_its_line,
  test_every_step_resolves_to_as_many_regions_as_it_names,
  test_a_head_picks_out_one_line_in_the_whole_script,
  test_a_tail_picks_out_one_line_after_its_head,
  test_no_region_is_too_long_to_read,
  test_regions_run_forwards_and_land_inside_the_script,
  test_the_blotto_step_reaches_the_revolution,
  test_the_matcher_here_is_the_one_that_ships
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
