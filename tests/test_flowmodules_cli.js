/*
 * Tests of tools/flowmodules_cli.js, the command line the book measures flow
 * modules with: it must hand the page's arithmetic the frames it is given,
 * and its nulls must keep what they say they keep.
 *
 *     node tests/test_flowmodules_cli.js
 */
const { spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const CLI = path.join(__dirname, '..', 'tools', 'flowmodules_cli.js');

/** Frames in which the same four groups of six trade only among themselves, game after game. */
function steadyGroups(frames = 8, groups = 4, size = 6) {
  const ids = Array.from({ length: groups * size }, (_, i) => 100 + i);
  const out = [];
  for (let t = 0; t < frames; t++) {
    const allocations = [];
    for (let g = 0; g < groups; g++) {
      const members = ids.slice(g * size, (g + 1) * size);
      for (const agent of members) {
        const targets = members.filter(other => other !== agent);
        allocations.push({ agent, targets, alloc: targets.map(() => 1) });
      }
    }
    out.push({ iteration: 1000 + t, phase: 2, ids, decisions: { allocations } });
  }
  return out;
}

function run(frames, args) {
  const input = frames.map(frame => JSON.stringify(frame)).join('\n');
  const done = spawnSync(process.execPath, [CLI, ...args], { input, encoding: 'utf8' });
  if (done.status !== 0) throw new Error(`flowmodules_cli.js ${args.join(' ')}: ${done.stderr}`);
  return JSON.parse(done.stdout);
}

function test_steady_groups_are_followed_through_every_frame() {
  const result = run(steadyGroups(), []);
  if (result.modulesPerFrame.some(n => n !== 4)) {
    throw new Error(`four groups, but frames held ${result.modulesPerFrame.join(', ')} modules`);
  }
  if (result.lives['8'] !== 4) {
    throw new Error(`each of the four groups lives all eight frames; lives were ${JSON.stringify(result.lives)}`);
  }
}

function test_shuffled_modules_keep_their_sizes_but_not_their_members() {
  const frames = steadyGroups();
  const real = run(frames, []);
  const dealt = run(frames, ['--shuffle', '1']);
  if (JSON.stringify(dealt.sizes) !== JSON.stringify(real.sizes)) {
    throw new Error(`shuffling must keep the modules' sizes: ${JSON.stringify(dealt.sizes)} against ${JSON.stringify(real.sizes)}`);
  }
  if (JSON.stringify(dealt.modulesPerFrame) !== JSON.stringify(real.modulesPerFrame)) {
    throw new Error('shuffling must keep the number of modules in every frame');
  }
  if ((dealt.lives['8'] || 0) >= real.lives['8']) {
    throw new Error(`groups dealt at random should not all live eight frames: ${JSON.stringify(dealt.lives)}`);
  }
  if (!dealt.shuffled || real.shuffled) throw new Error('the output must say whether its modules were shuffled');
}

function test_the_same_shuffle_seed_gives_the_same_answer() {
  const frames = steadyGroups();
  const a = run(frames, ['--shuffle', '7']);
  const b = run(frames, ['--shuffle', '7']);
  if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error('a null must be the same null every time');
}

function test_partitions_name_every_agents_module_in_every_frame() {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'flowmodules-')), 'partitions.jsonl');
  const frames = steadyGroups();
  run(frames, ['--partitions', file]);
  const rows = fs.readFileSync(file, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  if (rows.length !== frames.length) throw new Error(`${rows.length} partitions for ${frames.length} frames`);
  for (const row of rows) {
    if (row.ids.length !== row.module.length) throw new Error('every agent needs a module, or -1');
    const counts = {};
    for (const label of row.module) counts[label] = (counts[label] || 0) + 1;
    if (Object.values(counts).some(n => n !== 6) || Object.keys(counts).length !== 4) {
      throw new Error(`frame ${row.iteration} should hold four modules of six: ${JSON.stringify(counts)}`);
    }
    // Agents of one group share a module, whatever the labels are called.
    for (let g = 0; g < 4; g++) {
      const labels = new Set(row.module.slice(g * 6, (g + 1) * 6));
      if (labels.size !== 1) throw new Error(`group ${g} was split in frame ${row.iteration}`);
    }
  }
}

function test_a_module_that_keeps_its_members_has_not_outlasted_them() {
  const result = run(steadyGroups(), ['--outlast']);
  const five = result.outlast['5'];
  if (five.modules !== 4 || five.renewed !== 0 || five.stayed !== 1) {
    throw new Error(`steady groups keep every first member: ${JSON.stringify(five)}`);
  }
}

function test_a_module_whose_members_are_replaced_one_by_one_outlasts_them() {
  // One group of six; each game its oldest member is gone and a newcomer joins.
  // After six games none of the first six is left, and the group goes on.
  const frames = [];
  for (let t = 0; t < 8; t++) {
    const members = Array.from({ length: 6 }, (_, i) => 100 + t + i);
    const allocations = members.map(agent => {
      const targets = members.filter(other => other !== agent);
      return { agent, targets, alloc: targets.map(() => 1) };
    });
    frames.push({ iteration: 1000 + t, phase: 2, ids: members, decisions: { allocations } });
  }
  const result = run(frames, ['--outlast']);
  if (result.lives['8'] !== 1) throw new Error(`the group should be followed all eight games: ${JSON.stringify(result.lives)}`);
  const five = result.outlast['5'];
  if (five.modules !== 1 || five.renewed !== 1 || five.stayed !== 0) {
    throw new Error(`the group outlasted all its first members: ${JSON.stringify(five)}`);
  }
}

function test_dealing_among_neighbours_changes_nothing_where_everyone_splits_evenly() {
  // Every agent stakes one token on each of its neighbours: dealing the
  // amounts among the same neighbours gives the same flow, and the same modules.
  const frames = steadyGroups();
  const real = run(frames, []);
  const dealt = run(frames, ['--neighbours', '3']);
  for (const key of ['modulesPerFrame', 'sizes', 'lives']) {
    if (JSON.stringify(dealt[key]) !== JSON.stringify(real[key])) {
      throw new Error(`${key} changed: ${JSON.stringify(dealt[key])} against ${JSON.stringify(real[key])}`);
    }
  }
  if (!dealt.neighbours) throw new Error('the output must say its stakes were dealt among neighbours');
}

function test_dealing_among_neighbours_undoes_a_choice_of_partner() {
  // Two groups of four, each agent staking everything on one partner inside
  // its group though it also neighbours the other group: dealt at random, the
  // stakes cross between the groups, and the same seed deals them alike.
  const ids = [1, 2, 3, 4, 5, 6, 7, 8];
  const partner = { 1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7 };
  const frames = Array.from({ length: 6 }, (_, t) => ({
    iteration: t, phase: 2, ids,
    decisions: { allocations: ids.map(agent => {
      const targets = [agent, ...ids.filter(other => other !== agent)];
      return { agent, targets, alloc: targets.map(target => (target === partner[agent] ? 5 : 0)) };
    }) }
  }));
  const real = run(frames, []);
  const a = run(frames, ['--neighbours', '3']);
  const b = run(frames, ['--neighbours', '3']);
  if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error('a null must be the same null every time');
  if (real.lives['6'] !== 4) throw new Error(`the four chosen pairs last every game: ${JSON.stringify(real.lives)}`);
  if ((a.lives['6'] || 0) >= 4) throw new Error(`partners drawn at random should not all last: ${JSON.stringify(a.lives)}`);
}

function test_measures_are_only_added_when_asked_for() {
  const result = run(steadyGroups(), []);
  if ('outlast' in result) throw new Error('without --outlast the answer must be what it always was');
}

require('./harness').run(__filename, name => eval(name));
