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

require('./harness').run(__filename, name => eval(name));
