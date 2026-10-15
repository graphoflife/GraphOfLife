#!/usr/bin/env node
/*
 * Flow modules of a window of frames, worked out by the page's own arithmetic.
 *
 *     node tools/flowmodules_cli.js [--floor 0.3] [--mutual] [--null SEED] < frames.jsonl
 *
 * Reads one frame per line — iteration, phase, ids and decisions.allocations,
 * the fields the Research tab's Flow modules view reads — follows the modules
 * of the frames that carry flow (web/js/flowmodules.js), and prints one JSON
 * object: the summary the view shows, the number of modules in every frame,
 * and how many frames had no flow.
 *
 * `--mutual` matches a module to the next frame's only when each is the
 * other's best overlap, at any overlap, instead of best-first above a floor.
 * `--null SEED` first rewires every frame's flow at random, keeping how much
 * flow each agent sends and receives: each token that crossed a link is a
 * stub at both ends, the stubs are paired again at random, and the modules are
 * found in that. It says what the same amounts of flow give without their
 * structure.
 *
 * There is one implementation of the modules, the page's; this only loads it
 * the way tests/test_flowmodules.js does and hands it frames.
 */
const fs = require('fs');
const path = require('path');

const read = name => fs.readFileSync(path.join(__dirname, '..', 'web', 'js', name), 'utf8');
const GraphStats = new Function(`${read('graphstats.js')}; return GraphStats;`)();
const FlowModules = new Function('GraphStats', `${read('flowmodules.js')}; return FlowModules;`)(GraphStats);

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const at = args.indexOf(name);
  return at < 0 ? fallback : (args[at + 1] === undefined || args[at + 1].startsWith('--') ? true : args[at + 1]);
};
const floor = Number(option('--floor', 0.3));
const mutual = !!option('--mutual', false);
const nullSeed = option('--null', null);

/** A seeded stream (mulberry32), so a null is the same null every time. */
function stream(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The same frame with its flow paired again at random, every agent's amount of flow kept. */
function rewired(frame, random) {
  const { ids, edges } = FlowModules.flowOf(frame);
  const stubs = [];
  for (const [a, b, w] of edges) {
    for (let k = 0; k < w; k++) { stubs.push(a); stubs.push(b); }
  }
  for (let i = stubs.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [stubs[i], stubs[j]] = [stubs[j], stubs[i]];
  }
  const allocations = [];
  for (let i = 0; i + 1 < stubs.length; i += 2) {
    if (stubs[i] === stubs[i + 1]) continue;          // a token cannot cross to its own sender
    allocations.push({ agent: ids[stubs[i]], targets: [ids[stubs[i + 1]]], alloc: [1] });
  }
  return { ...frame, decisions: { allocations } };
}

const lines = fs.readFileSync(0, 'utf8').split('\n').filter(line => line.trim());
let frames = lines.map(line => JSON.parse(line));
if (nullSeed !== null) {
  const random = stream(Number(nullSeed));
  frames = frames.map(frame => rewired(frame, random));
}
const { history, withoutFlow } = FlowModules.follow(frames, { floor, mutual });
const perFrame = new Map();
for (const row of history) perFrame.set(row.iteration * 2 + row.phase, row.modules);
// How many games each module lived, and how many agents it had, as counts.
const lives = new Map();
for (const row of history) lives.set(row.id, (lives.get(row.id) || 0) + 1);
const lifeCounts = {}, sizeCounts = {};
for (const life of lives.values()) lifeCounts[life] = (lifeCounts[life] || 0) + 1;
for (const row of history) sizeCounts[row.size] = (sizeCounts[row.size] || 0) + 1;
process.stdout.write(JSON.stringify({
  frames: frames.length,
  withoutFlow,
  floor: mutual ? null : floor,
  mutual,
  null: nullSeed !== null,
  modulesPerFrame: [...perFrame.values()],
  agentsPerFrame: frames.filter(f => (f.decisions || {}).allocations?.length).map(f => (f.ids || []).length),
  lives: lifeCounts,
  sizes: sizeCounts,
  summary: FlowModules.summarise(history)
}));
