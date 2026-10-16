#!/usr/bin/env node
/*
 * Flow modules of a window of frames, worked out by the page's own arithmetic.
 *
 *     node tools/flowmodules_cli.js [--floor 0.3] [--mutual] [--null SEED] [--neighbours SEED]
 *                                   [--shuffle SEED] [--partitions FILE] [--outlast] < frames.jsonl
 *
 * Reads one frame per line — iteration, phase, ids and decisions.allocations,
 * the fields the Research tab's Flow modules view reads — follows the modules
 * of the frames that carry flow (web/js/flowmodules.js), and prints one JSON
 * object: the summary the view shows, the number of modules in every frame,
 * and how many frames had no flow.
 *
 * `--mutual` keeps, of the pairs above the floor, only those in which each
 * module is the other's best overlap; with `--floor 0` that is at any overlap.
 * `--null SEED` first rewires every frame's flow at random, keeping how much
 * flow each agent sends and receives: each token that crossed a link is a
 * stub at both ends, the stubs are paired again at random, and the modules are
 * found in that. It says what the same amounts of flow give without their
 * structure — but its modules are pairs and threes, so it is no yardstick for
 * how long modules live, and its tokens cross between agents that are not
 * neighbours, which no token can. `--neighbours SEED` keeps the network: every
 * agent keeps its stake at home and its amounts, which are dealt at random
 * among its own neighbours. What survives it is not the network's doing but
 * the agents' choice of whom to stake on. `--shuffle SEED` is: it keeps every frame's modules
 * and their sizes, deals the agents among them at random, frame by frame, and
 * follows those. A module of the real flow that lives longer than shuffled
 * modules of its size lives on something other than chance overlaps.
 * `--partitions FILE` also writes every frame's modules, one line a frame
 * ({iteration, ids, module}: the module of each agent, -1 for none), for
 * measures that need the whole partition.
 * `--outlast` adds, for the modules followed at least 5, 10, 20 and 40
 * frames, how many there were, the share of them none of whose first members
 * was still a member at its last frame, and the mean share of first members
 * still there: whether a module outlasts the agents it began with.
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
const neighboursSeed = option('--neighbours', null);
const shuffleSeed = option('--shuffle', null);
const partitionsFile = option('--partitions', null);
const outlast = !!option('--outlast', false);

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

/**
 * Every frame's modules with the agents dealt among them at random, the sizes
 * kept: what following modules gives when nothing but their sizes carries over.
 */
function shuffleModules(random) {
  const found = FlowModules.modulesOf.bind(FlowModules);
  const dealt = new WeakMap();
  FlowModules.modulesOf = frame => {
    let known = dealt.get(frame);
    if (!known) {
      const real = found(frame);
      const agents = [];
      for (const members of real.groups.values()) for (const agent of members) agents.push(agent);
      for (let i = agents.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [agents[i], agents[j]] = [agents[j], agents[i]];
      }
      const groups = new Map();
      let at = 0;
      for (const [label, members] of real.groups) {
        groups.set(label, new Set(agents.slice(at, at + members.size)));
        at += members.size;
      }
      known = { ...real, groups };
      dealt.set(frame, known);
    }
    return known;
  };
}

/** The same frame with each agent's stakes on its neighbours dealt among them at random. */
function reassigned(frame, random) {
  const allocations = ((frame.decisions || {}).allocations || []).map(record => {
    const others = [];
    record.targets.forEach((target, i) => { if (target !== record.agent) others.push(i); });
    const alloc = record.alloc.slice();
    for (let k = others.length - 1; k > 0; k--) {
      const j = Math.floor(random() * (k + 1));
      [alloc[others[k]], alloc[others[j]]] = [alloc[others[j]], alloc[others[k]]];
    }
    return { ...record, alloc };
  });
  return { ...frame, decisions: { ...frame.decisions, allocations } };
}

const lines = fs.readFileSync(0, 'utf8').split('\n').filter(line => line.trim());
let frames = lines.map(line => JSON.parse(line));
if (nullSeed !== null) {
  const random = stream(Number(nullSeed));
  frames = frames.map(frame => rewired(frame, random));
}
if (neighboursSeed !== null) {
  const random = stream(Number(neighboursSeed));
  frames = frames.map(frame => reassigned(frame, random));
}
if (shuffleSeed !== null) shuffleModules(stream(Number(shuffleSeed)));
const { history, withoutFlow } = FlowModules.follow(frames, { floor, mutual, keepMembers: outlast });
const perFrame = new Map();
for (const row of history) perFrame.set(row.iteration * 2 + row.phase, row.modules);
// How many games each module lived, and how many agents it had, as counts.
const lives = new Map();
for (const row of history) lives.set(row.id, (lives.get(row.id) || 0) + 1);
const lifeCounts = {}, sizeCounts = {};
for (const life of lives.values()) lifeCounts[life] = (lifeCounts[life] || 0) + 1;
for (const row of history) sizeCounts[row.size] = (sizeCounts[row.size] || 0) + 1;
if (partitionsFile !== null) {
  const rows = [];
  for (const frame of frames) {
    const { ids, hasFlow, groups } = FlowModules.modulesOf(frame);
    if (!hasFlow) continue;
    const moduleOf = new Map();
    for (const [label, members] of groups) for (const agent of members) moduleOf.set(agent, label);
    rows.push(JSON.stringify({ iteration: frame.iteration, ids, module: ids.map(id => (moduleOf.has(id) ? moduleOf.get(id) : -1)) }));
  }
  fs.writeFileSync(partitionsFile, rows.join('\n') + '\n');
}
/** For modules followed at least so many frames: how many, how many lost every first member, how many stayed. */
function outlasting(rows) {
  const first = new Map(), last = new Map(), life = new Map();
  for (const row of rows) {
    if (!first.has(row.id)) first.set(row.id, new Set(row.members));
    last.set(row.id, row.members);
    life.set(row.id, (life.get(row.id) || 0) + 1);
  }
  const out = {};
  for (const atLeast of [5, 10, 20, 40]) {
    let modules = 0, renewed = 0, stayed = 0;
    for (const [id, games] of life) {
      if (games < atLeast) continue;
      const began = first.get(id);
      const still = last.get(id).filter(agent => began.has(agent)).length;
      modules++;
      if (still === 0) renewed++;
      stayed += still / began.size;
    }
    out[atLeast] = { modules, renewed: modules ? renewed / modules : null, stayed: modules ? stayed / modules : null };
  }
  return out;
}

process.stdout.write(JSON.stringify({
  frames: frames.length,
  withoutFlow,
  floor: mutual ? null : floor,
  mutual,
  null: nullSeed !== null,
  shuffled: shuffleSeed !== null,
  ...(neighboursSeed !== null ? { neighbours: true } : {}),
  ...(outlast ? { outlast: outlasting(history) } : {}),
  modulesPerFrame: [...perFrame.values()],
  agentsPerFrame: frames.filter(f => (f.decisions || {}).allocations?.length).map(f => (f.ids || []).length),
  lives: lifeCounts,
  sizes: sizeCounts,
  summary: FlowModules.summarise(history)
}));
