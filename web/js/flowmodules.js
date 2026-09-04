/*
 * Flow modules, and following them through time.
 *
 * L0 — the modules. Group agents by where the tokens actually go: not by who
 * is joined to whom, and not by who is related to whom, but by who sends
 * tokens to whom. A module is a set of agents a token wandering the flow
 * network tends to stay inside.
 *
 * The objective is the two-level map equation (Rosvall & Bergstrom 2008),
 * which is the flow-native one: it asks how many bits it takes to describe a
 * random walk on the flow, given that walk's tendency to linger. If naming the
 * groups and then naming positions within them is shorter than naming every
 * position outright, the groups are carrying real structure. The number that
 * says how much is the difference between the code length of a partition and
 * the code length of one module holding everything — no structure at all.
 *
 * L1 — following them. A module at one iteration and a module at the next are
 * the same module when enough of their membership overlaps. That lets a module
 * survive while its agents are gradually replaced and lets it drift across the
 * graph, and it makes **turnover** measurable: a module that persists while
 * its members churn is a pattern whose matter is being replaced under it,
 * which is the thing that no definition anchored on membership can see.
 *
 * Both need per-edge flow, which a run only records with decisions turned on.
 * Without it there is nothing here to compute and the view says so.
 */
const FlowModules = {

  // ---- L0: the map equation ----------------------------------------------

  /**
   * x·log2(x), and 0 at 0.
   *
   * The map equation is a sum of these and nothing else, which is what makes a
   * single node's move between modules an O(1) question instead of a recount.
   */
  plogp(x) {
    return x > 0 ? x * Math.log2(x) : 0;
  },

  /**
   * How many bits a walk on this flow costs, given this partition.
   *
   *   L = plogp(q) − 2·Σ plogp(q_i) − Σ plogp(p_a) + Σ plogp(q_i + p_i)
   *
   * with p_a the share of the walk spent at agent `a`, q_i the share of it
   * that leaves module `i`, and p_i the share spent anywhere inside `i`. The
   * `Σ plogp(p_a)` term does not depend on the partition, so it is the only
   * part of this that a move never has to revisit.
   */
  codeLength(p, exit, inside) {
    let q = 0, sumExit = 0, sumBoth = 0;
    for (let i = 0; i < exit.length; i++) {
      if (inside[i] <= 0 && exit[i] <= 0) continue;
      q += exit[i];
      sumExit += this.plogp(exit[i]);
      sumBoth += this.plogp(exit[i] + inside[i]);
    }
    let sumNodes = 0;
    for (const share of p) sumNodes += this.plogp(share);
    return this.plogp(q) - 2 * sumExit - sumNodes + sumBoth;
  },

  /**
   * Partition agents by flow, by moving each to whichever neighbouring module
   * shortens the code most, until nothing more will.
   *
   * `edges` is [a, b, weight] over indices into `n`, undirected, weight being
   * the tokens that crossed. Louvain's local-moving step without the
   * aggregation phase: at these sizes the extra level buys little and costs a
   * rewrite of the graph on every pass.
   */
  partition(n, edges, { passes = 12, seed = 1 } = {}) {
    const strength = new Float64Array(n);
    let totalWeight = 0;
    for (const [a, b, w] of edges) {
      if (a === b || !(w > 0)) continue;
      strength[a] += w;
      strength[b] += w;
      totalWeight += w;
    }
    if (!(totalWeight > 0)) {
      return { module: new Int32Array(n).fill(-1), codeLength: 0,
               baseline: 0, modules: 0, isolated: n };
    }

    // Where a walker spends its time, and who each agent trades with.
    const p = new Float64Array(n);
    for (let a = 0; a < n; a++) p[a] = strength[a] / (2 * totalWeight);

    const near = Array.from({ length: n }, () => []);
    for (const [a, b, w] of edges) {
      if (a === b || !(w > 0)) continue;
      near[a].push([b, w]);
      near[b].push([a, w]);
    }

    // Everyone alone to begin with: every edge leaves its module.
    const module = new Int32Array(n);
    const inside = new Float64Array(n);
    const exit = new Float64Array(n);
    for (let a = 0; a < n; a++) {
      module[a] = a;
      inside[a] = p[a];
      exit[a] = strength[a] / (2 * totalWeight);
    }

    // A deterministic shuffle, so the answer does not depend on node order but
    // does not change between two runs of the same data either.
    const order = Int32Array.from({ length: n }, (_, i) => i);
    let state = seed >>> 0;
    const nextRandom = () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 4294967296;
    };
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(nextRandom() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }

    const cost = () => this.codeLength(p, exit, inside);
    let best = cost();

    for (let pass = 0; pass < passes; pass++) {
      let moved = 0;
      for (const a of order) {
        const from = module[a];

        // What `a` trades with each candidate module, so the exit weight after
        // a move can be worked out rather than recounted.
        const toModule = new Map();
        for (const [b, w] of near[a]) {
          toModule.set(module[b], (toModule.get(module[b]) || 0) + w);
        }
        const own = strength[a] / (2 * totalWeight);
        const linkedHome = (toModule.get(from) || 0) / totalWeight;

        // Take it out of its module first.
        inside[from] -= p[a];
        exit[from] -= own - linkedHome;

        let bestModule = from, bestCost = Infinity;
        for (const [candidate, weight] of toModule) {
          const linked = weight / totalWeight;
          inside[candidate] += p[a];
          exit[candidate] += own - linked;
          const here = cost();
          if (here < bestCost) { bestCost = here; bestModule = candidate; }
          inside[candidate] -= p[a];
          exit[candidate] -= own - linked;
        }
        // Staying put is always on the table.
        inside[from] += p[a];
        exit[from] += own - linkedHome;
        if (cost() <= bestCost + 1e-12) { bestModule = from; bestCost = cost(); }

        if (bestModule !== from) {
          inside[from] -= p[a];
          exit[from] -= own - linkedHome;
          const linked = (toModule.get(bestModule) || 0) / totalWeight;
          inside[bestModule] += p[a];
          exit[bestModule] += own - linked;
          module[a] = bestModule;
          moved++;
        }
      }
      const now = cost();
      if (!moved || best - now < 1e-10) { best = now; break; }
      best = now;
    }

    // One module holding everything is the "no structure" answer to compare
    // against: nothing ever leaves it, so its code is just the walk itself.
    const allInside = new Float64Array(n);
    const allExit = new Float64Array(n);
    for (let a = 0; a < n; a++) allInside[0] += p[a];
    const baseline = this.codeLength(p, allExit, allInside);

    const seen = new Set();
    let isolated = 0;
    for (let a = 0; a < n; a++) {
      if (strength[a] > 0) seen.add(module[a]);
      else { module[a] = -1; isolated++; }
    }
    return { module, codeLength: best, baseline, modules: seen.size, isolated };
  },

  // ---- reading a frame ---------------------------------------------------

  /**
   * The flow network of one recorded frame.
   *
   * Tokens are staked by agents on their neighbours, so what crossed an edge
   * is in the allocations — which a run only writes down with decisions
   * turned on. Self-stakes are dropped: a token an agent keeps says nothing
   * about who it is grouped with.
   */
  flowOf(frame) {
    const index = new Map();
    (frame.ids || []).forEach((id, i) => index.set(id, i));
    const weight = new Map();
    const allocations = ((frame.decisions || {}).allocations) || [];

    for (const record of allocations) {
      const from = index.get(record.agent);
      if (from === undefined) continue;
      const targets = record.targets || [];
      const alloc = record.alloc || [];
      for (let i = 0; i < targets.length; i++) {
        const amount = alloc[i] || 0;
        if (amount <= 0) continue;
        const to = index.get(targets[i]);
        if (to === undefined || to === from) continue;
        const key = from < to ? `${from},${to}` : `${to},${from}`;
        weight.set(key, (weight.get(key) || 0) + amount);
      }
    }

    const edges = [];
    for (const [key, w] of weight) {
      const [a, b] = key.split(",");
      edges.push([Number(a), Number(b), w]);
    }
    return { ids: frame.ids || [], edges, hasFlow: allocations.length > 0 };
  },

  // ---- L1: the same module, one iteration later --------------------------

  /**
   * Follow modules through time by how much of their membership they share.
   *
   * Matched greedily on the Jaccard overlap of their agents, best pair first,
   * above a floor. Sharing agents rather than sharing *all* agents is what
   * lets a module keep its identity while its members are replaced one by one,
   * and lets it walk across the graph. What it cannot follow is a module that
   * teleports, and nothing here pretends otherwise.
   *
   * `turnover` is the part worth watching: the share of a module's agents that
   * were not in it last time. A module with a long life and a high turnover is
   * a pattern being carried by different matter each iteration.
   */
  track(previous, current, { floor = 0.3 } = {}) {
    const pairs = [];
    for (const [oldId, oldSet] of previous) {
      for (const [newKey, newSet] of current) {
        let shared = 0;
        const [small, large] = oldSet.size < newSet.size ? [oldSet, newSet] : [newSet, oldSet];
        for (const member of small) if (large.has(member)) shared++;
        if (!shared) continue;
        const overlap = shared / (oldSet.size + newSet.size - shared);
        if (overlap >= floor) pairs.push({ oldId, newKey, overlap, shared });
      }
    }
    pairs.sort((a, b) => b.overlap - a.overlap);

    const takenOld = new Set(), takenNew = new Set(), assigned = new Map();
    for (const pair of pairs) {
      if (takenOld.has(pair.oldId) || takenNew.has(pair.newKey)) continue;
      takenOld.add(pair.oldId);
      takenNew.add(pair.newKey);
      assigned.set(pair.newKey, { id: pair.oldId, shared: pair.shared });
    }
    return assigned;
  },

  /**
   * Every module of every frame, given a lasting identity.
   *
   * Returns one record per module-appearance, carrying the identity it keeps
   * across iterations, its members, and how much of it is new since last time.
   */
  follow(frames, { floor = 0.3 } = {}) {
    const history = [];
    let previous = new Map();
    let nextId = 0;
    let withoutFlow = 0;

    for (const frame of frames) {
      const { ids, edges, hasFlow } = this.flowOf(frame);
      if (!hasFlow) { withoutFlow++; continue; }

      const found = this.partition(ids.length, edges);
      const groups = new Map();
      for (let i = 0; i < ids.length; i++) {
        const label = found.module[i];
        if (label < 0) continue;
        if (!groups.has(label)) groups.set(label, new Set());
        groups.get(label).add(ids[i]);
      }

      const carried = this.track(previous, groups, { floor });
      const now = new Map();
      for (const [label, members] of groups) {
        const match = carried.get(label);
        const id = match ? match.id : nextId++;
        const fresh = match ? members.size - match.shared : members.size;
        history.push({
          iteration: frame.iteration,
          phase: frame.phase,
          id,
          size: members.size,
          turnover: members.size ? fresh / members.size : 0,
          isNew: !match,
          codeLength: found.codeLength,
          baseline: found.baseline,
          modules: found.modules
        });
        now.set(id, members);
      }
      previous = now;
    }
    return { history, withoutFlow };
  },

  /** What a run's module structure amounts to, in a handful of numbers. */
  summarise(history) {
    if (!history.length) return null;
    const byId = new Map();
    for (const row of history) {
      if (!byId.has(row.id)) byId.set(row.id, []);
      byId.get(row.id).push(row);
    }
    const lives = [...byId.values()];
    const saving = history.reduce(
      (sum, r) => sum + (r.baseline > 0 ? (r.baseline - r.codeLength) / r.baseline : 0), 0)
      / history.length;

    const persistent = lives.filter(l => l.length >= 3);
    const churn = persistent.length
      ? persistent.reduce((sum, l) =>
          sum + l.slice(1).reduce((s, r) => s + r.turnover, 0) / Math.max(1, l.length - 1), 0)
        / persistent.length
      : 0;

    return {
      appearances: history.length,
      distinct: byId.size,
      longestLife: Math.max(...lives.map(l => l.length)),
      largest: Math.max(...history.map(r => r.size)),
      meanModules: history.reduce((s, r) => s + r.modules, 0) / history.length,
      compression: saving,
      persistent: persistent.length,
      meanTurnover: churn
    };
  }
};
