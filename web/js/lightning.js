/*
 * Lightning: how much of a Blotto phase's token flow goes round in circles.
 *
 * A lightning is a closed loop of token flow — tokens leaving an agent,
 * crossing links, arriving back where they started, like current round a
 * circuit. A loop of L hops carries L tokens, one per edge, and scores L², so
 * one token round a triangle is a small thing and ten round a ten-edge loop is
 * a large one. Every token belongs to at most one lightning.
 *
 * Maximising the total is NP-hard — packing flow into few long loops contains
 * "find the longest cycle", which is Hamiltonian cycle in disguise — so what is
 * reported is a bracket with both ends labelled. See gol_lightning.py, which
 * this mirrors step for step and which carries the full argument.
 */
const Lightning = {

  /** Matching the two limits in gol_lightning.py. */
  MAX_LOOPS: 20000,
  MAX_WALK: 4096,

  /**
   * The same flow with reciprocal amounts cancelled against each other.
   *
   * Five one way and three the other becomes two one way and nothing back.
   * Every edge then carries flow in at most one direction, which makes a
   * two-hop loop impossible and every remaining lightning a real circuit
   * rather than two neighbours sloshing tokens between them.
   */
  netted(flow) {
    const out = new Map();
    let total = 0;
    const seen = new Set();
    for (const [source, targets] of flow) {
      for (const [target, amount] of targets) {
        const pair = source < target ? `${source},${target}` : `${target},${source}`;
        if (seen.has(pair)) continue;
        seen.add(pair);
        const back = (flow.get(target) || new Map()).get(source) || 0;
        const net = amount - back;
        if (net > 0) {
          if (!out.has(source)) out.set(source, new Map());
          out.get(source).set(target, net);
          total += net;
        } else if (net < 0) {
          if (!out.has(target)) out.set(target, new Map());
          out.get(target).set(source, -net);
          total += -net;
        }
      }
    }
    return { flow: out, total };
  },

  /** Who sent how many tokens to whom, and how many moved in total. */
  network(frame) {
    const flow = new Map();
    let total = 0;
    const allocations = ((frame.decisions || {}).allocations) || [];
    for (const entry of allocations) {
      const source = Number(entry.agent);
      const targets = entry.targets || [];
      const amounts = entry.alloc || [];
      for (let position = 0; position < targets.length; position++) {
        const target = Number(targets[position]);
        if (target === source || position >= amounts.length) continue;
        const amount = Number(amounts[position]);
        if (!(amount > 0)) continue;
        if (!flow.has(source)) flow.set(source, new Map());
        const row = flow.get(source);
        row.set(target, (row.get(target) || 0) + amount);
        total += amount;
      }
    }
    return { flow, total };
  },

  /**
   * The share of flow conservation forbids from circulating.
   *
   * Every token in a loop leaves its agent and returns to it, so a loop moves
   * nobody's net balance. Whatever imbalance is left is one-way transport that
   * no decomposition can bend into a circle. Halved because each stranded
   * token appears twice, once as a surplus and once as a deficit.
   */
  imbalance(flow, total) {
    if (!(total > 0)) return 0;
    const net = new Map();
    for (const [source, targets] of flow) {
      for (const [target, amount] of targets) {
        net.set(source, (net.get(source) || 0) - amount);
        net.set(target, (net.get(target) || 0) + amount);
      }
    }
    let stranded = 0;
    for (const value of net.values()) stranded += Math.abs(value);
    return Math.min(1, (stranded / 2) / total);
  },

  /**
   * Peel loops out of one frame's flow.
   *
   * The walk prefers a step to an agent it has not been to yet, which is what
   * finds long loops rather than closing on the first triangle it stumbles
   * into. When there is nowhere new it takes any step it can, and the moment it
   * lands on an agent already in the walk the loop is closed.
   *
   * Deterministic: neighbours in sorted order, starts in sorted order, so the
   * same frame gives the same number here and in the Python.
   */
  peel(flow) {
    const residual = new Map();
    const order = new Map();
    for (const [source, targets] of flow) {
      residual.set(source, new Map(targets));
      order.set(source, [...targets.keys()].sort((a, b) => a - b));
    }

    let score = 0, used = 0, longest = 0, loops = 0;
    const starts = [...residual.keys()].sort((a, b) => a - b);

    for (const start of starts) {
      while (loops < this.MAX_LOOPS) {
        const walk = [start];
        const where = new Map([[start, 0]]);
        let closed = null;

        for (let step = 0; step < this.MAX_WALK; step++) {
          const here = walk[walk.length - 1];
          const options = order.get(here) || [];
          const row = residual.get(here);
          let next = null, fallback = null;
          for (const candidate of options) {
            if (!row || !(row.get(candidate) > 0)) continue;
            if (!where.has(candidate)) { next = candidate; break; }
            if (fallback === null) fallback = candidate;
          }
          if (next === null) next = fallback;
          if (next === null) break;
          if (where.has(next)) { closed = where.get(next); walk.push(next); break; }
          where.set(next, walk.length);
          walk.push(next);
        }

        if (closed === null) break;
        const hops = walk.length - 1 - closed;
        if (hops < 2) break;

        for (let i = closed; i < walk.length - 1; i++) {
          const a = walk[i], b = walk[i + 1];
          const row = residual.get(a);
          row.set(b, row.get(b) - 1);
          if (!(row.get(b) > 0)) {
            row.delete(b);
            order.set(a, order.get(a).filter(n => n !== b));
          }
        }

        score += hops * hops;
        used += hops;
        longest = Math.max(longest, hops);
        loops++;
      }
    }

    return { score, used, longest };
  },

  /** Both readings of one frame: the flow as sent, and the net flow. */
  of(frame) {
    const { flow, total } = this.network(frame);
    if (!(total > 0)) {
      return { lightningScore: null, cyclingShare: null,
               lightningLongest: null, flowImbalance: null,
               netLightningScore: null, netCyclingShare: null,
               netLightningLongest: null, netFlowShare: null };
    }
    const gross = this.peel(flow);
    const net = this.netted(flow);
    const netPeel = this.peel(net.flow);

    return {
      lightningScore: gross.score,
      cyclingShare: Math.min(1, gross.used / total),
      lightningLongest: gross.longest,
      flowImbalance: this.imbalance(flow, total),
      netLightningScore: netPeel.score,
      // Against the net total, so it answers "of the tokens that actually went
      // somewhere, how many went round" rather than being diluted by the
      // reciprocity netting just removed.
      netCyclingShare: net.total ? Math.min(1, netPeel.used / net.total) : 0,
      netLightningLongest: netPeel.longest,
      // How much of the gross flow survived cancellation. Low means neighbours
      // are mostly trading back and forth.
      netFlowShare: Math.min(1, net.total / total)
    };
  }
};
