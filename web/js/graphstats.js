/*
 * What colouring a frame by its structure needs: its adjacency, the loops and
 * triangles each agent and link lies on, and how much flowed between each two.
 *
 * These cost more than reading a token count, so each is computed once per
 * frame and only when something asks for it. The statistics of a whole frame —
 * bridges, dimension, power laws and the rest — are gol_series', worked out in
 * Python and asked for (gol_framestats); this file used to hold a second copy.
 */
const GraphStats = {

  /** Adjacency as sets, which is what the loop and triangle work needs. */
  adjacency(ids, edges) {
    const adj = new Map();
    for (const id of ids) adj.set(id, new Set());
    for (const [a, b] of edges) {
      const sa = adj.get(a), sb = adj.get(b);
      if (sa && sb && a !== b) { sa.add(b); sb.add(a); }
    }
    return adj;
  },

  /** One key for an undirected edge, whichever end it is named from. */
  pairKey(a, b) {
    return a < b ? `${a},${b}` : `${b},${a}`;
  },

  /**
   * What crossed each link during a game phase, read from its allocations: a
   * map from a pair of positions in `ids`, as lower × (n + 1) + higher, to the
   * tokens staked across that pair either way.
   *
   * A token an agent keeps is not flow, and a stake on an agent gone from the
   * frame had no link left to cross. Numbers rather than strings for keys, so
   * the renderer can look an edge up without building one per edge. The
   * statistics under the canvas and Flow modules each read allocations this
   * way, in code of their own.
   */
  flowByPair(ids, allocations, index = null) {
    const at = index || new Map(ids.map((id, i) => [id, i]));
    const stride = ids.length + 1;
    const flow = new Map();
    for (const record of allocations) {
      const from = at.get(record.agent);
      if (from === undefined) continue;
      const targets = record.targets || [], alloc = record.alloc || [];
      for (let i = 0; i < targets.length; i++) {
        const amount = alloc[i];
        if (!(amount > 0)) continue;
        const to = at.get(targets[i]);
        if (to === undefined || to === from) continue;
        const key = from < to ? from * stride + to : to * stride + from;
        flow.set(key, (flow.get(key) || 0) + amount);
      }
    }
    return flow;
  },

  /**
   * How many loops each node and edge actually lies on.
   *
   * Counting every loop through an element is intractable — the number of
   * cycles in a graph grows exponentially — so this counts a *basis* instead.
   * Take a spanning tree; each of the remaining edges closes exactly one loop,
   * its fundamental cycle, and those cycles form an independent set that
   * generates every other loop in the graph. There are exactly `cycleRank` of
   * them, so the counts are on the same footing as the total.
   *
   * The tree is built breadth-first, which keeps the fundamental cycles short
   * and local rather than sending them on long detours through the graph. A
   * different tree would give a different basis, so treat these as a fair
   * sample of the loop structure rather than a canonical answer — but unlike
   * the cycle rank of a whole 2-edge-connected block, which hands every node in
   * the core the same large number, this actually distinguishes a node that
   * many loops run through from one that only a couple do.
   */
  cycleParticipation(ids, edges, adj) {
    const perNode = new Map();
    for (const id of ids) perNode.set(id, 0);
    const perEdge = new Int32Array(edges.length);

    const edgeIndex = new Map();
    for (let i = 0; i < edges.length; i++) {
      const [a, b] = edges[i];
      edgeIndex.set(this.pairKey(a, b), i);
    }

    // Breadth-first spanning forest.
    const parent = new Map(), depth = new Map();
    const seen = new Set();
    for (const root of ids) {
      if (seen.has(root)) continue;
      seen.add(root); parent.set(root, null); depth.set(root, 0);
      const queue = [root];
      for (let qi = 0; qi < queue.length; qi++) {
        const u = queue[qi];
        for (const v of adj.get(u) || []) {
          if (seen.has(v)) continue;
          seen.add(v); parent.set(v, u); depth.set(v, depth.get(u) + 1);
          queue.push(v);
        }
      }
    }

    const treeEdges = new Set();
    for (const [child, up] of parent) {
      if (up === null) continue;
      treeEdges.add(child < up ? `${child},${up}` : `${up},${child}`);
    }

    const bump = (a, b) => {
      const i = edgeIndex.get(this.pairKey(a, b));
      if (i !== undefined) perEdge[i]++;
    };

    for (let i = 0; i < edges.length; i++) {
      const [a, b] = edges[i];
      const key = this.pairKey(a, b);
      if (treeEdges.has(key)) continue;          // tree edges close no new loop
      if (!depth.has(a) || !depth.has(b)) continue;

      // Climb both ends to their common ancestor; that path plus this edge is
      // the fundamental cycle.
      let x = a, y = b;
      const left = [], right = [];
      let guard = 0;
      const limit = ids.length + 1;

      while (depth.get(x) > depth.get(y) && guard++ < limit) { left.push(x); x = parent.get(x); }
      while (depth.get(y) > depth.get(x) && guard++ < limit) { right.push(y); y = parent.get(y); }
      while (x !== y && guard++ < limit) {
        left.push(x); right.push(y);
        x = parent.get(x); y = parent.get(y);
      }
      if (x !== y) continue;                     // different trees; no cycle

      const meeting = x;

      // Nodes on the cycle: both climbs plus the ancestor they met at.
      for (const n of left) perNode.set(n, perNode.get(n) + 1);
      for (const n of right) perNode.set(n, perNode.get(n) + 1);
      perNode.set(meeting, perNode.get(meeting) + 1);

      // Edges: the closing edge, then each tree step taken on the way up.
      perEdge[i]++;
      let prev = a;
      for (const n of left) { if (n !== prev) bump(prev, n); prev = n; }
      bump(prev, meeting);
      prev = b;
      for (const n of right) { if (n !== prev) bump(prev, n); prev = n; }
      bump(prev, meeting);
    }

    return { perNode, perEdge };
  },

  /**
   * Triangles per node and per edge.
   *
   * A triangle is the shortest loop there is, and unlike loops in general they
   * can be counted exactly and cheaply: the triangles on an edge are simply the
   * neighbours its two endpoints share.
   */
  triangles(ids, edges, adj) {
    const perEdge = new Int32Array(edges.length);
    const perNode = new Map();
    for (const id of ids) perNode.set(id, 0);

    let total = 0;
    for (let i = 0; i < edges.length; i++) {
      const [a, b] = edges[i];
      const na = adj.get(a), nb = adj.get(b);
      if (!na || !nb) continue;

      // Walk the smaller neighbourhood and test against the larger.
      const [small, large] = na.size <= nb.size ? [na, nb] : [nb, na];
      let shared = 0;
      for (const w of small) if (large.has(w)) shared++;

      perEdge[i] = shared;
      total += shared;
    }
    // Each triangle shows up on all three of its edges.
    total = Math.round(total / 3);

    // Summing the triangles on a node's edges counts each of its triangles twice.
    for (let i = 0; i < edges.length; i++) {
      const [a, b] = edges[i];
      if (perNode.has(a)) perNode.set(a, perNode.get(a) + perEdge[i]);
      if (perNode.has(b)) perNode.set(b, perNode.get(b) + perEdge[i]);
    }
    for (const id of ids) perNode.set(id, perNode.get(id) / 2);

    return { total, perEdge, perNode };
  }
};
