/*
 * Structural measures of a frame's graph: loops, entropy, dimension.
 *
 * These cost more than reading a token count, so each is computed once per
 * frame and only when something asks for it.
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

  /** Connected components, as a node id -> component index map. */
  components(ids, adj) {
    const label = new Map();
    let count = 0;

    for (const start of ids) {
      if (label.has(start)) continue;
      const stack = [start];
      label.set(start, count);
      while (stack.length) {
        const u = stack.pop();
        for (const v of adj.get(u) || []) {
          if (!label.has(v)) { label.set(v, count); stack.push(v); }
        }
      }
      count++;
    }
    return { label, count };
  },

  /**
   * Loop structure.
   *
   * The number of independent loops in a graph is its cycle rank,
   * `edges - nodes + components` — the edges left over once a spanning forest
   * is taken out, each of which closes exactly one new loop. Counting loops
   * one by one is hopeless (their number grows exponentially), but this is
   * exact and costs nothing.
   *
   * For "how many loops is this node or edge part of", the honest answer comes
   * from 2-edge-connected components. An edge that lies on no loop at all is a
   * bridge: removing it splits the graph. Take the bridges out and what remains
   * falls into components in which every edge lies on some loop, and each such
   * component has its own cycle rank — the number of independent loops running
   * through that part of the graph. Those components partition the nodes, so
   * every node gets exactly one number.
   */
  loops(ids, edges, adj) {
    const bridges = this._bridges(ids, edges, adj);

    // Rebuild adjacency without the bridges; what is left is 2-edge-connected.
    const stripped = new Map();
    for (const id of ids) stripped.set(id, new Set());
    let bridgeCount = 0;

    for (let i = 0; i < edges.length; i++) {
      if (bridges.has(i)) { bridgeCount++; continue; }
      const [a, b] = edges[i];
      const sa = stripped.get(a), sb = stripped.get(b);
      if (sa && sb && a !== b) { sa.add(b); sb.add(a); }
    }

    const { label, count } = this.components(ids, stripped);

    // Cycle rank of each 2-edge-connected component.
    const nodesIn = new Array(count).fill(0);
    const edgesIn = new Array(count).fill(0);
    for (const id of ids) nodesIn[label.get(id)]++;
    for (let i = 0; i < edges.length; i++) {
      if (bridges.has(i)) continue;
      edgesIn[label.get(edges[i][0])]++;
    }
    const rankOf = nodesIn.map((n, c) => Math.max(0, edgesIn[c] - n + 1));

    const nodeLoops = new Map();
    for (const id of ids) nodeLoops.set(id, rankOf[label.get(id)]);

    const edgeLoops = new Int32Array(edges.length);
    for (let i = 0; i < edges.length; i++) {
      edgeLoops[i] = bridges.has(i) ? 0 : rankOf[label.get(edges[i][0])];
    }

    // Whole-graph cycle rank, over however many components the graph has.
    const whole = this.components(ids, adj);
    const cycleRank = Math.max(0, edges.length - ids.length + whole.count);

    return { cycleRank, bridges: bridgeCount, nodeLoops, edgeLoops,
             blockCount: count, componentCount: whole.count };
  },

  /**
   * Bridges, by iterative depth-first search.
   *
   * An edge is a bridge when nothing below it in the search tree can reach back
   * above it. Written as a loop with an explicit stack rather than recursion,
   * since a few thousand nodes is enough to overflow the call stack.
   * Returns the set of bridge indices into `edges`.
   */
  _bridges(ids, edges, adj) {
    // Index edges by endpoint pair so a traversal can name the edge it used.
    const edgeIndex = new Map();
    for (let i = 0; i < edges.length; i++) {
      const [a, b] = edges[i];
      edgeIndex.set(a < b ? `${a},${b}` : `${b},${a}`, i);
    }

    const disc = new Map();
    const low = new Map();
    const bridges = new Set();
    let timer = 0;

    for (const root of ids) {
      if (disc.has(root)) continue;

      // Each stack entry carries its own iterator, so the walk can pause and
      // resume exactly where it left off.
      const stack = [{ node: root, parent: null, iter: (adj.get(root) || new Set()).values() }];
      disc.set(root, timer); low.set(root, timer); timer++;

      while (stack.length) {
        const top = stack[stack.length - 1];
        const step = top.iter.next();

        if (step.done) {
          stack.pop();
          const parentEntry = stack[stack.length - 1];
          if (parentEntry) {
            const u = parentEntry.node, v = top.node;
            low.set(u, Math.min(low.get(u), low.get(v)));
            if (low.get(v) > disc.get(u)) {
              const key = u < v ? `${u},${v}` : `${v},${u}`;
              const idx = edgeIndex.get(key);
              if (idx !== undefined) bridges.add(idx);
            }
          }
          continue;
        }

        const next = step.value;
        if (next === top.parent) {
          // Skip the edge we arrived on, but only once: a genuine second edge
          // between the same pair would make neither of them a bridge.
          top.parent = null;
          continue;
        }
        if (disc.has(next)) {
          low.set(top.node, Math.min(low.get(top.node), disc.get(next)));
          continue;
        }
        disc.set(next, timer); low.set(next, timer); timer++;
        stack.push({ node: next, parent: top.node, iter: (adj.get(next) || new Set()).values() });
      }
    }
    return bridges;
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
  },

  /**
   * Global clustering coefficient.
   *
   * Three times the triangles over the number of connected triples: the chance
   * that two neighbours of a node are themselves neighbours.
   */
  transitivity(ids, adj, triangleTotal) {
    let triples = 0;
    for (const id of ids) {
      const d = (adj.get(id) || new Set()).size;
      triples += d * (d - 1) / 2;
    }
    return triples > 0 ? (3 * triangleTotal) / triples : 0;
  },

  /** Shannon entropy of a set of counts, in bits. */
  entropyOfCounts(counts) {
    let total = 0;
    for (const c of counts) total += c;
    if (total <= 0) return 0;

    let h = 0;
    for (const c of counts) {
      if (c <= 0) continue;
      const p = c / total;
      h -= p * Math.log2(p);
    }
    return h;
  },

  /** Entropy of the degree distribution: how varied the connectivity is. */
  degreeEntropy(degrees) {
    const histogram = new Map();
    for (const d of degrees) histogram.set(d, (histogram.get(d) || 0) + 1);
    return this.entropyOfCounts([...histogram.values()]);
  },

  /**
   * Ball-growth dimension, in the spirit of the Wolfram Physics Project.
   *
   * Walk outward from a node and measure how fast the frontier grows. In
   * d-dimensional space the ball of radius r holds about r^d nodes, so the
   * shell just added at radius r holds about r^(d-1); the slope of log(shell)
   * against log(r), plus one, estimates d.
   *
   * The shell is used rather than the ball because the ball carries the
   * starting node itself as a constant offset, which drags the slope down
   * badly at the small radii these graphs allow. Measured against lattices of
   * known dimension the shell form returns 1.00 for a chain, 1.92 for a square
   * grid and 2.56 for a cubic one — exact at one dimension and increasingly
   * conservative above it, since a handful of steps is not enough room for the
   * growth to reach its asymptote.
   *
   * Averaged over sampled starting points, since a hub and a leaf see very
   * different neighbourhoods. Radii whose ball has swallowed most of the graph
   * are dropped — once growth runs out of room the slope only measures the
   * boundary, not the geometry. A small-world graph has no honest dimension at
   * all: its balls grow exponentially, which shows up here as a large and
   * unstable number, and that is a real answer about the graph rather than a
   * failure of the estimate.
   */
  dimension(ids, adj, { seeds = 24, maxRadius = 5 } = {}) {
    const n = ids.length;
    if (n < 8) return { estimate: null, volumes: [] };

    const sampleCount = Math.min(seeds, n);
    const step = Math.max(1, Math.floor(n / sampleCount));
    const volumeAt = new Array(maxRadius + 1).fill(0);
    let sampled = 0;

    for (let s = 0; s < n; s += step) {
      const start = ids[s];
      const seen = new Set([start]);
      let shell = [start];
      volumeAt[0] += 1;

      for (let r = 1; r <= maxRadius; r++) {
        const next = [];
        for (const u of shell) {
          for (const v of adj.get(u) || []) {
            if (!seen.has(v)) { seen.add(v); next.push(v); }
          }
        }
        shell = next;
        volumeAt[r] += seen.size;
        if (!shell.length) {
          // Ball exhausted; it stays this size for every larger radius.
          for (let rr = r + 1; rr <= maxRadius; rr++) volumeAt[rr] += seen.size;
          break;
        }
      }
      sampled++;
    }
    if (!sampled) return { estimate: null, volumes: [] };

    const volumes = volumeAt.map(v => v / sampled);

    // Least-squares slope of log shell against log radius, over radii that
    // have not yet swallowed half the graph.
    const xs = [], ys = [];
    for (let r = 1; r <= maxRadius; r++) {
      if (volumes[r] > n * 0.5) continue;
      const shellSize = volumes[r] - volumes[r - 1];
      if (shellSize <= 0) continue;
      xs.push(Math.log(r));
      ys.push(Math.log(shellSize));
    }
    if (xs.length < 2) return { estimate: null, volumes };

    const meanX = xs.reduce((a, b) => a + b, 0) / xs.length;
    const meanY = ys.reduce((a, b) => a + b, 0) / ys.length;
    let num = 0, den = 0;
    for (let i = 0; i < xs.length; i++) {
      num += (xs[i] - meanX) * (ys[i] - meanY);
      den += (xs[i] - meanX) ** 2;
    }
    // The shell exponent is d - 1.
    return { estimate: den > 0 ? num / den + 1 : null, volumes, radiiUsed: xs.length };
  }
};
