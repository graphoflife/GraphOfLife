/*
 * Turning a frame into numbers: what drives colour, what drives size, and the
 * summary statistics and histograms shown under the canvas.
 *
 * FrameMetrics is rebuilt whenever the frame or the relevant settings change,
 * so the renderer can stay a dumb value-to-pixel mapper.
 */
class FrameMetrics {
  constructor(frame, settings) {
    this.frame = frame;
    this.settings = settings;

    this.index = new Map();
    frame.ids.forEach((id, i) => this.index.set(id, i));

    this.degree = this._degrees();
    // Runs recorded before deltas existed simply have none; treat that as no
    // change rather than letting undefined leak into the colour maths.
    this.delta = frame.delta || new Array(frame.ids.length).fill(0);
    this.hasDelta = Boolean(frame.delta);
    this.totalTokens = frame.tokens.reduce((a, b) => a + b, 0);
    this.curvature = this._curvature();

    this.colorValues = this.scaledNodeValues(settings.nodeColorBy, settings.nodeColorLog);
    this.colorRange = this.nodeRange(settings.nodeColorBy, settings.nodeColorLog);
    this.sizeValues = this.scaledNodeValues(settings.nodeSizeBy, settings.nodeSizeLog);
    this.sizeRange = this.nodeRange(settings.nodeSizeBy, settings.nodeSizeLog);

    this.colorLabel = Metrics.label('node', settings.nodeColorBy)
      + (settings.nodeColorLog ? ' (log)' : '');
    this.colorRangeText = [
      Metrics.format('node', settings.nodeColorBy, this.colorRange[0], settings.nodeColorLog),
      Metrics.format('node', settings.nodeColorBy, this.colorRange[1], settings.nodeColorLog)
    ];
  }

  _degrees() {
    const deg = new Int32Array(this.frame.ids.length);
    for (const [a, b] of this.frame.edges) {
      const ia = this.index.get(a), ib = this.index.get(b);
      if (ia !== undefined) deg[ia]++;
      if (ib !== undefined) deg[ib]++;
    }
    return deg;
  }

  /**
   * Tokens sent along each edge during phase 2.
   *
   * Only available when the run recorded decisions, and only on phase-2 frames
   * — phase 1 has no allocations. Absent that, flow-based options fall back to
   * zero and the UI says so.
   */
  _edgeFlow() {
    const flow = new Map();
    const decisions = this.frame.decisions;
    if (!decisions || !decisions.allocations) return flow;

    const index = this.index;
    const stride = this.frame.ids.length + 1;

    for (const record of decisions.allocations) {
      const source = record.agent;
      const is = index.get(source);
      if (is === undefined) continue;

      for (let i = 0; i < record.targets.length; i++) {
        const target = record.targets[i];
        const amount = record.alloc[i];
        if (!amount || target === source) continue;
        const it = index.get(target);
        if (it === undefined) continue;

        const key = is < it ? is * stride + it : it * stride + is;
        flow.set(key, (flow.get(key) || 0) + amount);
      }
    }
    return flow;
  }

  get flow() {
    if (!this._flow) this._flow = this._edgeFlow();
    return this._flow;
  }

  get hasFlow() { return this.flow.size > 0; }

  /**
   * Raw per-node values for one metric, in frame order.
   *
   * No scaling is applied here: a metric has one set of values, and whether
   * they are read linearly or logarithmically is the reader's choice. Cached,
   * since the charts and the renderer often want the same one.
   */
  nodeValues(key) {
    if (!this._nodeCache) this._nodeCache = new Map();
    const hit = this._nodeCache.get(key);
    if (hit) return hit;

    const f = this.frame;
    const n = f.ids.length;
    const out = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      switch (key) {
        case 'tokens':           out[i] = f.tokens[i]; break;
        case 'degree':           out[i] = this.degree[i]; break;
        case 'token_delta':      out[i] = this.delta[i]; break;
        case 'abs_token_delta':  out[i] = Math.abs(this.delta[i]); break;
        case 'token_curvature':  out[i] = this.curvature[i]; break;
        case 'token_curvature_pre': out[i] = this.curvatureBefore[i]; break;
        case 'loops':      out[i] = this.structure.loops.nodeLoops.get(f.ids[i]) || 0; break;
        case 'triangles':  out[i] = this.structure.triangles.perNode.get(f.ids[i]) || 0; break;
        case 'brain_id':         out[i] = f.brain_ids[i]; break;
        case 'parent_brain_id':  out[i] = f.parent_brain_ids[i]; break;
        case 'age':              out[i] = f.ids[i]; break;
        case 'token_share':      out[i] = this.totalTokens ? f.tokens[i] / this.totalTokens : 0; break;
        default:                 out[i] = 0.5;
      }
    }
    this._nodeCache.set(key, out);
    return out;
  }

  /** The same values under the reader's choice of scale. */
  scaledNodeValues(key, log) {
    const raw = this.nodeValues(key);
    if (!log) return raw;
    const signed = Metrics.isSigned('node', key);
    const out = new Float64Array(raw.length);
    for (let i = 0; i < raw.length; i++) out[i] = Metrics.applyLog(raw[i], signed);
    return out;
  }

  /**
   * The same curvature, but on the graph as it stood before this phase.
   *
   * Read against the token change the phase produced, this is the pair a
   * diffusion law is written in: how far a node sat below its neighbourhood,
   * and how much it then gained. Taking the curvature from the current frame
   * instead would pair the change with the state it had already produced,
   * which answers a different and much less interesting question.
   *
   * Both the tokens and the wiring come from the earlier frame, since the
   * phase moves edges as well as tokens. A node that was not there yet has no
   * before to speak of and gets NaN, which the charts drop — those are the
   * points the reader is told will not line up.
   */
  get curvatureBefore() {
    if (this._curvatureBefore) return this._curvatureBefore;

    const f = this.frame;
    const n = f.ids.length;
    const out = new Float64Array(n);
    const previous = f.previous;

    if (!previous) {
      out.fill(NaN);
      this._curvatureBefore = out;
      return out;
    }

    // Curvature on the earlier graph, in that graph's own node order.
    const m = previous.ids.length;
    const slot = new Map();
    for (let j = 0; j < m; j++) slot.set(previous.ids[j], j);

    const degree = new Int32Array(m);
    const curve = new Float64Array(m);
    for (const [a, b] of previous.edges) {
      const ja = slot.get(a), jb = slot.get(b);
      if (ja === undefined || jb === undefined || ja === jb) continue;
      degree[ja]++; degree[jb]++;
      curve[ja] += previous.tokens[jb];
      curve[jb] += previous.tokens[ja];
    }
    for (let j = 0; j < m; j++) curve[j] -= degree[j] * previous.tokens[j];

    // Carried across to this frame's node order; anything newly born is NaN.
    for (let i = 0; i < n; i++) {
      const j = slot.get(f.ids[i]);
      out[i] = (j === undefined) ? NaN : curve[j];
    }
    this._curvatureBefore = out;
    return out;
  }

  /**
   * How far each node's pile sits below the average of what surrounds it.
   *
   * The sum of the neighbours' tokens less the node's own times its degree —
   * the graph Laplacian applied to wealth. Positive means a node is poorer
   * than its neighbourhood and sits in a valley; negative means it is a peak
   * its neighbours drain toward. Zero means it is exactly level with them,
   * which is what a flat stretch of the graph looks like.
   *
   * Walked over edges rather than through an adjacency map, so it costs one
   * pass and does not force the structure to be built.
   */
  _curvature() {
    const f = this.frame;
    const n = f.ids.length;
    const out = new Float64Array(n);

    for (const [a, b] of f.edges) {
      const ia = this.index.get(a), ib = this.index.get(b);
      if (ia === undefined || ib === undefined || ia === ib) continue;
      out[ia] += f.tokens[ib];
      out[ib] += f.tokens[ia];
    }
    for (let i = 0; i < n; i++) out[i] -= this.degree[i] * f.tokens[i];
    return out;
  }

    /**
   * Quantities that read as "up or down" rather than "more or less".
   *
   * These get a range centred on zero so the middle of the colour map means no
   * change, and a gain of 50 is the same distance from centre as a loss of 50.
   * Stretching them to fit min..max would put the neutral point wherever the
   * data happened to land.
   */
  nodeRange(key, log) {
    return this._rangeOf(this.scaledNodeValues(key, log), Metrics.isSigned('node', key));
  }

  _rangeOf(values, signed) {
    if (!signed) return this._rangeLinear(values);
    // Centred, so the middle of the colour map is no change and a gain of 50
    // sits as far from centre as a loss of 50. Stretching to fit min..max
    // would put the neutral point wherever the data happened to land.
    let extent = 0;
    for (const v of values) {
      if (Number.isNaN(v)) continue;
      extent = Math.max(extent, Math.abs(v));
    }
    if (extent < 1e-9) extent = 1;
    return [-extent, extent];
  }

  _rangeLinear(values) {
    let lo = Infinity, hi = -Infinity;
    for (const v of values) {
      if (Number.isNaN(v)) continue;   // a gap, not a value
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!Number.isFinite(lo)) return [0, 1];
    return (hi - lo < 1e-9) ? [lo, lo + 1] : [lo, hi];
  }

  static _norm(v, [lo, hi]) {
    if (Number.isNaN(v)) return 0.5;
    return Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
  }

  nodeColorNorm(i)  { return FrameMetrics._norm(this.colorValues[i], this.colorRange); }
  nodeSizeNorm(i)   { return FrameMetrics._norm(this.sizeValues[i], this.sizeRange); }

  nodeColorCssByIndex(i, alpha) {
    const s = this.settings;
    if (s.nodeColorBy === 'constant') {
      return colormapCss(s.nodeColormap, 0.6, alpha, s.nodeColorReverse);
    }
    return colormapCss(s.nodeColormap, this.nodeColorNorm(i), alpha, s.nodeColorReverse);
  }

  nodeColorCss(id, alpha) {
    const i = this.index.get(id);
    return (i === undefined) ? 'rgba(120,120,120,1)' : this.nodeColorCssByIndex(i, alpha);
  }

  // ---- edge-derived quantities ---------------------------------------

  _edgeRaw(kind, a, b) {
    const ia = this.index.get(a), ib = this.index.get(b);
    if (ia === undefined || ib === undefined) return 0;
    const f = this.frame;

    switch (kind) {
      case 'avg_tokens': return (f.tokens[ia] + f.tokens[ib]) / 2;
      case 'min_tokens': return Math.min(f.tokens[ia], f.tokens[ib]);
      case 'max_tokens': return Math.max(f.tokens[ia], f.tokens[ib]);
      case 'token_gap':  return Math.abs(f.tokens[ia] - f.tokens[ib]);
      case 'avg_degree': return (this.degree[ia] + this.degree[ib]) / 2;
      case 'min_degree': return Math.min(this.degree[ia], this.degree[ib]);
      case 'max_degree': return Math.max(this.degree[ia], this.degree[ib]);
      case 'avg_curvature': return (this.curvature[ia] + this.curvature[ib]) / 2;
      case 'flow': {
        const stride = this.frame.ids.length + 1;
        const key = ia < ib ? ia * stride + ib : ib * stride + ia;
        return this.flow.get(key) || 0;
      }
      case 'loops': {
        const i = this.edgeSlot(a, b);
        return i < 0 ? 0 : this.structure.loops.edgeLoops[i];
      }
      case 'triangles': {
        const i = this.edgeSlot(a, b);
        return i < 0 ? 0 : this.structure.triangles.perEdge[i];
      }
      case 'bridge': {
        // 1 when the edge lies on no loop at all.
        const i = this.edgeSlot(a, b);
        return (i >= 0 && this.structure.loops.edgeLoops[i] === 0) ? 1 : 0;
      }
      default: return 0;
    }
  }

  /** Raw per-edge values for one metric, in frame edge order. Cached. */
  edgeValues(key) {
    if (!this._edgeCache) this._edgeCache = new Map();
    const hit = this._edgeCache.get(key);
    if (hit) return hit;

    const edges = this.frame.edges;
    const out = new Float64Array(edges.length);
    for (let e = 0; e < edges.length; e++) {
      out[e] = this._edgeRaw(key, edges[e][0], edges[e][1]);
    }
    this._edgeCache.set(key, out);
    return out;
  }

  scaledEdgeValues(key, log) {
    const raw = this.edgeValues(key);
    if (!log) return raw;
    const signed = Metrics.isSigned('edge', key);
    const out = new Float64Array(raw.length);
    for (let i = 0; i < raw.length; i++) out[i] = Metrics.applyLog(raw[i], signed);
    return out;
  }

  edgeRange(key, log) {
    if (!this._edgeRanges) this._edgeRanges = new Map();
    const cacheKey = `${key}|${log ? 1 : 0}`;
    if (this._edgeRanges.has(cacheKey)) return this._edgeRanges.get(cacheKey);

    const range = this._rangeOf(this.scaledEdgeValues(key, log),
                               Metrics.isSigned('edge', key));
    this._edgeRanges.set(cacheKey, range);
    return range;
  }

  _edgeNorm(kind, log, a, b) {
    if (kind === 'constant' || kind === 'source') return 0.5;
    const value = this._edgeRaw(kind, a, b);
    const scaled = log ? Metrics.applyLog(value, Metrics.isSigned('edge', kind)) : value;
    return FrameMetrics._norm(scaled, this.edgeRange(kind, log));
  }

  edgeColorNorm(a, b) {
    return this._edgeNorm(this.settings.edgeColorBy, this.settings.edgeColorLog, a, b);
  }

  edgeWidthNorm(a, b) {
    const kind = this.settings.edgeWidthBy;
    if (kind === 'constant') return 0;
    return this._edgeNorm(kind, this.settings.edgeWidthLog, a, b);
  }

  // ---- structure: loops, triangles, dimension -------------------------

  /**
   * Adjacency, loops, triangles and dimension for this frame.
   *
   * Built on first use and kept, since these cost real work and most frames
   * are drawn without anyone asking for them.
   */
  get structure() {
    if (this._structure) return this._structure;

    const f = this.frame;
    const adj = GraphStats.adjacency(f.ids, f.edges);
    const loops = GraphStats.loops(f.ids, f.edges, adj);
    const triangles = GraphStats.triangles(f.ids, f.edges, adj);
    const dimension = GraphStats.dimension(f.ids, adj);
    const distances = GraphStats.distances(f.ids, adj);

    // Edge lookups are by endpoint pair, since the renderer walks edges by id.
    const edgeKey = new Map();
    for (let i = 0; i < f.edges.length; i++) {
      const [a, b] = f.edges[i];
      edgeKey.set(a < b ? `${a},${b}` : `${b},${a}`, i);
    }

    this._structure = { adj, loops, triangles, dimension, distances, edgeKey };
    return this._structure;
  }

  edgeSlot(a, b) {
    const key = a < b ? `${a},${b}` : `${b},${a}`;
    const i = this.structure.edgeKey.get(key);
    return i === undefined ? -1 : i;
  }

  // ---- per-node decisions --------------------------------------------

  /**
   * What each agent actually did this phase, keyed by node id.
   *
   * Built once and reused by the hover card. Which fields exist depends on the
   * phase: a reproduction frame knows about births, a game frame about
   * allocations and conquests.
   */
  get decisionIndex() {
    if (this._decisions) return this._decisions;

    const d = this.frame.decisions || {};
    const births = new Map();     // parent id -> birth record
    const newborns = new Map();   // child id  -> parent id
    const allocations = new Map();// agent id  -> allocation record
    const winners = new Map();    // node id   -> winner record
    const conquests = new Map();  // agent id  -> nodes it took

    for (const b of d.births || []) {
      births.set(b.agent, b);
      newborns.set(b.child, b.agent);
    }
    const rewires = new Map();
    for (const r of d.rewires || []) rewires.set(r.agent, r);
    for (const a of d.allocations || []) allocations.set(a.agent, a);
    for (const w of d.winners || []) {
      winners.set(w.node, w);
      conquests.set(w.winner, (conquests.get(w.winner) || 0) + 1);
    }

    this._decisions = { births, newborns, allocations, winners, conquests, rewires };
    return this._decisions;
  }

  /** Everything worth telling the reader about one node, for the hover card. */
  nodeDetail(i) {
    const f = this.frame;
    const id = f.ids[i];
    const idx = this.decisionIndex;

    const detail = {
      id,
      tokens: f.tokens[i],
      degree: this.degree[i],
      tokenShare: this.totalTokens ? f.tokens[i] / this.totalTokens : 0,
      brainId: f.brain_ids[i],
      parentBrainId: f.parent_brain_ids[i],
      spawnedBy: f.parent_ids[i] >= 0 ? f.parent_ids[i] : null,
      rank: this.wealthRank(i),
      curvature: this.curvature[i],
      phase: f.phase,
      delta: this.delta[i],
      hasDelta: this.hasDelta
    };

    if (f.phase === 1) {
      const birth = idx.births.get(id);
      detail.reproduced = idx.births.size ? Boolean(birth) : null;
      if (birth) {
        detail.invested = birth.invested;
        detail.handedOver = birth.handed_over ? birth.handed_over.length : null;
        detail.investedShare = birth.tokens_before ? birth.invested / birth.tokens_before : null;
        detail.child = birth.child;
        detail.childLinks = birth.links ? birth.links.length : 0;
      }
      const bornFrom = idx.newborns.get(id);
      if (bornFrom !== undefined) detail.newbornOf = bornFrom;

      const moved = idx.rewires.get(id);
      if (moved) detail.rewire = { edge: moved.edge, to: moved.to };
    } else {
      const alloc = idx.allocations.get(id);
      if (alloc) {
        const total = alloc.alloc.reduce((a, b) => a + b, 0);
        const selfIndex = alloc.targets.indexOf(id);
        detail.allocated = total;
        detail.keptAtHome = selfIndex >= 0 ? alloc.alloc[selfIndex] : 0;
        detail.revolted = alloc.revolt ? alloc.revolt.reduce((a, b) => a + b, 0) : 0;
        detail.doctrine = alloc.spread ? 'spread' : 'all-in';
      }

      const win = idx.winners.get(id);
      if (win) {
        // "Held home" means the agent standing on this node kept it; anything
        // else means a neighbour moved its genome in.
        detail.heldHome = win.winner === id;
        detail.takenBy = win.winner === id ? null : win.winner;
        detail.winningBid = win.amount;
        detail.wonByRevolt = Boolean(win.revolt);
      }
      detail.nodesWon = idx.conquests.get(id) || 0;
    }

    return detail;
  }

  wealthRank(i) {
    if (!this._wealthRank) {
      const order = Array.from(this.frame.tokens.keys())
        .sort((a, b) => this.frame.tokens[b] - this.frame.tokens[a]);
      const rank = new Int32Array(order.length);
      order.forEach((nodeIndex, position) => { rank[nodeIndex] = position + 1; });
      this._wealthRank = rank;
    }
    return this._wealthRank[i];
  }

  // ---- summary --------------------------------------------------------

  /** Tokens carried by each edge this phase, as a plain array. */
  flowValues() {
    return Array.from(this.flow.values());
  }

  /**
   * The numbers under the canvas.
   *
   * `includeStructure` gates the handful that need the whole graph walked;
   * without it they stay null and the strip simply omits them.
   */
  summary(includeStructure = true) {
    const f = this.frame;
    const n = f.ids.length;
    const degrees = Array.from(this.degree);
    const tokens = f.tokens;
    const d = f.decisions || {};

    const sum = arr => arr.reduce((a, b) => a + b, 0);
    const mean = arr => arr.length ? sum(arr) / arr.length : 0;
    const median = arr => {
      if (!arr.length) return 0;
      const s = [...arr].sort((a, b) => a - b);
      const m = Math.floor(s.length / 2);
      return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    };

    // How concentrated is wealth? 0 = perfectly equal, 1 = one agent holds all.
    const sorted = [...tokens].sort((a, b) => a - b);
    let cumulative = 0, weighted = 0;
    for (let i = 0; i < sorted.length; i++) {
      cumulative += sorted[i];
      weighted += cumulative;
    }
    const gini = cumulative > 0
      ? (sorted.length + 1 - 2 * weighted / cumulative) / sorted.length
      : 0;

    // Share held by the richest tenth — a blunter, more readable companion to
    // the Gini coefficient.
    const topCount = Math.max(1, Math.round(n * 0.1));
    const topShare = cumulative > 0
      ? sum(sorted.slice(-topCount)) / cumulative
      : 0;

    const distinctBrains = new Set(f.brain_ids).size;
    const distinctLineages = new Set(f.parent_brain_ids).size;

    const out = {
      iteration: f.iteration,
      phase: f.phase,
      // How many nodes entered this phase. Older runs predate the field, in
      // which case the caller falls back to the previous frame's node count.
      nodesBefore: (typeof f.nodes_before === 'number') ? f.nodes_before : null,

      // Topology
      nodes: n,
      edges: f.edges.length,
      density: n > 1 ? (2 * f.edges.length) / (n * (n - 1)) : 0,
      meanDegree: mean(degrees),
      medianDegree: median(degrees),
      maxDegree: degrees.length ? Math.max(...degrees) : 0,
      minDegree: degrees.length ? Math.min(...degrees) : 0,
      leaves: degrees.filter(x => x === 1).length,

      // Wealth
      tokens: this.totalTokens,
      meanTokens: mean(Array.from(tokens)),
      medianTokens: median(tokens),
      maxTokens: tokens.length ? Math.max(...tokens) : 0,
      minTokens: tokens.length ? Math.min(...tokens) : 0,
      gini,
      topDecileShare: topShare,

      // Biggest single swing either way this phase. Losses are reported as a
      // positive magnitude so the two read side by side.
      maxTokenAdded: this.delta.length ? Math.max(0, ...this.delta) : 0,
      maxTokenLost: this.delta.length ? Math.max(0, ...this.delta.map(v => -v)) : 0,
      gainers: this.delta.filter(v => v > 0).length,
      losers: this.delta.filter(v => v < 0).length,

      // Structure
      cycleRank: null, loopDensity: null, bridges: null, triangles: null,
      transitivity: null, degreeEntropy: null, degreeEvenness: null,
      radius: null, diameter: null, meanPathLength: null,
      tokenEntropy: null, tokenEvenness: null, dimension: null, components: null,

      // Genome
      distinctBrains,
      brainDiversity: n ? distinctBrains / n : 0,
      distinctLineages,

      // Cleanup, present on both phases
      starved: f.cleanup ? f.cleanup.starved : null,
      orphaned: f.cleanup ? f.cleanup.orphaned : null,
      redistributed: f.cleanup ? f.cleanup.redistributed : null,

      births: null, meanInvestedShare: null, meanChildLinks: null,
      reproTokenShare: null, handovers: null, rewires: null,
      revolutions: null, totalFlow: null, meanEdgeFlow: null, maxEdgeFlow: null,
      selfAllocationShare: null, revoltShare: null, spreadShare: null,
      heldHomeShare: null, prunedEdges: null
    };

    // Rewiring happens during the reproduction phase but is nobody's child:
    // absent when the rule is off, rather than a zero that reads as "nobody
    // chose to".
    const framedRewires = (f.summary || {}).rewires;
    if (framedRewires !== undefined) out.rewires = framedRewires;
    else if (d.rewires !== undefined) out.rewires = d.rewires.length;

    // ---- reproduction phase ----
    if (d.births) {
      const births = d.births;
      out.births = births.length;
      out.meanInvestedShare = births.length
        ? mean(births.map(b => b.tokens_before ? b.invested / b.tokens_before : 0))
        : 0;
      out.meanChildLinks = births.length
        ? mean(births.map(b => (b.links || []).length))
        : 0;

      // What share of the entire economy was committed to newborns this
      // phase. Distinct from meanInvestedShare, which averages each parent's
      // share of its own pile and so says nothing about how much of the world
      // that amounted to.
      const invested = sum(births.map(b => b.invested));
      out.reproTokenShare = this.totalTokens ? invested / this.totalTokens : 0;

      // Only present on runs with handover enabled; absent means the mechanic
      // was off, which is not the same as it being on and never used.
      if (births.some(b => b.handed_over !== undefined)) {
        out.handovers = sum(births.map(b => (b.handed_over || []).length));
      }
    }

    // ---- game phase ----
    if (d.allocations) {
      const allocations = d.allocations;
      let allocatedTotal = 0, keptAtHome = 0, revolted = 0, spreadCount = 0;

      for (const a of allocations) {
        const total = sum(a.alloc);
        allocatedTotal += total;
        const selfIndex = a.targets.indexOf(a.agent);
        if (selfIndex >= 0) keptAtHome += a.alloc[selfIndex];
        if (a.revolt) revolted += sum(a.revolt);
        if (a.spread) spreadCount++;
      }

      out.selfAllocationShare = allocatedTotal ? keptAtHome / allocatedTotal : 0;
      out.spreadShare = allocations.length ? spreadCount / allocations.length : 0;
      // Left null when the run has revolutions switched off, so the reader can
      // tell that apart from a phase where nobody revolted.
      if (allocations.some(a => a.revolt !== undefined)) {
        out.revoltShare = allocatedTotal ? revolted / allocatedTotal : 0;
      }

      const flows = this.flowValues();
      out.totalFlow = sum(flows);
      out.meanEdgeFlow = mean(flows);
      out.maxEdgeFlow = flows.length ? Math.max(...flows) : 0;
    }

    if (d.winners) {
      if (d.winners.some(w => w.revolt !== undefined)) {
        out.revolutions = d.winners.filter(w => w.revolt).length;
      }
      out.heldHomeShare = d.winners.length
        ? d.winners.filter(w => w.winner === w.node).length / d.winners.length
        : 0;
    }

    if (d.pruned_edges) out.prunedEdges = d.pruned_edges.length;

    // ---- structure ----
    //
    // Loops, bridges, triangles, dimension and the distance sweeps are by far
    // the most expensive thing here — around 250ms of a 280ms summary at
    // twenty thousand nodes, which is what made stepping between frames feel
    // heavy. The group they feed is collapsed by default, so the reader was
    // usually paying for numbers nobody was looking at. The caller asks for
    // them when the group is open, and the rest of the summary no longer
    // waits on them.
    if (includeStructure) {
      const st = this.structure;
      out.cycleRank = st.loops.cycleRank;
      out.loopDensity = f.edges.length ? st.loops.cycleRank / f.edges.length : 0;
      out.bridges = st.loops.bridges;
      out.components = st.loops.componentCount;
      out.triangles = st.triangles.total;
      out.transitivity = GraphStats.transitivity(f.ids, st.adj, st.triangles.total);
      out.dimension = st.dimension.estimate;
      out.radius = st.distances.radius;
      out.diameter = st.distances.diameter;
      out.meanPathLength = st.distances.meanPathLength;
    }

    out.degreeEntropy = GraphStats.degreeEntropy(degrees);
    // Against the most even the same number of classes could be, so 1 means
    // every degree is equally common.
    const degreeClasses = new Set(degrees).size;
    out.degreeEvenness = degreeClasses > 1 ? out.degreeEntropy / Math.log2(degreeClasses) : 0;

    out.tokenEntropy = GraphStats.entropyOfCounts(Array.from(tokens));
    out.tokenEvenness = n > 1 ? out.tokenEntropy / Math.log2(n) : 0;

    return out;
  }
}

// --------------------------------------------------------------------------
// Charts
// --------------------------------------------------------------------------

/** Shared canvas setup: size to the element's box in device pixels. */
function _prepareCanvas(canvas) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  return { ctx, w: rect.width, h: rect.height };
}

function _noData(ctx, w, h, message = 'no data') {
  ctx.fillStyle = '#5b6b7c';
  ctx.font = '11px system-ui, sans-serif';
  ctx.fillText(message, 8, h / 2);
}

/**
 * Draw a bar histogram of one metric.
 *
 * Values arrive raw; `logScale` is applied here rather than by the caller so
 * the axis can be labelled in the units the reader actually chose. `signed`
 * keeps the direction of quantities that run either way, which is what lets
 * token change and curvature be read on a log scale at all.
 */
function drawHistogram(canvas, values, options = {}) {
  const { bins = 32, colormap = 'viridis', reverse = false,
          logScale = false, logCount = false, signed = false,
          format = v => Math.round(v).toLocaleString('en-US') } = options;

  const { ctx, w, h } = _prepareCanvas(canvas);
  if (!values || !values.length) { _noData(ctx, w, h); return; }

  // Values can be missing — a "before the phase" quantity has none for a node
  // that did not exist yet. Those are dropped rather than counted as zero.
  const mapped = [];
  let skipped = 0;
  for (const raw of values) {
    if (Number.isNaN(raw)) { skipped++; continue; }
    mapped.push(logScale ? Metrics.applyLog(raw, signed) : raw);
  }
  if (!mapped.length) { _noData(ctx, w, h, 'no values for this frame'); return; }

  let lo = Infinity, hi = -Infinity;
  for (const v of mapped) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!Number.isFinite(lo)) { _noData(ctx, w, h); return; }
  if (hi - lo < 1e-9) hi = lo + 1;

  const counts = new Array(bins).fill(0);
  for (const v of mapped) {
    counts[Math.min(bins - 1, Math.floor((v - lo) / (hi - lo) * bins))]++;
  }
  const peak = Math.max(...counts) || 1;

  const padBottom = 16, padTop = 6;
  const plotH = h - padBottom - padTop;
  const barW = w / bins;

  // A log count axis lets a long tail of rare values stay visible next to a
  // spike that would otherwise flatten everything else to nothing.
  const barFraction = c => logCount
    ? (c > 0 ? Math.log1p(c) / Math.log1p(peak) : 0)
    : c / peak;

  for (let i = 0; i < bins; i++) {
    const barH = barFraction(counts[i]) * plotH;
    ctx.fillStyle = colormapCss(colormap, i / (bins - 1), 0.92, reverse);
    ctx.fillRect(i * barW, padTop + plotH - barH, Math.max(1, barW - 1), barH);
  }

  const back = v => logScale ? Metrics.undoLog(v, signed) : v;
  ctx.fillStyle = '#8fa3b5';
  ctx.font = '10px system-ui, sans-serif';
  ctx.fillText(format(back(lo)), 2, h - 4);
  const hiText = format(back(hi));
  ctx.fillText(hiText, w - ctx.measureText(hiText).width - 2, h - 4);
  const peakText = `peak ${peak}${logCount ? ' \u00b7 log' : ''}`
    + (skipped ? ` \u00b7 ${skipped.toLocaleString('en-US')} without a value` : '');
  ctx.fillText(peakText, (w - ctx.measureText(peakText).width) / 2, h - 4);
}

/**
 * Draw a two-dimensional binned heatmap of one metric against another.
 *
 * Each item — a node, or an edge — contributes one point, so the two arrays
 * must describe the same things in the same order. That is why the caller is
 * required to keep both axes in one domain: pairing a node value with an edge
 * value would count things that have no correspondence at all.
 *
 * Cell colour is the count in that bin, which is nearly always the quantity
 * with the widest spread on the chart: a handful of cells hold most of the
 * population. `logCount` is usually what makes the rest of the grid visible.
 */
function drawHeatmap(canvas, xs, ys, options = {}) {
  const { binsX = 34, binsY = 22, colormap = 'viridis', reverse = false,
          logX = false, logY = false, logCount = true,
          signedX = false, signedY = false,
          formatX = v => Math.round(v).toLocaleString('en-US'),
          formatY = v => Math.round(v).toLocaleString('en-US'),
          message = null } = options;

  const { ctx, w, h } = _prepareCanvas(canvas);
  if (message) { _noData(ctx, w, h, message); return; }
  if (!xs || !ys || !xs.length || xs.length !== ys.length) { _noData(ctx, w, h); return; }

  // A point needs a value on both axes. Where either is missing there is
  // nothing to plot it against, so it is dropped — which is why the two
  // quantities can have different counts and still be compared honestly.
  const mx = [], my = [];
  let dropped = 0;
  for (let i = 0; i < xs.length; i++) {
    const a = xs[i], b = ys[i];
    if (Number.isNaN(a) || Number.isNaN(b)) { dropped++; continue; }
    mx.push(logX ? Metrics.applyLog(a, signedX) : a);
    my.push(logY ? Metrics.applyLog(b, signedY) : b);
  }
  if (!mx.length) { _noData(ctx, w, h, 'nothing has a value on both axes'); return; }

  const extent = (arr) => {
    let lo = Infinity, hi = -Infinity;
    for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!Number.isFinite(lo)) return null;
    return hi - lo < 1e-9 ? [lo, lo + 1] : [lo, hi];
  };
  const ex = extent(mx), ey = extent(my);
  if (!ex || !ey) { _noData(ctx, w, h); return; }

  const padLeft = 38, padBottom = 15, padTop = 4, padRight = 2;
  const plotW = Math.max(1, w - padLeft - padRight);
  const plotH = Math.max(1, h - padBottom - padTop);

  const counts = new Int32Array(binsX * binsY);
  for (let i = 0; i < mx.length; i++) {
    const bx = Math.min(binsX - 1, Math.floor((mx[i] - ex[0]) / (ex[1] - ex[0]) * binsX));
    const by = Math.min(binsY - 1, Math.floor((my[i] - ey[0]) / (ey[1] - ey[0]) * binsY));
    counts[by * binsX + bx]++;
  }
  let peak = 0;
  for (const c of counts) if (c > peak) peak = c;
  if (!peak) { _noData(ctx, w, h); return; }

  const shade = c => logCount ? Math.log1p(c) / Math.log1p(peak) : c / peak;

  // Empty cells stay as background rather than taking the colour map's zero,
  // so "nothing here" reads differently from "the lowest value on the scale".
  const cellW = plotW / binsX, cellH = plotH / binsY;
  for (let by = 0; by < binsY; by++) {
    for (let bx = 0; bx < binsX; bx++) {
      const c = counts[by * binsX + bx];
      if (!c) continue;
      ctx.fillStyle = colormapCss(colormap, shade(c), 1, reverse);
      // y runs upward on screen, so the top row is the last bin.
      const px = padLeft + bx * cellW;
      const py = padTop + (binsY - 1 - by) * cellH;
      ctx.fillRect(px, py, Math.ceil(cellW), Math.ceil(cellH));
    }
  }

  const backX = v => logX ? Metrics.undoLog(v, signedX) : v;
  const backY = v => logY ? Metrics.undoLog(v, signedY) : v;

  ctx.fillStyle = '#8fa3b5';
  ctx.font = '10px system-ui, sans-serif';

  // y axis: high at the top, low at the bottom of the plot.
  ctx.fillText(formatY(backY(ey[1])), 2, padTop + 8);
  ctx.fillText(formatY(backY(ey[0])), 2, padTop + plotH - 1);

  // x axis, plus what the colour means.
  ctx.fillText(formatX(backX(ex[0])), padLeft, h - 3);
  const hiText = formatX(backX(ex[1]));
  ctx.fillText(hiText, w - ctx.measureText(hiText).width - 2, h - 3);
  const peakText = `${mx.length.toLocaleString('en-US')} paired \u00b7 peak ${peak}`
    + (logCount ? ' \u00b7 log' : '')
    + (dropped ? ` \u00b7 ${dropped.toLocaleString('en-US')} unpaired` : '');
  ctx.fillText(peakText, padLeft + (plotW - ctx.measureText(peakText).width) / 2, h - 3);
}
